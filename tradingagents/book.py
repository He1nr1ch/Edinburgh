"""Build the day's book from a list run's calls, with its bench, and save both.

The `size` command and the 4 AM job share this: the sizing rules make the
book (the top names each side), the bench is the next-best calls after it,
long or short together, and both go to the sizing folder as CSVs the page
and the Watcher read back.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from tradingagents.ranking import age_days, balances_book, levels_pct, score_call
from tradingagents.sizing import SIDE, Book, Call, bench_for, last_closes, load_calls, size_book, stop_for


@dataclass
class BuiltBook:
    date: str
    run_ids: list[str]
    equity: float
    book: Book
    bench: list[dict]
    book_path: Path | None = None
    bench_path: Path | None = None
    timing: dict | None = None
    timing_path: Path | None = None
    hedge: object | None = None        # the hedge.Hedge plan when the daily job priced one


def recent_calls(results_dir: Path, as_of: str, max_age_days: int) -> list[Call]:
    """Each ticker's newest call dated within ``max_age_days`` of ``as_of``.

    The nightly sweep covers part of the index each night (thinking makes it
    slow), so the book draws on every decision still fresh, not one night's.
    """
    import json
    from datetime import date as _date, timedelta

    from tradingagents.sizing import call_from_state
    from tradingagents.web.data import all_decisions

    oldest = (_date.fromisoformat(as_of) - timedelta(days=max_age_days)).isoformat()
    calls = []
    for ticker, refs in all_decisions(Path(results_dir)).items():
        fresh = next((r for r in refs if oldest <= r.date <= as_of), None)
        if fresh is not None:
            calls.append(call_from_state(json.loads(fresh.path.read_text(encoding="utf-8")), ticker, fresh.date))
    return calls


def build_book(run_ids: list[str], date: str, config: dict, equity: float,
               gross_pct: float | None = None, cap_pct: float | None = None,
               min_target_pct: float | None = None, names_per_side: int | None = None,
               calls: list[Call] | None = None) -> BuiltBook:
    """Size the calls of ``run_ids`` on ``date`` (or the ``calls`` given) into a book and a bench."""
    root = Path(config["results_dir"]) / "backtest"
    if calls is None:
        calls = [c for run_id in run_ids for c in load_calls(root / run_id, date)]
    if not calls:
        raise ValueError(f"no calls for {date} under {', '.join(run_ids)} in {root}")
    floor = config["min_target_pct"] if min_target_pct is None else min_target_pct
    per_side = config["names_per_side"] if names_per_side is None else names_per_side
    prices = last_closes([c.ticker for c in calls], date)
    book = size_book(calls, prices, equity, config["gross_pct"] if gross_pct is None else gross_pct,
                     config["position_cap_pct"] if cap_pct is None else cap_pct, floor, per_side,
                     config["default_stop_pct"])
    chosen = {o.ticker for o in book.orders}
    by_ticker = {c.ticker: c for c in calls}
    net = sum(o.value * (1 if o.shares > 0 else -1) for o in book.orders)
    for o in book.orders:
        o.score = scored(by_ticker[o.ticker], o.price, date, config["default_stop_pct"], net).score
    # The bench: every call that clears the rules and is not in the book, the
    # best-scored bench_size of them, so the wings are ranked the same way the
    # morning placement and the Watcher rank them.
    bench = []
    for c in bench_for(calls, prices, chosen, floor, len(calls)):
        price = prices[c.ticker]
        stop, stop_from = stop_for(c, price, config["default_stop_pct"])
        s = scored(c, price, date, config["default_stop_pct"], net)
        bench.append({"ticker": c.ticker, "side": "long" if c.rating in ("Buy", "Overweight") else "short",
                      "rating": c.rating, "risk_level": c.risk_level or "", "ceiling_pct": c.ceiling_pct,
                      "price": price, "stop": stop, "stop_from": stop_from, "target": c.target,
                      "expected_move_pct": round(c.target_move_pct(price) or 0.0, 2),
                      "decided": c.decided, "score": s.score, "score_parts": s.parts()})
    bench.sort(key=lambda b: (-b["score"], b["ticker"]))
    return BuiltBook(date, list(run_ids), equity, book, bench[:config["bench_size"]])


def scored(call: Call, price: float, as_of: str, default_stop_pct: float, net_value: float = 0.0):
    """The call's score at ``price``: reward and risk from its own levels (the sizing rules' stop)."""
    side = SIDE.get(call.rating, 0)
    stop, _ = stop_for(call, price, default_stop_pct)
    reward, risk = levels_pct(side, price, call.target, stop, default_stop_pct)
    return score_call(reward, risk, call.rating, call.risk_level, age_days(call.decided, as_of),
                      balances_book(side, net_value), min_risk_pct=default_stop_pct)


BOOK_COLUMNS = ["ticker", "side", "rating", "risk_level", "ceiling_pct", "weight_pct", "price", "shares", "value",
                "stop", "stop_from", "target", "score", "decided"]
BENCH_COLUMNS = ["ticker", "side", "rating", "risk_level", "ceiling_pct", "price", "stop", "stop_from", "target",
                 "expected_move_pct", "decided", "score", "score_parts"]


def save_book(built: BuiltBook, config: dict, hedge_row: list | None = None) -> BuiltBook:
    """Write the orders CSV and the bench CSV beside the run logs; returns ``built`` with the paths."""
    folder = Path(config["results_dir"]) / "backtest" / "sizing"
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{'+'.join(built.run_ids)}_{built.date}"
    built.book_path = folder / f"{stem}.csv"
    with built.book_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(BOOK_COLUMNS)
        for o in built.book.orders:
            writer.writerow([o.ticker, o.side, o.rating, o.risk_level or "", o.ceiling_pct,
                             round(o.weight_pct, 4), o.price, o.shares, round(o.value, 2),
                             o.stop, o.stop_from, o.target or "", getattr(o, "score", ""), o.decided])
        if hedge_row:
            writer.writerow(list(hedge_row) + [""] * (len(BOOK_COLUMNS) - len(hedge_row)))
    built.bench_path = folder / f"{stem}_bench.csv"
    with built.bench_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=BENCH_COLUMNS)
        writer.writeheader()
        writer.writerows(built.bench)
    return built


def load_rows(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def latest_bench(results_dir: Path) -> tuple[Path | None, list[dict]]:
    """The newest bench CSV, as rows."""
    folder = Path(results_dir) / "backtest" / "sizing"
    files = sorted(folder.glob("*_bench.csv"), key=lambda p: p.stat().st_mtime) if folder.is_dir() else []
    return (files[-1], load_rows(files[-1])) if files else (None, [])
