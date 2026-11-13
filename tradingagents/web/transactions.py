"""The day's transactions: what was entered, what was exited, and why.

Every order the worker sends is recorded with its kind, size and the reason
it was sent (the stop it hit, the news that moved it, the book it came from,
the position it replaces). This reads those records back as a list for the
page, the close-of-day email and the terminal, with the fill price from
Alpaca where it has one.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from tradingagents.feedback.scoring import r_multiple, r_summary, risk_fraction
from tradingagents.worker.state import WorkerState

logger = logging.getLogger(__name__)

# What each order kind means in the list, and whether it opens or closes exposure.
KINDS = {
    "open": ("Entered", "in"),
    "replace": ("Entered (replacement)", "in"),
    "add": ("Added to", "in"),
    "trim": ("Trimmed", "out"),
    "exit": ("Exited", "out"),
}


def transactions_on(state: WorkerState, day: str, fills: dict[str, dict] | None = None,
                    fills_for: Callable[[str], dict[str, dict]] | None = None,
                    config: dict | None = None) -> list[dict]:
    """The worker's orders on ``day`` as rows, in time order.

    An exit or a trim shows its result: the position's entry price (recorded
    with the order, else the fill of the last order that opened or added to the
    position, fetched for its own day through ``fills_for``) against the exit's
    fill, in dollars and as a percentage of the capital closed. With ``config``
    the result is also stated in R: over the risk the position was opened
    with, the distance from its entry to its stop (see ``entry_stop``).
    """
    rows = []
    lookups: dict = {}
    for o in state.orders_on(day):
        label, flow = KINDS.get(o["kind"], (o["kind"].capitalize(), ""))
        fill = (fills or {}).get(o.get("order_id", ""), {})
        side = o["side"]
        # The position's side, not the order's: a sell that opens is a short, a buy that closes was one.
        position = ("long" if side == "buy" else "short") if flow == "in" else ("long" if side == "sell" else "short")
        price = fill.get("filled_avg_price")
        row = {
            "time": o["at"][11:16], "at": o["at"], "ticker": o["symbol"], "action": label, "flow": flow,
            "kind": o["kind"], "order_side": side, "position": position, "qty": int(o["qty"]),
            "price": price, "value": round(price * int(o["qty"]), 2) if price else None,
            "status": fill.get("status", ""), "reason": o.get("reason", ""), "order_id": o.get("order_id", ""),
            "entry_price": None, "pnl": None, "pnl_pct": None, "stop": None, "risk_pct": None, "r": None,
        }
        if flow == "out":
            entry = o.get("entry_price") or _entry_from_fills(state, o, fills, fills_for)
            if entry and price:
                direction = 1 if position == "long" else -1
                row["entry_price"] = round(entry, 4)
                row["pnl"] = round((price - entry) * int(o["qty"]) * direction, 2)
                row["pnl_pct"] = round((price / entry - 1) * 100 * direction, 2)
                if config is not None:
                    stop = entry_stop(state, o, position, config, lookups)
                    risk = risk_fraction(direction, entry, stop, float(config.get("default_stop_pct") or 0))
                    r = r_multiple((price / entry - 1) * direction, risk)
                    if r is not None:
                        # The stop the risk was measured to: the desk's, or the default from the entry.
                        row["stop"] = round(entry * (1 - direction * risk), 2)
                        row["risk_pct"] = round(risk * 100, 2)
                        row["r"] = round(r, 2)
        rows.append(row)
    return rows


def _entry_from_fills(state: WorkerState, exit_order: dict, fills: dict | None,
                      fills_for: Callable[[str], dict] | None) -> float | None:
    """The fill price of the last order that opened or added to this position before the exit."""
    symbol, at = exit_order["symbol"], exit_order["at"]
    opened = [o for o in state.orders if o["symbol"] == symbol and o["kind"] in ("open", "replace", "add") and o["at"] < at]
    if not opened:
        return None
    entry = opened[-1]
    day = entry["at"][:10]
    pool = fills if fills and entry["order_id"] in fills else (fills_for(day) if fills_for else {}) or {}
    return (pool.get(entry["order_id"]) or {}).get("filled_avg_price")


def entry_stop(state: WorkerState, exit_order: dict, position: str, config: dict, lookups: dict | None = None) -> float | None:
    """The stop the position was opened with; None when the desk gave none (the default then applies).

    The stop kept with the order that opened it; for an order from before
    stops were kept, the stop in that day's book, else the stop of the
    decision that stood for the ticker on the day it was opened. Later
    decisions on the ticker do not move it: R is measured against the risk
    taken at entry. Never raises: a stop that cannot be read is no stop.
    """
    lookups = lookups if lookups is not None else {}
    symbol, at = exit_order["symbol"], exit_order["at"]
    opened = next((o for o in reversed(state.orders)
                   if o["symbol"] == symbol and o["kind"] in ("open", "replace") and o["at"] < at), None)
    if opened and opened.get("stop"):
        return float(opened["stop"])
    day = (opened or exit_order)["at"][:10]
    try:
        book_path = (state.days.get(day) or {}).get("book_path") if opened else None
        if book_path and Path(book_path).is_file():
            from tradingagents.book import load_rows

            if book_path not in lookups:
                lookups[book_path] = load_rows(Path(book_path))
            row = next((r for r in lookups[book_path] if r["ticker"].upper() == symbol.upper() and r.get("side") == position), None)
            if row and row.get("stop"):
                return float(row["stop"])
        if config.get("results_dir"):
            from tradingagents.web import data

            if "decisions" not in lookups:
                lookups["decisions"] = data.all_decisions(Path(config["results_dir"]))
            ref = next((r for r in lookups["decisions"].get(symbol.upper(), []) if r.date <= day), None)
            if ref is not None:
                summary = data.decision_summary(data.load_decision(ref), None, 0)
                if summary["side"] == position and summary["stop"]:
                    return float(summary["stop"])
    except Exception as exc:
        logger.warning("The stop %s was opened with could not be read: %s", symbol, exc)
    return None


def realised_r(rows: list[dict]) -> dict | None:
    """The day's result in R over the exits and trims whose R is known: ``{"n", "total", "mean"}``; None when none is."""
    return r_summary(r.get("r") for r in rows if r["flow"] == "out")


def r_text(summary: dict | None) -> str:
    """``-2.40R over 5 exits, -0.48R each``; empty when there is nothing to state."""
    if not summary:
        return ""
    return f"{summary['total']:+.2f}R over {summary['n']} exit{'s' if summary['n'] != 1 else ''}, {summary['mean']:+.2f}R each"


def realised(rows: list[dict]) -> float | None:
    """The day's realised result over the exits and trims whose result is known; None when none is."""
    known = [r["pnl"] for r in rows if r["flow"] == "out" and r["pnl"] is not None]
    return round(sum(known), 2) if known else None


def summarize(rows: list[dict], day: str) -> str:
    """Plain text for the email and the terminal: exits first with their reasons, then entries."""
    outs = [r for r in rows if r["flow"] == "out"]
    ins = [r for r in rows if r["flow"] == "in"]

    def line(r):
        price = f" @ {r['price']:,.2f}" if r["price"] else ""
        in_r = f", {r['r']:+.2f}R on a {r['risk_pct']:.1f}% stop" if r.get("r") is not None else ""
        result = (f" — entered {r['entry_price']:,.2f}, P/L {r['pnl']:+,.2f} ({r['pnl_pct']:+.1f}%{in_r})"
                  if r["pnl"] is not None else "")
        return f"- {r['time']} {r['action']} {r['position']} {r['ticker']}: {r['qty']} shares{price}{result}\n    {r['reason']}"

    total = realised(rows)
    text = [f"Edinburgh transactions for {day}: {len(outs)} exit(s), {len(ins)} entr{'y' if len(ins) == 1 else 'ies'}"
            + (f"; realised {total:+,.2f}" if total is not None else "")
            + (f"; {r_text(realised_r(rows))}" if realised_r(rows) else "") + "."]
    if outs:
        text += ["", "Exited:"] + [line(r) for r in outs]
    if ins:
        text += ["", "Entered:"] + [line(r) for r in ins]
    if not rows:
        text.append("No orders were sent.")
    return "\n".join(text)


def days_with_transactions(state: WorkerState) -> list[tuple[str, int, int]]:
    """Each day with orders, newest first: (day, exits, entries)."""
    days: dict[str, list[int]] = {}
    for o in state.orders:
        day = str(o.get("at", ""))[:10]
        if not day:
            continue
        flow = KINDS.get(o["kind"], ("", ""))[1]
        counts = days.setdefault(day, [0, 0])
        counts[0 if flow == "out" else 1] += 1
    return [(d, c[0], c[1]) for d, c in sorted(days.items(), reverse=True)]
