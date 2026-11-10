"""The 4 AM job: analyse the watchlist on the last close, size the book, bench the next-best.

Run by the worker on each trading day at ``daily_analysis_time`` (ET), or by
hand with ``tradingagents daily``. The sweep runs as separate processes, one
per part, as the hand-run sweeps did, so PAIR spreads them over its nodes;
the book and the bench are then built from the parts' calls and saved where
the page and the Watcher read them.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from tradingagents.book import BuiltBook, build_book, save_book
from tradingagents.hedge import describe_hedge, hedge_row

logger = logging.getLogger(__name__)


def watchlist_tickers(path: Path) -> list[str]:
    return [t for t in Path(path).read_text(encoding="utf-8").replace("\n", "").split(",") if t.strip()]


def run_ids_for(date: str, parts: int, tag: str = "daily") -> list[str]:
    return [f"sp500_{date.replace('-', '')}_{tag}_{chr(97 + i)}" for i in range(parts)]


def order_tickers(watchlist: list[str], holdings: list[str], last_analysed: dict[str, str],
                  date: str | None = None) -> list[str]:
    """The cycle's order: the held positions first, then the watchlist least recently analysed first.

    A holding already analysed on ``date`` (this cycle's close) is not reviewed
    again; it takes its place by recency with the rest. A ticker never analysed
    comes before any that has been; ties keep the watchlist's order.
    """
    held = [t for t in dict.fromkeys(h.upper() for h in holdings)
            if not (date and last_analysed.get(t) == date)]
    rest = [t for t in watchlist if t.upper() not in held]
    position = {t: i for i, t in enumerate(rest)}
    rest.sort(key=lambda t: (last_analysed.get(t.upper(), ""), position[t]))
    return held + rest


def _parse_deadline(text: str | None, now: datetime) -> datetime | None:
    """Today's (or, past it, tomorrow's) ``HH:MM`` in Eastern time."""
    if not text:
        return None
    from zoneinfo import ZoneInfo

    et = now.astimezone(ZoneInfo("America/New_York"))
    hour, minute = (int(x) for x in str(text).split(":"))
    at = et.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return at if at > et else at + timedelta(days=1)


def run_sweep(tickers: list[str], date: str, parts: int, log_dir: Path, tag: str = "daily",
              analysts: str | None = None, timing: dict | None = None,
              deadline: datetime | None = None) -> list[str]:
    """Analyse ``tickers`` on ``date`` in ``parts`` processes at once; returns the run ids.

    A part that fails is logged, not fatal: the others' calls still make a
    book, and a rerun with the same run ids resumes the cells left undone.
    ``timing``, when given, collects each part's ticker count, seconds and exit.
    The tickers are dealt round-robin, so every part starts with the front of
    the list (the held positions). At ``deadline`` the parts still running are
    stopped: the tickers they finished stand, the ticker in progress is lost,
    and the rest wait for the next night.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    run_ids = run_ids_for(date, parts, tag)
    procs = []
    started = time.monotonic()
    for i, run_id in enumerate(run_ids):
        slice_ = tickers[i::parts]
        if not slice_:
            continue
        cmd = [sys.executable, "-m", "cli.main", "backtest", ",".join(slice_),
               "--start", date, "--end", date, "--run-id", run_id]
        if analysts:
            cmd += ["--analysts", analysts]
        log = (log_dir / f"{run_id}.log").open("w", encoding="utf-8")
        procs.append((run_id, len(slice_), subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL), log))
        logger.info("Sweep part %s: %d tickers", run_id, len(slice_))
    if deadline is not None:
        while any(proc.poll() is None for _, _, proc, _ in procs):
            if datetime.now(deadline.tzinfo) >= deadline:
                for run_id, _, proc, _ in procs:
                    if proc.poll() is None:
                        logger.info("Sweep part %s stopped at the deadline, %s", run_id, deadline.strftime("%H:%M"))
                        proc.terminate()
                if timing is not None:
                    timing["stopped_at_deadline"] = True
                break
            time.sleep(15)
    done = []
    for run_id, count, proc, log in procs:
        try:
            rc = proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            proc.kill()
            rc = proc.wait()
        log.close()
        stopped = deadline is not None and (timing or {}).get("stopped_at_deadline")
        seconds = round(time.monotonic() - started, 1)
        if timing is not None:
            timing.setdefault("parts", {})[run_id] = {"tickers": count, "seconds": seconds, "exit": rc}
        if rc == 0 or stopped:
            done.append(run_id)       # a part stopped at the deadline still has its finished tickers
            logger.info("Sweep part %s %s: %d tickers given, in %s", run_id,
                        "done" if rc == 0 else "stopped", count, timedelta(seconds=int(seconds)))
        else:
            logger.error("Sweep part %s exited %d after %s; see %s", run_id, rc, timedelta(seconds=int(seconds)), log.name)
    return done


def run_daily(config: dict, equity: float, date: str | None = None, parts: int | None = None,
              refresh_watchlist: bool = True, holdings: list[str] | None = None) -> BuiltBook:
    """The whole job: holdings and watchlist, sweep to the deadline, book, bench. Returns the saved book."""
    from tradingagents import sp500
    from tradingagents.dataflows.vendors.alpaca import trading

    date = date or trading.last_trading_day(datetime.now().astimezone())
    parts = parts or int(config["daily_parts"])
    watchlist = Path(config["daily_watchlist"])
    if not watchlist.is_file() and sp500.WATCHLIST.is_file():
        watchlist.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(sp500.WATCHLIST, watchlist)   # the repo's copy seeds the data folder's
        logger.info("Seeded %s from %s", watchlist, sp500.WATCHLIST)
    if refresh_watchlist:
        try:
            sp500.refresh_if_stale(path=watchlist)
        except Exception as exc:     # a stale list still runs; a missing one cannot
            logger.warning("The watchlist could not be refreshed: %s", exc)
    from tradingagents.web.data import all_decisions

    if holdings is None:
        holdings = [h.symbol for h in trading.get_account().holdings if h.asset_class == "us_equity"]
    last = {t: refs[0].date for t, refs in all_decisions(Path(config["results_dir"])).items()}
    tickers = order_tickers(watchlist_tickers(watchlist), holdings, last)
    started = datetime.now().astimezone()
    deadline = _parse_deadline(config.get("daily_deadline"), started)
    timing = {"date": date, "tickers": len(tickers), "holdings": len(holdings), "parts": {},
              "started": started.isoformat(timespec="seconds"),
              "deadline": deadline.isoformat(timespec="minutes") if deadline else None}
    logger.info("Daily analysis on %s: %d held positions first, then %d of the watchlist, in %d parts, until %s",
                date, len(holdings), len(tickers) - len(holdings), parts, deadline.strftime("%H:%M") if deadline else "done")
    run_ids = run_sweep(tickers, date, parts, Path(config["results_dir"]) / "daily",
                        analysts=config.get("daily_analysts") or None, timing=timing, deadline=deadline)
    sweep_seconds = (datetime.now().astimezone() - started).total_seconds()
    timing["sweep_seconds"] = round(sweep_seconds, 1)
    if not run_ids:
        raise RuntimeError("every sweep part failed; no book")
    logger.info("Sweep done in %s; sizing the book", timedelta(seconds=int(sweep_seconds)))
    from tradingagents.book import recent_calls

    calls = recent_calls(Path(config["results_dir"]), date, int(config.get("decision_max_age_days") or 0))
    timing["analysed_tonight"] = sum(1 for t in tickers if last_analysed_on(Path(config["results_dir"]), t, date))
    timing["calls_in_window"] = len(calls)
    built = build_book(run_ids, date, config, equity, calls=calls or None)
    if config.get("daily_hedge", True):
        try:
            built.hedge = price_hedge(built, date, config)
        except Exception as exc:     # the book stands unhedged; the log and the email say so
            logger.warning("The hedge could not be priced: %s", exc)
    built = save_book(built, config, hedge_row(built.hedge) if built.hedge else None)
    logger.info("Hedge: %s", describe_hedge(built.hedge))
    finished = datetime.now().astimezone()
    timing.update(finished=finished.isoformat(timespec="seconds"),
                  total_seconds=round((finished - started).total_seconds(), 1),
                  book=str(built.book_path), orders=len(built.book.orders), bench=len(built.bench),
                  hedge=describe_hedge(built.hedge))
    built.timing = timing
    built.timing_path = built.book_path.with_name(built.book_path.stem + "_timing.json")
    built.timing_path.write_text(json.dumps(timing, indent=2) + "\n", encoding="utf-8")
    logger.info("Book: %d orders, bench %d, in %s all told; %s", len(built.book.orders), len(built.bench),
                timedelta(seconds=int(timing["total_seconds"])), built.book_path)
    return built


def build_morning_book(config: dict, equity: float, date: str | None = None) -> BuiltBook:
    """The day's book from every decision still fresh, however it was analysed.

    The continuous sweep leaves decisions behind as it goes; this sizes the
    book from those under ``decision_max_age_days`` old, prices the hedge and
    saves book, bench and a coverage note. The run ids name the book after the
    newest decisions it drew on.
    """
    from tradingagents.book import recent_calls
    from tradingagents.dataflows.vendors.alpaca import trading
    from tradingagents.web.data import all_decisions

    date = date or trading.last_trading_day(datetime.now().astimezone())
    results = Path(config["results_dir"])
    calls = recent_calls(results, date, int(config.get("decision_max_age_days") or 0))
    if not calls:
        raise RuntimeError(f"no decision under {config.get('decision_max_age_days')} days old to build a book from")
    newest = sorted({r[0].date for r in all_decisions(results).values()}, reverse=True)[:1]
    run_ids = [f"fresh_{(newest or [date])[0].replace('-', '')}"]
    started = datetime.now().astimezone()
    built = build_book(run_ids, date, config, equity, calls=calls)
    if config.get("daily_hedge", True):
        try:
            built.hedge = price_hedge(built, date, config)
        except Exception as exc:
            logger.warning("The hedge could not be priced: %s", exc)
    built = save_book(built, config, hedge_row(built.hedge) if built.hedge else None)
    fresh_days = sorted({c.decided for c in calls if c.decided})
    timing = {"date": date, "started": started.isoformat(timespec="seconds"),
              "finished": datetime.now().astimezone().isoformat(timespec="seconds"),
              "total_seconds": round((datetime.now().astimezone() - started).total_seconds(), 1),
              "calls_in_window": len(calls), "decided_between": [fresh_days[0], fresh_days[-1]] if fresh_days else [],
              "book": str(built.book_path), "orders": len(built.book.orders), "bench": len(built.bench),
              "hedge": describe_hedge(built.hedge)}
    built.timing = timing
    built.timing_path = built.book_path.with_name(built.book_path.stem + "_timing.json")
    built.timing_path.write_text(json.dumps(timing, indent=2) + "\n", encoding="utf-8")
    logger.info("Morning book: %d orders, bench %d, from %d decisions dated %s; %s",
                len(built.book.orders), len(built.bench), len(calls), " to ".join(timing["decided_between"]), built.book_path)
    return built


def last_analysed_on(results_dir: Path, ticker: str, date: str) -> bool:
    return any(Path(results_dir).glob(f"backtest/*/{ticker}/TradingAgentsStrategy_logs/full_states_log_{date}.json"))


def price_hedge(built: BuiltBook, date: str, config: dict):
    """The SPY call hedge for the sized book, net of the calls the ledger says are still held."""
    from tradingagents.hedge import hedge_book, open_hedges

    return hedge_book(built.book, date, monthly_ratio=float(config["hedge_monthly_ratio"]),
                      held=open_hedges(Path(config["hedge_ledger_path"]), date))


def describe_timing(timing: dict) -> str:
    """One line for the log and the email: how long the job took, by part."""
    total = timedelta(seconds=int(timing.get("total_seconds") or 0))
    parts = ", ".join(f"{rid.rsplit('_', 1)[-1]} {p['tickers']} in {timedelta(seconds=int(p['seconds']))}"
                      for rid, p in timing.get("parts", {}).items())
    if "decided_between" in timing:
        span = " to ".join(timing["decided_between"]) or "-"
        return f"book from {timing.get('calls_in_window')} decisions dated {span}, built in {timing.get('total_seconds')}s"
    covered = (f"{timing.get('analysed_tonight')} of {timing.get('tickers')} tickers analysed"
               if "analysed_tonight" in timing else f"{timing.get('tickers')} tickers")
    stop = ", stopped at the deadline" if timing.get("stopped_at_deadline") else ""
    window = f"; book from {timing['calls_in_window']} fresh decisions" if "calls_in_window" in timing else ""
    return (f"{covered} in {total} ({timing.get('started', '')[11:16]} to "
            f"{timing.get('finished', '')[11:16]}{stop}){window}; parts: {parts}")
