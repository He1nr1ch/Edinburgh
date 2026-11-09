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


def run_sweep(tickers: list[str], date: str, parts: int, log_dir: Path, tag: str = "daily",
              analysts: str | None = None, timing: dict | None = None) -> list[str]:
    """Analyse ``tickers`` on ``date`` in ``parts`` processes at once; returns the run ids.

    A part that fails is logged, not fatal: the others' calls still make a
    book, and a rerun with the same run ids resumes the cells left undone.
    ``timing``, when given, collects each part's ticker count, seconds and exit.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    per = -(-len(tickers) // parts)
    run_ids = run_ids_for(date, parts, tag)
    procs = []
    started = time.monotonic()
    for i, run_id in enumerate(run_ids):
        slice_ = tickers[i * per:(i + 1) * per]
        if not slice_:
            continue
        cmd = [sys.executable, "-m", "cli.main", "backtest", ",".join(slice_),
               "--start", date, "--end", date, "--run-id", run_id]
        if analysts:
            cmd += ["--analysts", analysts]
        log = (log_dir / f"{run_id}.log").open("w", encoding="utf-8")
        procs.append((run_id, len(slice_), subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL), log))
        logger.info("Sweep part %s: %d tickers", run_id, len(slice_))
    done = []
    for run_id, count, proc, log in procs:
        rc = proc.wait()
        log.close()
        seconds = round(time.monotonic() - started, 1)
        if timing is not None:
            timing.setdefault("parts", {})[run_id] = {"tickers": count, "seconds": seconds, "exit": rc}
        if rc == 0:
            done.append(run_id)
            logger.info("Sweep part %s done: %d tickers in %s", run_id, count, timedelta(seconds=int(seconds)))
        else:
            logger.error("Sweep part %s exited %d after %s; see %s", run_id, rc, timedelta(seconds=int(seconds)), log.name)
    return done


def run_daily(config: dict, equity: float, date: str | None = None, parts: int | None = None,
              refresh_watchlist: bool = True) -> BuiltBook:
    """The whole job: watchlist, sweep, book, bench. Returns the saved book."""
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
    tickers = watchlist_tickers(watchlist)
    started = datetime.now().astimezone()
    timing = {"date": date, "tickers": len(tickers), "parts": {}, "started": started.isoformat(timespec="seconds")}
    logger.info("Daily analysis of %d tickers on %s in %d parts", len(tickers), date, parts)
    run_ids = run_sweep(tickers, date, parts, Path(config["results_dir"]) / "daily",
                        analysts=config.get("daily_analysts") or None, timing=timing)
    sweep_seconds = (datetime.now().astimezone() - started).total_seconds()
    timing["sweep_seconds"] = round(sweep_seconds, 1)
    if not run_ids:
        raise RuntimeError("every sweep part failed; no book")
    logger.info("Sweep done in %s; sizing the book", timedelta(seconds=int(sweep_seconds)))
    built = build_book(run_ids, date, config, equity)
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
    return (f"{timing.get('tickers')} tickers in {total} ({timing.get('started', '')[11:16]} to "
            f"{timing.get('finished', '')[11:16]}); parts: {parts}")
