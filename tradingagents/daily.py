"""The 4 AM job: analyse the watchlist on the last close, size the book, bench the next-best.

Run by the worker on each trading day at ``daily_analysis_time`` (ET), or by
hand with ``tradingagents daily``. The sweep runs as separate processes, one
per part, as the hand-run sweeps did, so PAIR spreads them over its nodes;
the book and the bench are then built from the parts' calls and saved where
the page and the Watcher read them.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from tradingagents.book import BuiltBook, build_book, save_book

logger = logging.getLogger(__name__)


def watchlist_tickers(path: Path) -> list[str]:
    return [t for t in Path(path).read_text(encoding="utf-8").replace("\n", "").split(",") if t.strip()]


def run_ids_for(date: str, parts: int, tag: str = "daily") -> list[str]:
    return [f"sp500_{date.replace('-', '')}_{tag}_{chr(97 + i)}" for i in range(parts)]


def run_sweep(tickers: list[str], date: str, parts: int, log_dir: Path, tag: str = "daily",
              analysts: str | None = None) -> list[str]:
    """Analyse ``tickers`` on ``date`` in ``parts`` processes at once; returns the run ids.

    A part that fails is logged, not fatal: the others' calls still make a
    book, and a rerun with the same run ids resumes the cells left undone.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    per = -(-len(tickers) // parts)
    run_ids = run_ids_for(date, parts, tag)
    procs = []
    for i, run_id in enumerate(run_ids):
        slice_ = tickers[i * per:(i + 1) * per]
        if not slice_:
            continue
        cmd = [sys.executable, "-m", "cli.main", "backtest", ",".join(slice_),
               "--start", date, "--end", date, "--run-id", run_id]
        if analysts:
            cmd += ["--analysts", analysts]
        log = (log_dir / f"{run_id}.log").open("w", encoding="utf-8")
        procs.append((run_id, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL), log))
        logger.info("Sweep part %s: %d tickers", run_id, len(slice_))
    done = []
    for run_id, proc, log in procs:
        rc = proc.wait()
        log.close()
        if rc == 0:
            done.append(run_id)
        else:
            logger.error("Sweep part %s exited %d; see %s", run_id, rc, log.name)
    return done


def run_daily(config: dict, equity: float, date: str | None = None, parts: int | None = None,
              refresh_watchlist: bool = True) -> BuiltBook:
    """The whole job: watchlist, sweep, book, bench. Returns the saved book."""
    from tradingagents import sp500
    from tradingagents.dataflows.vendors.alpaca import trading

    date = date or trading.last_trading_day(datetime.now().astimezone())
    parts = parts or int(config["daily_parts"])
    watchlist = Path(config["daily_watchlist"])
    if refresh_watchlist:
        try:
            sp500.refresh_if_stale(path=watchlist)
        except Exception as exc:     # a stale list still runs; a missing one cannot
            logger.warning("The watchlist could not be refreshed: %s", exc)
    tickers = watchlist_tickers(watchlist)
    started = datetime.now()
    logger.info("Daily analysis of %d tickers on %s in %d parts", len(tickers), date, parts)
    run_ids = run_sweep(tickers, date, parts, Path(config["results_dir"]) / "daily",
                        analysts=config.get("daily_analysts") or None)
    if not run_ids:
        raise RuntimeError("every sweep part failed; no book")
    logger.info("Sweep done in %s; sizing the book", datetime.now() - started)
    built = save_book(build_book(run_ids, date, config, equity), config)
    logger.info("Book: %d orders, bench %d; %s", len(built.book.orders), len(built.bench), built.book_path)
    return built
