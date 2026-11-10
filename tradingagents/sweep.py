"""The sweep that never stops: the watchlist analysed around the clock, the Watcher first.

The worker drives it with :meth:`Sweep.tick` on every turn of its loop. A
cycle analyses the held positions first, then the watchlist least recently
analysed first, in rounds of a few tickers per process; when a cycle ends the
next begins, and when the session's close arrives (the last trading day
changes) the cycle restarts on the holdings against the new close, which is
when a fresh review matters most.

Priority for the Watcher during market hours is kept the only way shared
GPUs allow: the sweep then runs one process instead of three, so two of the
three PAIR nodes are free when the Watcher's own call comes, and its
processes run at a lower CPU priority than the worker. The stop and target
guards never touch the model and are unaffected either way. Rounds are
short, so the change of parallelism at the open and the close takes effect
within the hour without stopping a ticker midway.

State lives in ``sweep_state.json``: the cycle's date, the queue still to
analyse, the round in flight, so a restart resumes where it was.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from tradingagents.daily import order_tickers, watchlist_tickers

logger = logging.getLogger(__name__)

NICE = 10       # the sweep's processes yield the CPU to the worker


@dataclass
class Round:
    run_ids: list[str]
    tickers: list[str]
    started: float                      # time.monotonic()
    procs: list = field(default_factory=list)   # (run_id, Popen, log handle); empty after a restart


class Sweep:
    def __init__(self, config: dict):
        self.config = config
        self.path = Path(config["sweep_state_path"])
        self.log_dir = Path(config["results_dir"]) / "daily"
        self.date = ""              # the close the cycle analyses
        self.cycle = 0
        self.queue: list[str] = []
        self.round_no = 0
        self.round: Round | None = None
        self.analysed_this_cycle = 0
        self._load()

    # --- state -----------------------------------------------------------

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        self.date, self.cycle, self.queue = data.get("date", ""), int(data.get("cycle", 0)), list(data.get("queue", []))
        self.round_no = int(data.get("round_no", 0))
        self.analysed_this_cycle = int(data.get("analysed_this_cycle", 0))
        if data.get("round"):
            # A round a restart interrupted: its processes are gone. The tickers
            # go back to the front of the queue; the cells they finished are
            # skipped by the resumed run ids.
            r = data["round"]
            self.queue = [t for t in r.get("tickers", []) if t not in self.queue] + self.queue
            self.round = None
            logger.info("Sweep round %s was interrupted by a restart; its tickers are queued again.", r.get("run_ids"))

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {"date": self.date, "cycle": self.cycle, "queue": self.queue, "round_no": self.round_no,
                "analysed_this_cycle": self.analysed_this_cycle,
                "round": {"run_ids": self.round.run_ids, "tickers": self.round.tickers} if self.round else None,
                "saved": datetime.now().astimezone().isoformat(timespec="seconds")}
        self.path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    # --- the loop's hook -------------------------------------------------

    def tick(self, date: str, market_open: bool, holdings: list[str], last_analysed: dict[str, str]) -> None:
        """Advance the sweep: finish a round that is done, start a cycle or a round if none is running.

        ``date`` is the last close (the cycle restarts when it changes),
        ``holdings`` the held symbols (first in a new cycle), ``last_analysed``
        each ticker's newest decision date (least recent first after them).
        """
        if self.round is not None and self._round_done():
            self._finish_round()
        if self.round is not None:
            return
        if date != self.date or not self.queue:
            self._start_cycle(date, holdings, last_analysed)
        if self.queue:
            self._start_round(market_open)

    def parts_now(self, market_open: bool) -> int:
        return int(self.config["daily_parts_market_hours"] if market_open else self.config["daily_parts"])

    # --- cycles and rounds -----------------------------------------------

    def _start_cycle(self, date: str, holdings: list[str], last_analysed: dict[str, str]) -> None:
        watchlist = watchlist_tickers(Path(self.config["daily_watchlist"]))
        if self.date and self.date == date:
            # The same close, the list exhausted: go round again, holdings first.
            pass
        self.date, self.cycle, self.round_no, self.analysed_this_cycle = date, self.cycle + 1, 0, 0
        self.queue = order_tickers(watchlist, holdings, last_analysed)
        logger.info("Sweep cycle %d on %s's close: %d held positions first, then %d of the watchlist.",
                    self.cycle, date, len(holdings), len(self.queue) - len(holdings))
        self._save()

    def _start_round(self, market_open: bool) -> None:
        parts = max(1, self.parts_now(market_open))
        take = min(len(self.queue), parts * int(self.config["sweep_round_size"]))
        tickers, self.queue = self.queue[:take], self.queue[take:]
        self.round_no += 1
        tag = f"c{self.cycle}r{self.round_no}"
        run_ids = [f"sp500_{self.date.replace('-', '')}_{tag}_{chr(97 + i)}" for i in range(parts)]
        self.round = Round(run_ids, tickers, time.monotonic())
        self.log_dir.mkdir(parents=True, exist_ok=True)
        for i, run_id in enumerate(run_ids):
            slice_ = tickers[i::parts]
            if not slice_:
                continue
            cmd = [sys.executable, "-m", "cli.main", "backtest", ",".join(slice_),
                   "--start", self.date, "--end", self.date, "--run-id", run_id]
            if self.config.get("daily_analysts"):
                cmd += ["--analysts", self.config["daily_analysts"]]
            log = (self.log_dir / f"{run_id}.log").open("w", encoding="utf-8")
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                    preexec_fn=lambda: os.nice(NICE))
            self.round.procs.append((run_id, proc, log))
        logger.info("Sweep round %s: %d tickers in %d process(es)%s; %d left in the cycle.",
                    tag, len(tickers), parts, " (market hours: the Watcher has the other nodes)" if market_open else "",
                    len(self.queue))
        self._save()

    def _round_done(self) -> bool:
        return all(proc.poll() is not None for _, proc, _ in self.round.procs)

    def _finish_round(self) -> None:
        r = self.round
        seconds = time.monotonic() - r.started
        failed = [run_id for run_id, proc, _ in r.procs if proc.returncode != 0]
        for _, _, log in r.procs:
            log.close()
        self.analysed_this_cycle += len(r.tickers)
        logger.info("Sweep round %s done: %d tickers in %d min%s", r.run_ids[0].rsplit("_", 1)[0].rsplit("_", 1)[-1],
                    len(r.tickers), int(seconds // 60), f"; parts exited non-zero: {failed}" if failed else "")
        self.round = None
        self._save()

    def stop(self) -> None:
        """Stop the round in flight (a shutdown); its tickers return to the queue on the next start."""
        if self.round is None:
            return
        for _, proc, _ in self.round.procs:
            if proc.poll() is None:
                proc.terminate()
        self._save()

    def status(self) -> str:
        in_flight = f", round {self.round_no} running ({len(self.round.tickers)} tickers)" if self.round else ""
        return (f"cycle {self.cycle} on {self.date or '-'}: {self.analysed_this_cycle} analysed, "
                f"{len(self.queue)} queued{in_flight}")
