"""The sweep that never stops: the watchlist analysed around the clock, the Watcher first.

The worker drives it with :meth:`Sweep.tick` on every turn of its loop. A
cycle analyses the held positions first, then the watchlist least recently
analysed first; when a cycle ends the next begins, and when the session's
close arrives (the last trading day changes) the cycle restarts on the
holdings against the new close, which is when a fresh review matters most.

The work is done by a pool of processes, each taking a few tickers from the
front of the queue (``sweep_round_size``) and, the moment it is done, the
next few. No process waits for another: when they ran in rounds, a round
ended only when its slowest process did, and each process stood idle for six
minutes in every fifty-four. The pool holds ``daily_parts`` processes
outside market hours and ``daily_parts_market_hours`` during them. There are
more processes than PAIR nodes on purpose: a process spends part of its
time fetching, parsing and waiting on its own turn, and with one process a
node the nodes stood idle a third of the night.

The Watcher's priority during market hours is kept by the smaller pool, and
by the sweep's processes running at a lower CPU priority than the worker.
The stop and target guards never touch the model and are unaffected either
way. A run is short, so the change of pool size at the open and the close
takes effect within the hour without stopping a ticker midway.

State lives in ``sweep_state.json``: the cycle's date, the queue still to
analyse, the runs in flight and the runs finished, so a restart resumes
where it was and anything that reads the file (the positions page, the
daily report, ``/health``) can say how far the cycle has come and when it
should end: see :func:`progress`.
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

from tradingagents.daily import ensure_watchlist, order_tickers, watchlist_tickers

logger = logging.getLogger(__name__)

NICE = 10       # the sweep's processes yield the CPU to the worker
PACE_ROUNDS = 10  # how many finished runs the pace (and the finish estimate) is taken from


@dataclass
class Run:
    """One process of the pool at work on a few tickers."""
    run_id: str
    tickers: list[str]
    date: str                           # the close it analyses
    started: float                      # time.monotonic()
    started_at: str = ""                # wall clock, for the state file
    proc: object = None
    log: object = None


class Sweep:
    def __init__(self, config: dict):
        self.config = config
        self.path = Path(config["sweep_state_path"])
        self.log_dir = Path(config["results_dir"]) / "daily"
        self.date = ""              # the close the cycle analyses
        self.cycle = 0
        self.queue: list[str] = []
        self.round_no = 0           # runs started this cycle
        self.runs: list[Run] = []   # the pool: the processes at work now
        self.analysed_this_cycle = 0
        self.rounds: list[dict] = []     # finished runs this cycle: round, tickers, parts, minutes, finished
        self.retried: list[str] = []     # tickers whose failed run was queued again this cycle (once each)
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
        self.rounds = list(data.get("rounds", []))
        self.retried = list(data.get("retried", []))
        # Runs a restart interrupted: their processes are gone. Their tickers go
        # back to the front of the queue; the cells they finished are skipped
        # by the resumed analysis. ("round" is the state of before the pool.)
        interrupted = [t for r in data.get("runs") or ([data["round"]] if data.get("round") else [])
                       for t in r.get("tickers", [])]
        if interrupted:
            self.queue = [t for t in dict.fromkeys(interrupted) if t not in self.queue] + self.queue
            logger.info("A restart interrupted the sweep's runs; %d tickers are queued again.", len(interrupted))

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        runs = [{"run_id": r.run_id, "tickers": r.tickers, "started": r.started_at} for r in self.runs]
        data = {"date": self.date, "cycle": self.cycle, "queue": self.queue, "round_no": self.round_no,
                "analysed_this_cycle": self.analysed_this_cycle, "rounds": self.rounds[-50:], "retried": self.retried,
                "runs": runs,
                "saved": datetime.now().astimezone().isoformat(timespec="seconds")}
        self.path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    # --- the loop's hook -------------------------------------------------

    def due(self, market_open: bool) -> bool:
        """Whether a tick has work to do now: a run has ended, or the pool has a place free."""
        return any(r.proc.poll() is not None for r in self.runs) or len(self.runs) < self.parts_now(market_open)

    def tick(self, date: str, market_open: bool, holdings: list[str], last_analysed: dict[str, str]) -> None:
        """Advance the sweep: take in the runs that have ended, start a cycle if one is due, fill the pool.

        ``date`` is the last close (the cycle restarts when it changes),
        ``holdings`` the held symbols (first in a new cycle), ``last_analysed``
        each ticker's newest decision date (least recent first after them).
        """
        for run in [r for r in self.runs if r.proc.poll() is not None]:
            self._finish_run(run)
        # A new close starts the next cycle at once; the runs still at work on
        # the old one finish on their own. A queue that has run out starts the
        # list again once the last run is in (a failure may yet queue a ticker).
        if date != self.date or (not self.queue and not self.runs):
            self._start_cycle(date, holdings, last_analysed)
        while self.queue and len(self.runs) < self.parts_now(market_open):
            self._start_run(market_open)

    def parts_now(self, market_open: bool) -> int:
        return max(1, int(self.config["daily_parts_market_hours"] if market_open else self.config["daily_parts"]))

    # --- cycles and runs -------------------------------------------------

    def _start_cycle(self, date: str, holdings: list[str], last_analysed: dict[str, str]) -> None:
        watchlist = watchlist_tickers(ensure_watchlist(self.config))
        self.date, self.cycle, self.round_no, self.analysed_this_cycle = date, self.cycle + 1, 0, 0
        self.rounds, self.retried = [], []
        self.queue = order_tickers(watchlist, holdings, last_analysed, date)
        logger.info("Sweep cycle %d on %s's close: %d held positions first, then %d of the watchlist.",
                    self.cycle, date, len(holdings), len(self.queue) - len(holdings))
        self._save()

    def _start_run(self, market_open: bool) -> None:
        take = min(len(self.queue), max(1, int(self.config["sweep_round_size"])))
        tickers, self.queue = self.queue[:take], self.queue[take:]
        self.round_no += 1
        # A letter per place in the pool, the first one free, so two runs at
        # work never share a name; the number counts the cycle's runs.
        letters = {r.run_id[-1] for r in self.runs}
        letter = next(chr(97 + i) for i in range(26 + len(letters)) if chr(97 + i) not in letters)
        run_id = f"sp500_{self.date.replace('-', '')}_c{self.cycle}r{self.round_no}_{letter}"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-m", "cli.main", "backtest", ",".join(tickers),
               "--start", self.date, "--end", self.date, "--run-id", run_id]
        if self.config.get("daily_analysts"):
            cmd += ["--analysts", self.config["daily_analysts"]]
        log = (self.log_dir / f"{run_id}.log").open("w", encoding="utf-8")
        # The child reads its config from the environment; the folders the
        # worker uses go with it, so both write the same results.
        env = {**os.environ, "TRADINGAGENTS_RESULTS_DIR": str(self.config["results_dir"]),
               "TRADINGAGENTS_CACHE_DIR": str(self.config["data_cache_dir"])}
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                preexec_fn=lambda: os.nice(NICE), env=env)
        self.runs.append(Run(run_id, tickers, self.date, time.monotonic(),
                             datetime.now().astimezone().isoformat(timespec="seconds"), proc, log))
        logger.info("Sweep run %s: %s; %d process(es) at work%s; %d left in the cycle.",
                    run_id, ", ".join(tickers), len(self.runs),
                    " (market hours: the pool is smaller, for the Watcher)" if market_open else "", len(self.queue))
        self._save()

    def _finish_run(self, run: Run) -> None:
        seconds = time.monotonic() - run.started
        run.log.close()
        parts = len(self.runs)                 # how many were at work beside it, itself included
        self.runs.remove(run)
        tag = run.run_id.split("_")[-2]
        if run.date != self.date:
            # It analysed the close before this cycle's: its decisions stand,
            # and the new cycle has those tickers in its own queue.
            logger.info("Sweep run %s (on %s's close) done: %d tickers in %d min.", run.run_id, run.date, len(run.tickers), int(seconds // 60))
            self._save()
            return
        again = [t for t in self._failed(run) if t not in self.queue]
        self.queue += again
        self.retried += again
        self.analysed_this_cycle += len(run.tickers) - len(again)
        self.rounds.append({"round": tag, "tickers": len(run.tickers) - len(again), "parts": parts, "pool": True,
                            "minutes": round(seconds / 60, 1),
                            "finished": datetime.now().astimezone().isoformat(timespec="seconds")})
        self._save()
        p = self.progress()
        logger.info("Sweep run %s done: %d tickers in %d min%s%s; %d of %d analysed this cycle, %d queued%s", run.run_id,
                    len(run.tickers), int(seconds // 60), f"; exited {run.proc.returncode}" if run.proc.returncode else "",
                    f"; failed and queued again: {', '.join(again)}" if again else "",
                    p["analysed"], p["total"], p["queued"], f", about {describe_eta(p)} to go" if p.get("eta_hours") else "")

    def _failed(self, r: Run) -> list[str]:
        """The run's tickers whose analysis failed (see ``tradingagents.errors``) and may be tried once more this cycle.

        They go to the back of the queue: whatever failed them (the model
        silent, a vendor down) has the rest of the cycle to clear. A ticker
        that fails twice in a cycle waits for the next one.
        """
        from tradingagents import errors

        try:
            days = sorted({r.started_at[:10], datetime.now().astimezone().isoformat()[:10]} - {""})
            failed = {t for day in days for t in errors.failed_tickers(day, r.started_at, r.tickers, self.config)}
        except Exception as exc:
            logger.warning("The run's failures could not be read: %s", exc)
            return []
        return [t for t in r.tickers if t in failed and t not in self.retried]

    def stop(self) -> None:
        """Stop the runs in flight (a shutdown); their tickers return to the queue on the next start."""
        for run in self.runs:
            if run.proc.poll() is None:
                run.proc.terminate()
        self._save()

    def progress(self) -> dict:
        return progress(json.loads(self.path.read_text(encoding="utf-8")) if self.path.is_file() else {})

    def status(self) -> str:
        in_flight = f", {len(self.runs)} run(s) at work ({sum(len(r.tickers) for r in self.runs)} tickers)" if self.runs else ""
        return (f"cycle {self.cycle} on {self.date or '-'}: {self.analysed_this_cycle} analysed, "
                f"{len(self.queue)} queued{in_flight}")


def progress(state: dict, now: datetime | None = None) -> dict:
    """How far the cycle has come, from the state file: counts, the pace of recent runs, a finish estimate.

    The pace is wall-clock minutes per ticker over the last few finished runs:
    a run's own minutes over its tickers, divided by the number of processes
    at work beside it, so a change of pool size at the open or the close shows
    within the hour.
    """
    if not state:
        return {}
    runs = state.get("runs")
    if runs is None:                    # the state of before the pool: one round of several processes
        old = state.get("round") or {}
        runs = [{"tickers": old.get("tickers", []), "started": old.get("started")}] if old else []
        pool = int(old.get("parts") or 0)
    else:
        pool = len(runs)
    in_flight = sum(len(r.get("tickers", [])) for r in runs)
    started = min((r["started"] for r in runs if r.get("started")), default=None)
    analysed, queued = int(state.get("analysed_this_cycle", 0)), len(state.get("queue", []))
    recent = [r for r in state.get("rounds", [])[-PACE_ROUNDS:] if r.get("tickers") and r.get("minutes")]
    out = {"cycle": state.get("cycle"), "date": state.get("date"), "analysed": analysed, "in_flight": in_flight,
           "queued": queued, "total": analysed + in_flight + queued, "round": state.get("round_no"),
           "round_started": started, "parts": pool or None,
           "rounds_done": len(state.get("rounds", [])), "saved": state.get("saved")}
    if recent:
        # A pool's run took its minutes beside ``parts`` - 1 others; a round of
        # before the pool recorded the whole round's minutes already.
        wall = sum(r["minutes"] / (max(1, r.get("parts") or 1) if r.get("pool") else 1) for r in recent)
        minutes_per_ticker = wall / sum(r["tickers"] for r in recent)
        out["minutes_per_round"] = round(sum(r["minutes"] for r in recent) / len(recent), 1)
        out["tickers_per_round"] = round(sum(r["tickers"] for r in recent) / len(recent), 1)
        out["minutes_per_ticker"] = round(minutes_per_ticker, 2)
        # What is at work now is taken as half done, on average.
        remaining = queued + in_flight / 2
        out["eta_hours"] = round(max(0.0, remaining * minutes_per_ticker) / 60, 1)
        out["pct"] = round(100 * analysed / out["total"], 1) if out["total"] else 0.0
    return out


def describe_eta(p: dict) -> str:
    hours = p.get("eta_hours")
    if hours is None:
        return ""
    if hours < 1:
        return f"{int(round(hours * 60))} min"
    if hours < 48:
        return f"{hours:.1f} h"
    return f"{hours / 24:.1f} days"


def describe(p: dict) -> str:
    """One line: 'cycle 2 on 2026-11-10: 36 of 515 analysed (7%), 12 in flight, 467 queued, 43 min per run of 4, about 29 h to go'."""
    if not p:
        return "no sweep yet"
    text = f"cycle {p['cycle']} on {p['date']}: {p['analysed']} of {p['total']} analysed"
    if "pct" in p:
        text += f" ({p['pct']:.0f}%)"
    text += f", {p['in_flight']} in flight, {p['queued']} queued"
    if p.get("minutes_per_round"):
        text += f", {p['minutes_per_round']:.0f} min per run of {p['tickers_per_round']:.0f}"
    if p.get("eta_hours") is not None:
        text += f", about {describe_eta(p)} to go"
    return text
