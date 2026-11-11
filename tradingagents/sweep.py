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
analyse, the round in flight and the rounds finished, so a restart resumes
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
PACE_ROUNDS = 6  # how many finished rounds the pace (and the finish estimate) is taken from


@dataclass
class Round:
    run_ids: list[str]
    tickers: list[str]
    started: float                      # time.monotonic()
    parts: int = 1
    started_at: str = ""                # wall clock, for the state file
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
        self.rounds: list[dict] = []     # finished rounds this cycle: round, tickers, parts, minutes, finished
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
                "analysed_this_cycle": self.analysed_this_cycle, "rounds": self.rounds[-50:], "retried": self.retried,
                "round": {"run_ids": self.round.run_ids, "tickers": self.round.tickers, "parts": self.round.parts,
                          "started": self.round.started_at} if self.round else None,
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
        watchlist = watchlist_tickers(ensure_watchlist(self.config))
        self.date, self.cycle, self.round_no, self.analysed_this_cycle = date, self.cycle + 1, 0, 0
        self.rounds, self.retried = [], []
        self.queue = order_tickers(watchlist, holdings, last_analysed, date)
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
        self.round = Round(run_ids, tickers, time.monotonic(), parts, datetime.now().astimezone().isoformat(timespec="seconds"))
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
            # The child reads its config from the environment; the folders the
            # worker uses go with it, so both write the same results.
            env = {**os.environ, "TRADINGAGENTS_RESULTS_DIR": str(self.config["results_dir"]),
                   "TRADINGAGENTS_CACHE_DIR": str(self.config["data_cache_dir"])}
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                    preexec_fn=lambda: os.nice(NICE), env=env)
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
        again = self._failed(r)
        self.queue += again
        self.retried += again
        self.analysed_this_cycle += len(r.tickers) - len(again)
        tag = r.run_ids[0].rsplit("_", 1)[0].rsplit("_", 1)[-1]
        self.rounds.append({"round": tag, "tickers": len(r.tickers) - len(again), "parts": r.parts, "minutes": round(seconds / 60, 1),
                            "finished": datetime.now().astimezone().isoformat(timespec="seconds")})
        self.round = None
        self._save()
        p = self.progress()
        logger.info("Sweep round %s done: %d tickers in %d min%s%s; %d of %d analysed this cycle, %d queued%s", tag,
                    len(r.tickers), int(seconds // 60), f"; parts exited non-zero: {failed}" if failed else "",
                    f"; failed and queued again: {', '.join(again)}" if again else "",
                    p["analysed"], p["total"], p["queued"], f", about {describe_eta(p)} to go" if p.get("eta_hours") else "")

    def _failed(self, r: Round) -> list[str]:
        """The round's tickers whose run failed (see ``tradingagents.errors``) and may be tried once more this cycle.

        They go to the back of the queue: whatever failed them (the model
        silent, a vendor down) has the rest of the cycle to clear. A ticker
        that fails twice in a cycle waits for the next one.
        """
        from tradingagents import errors

        try:
            days = sorted({r.started_at[:10], datetime.now().astimezone().isoformat()[:10]} - {""})
            failed = {t for day in days for t in errors.failed_tickers(day, r.started_at, r.tickers, self.config)}
        except Exception as exc:
            logger.warning("The round's failures could not be read: %s", exc)
            return []
        return [t for t in r.tickers if t in failed and t not in self.retried]

    def stop(self) -> None:
        """Stop the round in flight (a shutdown); its tickers return to the queue on the next start."""
        if self.round is None:
            return
        for _, proc, _ in self.round.procs:
            if proc.poll() is None:
                proc.terminate()
        self._save()

    def progress(self) -> dict:
        return progress(json.loads(self.path.read_text(encoding="utf-8")) if self.path.is_file() else {})

    def status(self) -> str:
        in_flight = f", round {self.round_no} running ({len(self.round.tickers)} tickers)" if self.round else ""
        return (f"cycle {self.cycle} on {self.date or '-'}: {self.analysed_this_cycle} analysed, "
                f"{len(self.queue)} queued{in_flight}")


def progress(state: dict, now: datetime | None = None) -> dict:
    """How far the cycle has come, from the state file: counts, the pace of recent rounds, a finish estimate.

    The pace is minutes per ticker over the last few finished rounds, so a
    change of parallelism at the open or the close shows within an hour.
    """
    if not state:
        return {}
    in_flight = len((state.get("round") or {}).get("tickers", []))
    analysed, queued = int(state.get("analysed_this_cycle", 0)), len(state.get("queue", []))
    recent = [r for r in state.get("rounds", [])[-PACE_ROUNDS:] if r.get("tickers") and r.get("minutes")]
    out = {"cycle": state.get("cycle"), "date": state.get("date"), "analysed": analysed, "in_flight": in_flight,
           "queued": queued, "total": analysed + in_flight + queued, "round": state.get("round_no"),
           "round_started": (state.get("round") or {}).get("started"), "parts": (state.get("round") or {}).get("parts"),
           "rounds_done": len(state.get("rounds", [])), "saved": state.get("saved")}
    if recent:
        minutes_per_ticker = sum(r["minutes"] for r in recent) / sum(r["tickers"] for r in recent)
        out["minutes_per_round"] = round(sum(r["minutes"] for r in recent) / len(recent), 1)
        out["tickers_per_round"] = round(sum(r["tickers"] for r in recent) / len(recent), 1)
        out["minutes_per_ticker"] = round(minutes_per_ticker, 2)
        remaining = queued + in_flight
        elapsed = 0.0
        started = out["round_started"]
        if started and in_flight:
            try:
                elapsed = max(0.0, ((now or datetime.now().astimezone()) - datetime.fromisoformat(started)).total_seconds() / 60)
            except ValueError:
                elapsed = 0.0
        out["eta_hours"] = round(max(0.0, remaining * minutes_per_ticker - elapsed) / 60, 1)
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
    """One line: 'cycle 2 on 2026-11-10: 36 of 515 analysed (7%), 12 in flight, 467 queued, 43 min per round, about 29 h to go'."""
    if not p:
        return "no sweep yet"
    text = f"cycle {p['cycle']} on {p['date']}: {p['analysed']} of {p['total']} analysed"
    if "pct" in p:
        text += f" ({p['pct']:.0f}%)"
    text += f", {p['in_flight']} in flight, {p['queued']} queued"
    if p.get("minutes_per_round"):
        text += f", {p['minutes_per_round']:.0f} min per round of {p['tickers_per_round']:.0f}"
    if p.get("eta_hours") is not None:
        text += f", about {describe_eta(p)} to go"
    return text
