"""What went wrong with the models, kept where a person will see it.

A model call that fails is raised and handled where it happened: a ticker's
run fails, the Watcher waits, a contract closes a run. Each of those leaves a
line in a log file of its own, and none of them told anyone. This is the
desk's record of them: one line per failure in
``<results_dir>/errors/<date>.jsonl``, by kind:

- ``timeout``      a call passed its idle or total time limit and was dropped
- ``unavailable``  the model server could not be reached
- ``context``      the prompt did not fit the context window
- ``contract``     an agent's answer did not meet its schema after its retry
- ``run``          a ticker's run failed, whatever the cause (a vendor, a model)

The daily report counts them (:func:`summarize`), the sweep reads the ``run``
lines to put a failed ticker back in its queue, and the worker calls
:func:`watch` on every turn of its loop: when the model has given no answer
for ``model_outage_minutes`` it sends one email, and one more when the model
answers again.
"""

from __future__ import annotations

import json
import logging
import socket
from collections import Counter
from collections.abc import Callable
from datetime import date, datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

KINDS = ("timeout", "unavailable", "context", "contract", "run")
NO_ANSWER = ("timeout", "unavailable")      # the model gave no answer at all: what an outage is made of
_HOST = socket.gethostname()


def errors_dir(config: dict | None = None) -> Path:
    if config is None:
        from tradingagents.dataflows.config import get_config

        config = get_config()
    return Path(config.get("errors_dir") or Path(config["results_dir"]) / "errors")


def record(kind: str, message: str, agent: str = "", model: str = "", ticker: str | None = None) -> dict | None:
    """Add one failure to the day's log; the ticker and run come from the run in progress when not given.

    Never raises: a failure to write the record must not replace the failure it records.
    """
    try:
        from tradingagents import usage

        where = usage.context()
        rec = {
            "ts": datetime.now().astimezone().isoformat(timespec="seconds"), "host": _HOST, "kind": kind,
            "agent": agent or where["agent"], "ticker": ticker if ticker is not None else where["ticker"],
            "run_id": where["run_id"], "model": model, "message": " ".join(str(message).split())[:400],
        }
        folder = errors_dir()
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / f"{rec['ts'][:10]}.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        return rec
    except Exception as exc:
        logger.debug("Error record not written: %s", exc)
        return None


def read_day(day: str, config: dict | None = None) -> list[dict]:
    path = errors_dir(config) / f"{day}.jsonl"
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def summarize(rows: list[dict]) -> dict:
    """Counts by kind and by agent, the tickers whose run failed, and the last few messages."""
    model_rows = [r for r in rows if r.get("kind") != "run"]
    failed: dict[str, str] = {}
    for r in rows:
        if r.get("kind") == "run" and r.get("ticker"):
            failed[r["ticker"]] = r.get("message", "")
    return {
        "total": len(model_rows),
        "by_kind": dict(Counter(r.get("kind", "?") for r in model_rows).most_common()),
        "by_agent": dict(Counter(r.get("agent") or "unattributed" for r in model_rows).most_common()),
        "failed_tickers": failed,
        "recent": [f"{r['ts'][11:16]} {r.get('kind')} {r.get('agent') or '-'} {r.get('ticker') or '-'}: {r.get('message', '')[:160]}"
                   for r in model_rows[-5:]],
    }


def describe(summary: dict) -> str:
    """One line for a report: the counts, or that there were none."""
    parts = [f"{n} {kind}" for kind, n in summary.get("by_kind", {}).items()]
    text = ", ".join(parts) if parts else "no model errors"
    if summary.get("failed_tickers"):
        text += f"; {len(summary['failed_tickers'])} ticker run(s) failed"
    return text


def failed_tickers(day: str, since: str, tickers: list[str], config: dict | None = None) -> list[str]:
    """Which of ``tickers`` had a run fail on ``day`` at or after ``since`` (an ISO time), in their order."""
    failed = {r.get("ticker") for r in read_day(day, config) if r.get("kind") == "run" and r.get("ts", "") >= since}
    return [t for t in tickers if t in failed]


# --- the outage alert ---------------------------------------------------------

def _when(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return None


def outage(config: dict, now: datetime) -> dict | None:
    """The outage in progress, or None: no answer from the model for ``model_outage_minutes``.

    An outage is at least two calls that got no answer since the last one that
    did, the first of them that long ago. One slow call is not an outage, and
    neither is a contract failure: the model answered.
    """
    from tradingagents import usage

    days = [(now - timedelta(days=1)).date().isoformat(), now.date().isoformat()]
    answered = [t for day in days for r in usage.read_day(day, config) if (t := _when(r.get("ts", "")))]
    last_ok = max(answered, default=None)
    silent = [(t, r) for day in days for r in read_day(day, config)
              if r.get("kind") in NO_ANSWER and (t := _when(r.get("ts", ""))) and (last_ok is None or t > last_ok)]
    if len(silent) < 2:
        return None
    first = min(t for t, _ in silent)
    minutes = float(config.get("model_outage_minutes") or 15)
    if now - first < timedelta(minutes=minutes):
        return None
    last = max(silent, key=lambda x: x[0])[1]
    return {"since": first.isoformat(timespec="seconds"), "errors": len(silent),
            "last_ok": last_ok.isoformat(timespec="seconds") if last_ok else "",
            "by_kind": dict(Counter(r["kind"] for _, r in silent)), "last": last.get("message", ""),
            "hosts": sorted({r.get("host", "") for _, r in silent})}


def watch(config: dict, now: datetime | None = None, send: Callable | None = None) -> str | None:
    """Email once when the model has gone silent, and once when it answers again; what was sent, if anything.

    The state is a small file beside the log, so a restarted worker neither
    repeats the alert nor forgets to say the model is back.
    """
    if send is None:
        from tradingagents.worker.notify import send_email as send
    now = now or datetime.now().astimezone()
    state_path = errors_dir(config) / "alert.json"
    try:
        alerted = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else None
    except (OSError, ValueError):
        alerted = None
    down = outage(config, now)
    if down and not alerted:
        body = (f"The model has given no answer since {down['since']} ({down['errors']} call(s): "
                + ", ".join(f"{n} {k}" for k, n in down["by_kind"].items()) + ").\n"
                f"Last answered call: {down['last_ok'] or 'none on record'}.\n"
                f"Last error: {down['last']}\nFrom: {', '.join(down['hosts'])}\n\n"
                "Until it answers: ticker runs fail and are queued again, the Watcher leaves news unread and opens "
                "no replacement. The stop and target guards do not use the model and keep working.\n"
                "One more email follows when the model answers again.")
        send(f"Edinburgh: the model is not answering (since {down['since'][11:16]})", body, config)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(down) + "\n", encoding="utf-8")
        logger.error("The model has given no answer since %s (%d calls); alert sent.", down["since"], down["errors"])
        return "down"
    if alerted and not down:
        since = _when(alerted.get("since", ""))
        lasted = f" after {int((now - since).total_seconds() // 60)} minutes" if since else ""
        send("Edinburgh: the model is answering again",
             f"The model answered again{lasted} (silent since {alerted.get('since', '?')}, {alerted.get('errors', '?')} "
             "call(s) lost before the alert). Runs that failed are back in the sweep's queue.", config)
        state_path.unlink(missing_ok=True)
        logger.info("The model is answering again; recovery notice sent.")
        return "up"
    return None


def today_summary(config: dict | None = None, day: str | None = None) -> dict:
    return summarize(read_day(day or date.today().isoformat(), config))
