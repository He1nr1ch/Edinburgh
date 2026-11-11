"""What the models cost the desk: every call's tokens and seconds, by agent, ticker and day.

The Ollama client records each call it makes (see
:meth:`NormalizedChatOllama.invoke`): the model, the agent that asked (the
LangGraph node, or the name a structured caller such as the Watcher gives),
the ticker and run under analysis, the tokens in and out and the seconds the
server spent. Each record is appended to ``<results_dir>/usage/<date>.jsonl``,
one file per day, so a day's spend can be totalled later by anything that
reads the folder: the daily report, the positions page, ``tradingagents
usage``.

The same records feed an in-process :class:`Tally`, which the single-ticker
run writes into its ``timing.json`` by agent, so one ticker's cost can be
compared with another's.

Which PAIR node served a call is not in the response; PAIR keeps that record
itself (see :mod:`tradingagents.pair`).
"""

from __future__ import annotations

import contextvars
import json
import logging
import socket
import threading
from collections import defaultdict
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_agent: contextvars.ContextVar[str] = contextvars.ContextVar("usage_agent", default="")
_ticker: contextvars.ContextVar[str] = contextvars.ContextVar("usage_ticker", default="")
_run_id: contextvars.ContextVar[str] = contextvars.ContextVar("usage_run_id", default="")
_capture: contextvars.ContextVar[list | None] = contextvars.ContextVar("usage_capture", default=None)
_HOST = socket.gethostname()

# LangGraph's inner nodes, whose outer node names the agent.
_INNER_NODES = {"agent", "tools", "wrap_up"}


@contextmanager
def as_agent(name: str):
    """Attribute the model calls made inside the block to ``name`` (the Watcher, a reviewer)."""
    token = _agent.set(name or "")
    try:
        yield
    finally:
        _agent.reset(token)


@contextmanager
def capture():
    """Collect the records of the model calls made inside the block, for the caller to stamp its answer with."""
    records: list[dict] = []
    token = _capture.set(records)
    try:
        yield records
    finally:
        _capture.reset(token)


def set_run(ticker: str | None = None, run_id: str | None = None) -> None:
    """Name the ticker and run the calls that follow belong to (the graph sets the ticker per run)."""
    if ticker is not None:
        _ticker.set(ticker)
    if run_id is not None:
        _run_id.set(run_id)


def agent_from(config: dict | None) -> str:
    """The agent a LangChain run config names: the LangGraph node, outermost when nested."""
    meta = (config or {}).get("metadata") or {}
    node = meta.get("langgraph_node") or ""
    ns = meta.get("langgraph_checkpoint_ns") or ""
    if node in _INNER_NODES and ns:
        node = ns.split("|")[0].split(":")[0] or node
    return node or _agent.get()


class Tally:
    """Totals by agent for the process (reset at the start of a run)."""

    def __init__(self):
        self._lock = threading.Lock()
        self.by_agent: dict[str, dict] = defaultdict(lambda: {"calls": 0, "tokens_in": 0, "tokens_out": 0, "seconds": 0.0})

    def add(self, rec: dict) -> None:
        with self._lock:
            row = self.by_agent[rec.get("agent") or "unattributed"]
            row["calls"] += 1
            row["tokens_in"] += rec.get("tokens_in") or 0
            row["tokens_out"] += rec.get("tokens_out") or 0
            row["seconds"] = round(row["seconds"] + (rec.get("seconds") or 0.0), 2)

    def reset(self) -> None:
        with self._lock:
            self.by_agent.clear()

    def snapshot(self) -> dict:
        with self._lock:
            rows = {k: dict(v) for k, v in self.by_agent.items()}
        total = {"calls": sum(r["calls"] for r in rows.values()), "tokens_in": sum(r["tokens_in"] for r in rows.values()),
                 "tokens_out": sum(r["tokens_out"] for r in rows.values()),
                 "seconds": round(sum(r["seconds"] for r in rows.values()), 2)}
        return {"total": total, "by_agent": rows}


tally = Tally()


def usage_dir(config: dict | None = None) -> Path:
    if config is None:
        from tradingagents.dataflows.config import get_config

        config = get_config()
    return Path(config.get("usage_dir") or Path(config["results_dir"]) / "usage")


def record_response(response, model: str, config: dict | None = None) -> dict | None:
    """Record one chat response (its ``usage_metadata`` and Ollama's durations); None when it carries none."""
    usage = getattr(response, "usage_metadata", None) or {}
    meta = getattr(response, "response_metadata", None) or {}
    if not usage and not meta.get("total_duration"):
        return None
    rec = {
        "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
        "host": _HOST, "model": meta.get("model") or model,
        "agent": agent_from(config), "ticker": _ticker.get(), "run_id": _run_id.get(),
        "tokens_in": int(usage.get("input_tokens") or 0), "tokens_out": int(usage.get("output_tokens") or 0),
        "seconds": round((meta.get("total_duration") or 0) / 1e9, 2),
        "gen_seconds": round((meta.get("eval_duration") or 0) / 1e9, 2),
        "prompt_seconds": round((meta.get("prompt_eval_duration") or 0) / 1e9, 2),
    }
    if rec["gen_seconds"] and rec["tokens_out"]:
        rec["tokens_per_s"] = round(rec["tokens_out"] / rec["gen_seconds"], 1)
    record(rec)
    return rec


def record(rec: dict) -> None:
    tally.add(rec)
    collected = _capture.get()
    if collected is not None:
        collected.append(rec)
    try:
        folder = usage_dir()
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / f"{rec['ts'][:10]}.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
    except Exception as exc:     # a full disk must not fail the call that was answered
        logger.debug("Usage record not written: %s", exc)


def read_day(day: str, config: dict | None = None) -> list[dict]:
    path = usage_dir(config) / f"{day}.jsonl"
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
    """Totals, and by agent, model, host and ticker, from a day's records."""
    def bucket() -> dict:
        return {"calls": 0, "tokens_in": 0, "tokens_out": 0, "seconds": 0.0}

    groups = {k: defaultdict(bucket) for k in ("agent", "model", "host", "ticker")}
    total = bucket()
    for r in rows:
        for key, g in groups.items():
            row = g[r.get(key) or ("unattributed" if key == "agent" else "-")]
            for k in total:
                row[k] += r.get(k) or (1 if k == "calls" else 0)
        for k in total:
            total[k] += r.get(k) or (1 if k == "calls" else 0)
    for g in groups.values():
        for row in g.values():
            row["seconds"] = round(row["seconds"], 1)
    total["seconds"] = round(total["seconds"], 1)
    total["hours"] = round(total["seconds"] / 3600, 2)
    total["tickers"] = len(groups["ticker"]) - (1 if "-" in groups["ticker"] else 0)
    return {"total": total, **{f"by_{k}": dict(sorted(g.items(), key=lambda kv: -kv[1]["seconds"])) for k, g in groups.items()}}


def today_summary(config: dict | None = None, day: str | None = None) -> dict:
    return summarize(read_day(day or date.today().isoformat(), config))


def describe(summary: dict) -> str:
    t = summary["total"]
    if not t["calls"]:
        return "no model calls recorded"
    text = (f"{t['calls']:,} calls, {t['tokens_in']:,} tokens in, {t['tokens_out']:,} out, "
            f"{t['hours']:.1f} h of model time")
    if t.get("tickers"):
        text += f" over {t['tickers']} ticker(s)"
    return text
