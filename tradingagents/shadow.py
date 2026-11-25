"""The Shadow Portfolio Manager: the same decision asked of a second model, kept beside the first.

The Portfolio Manager can run on a model of its own (the large one on the PGX,
``agent_calls``). Whether that model's decisions are worth what they cost is a
question only a comparison answers, so with ``shadow_portfolio_manager`` on,
every symbol's Portfolio Manager prompt is also put, word for word and at the
same moment, to the deep tier's own model. The desk acts on the Portfolio
Manager's answer alone: the shadow's is never traded, never shown to another
agent and never enters the run's state. Both are written as one line in
``<results_dir>/shadow/<date>.jsonl``.

A shadow call that fails costs the run nothing: the line records the failure
and the run goes on. A difference between the two is not proof that either is
right; the outcomes settle that, which is what the record is for.
"""

from __future__ import annotations

import contextvars
import json
import logging
import threading
from collections import Counter
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

AGENT = "Shadow Portfolio Manager"
FIELDS = ("rating", "position_size_pct", "risk_level", "price_target", "time_horizon",
          "executive_summary", "investment_thesis")
_SIDE = {"buy": "long", "overweight": "long", "sell": "short", "underweight": "short", "hold": "flat"}


def side(rating: Any) -> str:
    """Long, short or flat for a rating; empty when there is none."""
    return _SIDE.get(str(rating or "").strip().lower(), "")


def shadow_dir(config: dict | None = None) -> Path:
    if config is None:
        from tradingagents.dataflows.config import get_config

        config = get_config()
    return Path(config.get("shadow_dir") or Path(config["results_dir"]) / "shadow")


class Shadow:
    """One shadow call, made beside the Portfolio Manager's own: started before it, read after it."""

    def __init__(self, ask: Callable[[], tuple]):
        self._answer: tuple | None = None
        self._error = ""
        context = contextvars.copy_context()          # the run's config, ticker and run id come along
        self._thread = threading.Thread(target=context.run, args=(self._run, ask), name="shadow-pm", daemon=True)
        self._thread.start()

    def _run(self, ask: Callable[[], tuple]) -> None:
        try:
            # The graph's own run config would name these calls the Portfolio Manager's: they are the shadow's.
            from langchain_core.runnables.config import var_child_runnable_config

            var_child_runnable_config.set(None)
            self._answer = ask()
        except BaseException as exc:        # the shadow never costs the run anything
            self._error = f"{type(exc).__name__}: {' '.join(str(exc).split())[:300]}"
            logger.warning("%s: no answer (%s)", AGENT, self._error)

    def verdict(self) -> dict:
        """The shadow's answer as a record, waiting for it; why there is none, when there is none."""
        self._thread.join()
        if self._answer is None:
            return {"ok": False, "error": self._error or "no answer"}
        return entry(self._answer[1])


def entry(verdict: Any) -> dict:
    """What the record keeps of one model's answer: whether it answered, with what, and what it took."""
    fields = getattr(verdict, "fields", None) or {}
    out = {"ok": bool(getattr(verdict, "ok", False)), "model": getattr(verdict, "model", "") or "",
           "seconds": getattr(verdict, "latency_s", None), "tokens_out": getattr(verdict, "tokens_out", None),
           "attempts": getattr(verdict, "attempts", None)}
    if not out["ok"]:
        out["error"] = getattr(verdict, "error", "") or ""
    out.update({k: fields.get(k) for k in FIELDS if k in fields})
    return out


def record(ticker: str, trade_date: str, primary: dict, shadow: dict, config: dict | None = None) -> dict | None:
    """Add one comparison to the day's file. Never raises."""
    try:
        from tradingagents import usage

        both = primary.get("ok") and shadow.get("ok")
        rec = {"ts": datetime.now().astimezone().isoformat(timespec="seconds"), "ticker": ticker, "trade_date": trade_date,
               "run_id": usage.context()["run_id"], "portfolio_manager": primary, "shadow": shadow,
               "same_rating": bool(both and primary.get("rating") == shadow.get("rating")),
               "same_side": bool(both and side(primary.get("rating")) == side(shadow.get("rating")))}
        folder = shadow_dir(config)
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / f"{rec['ts'][:10]}.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
        return rec
    except Exception as exc:
        logger.debug("Shadow record not written: %s", exc)
        return None


def read_day(day: str, config: dict | None = None) -> list[dict]:
    path = shadow_dir(config) / f"{day}.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def summarize(rows: list[dict]) -> dict:
    """How far the two agree: over the symbols both answered, by rating, by side and by size."""
    both = [r for r in rows if r["portfolio_manager"].get("ok") and r["shadow"].get("ok")]
    sizes = [(r["portfolio_manager"].get("position_size_pct"), r["shadow"].get("position_size_pct")) for r in both]
    sizes = [(a, b) for a, b in sizes if isinstance(a, (int, float)) and isinstance(b, (int, float))]
    pairs = Counter((r["portfolio_manager"].get("rating"), r["shadow"].get("rating")) for r in both)
    return {
        "symbols": len(rows), "both_answered": len(both),
        "shadow_failed": sum(1 for r in rows if not r["shadow"].get("ok")),
        "same_rating": sum(1 for r in both if r["same_rating"]),
        "same_side": sum(1 for r in both if r["same_side"]),
        "opposite_sides": sum(1 for r in both if {side(r["portfolio_manager"].get("rating")),
                                                  side(r["shadow"].get("rating"))} == {"long", "short"}),
        "mean_size_difference_pct": round(sum(abs(a - b) for a, b in sizes) / len(sizes), 2) if sizes else None,
        "ratings": [{"portfolio_manager": a, "shadow": b, "symbols": n} for (a, b), n in pairs.most_common()],
    }
