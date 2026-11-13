"""The feedback store: decisions in, outcomes and scores out, a track record back to the agents."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from sqlalchemy import create_engine, desc, select
from sqlalchemy.engine import Engine

from tradingagents.agents.rating import RATING_REVIEW, run_rating
from tradingagents.feedback import extract, scoring
from tradingagents.feedback.schema import (
    add_missing_columns,
    agent_scores,
    agent_verdicts,
    decisions,
    desk_verdicts,
    metadata,
    outcomes,
)

logger = logging.getLogger(__name__)

AGENT_LABELS = {
    "research_manager": "Research Manager",
    "trader": "Trader",
    "risk_manager": "Risk Manager",
    "portfolio_manager": "Portfolio Manager",
}


@dataclass
class Settled:
    """What a settlement pass did."""

    settled: list[tuple[str, str]] = field(default_factory=list)       # (ticker, date)
    failed: list[tuple[str, str, str]] = field(default_factory=list)   # (ticker, date, reason)


class FeedbackStore:
    """Decisions, outcomes and agent scores behind ``feedback_db_url``.

    With no URL configured the store is disabled: every call is a no-op and the
    track record is empty, so a run without a database still completes.
    """

    def __init__(self, config: dict | None = None, engine: Engine | None = None):
        cfg = config or {}
        self.hold_band_pct = float(cfg.get("feedback_hold_band_pct", 2.0))
        self.context_decisions = int(cfg.get("feedback_context_decisions", 30))
        self.default_stop_pct = float(cfg.get("default_stop_pct") or 2.0)
        self.engine = engine
        url = cfg.get("feedback_db_url")
        if self.engine is None and url:
            if url.startswith("sqlite:///"):
                path = url[len("sqlite:///"):]
                if path and path != ":memory:":
                    Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
                    url = "sqlite:///" + str(Path(path).expanduser())
            self.engine = create_engine(url, future=True)
        if self.engine is not None:
            metadata.create_all(self.engine)
            for column in add_missing_columns(self.engine):
                logger.info("The feedback store gained the column %s", column)

    @property
    def enabled(self) -> bool:
        return self.engine is not None

    # --- record ---------------------------------------------------------------

    def record(self, ticker: str, trade_date: str, final_state: dict, settings: dict | None = None,
               holding_days: int = 5, benchmark: str = "SPY") -> int | None:
        """Record a finished run's calls. A pending row for the same ticker and
        date is replaced; a settled one is kept, as the log keeps its entry."""
        if not self.enabled:
            return None
        settings = settings or {}
        row = {
            "ticker": ticker, "trade_date": str(trade_date),
            "asset_type": final_state.get("asset_type") or "stock",
            "recorded_at": datetime.now(), "status": "pending",
            "rating": run_rating(final_state),
            "position_size_pct": final_state.get("final_position_size_pct"),
            "risk_level": final_state.get("final_risk_level") or None,
            "rm_risk_level": (final_state.get("risk_level") or None) if final_state.get("risk_level") != RATING_REVIEW else None,
            **typed_calls(final_state),
            "holding_days": holding_days, "benchmark": benchmark,
            "llm_provider": settings.get("llm_provider"),
            "quick_think_llm": settings.get("quick_think_llm"),
            "deep_think_llm": settings.get("deep_think_llm"),
            "analysts": ",".join(settings.get("analysts") or []) or None,
            "decision_text": final_state.get("final_trade_decision"),
        }
        with self.engine.begin() as conn:
            existing = conn.execute(select(decisions.c.id, decisions.c.status)
                                    .where(decisions.c.ticker == ticker, decisions.c.trade_date == str(trade_date))).first()
            if existing is not None:
                if existing.status != "pending":
                    return existing.id
                conn.execute(decisions.update().where(decisions.c.id == existing.id).values(**row))
                decision_id = existing.id
            else:
                decision_id = conn.execute(decisions.insert().values(**row)).inserted_primary_key[0]
            conn.execute(agent_verdicts.delete().where(agent_verdicts.c.decision_id == decision_id))
            for agent, v in (final_state.get("verdicts") or {}).items():
                conn.execute(agent_verdicts.insert().values(
                    decision_id=decision_id, agent=agent, ok=bool(v.get("ok")), call=v.get("call"),
                    confidence=v.get("confidence"), thesis=v.get("thesis") or None,
                    evidence=json.dumps(v.get("evidence") or []), model=v.get("model") or None,
                    latency_s=v.get("latency_s"), tokens_in=v.get("tokens_in"), tokens_out=v.get("tokens_out"),
                    attempts=v.get("attempts"), error=v.get("error") or None))
            return decision_id

    def verdicts(self, ticker: str, trade_date: str) -> list[dict]:
        """Every agent's typed answer for one run, as recorded."""
        if not self.enabled:
            return []
        with self.engine.connect() as conn:
            rows = conn.execute(select(agent_verdicts).join(decisions)
                                .where(decisions.c.ticker == ticker, decisions.c.trade_date == str(trade_date))
                                .order_by(agent_verdicts.c.id)).mappings().all()
        return [{**dict(r), "evidence": json.loads(r["evidence"] or "[]")} for r in rows]

    # --- calls made outside a run ----------------------------------------------

    def record_desk_verdict(self, verdict: dict, kind: str, day: str, ticker: str | None = None,
                            action: str | None = None, context: dict | None = None) -> int | None:
        """Record a call the Watcher or a reviewer made: its typed answer and what the desk did with it."""
        if not self.enabled:
            return None
        v = verdict
        with self.engine.begin() as conn:
            return conn.execute(desk_verdicts.insert().values(
                recorded_at=datetime.now(), agent=v.get("agent") or "", kind=kind, ticker=ticker, day=str(day),
                ok=bool(v.get("ok")), call=(v.get("call") or None), confidence=v.get("confidence"),
                thesis=v.get("thesis") or None, evidence=json.dumps(v.get("evidence") or []),
                fields=json.dumps(v.get("fields") or {}, default=str), action=action,
                context=json.dumps(context, default=str) if context else None, model=v.get("model") or None,
                latency_s=v.get("latency_s"), tokens_in=v.get("tokens_in"), tokens_out=v.get("tokens_out"),
                attempts=v.get("attempts"), error=v.get("error") or None)).inserted_primary_key[0]

    def desk_verdicts(self, kind: str | None = None, ticker: str | None = None, since: str | None = None) -> list[dict]:
        """The calls made outside a run, oldest first; by kind, ticker and first day when given."""
        if not self.enabled:
            return []
        query = select(desk_verdicts).order_by(desk_verdicts.c.id)
        if kind:
            query = query.where(desk_verdicts.c.kind == kind)
        if ticker:
            query = query.where(desk_verdicts.c.ticker == ticker)
        if since:
            query = query.where(desk_verdicts.c.day >= since)
        with self.engine.connect() as conn:
            rows = conn.execute(query).mappings().all()
        return [{**dict(r), "evidence": json.loads(r["evidence"] or "[]"), "fields": json.loads(r["fields"] or "{}"),
                 "context": json.loads(r["context"] or "null")} for r in rows]

    # --- settle -----------------------------------------------------------------

    def pending(self) -> list[dict]:
        if not self.enabled:
            return []
        with self.engine.connect() as conn:
            rows = conn.execute(select(decisions).where(decisions.c.status == "pending")
                                .order_by(decisions.c.trade_date)).mappings().all()
        return [dict(r) for r in rows]

    def settle(self, decision: dict, window, reflection: str | None = None) -> None:
        """Write the outcome and the agent scores for ``decision`` over ``window``
        (a ``memory.settlement.Window``)."""
        scored = scoring.score(decision, window.raw_return, float(window.closes.min()),
                               float(window.closes.max()), self.hold_band_pct,
                               closes=window.closes, default_stop_pct=self.default_stop_pct)
        entry = window.entry
        outcome = {
            "decision_id": decision["id"], "settled_at": datetime.now(),
            "entry_date": window.entry_date, "entry_price": entry,
            "exit_date": window.exit_date, "exit_price": window.exit,
            "raw_return": window.raw_return, "benchmark_return": window.benchmark_return, "alpha": window.alpha,
            "max_drawdown": float(window.closes.min() / entry - 1), "max_runup": float(window.closes.max() / entry - 1),
            "direction": scored.direction, "directional_return": scored.directional_return,
            "weighted_return": scored.weighted_return, "stop_hit": scored.stop_hit,
            "r_multiple": scored.r_multiple, "verdict": scored.verdict, "reflection": reflection,
        }
        with self.engine.begin() as conn:
            conn.execute(outcomes.delete().where(outcomes.c.decision_id == decision["id"]))
            conn.execute(agent_scores.delete().where(agent_scores.c.decision_id == decision["id"]))
            conn.execute(outcomes.insert().values(**outcome))
            for row in scored.agent_scores:
                conn.execute(agent_scores.insert().values(decision_id=decision["id"], **row))
            conn.execute(decisions.update().where(decisions.c.id == decision["id"]).values(status="settled"))

    def settle_due(self, fetch_window: Callable, reflections: Callable[[str, str], str | None] | None = None) -> Settled:
        """Settle every pending decision whose window has traded.

        ``fetch_window(ticker, trade_date, holding_days, benchmark)`` returns the
        window or None while it has not traded; ``reflections(ticker, date)``
        supplies the memory log's lesson for the row, when it has one.
        """
        result = Settled()
        for decision in self.pending():
            ticker, date = decision["ticker"], decision["trade_date"]
            try:
                window = fetch_window(ticker, date, decision["holding_days"], decision["benchmark"])
            except Exception as exc:
                logger.warning("Prices for scoring %s on %s were unavailable: %s", ticker, date, exc)
                result.failed.append((ticker, date, f"prices unavailable: {exc}"))
                continue
            if window is None:
                continue
            self.settle(decision, window, reflections(ticker, date) if reflections else None)
            result.settled.append((ticker, date))
        return result

    def score_r(self, fetch_window: Callable) -> tuple[int, list[tuple[str, str, str]]]:
        """Fill in R for the decisions settled before R was kept: how many were scored, and those that could not be.

        Each directional outcome without an R has its window fetched again
        (``fetch_window`` as in ``settle_due``) and its R written; nothing else
        about the outcome changes.
        """
        if not self.enabled:
            return 0, []
        with self.engine.connect() as conn:
            rows = [dict(r) for r in self._settled(conn, None) if r["r_multiple"] is None and r["direction"]]
        done, failed = 0, []
        for row in rows:
            ticker, date = row["ticker"], row["trade_date"]
            try:
                window = fetch_window(ticker, date, row["holding_days"], row["benchmark"])
            except Exception as exc:
                failed.append((ticker, date, f"prices unavailable: {exc}"))
                continue
            stop = row["rm_stop_loss"] if row["rm_stop_loss"] is not None else row["trader_stop_loss"]
            r = scoring.window_r(row["direction"], window.closes, stop, row["price_target"],
                                 self.default_stop_pct) if window is not None else None
            if r is None:
                failed.append((ticker, date, "no window or no risk to measure against"))
                continue
            with self.engine.begin() as conn:
                conn.execute(outcomes.update().where(outcomes.c.decision_id == row["id"]).values(r_multiple=r))
            done += 1
        return done, failed

    def r_record(self, as_of: str | None = None, exited_on: str | None = None) -> dict:
        """R over the settled directional decisions: all of them, and by side.

        ``exited_on`` keeps only the outcomes whose window closed that day.
        Each entry is ``{"n", "total", "mean"}``; a side with nothing scored is left out.
        """
        if not self.enabled:
            return {}
        with self.engine.connect() as conn:
            rows = [r for r in self._settled(conn, as_of) if exited_on is None or r["exit_date"] == exited_on]
        out = {}
        for name, keep in (("all", lambda r: True), ("long", lambda r: r["direction"] > 0), ("short", lambda r: r["direction"] < 0)):
            summary = scoring.r_summary(r["r_multiple"] for r in rows if keep(r))
            if summary:
                out[name] = summary
        return out

    def attach_reflection(self, ticker: str, trade_date: str, reflection: str) -> None:
        """Keep the memory log's lesson with the outcome once it exists."""
        if not self.enabled:
            return
        with self.engine.begin() as conn:
            decision_id = conn.execute(select(decisions.c.id).where(
                decisions.c.ticker == ticker, decisions.c.trade_date == trade_date)).scalar()
            if decision_id is not None:
                conn.execute(outcomes.update().where(outcomes.c.decision_id == decision_id).values(reflection=reflection))

    # --- read back ----------------------------------------------------------------

    def _settled(self, conn, as_of: str | None):
        """Settled decisions joined to their outcomes, newest first, known by ``as_of``."""
        query = (select(decisions, outcomes).join(outcomes, outcomes.c.decision_id == decisions.c.id)
                 .where(decisions.c.status == "settled").order_by(desc(decisions.c.trade_date)))
        if as_of:
            query = query.where(outcomes.c.exit_date <= as_of)
        return conn.execute(query).mappings().all()

    def scorecard(self, as_of: str | None = None, limit: int | None = None) -> dict:
        """Per agent: how many calls were judged, the hit rate, the mean score, and the same by call."""
        if not self.enabled:
            return {}
        limit = limit or self.context_decisions
        with self.engine.connect() as conn:
            recent = [r["id"] for r in self._settled(conn, as_of)][:limit]
            if not recent:
                return {}
            rows = conn.execute(select(agent_scores).where(agent_scores.c.decision_id.in_(recent))).mappings().all()
        card: dict = {}
        for r in rows:
            agent = card.setdefault(r["agent"], {"n": 0, "hits": 0, "scores": [], "by_call": {}})
            by_call = agent["by_call"].setdefault(r["call"], {"n": 0, "hits": 0, "scores": []})
            for bucket in (agent, by_call):
                if r["correct"] is not None:
                    bucket["n"] += 1
                    bucket["hits"] += int(r["correct"])
                if r["score"] is not None:
                    bucket["scores"].append(r["score"])
        for agent in card.values():
            _finish(agent)
            for by_call in agent["by_call"].values():
                _finish(by_call)
        return card

    def ticker_history(self, ticker: str, as_of: str | None = None, limit: int = 5) -> list[dict]:
        if not self.enabled:
            return []
        with self.engine.connect() as conn:
            rows = [dict(r) for r in self._settled(conn, as_of) if r["ticker"] == ticker][:limit]
        return rows

    def settled_count(self, as_of: str | None = None) -> int:
        if not self.enabled:
            return 0
        with self.engine.connect() as conn:
            return len(self._settled(conn, as_of))

    def render_context(self, ticker: str, as_of: str | None = None) -> str:
        """The track record as the deciding agents read it; empty when nothing is settled."""
        card = self.scorecard(as_of)
        if not card:
            return ""
        counted = min(self.context_decisions, self.settled_count(as_of))
        lines = [f"Track record over the last {counted} settled decision{'s' if counted != 1 else ''} "
                 f"(a call is right when the price moved its way over the holding window; "
                 f"a Hold is right within ±{self.hold_band_pct:g}%):"]
        for agent in scoring.AGENTS:
            stats = card.get(agent)
            if not stats or not stats["n"]:
                continue
            calls = ", ".join(f"{call} {s['hits']}/{s['n']}" + (f" ({s['mean']:+.1%})" if s["mean"] is not None else "")
                              for call, s in sorted(stats["by_call"].items()))
            mean = f", mean return on its calls {stats['mean']:+.1%}" if stats["mean"] is not None else ""
            lines.append(f"- {AGENT_LABELS[agent]}: {stats['hits']}/{stats['n']} right{mean}. By call: {calls}")
        history = self.ticker_history(ticker, as_of)
        if history:
            lines.append(f"Past settled decisions on {ticker}:")
            for h in history:
                size = f", size {h['position_size_pct']:g}%" if h["position_size_pct"] is not None else ""
                stop = ", stop hit" if h["stop_hit"] else ""
                lesson = f" Lesson: {h['reflection']}" if h.get("reflection") else ""
                lines.append(f"- {h['trade_date']}: {h['rating']}{size}, risk {h['risk_level'] or h['rm_risk_level'] or '?'} "
                             f"→ {h['verdict']}, return {h['raw_return']:+.1%}, alpha {h['alpha']:+.1%}{stop}.{lesson}")
        lines.append("Favour the calls and stances that have been right, and be slower to repeat those that have not; "
                     "a short record proves little, so weigh it with the evidence of this run.")
        return "\n".join(lines)


def typed_calls(final_state: dict) -> dict:
    """The deciding agents' calls for the decisions row, from their typed verdicts.

    An agent with no verdict in the state (a run from before verdicts were
    kept, or prose under the free-text opt-out) has its call read from its
    rendered text instead, as before.
    """
    verdicts = final_state.get("verdicts") or {}

    def typed(agent: str) -> dict | None:
        v = verdicts.get(agent)
        return v if v and v.get("ok") else None

    out: dict = {}
    pm, risk, trader, research = (typed(a) for a in ("portfolio_manager", "risk_manager", "trader", "research_manager"))
    out["price_target"] = (pm["fields"].get("price_target") if pm
                           else extract.labelled_number(final_state.get("final_trade_decision") or "", "Price Target"))
    if risk:
        out.update(rm_max_position_pct=risk["fields"].get("max_position_pct"), rm_stop_loss=risk["fields"].get("stop_loss"))
    else:
        out.update(extract.risk_fields(final_state.get("risk_assessment") or ""))
    if trader:
        out.update(trader_action=trader["call"], trader_entry_price=trader["fields"].get("entry_price"),
                   trader_stop_loss=trader["fields"].get("stop_loss"))
    else:
        out.update(extract.trader_fields(final_state.get("trader_investment_plan") or ""))
    out["research_recommendation"] = (research["call"] if research
                                      else extract.research_recommendation(final_state.get("investment_plan") or ""))
    return out


def _finish(bucket: dict) -> None:
    scores = bucket.pop("scores")
    bucket["mean"] = sum(scores) / len(scores) if scores else None
    bucket["hit_rate"] = bucket["hits"] / bucket["n"] if bucket["n"] else None


def keep_desk_verdict(config: dict, verdict, kind: str, day: str, ticker: str | None = None,
                      action: str | None = None, context: dict | None = None) -> None:
    """Record a Watcher's or a reviewer's verdict in the store. Never raises: the record must not stop the desk."""
    try:
        data = verdict.model_dump(mode="json") if hasattr(verdict, "model_dump") else dict(verdict)
        FeedbackStore(config).record_desk_verdict(data, kind, day, ticker, action, context)
    except Exception as exc:
        logging.getLogger(__name__).warning("The %s verdict was not recorded: %s", kind, exc)
