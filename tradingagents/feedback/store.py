"""The feedback store: decisions in, outcomes and scores out, a track record back to the agents."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from sqlalchemy import create_engine, desc, select
from sqlalchemy.engine import Engine

from tradingagents.agents.rating import RATING_REVIEW, run_rating
from tradingagents.feedback import extract, scoring
from tradingagents.feedback.schema import agent_scores, decisions, metadata, outcomes

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
        risk_text = final_state.get("risk_assessment") or ""
        row = {
            "ticker": ticker, "trade_date": str(trade_date),
            "asset_type": final_state.get("asset_type") or "stock",
            "recorded_at": datetime.now(), "status": "pending",
            "rating": run_rating(final_state),
            "position_size_pct": final_state.get("final_position_size_pct"),
            "risk_level": final_state.get("final_risk_level") or None,
            "price_target": extract.labelled_number(final_state.get("final_trade_decision") or "", "Price Target"),
            "rm_risk_level": (final_state.get("risk_level") or None) if final_state.get("risk_level") != RATING_REVIEW else None,
            **extract.risk_fields(risk_text),
            **extract.trader_fields(final_state.get("trader_investment_plan") or ""),
            "research_recommendation": extract.research_recommendation(final_state.get("investment_plan") or ""),
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
                return existing.id
            return conn.execute(decisions.insert().values(**row)).inserted_primary_key[0]

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
                               float(window.closes.max()), self.hold_band_pct)
        entry = window.entry
        outcome = {
            "decision_id": decision["id"], "settled_at": datetime.now(),
            "entry_date": window.entry_date, "entry_price": entry,
            "exit_date": window.exit_date, "exit_price": window.exit,
            "raw_return": window.raw_return, "benchmark_return": window.benchmark_return, "alpha": window.alpha,
            "max_drawdown": float(window.closes.min() / entry - 1), "max_runup": float(window.closes.max() / entry - 1),
            "direction": scored.direction, "directional_return": scored.directional_return,
            "weighted_return": scored.weighted_return, "stop_hit": scored.stop_hit,
            "verdict": scored.verdict, "reflection": reflection,
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


def _finish(bucket: dict) -> None:
    scores = bucket.pop("scores")
    bucket["mean"] = sum(scores) / len(scores) if scores else None
    bucket["hit_rate"] = bucket["hits"] / bucket["n"] if bucket["n"] else None

