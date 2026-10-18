"""Tables of the feedback loop, written for Postgres and served by SQLite until then.

decisions      one row per run: the typed calls every stage made
outcomes       one row per settled decision: what the market did over the window
agent_scores   one row per agent per settled decision: its call and whether it was right
"""

from __future__ import annotations

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
)

metadata = MetaData()

decisions = Table(
    "decisions", metadata,
    Column("id", Integer, primary_key=True),
    Column("ticker", String(32), nullable=False),
    Column("trade_date", String(10), nullable=False),          # YYYY-MM-DD, the analysis date
    Column("asset_type", String(16), nullable=False, default="stock"),
    Column("recorded_at", DateTime, nullable=False),
    Column("status", String(16), nullable=False, default="pending"),   # pending | settled | unscorable
    # The Portfolio Manager's call
    Column("rating", String(16), nullable=False),                # Buy/Overweight/Hold/Underweight/Sell/REVIEW
    Column("position_size_pct", Float),                          # target % of portfolio equity
    Column("risk_level", String(16)),                            # the stance the PM adopted
    Column("price_target", Float),
    # Risk Management's verdict
    Column("rm_risk_level", String(16)),
    Column("rm_max_position_pct", Float),
    Column("rm_stop_loss", Float),
    # The Trader's proposal
    Column("trader_action", String(16)),
    Column("trader_entry_price", Float),
    Column("trader_stop_loss", Float),
    # The Research Manager's recommendation
    Column("research_recommendation", String(16)),
    # How the outcome is measured
    Column("holding_days", Integer, nullable=False),
    Column("benchmark", String(32), nullable=False),
    # What produced it
    Column("llm_provider", String(32)),
    Column("quick_think_llm", String(128)),
    Column("deep_think_llm", String(128)),
    Column("analysts", String(128)),
    Column("decision_text", Text),
    UniqueConstraint("ticker", "trade_date", name="uq_decisions_ticker_date"),
)

outcomes = Table(
    "outcomes", metadata,
    Column("decision_id", Integer, ForeignKey("decisions.id", ondelete="CASCADE"), primary_key=True),
    Column("settled_at", DateTime, nullable=False),
    Column("entry_date", String(10), nullable=False),            # first close after the analysis date
    Column("entry_price", Float, nullable=False),
    Column("exit_date", String(10), nullable=False),             # holding_days sessions later
    Column("exit_price", Float, nullable=False),
    Column("raw_return", Float, nullable=False),
    Column("benchmark_return", Float, nullable=False),
    Column("alpha", Float, nullable=False),
    Column("max_drawdown", Float, nullable=False),               # worst close vs entry over the window, <= 0
    Column("max_runup", Float, nullable=False),                  # best close vs entry over the window, >= 0
    Column("direction", Integer, nullable=False),                # +1 long, -1 short, 0 hold
    Column("directional_return", Float, nullable=False),         # raw_return * direction
    Column("weighted_return", Float),                            # directional_return * position_size_pct / 100
    Column("stop_hit", Boolean),
    Column("verdict", String(16), nullable=False),               # win | loss | flat | missed | avoided
    Column("reflection", Text),
)

agent_scores = Table(
    "agent_scores", metadata,
    Column("id", Integer, primary_key=True),
    Column("decision_id", Integer, ForeignKey("decisions.id", ondelete="CASCADE"), nullable=False),
    Column("agent", String(32), nullable=False),                 # research_manager | trader | risk_manager | portfolio_manager
    Column("call", String(16), nullable=False),                  # what it said
    Column("correct", Boolean),                                  # NULL when the call cannot be judged
    Column("score", Float),                                      # the return its call earned, signed
    UniqueConstraint("decision_id", "agent", name="uq_agent_scores_decision_agent"),
)
