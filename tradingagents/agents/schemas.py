"""Pydantic schemas used by agents that produce structured output.

The framework's primary artifact is still prose: each agent's natural-language
reasoning is what users read in the saved markdown reports and what the
downstream agents read as context.  Structured output is layered onto the
three decision-making agents (Research Manager, Trader, Portfolio Manager)
so that:

- Their outputs follow consistent section headers across runs and providers
- Each provider's native structured-output mode is used (json_schema for
  OpenAI/xAI, response_schema for Gemini, tool-use for Anthropic)
- Schema field descriptions become the model's output instructions, freeing
  the prompt body to focus on context and the rating-scale guidance
- A render helper turns the parsed Pydantic instance back into the same
  markdown shape the rest of the system already consumes, so display,
  memory log, and saved reports keep working unchanged
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

# LLMs sometimes write a placeholder string ("None", "N/A", ...) into an optional
# numeric field instead of omitting it. Coerce those to None so the structured
# call validates instead of erroring (#1058). Pydantic still parses real numeric
# strings ("189.5") to float.
_NULLISH_FLOAT = {"", "none", "n/a", "na", "null", "nil", "-", "tbd", "unknown"}


def _coerce_optional_float(value):
    """Normalise an LLM-written optional numeric field before validation.

    Three shapes show up in practice: a placeholder string ("None", "N/A") in
    place of an omitted value (#1058); a percentage where a price was asked for
    ("15%", #1288); and a human-formatted price ("$1,234.50"). A percentage
    cannot be salvaged into an absolute level -- reading "15%" as 15 would put a
    stop at $15 on a $600 stock -- so it is dropped like a placeholder, leaving
    one bad field to null out instead of failing the whole proposal. A formatted
    price is reduced to its number.

    Anything that is not a single number is dropped the same way. A range
    ("150-160") or a hedge ("around 150") would otherwise reach pydantic, fail
    validation, and discard the whole decision, losing every field the model got
    right along with the price.
    """
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text.lower() in _NULLISH_FLOAT or text.endswith("%"):
        return None
    cleaned = text.replace(",", "").lstrip("$€£¥").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Shared rating types
# ---------------------------------------------------------------------------


class PortfolioRating(StrEnum):
    """5-tier rating used by the Research Manager and Portfolio Manager."""

    BUY = "Buy"
    OVERWEIGHT = "Overweight"
    HOLD = "Hold"
    UNDERWEIGHT = "Underweight"
    SELL = "Sell"


def _always_asked(*names):
    """Mark nullable fields as required in the JSON schema the model is bound to.

    A field with a default is optional in the schema, and a model served a
    schema-constrained grammar (Ollama's structured output) then leaves the key
    out and writes the value into the prose instead. Requiring the key makes the
    model answer it, as a number or an explicit null, while the Python side keeps
    treating an unparseable value as None rather than failing the decision.
    """
    def extend(schema: dict) -> None:
        required = schema.setdefault("required", [])
        for name in names:
            if name not in required:
                required.append(name)
    return extend


class RiskLevel(StrEnum):
    """The stance Risk Management settles on for a trade."""

    AGGRESSIVE = "Aggressive"
    NEUTRAL = "Neutral"
    CONSERVATIVE = "Conservative"


def _coerce_optional_percent(value):
    """A percentage of portfolio equity as a number in [0, 100], or None.

    Models write "4.5%", "4.5", "4–5%" or a placeholder. A range or anything
    else that is not one number is dropped, like a price, rather than failing
    the whole decision; a number outside [0, 100] is not a weight and is dropped too.
    """
    if isinstance(value, str):
        value = _coerce_optional_float(value.rstrip("%").strip() if value.strip().endswith("%") else value)
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if 0.0 <= number <= 100.0 else None


class TraderAction(StrEnum):
    """3-tier transaction direction used by the Trader.

    The Trader's job is to translate the Research Manager's investment plan
    into a concrete transaction proposal: should the desk execute a Buy, a
    Sell, or sit on Hold this round.  Position sizing and the nuanced
    Overweight / Underweight calls happen later at the Portfolio Manager.
    """

    BUY = "Buy"
    HOLD = "Hold"
    SELL = "Sell"


# ---------------------------------------------------------------------------
# Research Manager
# ---------------------------------------------------------------------------


class ResearchPlan(BaseModel):
    """Structured investment plan produced by the Research Manager.

    Hand-off to the Trader: the recommendation pins the directional view,
    the rationale captures which side of the bull/bear debate carried the
    argument, and the strategic actions translate that into concrete
    instructions the trader can execute against.
    """

    recommendation: PortfolioRating = Field(
        description=(
            "The investment recommendation. Exactly one of Buy / Overweight / "
            "Hold / Underweight / Sell. Conflicting arguments alone are not a "
            "reason to Hold: commit to the stronger side, sized by how "
            "decisively it wins. Choose Hold only when the evidence is still "
            "balanced after weighing, or too thin to support a call."
        ),
    )
    rationale: str = Field(
        description=(
            "Conversational summary of the key points from both sides of the "
            "debate, ending with which arguments led to the recommendation. "
            "Speak naturally, as if to a teammate."
        ),
    )
    strategic_actions: str = Field(
        description=(
            "Concrete steps for the trader to implement the recommendation, "
            "including sizing guidance relative to a standard allocation. The "
            "research team does not see the caller's holdings; the trader and "
            "portfolio manager apply the actual position."
        ),
    )


def render_research_plan(plan: ResearchPlan) -> str:
    """Render a ResearchPlan to markdown for storage and the trader's prompt context."""
    return "\n".join([
        f"**Recommendation**: {plan.recommendation.value}",
        "",
        f"**Rationale**: {plan.rationale}",
        "",
        f"**Strategic Actions**: {plan.strategic_actions}",
    ])


# ---------------------------------------------------------------------------
# Trader
# ---------------------------------------------------------------------------


class TraderProposal(BaseModel):
    """Structured transaction proposal produced by the Trader.

    The trader reads the Research Manager's investment plan and the analyst
    reports, then turns them into a concrete transaction: what action to
    take, the reasoning that justifies it, and the practical levels for
    entry, stop-loss, and sizing.
    """

    action: TraderAction = Field(
        description="The transaction direction. Exactly one of Buy / Hold / Sell.",
    )
    reasoning: str = Field(
        description=(
            "The case for this action, anchored in the analysts' reports and "
            "the research plan. Two to four sentences."
        ),
    )
    entry_price: float | None = Field(
        default=None,
        description=(
            "Optional entry price target as an absolute number in the instrument's "
            "quote currency (e.g. 189.5), never a percentage or a range. Omit it "
            "if you cannot state a specific level."
        ),
    )
    stop_loss: float | None = Field(
        default=None,
        description=(
            "Optional stop-loss as an absolute price in the instrument's quote "
            "currency (e.g. 172.0), never a percentage. Convert a percentage "
            "distance to the price level it implies, or omit it."
        ),
    )
    position_sizing: str | None = Field(
        default=None,
        description="Optional sizing guidance, e.g. '5% of portfolio'.",
    )

    @field_validator("entry_price", "stop_loss", mode="before")
    @classmethod
    def _nullish_float_to_none(cls, v):
        return _coerce_optional_float(v)


def render_trader_proposal(proposal: TraderProposal) -> str:
    """Render a TraderProposal to markdown.

    The trailing ``FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**`` line is
    preserved for backward compatibility with the analyst stop-signal text
    and any external code that greps for it.
    """
    parts = [
        f"**Action**: {proposal.action.value}",
        "",
        f"**Reasoning**: {proposal.reasoning}",
    ]
    # Named even when absent, so a reader can tell a level the trader chose not
    # to give from one the schema never asked for.
    for label, value in (("Entry Price", proposal.entry_price),
                         ("Stop Loss", proposal.stop_loss),
                         ("Position Sizing", proposal.position_sizing)):
        parts.extend(["", f"**{label}**: {value if value is not None and value != '' else 'not provided'}"])
    parts.extend([
        "",
        f"FINAL TRANSACTION PROPOSAL: **{proposal.action.value.upper()}**",
    ])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Risk Manager
# ---------------------------------------------------------------------------


class RiskAssessment(BaseModel):
    """Structured verdict of Risk Management on the trader's proposal.

    The three risk analysts argue; this is where the desk settles on a stance
    and a ceiling. The Portfolio Manager sizes within the ceiling, and the
    feedback loop scores the stance against the outcome.
    """

    model_config = ConfigDict(json_schema_extra=_always_asked("stop_loss"))

    risk_level: RiskLevel = Field(
        description=(
            "The risk stance the desk adopts for this trade. Exactly one of "
            "Aggressive / Neutral / Conservative. Aggressive: the upside case is "
            "strong enough to carry a full-size position; Conservative: the "
            "downside case warrants a small position or none; Neutral: a standard "
            "position with ordinary stops."
        ),
    )
    max_position_pct: float = Field(
        ge=0.0, le=100.0,
        description=(
            "The ceiling on this position as a percentage of total portfolio "
            "equity, a number from 0 to 100 (e.g. 5 for five percent). 0 means "
            "no position should be held."
        ),
    )
    stop_loss: float | None = Field(
        default=None,
        description=(
            "Recommended stop-loss as an absolute price in the instrument's quote "
            "currency (e.g. 172.0), never a percentage. The stop belongs here, not "
            "in the rationale; null only when no stop applies."
        ),
    )
    rationale: str = Field(
        description=(
            "Which risk arguments decided the stance and the ceiling, in two to "
            "four sentences, naming the specific risks that bound the size. The "
            "level, ceiling and stop are given in their own fields, not repeated here."
        ),
    )

    @field_validator("max_position_pct", mode="before")
    @classmethod
    def _percent(cls, v):
        coerced = _coerce_optional_percent(v)
        return v if coerced is None else coerced

    @field_validator("stop_loss", mode="before")
    @classmethod
    def _nullish_float_to_none(cls, v):
        return _coerce_optional_float(v)


def render_risk_assessment(assessment: RiskAssessment) -> str:
    """Render a RiskAssessment to markdown, the risk level on its own first line."""
    stop = assessment.stop_loss if assessment.stop_loss is not None else "not provided"
    return "\n".join([
        f"**Risk Level**: {assessment.risk_level.value}",
        "",
        f"**Maximum Position**: {assessment.max_position_pct:g}% of portfolio equity",
        "",
        f"**Stop Loss**: {stop}",
        "",
        f"**Rationale**: {assessment.rationale}",
    ])


# ---------------------------------------------------------------------------
# Portfolio Manager
# ---------------------------------------------------------------------------


class PortfolioDecision(BaseModel):
    """Structured output produced by the Portfolio Manager.

    The model fills every field as part of its primary LLM call; no separate
    extraction pass is required. Field descriptions double as the model's
    output instructions, so the prompt body only needs to convey context and
    the rating-scale guidance.
    """

    model_config = ConfigDict(json_schema_extra=_always_asked("position_size_pct", "risk_level"))

    rating: PortfolioRating = Field(
        description=(
            "The final position rating. Exactly one of Buy / Overweight / Hold / "
            "Underweight / Sell, picked based on the analysts' debate. "
            "Conflicting arguments alone are not a reason to Hold: commit to the "
            "stronger side, sized by how decisively it wins. Choose Hold only "
            "when the evidence is still balanced after weighing, or too thin to "
            "support a call."
        ),
    )
    position_size_pct: float | None = Field(
        default=None,
        description=(
            "The target position after this decision, as a percentage of total "
            "portfolio equity: a number from 0 to 100 (e.g. 4.5), never a range. "
            "0 means hold no position (a Sell exits; a Hold with no position "
            "stays out). Stay within Risk Management's maximum unless the thesis "
            "says why not."
        ),
    )
    risk_level: RiskLevel | None = Field(
        default=None,
        description=(
            "The risk stance this decision adopts. Exactly one of Aggressive / "
            "Neutral / Conservative, normally Risk Management's; say so in the "
            "thesis if you depart from it."
        ),
    )
    executive_summary: str = Field(
        description=(
            "A concise action plan covering entry strategy, position sizing, "
            "key risk levels, and time horizon. Two to four sentences."
        ),
    )
    investment_thesis: str = Field(
        description=(
            "Detailed reasoning anchored in specific evidence from the analysts' "
            "debate. If prior lessons are referenced in the prompt context, "
            "incorporate them; otherwise rely solely on the current analysis."
        ),
    )
    price_target: float | None = Field(
        default=None,
        description="Optional target price in the instrument's quote currency.",
    )
    time_horizon: str | None = Field(
        default=None,
        description="Optional recommended holding period, e.g. '3-6 months'.",
    )

    @field_validator("price_target", mode="before")
    @classmethod
    def _nullish_float_to_none(cls, v):
        return _coerce_optional_float(v)

    @field_validator("position_size_pct", mode="before")
    @classmethod
    def _percent_or_none(cls, v):
        return _coerce_optional_percent(v)


def render_pm_decision(decision: PortfolioDecision) -> str:
    """Render a PortfolioDecision back to the markdown shape the rest of the system expects.

    Memory log, CLI display, and saved report files all read this markdown,
    so the rendered output preserves the exact section headers (``**Rating**``,
    ``**Executive Summary**``, ``**Investment Thesis**``) that downstream
    parsers and the report writers already handle.
    """
    size = (f"{decision.position_size_pct:g}% of portfolio equity"
            if decision.position_size_pct is not None else "not provided")
    parts = [
        f"**Rating**: {decision.rating.value}",
        "",
        f"**Position Size**: {size}",
        "",
        f"**Risk Level**: {decision.risk_level.value if decision.risk_level else 'not provided'}",
        "",
        f"**Executive Summary**: {decision.executive_summary}",
        "",
        f"**Investment Thesis**: {decision.investment_thesis}",
    ]
    # Named even when absent: a missing line reads as a field nobody asked for,
    # so a reader cannot tell "no target" from "target not reported".
    target = decision.price_target if decision.price_target is not None else "not provided"
    parts.extend(["", f"**Price Target**: {target}"])
    parts.extend(["", f"**Time Horizon**: {decision.time_horizon or 'not provided'}"])
    return "\n".join(parts)
