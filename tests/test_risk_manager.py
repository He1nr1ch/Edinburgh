"""Risk Management's verdict and the Portfolio Manager's size are typed, so the
feedback loop scores values rather than prose."""

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
from tradingagents.agents.managers.risk_manager import create_risk_manager
from tradingagents.agents.rating import RATING_REVIEW, extract_risk_level, parse_risk_level
from tradingagents.agents.schemas import (
    PortfolioDecision,
    PortfolioRating,
    RiskAssessment,
    RiskLevel,
    render_pm_decision,
    render_risk_assessment,
)

RISK_STATE = {
    "history": "Aggressive: go big. Conservative: trim. Neutral: standard.",
    "aggressive_history": "a", "conservative_history": "c", "neutral_history": "n",
    "current_aggressive_response": "", "current_conservative_response": "",
    "current_neutral_response": "", "latest_speaker": "Neutral", "count": 3,
}


def _state(**extra):
    return {
        "company_of_interest": "NVDA", "trade_date": "2026-10-16", "asset_type": "stock",
        "instrument_context": "", "portfolio_context": "", "past_context": "",
        "market_report": "Price 214.13, ATR 7.9, 200-SMA 202.66.", "news_report": "", "fundamentals_report": "",
        "investment_plan": "**Recommendation**: Hold", "trader_investment_plan": "**Action**: Hold",
        "risk_debate_state": dict(RISK_STATE), **extra,
    }


def _structured(result, captured=None):
    seen = captured if captured is not None else {}
    structured = MagicMock()
    structured.invoke.side_effect = lambda prompt: (seen.__setitem__("prompt", prompt) or result)
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


def _free_text(text):
    llm = MagicMock()
    llm.with_structured_output.side_effect = NotImplementedError
    llm.invoke.return_value = MagicMock(content=text)
    return llm


# --- the schema ---------------------------------------------------------------

@pytest.mark.unit
def test_a_risk_assessment_renders_with_its_level_first():
    rendered = render_risk_assessment(RiskAssessment(
        risk_level=RiskLevel.CONSERVATIVE, max_position_pct="4.5%", stop_loss="$209", rationale="Credit stress."))

    assert rendered.startswith("**Risk Level**: Conservative")
    assert "**Maximum Position**: 4.5% of portfolio equity" in rendered
    assert "**Stop Loss**: 209.0" in rendered and "Credit stress." in rendered
    assert extract_risk_level(rendered) == "Conservative"


@pytest.mark.unit
@pytest.mark.parametrize("bad", ["150", -1, "4-5%", "a lot"])
def test_a_ceiling_that_is_not_a_percentage_fails_validation(bad):
    with pytest.raises(ValidationError):
        RiskAssessment(risk_level=RiskLevel.NEUTRAL, max_position_pct=bad, rationale="r")


@pytest.mark.unit
@pytest.mark.parametrize("written, expected", [
    ("4.5", 4.5), ("4.5%", 4.5), (0, 0.0), ("0%", 0.0), ("100", 100.0),
    ("4-5%", None), ("150", None), ("N/A", None), (None, None), ("about 5", None),
])
def test_the_portfolio_managers_size_is_one_percentage_or_nothing(written, expected):
    decision = PortfolioDecision(rating=PortfolioRating.HOLD, executive_summary="s", investment_thesis="t",
                                 position_size_pct=written)
    assert decision.position_size_pct == expected


@pytest.mark.unit
def test_the_decision_renders_size_and_stance_after_the_rating():
    rendered = render_pm_decision(PortfolioDecision(
        rating=PortfolioRating.OVERWEIGHT, executive_summary="s", investment_thesis="t",
        position_size_pct=4.5, risk_level=RiskLevel.NEUTRAL))
    lines = [line for line in rendered.splitlines() if line]

    assert lines[:3] == ["**Rating**: Overweight", "**Position Size**: 4.5% of portfolio equity", "**Risk Level**: Neutral"]


@pytest.mark.unit
def test_a_decision_without_size_or_stance_says_so():
    rendered = render_pm_decision(PortfolioDecision(rating=PortfolioRating.HOLD, executive_summary="s", investment_thesis="t"))
    assert "**Position Size**: not provided" in rendered and "**Risk Level**: not provided" in rendered


@pytest.mark.unit
def test_the_model_is_always_asked_for_size_stance_and_stop():
    """Optional keys are left out by a schema-constrained model, which then writes the
    value into the prose; the keys are required in the schema while staying nullable."""
    assert {"position_size_pct", "risk_level"} <= set(PortfolioDecision.model_json_schema()["required"])
    assert "stop_loss" in RiskAssessment.model_json_schema()["required"]
    # ...and a decision built without them still validates on the Python side.
    assert PortfolioDecision(rating=PortfolioRating.HOLD, executive_summary="s", investment_thesis="t").risk_level is None


# --- reading a stance from text -----------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("text, level", [
    ("**Risk Level**: Neutral\n\nok", "Neutral"),
    ("Risk level - aggressive", "Aggressive"),
    ("Risk Level：Conservative", "Conservative"),            # fullwidth colon
    ("Risk Level: Neutral ... Risk Level: Conservative", None),  # disagreeing labels
    ("The neutral analyst spoke last.", None),                   # no label
    ("", None),
])
def test_extract_risk_level(text, level):
    assert extract_risk_level(text) == level
    assert parse_risk_level(text) == (level or RATING_REVIEW)


# --- the Risk Manager node ----------------------------------------------------

@pytest.mark.unit
def test_the_risk_manager_settles_the_debate_into_a_typed_verdict():
    captured = {}
    assessment = RiskAssessment(risk_level=RiskLevel.CONSERVATIVE, max_position_pct=3, stop_loss=209.0, rationale="r")

    out = create_risk_manager(_structured(assessment, captured))(_state())

    assert out["risk_level"] == "Conservative"
    assert out["risk_assessment"].startswith("**Risk Level**: Conservative")
    prompt = captured["prompt"]
    assert "Aggressive: go big." in prompt and "**Action**: Hold" in prompt and "ATR 7.9" in prompt
    assert "## Output" in prompt and "**Risk Level**" in prompt.split("## Output", 1)[1]


@pytest.mark.unit
def test_the_risk_manager_falls_back_to_text_and_reads_the_stance_from_it():
    out = create_risk_manager(_free_text("**Risk Level**: Aggressive\n\nFull size."))(_state())
    assert out == {"risk_assessment": "**Risk Level**: Aggressive\n\nFull size.", "risk_level": "Aggressive"}


@pytest.mark.unit
def test_a_verdict_naming_no_stance_is_marked_for_review():
    out = create_risk_manager(_free_text("Be careful out there."))(_state())
    assert out["risk_level"] == RATING_REVIEW


# --- the Portfolio Manager takes the verdict ------------------------------------

@pytest.mark.unit
def test_the_portfolio_manager_is_shown_the_verdict_and_records_size_and_stance():
    captured = {}
    decision = PortfolioDecision(rating=PortfolioRating.HOLD, executive_summary="s", investment_thesis="t",
                                 position_size_pct=4, risk_level=RiskLevel.CONSERVATIVE)
    state = _state(risk_assessment="**Risk Level**: Conservative\n\n**Maximum Position**: 5% of portfolio equity",
                   risk_level="Conservative")

    out = create_portfolio_manager(_structured(decision, captured))(state)

    assert "Risk Management's verdict" in captured["prompt"] and "Maximum Position" in captured["prompt"]
    assert (out["final_rating"], out["final_position_size_pct"], out["final_risk_level"]) == ("Hold", 4.0, "Conservative")


@pytest.mark.unit
def test_a_decision_that_names_no_stance_takes_risk_managements():
    decision = PortfolioDecision(rating=PortfolioRating.BUY, executive_summary="s", investment_thesis="t")
    out = create_portfolio_manager(_structured(decision))(_state(risk_assessment="**Risk Level**: Aggressive", risk_level="Aggressive"))
    assert out["final_risk_level"] == "Aggressive" and out["final_position_size_pct"] is None


@pytest.mark.unit
def test_without_a_verdict_the_portfolio_manager_is_told_to_set_the_stance_itself():
    captured = {}
    decision = PortfolioDecision(rating=PortfolioRating.HOLD, executive_summary="s", investment_thesis="t")

    out = create_portfolio_manager(_structured(decision, captured))(_state())

    assert "not available this run" in captured["prompt"]
    assert out["final_risk_level"] == RATING_REVIEW


@pytest.mark.unit
def test_the_free_text_fallback_reads_size_as_unknown_and_the_stance_from_the_text():
    out = create_portfolio_manager(_free_text("**Rating**: Sell\n**Risk Level**: Conservative\n\nExit."))(_state(risk_level="Neutral"))
    assert (out["final_rating"], out["final_position_size_pct"], out["final_risk_level"]) == ("Sell", None, "Conservative")


# --- the graph ------------------------------------------------------------------

@pytest.mark.unit
def test_the_risk_manager_sits_between_the_risk_debate_and_the_portfolio_manager():
    from tradingagents.graph.conditional_logic import ConditionalLogic
    from tradingagents.graph.setup import RISK_ANALYSIS_PATH_MAP, GraphSetup

    workflow = GraphSetup(MagicMock(), MagicMock(), ConditionalLogic(), 20).setup_graph(("market",))
    graph = workflow.compile().get_graph()
    edges = {(e.source, e.target) for e in graph.edges}

    assert "Risk Manager" in RISK_ANALYSIS_PATH_MAP
    assert ("Risk Manager", "Portfolio Manager") in edges
    assert ("Neutral Analyst", "Risk Manager") in edges and ("Neutral Analyst", "Portfolio Manager") in edges
