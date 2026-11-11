"""The typed contract between agents: the envelope, the verdicts in the run's state, and the closed run."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tests.test_graph_end_to_end import TRADE_DATE, ScriptedModel, _graph, offline  # noqa: F401  (offline is a fixture)
from tradingagents import usage
from tradingagents.agents import schemas
from tradingagents.agents.rating import RATING_REVIEW
from tradingagents.agents.structured import decide, settle
from tradingagents.feedback.store import typed_calls

AGENTS = {"market_analyst", "news_analyst", "fundamentals_analyst", "bull_researcher", "bear_researcher",
          "research_manager", "trader", "aggressive_analyst", "conservative_analyst", "neutral_analyst",
          "risk_manager", "portfolio_manager"}


# --- the envelope -----------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("raw,read", [(0.7, 0.7), ("0.7", 0.7), ("70%", 0.7), (70, 0.7), ("high", 0.8),
                                      ("fairly sure", None), (None, None), (140, None), (True, None)])
def test_a_confidence_is_read_as_a_number_from_0_to_1_or_dropped(raw, read):
    proposal = schemas.TraderProposal(action="Buy", reasoning="r", confidence=raw)
    assert proposal.confidence == read          # an unreadable confidence never sinks the decision it came with


@pytest.mark.unit
def test_evidence_is_a_list_however_the_model_wrote_it():
    assert schemas.DebateArgument(argument="a", evidence="- RSI 71\n- ATR 7.9\n").evidence == ["RSI 71", "ATR 7.9"]
    assert schemas.DebateArgument(argument="a", evidence=None).evidence == []


@pytest.mark.unit
@pytest.mark.parametrize("schema", [schemas.ResearchPlan, schemas.TraderProposal, schemas.RiskAssessment,
                                    schemas.PortfolioDecision, schemas.DebateArgument, schemas.AnalystVerdict])
def test_every_agent_schema_asks_for_the_envelope_after_its_own_fields(schema):
    js = schema.model_json_schema()
    assert {"confidence", "evidence"} <= set(js["required"])
    assert list(js["properties"])[-2:] == ["confidence", "evidence"]     # reason first, then say how sure
    assert not {"model", "latency_s", "ok"} & set(js["properties"])      # stamped by the code, never asked of the model


@pytest.mark.unit
def test_a_render_carries_the_envelope_only_when_the_answer_gave_one():
    bare = schemas.RiskAssessment(risk_level="Neutral", max_position_pct=5, rationale="r")
    assert "Confidence" not in schemas.render_risk_assessment(bare)
    full = bare.model_copy(update={"confidence": 0.65, "evidence": ["ATR 7.9", "stop 2% away"]})
    assert schemas.render_risk_assessment(full).endswith("**Confidence**: 0.65\n\n**Evidence**: ATR 7.9; stop 2% away")
    trade = schemas.TraderProposal(action="Sell", reasoning="r", confidence=0.6)
    assert schemas.render_trader_proposal(trade).splitlines()[-1] == "FINAL TRANSACTION PROPOSAL: **SELL**"


# --- asking, and what a miss does ---------------------------------------------------

def _answers(*answers):
    """A structured model that gives ``answers`` in turn (an exception is raised), recording what it was asked."""
    asked, queue = [], list(answers)
    structured = MagicMock()

    def invoke(prompt):
        asked.append(prompt)
        answer = queue.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    structured.invoke.side_effect = invoke
    return structured, asked


@pytest.mark.unit
def test_the_verdict_is_stamped_with_the_model_and_the_time_by_the_code_that_called(tmp_path):
    from tradingagents.dataflows.config import run_config

    structured = MagicMock()

    def invoke(prompt):       # what the Ollama client does on every answered call
        usage.record({"ts": "2026-11-11T09:00:00", "model": "gemma4:27b", "tokens_in": 900, "tokens_out": 120, "seconds": 3.2})
        return schemas.TraderProposal(action="Sell", reasoning="Below the 50 SMA.", stop_loss=212.0,
                                      confidence=0.7, evidence=["close 204 under SMA 209"])

    structured.invoke.side_effect = invoke
    with run_config({"results_dir": str(tmp_path)}):
        answer, verdict = decide(structured, "prompt", "Trader")
    assert answer.stop_loss == 212.0
    assert (verdict.agent, verdict.ok, verdict.call, verdict.thesis) == ("Trader", True, "Sell", "Below the 50 SMA.")
    assert (verdict.confidence, verdict.evidence) == (0.7, ["close 204 under SMA 209"])
    assert (verdict.model, verdict.tokens_in, verdict.tokens_out, verdict.attempts) == ("gemma4:27b", 900, 120, 1)
    assert verdict.latency_s is not None and verdict.fields["stop_loss"] == 212.0 and "confidence" not in verdict.fields


@pytest.mark.unit
def test_an_answer_that_does_not_validate_is_asked_for_once_more_with_the_error():
    good = schemas.ResearchPlan(recommendation="Sell", rationale="r", strategic_actions="a")
    structured, asked = _answers(ValueError("recommendation: Input should be 'Buy', 'Sell'"), good)
    outcome = settle(structured, MagicMock(), "PROMPT", "Research Manager")
    assert outcome.result is good and outcome.verdict.ok and outcome.verdict.attempts == 2
    assert asked[0] == "PROMPT" and asked[1].startswith("PROMPT") and "Input should be 'Buy', 'Sell'" in asked[1]
    assert "contract_failure" not in outcome.update("research_manager")


@pytest.mark.unit
def test_two_misses_close_the_run_and_no_prose_is_asked_for():
    structured, asked = _answers(ValueError("bad json"), None)      # None: the model answered in plain text
    plain = MagicMock()
    outcome = settle(structured, plain, [{"role": "user", "content": "PROMPT"}], "Trader")
    assert outcome.closed and len(asked) == 2 and asked[1][-1]["role"] == "user"
    plain.invoke.assert_not_called()
    update = outcome.update("trader")
    assert update["contract_failure"].startswith("Trader: ") and update["verdicts"]["trader"]["ok"] is False
    assert "No valid answer from the Trader" in outcome.markdown(schemas.render_trader_proposal)


@pytest.mark.unit
def test_an_unreachable_model_is_raised_not_recorded_as_a_failed_answer():
    structured, _ = _answers(ConnectionError("PAIR is down"))
    with pytest.raises(ConnectionError):
        settle(structured, MagicMock(), "PROMPT", "Trader")


# --- the run ------------------------------------------------------------------------

@pytest.mark.unit
def test_a_run_keeps_every_agents_typed_verdict_and_the_store_reads_them(tmp_path, monkeypatch, offline):  # noqa: F811
    graph = _graph(tmp_path, monkeypatch, ScriptedModel(structured=True),
                   feedback_db_url=f"sqlite:///{tmp_path / 'feedback.db'}")
    state, signal = graph.propagate("NVDA", TRADE_DATE)

    assert signal == "Overweight" and state["contract_failure"] == ""
    verdicts = state["verdicts"]
    assert set(verdicts) == AGENTS and all(v["ok"] for v in verdicts.values())
    assert {k: verdicts[k]["call"] for k in ("market_analyst", "bull_researcher", "bear_researcher", "trader",
                                             "conservative_analyst", "risk_manager", "portfolio_manager")} == \
        {"market_analyst": "Bullish", "bull_researcher": "Bullish", "bear_researcher": "Bearish", "trader": "Buy",
         "conservative_analyst": "Conservative", "risk_manager": "Neutral", "portfolio_manager": "Overweight"}
    assert verdicts["market_analyst"]["confidence"] == 0.8 and verdicts["bull_researcher"]["evidence"] == ["RSI 71"]
    assert "Bull Analyst: My case." in state["investment_debate_state"]["history"]

    rows = {r["agent"]: r for r in graph.feedback.verdicts("NVDA", TRADE_DATE)}
    assert set(rows) == AGENTS
    assert (rows["risk_manager"]["call"], rows["risk_manager"]["confidence"], rows["risk_manager"]["evidence"]) == \
        ("Neutral", 0.6, ["ATR 7.9"])
    decision = graph.feedback.pending()[0]
    assert (decision["trader_action"], decision["rm_stop_loss"], decision["rm_max_position_pct"],
            decision["research_recommendation"]) == ("Buy", 180.0, 5.0, "Overweight")


@pytest.mark.unit
def test_a_deciding_agent_that_cannot_meet_its_contract_closes_the_run_as_review(tmp_path, monkeypatch, offline):  # noqa: F811
    model = ScriptedModel(structured=True, broken=(schemas.TraderProposal,))
    graph = _graph(tmp_path, monkeypatch, model)
    state, signal = graph.propagate("NVDA", TRADE_DATE)

    assert signal == state["final_rating"] == RATING_REVIEW and state["final_position_size_pct"] is None
    assert state["contract_failure"].startswith("Trader: ")
    assert "**Failed**: Trader" in state["final_trade_decision"]
    verdicts = state["verdicts"]
    assert verdicts["trader"]["ok"] is False and verdicts["trader"]["attempts"] == 2 and verdicts["research_manager"]["ok"]
    # Nobody after the Trader was asked: not the risk analysts, not Risk Management, not the Portfolio Manager.
    assert not {"aggressive_analyst", "conservative_analyst", "neutral_analyst", "risk_manager", "portfolio_manager"} & set(verdicts)
    assert state["risk_debate_state"]["count"] == 0
    assert [e["rating"] for e in graph.memory_log.load_entries()] == [RATING_REVIEW]


@pytest.mark.unit
def test_an_analyst_without_a_verdict_keeps_its_report_and_the_run_goes_on(tmp_path, monkeypatch, offline):  # noqa: F811
    graph = _graph(tmp_path, monkeypatch, ScriptedModel(structured=True, broken=(schemas.AnalystVerdict,)))
    state, signal = graph.propagate("NVDA", TRADE_DATE)
    assert signal == "Overweight" and state["contract_failure"] == "" and state["market_report"].strip()
    assert state["verdicts"]["market_analyst"]["ok"] is False and state["verdicts"]["market_analyst"]["call"] is None


@pytest.mark.unit
def test_analyst_verdicts_can_be_switched_off(tmp_path, monkeypatch, offline):  # noqa: F811
    graph = _graph(tmp_path, monkeypatch, ScriptedModel(structured=True), analyst_verdicts=False)
    state, _ = graph.propagate("NVDA", TRADE_DATE)
    assert set(state["verdicts"]) == AGENTS - {"market_analyst", "news_analyst", "fundamentals_analyst"}


# --- the store reads the typed call, not the prose ------------------------------------

@pytest.mark.unit
def test_the_store_takes_the_typed_call_over_what_the_prose_says():
    state = {
        "trader_investment_plan": "**Action**: Buy\n\n**Stop Loss**: 1.0\n\nFINAL TRANSACTION PROPOSAL: **BUY**",
        "verdicts": {"trader": {"ok": True, "call": "Sell", "fields": {"entry_price": 204.0, "stop_loss": 212.0}}},
        "risk_assessment": "**Maximum Position**: 3% of portfolio equity\n\n**Stop Loss**: 209.0",   # no verdict: read from text
    }
    calls = typed_calls(state)
    assert (calls["trader_action"], calls["trader_entry_price"], calls["trader_stop_loss"]) == ("Sell", 204.0, 212.0)
    assert (calls["rm_max_position_pct"], calls["rm_stop_loss"]) == (3.0, 209.0)
    failed = typed_calls({"verdicts": {"trader": {"ok": False, "call": None, "fields": {}}}, "trader_investment_plan": ""})
    assert failed["trader_action"] is None
