"""The feedback loop: typed calls recorded, outcomes scored per agent, a track record shown back."""

from unittest.mock import MagicMock

import pandas as pd
import pytest
from sqlalchemy import create_engine, select

from tradingagents.agents.rating import RATING_REVIEW
from tradingagents.feedback import FeedbackStore, extract, scoring
from tradingagents.feedback.schema import agent_scores, decisions, outcomes
from tradingagents.memory.settlement import Window

DECISION_TEXT = "**Rating**: Buy\n\n**Position Size**: 4% of portfolio equity\n\n**Risk Level**: Neutral\n\n**Price Target**: 250.0"
RISK_TEXT = "**Risk Level**: Neutral\n\n**Maximum Position**: 5% of portfolio equity\n\n**Stop Loss**: 209.0\n\n**Rationale**: r"
TRADER_TEXT = "**Action**: Buy\n\n**Entry Price**: 214.5\n\n**Stop Loss**: 205.0\n\nFINAL TRANSACTION PROPOSAL: **BUY**"
PLAN_TEXT = "**Recommendation**: Overweight\n\n**Rationale**: r"


def _state(**over):
    state = {
        "asset_type": "stock", "final_trade_decision": DECISION_TEXT, "final_rating": "Buy",
        "final_position_size_pct": 4.0, "final_risk_level": "Neutral", "risk_level": "Neutral",
        "risk_assessment": RISK_TEXT, "trader_investment_plan": TRADER_TEXT, "investment_plan": PLAN_TEXT,
    }
    state.update(over)
    return state


def _window(closes, bench=(400.0, 404.0), start="2026-01-06"):
    dates = pd.bdate_range(start, periods=len(closes))
    return Window(entry_date=dates[0].strftime("%Y-%m-%d"), exit_date=dates[-1].strftime("%Y-%m-%d"),
                  entry=float(closes[0]), exit=float(closes[-1]), benchmark_entry=bench[0], benchmark_exit=bench[1],
                  closes=pd.Series(closes, index=dates, dtype=float))


@pytest.fixture
def store():
    return FeedbackStore({"feedback_hold_band_pct": 2.0}, engine=create_engine("sqlite://"))


# --- the rules ---------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("call, direction", [("Buy", 1), ("Overweight", 1), ("Hold", 0), ("Underweight", -1),
                                             ("Sell", -1), ("BUY", 1), (RATING_REVIEW, None), (None, None), ("Maybe", None)])
def test_direction_of_every_call(call, direction):
    assert scoring.direction_of(call) == direction


@pytest.mark.unit
@pytest.mark.parametrize("direction, raw, expected", [
    (1, 0.05, "win"), (1, -0.05, "loss"), (-1, -0.05, "win"), (-1, 0.05, "loss"), (1, 0.0, "flat"),
    (0, 0.01, "flat"), (0, 0.05, "missed"), (0, -0.05, "avoided"), (None, 0.1, None),
])
def test_verdicts(direction, raw, expected):
    assert scoring.verdict(direction, raw, hold_band_pct=2.0) == expected


@pytest.mark.unit
@pytest.mark.parametrize("level, exposure, stop_hit, expected", [
    ("Aggressive", 0.05, False, True), ("Aggressive", 0.01, False, False), ("Aggressive", -0.03, False, False),
    ("Conservative", -0.01, False, True), ("Conservative", 0.03, True, True), ("Conservative", 0.03, False, False),
    ("Neutral", 0.01, False, True), ("Neutral", -0.01, False, True), ("Neutral", -0.05, False, False),
    ("Neutral", 0.04, True, False), (RATING_REVIEW, 0.1, False, None), (None, 0.1, False, None),
])
def test_a_stance_is_right_when_the_outcome_justified_it(level, exposure, stop_hit, expected):
    assert scoring.stance_correct(level, exposure, stop_hit, hold_band_pct=2.0) is expected


@pytest.mark.unit
def test_a_stop_is_hit_below_for_a_long_and_above_for_a_short():
    assert scoring.stop_was_hit(1, 205.0, low=204.0, high=220.0) is True
    assert scoring.stop_was_hit(1, 205.0, low=206.0, high=220.0) is False
    assert scoring.stop_was_hit(-1, 220.0, low=200.0, high=221.0) is True
    assert scoring.stop_was_hit(0, 205.0, low=204.0, high=220.0) is True      # a Hold is long exposure
    assert scoring.stop_was_hit(1, None, low=0, high=1) is None


@pytest.mark.unit
def test_scoring_a_buy_that_won_marks_every_agent():
    decision = {"rating": "Buy", "position_size_pct": 4.0, "rm_risk_level": "Neutral", "rm_stop_loss": 209.0,
                "trader_action": "Buy", "trader_stop_loss": 205.0, "research_recommendation": "Overweight"}

    scored = scoring.score(decision, raw_return=0.05, low=212.0, high=226.0, hold_band_pct=2.0)

    assert (scored.direction, scored.verdict, scored.stop_hit) == (1, "win", False)
    assert scored.directional_return == pytest.approx(0.05) and scored.weighted_return == pytest.approx(0.002)
    marks = {row["agent"]: (row["call"], row["correct"]) for row in scored.agent_scores}
    assert marks == {"research_manager": ("Overweight", True), "trader": ("Buy", True),
                     "portfolio_manager": ("Buy", True), "risk_manager": ("Neutral", True)}


@pytest.mark.unit
def test_scoring_a_sell_that_lost_and_a_stop_that_was_hit():
    decision = {"rating": "Sell", "position_size_pct": 3.0, "rm_risk_level": "Conservative", "rm_stop_loss": 230.0,
                "trader_action": "Sell", "research_recommendation": "Underweight"}

    scored = scoring.score(decision, raw_return=0.08, low=214.0, high=232.0, hold_band_pct=2.0)

    assert (scored.direction, scored.verdict, scored.stop_hit) == (-1, "loss", True)
    assert scored.directional_return == pytest.approx(-0.08)
    marks = {row["agent"]: row["correct"] for row in scored.agent_scores}
    assert marks["portfolio_manager"] is False and marks["trader"] is False
    assert marks["risk_manager"] is True          # caution was warranted: the short lost and the stop was hit


@pytest.mark.unit
def test_an_unreadable_decision_is_scored_as_no_call():
    scored = scoring.score({"rating": RATING_REVIEW, "trader_action": None}, raw_return=0.05, low=1, high=2, hold_band_pct=2.0)
    assert scored.direction == 0 and scored.verdict == "flat" and scored.agent_scores == []


# --- reading the calls out of the text --------------------------------------------

@pytest.mark.unit
def test_the_typed_calls_are_read_from_the_rendered_text():
    assert extract.trader_fields(TRADER_TEXT) == {"trader_action": "Buy", "trader_entry_price": 214.5, "trader_stop_loss": 205.0}
    assert extract.risk_fields(RISK_TEXT) == {"rm_max_position_pct": 5.0, "rm_stop_loss": 209.0}
    assert extract.research_recommendation(PLAN_TEXT) == "Overweight"
    assert extract.labelled_number(DECISION_TEXT, "Price Target") == 250.0
    assert extract.labelled_number("**Stop Loss**: not provided", "Stop Loss") is None
    assert extract.labelled_number("Stop Loss: $1,234.50", "Stop Loss") == 1234.5
    assert extract.trader_action("Hold for now.\n\nFINAL TRANSACTION PROPOSAL: **HOLD**") == "Hold"
    assert extract.trader_action("") is None


# --- the store -------------------------------------------------------------------

@pytest.mark.unit
def test_a_store_without_a_url_is_disabled_and_silent():
    store = FeedbackStore({})
    assert not store.enabled
    assert store.record("NVDA", "2026-01-05", _state()) is None
    assert store.pending() == [] and store.render_context("NVDA") == "" and store.scorecard() == {}


@pytest.mark.unit
def test_recording_a_run_keeps_every_typed_call(store):
    settings = {"llm_provider": "ollama", "quick_think_llm": "q", "deep_think_llm": "d", "analysts": ["market", "news"]}

    decision_id = store.record("NVDA", "2026-01-05", _state(), settings, holding_days=5, benchmark="SPY")

    with store.engine.connect() as conn:
        row = conn.execute(select(decisions).where(decisions.c.id == decision_id)).mappings().one()
    assert (row["rating"], row["position_size_pct"], row["risk_level"], row["price_target"]) == ("Buy", 4.0, "Neutral", 250.0)
    assert (row["rm_risk_level"], row["rm_max_position_pct"], row["rm_stop_loss"]) == ("Neutral", 5.0, 209.0)
    assert (row["trader_action"], row["trader_entry_price"], row["trader_stop_loss"]) == ("Buy", 214.5, 205.0)
    assert row["research_recommendation"] == "Overweight"
    assert (row["status"], row["holding_days"], row["benchmark"], row["analysts"]) == ("pending", 5, "SPY", "market,news")
    assert row["llm_provider"] == "ollama" and row["decision_text"] == DECISION_TEXT


@pytest.mark.unit
def test_a_rerun_replaces_a_pending_row_but_not_a_settled_one(store):
    first = store.record("NVDA", "2026-01-05", _state())
    again = store.record("NVDA", "2026-01-05", _state(final_rating="Sell", final_trade_decision="**Rating**: Sell"))
    assert again == first and store.pending()[0]["rating"] == "Sell"

    store.settle(store.pending()[0], _window([100, 101, 102, 103, 104, 105]))
    third = store.record("NVDA", "2026-01-05", _state(final_rating="Hold", final_trade_decision="**Rating**: Hold"))
    with store.engine.connect() as conn:
        rating = conn.execute(select(decisions.c.rating).where(decisions.c.id == third)).scalar()
    assert third == first and rating == "Sell"


@pytest.mark.unit
def test_a_review_stance_from_risk_management_is_recorded_as_no_stance(store):
    store.record("NVDA", "2026-01-05", _state(risk_level=RATING_REVIEW, risk_assessment="unclear"))
    assert store.pending()[0]["rm_risk_level"] is None


@pytest.mark.unit
def test_settling_writes_the_outcome_and_the_scores(store):
    store.record("NVDA", "2026-01-05", _state())
    closes = [220.0, 224.0, 217.8, 226.0, 230.0, 233.2]      # +6%, dipped to -1%, never near the 209 stop

    done = store.settle_due(lambda t, d, days, bench: _window(closes), lambda t, d: "Momentum carried it.")

    assert done.settled == [("NVDA", "2026-01-05")] and store.pending() == []
    with store.engine.connect() as conn:
        outcome = conn.execute(select(outcomes)).mappings().one()
        scores = {r["agent"]: r for r in conn.execute(select(agent_scores)).mappings().all()}
    assert outcome["raw_return"] == pytest.approx(0.06) and outcome["alpha"] == pytest.approx(0.05)
    assert outcome["max_drawdown"] == pytest.approx(-0.01) and outcome["max_runup"] == pytest.approx(0.06)
    assert (outcome["direction"], outcome["verdict"], outcome["stop_hit"]) == (1, "win", False)
    assert outcome["weighted_return"] == pytest.approx(0.0024) and outcome["reflection"] == "Momentum carried it."
    assert {a: s["correct"] for a, s in scores.items()} == {
        "research_manager": True, "trader": True, "portfolio_manager": True, "risk_manager": True}


@pytest.mark.unit
def test_a_window_that_has_not_traded_leaves_the_decision_pending(store):
    store.record("NVDA", "2026-01-05", _state())
    done = store.settle_due(lambda *a: None)
    assert done.settled == [] and len(store.pending()) == 1


@pytest.mark.unit
def test_a_price_failure_is_reported_and_the_rest_still_settle(store):
    store.record("AAPL", "2026-01-05", _state())
    store.record("NVDA", "2026-01-05", _state())

    def fetch(ticker, *a):
        if ticker == "AAPL":
            raise OSError("vendor down")
        return _window([100, 101, 102, 103, 104, 105])

    done = store.settle_due(fetch)

    assert done.settled == [("NVDA", "2026-01-05")]
    assert done.failed == [("AAPL", "2026-01-05", "prices unavailable: vendor down")]


@pytest.mark.unit
def test_the_track_record_summarises_each_agent_and_the_tickers_own_history(store):
    store.record("NVDA", "2026-01-05", _state())
    store.record("NVDA", "2026-01-12", _state(final_rating="Sell", final_trade_decision="**Rating**: Sell", final_position_size_pct=0.0,
                                               trader_investment_plan="FINAL TRANSACTION PROPOSAL: **SELL**", risk_level="Conservative",
                                               final_risk_level="Conservative", risk_assessment="**Risk Level**: Conservative\n**Maximum Position**: 2%"))
    store.record("AAPL", "2026-01-12", _state(final_rating="Hold", final_trade_decision="**Rating**: Hold",
                                               trader_investment_plan="FINAL TRANSACTION PROPOSAL: **HOLD**"))
    windows = {("NVDA", "2026-01-05"): _window([220, 224, 228, 232, 236, 242]),            # Buy, +10%: win
               ("NVDA", "2026-01-12"): _window([220, 222, 224, 226, 228, 231], start="2026-01-13"),  # Sell, +5%: loss
               ("AAPL", "2026-01-12"): _window([220, 220, 221, 220, 222, 222.2], start="2026-01-13")}  # Hold, +1%: flat
    store.settle_due(lambda t, d, *a: windows[(t, d)], lambda t, d: "Lesson for " + t)

    text = store.render_context("NVDA")

    assert text.startswith("Track record over the last 3 settled decisions")
    assert "Portfolio Manager: 2/3 right" in text and "Buy 1/1 (+10.0%)" in text and "Sell 0/1 (-5.0%)" in text
    assert "Hold 1/1" in text and "Trader: 2/3 right" in text
    # Neutral on a +10% win, Conservative on a short that lost, Neutral on a flat Hold: all justified.
    assert "Risk Manager: 3/3 right" in text and "Conservative 1/1 (-5.0%)" in text
    assert "Past settled decisions on NVDA:" in text
    assert "2026-01-12: Sell, size 0%, risk Conservative → loss, return +5.0%" in text
    assert "2026-01-05: Buy, size 4%, risk Neutral → win, return +10.0%, alpha +9.0%. Lesson: Lesson for NVDA" in text
    assert "Favour the calls and stances that have been right" in text
    card = store.scorecard()
    assert card["portfolio_manager"]["hit_rate"] == pytest.approx(2 / 3)
    assert card["trader"]["by_call"]["Buy"]["mean"] == pytest.approx(0.10)


@pytest.mark.unit
def test_the_track_record_is_point_in_time(store):
    store.record("NVDA", "2026-01-05", _state())
    store.settle_due(lambda *a: _window([100, 102, 104, 106, 108, 110]))       # exits 2026-01-13

    assert store.render_context("NVDA", as_of="2026-01-12") == ""
    assert "Buy 1/1" in store.render_context("NVDA", as_of="2026-01-13")


@pytest.mark.unit
def test_a_lesson_settled_later_is_attached_to_the_outcome(store):
    store.record("NVDA", "2026-01-05", _state())
    store.settle_due(lambda *a: _window([100, 102, 104, 106, 108, 110]))
    store.attach_reflection("NVDA", "2026-01-05", "Trend-following paid.")
    assert store.ticker_history("NVDA")[0]["reflection"] == "Trend-following paid."


# --- the graph records and scores -----------------------------------------------------

@pytest.mark.unit
def test_the_graph_records_its_decision_scores_it_later_and_shows_the_record(tmp_path, monkeypatch):
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    from tradingagents.memory import TradingMemoryLog, settlement

    graph = object.__new__(TradingAgentsGraph)
    graph.config = {"results_dir": str(tmp_path), "llm_provider": "ollama", "deep_think_llm": "d", "quick_think_llm": "q",
                    "max_debate_rounds": 1, "max_risk_discuss_rounds": 1, "data_vendors": {}, "tool_vendors": {},
                    "holding_period_days": 5, "benchmark_ticker": None, "memory_log_path": str(tmp_path / "m.md"),
                    "feedback_db_url": f"sqlite:///{tmp_path / 'feedback.db'}"}
    graph.selected_analysts = ("market",)
    graph.memory_log = TradingMemoryLog(graph.config)
    graph.feedback = FeedbackStore(graph.config)
    graph.reflector = MagicMock()
    graph.reflector.reflect_on_final_decision.return_value = "Buying the dip worked."
    state = {"company_of_interest": "NVDA", "trade_date": "2026-01-05", "market_report": "M", "news_report": "",
             "fundamentals_report": "", "investment_plan": PLAN_TEXT, "trader_investment_plan": TRADER_TEXT,
             "risk_assessment": RISK_TEXT, "risk_level": "Neutral", "final_position_size_pct": 4.0, "final_risk_level": "Neutral",
             "investment_debate_state": {"bull_history": "", "bear_history": "", "history": "", "current_response": ""},
             "risk_debate_state": {"aggressive_history": "", "conservative_history": "", "neutral_history": "", "history": ""},
             "final_trade_decision": DECISION_TEXT, "final_rating": "Buy"}

    graph.record_decision("NVDA", "2026-01-05", state)
    assert [p["ticker"] for p in graph.feedback.pending()] == ["NVDA"]

    closes = pd.Series([100.0, 102, 104, 106, 108, 110, 112, 114], index=pd.bdate_range("2026-01-06", periods=8))
    monkeypatch.setattr(settlement, "get_closes", lambda *a, **k: closes)
    graph.settle_all_pending()

    assert graph.feedback.pending() == []
    history = graph.feedback.ticker_history("NVDA")
    assert history[0]["verdict"] == "win" and history[0]["reflection"] == "Buying the dip worked."
    step = graph._memory_step({"company_of_interest": "NVDA", "trade_date": "2026-02-01"})
    assert "Track record over the last 1 settled decision" in step["past_context"]
    assert "Lesson: Buying the dip worked." in step["past_context"]
    assert "Past analyses of NVDA" in step["past_context"]      # the memory log's own lessons follow


@pytest.mark.unit
def test_the_risk_manager_is_shown_the_track_record():
    from tradingagents.agents.managers.risk_manager import create_risk_manager
    from tradingagents.agents.schemas import RiskAssessment, RiskLevel

    captured = {}
    structured = MagicMock()
    structured.invoke.side_effect = lambda prompt: captured.__setitem__("prompt", prompt) or RiskAssessment(
        risk_level=RiskLevel.NEUTRAL, max_position_pct=5, rationale="r")
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    state = {"company_of_interest": "NVDA", "trade_date": "2026-01-05", "asset_type": "stock", "instrument_context": "",
             "portfolio_context": "", "market_report": "", "trader_investment_plan": "T",
             "past_context": "Track record over the last 3 settled decisions: Aggressive 0/2 right.",
             "risk_debate_state": {"history": "h"}}

    create_risk_manager(llm)(state)

    assert "Aggressive 0/2 right" in captured["prompt"]
