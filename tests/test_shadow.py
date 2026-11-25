"""The Shadow Portfolio Manager: the same decision asked of a second model, recorded beside the first and never used."""

from __future__ import annotations

import contextlib
from unittest.mock import MagicMock

import pytest

from tradingagents import shadow, usage
from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
from tradingagents.agents.schemas import PortfolioDecision, PortfolioRating
from tradingagents.dataflows.config import run_config
from tradingagents.default_config import DEFAULT_CONFIG

RISK = {"history": "h", "aggressive_history": "a", "conservative_history": "c", "neutral_history": "n",
        "current_aggressive_response": "", "current_conservative_response": "", "current_neutral_response": "",
        "latest_speaker": "Neutral", "count": 1}
STATE = {"company_of_interest": "NVDA", "trade_date": "2026-11-24", "risk_debate_state": RISK,
         "investment_plan": "plan", "trader_investment_plan": "trader plan"}


def _llm(answer, seen: list):
    """A model whose structured binding gives ``answer`` (raised when it is an exception) and notes what it was asked."""
    def invoke(prompt):
        seen.append((prompt, usage.context()["agent"], usage.context()["ticker"]))
        if isinstance(answer, BaseException):
            raise answer
        return answer

    structured = MagicMock()
    structured.invoke.side_effect = invoke
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


def _decision(rating, size):
    return PortfolioDecision(rating=rating, position_size_pct=size, executive_summary="x", investment_thesis="y")


def _config(tmp_path):
    return {**DEFAULT_CONFIG, "results_dir": str(tmp_path), "shadow_dir": ""}


@pytest.mark.unit
def test_the_shadow_is_asked_the_same_prompt_and_its_answer_is_recorded_and_never_used(tmp_path):
    asked, shadow_asked = [], []
    pm = _llm(_decision(PortfolioRating.BUY, 4.0), asked)
    second = _llm(_decision(PortfolioRating.SELL, 2.0), shadow_asked)
    with run_config(_config(tmp_path)):
        usage.set_run(ticker="NVDA", run_id="r1")
        out = create_portfolio_manager(pm, second)(STATE)
        rows = shadow.read_day(next((tmp_path / "shadow").iterdir()).stem)

    assert (out["final_rating"], out["final_position_size_pct"]) == ("Buy", 4.0)      # the desk acts on the Portfolio Manager's
    assert "Sell" not in out["final_trade_decision"] and "shadow" not in str(out["verdicts"]).lower()
    assert shadow_asked[0][0] == asked[0][0]                                           # word for word
    assert shadow_asked[0][1:] == ("Shadow Portfolio Manager", "NVDA")                 # its calls are its own, on the run's ticker
    (row,) = rows
    assert (row["ticker"], row["trade_date"], row["run_id"]) == ("NVDA", "2026-11-24", "r1")
    assert (row["portfolio_manager"]["rating"], row["shadow"]["rating"]) == ("Buy", "Sell")
    assert (row["portfolio_manager"]["position_size_pct"], row["shadow"]["position_size_pct"]) == (4.0, 2.0)
    assert row["same_rating"] is False and row["same_side"] is False


@pytest.mark.unit
def test_a_shadow_that_fails_costs_the_run_nothing_and_says_so_in_the_record(tmp_path):
    pm = _llm(_decision(PortfolioRating.OVERWEIGHT, 3.0), [])
    with run_config(_config(tmp_path)):
        out = create_portfolio_manager(pm, _llm(ConnectionError("PAIR is down"), []))(STATE)
        (row,) = shadow.read_day(next((tmp_path / "shadow").iterdir()).stem)
    assert out["final_rating"] == "Overweight"
    assert row["shadow"]["ok"] is False and "PAIR is down" in row["shadow"]["error"]
    assert row["portfolio_manager"]["ok"] is True and row["same_side"] is None       # nothing to compare


@pytest.mark.unit
def test_without_a_shadow_nothing_is_asked_twice_or_written(tmp_path):
    asked = []
    with run_config(_config(tmp_path)):
        create_portfolio_manager(_llm(_decision(PortfolioRating.HOLD, 0.0), asked))(STATE)
    assert len(asked) == 1 and not (tmp_path / "shadow").exists()


@pytest.mark.unit
def test_the_graph_gives_the_shadow_the_deep_tiers_model_and_only_when_switched_on(monkeypatch):
    from tradingagents.graph import setup

    made = {}
    monkeypatch.setattr(setup, "create_portfolio_manager", lambda llm, shadow_llm=None: made.update(pm=llm, shadow=shadow_llm) or (lambda s: {}))
    monkeypatch.setattr(setup, "for_agent", lambda llm, config, agent, tier: f"{tier}:{agent}")

    def build(**config):
        made.clear()
        graph = setup.GraphSetup.__new__(setup.GraphSetup)
        graph.quick_thinking_llm = graph.deep_thinking_llm = object()
        graph.config = {**DEFAULT_CONFIG, **config}
        graph.conditional_logic = MagicMock()
        graph.max_tool_rounds, graph.max_tool_output_chars, graph.max_tool_history_chars = 1, 1000, 1000
        with contextlib.suppress(Exception):     # the nodes are stand-ins; only what the Portfolio Manager was given matters
            graph.setup_graph(["market"])
        return dict(made)

    assert build(shadow_portfolio_manager=True) == {"pm": "deep:Portfolio Manager", "shadow": "deep:Shadow Portfolio Manager"}
    assert build() == {"pm": "deep:Portfolio Manager", "shadow": None}


@pytest.mark.unit
def test_the_summary_counts_agreement_by_rating_side_and_size():
    def row(a, b, sa=3.0, sb=3.0, ok=True):
        first = {"ok": True, "rating": a, "position_size_pct": sa}
        second = {"ok": ok, "rating": b, "position_size_pct": sb} if ok else {"ok": False, "error": "timeout"}
        both = ok
        return {"portfolio_manager": first, "shadow": second, "same_rating": both and a == b,
                "same_side": both and shadow.side(a) == shadow.side(b)}

    s = shadow.summarize([row("Buy", "Buy"), row("Buy", "Overweight", 4.0, 2.0), row("Sell", "Buy"),
                          row("Hold", "Underweight"), row("Buy", None, ok=False)])
    assert (s["symbols"], s["both_answered"], s["shadow_failed"]) == (5, 4, 1)
    assert (s["same_rating"], s["same_side"], s["opposite_sides"]) == (1, 2, 1)
    assert s["mean_size_difference_pct"] == 0.5
    assert {"portfolio_manager": "Sell", "shadow": "Buy", "symbols": 1} in s["ratings"]
