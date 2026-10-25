"""report_style: a concise run gives each output a length; a detailed one does not."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tradingagents.agents.context import length_brief, report_style
from tradingagents.dataflows.config import run_config


def _trader_prompt(style: str) -> str:
    from tradingagents.agents.schemas import TraderAction, TraderProposal
    from tradingagents.agents.trader.trader import create_trader

    captured = {}
    structured = MagicMock()
    structured.invoke.side_effect = lambda prompt: (
        captured.__setitem__("prompt", prompt) or TraderProposal(action=TraderAction.BUY, reasoning="x")
    )
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    with run_config({"report_style": style}):
        create_trader(llm)({
            "company_of_interest": "NVDA",
            "investment_plan": "**Recommendation**: Buy",
            "market_report": "Current price $189.5; ATR 4.2.",
        })
    return "\n".join(m["content"] for m in captured["prompt"])


def _bull_prompt(style: str) -> str:
    from tradingagents.agents.researchers.bull_researcher import create_bull_researcher

    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="growth")
    with run_config({"report_style": style}):
        create_bull_researcher(llm)({
            "company_of_interest": "NVDA",
            "market_report": "m", "news_report": "", "fundamentals_report": "",
            "investment_debate_state": {
                "history": "", "bull_history": "", "bear_history": "",
                "current_response": "", "count": 0,
            },
        })
    return llm.invoke.call_args.args[0]


@pytest.mark.unit
def test_a_run_is_concise_unless_it_asks_for_detail():
    assert report_style() == "concise"
    with run_config({"report_style": "Detailed"}):
        assert report_style() == "detailed"
        assert length_brief("report", before=" ") == ""


@pytest.mark.unit
def test_a_misspelled_style_is_refused_not_guessed():
    with run_config({"report_style": "brief"}), pytest.raises(ValueError, match="report_style"):
        report_style()


@pytest.mark.unit
@pytest.mark.parametrize("kind", ["report", "argument", "decision"])
def test_each_kind_of_output_has_a_concise_brief(kind):
    with run_config({"report_style": "concise"}):
        assert length_brief(kind)
        assert length_brief(kind, before=" ").startswith(" ")


@pytest.mark.unit
def test_the_brief_reaches_the_prompt_only_on_a_concise_run():
    with run_config({"report_style": "concise"}):
        decision, argument = length_brief("decision"), length_brief("argument")
    assert decision in _trader_prompt("concise")
    assert decision not in _trader_prompt("detailed")
    assert argument in _bull_prompt("concise")
    assert argument not in _bull_prompt("detailed")


@pytest.mark.unit
def test_a_detailed_run_keeps_the_original_wording():
    """``detailed`` is the prompts as they were, so a run can be compared against them."""
    import inspect

    import tradingagents.agents.analysts.fundamentals_analyst as fa
    import tradingagents.agents.analysts.market_analyst as ma

    assert "Write a very detailed and nuanced report of the trends you observe. " in inspect.getsource(ma)
    assert "Make sure to include as much detail as possible." in inspect.getsource(fa)
    assert _bull_prompt("detailed").endswith("strengths of the bull position.\n")
