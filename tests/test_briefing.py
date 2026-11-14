"""The analysts' briefing: their data fetched for them and handed over in one message."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from tradingagents.agents.analysts import briefing
from tradingagents.dataflows.config import run_config
from tradingagents.default_config import DEFAULT_CONFIG


@pytest.fixture
def routed(tmp_path, monkeypatch):
    """Every vendor call recorded and answered; the config has briefings on and a cache folder of its own."""
    calls = []

    def route(method, *args):
        calls.append((method, *args))
        if method == "get_cashflow":
            raise RuntimeError("EDGAR is down")
        return f"<{method} {' '.join(str(a) for a in args)}>"

    monkeypatch.setattr(briefing, "route_to_vendor", route)
    with run_config({**DEFAULT_CONFIG, "analyst_briefing": True, "data_cache_dir": str(tmp_path)}):
        yield calls


@pytest.mark.unit
def test_the_market_analyst_is_handed_its_prices_its_snapshot_and_every_indicator(routed):
    text = briefing.market("FDX", "2026-11-13")

    assert routed[0] == ("get_stock_data", "FDX", "2026-07-16", "2026-11-13")           # 120 days back, never past the run's date
    assert routed[1] == ("get_market_snapshot", "FDX", "2026-11-13", 30)
    assert [c[2] for c in routed[2:]] == list(briefing.INDICATORS) and all(c[3] == "2026-11-13" for c in routed[2:])
    assert "[get_verified_market_snapshot returned]\n<get_market_snapshot FDX 2026-11-13 30>" in text
    assert "[get_indicators(rsi, 30 days) returned]" in text
    assert text.rstrip().endswith("than fill the gap.") and "No tool is available in this turn" in text


@pytest.mark.unit
def test_what_every_ticker_shares_is_fetched_once_a_day(routed):
    """The market-wide news and the macro series are the same for all 503 tickers of a sweep."""
    first = briefing.news("FDX", "2026-11-13")
    second = briefing.news("PSX", "2026-11-13")

    assert [c[0] for c in routed].count("get_global_news") == 1
    assert [c[0] for c in routed].count("get_macro_indicators") == len(briefing.MACRO)
    assert [c for c in routed if c[0] == "get_news"] == [("get_news", "FDX", "2026-11-06", "2026-11-13"),
                                                           ("get_news", "PSX", "2026-11-06", "2026-11-13")]
    assert "<get_global_news 2026-11-13 None None>" in first and "<get_global_news 2026-11-13 None None>" in second
    briefing.news("FDX", "2026-11-16")                                                    # another day: asked again
    assert [c[0] for c in routed].count("get_global_news") == 2


@pytest.mark.unit
def test_a_section_that_cannot_be_fetched_is_said_so_and_the_rest_stands(routed, monkeypatch):
    text = briefing.fundamentals("FDX", "2026-11-13")
    assert len(routed) == 6                                                              # three statements, quarterly and annual
    assert "[get_cashflow(quarterly) returned]\nCould not be retrieved (RuntimeError: EDGAR is down)." in text
    assert "<get_income_statement FDX annual 2026-11-13>" in text

    def down(*a):
        raise RuntimeError("no network")
    monkeypatch.setattr(briefing, "route_to_vendor", down)
    assert briefing.fundamentals("FDX", "2026-11-13") is None                             # nothing at all: the tools remain


@pytest.mark.unit
def test_a_briefing_is_built_only_for_an_analyst_that_has_not_begun_with_its_tools(routed):
    state = {"company_of_interest": "FDX", "trade_date": "2026-11-13", "messages": [HumanMessage("FDX")]}
    assert briefing.brief(briefing.fundamentals, state)
    resumed = {**state, "messages": [HumanMessage("FDX"), AIMessage("", tool_calls=[{"name": "get_cashflow", "args": {}, "id": "1"}]),
                                     ToolMessage("...", tool_call_id="1")]}
    assert briefing.brief(briefing.fundamentals, resumed) is None
    with run_config({**DEFAULT_CONFIG, "analyst_briefing": False}):
        assert briefing.brief(briefing.fundamentals, state) is None


@pytest.mark.unit
def test_a_briefed_analyst_writes_its_report_in_one_call_and_is_offered_no_tools():
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.runnables import RunnableLambda

    from tradingagents.agents.analysts.turn import take_turn

    seen = []

    class Llm(RunnableLambda):
        def bind_tools(self, tools):
            raise AssertionError("a briefed analyst is offered no tools")

    llm = Llm(lambda value: seen.append(value.to_messages()) or AIMessage("The report."))
    prompt = ChatPromptTemplate.from_messages([("system", "Tools: {tool_names}."), MessagesPlaceholder("messages")])

    result, report = take_turn(prompt, llm, (), [HumanMessage("FDX")], "[get_stock_data returned]\n...")
    assert report == "The report." and not result.tool_calls and len(seen) == 1
    system, asked, handed = seen[0]
    assert "none in this turn" in system.content and asked.content == "FDX" and handed.content.startswith("[get_stock_data returned]")
