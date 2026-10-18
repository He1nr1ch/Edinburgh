"""An analyst's tool results are bounded, so its prompts fit the model's context window."""

import pandas as pd
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from tests.test_graph_end_to_end import TRADE_DATE, LoopingModel, _graph
from tradingagents.agents import context
from tradingagents.dataflows import router
from tradingagents.dataflows.vendors.alpaca import market as alpaca_market
from tradingagents.graph import setup


@pytest.mark.unit
def test_a_long_tool_result_is_cut_and_marked():
    long = ToolMessage(content="x" * 50_000, tool_call_id="1", name="get_stock_data")
    short = ToolMessage(content="ok", tool_call_id="2", name="get_news")
    plain = HumanMessage("NVDA")

    setup.clamp_tool_output([long, short, plain], limit=16_000)

    assert long.content.startswith("x" * 16_000) and "34,000 more characters withheld" in long.content
    assert short.content == "ok" and plain.content == "NVDA"


@pytest.mark.unit
def test_tool_history_counts_only_tool_results():
    messages = [HumanMessage("a" * 100), AIMessage("b" * 100), ToolMessage(content="c" * 300, tool_call_id="1"),
                ToolMessage(content="d" * 200, tool_call_id="2")]
    assert setup.tool_history_chars(messages) == 500


@pytest.fixture
def verbose_vendors(monkeypatch):
    """Every vendor answers with far more than one result may carry."""
    for _method, vendors in router.VENDOR_METHODS.items():
        for vendor in vendors:
            monkeypatch.setitem(vendors, vendor, lambda *a, **k: "9" * 30_000)
    prices = pd.DataFrame({"Date": pd.bdate_range(end=TRADE_DATE, periods=60), "Open": 100.0, "High": 101.0,
                           "Low": 99.0, "Close": 100.5, "Volume": 1_000_000})
    monkeypatch.setattr(alpaca_market, "load_ohlcv", lambda *a, **k: prices.copy())
    monkeypatch.setattr(context, "get_asset_profile", lambda ticker: {})
    context._identity.cache_clear()


@pytest.mark.unit
def test_an_analyst_past_its_tool_budget_writes_its_report(tmp_path, monkeypatch, verbose_vendors):
    """A model that keeps asking for data is stopped by the budget, not by the window:
    each result is cut to 16k characters, and after three (48k, past a 40k budget)
    the analyst is asked for its report, well short of its twenty rounds."""
    model = LoopingModel(structured=True)
    graph = _graph(tmp_path, monkeypatch, model, max_tool_rounds=20,
                   max_tool_output_chars=16_000, max_tool_history_chars=40_000)

    final_state, rating = graph.propagate("NVDA", TRADE_DATE)

    assert len(model.tool_turns) == 3 * 3          # three analysts, three rounds each
    assert rating == "Overweight"
    histories = [h for h in model.last_turns if isinstance(h[-1], HumanMessage) and "tool round" in h[-1].content]
    assert len(histories) == 3
    for history in histories:
        results = [str(m.content) for m in history if "returned]" in str(m.content)]
        assert results and all("more characters withheld" in r for r in results)
        assert max(len(r) for r in results) < 17_000
