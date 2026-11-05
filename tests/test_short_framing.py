"""The desk runs long and short: every agent that frames direction is told a Sell is a short.

Before this, the rating scale read "Sell: exit position or avoid entry", so the
Risk Manager's stop for a Sell was a long holder's exit level below the price
and the Portfolio Manager reused it as the target: across the S&P 500, 192 of
241 bearish calls had their stop or target on the wrong side for a short.
"""

from __future__ import annotations

import importlib
import inspect

import pytest

from tradingagents.agents.schemas import PortfolioDecision, ResearchPlan, RiskAssessment, TraderProposal


@pytest.mark.unit
@pytest.mark.parametrize("module", [
    "tradingagents.agents.managers.research_manager",
    "tradingagents.agents.managers.portfolio_manager",
    "tradingagents.agents.managers.risk_manager",
    "tradingagents.agents.trader.trader",
])
def test_each_deciding_agent_is_told_the_desk_runs_long_and_short(module):
    source = inspect.getsource(importlib.import_module(module))
    assert "long and short" in source
    assert "avoid entry" not in source and "avoiding the position" not in source


@pytest.mark.unit
def test_the_risk_managers_stop_is_placed_by_the_positions_side():
    source = inspect.getsource(importlib.import_module("tradingagents.agents.managers.risk_manager"))
    assert "its stop is above the current price" in source
    assert "above the price for a short and below it for a long" in source


@pytest.mark.unit
def test_the_schemas_name_a_sell_as_a_short():
    def desc(model, field):
        return model.model_fields[field].description or ""
    assert "short" in desc(ResearchPlan, "recommendation")
    assert "short sale" in desc(TraderProposal, "action")
    assert "above it for a Sell (a short)" in desc(TraderProposal, "stop_loss")
    assert "above the current price for a short" in desc(RiskAssessment, "stop_loss")
    assert "a short of this size" in desc(PortfolioDecision, "position_size_pct")
    assert "a short covers there" in inspect.getsource(importlib.import_module("tradingagents.agents.managers.portfolio_manager"))
