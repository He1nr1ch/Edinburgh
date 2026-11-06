"""The positions page: open positions joined to the desk's newest decision, each agent's text on its report."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from tradingagents.dataflows.vendors.alpaca.trading import Account, Holding
from tradingagents.web import data


def _save(results_dir, run, ticker, date, state):
    folder = (results_dir / "backtest" / run / ticker if run else results_dir / ticker) / "TradingAgentsStrategy_logs"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"full_states_log_{date}.json"
    path.write_text(json.dumps({"company_of_interest": ticker, **state}), encoding="utf-8")
    return path


COIN = {
    "final_rating": "Underweight", "final_risk_level": "Neutral", "final_position_size_pct": 3.0,
    "market_report": "## Trend\n\nBelow the 50 SMA.\n\n| Indicator | Value |\n|---|---|\n| RSI | 41 |",
    "investment_debate_state": {"bull_history": "Bull: adoption grows.", "bear_history": "Bear: volumes fall."},
    "investment_plan": "**Recommendation**: Underweight",
    "trader_investment_plan": "**Action**: Sell\n\n**Stop Loss**: 165",
    "risk_debate_state": {"aggressive_history": "Aggressive: press it.", "conservative_history": "", "neutral_history": "Neutral: small."},
    "risk_assessment": "**Risk Level**: Neutral\n\n**Maximum Position**: 3\n\n**Stop Loss**: 162.02",
    "final_trade_decision": "**Rating**: Underweight\n\n**Executive Summary**: Short it.\n\n**Price Target**: 120\n\n**Time Horizon**: 1-3 months",
}


@pytest.mark.unit
def test_the_newest_decision_per_ticker_wins(tmp_path):
    _save(tmp_path, "sp500_a", "COIN", "2026-10-27", COIN)
    newest = _save(tmp_path, "sp500_b", "COIN", "2026-11-04", COIN)
    single = _save(tmp_path, None, "NVDA", "2026-10-23", {"final_rating": "Hold"})

    found = data.find_decisions(tmp_path)
    assert found["COIN"].path == newest and found["COIN"].run == "sp500_b"
    assert found["NVDA"].path == single and found["NVDA"].run == "single"


@pytest.mark.unit
def test_the_summary_carries_the_sizing_rules_stop_and_the_expected_move():
    summary = data.decision_summary(COIN | {"company_of_interest": "COIN"}, close=151.50, default_stop_pct=2)
    assert (summary["rating"], summary["side"], summary["stop"], summary["stop_from"]) == ("Underweight", "short", 162.02, "Risk Manager")
    assert summary["target"] == 120 and summary["expected_move_pct"] == pytest.approx(20.79, abs=0.01)

    no_stop = data.decision_summary({"final_rating": "Overweight", "final_trade_decision": "**Price Target**: 110"}, 100.0, 2)
    assert (no_stop["stop"], no_stop["stop_from"]) == (98.0, "default 2%")


@pytest.mark.unit
def test_every_agent_that_ran_is_listed_in_desk_order():
    names = [name for name, _ in data.agent_outputs(COIN)]
    assert names == ["Market Analyst", "Bull Researcher", "Bear Researcher", "Research Manager", "Trader",
                     "Aggressive Risk Analyst", "Neutral Risk Analyst", "Risk Manager", "Portfolio Manager"]


@pytest.fixture
def client(tmp_path, monkeypatch):
    from tradingagents.web import app as web

    _save(tmp_path, "sp500_b", "COIN", "2026-11-04", COIN)
    (tmp_path / "backtest" / "sizing").mkdir(parents=True)
    (tmp_path / "backtest" / "sizing" / "book.csv").write_text(
        "ticker,side,rating,risk_level,ceiling_pct,weight_pct,price,shares,value,stop,stop_from,target\n"
        "COIN,short,Underweight,Neutral,3,-3.46,151.5,-22,3333,162.02,Risk Manager,120\n"
        "GD,long,Overweight,Neutral,5,5.77,364.87,16,5838,344.41,Risk Manager,480\n")
    monkeypatch.setattr(web, "_results_dir", lambda: tmp_path)
    monkeypatch.setattr(web, "_default_stop", lambda: 2.0)
    holdings = (Holding("COIN", -22, -3_200.0, 151.50, "us_equity"), Holding("AAPL", 10, 3_500.0, 340.0, "us_equity"))
    monkeypatch.setattr(web, "_holdings", lambda: Account(100_000, 60_000, True, holdings))
    return TestClient(web.app)


@pytest.mark.unit
def test_the_index_lists_positions_with_their_levels_and_a_link_to_the_agents(client):
    page = client.get("/").text
    assert "COIN" in page and "Underweight" in page and "162.02" in page and "120.00" in page
    assert '/report/COIN/2026-11-04' in page
    assert "AAPL" in page and ">none<" in page           # held, but the desk never decided on it
    assert "The day&#x27;s book" in page or "The day's book" in page
    assert "GD" in page and ">no<" in page               # in the book, not yet held


@pytest.mark.unit
def test_the_report_shows_each_agents_text_rendered(client):
    page = client.get("/report/COIN/2026-11-04").text
    assert "<h3>Market Analyst</h3>" in page and "<table>" in page and "<td>41</td>" in page
    assert "Bull: adoption grows." in page and "<h3>Portfolio Manager</h3>" in page
    assert "Conservative Risk Analyst" not in page      # did not speak
    assert client.get("/report/COIN").status_code == 200
    assert client.get("/report/XYZ").status_code == 404


@pytest.mark.unit
def test_the_json_routes_carry_the_same_data(client):
    positions = client.get("/api/positions").json()
    coin = next(p for p in positions["positions"] if p["ticker"] == "COIN")
    assert coin["decision"]["target"] == 120 and coin["decision_date"] == "2026-11-04"
    report = client.get("/api/report/COIN/2026-11-04").json()
    assert [a["agent"] for a in report["agents"]][0] == "Market Analyst"
    assert client.get("/health").json()["decisions"] == 1


@pytest.mark.unit
def test_a_models_raw_html_is_not_rendered(tmp_path, monkeypatch):
    from tradingagents.web import app as web

    _save(tmp_path, None, "X", "2026-11-04", {"final_rating": "Hold", "market_report": "<script>alert(1)</script> plain"})
    monkeypatch.setattr(web, "_results_dir", lambda: tmp_path)
    page = TestClient(web.app).get("/report/X/2026-11-04").text
    assert "<script>" not in page and "&lt;script&gt;" in page
