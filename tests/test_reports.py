"""The daily and weekly reports, and the lessons that flow back to the agents."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from unittest.mock import MagicMock

import pytest

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.lessons import Lessons, desk_lessons
from tradingagents.reports import daily, weekly
from tradingagents.worker.state import WorkerState


def _cfg(tmp_path, **more):
    return {**DEFAULT_CONFIG, "results_dir": str(tmp_path), "worker_state_path": str(tmp_path / "memory" / "state.json"),
            "sweep_state_path": str(tmp_path / "memory" / "sweep.json"), "lessons_path": str(tmp_path / "memory" / "lessons.json"),
            "feedback_db_url": "sqlite:///" + str(tmp_path / "memory" / "feedback.db"), **more}


def _day_data(tmp_path, cfg, monkeypatch, day="2026-11-09"):
    """A day as the desk records it: a snapshot, orders with reasons, a worker log, a book, the broker's fills."""
    from tradingagents.dataflows.vendors.alpaca import trading
    from tradingagents.web import snapshots
    from tradingagents.dataflows.vendors.alpaca.trading import Account, Holding

    monkeypatch.setattr(trading, "fills_on", lambda d: {"o3": {"filled_avg_price": 111.50, "filled_qty": 18, "status": "filled"}} if d == day else {})

    (tmp_path / "memory").mkdir(exist_ok=True)
    state = WorkerState.load(cfg["worker_state_path"])
    at = lambda h, m: datetime(2026, 11, 9, h, m, tzinfo=UTC)  # noqa: E731
    state.record_order("GD", "open", "buy", 16, "o1", at(14, 30), "the day's book (b.csv)")
    state.record_order("TROW", "open", "sell", 18, "o2", at(14, 30), "the day's book (b.csv)")
    state.record_order("TROW", "exit", "buy", 18, "o3", at(15, 50), "price 111.41 is through the stop 111.36 (Risk Manager); -2.1% on the position", entry_price=109.10)
    state.record_order("ACGL", "replace", "buy", 18, "o4", at(15, 56), "replaces TROW: best score on the bench (score 6.5, +17.5% to target)")
    state.day(day).update(build="1.2026.11.09.22 (abc1234)", book_path=str(tmp_path / "b.csv"))
    state.save()
    (tmp_path / "b.csv").write_text("ticker,side,rating,risk_level,ceiling_pct,weight_pct,price,shares,value,stop,stop_from,target,score\n"
                                    "GD,long,Overweight,Neutral,5,5.77,364.87,16,5838,344.41,Risk Manager,480,5.2\n"
                                    "TROW,short,Underweight,Neutral,2,-2.31,109.1,-18,1964,111.36,Risk Manager,100,3.1\n")
    folder = tmp_path / "backtest" / "r" / "GD" / "TradingAgentsStrategy_logs"
    folder.mkdir(parents=True)
    (folder / f"full_states_log_{day}.json").write_text(json.dumps({
        "company_of_interest": "GD", "final_rating": "Overweight", "final_risk_level": "Neutral",
        "risk_assessment": "**Maximum Position**: 5\n\n**Stop Loss**: 344.41\n", "final_trade_decision": "**Price Target**: 480\n"}))
    account = Account(99_800, 60_000, True, (Holding("GD", 16, 16 * 370.0, 364.87, "us_equity"), Holding("ACGL", 18, 18 * 107.0, 106.52, "us_equity")))
    snapshots.write_snapshot(tmp_path, account, 2.0, datetime(2026, 11, 9, 16, 5))
    snapshots.write_snapshot(tmp_path, Account(100_000, 100_000, True, ()), 2.0, datetime(2026, 11, 6, 16, 5))
    (tmp_path / "worker").mkdir()
    (tmp_path / "worker" / f"{day}.log").write_text("15:50:59   [urgent] TROW short TROW: stop hit at 111.36 (Risk Manager) -> exit buy 18 [sent, order o3]\n"
                                                    "15:56:00   [urgent] ACGL long ACGL replaces TROW -> replace buy 18 [sent, order o4]\n")
    (tmp_path / "memory" / "sweep.json").write_text(json.dumps({"cycle": 1, "date": day, "queue": ["A", "B"], "analysed_this_cycle": 40}))
    return state


def _fake_llm(result):
    structured = MagicMock()
    structured.invoke.side_effect = lambda prompt: (setattr(structured, "prompt", prompt) or result)
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm, structured


@pytest.mark.unit
def test_the_lessons_store_keeps_daily_lessons_thirty_days_and_the_standing_set_small(tmp_path):
    store = Lessons(tmp_path / "lessons.json")
    store.add_daily("2026-10-01", [{"agent": "Risk Manager", "lesson": "old", "evidence": "x"}])
    store.add_daily("2026-11-09", [{"agent": "Risk Manager", "lesson": "A 2% default stop on a short of a volatile name is noise; use 1 ATR.", "evidence": "LEN, TROW stopped within a day"},
                                   {"agent": "Watcher", "lesson": "A closed short is best replaced by a short.", "evidence": "TROW → ACGL"}])
    store.add_daily("2026-11-09", [{"agent": "Watcher", "lesson": "A closed short is best replaced by a short.", "evidence": "TROW → ACGL"}])   # a rerun replaces the day
    again = Lessons(tmp_path / "lessons.json")
    assert [l["agent"] for l in again.daily] == ["Watcher"] and again.daily[0]["date"] == "2026-11-09"          # old pruned, rerun idempotent
    again.set_standing("2026-11-13", [{"agent": "Desk", "lesson": f"Rule {i}.", "evidence": "x"} for i in range(20)]
                       + [{"agent": "Desk", "lesson": "Rule 0.", "evidence": "a repeat"}])
    assert len(Lessons(tmp_path / "lessons.json").standing) == 12                                  # capped, repeats dropped
    again.set_standing("2026-11-13", [{"agent": "Desk", "lesson": "Default stops at 2% fire on noise; widen to 1 ATR.", "evidence": "4 of 6 stops hit within a day"}])
    text = again.render_for("Watcher", as_of="2026-11-10")
    assert "for the Watcher and the desk" in text and "closed short is best replaced" in text and "Default stops at 2%" in text and "standing" in text
    assert again.render_for("Market Analyst", as_of="2026-11-10").count("\n- ") == 1        # only the Desk lesson applies
    assert Lessons(tmp_path / "none.json").render_for("Trader") == ""


@pytest.mark.unit
def test_the_deciding_agents_and_the_watcher_read_their_lessons(tmp_path, monkeypatch):
    from tradingagents.dataflows.config import run_config

    store = Lessons(tmp_path / "lessons.json")
    store.add_daily(date.today().isoformat(), [{"agent": "Risk Manager", "lesson": "Place the stop beyond one ATR, never inside 2%.", "evidence": "LEN"},
                                               {"agent": "Watcher", "lesson": "Replace a closed short with a short.", "evidence": "TROW"}])
    with run_config({"lessons_path": str(tmp_path / "lessons.json")}):
        assert "beyond one ATR" in desk_lessons("Risk Manager")
        assert desk_lessons("Research Manager") == ""

        from tradingagents.agents.managers.risk_manager import create_risk_manager
        from tradingagents.agents.schemas import RiskAssessment, RiskLevel

        llm, structured = _fake_llm(RiskAssessment(risk_level=RiskLevel.NEUTRAL, max_position_pct=5, rationale="r"))
        create_risk_manager(llm)({"company_of_interest": "GD", "trader_investment_plan": "**Action**: Buy",
                                  "risk_debate_state": {"history": "h", "aggressive_history": "", "conservative_history": "", "neutral_history": "", "count": 1}})
        assert "beyond one ATR" in structured.prompt

        from tradingagents.worker import news
        from tradingagents.dataflows.vendors.alpaca.trading import Holding
        prompt = news._prompt(Holding("TROW", -18, -2000.0, 109.1, "us_equity"), [{"created_at": datetime(2026, 11, 9, 15, 0, tzinfo=UTC), "source": "s", "headline": "h", "summary": ""}])
        assert "Replace a closed short with a short" in prompt


@pytest.mark.unit
def test_the_daily_report_gathers_the_day_and_the_review_teaches(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    _day_data(tmp_path, cfg, monkeypatch)
    facts = daily.gather(cfg, "2026-11-09")
    assert facts["equity"] == 99_800 and facts["equity_change"] == -200 and facts["realised"] == -43.2
    assert facts["exits_by_cause"] == {"stop": 1} and facts["entries"] == 3 and facts["book_held"] == 1
    assert len(facts["findings"]) == 2 and facts["sweep"]["queued"] == 2 and facts["sweep"]["decided_today"] == 1
    assert facts["build"].startswith("1.2026.11.09.22")

    rev = daily.DailyReview(what_happened="TROW stopped out for -43.20.", what_worked="The stop guard fired within 5 s of the breach.",
                            what_did_not="TROW's stop sat 2.1% above entry, inside one day's range.",
                            lessons=[daily.Lesson(agent="Risk Manager", lesson="Place a short's stop beyond one ATR above entry.", evidence="TROW stop 111.36 hit at 10:50")],
                            rule_suggestions=["Raise default_stop_pct to 3% for names with ATR over 2% (LEN, TROW)."])
    llm, structured = _fake_llm(rev)
    path, got = daily.run(cfg, "2026-11-09", llm)
    assert "Today's record:" in structured.prompt and "TROW" in structured.prompt and "Feedback scorecard" not in structured.prompt   # nothing settled yet
    text = path.read_text()
    assert text.startswith("# Edinburgh daily report, 2026-11-09") and "**Risk Manager**: Place a short's stop" in text
    assert "| 15:50 | Exited | TROW |" in text and "-43.20" in text and "## Rule suggestions" in text
    assert Lessons(cfg["lessons_path"]).for_agent("Risk Manager", as_of="2026-11-10")[0]["lesson"].startswith("Place a short's stop")
    assert json.loads((tmp_path / "reports" / "daily" / "2026-11-09.json").read_text())["review"]["lessons"][0]["agent"] == "Risk Manager"


@pytest.mark.unit
def test_the_weekly_report_aggregates_the_days_and_sets_the_standing_lessons(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    _day_data(tmp_path, cfg, monkeypatch)
    llm, _ = _fake_llm(daily.DailyReview(what_happened="h", what_worked="w", what_did_not="n",
                                         lessons=[daily.Lesson(agent="Watcher", lesson="Replace a short with a short.", evidence="TROW")], rule_suggestions=[]))
    daily.run(cfg, "2026-11-09", llm)
    w = weekly.gather(cfg, "2026-11-13")
    assert [d["day"] for d in w["days"]] == ["2026-11-09"] and w["realised_week"] == -43.2
    assert w["by_cause"] == {"stop": {"n": 1, "pnl": -43.2}} and w["by_side"]["short"]["wins"] == 0
    assert w["daily_lessons"][0]["agent"] == "Watcher" and w["equity_start"] == 100_000 and w["equity_end"] == 99_800

    rev = weekly.WeeklyReview(summary="s", by_agent="Risk Manager: stops too tight.",
                              standing_lessons=[daily.Lesson(agent="Desk", lesson="Stops inside one day's range fire on noise.", evidence="2 of 2 stops hit within hours")],
                              rule_suggestions=[])
    wllm, structured = _fake_llm(rev)
    path, got = weekly.run(cfg, "2026-11-13", wllm)
    assert "The week's daily lessons:" in structured.prompt and "Replace a short with a short" in structured.prompt
    assert "## Standing lessons (carried forward)" in path.read_text() and "| stop | 1 | -43.20 |" in path.read_text()
    standing = Lessons(cfg["lessons_path"]).standing
    assert len(standing) == 1 and standing[0]["since"] == "2026-11-13"
    from tradingagents.dataflows.config import run_config
    with run_config({"lessons_path": cfg["lessons_path"]}):
        assert "Stops inside one day" in desk_lessons("Trader")          # a Desk lesson reaches every deciding agent


@pytest.mark.unit
def test_the_week_ends_when_the_next_open_is_in_a_later_week():
    from zoneinfo import ZoneInfo

    et = ZoneInfo("America/New_York")
    assert daily.is_week_end("2026-11-13", datetime(2026, 11, 16, 9, 30, tzinfo=et))        # Friday → Monday
    assert not daily.is_week_end("2026-11-12", datetime(2026, 11, 13, 9, 30, tzinfo=et))
    assert daily.is_week_end("2026-11-25", datetime(2026, 11, 30, 9, 30, tzinfo=et))        # Wednesday before a long holiday
    assert daily.week_days("2026-11-11") == ["2026-11-09", "2026-11-10", "2026-11-11"]


@pytest.mark.unit
def test_the_reports_pages(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from tradingagents.web import app as web

    cfg = _cfg(tmp_path)
    (tmp_path / "reports" / "daily").mkdir(parents=True)
    (tmp_path / "reports" / "daily" / "2026-11-09.md").write_text("# Edinburgh daily report, 2026-11-09\n\n## Lessons\n- **Risk Manager**: wider stops\n")
    Lessons(cfg["lessons_path"]).set_standing("2026-11-06", [{"agent": "Desk", "lesson": "Stops inside noise fire.", "evidence": "x"}])
    monkeypatch.setattr(web, "get_config", lambda: cfg)
    client = TestClient(web.app)
    index = client.get("/reports").text
    assert '/reports/2026-11-09' in index and "Stops inside noise fire." in index
    page = client.get("/reports/2026-11-09").text
    assert "<h1>Edinburgh daily report, 2026-11-09</h1>" in page and "wider stops" in page
    assert client.get("/reports/2026-01-01").status_code == 404
    assert client.get("/reports/week/2026-11-13").status_code == 404


@pytest.mark.unit
@pytest.mark.usefixtures("free_text_contract")
def test_the_analysts_and_the_researchers_read_their_lessons(tmp_path):
    """A lesson addressed to an analyst or a researcher reaches that agent's prompt, and no other's."""
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.runnables import RunnableLambda

    from tradingagents.agents.analysts.fundamentals_analyst import create_fundamentals_analyst
    from tradingagents.agents.analysts.market_analyst import create_market_analyst
    from tradingagents.agents.analysts.news_analyst import create_news_analyst
    from tradingagents.agents.researchers.bear_researcher import create_bear_researcher
    from tradingagents.agents.researchers.bull_researcher import create_bull_researcher
    from tradingagents.dataflows.config import run_config

    marks = {"Market Analyst": "MARKET-MARK {braces} stay literal", "News Analyst": "NEWS-MARK", "Fundamentals Analyst": "FUND-MARK",
             "Bull Researcher": "BULL-MARK", "Bear Researcher": "BEAR-MARK"}
    store = Lessons(tmp_path / "lessons.json")
    store.add_daily(date.today().isoformat(), [{"agent": a, "lesson": m, "evidence": "x"} for a, m in marks.items()])
    store.set_standing(date.today().isoformat(), [{"agent": "Desk", "lesson": "DESK-MARK", "evidence": "x"}])

    def analyst_prompt(factory) -> str:
        seen = {}

        class Llm:
            def bind_tools(self, tools):
                return RunnableLambda(lambda value: seen.__setitem__("text", value.to_string()) or AIMessage(content="report"))

        factory(Llm())({"trade_date": "2026-11-10", "company_of_interest": "GD", "asset_type": "stock",
                        "messages": [HumanMessage(content="GD")]})
        return seen["text"]

    def researcher_prompt(factory) -> str:
        seen = {}
        llm = MagicMock()
        llm.invoke.side_effect = lambda prompt: seen.__setitem__("text", prompt) or MagicMock(content="argument")
        factory(llm)({"company_of_interest": "GD", "asset_type": "stock", "market_report": "m", "news_report": "n",
                      "fundamentals_report": "f",
                      "investment_debate_state": {"history": "", "bull_history": "", "bear_history": "", "current_response": "", "count": 0}})
        return seen["text"]

    with run_config({"lessons_path": str(tmp_path / "lessons.json")}):
        prompts = {"Market Analyst": analyst_prompt(create_market_analyst), "News Analyst": analyst_prompt(create_news_analyst),
                   "Fundamentals Analyst": analyst_prompt(create_fundamentals_analyst),
                   "Bull Researcher": researcher_prompt(create_bull_researcher), "Bear Researcher": researcher_prompt(create_bear_researcher)}
    for agent, text in prompts.items():
        assert marks[agent] in text and "DESK-MARK" in text, agent
        assert not any(m in text for a, m in marks.items() if a != agent), agent

    with run_config({"lessons_path": str(tmp_path / "none.json")}):      # nothing learned yet: the prompt carries no block
        assert "Lessons from the desk" not in researcher_prompt(create_bull_researcher)
        assert "Lessons from the desk" not in analyst_prompt(create_market_analyst)
