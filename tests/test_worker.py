"""The daily worker: news judged for the holder, orders by fixed rules, guardrails on every order."""

from datetime import UTC, datetime

import pytest

from tradingagents.dataflows.vendors.alpaca.trading import Account, Holding
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.worker.duty import Action, WorkerContext
from tradingagents.worker.policy import action_for, refusal
from tradingagents.worker.state import WorkerState

CONFIG = {**DEFAULT_CONFIG, "worker_trim_fraction": 0.5, "worker_add_fraction": 0.25,
          "worker_position_cap_pct": 10.0, "worker_trading_enabled": True, "worker_allow_live": False}
LONG = Holding("JPM", 40, 13_263.2, 300.0, "us_equity")
SHORT = Holding("T", -216, -4_520.9, 21.0, "us_equity")


def _act(h, materiality, direction, surprise=True, equity=100_000):
    return action_for(h, materiality, direction, surprise, equity, CONFIG, "why")


@pytest.mark.unit
def test_adverse_unexpected_news_exits_when_urgent_and_trims_when_notable():
    assert _act(LONG, "high", "adverse") == Action("JPM", "exit", "sell", 40, "why")
    assert _act(LONG, "medium", "adverse") == Action("JPM", "trim", "sell", 20, "why")
    # For a short, closing is buying back.
    assert _act(SHORT, "high", "adverse") == Action("T", "exit", "buy", 216, "why")
    assert _act(SHORT, "medium", "adverse").side == "buy"


@pytest.mark.unit
def test_favorable_urgent_news_adds_up_to_the_cap():
    small = Holding("AMZN", 6, 1_488.0, 248.0, "us_equity")
    assert _act(small, "high", "favorable") == Action("AMZN", "add", "buy", 1, "why")      # 25% of 6
    # JPM is 13.3% of 100k already: past the 10% cap, so no add.
    assert _act(LONG, "high", "favorable") is None
    # A short adds by selling more.
    assert _act(SHORT, "high", "favorable") == Action("T", "add", "sell", 54, "why")


@pytest.mark.unit
@pytest.mark.parametrize("materiality, direction, surprise", [
    ("high", "adverse", False),       # expected: an alert, not an order
    ("high", "mixed", True),
    ("low", "adverse", True),
    ("medium", "favorable", True),    # good but not urgent: alert only
])
def test_expected_mixed_minor_or_merely_good_news_only_alerts(materiality, direction, surprise):
    assert _act(LONG, materiality, direction, surprise) is None


@pytest.mark.unit
def test_the_guardrails():
    a = Action("JPM", "exit", "sell", 40, "why")
    ok = dict(market_open=True, paper=True, pending=set(), config=CONFIG)
    assert refusal(a, **ok) is None
    assert "closed" in refusal(a, **{**ok, "market_open": False})
    assert "live" in refusal(a, **{**ok, "paper": False})
    assert refusal(a, **{**ok, "paper": False, "config": {**CONFIG, "worker_allow_live": True}}) is None
    assert "turned off" in refusal(a, **{**ok, "config": {**CONFIG, "worker_trading_enabled": False}})
    assert "not filled" in refusal(a, **{**ok, "pending": {"JPM"}})
    assert refusal(a, **{**ok, "pending": {"T"}}) is None     # another symbol's open order does not block it


@pytest.mark.unit
def test_there_is_no_daily_limit_on_orders():
    """A symbol can trade again the same day, and any number of symbols can, once filled."""
    a = Action("JPM", "trim", "sell", 20, "why")
    import inspect

    import tradingagents.worker.policy as policy
    assert "orders_today" not in inspect.signature(policy.refusal).parameters
    assert "worker_max_orders_per_day" not in DEFAULT_CONFIG
    assert refusal(a, market_open=True, paper=True, pending=set(), config=CONFIG) is None


@pytest.mark.unit
def test_only_equity_positions_are_watched_and_the_list_can_be_narrowed():
    call = Holding("SPY261120C00760000", 1, 1_464.0, 14.64, "us_option")
    account = Account(100_000, 90_000, True, (LONG, SHORT, call))
    ctx = WorkerContext(account, datetime.now(UTC), CONFIG, None)
    assert [h.symbol for h in ctx.watched()] == ["JPM", "T"]
    ctx = WorkerContext(account, datetime.now(UTC), {**CONFIG, "worker_symbols": "t"}, None)
    assert [h.symbol for h in ctx.watched()] == ["T"]


@pytest.mark.unit
def test_the_state_remembers_articles_and_the_days_orders(tmp_path):
    state = WorkerState.load(tmp_path / "state.json")
    state.mark_seen("JPM", ["a1", "a2"])
    state.record_order("JPM", "trim", "sell", 20, "ord-1", datetime(2026, 10, 26, 10, 5, tzinfo=UTC))
    state.save()

    again = WorkerState.load(tmp_path / "state.json")
    assert not again.is_new("JPM", "a1") and again.is_new("JPM", "a3")
    assert [o["symbol"] for o in again.orders_on("2026-10-26")] == ["JPM"]
    assert again.orders_on("2026-10-27") == []


@pytest.mark.unit
def test_a_close_cancels_the_symbols_waiting_orders_and_only_the_workers_own_orders_hold_it_back(monkeypatch, tmp_path):
    """A bracket's take-profit left after a close would later open the position the other way."""
    from tradingagents.worker import runner
    from tradingagents.worker.duty import Finding

    sent, cancelled = [], []
    monkeypatch.setattr(runner.trading, "submit_market_order", lambda s, q, side: sent.append((s, q, side)) or "ord-9")
    monkeypatch.setattr(runner.trading, "cancel_orders_for",
                        lambda s, open_now: cancelled.append(s) or sum(v == s for v in open_now.values()))
    state = WorkerState.load(tmp_path / "s.json")
    ctx = WorkerContext(Account(100_000, 0, True, (LONG,)), datetime.now(UTC), CONFIG, state)
    open_now = {"bracket-tp": "JPM", "bracket-sl": "JPM", "other": "T"}

    f = Finding("news", "JPM", "urgent", "h", "d", Action("JPM", "exit", "sell", 40, "why"))
    runner._act(f, ctx, market_open=True, pending=set(), open_now=open_now, dry_run=False)
    assert cancelled == ["JPM"] and sent == [("JPM", 40, "sell")]
    assert "cancelled 2 waiting order(s)" in f.outcome

    add = Finding("news", "T", "urgent", "h", "d", Action("T", "add", "sell", 5, "why"))
    runner._act(add, ctx, market_open=True, pending=set(), open_now=open_now, dry_run=False)
    assert cancelled == ["JPM"]          # an add leaves the waiting orders alone


@pytest.mark.unit
def test_the_forever_loop_sleeps_through_the_close_and_survives_errors(monkeypatch):
    """Closed: nap five minutes at a time so the sweep's rounds keep turning. An error: log it, wait a minute, carry on."""
    from datetime import timedelta

    from tradingagents.dataflows.vendors.alpaca.trading import Clock
    from tradingagents.worker import runner

    now = datetime(2026, 10, 25, 20, 0, tzinfo=UTC)
    clocks = iter([Clock(False, now, now + timedelta(hours=17, minutes=30), now), RuntimeError("down")])

    def clock():
        c = next(clocks)
        if isinstance(c, Exception):
            raise c
        return c

    slept = []

    class Stop(Exception):
        pass

    def sleep(seconds):
        slept.append(seconds)
        if len(slept) == 2:
            raise Stop

    monkeypatch.setattr(runner.trading, "get_clock", clock)
    monkeypatch.setattr(runner.time, "sleep", sleep)
    with pytest.raises(Stop):
        runner.run_forever(CONFIG, None, [], 5)
    assert slept == [300, 60]          # a five-minute nap (the sweep keeps turning), then the retry wait


def _decision(results_dir, ticker, date, **state):
    folder = results_dir / "backtest" / "run" / ticker / "TradingAgentsStrategy_logs"
    folder.mkdir(parents=True, exist_ok=True)
    import json as _json
    (folder / f"full_states_log_{date}.json").write_text(_json.dumps({"company_of_interest": ticker, **state}))


@pytest.mark.unit
def test_the_stop_guard_closes_a_position_through_its_stop_and_leaves_the_rest(tmp_path):
    from tradingagents.worker.stops import StopDuty, breached, standing_stop

    _decision(tmp_path, "JPM", "2026-11-04", final_rating="Overweight",
              risk_assessment="**Maximum Position**: 5\n\n**Stop Loss**: 341.64\n", final_trade_decision="**Price Target**: 385\n")
    _decision(tmp_path, "T", "2026-11-04", final_rating="Underweight",
              risk_assessment="**Maximum Position**: 2\n\n**Stop Loss**: 88.09\n", final_trade_decision="**Price Target**: 75\n")
    cfg = {**CONFIG, "results_dir": str(tmp_path), "default_stop_pct": 2.0}
    jpm_fine = Holding("JPM", 10, 3_500.0, 352.11, "us_equity")          # 350, above the 341.64 stop
    jpm_hit = Holding("JPM", 10, 3_400.0, 352.11, "us_equity")           # 340, through it
    t_hit = Holding("T", -100, -8_900.0, 83.25, "us_equity")             # short at 83.25, price 89 > 88.09
    orphan = Holding("ZZZ", 50, 4_800.0, 100.0, "us_equity")             # no decision: 2% from entry is 98; price 96
    ctx = WorkerContext(Account(100_000, 0, True, (jpm_fine, jpm_hit, t_hit, orphan)), datetime.now(UTC), cfg, None)

    from tradingagents.web import data
    decisions = data.find_decisions(tmp_path)
    assert standing_stop(jpm_fine, decisions, 2.0) == (341.64, "Risk Manager")
    assert standing_stop(orphan, decisions, 2.0) == (98.0, "default 2%")
    assert not breached(jpm_fine, 341.64) and breached(jpm_hit, 341.64) and breached(t_hit, 88.09)

    findings = StopDuty().run(ctx)
    assert [(f.symbol, f.action.kind, f.action.side, f.action.qty) for f in findings] == [
        ("JPM", "exit", "sell", 10), ("T", "exit", "buy", 100), ("ZZZ", "exit", "sell", 50)]
    assert all(f.severity == "urgent" for f in findings)
    assert "through the stop 341.64 (Risk Manager)" in findings[0].detail


@pytest.mark.unit
def test_the_watcher_judges_news_against_the_desks_case(tmp_path, monkeypatch):
    from unittest.mock import MagicMock

    from tradingagents.worker import news

    _decision(tmp_path, "JPM", "2026-11-04", final_rating="Overweight",
              risk_assessment="**Stop Loss**: 341.64\n",
              final_trade_decision="**Rating**: Overweight\n\n**Executive Summary**: Buy the dip.\n\n"
                                   "**Investment Thesis**: Net interest income grows; the known risk is the Basel capital rule.\n\n"
                                   "**Price Target**: 385\n\n**Time Horizon**: 1-3 months",
              news_report="## JPM News\n\n### Basel III endgame rule lands (Reuters)")
    cfg = {**CONFIG, "results_dir": str(tmp_path), "worker_news_lookback_hours": 18}
    article = {"id": "a1", "headline": "JPMorgan CFO resigns abruptly", "summary": "", "source": "WSJ",
               "created_at": datetime(2026, 11, 6, 15, 0, tzinfo=UTC), "url": "u"}
    monkeypatch.setattr(news, "articles_since", lambda s, since, limit=20: [article])
    monkeypatch.setattr(news, "_name", lambda s: "JPMorgan Chase")
    captured = {}
    structured = MagicMock()
    structured.invoke.side_effect = lambda prompt: captured.__setitem__("prompt", prompt) or news.NewsAssessment(
        materiality="high", direction="adverse", accounted_for=False, summary="CFO out.", rationale="Not in the thesis.")
    llm = MagicMock(); llm.with_structured_output.return_value = structured
    state = WorkerState.load(tmp_path / "state.json")
    ctx = WorkerContext(Account(100_000, 0, True, (LONG,)), datetime(2026, 11, 6, 15, 5, tzinfo=UTC), cfg, state, llm)

    (finding,) = news.WatcherDuty().run(ctx)
    prompt = captured["prompt"]
    assert "The desk's case, decided on 2026-11-04 (Overweight, target 385.00, stop 341.64, 1-3 months)" in prompt
    assert "the known risk is the Basel capital rule" in prompt and "Basel III endgame rule lands" in prompt
    assert "CFO resigns" in prompt
    assert finding.severity == "urgent" and finding.action.kind == "exit" and finding.action.qty == 40
    assert "not accounted for" in finding.headline

    structured.invoke.side_effect = lambda prompt: news.NewsAssessment(
        materiality="high", direction="adverse", accounted_for=True, summary="Basel rule.", rationale="Known.")
    state.last_check.clear(); state.seen.clear()
    (finding,) = news.WatcherDuty().run(ctx)
    assert finding.action is None and finding.severity == "notable" and "(accounted for)" in finding.headline
