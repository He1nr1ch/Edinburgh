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
    """Closed: nap half a minute at a time so the sweep's pool stays full. An error: log it, wait a minute, carry on."""
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
    assert slept == [30, 60]           # a half-minute nap (the sweep's pool is looked in on), then the retry wait


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
    cfg = {**CONFIG, "results_dir": str(tmp_path), "worker_news_lookback_hours": 18,
           "feedback_db_url": "sqlite:///" + str(tmp_path / "feedback.db")}
    article = {"id": "a1", "headline": "JPMorgan CFO resigns abruptly", "summary": "", "source": "WSJ",
               "created_at": datetime(2026, 11, 6, 15, 0, tzinfo=UTC), "url": "u"}
    monkeypatch.setattr(news, "articles_since", lambda s, since, limit=20: [article])
    monkeypatch.setattr(news, "_name", lambda s: "JPMorgan Chase")
    captured = {}
    structured = MagicMock()
    structured.invoke.side_effect = lambda prompt: captured.__setitem__("prompt", prompt) or news.NewsAssessment(
        materiality="high", direction="adverse", accounted_for=False, summary="CFO out.", rationale="Not in the thesis.",
        confidence="80%", evidence=["CFO resigned 2026-11-06", "thesis names only the Basel rule"])
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
    assert "**Confidence**: 0.8" in finding.detail and "CFO resigned 2026-11-06" in finding.detail

    structured.invoke.side_effect = lambda prompt: news.NewsAssessment(
        materiality="high", direction="adverse", accounted_for=True, summary="Basel rule.", rationale="Known.")
    state.last_check.clear(); state.seen.clear()
    (finding,) = news.WatcherDuty().run(ctx)
    assert finding.action is None and finding.severity == "notable" and "(accounted for)" in finding.headline

    # Both judgements are kept as typed verdicts, with what the desk did and what it was asked about.
    from tradingagents.feedback import FeedbackStore
    first, second = FeedbackStore(cfg).desk_verdicts(kind="news", ticker="JPM")
    assert (first["agent"], first["call"], first["confidence"], first["action"]) == ("Watcher", "high adverse", 0.8, "exit 40")
    assert first["evidence"] == ["CFO resigned 2026-11-06", "thesis names only the Basel rule"] and first["thesis"] == "Not in the thesis."
    assert first["fields"]["accounted_for"] is False and first["context"]["articles"][0]["headline"].startswith("JPMorgan CFO")
    assert first["day"] == "2026-11-06" and first["ok"] and first["latency_s"] is not None
    assert (second["action"], second["confidence"], second["fields"]["accounted_for"]) == ("none", None, True)

    # An answer that does not meet the contract is kept too, as a failed verdict.
    structured.invoke.side_effect = ValueError("not the schema")
    state.last_check.clear()
    state.seen.clear()
    news.WatcherDuty().run(ctx)
    failed = FeedbackStore(cfg).desk_verdicts(kind="news")[-1]
    assert not failed["ok"] and "not the schema" in failed["error"] and failed["call"] is None


@pytest.mark.unit
def test_a_decision_for_the_other_side_does_not_speak_for_the_position(tmp_path):
    """The review closes a long now rated Underweight; until it has, the short's stop above the price is not the long's."""
    from tradingagents.web import data
    from tradingagents.worker.stops import StopDuty, TargetDuty, standing_stop, standing_target

    _decision(tmp_path, "CPAY", "2026-11-12", final_rating="Underweight",
              risk_assessment="**Maximum Position**: 2\n\n**Stop Loss**: 483.50\n", final_trade_decision="**Price Target**: 405\n")
    _decision(tmp_path, "GD", "2026-11-12", final_rating="Overweight",
              risk_assessment="**Maximum Position**: 2\n\n**Stop Loss**: 337.00\n", final_trade_decision="**Price Target**: 395\n")
    cfg = {**CONFIG, "results_dir": str(tmp_path), "default_stop_pct": 2.0}
    long_cpay = Holding("CPAY", 9, 9 * 459.37, 459.74, "us_equity")       # long, the desk now says short
    short_gd = Holding("GD", -11, -11 * 364.94, 364.65, "us_equity")      # short, the desk now says long
    decisions = data.find_decisions(tmp_path)

    assert standing_stop(long_cpay, decisions, 2.0) == (pytest.approx(450.55, abs=0.01), "default 2%")
    assert standing_stop(short_gd, decisions, 2.0) == (pytest.approx(371.94, abs=0.01), "default 2%")
    assert standing_target(long_cpay, decisions) is None and standing_target(short_gd, decisions) is None

    ctx = WorkerContext(Account(100_000, 0, True, (long_cpay, short_gd)), datetime.now(UTC), cfg, None)
    assert StopDuty().run(ctx) == [] and TargetDuty().run(ctx) == []

    # The default still guards it: 2% against the long from its entry closes it.
    fallen = Holding("CPAY", 9, 9 * 449.0, 459.74, "us_equity")
    ctx = WorkerContext(Account(100_000, 0, True, (fallen,)), datetime.now(UTC), cfg, None)
    assert [(f.symbol, f.action.side, f.action.qty) for f in StopDuty().run(ctx)] == [("CPAY", "sell", 9)]


@pytest.mark.unit
def test_an_exit_never_sells_or_covers_more_than_is_held_now(monkeypatch, tmp_path):
    """The pass's picture of the account is minutes old when its orders go out: an exit for a position
    the review has already closed would open it the other way round."""
    from tradingagents.worker import runner
    from tradingagents.worker.duty import Finding

    sent = []
    monkeypatch.setattr(runner.trading, "submit_market_order", lambda s, q, side: sent.append((s, q, side)) or "ord-1")
    monkeypatch.setattr(runner.trading, "cancel_orders_for", lambda s, open_now: 0)
    state = WorkerState.load(tmp_path / "s.json")
    ctx = WorkerContext(Account(100_000, 0, True, (LONG, SHORT)), datetime.now(UTC), CONFIG, state)

    def act(action, live):
        f = Finding("stops", action.symbol, "urgent", "h", "d", action)
        runner._act(f, ctx, market_open=True, pending=set(), open_now={}, dry_run=False, live=live)
        return f

    gone = act(Action("JPM", "exit", "sell", 40, "stop"), {"T": -216})                  # already closed
    assert sent == [] and gone.outcome == "not sent: JPM is no longer held long"
    flipped = act(Action("JPM", "exit", "sell", 40, "stop"), {"JPM": -40})              # held the other way now
    assert sent == [] and "no longer held long" in flipped.outcome
    covered = act(Action("T", "exit", "buy", 216, "stop"), {})
    assert sent == [] and covered.outcome == "not sent: T is no longer held short"

    live = {"JPM": 15}
    part = act(Action("JPM", "exit", "sell", 40, "stop"), live)                         # trimmed since the pass began
    assert sent == [("JPM", 15, "sell")] and part.action.qty == 15 and live["JPM"] == 0
    again = act(Action("JPM", "exit", "sell", 40, "target"), live)                      # a second exit in the same batch
    assert sent == [("JPM", 15, "sell")] and "no longer held" in again.outcome

    act(Action("AMZN", "replace", "sell", 5, "bench"), {})                              # an opening order is not capped
    assert sent[-1] == ("AMZN", 5, "sell")


@pytest.mark.unit
def test_each_dutys_orders_go_out_before_the_next_duty_runs(monkeypatch, tmp_path):
    """A stop found in the first second does not wait for the news duty's ten minutes with the model."""
    from tradingagents.worker import runner
    from tradingagents.worker.duty import Finding

    events = []
    holdings = [LONG]
    monkeypatch.setattr(runner.trading, "get_account", lambda *a, **k: Account(100_000, 0, True, tuple(holdings)))
    monkeypatch.setattr(runner.trading, "get_clock", lambda: type("C", (), {"is_open": True})())
    monkeypatch.setattr(runner.trading, "open_orders", lambda *a, **k: {})
    monkeypatch.setattr(runner.trading, "cancel_orders_for", lambda s, open_now: 0)
    monkeypatch.setattr(runner, "write_snapshot", lambda *a, **k: None)
    monkeypatch.setattr(runner, "send_email", lambda *a, **k: True)

    def submit(symbol, qty, side):
        events.append(f"order {side} {qty} {symbol}")
        holdings.clear()                     # it fills at once: the position is gone
        return "ord-1"
    monkeypatch.setattr(runner.trading, "submit_market_order", submit)

    class Stops:
        name = "stops"
        def run(self, context):
            events.append("stops ran")
            return [Finding("stops", "JPM", "urgent", "h", "d", Action("JPM", "exit", "sell", 40, "stop"))]

    class News:
        name = "news"
        def run(self, context):              # slow, and still looking at the pass's first picture of the account
            events.append("news ran")
            return [Finding("news", "JPM", "urgent", "h", "d", Action("JPM", "exit", "sell", 40, "adverse news"))]

    cfg = {**CONFIG, "results_dir": str(tmp_path), "worker_state_path": str(tmp_path / "s.json"), "mirror_enabled": False,
           "worker_broker_stops": False}
    findings = runner.run_pass(cfg, None, [Stops(), News()])
    assert events == ["stops ran", "order sell 40 JPM", "news ran"]
    assert findings[0].outcome.startswith("sent") and findings[1].outcome == "not sent: JPM is no longer held long"
    assert len(WorkerState.load(tmp_path / "s.json").orders) == 1


@pytest.mark.unit
def test_no_order_goes_out_before_the_trading_window_opens(monkeypatch, tmp_path):
    from zoneinfo import ZoneInfo

    from tradingagents.worker import runner
    from tradingagents.worker.policy import too_early

    et = ZoneInfo("America/New_York")
    cfg = {**CONFIG, "worker_trade_from": "10:30"}
    at = lambda h, m: datetime(2026, 11, 16, h, m, tzinfo=et)       # noqa: E731
    exit_ = Action("JPM", "exit", "sell", 40, "stop")
    ok = dict(market_open=True, paper=True, pending=set())

    assert too_early(cfg, at(9, 30)) == "no trades before 10:30 ET (worker_trade_from)"
    assert too_early(cfg, at(10, 29)) and too_early(cfg, at(10, 30)) is None and too_early(cfg, at(15, 59)) is None
    assert too_early(cfg, datetime(2026, 11, 16, 15, 0, tzinfo=UTC)) is not None          # 10:00 ET, whatever the clock's zone
    assert too_early({**cfg, "worker_trade_from": ""}, at(9, 30)) is None
    assert "no trades before 10:30" in refusal(exit_, **ok, config=cfg, now=at(9, 45))    # a stop waits as well
    assert refusal(exit_, **ok, config=cfg, now=at(10, 30)) is None
    assert DEFAULT_CONFIG["worker_trade_from"] == "10:30"

    # The day's loop neither places the book nor runs a pass until the window opens, then does both, the book once.
    calls, clock = [], iter([True, True, False])
    early = iter(["no trades before 10:30 ET", None, None])
    monkeypatch.setattr(runner, "too_early", lambda config, now=None: next(early))
    monkeypatch.setattr(runner, "place_book", lambda config, dry_run=False: calls.append("book") or [])
    monkeypatch.setattr(runner, "run_pass", lambda *a, **k: calls.append("pass") or [])
    monkeypatch.setattr(runner.trading, "get_clock", lambda: type("C", (), {"is_open": next(clock)})())
    monkeypatch.setattr(runner.time, "sleep", lambda s: None)
    monkeypatch.setattr(runner, "send_email", lambda *a, **k: True)
    monkeypatch.setattr(runner, "summarize_day", lambda *a, **k: "")
    monkeypatch.setattr(runner, "run_reports", lambda *a, **k: None)
    runner.run_day({**cfg, "results_dir": str(tmp_path), "worker_state_path": str(tmp_path / "s.json")}, None, [], 5)
    assert calls == ["book", "pass", "pass"]


@pytest.mark.unit
def test_the_desks_stop_stands_for_a_position_entered_beyond_it(tmp_path):
    """Short FDX entered at 322.10 with the desk's stop at 318: the stop is the desk's, not 2% from the entry."""
    from tradingagents.web import data
    from tradingagents.worker.stops import StopDuty, standing_stop

    _decision(tmp_path, "FDX", "2026-11-13", final_rating="Underweight",
              risk_assessment="**Maximum Position**: 3\n\n**Stop Loss**: 318.0\n", final_trade_decision="**Price Target**: 288\n")
    cfg = {**CONFIG, "results_dir": str(tmp_path), "default_stop_pct": 2.0}
    above = Holding("FDX", -3, -3 * 320.0, 322.10, "us_equity")
    below = Holding("FDX", -3, -3 * 316.0, 322.10, "us_equity")
    assert standing_stop(above, data.find_decisions(tmp_path), 2.0) == (318.0, "Risk Manager")

    (finding,) = StopDuty().run(WorkerContext(Account(100_000, 0, True, (above,)), datetime.now(UTC), cfg, None))
    assert "stop hit at 318.00 (Risk Manager)" in finding.headline and finding.action.side == "buy"
    assert StopDuty().run(WorkerContext(Account(100_000, 0, True, (below,)), datetime.now(UTC), cfg, None)) == []


@pytest.mark.unit
def test_a_call_sized_on_the_close_is_judged_again_at_the_live_price():
    from tradingagents.worker.policy import stale_levels

    assert stale_levels("long", 237.14, 247.5, 296.56, 5) == "the price 237.14 is already through its stop 247.50"
    assert stale_levels("short", 325.29, 322.63, 250.77, 5) == "the price 325.29 is already through its stop 322.63"
    assert "only +1.7% is left to its target" in stale_levels("long", 29.5, 25.0, 30.0, 5)
    assert stale_levels("long", 269.0, 247.5, 296.56, 5) is None and stale_levels("short", 303.0, 318.0, 270.0, 5) is None
    assert stale_levels("long", None, 95.0, 120.0, 5) == "no live price"
    assert stale_levels("long", 100.0, None, 120.0, 5) is None                         # no stop given: the default applies


@pytest.mark.unit
def test_a_book_position_the_pass_has_not_seen_yet_is_not_replaced(tmp_path, monkeypatch):
    """2026-11-12, 09:30: PSX was opened as the pass began, so the pass's account did not show it; by the
    time the replacement duty ran the order had filled and was not waiting either. TRV "replaced" it
    while it stood. The duty reads the broker itself, and the broker shows PSX held."""
    from tradingagents.worker import replace

    state = WorkerState.load(tmp_path / "state.json")
    state.book = {"PSX": {"placed_on": "2026-11-12", "side": "short", "value": 2335.0}}
    monkeypatch.setattr(replace.trading, "open_orders", lambda: {})
    monkeypatch.setattr(replace.trading, "get_account",
                        lambda: Account(100_000, 0, True, (Holding("PSX", -9, -2335.0, 259.49, "us_equity"),)))
    monkeypatch.setattr(replace, "latest_bench", lambda *a: pytest.fail("nothing left the book"))
    stale = WorkerContext(Account(100_000, 0, True, ()), datetime(2026, 11, 12, 14, 30, tzinfo=UTC),
                          {**CONFIG, "results_dir": str(tmp_path)}, state)

    assert replace.ReplacementDuty().run(stale) == [] and "PSX" in state.book

    # Still on its way to the exchange: waiting, not gone.
    monkeypatch.setattr(replace.trading, "open_orders", lambda: {"ord-1": "PSX"})
    monkeypatch.setattr(replace.trading, "get_account", lambda: Account(100_000, 0, True, ()))
    assert replace.ReplacementDuty().run(stale) == [] and "PSX" in state.book


@pytest.mark.unit
def test_the_days_record_keeps_every_build_that_ran_in_the_session(tmp_path, monkeypatch):
    from tradingagents.worker import runner

    cfg = {**CONFIG, "results_dir": str(tmp_path), "worker_state_path": str(tmp_path / "state.json")}
    monkeypatch.setattr(runner, "too_early", lambda config, now=None: None)
    monkeypatch.setattr(runner, "place_book", lambda *a, **k: [])
    monkeypatch.setattr(runner, "run_pass", lambda *a, **k: [])
    monkeypatch.setattr(runner.trading, "get_clock", lambda: type("C", (), {"is_open": False})())
    monkeypatch.setattr(runner, "send_email", lambda *a, **k: True)
    monkeypatch.setattr(runner, "summarize_day", lambda *a, **k: "")
    monkeypatch.setattr(runner, "run_reports", lambda *a, **k: None)

    for running in ("1.2026.11.11.19 (d668d94)", "1.2026.11.11.19 (d668d94)", "1.2026.11.13.12 (4fc228e)"):
        monkeypatch.setattr(runner.build, "describe", lambda running=running: running)
        runner.run_day(cfg, None, [], 5)
    (record,) = [d["build"] for d in WorkerState.load(cfg["worker_state_path"]).days.values()]
    assert record.startswith("1.2026.11.11.19 (d668d94); then 1.2026.11.13.12 (4fc228e) from ") and record.count("then") == 1
