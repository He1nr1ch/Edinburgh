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
    """Closed: sleep toward the open. An error: log it, wait a minute, carry on."""
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
    assert slept == [15 * 60, 60]      # a 15-minute nap toward Monday's open, then the retry wait
