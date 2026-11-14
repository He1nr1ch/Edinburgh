"""The stops held at Alpaca: one day stop order behind every watched position, kept in line with it."""

from datetime import UTC, datetime

import pytest

from tradingagents.dataflows.vendors.alpaca.trading import Account, Holding, StopOrder
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.worker import protect as protect_module
from tradingagents.worker.duty import WorkerContext
from tradingagents.worker.protect import protect
from tradingagents.worker.state import WorkerState

LONG = Holding("JPM", 40, 13_263.2, 300.0, "us_equity")       # at 331.58; the 2% default stop is 294.00
SHORT = Holding("T", -216, -4_520.9, 21.0, "us_equity")       # at 20.93; the 2% default stop is 21.42
NOW = datetime(2026, 11, 16, 16, 0, tzinfo=UTC)               # 11:00 ET


class Broker:
    """Alpaca, as far as the stops go."""

    def __init__(self, monkeypatch, holdings, stops=(), open_orders=None, results=None):
        self.holdings, self.stops = list(holdings), list(stops)
        self.orders, self.results = open_orders or {}, results or {}
        self.sent, self.replaced, self.cancelled = [], [], []
        t = protect_module.trading
        monkeypatch.setattr(t, "get_account", lambda *a, **k: Account(100_000, 0, True, tuple(self.holdings)))
        monkeypatch.setattr(t, "open_orders", lambda *a, **k: {**self.orders, **{s.id: s.symbol for s in self.stops}})
        monkeypatch.setattr(t, "open_stops", lambda *a, **k: list(self.stops))
        monkeypatch.setattr(t, "order_result", lambda oid, *a, **k: self.results[oid])
        monkeypatch.setattr(t, "cancel_order", lambda oid, *a, **k: self.cancelled.append(oid))
        monkeypatch.setattr(t, "submit_stop_order", self._submit)
        monkeypatch.setattr(t, "replace_stop_order", self._replace)

    def _submit(self, symbol, qty, side, stop):
        self.sent.append((symbol, qty, side, stop))
        self.stops.append(StopOrder(f"stop-{symbol}", symbol, side, qty, stop))      # it waits at Alpaca from now on
        return f"stop-{symbol}"

    def _replace(self, order_id, qty, stop):
        self.replaced.append((order_id, qty, stop))
        return f"{order_id}-b"


@pytest.fixture
def desk(tmp_path, monkeypatch):
    monkeypatch.setattr("tradingagents.worker.policy._now", lambda: NOW)
    config = {**DEFAULT_CONFIG, "results_dir": str(tmp_path), "worker_trading_enabled": True,
              "worker_allow_live": False, "worker_trade_from": "10:30", "default_stop_pct": 2.0}
    state = WorkerState.load(tmp_path / "s.json")
    return WorkerContext(Account(100_000, 0, True, ()), NOW, config, state)


@pytest.mark.unit
def test_a_position_without_a_stop_gets_one_at_the_desks_level(desk, monkeypatch):
    broker = Broker(monkeypatch, [LONG, SHORT])
    findings = protect(desk, market_open=True)
    assert broker.sent == [("JPM", 40, "sell", 294.0), ("T", 216, "buy", 21.42)]
    assert desk.state.stops["JPM"]["order_id"] == "stop-JPM" and desk.state.stops["T"]["held"] == "short"
    assert [f.severity for f in findings] == ["info", "info"]       # logged, not an alert
    # The stop is not an order of the worker's that holds the name back.
    assert desk.state.orders == []


@pytest.mark.unit
def test_a_stop_that_fits_is_left_alone_and_one_found_at_alpaca_is_taken_over(desk, monkeypatch):
    broker = Broker(monkeypatch, [LONG], stops=[StopOrder("s1", "JPM", "sell", 40, 294.0)])
    assert protect(desk, market_open=True) == []
    assert not broker.sent and not broker.replaced and not broker.cancelled
    assert desk.state.stops["JPM"]["order_id"] == "s1"


@pytest.mark.unit
def test_a_trim_or_a_new_level_replaces_the_stop(desk, monkeypatch):
    broker = Broker(monkeypatch, [Holding("JPM", 20, 6_631.6, 300.0, "us_equity")],
                    stops=[StopOrder("s1", "JPM", "sell", 40, 294.0)])
    protect(desk, market_open=True)
    assert broker.replaced == [("s1", 20, 294.0)] and not broker.sent
    assert desk.state.stops["JPM"] == {"side": "sell", "held": "long", "qty": 20, "stop": 294.0,
                                       "entry_price": 300.0, "order_id": "s1-b"}


@pytest.mark.unit
def test_a_stop_alpaca_filled_is_recorded_as_the_exit_it_was(desk, monkeypatch):
    desk.state.record_order("JPM", "open", "buy", 40, "ord-1", datetime(2026, 11, 13, 16, 0, tzinfo=UTC), stop=294.0)
    desk.state.stops["JPM"] = {"order_id": "s1", "side": "sell", "held": "long", "qty": 40, "stop": 294.0, "entry_price": 300.0}
    broker = Broker(monkeypatch, [], results={"s1": {"status": "filled", "filled_qty": 40.0, "filled_avg_price": 293.8,
                                                     "filled_at": datetime(2026, 11, 16, 15, 41, tzinfo=UTC)}})
    findings = protect(desk, market_open=True)
    exit_order = desk.state.orders[-1]
    assert (exit_order["kind"], exit_order["side"], exit_order["qty"], exit_order["order_id"]) == ("exit", "sell", 40, "s1")
    assert exit_order["entry_price"] == 300.0 and exit_order["position_id"] == desk.state.orders[0]["position_id"]
    assert "held at Alpaca filled at 293.80" in exit_order["reason"]
    assert [f.severity for f in findings] == ["urgent"] and "closed by Alpaca" in findings[0].headline
    assert desk.state.stops == {} and not broker.sent


@pytest.mark.unit
def test_a_stop_that_lapsed_at_the_close_is_placed_again(desk, monkeypatch):
    desk.state.stops["JPM"] = {"order_id": "old", "side": "sell", "held": "long", "qty": 40, "stop": 294.0, "entry_price": 300.0}
    broker = Broker(monkeypatch, [LONG], results={"old": {"status": "expired", "filled_qty": 0.0,
                                                          "filled_avg_price": None, "filled_at": None}})
    protect(desk, market_open=True)
    assert broker.sent == [("JPM", 40, "sell", 294.0)] and desk.state.orders == []
    assert desk.state.stops["JPM"]["order_id"] == "stop-JPM"


@pytest.mark.unit
def test_a_stop_whose_position_has_gone_is_cancelled_but_a_strangers_is_not(desk, monkeypatch):
    """Left waiting, a sell stop with no long behind it would open a short."""
    desk.state.stops["JPM"] = {"order_id": "s1", "side": "sell", "held": "long", "qty": 40, "stop": 294.0, "entry_price": 300.0}
    broker = Broker(monkeypatch, [], stops=[StopOrder("s1", "JPM", "sell", 40, 294.0),
                                            StopOrder("by-hand", "IBM", "sell", 5, 200.0)])
    protect(desk, market_open=True)
    assert broker.cancelled == ["s1"]


@pytest.mark.unit
def test_nothing_is_left_at_alpaca_before_the_trading_window_or_when_switched_off(desk, monkeypatch):
    broker = Broker(monkeypatch, [LONG])
    monkeypatch.setattr("tradingagents.worker.policy._now", lambda: datetime(2026, 11, 16, 15, 0, tzinfo=UTC))   # 10:00 ET
    assert protect(desk, market_open=True) == [] and not broker.sent
    monkeypatch.setattr("tradingagents.worker.policy._now", lambda: NOW)
    assert protect(desk, market_open=False) == [] and not broker.sent
    desk.config["worker_broker_stops"] = False
    assert protect(desk, market_open=True) == [] and not broker.sent


@pytest.mark.unit
def test_a_position_with_an_order_in_flight_or_already_through_its_stop_waits(desk, monkeypatch):
    desk.state.record_order("JPM", "trim", "sell", 20, "ord-7", NOW)
    through = Holding("T", -216, -4_644.0, 21.0, "us_equity")       # at 21.50, past the 21.42 stop: the guard's to close
    broker = Broker(monkeypatch, [LONG, through], open_orders={"ord-7": "JPM"})
    assert protect(desk, market_open=True) == [] and not broker.sent


@pytest.mark.unit
def test_a_dry_run_says_what_it_would_leave_and_sends_nothing(desk, monkeypatch):
    broker = Broker(monkeypatch, [LONG])
    findings = protect(desk, market_open=True, dry_run=True)
    assert not broker.sent and desk.state.stops == {}
    assert "dry run" in findings[0].detail


@pytest.mark.unit
def test_a_refused_stop_is_reported_and_tried_again_next_time(desk, monkeypatch):
    broker = Broker(monkeypatch, [LONG])
    monkeypatch.setattr(protect_module.trading, "submit_stop_order",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("insufficient qty")))
    findings = protect(desk, market_open=True)
    assert findings[0].severity == "notable" and "insufficient qty" in findings[0].detail
    assert desk.state.stops == {} and not broker.sent


@pytest.mark.unit
def test_every_pass_brings_the_stops_in_line_after_each_duty(desk, monkeypatch, tmp_path):
    from tradingagents.worker import runner

    broker = Broker(monkeypatch, [LONG])
    monkeypatch.setattr(runner.trading, "get_clock", lambda: type("C", (), {"is_open": True})())
    monkeypatch.setattr(runner, "write_snapshot", lambda *a, **k: None)
    monkeypatch.setattr(runner, "send_email", lambda *a, **k: pytest.fail("a stop left at Alpaca is not an alert"))

    class Quiet:
        name = "quiet"
        def run(self, context):
            return []

    cfg = {**desk.config, "worker_state_path": str(tmp_path / "s.json"), "mirror_enabled": False}
    findings = runner.run_pass(cfg, None, [Quiet(), Quiet()])
    assert broker.sent == [("JPM", 40, "sell", 294.0)]          # the second look finds it waiting
    assert [f.duty for f in findings] == ["protect"]
    assert WorkerState.load(tmp_path / "s.json").stops["JPM"]["order_id"] == "stop-JPM"


@pytest.mark.unit
def test_cancelling_waits_until_alpaca_has_let_the_shares_go(monkeypatch):
    from tradingagents.dataflows.vendors.alpaca import trading

    statuses = iter(["pending_cancel", "canceled"])

    class Client:
        cancelled = []
        def cancel_order_by_id(self, oid):
            self.cancelled.append(oid)
        def get_order_by_id(self, oid):
            return type("O", (), {"status": next(statuses), "filled_qty": 0, "filled_avg_price": None, "filled_at": None})()

    monkeypatch.setattr("time.sleep", lambda s: None)
    client = Client()
    assert trading.cancel_orders_for("JPM", {"s1": "JPM", "x": "T"}, client) == 1
    assert client.cancelled == ["s1"] and next(statuses, "all asked") == "all asked"
