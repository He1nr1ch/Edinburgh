"""The mirror account: Valya holds the exact opposite of the desk, with the levels swapped."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from tradingagents import mirror
from tradingagents.dataflows.vendors.alpaca.trading import Account, Holding
from tradingagents.default_config import DEFAULT_CONFIG


def _acct(*holdings, equity=100_000.0, cash=50_000.0, paper=True):
    return Account(equity, cash, paper, tuple(holdings))


def _h(symbol, qty, price=100.0, entry=None, asset_class="us_equity"):
    return Holding(symbol, qty, qty * price, entry or price, asset_class)


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("API_KEY_VALYA", "k")
    monkeypatch.setenv("API_SECRET_VALYA", "s")
    return {**DEFAULT_CONFIG, "results_dir": str(tmp_path), "mirror_state_path": str(tmp_path / "memory" / "mirror.json")}


@pytest.mark.unit
def test_the_wanted_book_is_the_negation_of_the_desks_equities_only():
    main = _acct(_h("WM", 20), _h("NKE", -259), _h("SPY261120C00680000", 2, asset_class="us_option"))
    assert mirror.target_positions(main) == {"WM": -20, "NKE": 259}
    assert mirror.current_positions(_acct(_h("WM", -20), _h("BTCUSD", 1, asset_class="crypto"))) == {"WM": -20}


@pytest.mark.unit
def test_the_orders_take_her_book_to_the_opposite_one_name_at_a_time():
    wanted = {"WM": -20, "NKE": 259, "CRL": -12, "LEN": 54, "ACGL": -18}
    have = {"WM": -20, "NKE": 259, "CRL": -11, "LEN": 60, "TROW": 18, "OKTA": 17}
    plan = {o["symbol"]: o for o in mirror.orders_to_send(wanted, have, main_pending={"ACGL"}, mirror_pending={"OKTA"})}
    assert "WM" not in plan and "NKE" not in plan                       # already the opposite
    assert (plan["CRL"]["side"], plan["CRL"]["qty"]) == ("sell", 1) and "added" in plan["CRL"]["why"]
    assert (plan["LEN"]["side"], plan["LEN"]["qty"]) == ("sell", 6) and "trimmed" in plan["LEN"]["why"]
    assert (plan["TROW"]["side"], plan["TROW"]["qty"]) == ("sell", 18) and "closed" in plan["TROW"]["why"]
    assert plan["ACGL"]["side"] is None and "desk's own order" in plan["ACGL"]["why"]
    assert plan["OKTA"]["side"] is None and "mirror order" in plan["OKTA"]["why"]
    # A name that must change side is flattened first; the other way comes on the next pass.
    flip = mirror.orders_to_send({"GD": 11}, {"GD": -11}, set(), set())
    assert flip == [{"symbol": "GD", "side": "buy", "qty": 11, "why": "flatten short 11 before going the other way"}]
    assert mirror.orders_to_send({"GD": 11}, {}, set(), set())[0] == {"symbol": "GD", "side": "buy", "qty": 11,
                                                                         "why": "the desk opened a short of 11: open long 11"}


class FakeBroker:
    """Both accounts' orders, told apart by the client object."""

    def __init__(self, monkeypatch, main: Account, hers: Account, refuse=()):
        self.sent: list[tuple] = []
        self.her_client = object()
        monkeypatch.setattr(mirror, "client", lambda: self.her_client)
        monkeypatch.setattr(mirror.trading, "get_account", lambda client=None, paper=None: hers if client is self.her_client else main)
        monkeypatch.setattr(mirror.trading, "open_orders", lambda client=None: {})
        monkeypatch.setattr(mirror.trading, "get_clock", lambda: type("C", (), {"is_open": True})())

        def submit(symbol, qty, side, client=None):
            if symbol in refuse:
                raise RuntimeError(f"{symbol} is not shortable")
            self.sent.append((symbol, qty, side, client is self.her_client))
            return f"ord-{len(self.sent)}"

        monkeypatch.setattr(mirror.trading, "submit_market_order", submit)


@pytest.mark.unit
def test_follow_sends_her_the_opposite_orders_and_notes_what_alpaca_refused(cfg, monkeypatch, tmp_path):
    main = _acct(_h("WM", 20), _h("NKE", -259), _h("XYZ", 5))
    hers = _acct(_h("TROW", -18))
    broker = FakeBroker(monkeypatch, main, hers, refuse={"XYZ"})
    sent = mirror.follow(cfg)
    assert sorted(broker.sent) == [("NKE", 259, "buy", True), ("TROW", 18, "buy", True), ("WM", 20, "sell", True)]
    assert all(o["order_id"] for o in sent) and len(sent) == 3
    state = json.loads((tmp_path / "memory" / "mirror.json").read_text())
    assert [o["symbol"] for o in state["orders"]] == ["NKE", "TROW", "WM"]
    assert "not shortable" in state["refused"]["XYZ"]["why"] and state["last_follow"]
    log = (tmp_path / "worker" / f"{datetime.now(UTC).astimezone():%Y-%m-%d}.log").read_text()
    assert "mirror: sent sell 20 WM" in log and "mirror: refused: sell 5 XYZ" in log
    # The next pass, with her book now the opposite, sends nothing and clears nothing by itself.
    hers2 = _acct(_h("WM", -20), _h("NKE", 259))
    broker2 = FakeBroker(monkeypatch, _acct(_h("WM", 20), _h("NKE", -259)), hers2)
    assert mirror.follow(cfg) == [] and broker2.sent == []


@pytest.mark.unit
def test_follow_waits_for_the_open_respects_dry_run_and_needs_the_keys(cfg, monkeypatch):
    main, hers = _acct(_h("WM", 20)), _acct()
    broker = FakeBroker(monkeypatch, main, hers)
    assert mirror.follow(cfg, market_open=False) == [] and broker.sent == []
    wanted = mirror.follow(cfg, dry_run=True)
    assert [(o["symbol"], o["side"], o["qty"], o["order_id"]) for o in wanted] == [("WM", "sell", 20, None)] and broker.sent == []
    monkeypatch.delenv("API_KEY_VALYA")
    assert not mirror.configured(cfg) and mirror.follow(cfg) == []
    monkeypatch.setenv("API_KEY_VALYA", "k")
    assert not mirror.configured({**cfg, "mirror_enabled": False})


@pytest.mark.unit
def test_her_page_shows_the_desks_target_as_her_stop_and_the_desks_stop_as_her_target(cfg, monkeypatch, tmp_path):
    from tradingagents.web import data

    class Ref:
        date = "2026-11-10"

    monkeypatch.setattr(data, "load_decision", lambda ref: {})
    monkeypatch.setattr(data, "decision_summary", lambda state, close, dflt: {"stop": 95.0, "target": 120.0})
    main = _acct(_h("WM", 20, price=100.0), _h("NKE", -10, price=40.0), _h("NEW", 3))
    hers = _acct(_h("WM", -20, price=100.0, entry=98.0), _h("NKE", 10, price=40.0, entry=41.0))
    rows = {r["ticker"]: r for r in mirror.positions_view(hers, main, {"WM": Ref(), "NKE": Ref()}, 2.0,
                                                           refused={"NEW": {"why": "NEW is not shortable"}})}
    wm = rows["WM"]
    assert wm["side"] == "short" and wm["desk_side"] == "long" and wm["matched"]
    assert (wm["stop"], wm["target"]) == (120.0, 95.0)                   # swapped
    assert wm["to_stop_pct"] == pytest.approx(-20.0) and wm["to_target_pct"] == pytest.approx(5.0)
    assert wm["pnl_pct"] == pytest.approx(-100 * (100 / 98 - 1))
    assert rows["NEW"]["missing"] == "NEW is not shortable" and rows["NEW"]["side"] == "short"


@pytest.mark.unit
def test_the_mirror_page_and_its_json(cfg, monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from tradingagents.web import app as web

    main = _acct(_h("WM", 20), _h("NKE", -259))
    hers = _acct(_h("WM", -20), equity=99_500, cash=80_000)
    monkeypatch.setattr(web, "get_config", lambda: cfg)
    monkeypatch.setattr(web, "_results_dir", lambda: tmp_path)
    monkeypatch.setattr(web, "_holdings", lambda: main)
    monkeypatch.setattr(web, "_mirror_account", lambda: hers)
    client = TestClient(web.app)
    page = client.get("/mirror").text
    assert "<h1>Valya</h1>" in page and "equity 99,500" in page and "exact opposite" in page
    assert "<b>WM</b>" in page and "<b>NKE</b>" in page and "not held: not yet mirrored" in page
    js = client.get("/api/mirror").json()
    assert js["name"] == "Valya" and [p["ticker"] for p in js["positions"]] == ["WM", "NKE"]
    assert js["positions"][1]["missing"] == "not yet mirrored"
    assert 'href="/mirror">Valya</a>' in client.get("/").text
    monkeypatch.delenv("API_KEY_VALYA")
    assert client.get("/mirror").status_code == 404 and client.get("/api/mirror").status_code == 404


@pytest.mark.unit
def test_the_worker_follows_the_mirror_after_a_pass_and_a_failure_is_not_fatal(cfg, monkeypatch, caplog):
    import logging

    from tradingagents.worker import runner

    calls = []
    monkeypatch.setattr(mirror, "follow", lambda config, dry_run, market_open=None: calls.append((dry_run, market_open)))
    runner.follow_mirror(cfg, False, True)
    assert calls == [(False, True)]
    monkeypatch.setattr(mirror, "follow", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("broker down")))
    with caplog.at_level(logging.WARNING):
        runner.follow_mirror(cfg, False, True)
    assert "could not be followed: broker down" in caplog.text
    monkeypatch.delenv("API_KEY_VALYA")
    calls.clear()
    monkeypatch.setattr(mirror, "follow", lambda *a, **k: calls.append(1))
    runner.follow_mirror(cfg, False, True)
    assert calls == []                                   # no keys: never called


@pytest.mark.unit
def test_flatten_closes_everything_she_holds_and_notes_it(cfg, monkeypatch, tmp_path):
    hers = _acct(_h("WM", -20), _h("NKE", 259))
    calls = []

    class Client:
        def close_all_positions(self, cancel_orders):
            calls.append(cancel_orders)
            return []

    cl = Client()
    monkeypatch.setattr(mirror, "client", lambda: cl)
    monkeypatch.setattr(mirror.trading, "get_account", lambda client=None, paper=None: hers)
    assert mirror.flatten(cfg, dry_run=True) == ["short 20 WM", "long 259 NKE"] and calls == []
    assert mirror.flatten(cfg) == ["short 20 WM", "long 259 NKE"] and calls == [True]
    state = json.loads((tmp_path / "memory" / "mirror.json").read_text())
    assert [(o["symbol"], o["side"], o["qty"]) for o in state["orders"]] == [("WM", "buy", 20), ("NKE", "sell", 259)]
    assert "flattened: closed 2 position(s)" in (tmp_path / "worker" / f"{datetime.now(UTC).astimezone():%Y-%m-%d}.log").read_text()
