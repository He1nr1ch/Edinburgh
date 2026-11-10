"""The daily transaction list: what was exited and why, what was entered."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from tradingagents.web import transactions
from tradingagents.worker.state import WorkerState


def _state(tmp_path):
    s = WorkerState.load(tmp_path / "state.json")
    at = lambda h, m: datetime(2026, 11, 9, h, m, tzinfo=UTC)  # noqa: E731
    s.record_order("TROW", "open", "sell", 18, "o1", at(14, 30), "the day's book (night.csv)")
    s.record_order("GD", "open", "buy", 16, "o2", at(14, 30), "the day's book (night.csv)")
    s.record_order("CRL", "add", "buy", 1, "o3", at(14, 35), "high, favorable, not accounted for: acquisition of Frontage")
    s.record_order("TROW", "exit", "buy", 18, "o4", at(15, 50), "price 111.41 is through the stop 111.36 (Risk Manager); -2.1% on the position")
    s.record_order("ACGL", "replace", "buy", 18, "o5", at(15, 56), "replaces TROW: best score on the bench (score 6.5, +17.5% to target)")
    s.record_order("OLD", "exit", "sell", 5, "o0", datetime(2026, 11, 6, 15, 0, tzinfo=UTC), "yesterday")
    return s


@pytest.mark.unit
def test_the_rows_name_the_action_the_positions_side_and_the_fill(tmp_path):
    rows = transactions.transactions_on(_state(tmp_path), "2026-11-09", {"o4": {"filled_avg_price": 111.5, "filled_qty": 18, "status": "filled"}})
    assert [(r["time"], r["action"], r["ticker"], r["position"]) for r in rows] == [
        ("14:30", "Entered", "TROW", "short"), ("14:30", "Entered", "GD", "long"), ("14:35", "Added to", "CRL", "long"),
        ("15:50", "Exited", "TROW", "short"), ("15:56", "Entered (replacement)", "ACGL", "long")]
    trow_exit = rows[3]
    assert (trow_exit["price"], trow_exit["value"], trow_exit["status"]) == (111.5, 2007.0, "filled")
    assert "through the stop 111.36" in trow_exit["reason"] and rows[1]["price"] is None


@pytest.mark.unit
def test_the_summary_puts_exits_first_with_their_reasons(tmp_path):
    text = transactions.summarize(transactions.transactions_on(_state(tmp_path), "2026-11-09"), "2026-11-09")
    assert text.startswith("Edinburgh transactions for 2026-11-09: 1 exit(s), 4 entries.")
    assert text.index("Exited:") < text.index("Entered:")
    assert "- 15:50 Exited short TROW: 18 shares\n    price 111.41 is through the stop 111.36" in text
    assert "No orders were sent." in transactions.summarize([], "2026-11-10")


@pytest.mark.unit
def test_days_with_transactions_count_exits_and_entries_newest_first(tmp_path):
    assert transactions.days_with_transactions(_state(tmp_path)) == [("2026-11-09", 1, 4), ("2026-11-06", 1, 0)]


@pytest.mark.unit
def test_the_pages_list_the_transactions(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from tradingagents.dataflows.vendors.alpaca.trading import Account
    from tradingagents.web import app as web

    state = _state(tmp_path)
    state.save()
    monkeypatch.setattr(web, "_results_dir", lambda: tmp_path)
    monkeypatch.setattr(web, "_default_stop", lambda: 2.0)
    monkeypatch.setattr(web, "_holdings", lambda: Account(100_000, 100_000, True, ()))
    monkeypatch.setattr(web, "get_config", lambda: {"results_dir": str(tmp_path), "worker_state_path": str(state.path), "default_stop_pct": 2.0})
    monkeypatch.setattr(web, "_transactions_rows", lambda day: transactions.transactions_on(state, day))   # no Alpaca for fills
    client = TestClient(web.app)

    index = client.get("/transactions").text
    assert '/transactions/2026-11-09' in index and "<td>1</td><td>4</td>" in index
    page = client.get("/transactions/2026-11-09").text
    assert "Exited" in page and "TROW" in page and "through the stop 111.36" in page and "Entered (replacement)" in page
    assert "1 exit(s), 4 entries" in page
    assert client.get("/").status_code == 200


@pytest.mark.unit
def test_an_exit_states_its_result_from_the_recorded_entry_or_the_entry_orders_fill(tmp_path):
    s = WorkerState.load(tmp_path / "state.json")
    at = lambda d, h, m: datetime(2026, 11, d, h, m, tzinfo=UTC)  # noqa: E731
    s.record_order("TROW", "open", "sell", 18, "in1", at(9, 14, 30), "the day's book")
    s.record_order("TROW", "exit", "buy", 18, "out1", at(9, 15, 50), "stop")                     # no entry recorded: from in1's fill
    s.record_order("GD", "open", "buy", 16, "in2", at(6, 14, 30), "the day's book")              # entered on an earlier day
    s.record_order("GD", "exit", "sell", 16, "out2", at(9, 16, 0), "target", entry_price=358.05)  # entry recorded with the exit
    today = {"in1": {"filled_avg_price": 109.10, "filled_qty": 18, "status": "filled"},
             "out1": {"filled_avg_price": 111.50, "filled_qty": 18, "status": "filled"},
             "out2": {"filled_avg_price": 380.00, "filled_qty": 16, "status": "filled"}}
    asked = []
    rows = transactions.transactions_on(s, "2026-11-09", today, lambda d: asked.append(d) or {})

    trow = next(r for r in rows if r["ticker"] == "TROW" and r["flow"] == "out")
    assert (trow["entry_price"], trow["pnl"], trow["pnl_pct"]) == (109.10, -43.2, pytest.approx(-2.2, abs=0.01))   # a short: bought back higher
    gd = next(r for r in rows if r["ticker"] == "GD" and r["flow"] == "out")
    assert (gd["entry_price"], gd["pnl"], gd["pnl_pct"]) == (358.05, 351.2, pytest.approx(6.13, abs=0.01))
    assert asked == []                                       # GD's entry was recorded, so no other day's fills were needed
    assert transactions.realised(rows) == 308.0
    text = transactions.summarize(rows, "2026-11-09")
    assert "realised +308.00." in text and "entered 109.10, P/L -43.20 (-2.2%)" in text


@pytest.mark.unit
def test_an_exit_sent_by_the_watcher_records_the_positions_entry_price(tmp_path, monkeypatch):
    from unittest.mock import MagicMock

    from tradingagents.dataflows.vendors.alpaca.trading import Account, Holding
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.worker import runner
    from tradingagents.worker.duty import Action, Finding, WorkerContext

    monkeypatch.setattr(runner.trading, "submit_market_order", lambda s, q, side: "ord-x")
    monkeypatch.setattr(runner.trading, "cancel_orders_for", lambda s, o: 0)
    state = WorkerState.load(tmp_path / "s.json")
    cfg = {**DEFAULT_CONFIG, "worker_trading_enabled": True, "worker_allow_live": False}
    ctx = WorkerContext(Account(100_000, 0, True, (Holding("LEN", -54, -4_950.0, 90.08, "us_equity"),)), datetime.now(UTC), cfg, state)
    runner._act(Finding("stops", "LEN", "urgent", "h", "d", Action("LEN", "exit", "buy", 54, "stop")), ctx, True, set(), {}, False)
    assert state.orders[-1]["entry_price"] == 90.08
