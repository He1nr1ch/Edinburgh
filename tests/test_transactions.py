"""The daily transaction list: what was exited and why, what was entered."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from tradingagents.web import transactions
from tradingagents.web.transactions import summarize, transactions_on
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


@pytest.mark.unit
def test_an_exit_states_its_result_in_r_over_the_risk_it_was_opened_with(tmp_path):
    import json

    s = WorkerState.load(tmp_path / "state.json")
    at = lambda d, h, m: datetime(2026, 11, d, h, m, tzinfo=UTC)  # noqa: E731
    s.record_order("GD", "open", "buy", 16, "in1", at(6, 14, 30), "the day's book", stop=350.0)          # the stop kept with the order
    s.record_order("GD", "exit", "sell", 16, "out1", at(9, 16, 0), "target", entry_price=358.05)
    s.record_order("TROW", "open", "sell", 18, "in2", at(6, 14, 30), "the day's book (b.csv)")            # from before stops were kept
    s.record_order("TROW", "exit", "buy", 18, "out2", at(9, 15, 50), "stop", entry_price=109.10)
    s.record_order("ACGL", "replace", "buy", 18, "in3", at(6, 15, 0), "replaces X")                       # no book row: its decision's stop
    s.record_order("ACGL", "trim", "sell", 9, "out3", at(9, 15, 55), "news", entry_price=100.0)
    s.record_order("LEN", "open", "buy", 10, "in4", at(6, 14, 30), "the day's book (b.csv)")              # no stop anywhere: the default
    s.record_order("LEN", "exit", "sell", 10, "out4", at(9, 15, 58), "stop", entry_price=100.0)
    (tmp_path / "b.csv").write_text("ticker,side,stop\nTROW,short,111.36\nLEN,long,\n")
    s.day("2026-11-06")["book_path"] = str(tmp_path / "b.csv")
    for day, stop in (("2026-11-05", 96.0), ("2026-11-09", 99.0)):                                          # the later decision does not move the risk
        folder = tmp_path / "backtest" / "r" / "ACGL" / "TradingAgentsStrategy_logs"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"full_states_log_{day}.json").write_text(json.dumps({
            "company_of_interest": "ACGL", "final_rating": "Buy", "risk_assessment": f"**Stop Loss**: {stop}\n"}))
    fills = {"out1": {"filled_avg_price": 379.53}, "out2": {"filled_avg_price": 111.50},
             "out3": {"filled_avg_price": 106.0}, "out4": {"filled_avg_price": 97.0}}
    cfg = {"results_dir": str(tmp_path), "default_stop_pct": 2.0}

    rows = {r["ticker"]: r for r in transactions.transactions_on(s, "2026-11-09", fills, config=cfg)}

    assert (rows["GD"]["stop"], rows["GD"]["risk_pct"], rows["GD"]["r"]) == (350.0, 2.25, 2.67)           # +6% on a 2.25% stop
    assert (rows["TROW"]["stop"], rows["TROW"]["risk_pct"], rows["TROW"]["r"]) == (111.36, 2.07, -1.06)   # through the stop: a little over 1R
    assert (rows["ACGL"]["stop"], rows["ACGL"]["risk_pct"], rows["ACGL"]["r"]) == (96.0, 4.0, 1.5)        # a trim is scored like an exit
    assert (rows["LEN"]["stop"], rows["LEN"]["risk_pct"], rows["LEN"]["r"]) == (98.0, 2.0, -1.5)
    summary = transactions.realised_r(list(rows.values()))
    assert summary == {"n": 4, "total": 1.61, "mean": 0.4}
    text = transactions.summarize(list(rows.values()), "2026-11-09")
    assert "+1.61R over 4 exits, +0.40R each." in text and "(-2.2%, -1.06R on a 2.1% stop)" in text
    without = transactions.transactions_on(s, "2026-11-09", fills)                                        # no config: no R, as before
    assert all(r["r"] is None for r in without) and transactions.realised_r(without) is None


@pytest.mark.unit
def test_every_position_has_an_id_and_an_entry_date_and_its_exits_carry_them(tmp_path):
    """Q on 2026-11-13: a long closed by the review, then a short the desk never chose, closed at its target.
    Three exits in one ticker read as one position; they were two, one of them not the worker's."""
    from zoneinfo import ZoneInfo

    et = ZoneInfo("America/New_York")
    s = WorkerState.load(tmp_path / "state.json")
    s.record_order("Q", "open", "buy", 18, "in1", datetime(2026, 11, 12, 10, 30, tzinfo=et), "the day's book", stop=118.0)
    s.record_order("Q", "trim", "sell", 9, "t1", datetime(2026, 11, 12, 14, 0, tzinfo=et), "news", entry_price=124.34)
    assert s.positions["Q"] == {"id": "Q-20261112-L", "entered": "2026-11-12", "side": "long"}
    s.record_order("Q", "exit", "sell", 9, "out1", datetime(2026, 11, 13, 10, 30, tzinfo=et), "review", entry_price=124.34)
    assert "Q" not in s.positions                                                   # closed: the ID is spent
    s.record_order("Q", "exit", "buy", 18, "out2", datetime(2026, 11, 13, 10, 59, tzinfo=et), "target", entry_price=121.10)
    s.record_order("Q", "open", "buy", 5, "in2", datetime(2026, 11, 13, 11, 0, tzinfo=et), "the day's book")
    s.record_order("Q", "exit", "sell", 5, "out3", datetime(2026, 11, 13, 11, 30, tzinfo=et), "stop", entry_price=120.0)
    s.record_order("Q", "replace", "buy", 5, "in3", datetime(2026, 11, 13, 12, 0, tzinfo=et), "replaces X")
    s.save()

    tagged = [(o["order_id"], o["position_id"], o["entered"]) for o in WorkerState.load(tmp_path / "state.json").orders]
    assert tagged == [("in1", "Q-20261112-L", "2026-11-12"), ("t1", "Q-20261112-L", "2026-11-12"),
                      ("out1", "Q-20261112-L", "2026-11-12"),
                      ("out2", "Q-unrecorded-S", None),                              # a short the worker never opened
                      ("in2", "Q-20261113-L", "2026-11-13"), ("out3", "Q-20261113-L", "2026-11-13"),
                      ("in3", "Q-20261113-L-2", "2026-11-13")]                       # a second long in the name that day

    rows = {r["order_id"]: r for r in transactions_on(s, "2026-11-13")}
    assert (rows["out1"]["position_id"], rows["out1"]["carried"]) == ("Q-20261112-L", True)
    assert (rows["out3"]["position_id"], rows["out3"]["carried"]) == ("Q-20261113-L", False)
    assert rows["out2"]["carried"] is None
    text = summarize(list(rows.values()), "2026-11-13")
    assert "position Q-20261112-L, entered 2026-11-12 (carried over)" in text
    assert "position Q-unrecorded-S, no entry on the worker's record" in text


@pytest.mark.unit
def test_orders_from_before_positions_were_tagged_are_read_from_the_orders_before_them(tmp_path):
    """The record on Colossus has a week of untagged orders; an exit is still tied to its own entry, on its own side."""
    s = WorkerState.load(tmp_path / "state.json")
    at = lambda d, h, m: datetime(2026, 11, d, h, m, tzinfo=UTC).isoformat()
    s.orders = [
        {"symbol": "GLW", "kind": "open", "side": "sell", "qty": 25, "order_id": "in1", "at": at(10, 15, 30), "reason": "book", "stop": 118.0},
        {"symbol": "GLW", "kind": "exit", "side": "buy", "qty": 25, "order_id": "out1", "at": at(11, 14, 30), "reason": "review", "entry_price": 112.38},
        {"symbol": "GLW", "kind": "exit", "side": "buy", "qty": 25, "order_id": "out2", "at": at(11, 14, 31), "reason": "stop", "entry_price": 112.38},
        {"symbol": "GLW", "kind": "exit", "side": "sell", "qty": 25, "order_id": "out3", "at": at(11, 15, 33), "reason": "target", "entry_price": 119.70},
    ]
    fills = {"out1": {"filled_avg_price": 119.63}, "out2": {"filled_avg_price": 119.70}, "out3": {"filled_avg_price": 120.47}}
    rows = {r["order_id"]: r for r in transactions_on(s, "2026-11-11", fills, config={"default_stop_pct": 2.0})}

    assert (rows["out1"]["position_id"], rows["out1"]["entered"], rows["out1"]["carried"]) == ("GLW-20261110-S", "2026-11-10", True)
    assert rows["out2"]["position_id"] == "GLW-unrecorded-S" and rows["out2"]["entered"] is None      # the phantom: nothing left to close
    assert rows["out3"]["position_id"] == "GLW-unrecorded-L"                                          # the long it opened
    assert rows["out1"]["stop"] == 118.0                                # the short's own stop
    assert rows["out3"]["risk_pct"] == 2.0                              # not the short's 118 read as the long's (1.4%): the default


@pytest.mark.unit
def test_an_entry_says_how_old_the_desks_decision_was_and_its_exit_still_knows(tmp_path):
    """ROK on 2026-11-11: opened on a decision of 2026-11-06, five days old. Nothing in the record said so."""
    from zoneinfo import ZoneInfo

    from tradingagents.web.transactions import age_text, entry_ages

    et = ZoneInfo("America/New_York")
    s = WorkerState.load(tmp_path / "state.json")
    s.record_order("ROK", "replace", "sell", 4, "in1", datetime(2026, 11, 11, 12, 36, tzinfo=et), "replaces FDS", decided="2026-11-06")
    s.record_order("HAS", "open", "sell", 22, "in2", datetime(2026, 11, 11, 10, 30, tzinfo=et), "the day's book", decided="2026-11-10")
    s.record_order("OLD", "open", "buy", 5, "in3", datetime(2026, 11, 11, 10, 30, tzinfo=et), "the day's book")     # no decision kept
    s.record_order("ROK", "exit", "buy", 4, "out1", datetime(2026, 11, 12, 11, 0, tzinfo=et), "stop", entry_price=402.31)

    rows = {r["order_id"]: r for r in transactions_on(s, "2026-11-11")}
    assert (rows["in1"]["decided"], rows["in1"]["decision_age_days"], age_text(rows["in1"])) == ("2026-11-06", 5, "5 days old")
    assert (rows["in2"]["decision_age_days"], age_text(rows["in2"])) == (1, "1 day old")
    assert rows["in3"]["decided"] is None and age_text(rows["in3"]) == ""
    assert entry_ages(list(rows.values())) == {1: 1, 5: 1}
    assert "position ROK-20261111-S, entered 2026-11-11, on the decision of 2026-11-06 (5 days old at entry)" in summarize(list(rows.values()), "2026-11-11")

    (closed,) = transactions_on(s, "2026-11-12")
    assert (closed["decided"], closed["decision_age_days"], closed["carried"]) == ("2026-11-06", 5, True)
