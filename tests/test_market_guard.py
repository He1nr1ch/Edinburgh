"""The market guard: SPY read by arithmetic, and nothing new opened against a market that is running."""

import json
from datetime import UTC, date, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from tradingagents import market_guard
from tradingagents.dataflows.vendors.alpaca import OptionQuote
from tradingagents.dataflows.vendors.alpaca.trading import Account
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.hedge import (
    describe_hedge,
    hedge_row,
    is_hedge_row,
    kind_of,
    open_hedges,
    plan_hedge,
    record_hedge,
)
from tradingagents.market_guard import Tape, judge
from tradingagents.worker.duty import WorkerContext
from tradingagents.worker.state import WorkerState

# SPY's closes from 2026-11-10 to 11-19, the week the desk kept buying longs.
CLOSES = [749.92, 759.81, 757.12, 754.37, 739.95, 732.44, 730.20, 715.87]
FALLING = Tape(709.72, -0.9, -5.9, 5, -1)
RISING = Tape(790.0, 1.8, 3.4, 5, 1)


def _cfg(tmp_path, **more):
    return {**DEFAULT_CONFIG, "results_dir": str(tmp_path), "worker_state_path": str(tmp_path / "state.json"),
            "hedge_ledger_path": str(tmp_path / "hedges.json"), "feedback_db_url": "sqlite:///" + str(tmp_path / "feedback.db"),
            "worker_trading_enabled": True, "worker_allow_live": False, "market_guard": True, **more}


def _put(expiry, ask):
    return OptionQuote(f"SPY{expiry:%y%m%d}P00710000", expiry, 710.0, ask - 0.05, ask, None)


@pytest.mark.unit
def test_a_market_past_either_limit_is_running_and_one_inside_both_is_not():
    day = judge(739.95, CLOSES[:4], 1.5, 5, 3.0)                # 2026-11-16: -1.9% on the day, -1.3% over five
    assert day.falling and day.barred == "long" and day.day_pct == pytest.approx(-1.91, abs=0.01)
    week = judge(730.20, CLOSES[:6], 1.5, 5, 3.0)               # 2026-11-18: -0.3% on the day, -3.9% from the 11th's close
    assert week.falling and week.period_pct == pytest.approx(-3.90, abs=0.01)
    assert "falling" in week.describe() and "-3.9% over 5 sessions" in week.describe()

    calm = judge(756.0, CLOSES[:4], 1.5, 5, 3.0)
    assert calm.direction == 0 and calm.barred is None and calm.why_not("long") is None

    rally = judge(771.0, CLOSES[:4], 1.5, 5, 3.0)               # +2.2% on the day
    assert rally.barred == "short" and rally.why_not("long") is None
    assert "the market is rising" in rally.why_not("short") and rally.why_not("short").endswith("no new short")


@pytest.mark.unit
def test_a_bounce_inside_a_sell_off_is_neither_and_a_limit_of_nought_is_off():
    bounce = judge(728.0, CLOSES, 1.5, 5, 3.0)                  # +1.7% on the day, -3.5% over five
    assert bounce.day_pct > 1.5 and bounce.period_pct < -3.0 and bounce.direction == 0

    assert judge(700.0, CLOSES, 0, 5, 0).direction == 0         # both readings off
    assert judge(700.0, CLOSES, 0, 5, 3.0).falling              # the day's off, the week's on
    short_history = judge(752.0, CLOSES[:3], 1.5, 5, 3.0)       # three closes: no five-session reading
    assert short_history.period_pct is None and short_history.direction == 0


@pytest.mark.unit
def test_the_tape_is_read_from_spys_closed_sessions_and_its_live_price(monkeypatch):
    import tradingagents.dataflows.vendors.alpaca as alpaca

    asked = []
    index = pd.bdate_range("2026-11-10", periods=len(CLOSES))
    monkeypatch.setattr(alpaca, "get_closes", lambda s, a, b: asked.append((s, a, b)) or pd.Series(CLOSES, index=index))
    monkeypatch.setattr(alpaca, "latest_price", lambda s: 709.72)
    now = datetime(2026, 11, 20, 13, 53, tzinfo=UTC)            # 08:53 in New York

    tape = market_guard.read_tape({**DEFAULT_CONFIG, "market_guard": True}, now)
    assert asked == [("SPY", "2026-10-27", "2026-11-20")]       # up to, not including, today
    assert tape.falling and tape.price == 709.72
    assert tape.day_pct == pytest.approx(-0.86, abs=0.01) and tape.period_pct == pytest.approx(-5.92, abs=0.01)

    assert market_guard.read_tape({**DEFAULT_CONFIG, "market_guard": False}, now) is None

    # SPY out of reach: the desk trades as it did without a guard.
    monkeypatch.setattr(alpaca, "latest_price", lambda s: (_ for _ in ()).throw(OSError("no route")))
    assert market_guard.read_tape({**DEFAULT_CONFIG, "market_guard": True}, now) is None
    monkeypatch.setattr(alpaca, "latest_price", lambda s: None)
    assert market_guard.read_tape({**DEFAULT_CONFIG, "market_guard": True}, now) is None


def _book(tmp_path):
    sizing = tmp_path / "backtest" / "sizing"
    sizing.mkdir(parents=True)
    (sizing / "run_a_2026-11-05.csv").write_text(
        "ticker,side,rating,risk_level,ceiling_pct,weight_pct,price,shares,value,stop,stop_from,target\n"
        "GD,long,Overweight,Neutral,5,5.77,364.87,16,5838,344.41,Risk Manager,480\n"
        "FIS,short,Sell,Neutral,4,-4.62,29.76,-156,4643,31.68,Risk Manager,25\n")


def _placing(monkeypatch, tape):
    from tradingagents.worker import runner

    sent = []
    live = {"GD": 365.0, "FIS": 29.80}
    monkeypatch.setattr(runner.trading, "get_account", lambda: Account(100_000, 90_000, True, ()))
    monkeypatch.setattr(runner, "latest_price", lambda t: live.get(t))
    monkeypatch.setattr(runner.trading, "get_clock", lambda: MagicMock(is_open=True))
    monkeypatch.setattr(runner.trading, "open_orders", lambda: {})
    monkeypatch.setattr(runner.trading, "submit_market_order", lambda s, q, side: sent.append((s, q, side)) or f"ord-{s}")
    monkeypatch.setattr(runner, "send_email", lambda *a, **k: True)
    monkeypatch.setattr(runner, "read_tape", lambda config, now=None: tape)
    return runner, sent


@pytest.mark.unit
def test_the_book_opens_no_long_in_a_falling_market_and_no_short_in_a_rising_one(tmp_path, monkeypatch):
    _book(tmp_path)
    runner, sent = _placing(monkeypatch, FALLING)
    cfg = _cfg(tmp_path)
    assert runner.place_book(cfg) == ["FIS"] and sent == [("FIS", 155, "sell")]
    log = "\n".join(p.read_text() for p in Path(tmp_path).rglob("*.log"))
    assert "not placed: GD: the market is falling (SPY 709.72" in log and "no new long" in log

    other = tmp_path / "rising"
    _book(other)
    runner, sent = _placing(monkeypatch, RISING)
    assert runner.place_book(_cfg(other)) == ["GD"] and sent == [("GD", 15, "buy")]


@pytest.mark.unit
def test_the_book_opens_both_sides_when_the_market_is_not_running_or_cannot_be_read(tmp_path, monkeypatch):
    _book(tmp_path)
    runner, sent = _placing(monkeypatch, Tape(756.0, 0.2, -0.5, 5, 0))
    assert runner.place_book(_cfg(tmp_path)) == ["GD", "FIS"]

    other = tmp_path / "unread"
    _book(other)
    runner, sent = _placing(monkeypatch, None)
    assert runner.place_book(_cfg(other)) == ["GD", "FIS"]


def _bench_desk(tmp_path, monkeypatch, tape, bench_rows):
    from tradingagents.worker import replace

    sizing = tmp_path / "backtest" / "sizing"
    sizing.mkdir(parents=True)
    (sizing / "run_a_2026-11-05_bench.csv").write_text(
        "ticker,side,rating,risk_level,ceiling_pct,price,stop,stop_from,target,expected_move_pct\n" + bench_rows)
    live = {"NOW": 109.0, "KIM": 25.9, "PPL": 35.0}
    monkeypatch.setattr(replace, "latest_price", lambda t: live.get(t))
    monkeypatch.setattr(replace.trading, "open_orders", lambda: {})
    monkeypatch.setattr(replace, "read_tape", lambda config, now=None: tape)
    asked = []
    structured = MagicMock()
    structured.invoke.side_effect = lambda prompt: asked.append(prompt) or replace.ReplacementChoice(
        ticker="NOW", reason="the only call the market is not running against", confidence=0.7, evidence=["12.8% to target"])
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    cfg = _cfg(tmp_path)
    state = WorkerState.load(cfg["worker_state_path"])
    state.book = {"ADI": {"placed_on": "2026-11-06", "side": "long", "value": 4643.0}}
    account = Account(100_000, 0, True, ())                     # ADI is gone: its stop took it out
    monkeypatch.setattr(replace.trading, "get_account", lambda: account)
    return replace, WorkerContext(account, datetime(2026, 11, 6, 15, 0, tzinfo=UTC), cfg, state, llm), asked


LONGS = ("KIM,long,Overweight,Neutral,4,25.88,24.0,Risk Manager,30,15.9\n"
         "PPL,long,Overweight,Neutral,4,35.04,34.0,Risk Manager,40,14.2\n")
A_SHORT = "NOW,short,Underweight,Neutral,3,108.81,114.69,Risk Manager,95,12.7\n"


@pytest.mark.unit
def test_a_long_stopped_out_in_a_falling_market_is_not_replaced_with_another_long(tmp_path, monkeypatch):
    """2026-11-18: three stop-outs were each replaced with a long within minutes, on a bench of ten
    longs, while SPY stood 3.9% under its close of five sessions before. One of them, ADI, was
    stopped out the next day."""
    replace, ctx, asked = _bench_desk(tmp_path, monkeypatch, FALLING, LONGS)

    (finding,) = replace.ReplacementDuty().run(ctx)
    assert finding.action is None and asked == []               # nothing to choose from: the model is not asked
    assert finding.headline == "ADI left the book; no replacement on the bench"
    assert "the market is falling (SPY 709.72" in finding.detail and "no new long" in finding.detail
    assert "ADI" not in ctx.state.book                          # the slot lapses; tomorrow's book refills it


@pytest.mark.unit
def test_the_bench_still_fills_the_slot_from_the_side_the_market_is_not_running_against(tmp_path, monkeypatch):
    replace, ctx, asked = _bench_desk(tmp_path, monkeypatch, FALLING, LONGS + A_SHORT)

    (finding,) = replace.ReplacementDuty().run(ctx)
    assert (finding.action.symbol, finding.action.side) == ("NOW", "sell")
    assert "- NOW:" in asked[0] and "KIM" not in asked[0] and "PPL" not in asked[0]


@pytest.mark.unit
def test_the_bench_is_open_to_both_sides_when_the_market_is_not_running(tmp_path, monkeypatch):
    replace, ctx, asked = _bench_desk(tmp_path, monkeypatch, Tape(756.0, 0.2, -0.5, 5, 0), LONGS + A_SHORT)

    replace.ReplacementDuty().run(ctx)
    assert all(f"- {t}:" in asked[0] for t in ("NOW", "KIM", "PPL"))


WEEKLY_PUT = _put(date(2026, 11, 27), 8.10)


@pytest.mark.unit
def test_a_net_long_book_is_hedged_with_puts_only_while_the_market_falls():
    calm = plan_hedge(80_000, 709.72, WEEKLY_PUT, None)
    assert calm.contracts == 0 and "the market is not falling" in calm.reason and hedge_row(calm) is None

    plan = plan_hedge(80_000, 709.72, WEEKLY_PUT, _put(date(2026, 12, 18), 19.0), falling=True)
    assert (plan.contracts, plan.kind, plan.choice) == (1, "put", "weekly")        # 1.13 contracts; the month is 2.35x
    assert plan.cost == pytest.approx(810) and plan.coverage == pytest.approx(70_972 / 80_000)
    assert "the market is falling and the book is net long" in plan.reason
    row = hedge_row(plan)
    assert row[:2] == ["SPY261127P00710000", "put (weekly hedge)"] and is_hedge_row({"side": row[1]})
    assert describe_hedge(plan).startswith("buy 1 SPY261127P00710000 (weekly) at 8.10, 810 in all")

    small = plan_hedge(30_000, 709.72, WEEKLY_PUT, None, falling=True)               # 0.42 of a contract
    assert small.contracts == 0 and "under half a contract" in small.reason and "put would add more short" in small.reason
    assert plan_hedge(-80_000, 709.72, WEEKLY_PUT, None, falling=True).kind == "call"   # a net-short book is still hedged with calls


@pytest.mark.unit
def test_calls_held_do_not_stand_in_for_puts_and_puts_held_are_not_bought_again(tmp_path):
    ledger = tmp_path / "hedges.json"
    ledger.write_text(json.dumps([{"symbol": "SPY261127C00730000", "contracts": 1, "expiry": "2026-11-27",
                                   "bought_on": "2026-11-17", "cost": 900.0}]))
    assert kind_of("SPY261127C00730000") == "call" and kind_of("SPY261127P00710000") == "put"

    first = plan_hedge(80_000, 709.72, WEEKLY_PUT, None, held=open_hedges(ledger, "2026-11-20"), falling=True)
    assert first.contracts == 1 and first.held == ()            # the call covers a rally, not this fall
    record_hedge(ledger, first, "2026-11-20")

    again = plan_hedge(80_000, 709.72, WEEKLY_PUT, None, held=open_hedges(ledger, "2026-11-23"), falling=True)
    assert again.contracts == 0 and "already hedged: 1 SPY261127P00710000" in again.reason
    assert "the puts held cover 89%" in describe_hedge(again)


@pytest.mark.unit
def test_the_hedge_asks_for_put_quotes_for_a_net_long_book_in_a_falling_market(monkeypatch):
    import tradingagents.dataflows.vendors.alpaca as alpaca
    from tradingagents import hedge
    from tradingagents.sizing import Book, Order

    asked = []
    monkeypatch.setattr(alpaca, "latest_price", lambda s: 709.72)
    monkeypatch.setattr(alpaca, "get_closes", lambda s, a, b: pd.Series([720.0, 715.87], index=pd.to_datetime(["2026-11-18", "2026-11-19"])))
    monkeypatch.setattr(alpaca, "atm_put", lambda u, price, d: asked.append(("put", price)) or _put(d, 8.10))
    monkeypatch.setattr(alpaca, "atm_call", lambda u, price, d: asked.append(("call", price)) or None)
    book = Book(200_000, 75, 10, [Order("T", "Buy", "Neutral", 5, 40, 20.0, 4_000)])   # too little history: beta 1

    plan = hedge.hedge_book(book, "2026-11-19", falling=True)
    assert asked == [("put", 709.72), ("put", 709.72)] and (plan.kind, plan.contracts) == ("put", 1)

    asked.clear()
    assert hedge.hedge_book(book, "2026-11-19").contracts == 0 and {k for k, _ in asked} == {"call"}


@pytest.mark.unit
def test_the_puts_of_the_book_are_bought_though_calls_are_still_held(tmp_path, monkeypatch):
    sizing = tmp_path / "backtest" / "sizing"
    sizing.mkdir(parents=True)
    (sizing / "run_a_2026-11-05.csv").write_text(
        "ticker,side,rating,risk_level,ceiling_pct,weight_pct,price,shares,value,stop,stop_from,target\n"
        "SPY261127P00710000,put (weekly hedge),,,,,8.1,1,810,,,\n")
    cfg = _cfg(tmp_path)
    Path(cfg["hedge_ledger_path"]).write_text(json.dumps([{"symbol": "SPY261231C00730000", "contracts": 1,
                                                           "expiry": "2099-12-31", "bought_on": "2026-11-17", "cost": 900.0}]))
    runner, sent = _placing(monkeypatch, FALLING)

    assert runner.place_book(cfg) == ["SPY261127P00710000"] and sent == [("SPY261127P00710000", 1, "buy")]
    assert [e["symbol"] for e in json.loads(Path(cfg["hedge_ledger_path"]).read_text())] == ["SPY261231C00730000", "SPY261127P00710000"]
