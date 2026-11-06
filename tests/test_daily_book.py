"""The daily job and the book it leaves: the bench, placing the book, taking profit, replacing a position."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tradingagents.dataflows.vendors.alpaca.trading import Account, Holding
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.sizing import Call, bench_for
from tradingagents.worker.duty import WorkerContext
from tradingagents.worker.state import WorkerState


def _cfg(tmp_path, **more):
    return {**DEFAULT_CONFIG, "results_dir": str(tmp_path), "worker_state_path": str(tmp_path / "state.json"),
            "worker_trading_enabled": True, "worker_allow_live": False, **more}


def _decision(results_dir, run, ticker, date, rating, stop, target, ceiling=5):
    folder = Path(results_dir) / "backtest" / run / ticker / "TradingAgentsStrategy_logs"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"full_states_log_{date}.json").write_text(json.dumps({
        "company_of_interest": ticker, "final_rating": rating, "final_risk_level": "Neutral",
        "risk_assessment": f"**Maximum Position**: {ceiling}\n\n**Stop Loss**: {stop}\n",
        "final_trade_decision": f"**Rating**: {rating}\n\n**Price Target**: {target}\n",
    }))


@pytest.mark.unit
def test_the_bench_is_the_next_best_after_the_book_long_and_short_together():
    calls = [Call("A", "Overweight", 5, target=130.0), Call("B", "Overweight", 5, target=112.0),
             Call("C", "Underweight", 5, target=80.0), Call("D", "Underweight", 5, target=96.0),
             Call("E", "Hold", 5, target=140.0), Call("F", "Sell", 5, target=103.0)]   # Hold: no side; F: wrong side
    prices = {t: 100.0 for t in "ABCDEF"}
    assert [c.ticker for c in bench_for(calls, prices, chosen={"A"}, min_target_pct=5, n=10)] == ["C", "B"]   # -20%, +12%; D is under 5%


@pytest.mark.unit
def test_build_and_save_the_book_with_its_bench(tmp_path, monkeypatch):
    from tradingagents import book as bk

    for t, rating, stop, target in (("A", "Overweight", 94, 130), ("B", "Overweight", 95, 112), ("C", "Underweight", 104, 80),
                                    ("D", "Underweight", 103, 90), ("E", "Overweight", 96, 101)):
        _decision(tmp_path, "run_a", t, "2026-11-05", rating, stop, target)
    monkeypatch.setattr(bk, "last_closes", lambda tickers, date: {t: 100.0 for t in tickers})
    cfg = _cfg(tmp_path, names_per_side=1, bench_size=10)

    built = bk.save_book(bk.build_book(["run_a"], "2026-11-05", cfg, 100_000), cfg)
    assert sorted(o.ticker for o in built.book.orders) == ["A", "C"]          # the top long and the top short
    assert [b["ticker"] for b in built.bench] == ["B", "D"]                   # E's +1% is under the floor
    assert built.bench[0]["expected_move_pct"] == 12.0 and built.bench[1]["stop"] == 103.0
    assert built.book_path.name == "run_a_2026-11-05.csv" and built.bench_path.name == "run_a_2026-11-05_bench.csv"
    assert bk.latest_bench(tmp_path)[1][0]["ticker"] == "B"
    from tradingagents.web.data import latest_book
    assert latest_book(tmp_path)[0] == built.book_path                         # the page never takes the bench for the book


@pytest.mark.unit
def test_the_target_guard_takes_the_profit(tmp_path):
    from tradingagents.worker.stops import TargetDuty

    _decision(tmp_path, "run_a", "GD", "2026-11-05", "Overweight", 344.41, 480.0)
    _decision(tmp_path, "run_a", "FIS", "2026-11-05", "Sell", 31.68, 25.0)
    gd_short_of_target = Holding("GD", 16, 16 * 470.0, 364.87, "us_equity")
    gd_at_target = Holding("GD", 16, 16 * 481.0, 364.87, "us_equity")
    fis_covered = Holding("FIS", -156, -156 * 24.5, 29.76, "us_equity")
    ctx = WorkerContext(Account(100_000, 0, True, (gd_short_of_target, gd_at_target, fis_covered)), datetime.now(UTC), _cfg(tmp_path), None)

    findings = TargetDuty().run(ctx)
    assert [(f.symbol, f.action.side, f.action.qty) for f in findings] == [("GD", "sell", 16), ("FIS", "buy", 156)]
    assert "+31.8%" in findings[0].detail


@pytest.mark.unit
def test_placing_the_book_at_the_open_once_and_only_what_is_not_held(tmp_path, monkeypatch):
    from tradingagents.worker import runner

    sizing = tmp_path / "backtest" / "sizing"
    sizing.mkdir(parents=True)
    (sizing / "run_a_2026-11-05.csv").write_text(
        "ticker,side,rating,risk_level,ceiling_pct,weight_pct,price,shares,value,stop,stop_from,target\n"
        "GD,long,Overweight,Neutral,5,5.77,364.87,16,5838,344.41,Risk Manager,480\n"
        "FIS,short,Sell,Neutral,4,-4.62,29.76,-156,4643,31.68,Risk Manager,25\n"
        "SPY261204C00740000,call (monthly hedge),,,,,16.52,1,1652,,,\n")
    sent = []
    monkeypatch.setattr(runner.trading, "get_account", lambda: Account(100_000, 90_000, True, (Holding("GD", 16, 5_900.0, 364.87, "us_equity"),)))
    monkeypatch.setattr(runner.trading, "get_clock", lambda: MagicMock(is_open=True))
    monkeypatch.setattr(runner.trading, "open_orders", lambda: {})
    monkeypatch.setattr(runner.trading, "submit_market_order", lambda s, q, side: sent.append((s, q, side)) or f"ord-{s}")
    monkeypatch.setattr(runner, "send_email", lambda *a, **k: True)
    cfg = _cfg(tmp_path)

    assert runner.place_book(cfg) == ["FIS"]                   # GD is held already; the hedge row is not a stock
    assert sent == [("FIS", 156, "sell")]
    state = WorkerState.load(cfg["worker_state_path"])
    assert state.book["FIS"] == {"placed_on": datetime.now().strftime("%Y-%m-%d"), "side": "short", "value": 4643.0}
    assert runner.place_book(cfg) == []                        # once a day


@pytest.mark.unit
def test_a_book_position_that_sold_out_is_replaced_from_the_bench_by_the_watchers_choice(tmp_path, monkeypatch):
    from tradingagents.worker import replace

    sizing = tmp_path / "backtest" / "sizing"
    sizing.mkdir(parents=True)
    (sizing / "run_a_2026-11-05_bench.csv").write_text(
        "ticker,side,rating,risk_level,ceiling_pct,price,stop,stop_from,target,expected_move_pct\n"
        "NOW,short,Underweight,Neutral,3,108.81,114.69,Risk Manager,95,12.7\n"
        "KIM,long,Overweight,Neutral,4,25.88,25.0,Risk Manager,30,15.9\n"
        "DOW,short,Underweight,Conservative,2,32.70,33.08,Risk Manager,26.82,18.0\n")
    live = {"NOW": 109.0, "KIM": 29.5, "DOW": 33.5}            # KIM has run: +1.7% left; DOW is through its stop
    monkeypatch.setattr(replace, "latest_price", lambda t: live.get(t))
    monkeypatch.setattr(replace.trading, "open_orders", lambda: {})
    captured = {}
    structured = MagicMock()
    structured.invoke.side_effect = lambda prompt: captured.__setitem__("prompt", prompt) or replace.ReplacementChoice(
        ticker="NOW", reason="largest move left and keeps a short slot short")
    llm = MagicMock(); llm.with_structured_output.return_value = structured
    cfg = _cfg(tmp_path)
    state = WorkerState.load(cfg["worker_state_path"])
    state.book = {"FIS": {"placed_on": "2026-11-06", "side": "short", "value": 4643.0}, "GD": {"placed_on": "2026-11-06", "side": "long", "value": 5838.0}}
    state.record_order("FIS", "exit", "buy", 156, "ord-1", datetime.now(UTC), "price 31.90 is through the stop 31.68 (Risk Manager)")
    account = Account(100_000, 0, True, (Holding("GD", 16, 5_900.0, 364.87, "us_equity"),))   # FIS is gone
    ctx = WorkerContext(account, datetime(2026, 11, 6, 15, 0, tzinfo=UTC), cfg, state, llm)

    (finding,) = replace.ReplacementDuty().run(ctx)
    assert "closed because price 31.90 is through the stop" in captured["prompt"]
    assert "- NOW:" in captured["prompt"] and "KIM" not in captured["prompt"] and "DOW" not in captured["prompt"]
    assert (finding.action.symbol, finding.action.kind, finding.action.side, finding.action.qty) == ("NOW", "replace", "sell", 42)   # 4,643 / 109
    assert finding.action.value == pytest.approx(42 * 109.0)
    assert "FIS" not in state.book and state.day("2026-11-06")["replaced"] == {"FIS": "NOW"}


@pytest.mark.unit
def test_the_daily_job_is_due_once_after_its_time_on_weekdays(tmp_path):
    from zoneinfo import ZoneInfo

    from tradingagents.worker.runner import daily_job_due

    cfg = _cfg(tmp_path, daily_analysis_time="04:00")
    state = WorkerState.load(cfg["worker_state_path"])
    et = ZoneInfo("America/New_York")
    assert not daily_job_due(cfg, datetime(2026, 11, 6, 3, 59, tzinfo=et), state)     # Friday, too early
    assert daily_job_due(cfg, datetime(2026, 11, 6, 4, 0, tzinfo=et), state)
    assert not daily_job_due(cfg, datetime(2026, 11, 7, 9, 0, tzinfo=et), state)      # Saturday
    state.day("2026-11-06")["analysis"] = "done"
    assert not daily_job_due(cfg, datetime(2026, 11, 6, 12, 0, tzinfo=et), state)


@pytest.mark.unit
def test_the_sweep_runs_the_parts_as_processes_and_reports_the_ones_that_finished(tmp_path, monkeypatch):
    from tradingagents import daily

    launched = []

    class Proc:
        def __init__(self, cmd, **kw):
            launched.append(cmd); self.rc = 1 if "c" in cmd[cmd.index("--run-id") + 1][-1:] else 0
        def wait(self):
            return self.rc

    monkeypatch.setattr(daily.subprocess, "Popen", Proc)
    done = daily.run_sweep([f"T{i}" for i in range(10)], "2026-11-05", 3, tmp_path / "logs")
    assert done == ["sp500_20261105_daily_a", "sp500_20261105_daily_b"]
    assert launched[0][launched[0].index("backtest") + 1] == "T0,T1,T2,T3" and launched[2][-1] == "sp500_20261105_daily_c"
    assert (tmp_path / "logs" / "sp500_20261105_daily_c.log").exists()


@pytest.mark.unit
def test_the_job_times_itself_by_part_and_in_all(tmp_path, monkeypatch):
    from tradingagents import daily

    def fake_sweep(tickers, date, parts, log_dir, tag="daily", analysts=None, timing=None):
        timing["parts"] = {"sp500_x_daily_a": {"tickers": 2, "seconds": 61.0, "exit": 0},
                           "sp500_x_daily_b": {"tickers": 1, "seconds": 45.5, "exit": 0}}
        return ["run_a"]

    _decision(tmp_path, "run_a", "A", "2026-11-05", "Overweight", 94, 130)
    monkeypatch.setattr(daily, "run_sweep", fake_sweep)
    import tradingagents.book as bk
    monkeypatch.setattr(bk, "last_closes", lambda tickers, date: {t: 100.0 for t in tickers})
    watch = tmp_path / "w.txt"; watch.write_text("A,B,C\n")
    cfg = _cfg(tmp_path, daily_watchlist=str(watch), daily_parts=2)

    built = daily.run_daily(cfg, 100_000, date="2026-11-05", refresh_watchlist=False)
    t = json.loads(built.timing_path.read_text())
    assert t["tickers"] == 3 and t["orders"] == 1 and t["total_seconds"] >= t["sweep_seconds"] >= 0
    assert t["parts"]["sp500_x_daily_b"]["seconds"] == 45.5
    line = daily.describe_timing(t)
    assert line.startswith("3 tickers in 0:00:") and "a 2 in 0:01:01, b 1 in 0:00:45" in line
    assert built.timing_path.name.endswith("_2026-11-05_timing.json")


@pytest.mark.unit
def test_the_default_analysis_time_is_three_in_the_morning():
    assert DEFAULT_CONFIG["daily_analysis_time"] == "03:00"
    assert DEFAULT_CONFIG["daily_watchlist"].endswith("sp500.txt")
