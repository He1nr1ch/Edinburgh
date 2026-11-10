"""The continuous sweep: holdings first, rounds sized for the session, the Watcher's nodes kept free."""

from __future__ import annotations

import json

import pytest

from tradingagents import sweep as sw
from tradingagents.default_config import DEFAULT_CONFIG


class FakeProc:
    live: list = []

    def __init__(self, cmd, **kw):
        self.cmd, self.kw, self.rc = cmd, kw, None
        FakeProc.live.append(self)

    def poll(self):
        return self.rc

    @property
    def returncode(self):
        return self.rc

    def terminate(self):
        self.rc = -15

    def finish(self, rc=0):
        self.rc = rc

    def tickers(self):
        return self.cmd[self.cmd.index("backtest") + 1].split(",")


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    FakeProc.live = []
    monkeypatch.setattr(sw.subprocess, "Popen", FakeProc)
    watch = tmp_path / "sp500.txt"
    watch.write_text(",".join(f"T{i:02d}" for i in range(12)) + "\n")
    return {**DEFAULT_CONFIG, "results_dir": str(tmp_path), "sweep_state_path": str(tmp_path / "sweep.json"),
            "daily_watchlist": str(watch), "daily_parts": 3, "daily_parts_market_hours": 1, "sweep_round_size": 2}


@pytest.mark.unit
def test_a_cycle_starts_with_the_holdings_then_the_least_recently_analysed_in_rounds(cfg):
    s = sw.Sweep(cfg)
    s.tick("2026-11-09", market_open=False, holdings=["T05", "XYZ"], last_analysed={"T00": "2026-11-08", "T01": "2026-11-01"})

    assert s.cycle == 1 and s.round_no == 1 and len(FakeProc.live) == 3          # three processes outside market hours
    first_round = [t for p in FakeProc.live for t in p.tickers()]
    assert set(first_round) == {"T05", "XYZ", "T02", "T03", "T04", "T06"}        # holdings first, never-analysed next; T01, T00 later
    assert FakeProc.live[0].tickers()[0] == "T05"                                 # every process starts on a holding
    assert FakeProc.live[0].kw["preexec_fn"] is not None                         # the sweep yields the CPU to the worker
    assert FakeProc.live[0].kw["env"]["TRADINGAGENTS_RESULTS_DIR"] == cfg["results_dir"]   # the child writes where the worker reads
    assert FakeProc.live[0].cmd[FakeProc.live[0].cmd.index("--run-id") + 1] == "sp500_20261109_c1r1_a"
    assert len(s.queue) == 7 and s.queue[-2:] == ["T01", "T00"]                   # least recent first, newest last

    s.tick("2026-11-09", False, ["T05", "XYZ"], {})                               # the round is still running: nothing new
    assert s.round_no == 1 and len(FakeProc.live) == 3


@pytest.mark.unit
def test_during_market_hours_a_round_runs_on_one_process(cfg):
    s = sw.Sweep(cfg)
    s.tick("2026-11-09", market_open=True, holdings=[], last_analysed={})
    assert len(FakeProc.live) == 1 and len(FakeProc.live[0].tickers()) == 2       # one process, round_size tickers
    for p in FakeProc.live:
        p.finish()
    s.tick("2026-11-09", market_open=False, holdings=[], last_analysed={})       # the close: back to three
    assert s.round_no == 2 and len(FakeProc.live) == 4


@pytest.mark.unit
def test_a_new_close_restarts_the_cycle_on_the_holdings_and_an_exhausted_list_goes_round_again(cfg):
    s = sw.Sweep(cfg)
    s.tick("2026-11-09", False, ["T07"], {})
    for p in FakeProc.live:
        p.finish()
    s.tick("2026-11-10", False, ["T07"], {})                                     # the close moved on
    assert s.cycle == 2 and s.date == "2026-11-10" and s.round_no == 1
    assert FakeProc.live[-3].tickers()[0] == "T07"
    s.queue = []                                                                  # pretend the list ran out
    for p in FakeProc.live:
        p.finish()
    s.tick("2026-11-10", False, ["T07"], {})
    assert s.cycle == 3                                                           # round again, same close


@pytest.mark.unit
def test_a_restart_requeues_the_interrupted_round(cfg):
    s = sw.Sweep(cfg)
    s.tick("2026-11-09", False, ["T05"], {})
    in_flight = sorted(t for p in FakeProc.live for t in p.tickers())
    saved = json.loads((cfg["sweep_state_path"] and open(cfg["sweep_state_path"])).read())
    assert saved["round"]["run_ids"][0] == "sp500_20261109_c1r1_a"

    again = sw.Sweep(cfg)                                                         # the process came back; the round's processes did not
    assert again.round is None and again.queue[:len(in_flight)] == sorted(in_flight, key=in_flight.index) or set(again.queue[:6]) == set(in_flight)
    again.tick("2026-11-09", False, ["T05"], {})
    assert again.round_no == 2 and again.cycle == 1                               # the same cycle carries on


@pytest.mark.unit
def test_the_book_is_due_at_its_time_on_a_day_the_market_opens_and_once(tmp_path):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from tradingagents.worker.runner import book_due
    from tradingagents.worker.state import WorkerState

    cfg = {**DEFAULT_CONFIG, "daily_book_time": "09:00", "worker_state_path": str(tmp_path / "s.json")}
    state = WorkerState.load(cfg["worker_state_path"])
    et = ZoneInfo("America/New_York")
    opens = datetime(2026, 11, 10, 9, 30, tzinfo=et)
    assert not book_due(cfg, datetime(2026, 11, 10, 8, 59, tzinfo=et), state, opens)
    assert book_due(cfg, datetime(2026, 11, 10, 9, 0, tzinfo=et), state, opens)
    assert not book_due(cfg, datetime(2026, 11, 8, 9, 0, tzinfo=et), state, datetime(2026, 11, 9, 9, 30, tzinfo=et))   # Sunday
    state.day("2026-11-10")["analysis"] = "book built 09:00"
    assert not book_due(cfg, datetime(2026, 11, 10, 9, 5, tzinfo=et), state, opens)
    state.day("2026-11-10")["analysis"] = "failed: Alpaca down"
    assert book_due(cfg, datetime(2026, 11, 10, 9, 5, tzinfo=et), state, opens)      # a failure is retried
    state.day("2026-11-10")["analysis"] = "running"                                    # the old nightly job's marker
    assert book_due(cfg, datetime(2026, 11, 10, 9, 5, tzinfo=et), state, opens)


@pytest.mark.unit
def test_the_morning_book_is_built_from_the_fresh_decisions(tmp_path, monkeypatch):
    import tradingagents.book as bk
    from tradingagents import daily

    def decision(run, ticker, date, rating, stop, target):
        folder = tmp_path / "backtest" / run / ticker / "TradingAgentsStrategy_logs"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"full_states_log_{date}.json").write_text(json.dumps({
            "company_of_interest": ticker, "final_rating": rating, "final_risk_level": "Neutral",
            "risk_assessment": f"**Maximum Position**: 5\n\n**Stop Loss**: {stop}\n",
            "final_trade_decision": f"**Rating**: {rating}\n\n**Price Target**: {target}\n"}))

    decision("c1r1_a", "GD", "2026-11-09", "Overweight", 345, 386)
    decision("c1r2_b", "NKE", "2026-11-09", "Underweight", 41.5, 36.91)
    decision("old", "ADBE", "2026-11-01", "Underweight", 250, 207)              # 8 days old: out of the window
    monkeypatch.setattr(bk, "last_closes", lambda tickers, date: {t: 100.0 for t in tickers})
    cfg = {**DEFAULT_CONFIG, "results_dir": str(tmp_path), "daily_hedge": False, "decision_max_age_days": 5}

    built = daily.build_morning_book(cfg, 100_000, date="2026-11-09")
    assert sorted(o.ticker for o in built.book.orders) == ["GD", "NKE"]
    assert built.timing["calls_in_window"] == 2 and built.timing["decided_between"] == ["2026-11-09", "2026-11-09"]
    assert built.book_path.name.startswith("fresh_20261109_2026-11-09")
    assert "book from 2 decisions dated 2026-11-09 to 2026-11-09" in daily.describe_timing(built.timing)
    with pytest.raises(RuntimeError):
        daily.build_morning_book({**cfg, "decision_max_age_days": 0}, 100_000, date="2026-10-01")


@pytest.mark.unit
def test_the_watchlist_default_exists_and_the_sweep_seeds_the_file_when_missing(tmp_path, monkeypatch):
    """The first deploy of the continuous sweep stalled on a missing 'daily_watchlist' key."""
    assert DEFAULT_CONFIG["daily_watchlist"].endswith("/watchlists/sp500.txt")
    from tradingagents import daily, sp500

    repo_copy = tmp_path / "repo" / "sp500.txt"
    repo_copy.parent.mkdir()
    repo_copy.write_text("AAA,BBB\n")
    monkeypatch.setattr(sp500, "WATCHLIST", repo_copy)
    monkeypatch.setattr(sp500, "refresh_if_stale", lambda path=None: None)
    FakeProc.live = []
    monkeypatch.setattr(sw.subprocess, "Popen", FakeProc)
    cfg = {**DEFAULT_CONFIG, "results_dir": str(tmp_path), "sweep_state_path": str(tmp_path / "sweep.json"),
           "daily_watchlist": str(tmp_path / "data" / "sp500.txt"), "daily_parts": 1, "sweep_round_size": 4}
    s = sw.Sweep(cfg)
    s.tick("2026-11-10", market_open=False, holdings=[], last_analysed={})
    assert (tmp_path / "data" / "sp500.txt").read_text() == "AAA,BBB\n"
    assert FakeProc.live[0].tickers() == ["AAA", "BBB"]
    assert daily.ensure_watchlist(cfg, refresh=False) == tmp_path / "data" / "sp500.txt"
