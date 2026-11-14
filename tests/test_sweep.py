"""The continuous sweep: holdings first, a pool of processes sized for the session, none waiting for another."""

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
def test_a_cycle_starts_with_the_holdings_then_the_least_recently_analysed(cfg):
    s = sw.Sweep(cfg)
    s.tick("2026-11-09", market_open=False, holdings=["T05", "XYZ"], last_analysed={"T00": "2026-11-08", "T01": "2026-11-01"})

    assert s.cycle == 1 and s.round_no == 3 and len(FakeProc.live) == 3          # three processes outside market hours
    assert FakeProc.live[0].tickers() == ["T05", "XYZ"]                           # the holdings are the first run's
    assert {t for p in FakeProc.live[1:] for t in p.tickers()} == {"T02", "T03", "T04", "T06"}   # never analysed next; T01, T00 later
    assert FakeProc.live[0].kw["preexec_fn"] is not None                         # the sweep yields the CPU to the worker
    assert FakeProc.live[0].kw["env"]["TRADINGAGENTS_RESULTS_DIR"] == cfg["results_dir"]   # the child writes where the worker reads
    ids = [p.cmd[p.cmd.index("--run-id") + 1] for p in FakeProc.live]
    assert ids == ["sp500_20261109_c1r1_a", "sp500_20261109_c1r2_b", "sp500_20261109_c1r3_c"]
    assert len(s.queue) == 7 and s.queue[-2:] == ["T01", "T00"]                   # least recent first, newest last

    assert not s.due(market_open=False)
    s.tick("2026-11-09", False, ["T05", "XYZ"], {})                               # the pool is full: nothing new
    assert s.round_no == 3 and len(FakeProc.live) == 3


@pytest.mark.unit
def test_a_process_that_ends_is_replaced_at_once_and_does_not_wait_for_the_others(cfg):
    """In rounds, a round ended when its slowest process did: each process stood idle six minutes in fifty-four."""
    s = sw.Sweep(cfg)
    s.tick("2026-11-09", False, [], {})
    FakeProc.live[1].finish()                                                     # one of three is done; two still at work
    assert s.due(market_open=False)
    s.tick("2026-11-09", False, [], {})

    assert len(FakeProc.live) == 4 and len(s.runs) == 3 and s.analysed_this_cycle == 2
    assert FakeProc.live[3].cmd[FakeProc.live[3].cmd.index("--run-id") + 1] == "sp500_20261109_c1r4_b"   # the place it left
    assert FakeProc.live[0].poll() is None and FakeProc.live[2].poll() is None    # the others were not touched
    saved = json.loads(open(cfg["sweep_state_path"]).read())
    assert [r["round"] for r in saved["rounds"]] == ["c1r2"] and saved["rounds"][0]["parts"] == 3


@pytest.mark.unit
def test_during_market_hours_the_pool_is_smaller(cfg):
    s = sw.Sweep(cfg)
    s.tick("2026-11-09", market_open=True, holdings=[], last_analysed={})
    assert len(FakeProc.live) == 1 and len(FakeProc.live[0].tickers()) == 2       # one process, sweep_round_size tickers
    s.tick("2026-11-09", market_open=False, holdings=[], last_analysed={})       # the close: two more join it at once
    assert len(s.runs) == 3 and len(FakeProc.live) == 3
    for p in FakeProc.live:
        p.finish()
    s.tick("2026-11-09", market_open=True, holdings=[], last_analysed={})        # the open: back to one
    assert len(s.runs) == 1


@pytest.mark.unit
def test_a_new_close_restarts_the_cycle_on_the_holdings_and_an_exhausted_list_goes_round_again(cfg):
    s = sw.Sweep(cfg)
    s.tick("2026-11-09", False, ["T07"], {})
    FakeProc.live[0].finish()
    s.tick("2026-11-10", False, ["T07"], {})                                     # the close moved on, two runs still on the old one
    assert s.cycle == 2 and s.date == "2026-11-10" and s.round_no == 1
    assert FakeProc.live[-1].tickers()[0] == "T07" and "20261110_c2r1" in FakeProc.live[-1].cmd[-1]
    for p in FakeProc.live:
        p.finish()
    s.tick("2026-11-10", False, ["T07"], {})
    assert s.analysed_this_cycle == 2                                             # the old close's runs count for no cycle
    s.queue = []                                                                  # pretend the list ran out
    s.tick("2026-11-10", False, ["T07"], {})
    assert s.cycle == 2                                                           # runs still at work: the cycle is not over
    for p in FakeProc.live:
        p.finish()
    s.tick("2026-11-10", False, ["T07"], {})
    assert s.cycle == 3                                                           # round again, same close


@pytest.mark.unit
def test_a_restart_requeues_the_interrupted_runs(cfg):
    s = sw.Sweep(cfg)
    s.tick("2026-11-09", False, ["T05"], {})
    in_flight = [t for p in FakeProc.live for t in p.tickers()]
    saved = json.loads(open(cfg["sweep_state_path"]).read())
    assert saved["runs"][0]["run_id"] == "sp500_20261109_c1r1_a"

    again = sw.Sweep(cfg)                                                         # the worker came back; the runs' processes did not
    assert again.runs == [] and again.queue[:6] == in_flight
    again.tick("2026-11-09", False, ["T05"], {})
    assert again.round_no == 6 and again.cycle == 1                               # the same cycle carries on

    # A state file from before the pool, with its one round in flight, is taken up the same way.
    old = {**saved, "round": {"run_ids": ["x"], "tickers": ["T09", "T10"], "parts": 3, "started": "s"}, "queue": ["T11"]}
    del old["runs"]
    open(cfg["sweep_state_path"], "w").write(json.dumps(old))
    assert sw.Sweep(cfg).queue == ["T09", "T10", "T11"]


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
    monkeypatch.setattr(bk, "last_closes", lambda tickers, date: {"GD": 364.0, "NKE": 40.5, "ADBE": 240.0})
    cfg = {**DEFAULT_CONFIG, "results_dir": str(tmp_path), "daily_hedge": False, "decision_max_age_days": 5}

    built = daily.build_morning_book(cfg, 100_000, date="2026-11-09")
    assert sorted(o.ticker for o in built.book.orders) == ["GD", "NKE"]
    assert {o.decided for o in built.book.orders} == {"2026-11-09"}                      # each order knows its decision's date
    assert all(r["decided"] == "2026-11-09" for r in bk.load_rows(built.book_path)) if built.book_path else True
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
