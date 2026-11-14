"""Token accounting, the sweep's progress estimate, and PAIR's node shares."""

from __future__ import annotations

import json
import time

import pytest

from tradingagents import pair, usage
from tradingagents import sweep as sw


@pytest.fixture
def usage_dir(tmp_path, monkeypatch):
    from tradingagents.dataflows import config as cfg

    monkeypatch.setattr(cfg, "get_config", lambda: {"results_dir": str(tmp_path), "usage_dir": ""})
    usage.tally.reset()
    return tmp_path / "usage"


def _response(tokens_in=1200, tokens_out=300, total_ns=12e9, eval_ns=9e9):
    from langchain_core.messages import AIMessage

    return AIMessage("ok", usage_metadata={"input_tokens": tokens_in, "output_tokens": tokens_out, "total_tokens": tokens_in + tokens_out},
                     response_metadata={"model": "qwen3.8:27b", "total_duration": total_ns, "eval_duration": eval_ns,
                                        "prompt_eval_duration": 2e9})


@pytest.mark.unit
def test_every_answered_call_is_recorded_with_its_agent_ticker_and_run(usage_dir):
    usage.set_run(ticker="NVDA", run_id="sp500_20261110_c2r4_a")
    # Inside the graph: LangGraph names the node, the outermost one when nested.
    rec = usage.record_response(_response(), "qwen3.8:27b",
                                {"metadata": {"langgraph_node": "agent", "langgraph_checkpoint_ns": "Market Analyst:abc|agent:def"}})
    assert rec["agent"] == "Market Analyst" and rec["ticker"] == "NVDA" and rec["run_id"] == "sp500_20261110_c2r4_a"
    assert rec["tokens_in"] == 1200 and rec["tokens_out"] == 300 and rec["seconds"] == 12.0 and rec["tokens_per_s"] == pytest.approx(33.3)
    # Outside the graph: the structured caller names itself.
    with usage.as_agent("Watcher"):
        rec2 = usage.record_response(_response(500, 50), "qwen3.8:27b", None)
    assert rec2["agent"] == "Watcher"
    # A response without usage is not a record.
    from langchain_core.messages import AIMessage

    assert usage.record_response(AIMessage("bare"), "qwen3.8:27b", None) is None

    files = list(usage_dir.glob("*.jsonl"))
    assert len(files) == 1 and len(files[0].read_text().splitlines()) == 2
    snap = usage.tally.snapshot()
    assert snap["total"] == {"calls": 2, "tokens_in": 1700, "tokens_out": 350, "seconds": 24.0}
    assert set(snap["by_agent"]) == {"Market Analyst", "Watcher"}


@pytest.mark.unit
def test_the_ollama_client_records_what_it_is_answered(usage_dir, monkeypatch):
    from langchain_ollama import ChatOllama

    from tradingagents.llm_clients.ollama_client import NormalizedChatOllama

    monkeypatch.setattr(ChatOllama, "invoke", lambda self, *a, **k: _response())
    llm = NormalizedChatOllama(model="qwen3.8:27b", base_url="http://127.0.0.1:11435", num_ctx=32768)
    with usage.as_agent("Weekly review"):
        llm.invoke("hello")
    rows = usage.read_day(time.strftime("%Y-%m-%d"), {"results_dir": str(usage_dir.parent), "usage_dir": ""})
    assert rows and rows[0]["agent"] == "Weekly review" and rows[0]["model"] == "qwen3.8:27b"


@pytest.mark.unit
def test_a_days_records_are_summarised_by_agent_model_host_and_ticker(tmp_path):
    folder = tmp_path / "usage"
    folder.mkdir()
    rows = [{"ts": "2026-11-10T09:00:00-05:00", "host": "colossus", "model": "qwen3.8:27b", "agent": "Trader", "ticker": "NVDA",
             "tokens_in": 1000, "tokens_out": 100, "seconds": 60.0},
            {"ts": "2026-11-10T09:01:00-05:00", "host": "colossus", "model": "qwen3.8:27b", "agent": "Trader", "ticker": "AAPL",
             "tokens_in": 2000, "tokens_out": 200, "seconds": 120.0},
            {"ts": "2026-11-10T09:02:00-05:00", "host": "colossus", "model": "qwen3.8:27b", "agent": "Watcher", "ticker": "",
             "tokens_in": 500, "tokens_out": 50, "seconds": 30.0}]
    (folder / "2026-11-10.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n")
    cfg = {"results_dir": str(tmp_path), "usage_dir": ""}
    s = usage.summarize(usage.read_day("2026-11-10", cfg))
    assert s["total"] == {"calls": 3, "tokens_in": 3500, "tokens_out": 350, "seconds": 210.0, "hours": 0.06, "tickers": 2}
    assert list(s["by_agent"]) == ["Trader", "Watcher"] and s["by_agent"]["Trader"]["calls"] == 2
    assert s["by_ticker"]["AAPL"]["seconds"] == 120.0 and "-" in s["by_ticker"]
    assert "3 calls, 3,500 tokens in, 350 out" in usage.describe(s) and "2 ticker(s)" in usage.describe(s)
    assert usage.describe(usage.summarize([])) == "no model calls recorded"
    assert usage.read_day("2026-11-09", cfg) == []


# --- the sweep's progress ------------------------------------------------------

def _state(analysed=36, queued=467, in_flight=12, rounds=((12, 40.0), (12, 45.0)), started="2026-11-10T18:29:12-05:00"):
    return {"cycle": 2, "date": "2026-11-10", "analysed_this_cycle": analysed, "queue": [f"T{i}" for i in range(queued)],
            "round_no": 4, "round": {"tickers": [f"F{i}" for i in range(in_flight)], "parts": 3, "started": started} if in_flight else None,
            "rounds": [{"round": f"c2r{i + 2}", "tickers": t, "parts": 3, "minutes": m} for i, (t, m) in enumerate(rounds)],
            "saved": "2026-11-10T18:40:00-05:00"}


@pytest.mark.unit
def test_the_progress_counts_the_cycle_and_estimates_its_end_from_the_recent_pace():
    from datetime import datetime

    now = datetime.fromisoformat("2026-11-10T18:44:12-05:00")
    p = sw.progress(_state(), now)                                # a state file from before the pool still reads
    assert (p["analysed"], p["in_flight"], p["queued"], p["total"]) == (36, 12, 467, 515)
    assert p["minutes_per_round"] == 42.5 and p["tickers_per_round"] == 12 and p["minutes_per_ticker"] == 3.54
    # 467 queued and the 12 at work, taken as half done, × 3.54 min.
    assert p["eta_hours"] == pytest.approx((467 + 6) * 42.5 / 12 / 60, abs=0.06)
    # The pool: each run records its own minutes and how many processes were at work beside it.
    pool = {**_state(in_flight=0), "runs": [{"run_id": "x_a", "tickers": ["A", "B"], "started": "2026-11-10T18:29:12-05:00"},
                                             {"run_id": "x_b", "tickers": ["C", "D"], "started": "2026-11-10T18:31:00-05:00"}],
            "rounds": [{"round": "c2r7", "tickers": 2, "parts": 5, "pool": True, "minutes": 20.0},
                       {"round": "c2r8", "tickers": 2, "parts": 5, "pool": True, "minutes": 30.0}]}
    pp = sw.progress(pool, now)
    assert (pp["in_flight"], pp["parts"], pp["round_started"]) == (4, 2, "2026-11-10T18:29:12-05:00")
    assert pp["minutes_per_ticker"] == 2.5                        # 50 process-minutes over 4 tickers, five at work: 2.5 min of wall clock each
    assert p["pct"] == 7.0
    text = sw.describe(p)
    assert text.startswith("cycle 2 on 2026-11-10: 36 of 515 analysed (7%), 12 in flight, 467 queued, 42 min per run of 12, about ")
    assert sw.describe({}) == "no sweep yet" and sw.progress({}) == {}
    # No finished round yet: counts only, no pace or estimate.
    p0 = sw.progress(_state(analysed=0, rounds=()))
    assert "eta_hours" not in p0 and "min per run" not in sw.describe(p0)
    assert sw.describe_eta({"eta_hours": 0.5}) == "30 min" and sw.describe_eta({"eta_hours": 60}) == "2.5 days"


@pytest.mark.unit
def test_a_finished_round_is_kept_in_the_state_with_its_pace(tmp_path, monkeypatch, caplog):
    import logging

    from tests.test_sweep import FakeProc

    FakeProc.live = []
    monkeypatch.setattr(sw.subprocess, "Popen", FakeProc)
    watch = tmp_path / "sp500.txt"
    watch.write_text(",".join(f"T{i:02d}" for i in range(12)) + "\n")
    from tradingagents.default_config import DEFAULT_CONFIG

    cfg = {**DEFAULT_CONFIG, "results_dir": str(tmp_path), "sweep_state_path": str(tmp_path / "sweep.json"),
           "daily_watchlist": str(watch), "daily_parts": 3, "daily_parts_market_hours": 1, "sweep_round_size": 2}
    s = sw.Sweep(cfg)
    s.tick("2026-11-10", False, [], {})
    assert len(FakeProc.live) == 3
    for proc in FakeProc.live:
        proc.finish()
    with caplog.at_level(logging.INFO):
        s.tick("2026-11-10", False, [], {})      # takes in the three runs, starts three more
    saved = json.loads((tmp_path / "sweep.json").read_text())
    assert len(saved["rounds"]) == 3 and all(r["tickers"] == 2 and r["pool"] for r in saved["rounds"])
    assert len(saved["runs"]) == 3 and saved["runs"][0]["started"]
    p = s.progress()
    assert (p["analysed"], p["in_flight"], p["queued"], p["total"]) == (6, 6, 0, 12)
    assert "6 of 12 analysed this cycle, 6 queued" in caplog.text    # logged as the last of the three came in


# --- PAIR's record of who served what -------------------------------------------

def _pair_dir(tmp_path):
    folder = tmp_path / "pair"
    (folder / "cluster").mkdir(parents=True)
    (folder / "cluster" / "members.json").write_text(json.dumps([
        {"nodeUuid": "u-dev", "name": "InfinitiDev"}, {"nodeUuid": "u-col", "name": "colossus"}, {"nodeUuid": "u-pgx", "name": "thinkstationpgx"}]))
    t0 = 1_800_000_000_000
    rows = [{"id": "1", "model": "qwen3.8:27b", "state": "completed", "originatedFrom": "u-col", "scheduledOn": "u-dev",
             "createdAt": t0, "startedAt": t0, "completedAt": t0 + 60_000},
            {"id": "2", "model": "qwen3.8:27b", "state": "completed", "originatedFrom": "u-col", "scheduledOn": "u-pgx",
             "createdAt": t0 + 1000, "startedAt": t0 + 1000, "completedAt": t0 + 1000 + 180_000},
            {"id": "3", "model": "gemma4:26b", "state": "cancelled", "originatedFrom": "u-dev", "scheduledOn": "u-dev",
             "createdAt": t0 + 2000, "startedAt": t0 + 2000, "completedAt": None}]
    (folder / "workloads-history.json").write_text(json.dumps(rows))
    # The rotated file repeats a record and adds an old one outside the window.
    (folder / "workloads-history.json.1").write_text(json.dumps([rows[0], {**rows[0], "id": "0", "createdAt": t0 - 2 * 86_400_000,
                                                                           "startedAt": t0 - 2 * 86_400_000, "completedAt": t0 - 2 * 86_400_000 + 1000}]))
    return folder, t0


@pytest.mark.unit
def test_node_shares_come_from_the_routers_history_deduplicated_and_windowed(tmp_path):
    folder, t0 = _pair_dir(tmp_path)
    assert pair.available(folder) and not pair.available(tmp_path)
    assert len(pair.history(folder)) == 4
    s = pair.shares(24, folder, now_ms=t0 + 3600_000)
    assert s["calls"] == 3 and s["hours"] == 24
    assert [n["node"] for n in s["nodes"]] == ["thinkstationpgx", "InfinitiDev"]
    dev = s["nodes"][1]
    assert dev["calls"] == 2 and dev["cancelled"] == 1 and dev["busy_hours"] == round(1 / 60, 2)
    assert set(dev["models"]) == {"qwen3.8:27b", "gemma4:26b"} and dev["share_pct"] == pytest.approx(66.7)
    assert s["origins"] == {"colossus": 2, "InfinitiDev": 1}
    text = pair.render(s)
    assert "thinkstationpgx" in text and "originated from: colossus 2, InfinitiDev 1" in text
    assert pair.render({}) .startswith("No PAIR history found")
    assert pair.shares(24, tmp_path) == {}


@pytest.mark.unit
def test_the_pair_folder_comes_from_the_config_or_the_environment(monkeypatch, tmp_path):
    monkeypatch.delenv("TRADINGAGENTS_PAIR_DIR", raising=False)
    assert pair.pair_dir({}) == pair.DEFAULT_DIR
    assert pair.pair_dir({"pair_dir": str(tmp_path)}) == tmp_path
    monkeypatch.setenv("TRADINGAGENTS_PAIR_DIR", "/srv/pair")
    assert str(pair.pair_dir({})) == "/srv/pair"


# --- the page and the health endpoint ------------------------------------------

@pytest.mark.unit
def test_the_page_and_health_show_the_sweeps_progress_the_days_spend_and_the_nodes(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from tradingagents.dataflows.vendors.alpaca.trading import Account
    from tradingagents.web import app as web

    folder, t0 = _pair_dir(tmp_path)
    (tmp_path / "memory").mkdir()
    (tmp_path / "memory" / "sweep.json").write_text(json.dumps(_state()))
    (tmp_path / "usage").mkdir()
    (tmp_path / "usage" / f"{time.strftime('%Y-%m-%d')}.jsonl").write_text(json.dumps(
        {"ts": time.strftime("%Y-%m-%dT09:00:00"), "host": "colossus", "model": "qwen3.8:27b", "agent": "Trader", "ticker": "NVDA",
         "tokens_in": 1000, "tokens_out": 100, "seconds": 60.0}) + "\n")
    cfg = {"results_dir": str(tmp_path), "usage_dir": "", "sweep_state_path": str(tmp_path / "memory" / "sweep.json"),
           "pair_dir": str(folder), "default_stop_pct": 2, "worker_state_path": str(tmp_path / "memory" / "worker.json")}
    monkeypatch.setattr(web, "get_config", lambda: cfg)
    monkeypatch.setattr(web, "_results_dir", lambda: tmp_path)
    monkeypatch.setattr(web, "_holdings", lambda: Account(100_000, 60_000, True, ()))
    monkeypatch.setattr(pair.time, "time", lambda: (t0 + 3600_000) / 1000)
    client = TestClient(web.app)

    page = client.get("/").text
    assert "Desk status" in page and "36 of 515" in page and "cycle 2 on 2026-11-10" in code_free(page)
    assert "Model calls today" in page and "Trader" in page
    assert "PAIR nodes, last 24 h" in page and "thinkstationpgx" in page

    health = client.get("/health").json()
    assert health["sweep"]["queued"] == 467 and health["sweep_text"].startswith("cycle 2 on 2026-11-10: 36 of 515")
    assert health["usage_today"]["calls"] == 1

    status = client.get("/api/status").json()
    assert status["nodes"]["calls"] == 3 and status["usage_today"]["by_agent"]["Trader"]["tokens_in"] == 1000


def code_free(page: str) -> str:
    return page.replace("<code>", "").replace("</code>", "")
