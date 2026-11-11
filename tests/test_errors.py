"""The desk's record of model failures: the log, the daily report's section, the outage alert, the sweep's retry."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest
from langchain_ollama import ChatOllama

from tests.test_reports import _cfg, _day_data, _fake_llm
from tests.test_sweep import FakeProc, cfg  # noqa: F401  (cfg is a fixture)
from tradingagents import errors, sweep as sw, usage
from tradingagents.agents import schemas
from tradingagents.agents.structured import decide, settle
from tradingagents.dataflows.config import run_config
from tradingagents.llm_clients.ollama_client import (
    ContextWindowExceeded,
    InferenceTimeout,
    InferenceUnavailable,
    NormalizedChatOllama,
)
from tradingagents.reports import daily

NOW = datetime.now().astimezone().replace(microsecond=0)
TODAY = NOW.date().isoformat()


def _write(tmp_path, folder: str, rows: list[dict]) -> None:
    (tmp_path / folder).mkdir(exist_ok=True)
    with (tmp_path / folder / f"{TODAY}.jsonl").open("a", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)


def _ago(minutes: float) -> str:
    return (NOW - timedelta(minutes=minutes)).isoformat(timespec="seconds")


# --- the log --------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("raised,kind", [(InferenceTimeout("sent nothing for 600 s"), "timeout"),
                                         (InferenceUnavailable("could not be reached"), "unavailable"),
                                         (RuntimeError("no user query found in messages"), "context")])
def test_the_client_records_every_call_that_got_no_answer(tmp_path, monkeypatch, raised, kind):
    def fail(self, *a, **k):
        raise raised

    monkeypatch.setattr(ChatOllama, "invoke", fail)
    llm = NormalizedChatOllama(model="qwen3.8:27b", base_url="http://127.0.0.1:11435", num_ctx=32768)
    with run_config({"results_dir": str(tmp_path)}):
        usage.set_run(ticker="NVDA", run_id="r1")
        try:
            with usage.as_agent("Trader"), pytest.raises((InferenceTimeout, InferenceUnavailable, ContextWindowExceeded)):
                llm.invoke("hello")
        finally:
            usage.set_run(ticker="", run_id="")
        (row,) = errors.read_day(TODAY, {"results_dir": str(tmp_path)})
    assert (row["kind"], row["agent"], row["ticker"], row["run_id"], row["model"]) == (kind, "Trader", "NVDA", "r1", "qwen3.8:27b")


@pytest.mark.unit
def test_an_answer_that_fails_its_contract_is_recorded_once_and_no_answer_is_not_recorded_as_one(tmp_path):
    conf = {"results_dir": str(tmp_path)}
    with run_config(conf):
        bad = MagicMock()
        bad.invoke.side_effect = ValueError("not json")
        assert settle(bad, MagicMock(), "PROMPT", "Trader").closed
        silent = MagicMock()
        silent.invoke.side_effect = InferenceTimeout("no answer")       # the client records this one itself
        assert decide(silent, "PROMPT", "Watcher")[0] is None
        good = MagicMock()
        good.invoke.return_value = schemas.TraderProposal(action="Buy", reasoning="r")
        decide(good, "PROMPT", "Trader")
    rows = errors.read_day(TODAY, conf)
    assert [(r["kind"], r["agent"]) for r in rows] == [("contract", "Trader")] and "not json" in rows[0]["message"]


@pytest.mark.unit
def test_the_days_errors_are_counted_by_kind_and_agent_with_the_tickers_that_failed():
    rows = [{"ts": f"{TODAY}T09:00:00-05:00", "kind": "timeout", "agent": "Trader", "ticker": "NVDA", "message": "sent nothing"},
            {"ts": f"{TODAY}T09:10:00-05:00", "kind": "timeout", "agent": "Portfolio Manager", "ticker": "GD", "message": "sent nothing"},
            {"ts": f"{TODAY}T09:20:00-05:00", "kind": "contract", "agent": "Trader", "ticker": "LEN", "message": "bad"},
            {"ts": f"{TODAY}T09:00:05-05:00", "kind": "run", "agent": "", "ticker": "NVDA", "message": "InferenceTimeout: sent nothing"}]
    s = errors.summarize(rows)
    assert s["total"] == 3 and s["by_kind"] == {"timeout": 2, "contract": 1} and s["by_agent"] == {"Trader": 2, "Portfolio Manager": 1}
    assert s["failed_tickers"] == {"NVDA": "InferenceTimeout: sent nothing"} and len(s["recent"]) == 3
    assert errors.describe(s) == "2 timeout, 1 contract; 1 ticker run(s) failed"
    assert errors.describe(errors.summarize([])) == "no model errors"


# --- the daily report -------------------------------------------------------------

@pytest.mark.unit
def test_the_daily_report_says_what_went_wrong_with_the_models(tmp_path, monkeypatch):
    conf = _cfg(tmp_path)
    _day_data(tmp_path, conf, monkeypatch)
    (tmp_path / "errors").mkdir()
    (tmp_path / "errors" / "2026-11-09.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"ts": "2026-11-09T10:00:00-05:00", "kind": "timeout", "agent": "Trader", "ticker": "LEN", "message": "qwen3.8:27b sent nothing for 600 s"},
        {"ts": "2026-11-09T10:00:01-05:00", "kind": "run", "agent": "", "ticker": "LEN", "message": "InferenceTimeout: sent nothing for 600 s"},
        {"ts": "2026-11-09T11:00:00-05:00", "kind": "contract", "agent": "Risk Manager", "ticker": "KIM", "message": "ValueError: not json"}]) + "\n")
    facts = daily.gather(conf, "2026-11-09")
    assert facts["errors"]["by_kind"] == {"timeout": 1, "contract": 1}

    rev = daily.DailyReview(what_happened="x", what_worked="y", what_did_not="z", lessons=[], rule_suggestions=[])
    llm, structured = _fake_llm(rev)
    path, _ = daily.run(conf, "2026-11-09", llm)
    assert "Model errors today: 1 timeout, 1 contract; 1 ticker run(s) failed; by agent: Trader 1, Risk Manager 1" in structured.prompt
    text = path.read_text()
    assert "## Model errors" in text and "| timeout | 1 |" in text and "| Risk Manager | 1 |" in text
    assert "- **LEN**: InferenceTimeout: sent nothing for 600 s" in text and "10:00 timeout Trader LEN" in text


@pytest.mark.unit
def test_a_clean_day_says_so(tmp_path, monkeypatch):
    conf = _cfg(tmp_path)
    _day_data(tmp_path, conf, monkeypatch)
    llm, structured = _fake_llm(daily.DailyReview(what_happened="x", what_worked="y", what_did_not="z", lessons=[], rule_suggestions=[]))
    path, _ = daily.run(conf, "2026-11-09", llm)
    assert "## Model errors\nNo model errors." in path.read_text() and "Model errors today" not in structured.prompt


# --- the outage alert -------------------------------------------------------------

def _silent(minutes: float, kind: str = "timeout") -> dict:
    return {"ts": _ago(minutes), "host": "colossus", "kind": kind, "agent": "Trader", "ticker": "NVDA", "message": "sent nothing for 600 s"}


@pytest.mark.unit
def test_one_email_when_the_model_goes_silent_and_one_when_it_is_back(tmp_path):
    conf = {"results_dir": str(tmp_path), "usage_dir": "", "model_outage_minutes": 15}
    sent = []
    send = lambda subject, body, config: sent.append((subject, body))  # noqa: E731

    _write(tmp_path, "usage", [{"ts": _ago(40), "model": "qwen3.8:27b", "seconds": 5.0}])
    _write(tmp_path, "errors", [_silent(20)])
    assert errors.watch(conf, NOW, send) is None                       # one call lost is not an outage
    _write(tmp_path, "errors", [_silent(9, "unavailable"), {**_silent(5), "kind": "contract"}])
    assert errors.watch(conf, NOW, send) == "down" and len(sent) == 1
    subject, body = sent[0]
    assert "not answering" in subject and "2 call(s): 1 timeout, 1 unavailable" in body and "stop and target guards" in body
    assert errors.watch(conf, NOW, send) is None and len(sent) == 1    # said once, also after a restart (the state is a file)

    _write(tmp_path, "usage", [{"ts": _ago(1), "model": "qwen3.8:27b", "seconds": 5.0}])
    assert errors.watch(conf, NOW, send) == "up" and "answering again" in sent[1][0] and "after 20 minutes" in sent[1][1]
    assert errors.watch(conf, NOW, send) is None and len(sent) == 2


@pytest.mark.unit
def test_a_short_gap_and_a_bad_answer_are_not_an_outage(tmp_path):
    conf = {"results_dir": str(tmp_path), "usage_dir": "", "model_outage_minutes": 15}
    _write(tmp_path, "errors", [_silent(8), _silent(4)])                          # two lost, but only 8 minutes
    assert errors.outage(conf, NOW) is None
    _write(tmp_path, "errors", [{**_silent(30), "kind": "contract"}, {**_silent(25), "kind": "contract"}])
    assert errors.outage(conf, NOW) is None                                        # it answered; the answers were bad
    _write(tmp_path, "errors", [_silent(20)])
    assert errors.outage(conf, NOW)["errors"] == 3
    _write(tmp_path, "usage", [{"ts": _ago(6), "model": "qwen3.8:27b", "seconds": 5.0}])
    assert errors.outage(conf, NOW) is None                                        # one answer since: only one call lost after it


# --- the sweep tries a failed ticker again ------------------------------------------

@pytest.mark.unit
def test_a_ticker_whose_run_failed_goes_back_in_the_queue_once_per_cycle(cfg):  # noqa: F811
    s = sw.Sweep(cfg)
    s.tick("2026-11-09", market_open=True, holdings=[], last_analysed={})         # one process, two tickers
    first = FakeProc.live[0].tickers()
    with run_config(cfg):
        errors.record("run", "InferenceTimeout: sent nothing for 600 s", ticker=first[0])
    FakeProc.live[0].finish()
    queued = len(s.queue)
    s.tick("2026-11-09", True, [], {})

    assert s.retried == [first[0]] and s.analysed_this_cycle == 1                  # only the one that got a decision counts
    assert first[0] in s.queue + FakeProc.live[1].tickers() and len(s.queue) == queued + 1 - 2
    assert sw.Sweep(cfg).retried == [first[0]]                                     # remembered across a restart

    # It comes up again later in the cycle and fails again: it waits for the next cycle, not a third try.
    again = sw.Round(["r"], [first[0], "T99"], 0.0, 1, s.round.started_at)
    with run_config(cfg):
        errors.record("run", "InferenceTimeout: again", ticker=first[0])
        errors.record("run", "VendorError: no data", ticker="T99")
    assert s._failed(again) == ["T99"]


@pytest.mark.unit
def test_a_sweep_run_writes_its_usage_and_errors_where_the_desk_reads_them(tmp_path, monkeypatch):
    """A run has a results folder of its own; its model calls and failures still go to the shared logs."""
    import tradingagents.backtest as bt
    from tests import test_backtest as tb

    seen = {}

    class Graph(tb._FakeGraph):
        def __init__(self, selected_analysts=None, config=None, **kw):
            super().__init__(selected_analysts, config=config, **kw)
            seen.update(config)

        def propagate(self, ticker, trade_date, *a, **k):
            with run_config(seen):                     # as the real graph does for the length of a run
                if ticker == "NVDA":
                    errors.record("timeout", "sent nothing for 600 s", "Trader", "qwen3.8:27b", ticker="NVDA")
                    raise InferenceTimeout("sent nothing for 600 s")
                usage.record({"ts": NOW.isoformat(timespec="seconds"), "model": "qwen3.8:27b", "agent": "Trader", "seconds": 5.0})

    monkeypatch.setattr(bt, "TradingAgentsGraph", Graph)
    config = {"results_dir": str(tmp_path / "results"), "memory_log_path": str(tmp_path / "log.md")}
    with run_config(config):
        result = bt.run_backtest(["NVDA", "AAPL"], ["2026-01-05"], config, run_id="r1")

    assert seen["results_dir"].endswith("backtest/r1") and seen["usage_dir"] == str(tmp_path / "results" / "usage")
    assert [f[0] for f in result.failures] == ["NVDA"]
    rows = errors.read_day(TODAY, config)                 # read from the desk's folder, not the run's
    assert [(r["kind"], r["ticker"]) for r in rows] == [("timeout", "NVDA"), ("run", "NVDA")]
    assert rows[1]["message"] == "InferenceTimeout: sent nothing for 600 s"
    assert errors.failed_tickers(TODAY, "", ["AAPL", "NVDA"], config) == ["NVDA"]
    assert len(usage.read_day(TODAY, {**config, "usage_dir": ""})) == 1
