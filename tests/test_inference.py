"""The one inference client: its time limits, its settings per call, PAIR, and failing closed."""

from __future__ import annotations

import ast
import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest
from langchain_ollama import ChatOllama

from tradingagents import pair
from tradingagents.agents.structured import settle
from tradingagents.dataflows.vendors.alpaca.trading import Account, Holding
from tradingagents.default_config import DEFAULT_CONFIG, _coerce
from tradingagents.llm_clients import ollama_client
from tradingagents.llm_clients.factory import create_tier_client, for_agent
from tradingagents.llm_clients.ollama_client import (
    InferenceError,
    InferenceTimeout,
    InferenceUnavailable,
    NormalizedChatOllama,
    check_endpoint,
)
from tradingagents.worker.duty import WorkerContext
from tradingagents.worker.state import WorkerState

ROOT = Path(__file__).resolve().parents[1]
URL = "http://127.0.0.1:11435"


def _llm(**kwargs) -> NormalizedChatOllama:
    return NormalizedChatOllama(model="qwen3.8:27b", base_url=URL, num_ctx=32768, **kwargs)


def _stream(monkeypatch, parts, closed: list):
    """Make the server's stream ``parts`` (an exception among them is raised), noting when it is closed."""
    def stream(self, messages, stop=None, **kwargs):
        try:
            for part in parts:
                if isinstance(part, BaseException):
                    raise part
                yield part
        finally:
            closed.append(True)

    monkeypatch.setattr(ChatOllama, "_create_chat_stream", stream)


# --- time limits ----------------------------------------------------------------

@pytest.mark.unit
def test_a_call_past_its_total_limit_is_dropped_and_raised(monkeypatch):
    closed: list = []
    _stream(monkeypatch, ["a", "b", "c"], closed)
    clock = iter([0.0, 1.0, 5.0, 99.0])          # the deadline is set at 0; "b" arrives past it
    monkeypatch.setattr(ollama_client.time, "monotonic", lambda: next(clock))
    got = []
    with pytest.raises(InferenceTimeout, match="after 3 s"):
        for part in _llm(call_timeout=3)._create_chat_stream([]):
            got.append(part)
    assert got == ["a", "b"] and closed == [True]      # the server's stream is closed too, so PAIR frees the node


@pytest.mark.unit
def test_silence_past_the_idle_limit_is_a_timeout_and_a_dead_server_is_unavailable(monkeypatch):
    _stream(monkeypatch, ["a", httpx.ReadTimeout("timed out")], [])
    with pytest.raises(InferenceTimeout, match="sent nothing for 600 s"):
        list(_llm(idle_timeout=600)._create_chat_stream([]))
    for error in (httpx.ConnectError("refused"), ConnectionError("Failed to connect to Ollama")):
        _stream(monkeypatch, [error], [])
        with pytest.raises(InferenceUnavailable, match="could not be reached"):
            list(_llm()._create_chat_stream([]))
    assert issubclass(InferenceTimeout, InferenceError) and not issubclass(InferenceError, ValueError)


@pytest.mark.unit
def test_a_call_within_its_limits_is_passed_through_untouched(monkeypatch):
    closed: list = []
    _stream(monkeypatch, ["a", "b"], closed)
    assert list(_llm(idle_timeout=600, call_timeout=900)._create_chat_stream([])) == ["a", "b"] and closed == [True]


@pytest.mark.unit
def test_the_limits_come_from_the_config_and_the_idle_one_is_the_streams_read_timeout():
    config = {**DEFAULT_CONFIG, "llm_provider": "ollama", "quick_think_llm": "qwen3.8:27b", "backend_url": URL}
    llm = create_tier_client(config, "quick").get_llm()
    assert (llm.idle_timeout, llm.call_timeout) == (600.0, 900.0)
    timeout = llm._client._client.timeout
    assert (timeout.read, timeout.connect) == (600.0, 10.0)
    tight = create_tier_client({**config, "ollama_idle_timeout": 120, "ollama_call_timeout": 300}, "quick").get_llm()
    assert (tight.idle_timeout, tight.call_timeout, tight._client._client.timeout.read) == (120.0, 300.0, 120.0)


# --- failing closed -------------------------------------------------------------

@pytest.mark.unit
def test_a_timeout_fails_the_run_instead_of_closing_it_as_a_decision():
    structured = MagicMock()
    structured.invoke.side_effect = InferenceTimeout("qwen3.8:27b sent nothing for 600 s")
    plain = MagicMock()
    with pytest.raises(InferenceTimeout):
        settle(structured, plain, "PROMPT", "Trader")
    assert structured.invoke.call_count == 1          # no second attempt, and no prose asked for
    plain.invoke.assert_not_called()


def _news_context(tmp_path, monkeypatch, structured):
    from tradingagents.worker import news

    article = {"id": "a1", "headline": "JPMorgan CFO resigns abruptly", "summary": "", "source": "WSJ",
               "created_at": datetime(2026, 11, 6, 15, 0, tzinfo=UTC), "url": "u"}
    monkeypatch.setattr(news, "articles_since", lambda s, since, limit=20: [article])
    monkeypatch.setattr(news, "_name", lambda s: "JPMorgan Chase")
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    cfg = {**DEFAULT_CONFIG, "results_dir": str(tmp_path), "worker_news_lookback_hours": 18,
           "worker_state_path": str(tmp_path / "state.json")}
    state = WorkerState.load(tmp_path / "state.json")
    holding = Holding("JPM", 40, 12_000.0, 300.0, "us_equity")
    return news, state, WorkerContext(Account(100_000, 0, True, (holding,)), datetime(2026, 11, 6, 15, 5, tzinfo=UTC), cfg, state, llm)


@pytest.mark.unit
def test_news_the_model_could_not_be_asked_about_stays_unread_for_the_next_pass(tmp_path, monkeypatch):
    structured = MagicMock()
    structured.invoke.side_effect = InferenceTimeout("no answer in 600 s")
    news, state, ctx = _news_context(tmp_path, monkeypatch, structured)

    (finding,) = news.WatcherDuty().run(ctx)
    assert finding.severity == "info" and finding.action is None and "wait" in finding.headline   # info: no email every pass
    assert state.is_new("JPM", "a1") and "JPM" not in state.last_check                              # nothing was read

    structured.invoke.side_effect = lambda prompt: news.NewsAssessment(
        materiality="high", direction="adverse", accounted_for=False, summary="CFO out.", rationale="Not in the thesis.")
    (finding,) = news.WatcherDuty().run(ctx)                                                        # the model is back
    assert finding.severity == "urgent" and finding.action.kind == "exit"
    assert not state.is_new("JPM", "a1") and "JPM" in state.last_check


@pytest.mark.unit
def test_news_the_model_answered_badly_twice_is_shown_to_a_person_and_not_asked_again(tmp_path, monkeypatch):
    structured = MagicMock()
    structured.invoke.side_effect = ValueError("not json")
    news, state, ctx = _news_context(tmp_path, monkeypatch, structured)
    (finding,) = news.WatcherDuty().run(ctx)
    assert finding.severity == "notable" and "could not judge" in finding.headline and structured.invoke.call_count == 2
    assert not state.is_new("JPM", "a1")


def _replacement_context(tmp_path, monkeypatch, structured):
    from tradingagents.worker import replace

    sizing = tmp_path / "backtest" / "sizing"
    sizing.mkdir(parents=True)
    (sizing / "run_a_2026-11-05_bench.csv").write_text(
        "ticker,side,rating,risk_level,ceiling_pct,price,stop,stop_from,target,expected_move_pct\n"
        "NOW,short,Underweight,Neutral,3,108.81,114.69,Risk Manager,95,12.7\n")
    monkeypatch.setattr(replace, "latest_price", lambda t: 109.0)
    monkeypatch.setattr(replace.trading, "open_orders", lambda: {})
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    cfg = {**DEFAULT_CONFIG, "results_dir": str(tmp_path), "worker_state_path": str(tmp_path / "state.json")}
    state = WorkerState.load(cfg["worker_state_path"])
    state.book = {"FIS": {"placed_on": "2026-11-06", "side": "short", "value": 4643.0}}
    ctx = WorkerContext(Account(100_000, 0, True, ()), datetime(2026, 11, 6, 15, 0, tzinfo=UTC), cfg, state, llm)
    return replace, state, ctx


@pytest.mark.unit
def test_no_answer_opens_no_replacement_and_the_next_pass_asks_again(tmp_path, monkeypatch):
    structured = MagicMock()
    structured.invoke.side_effect = InferenceUnavailable("PAIR is down")
    replace, state, ctx = _replacement_context(tmp_path, monkeypatch, structured)

    (finding,) = replace.ReplacementDuty().run(ctx)
    assert finding.action is None and finding.severity == "info" and "waits for the model" in finding.headline
    assert state.book == {"FIS": {"placed_on": "2026-11-06", "side": "short", "value": 4643.0}}    # still to be replaced

    structured.invoke.side_effect = lambda prompt: replace.ReplacementChoice(ticker="NOW", reason="the move left")
    (finding,) = replace.ReplacementDuty().run(ctx)
    assert (finding.action.symbol, finding.action.kind) == ("NOW", "replace") and "FIS" not in state.book


@pytest.mark.unit
def test_an_answer_that_fails_its_contract_opens_nothing_either(tmp_path, monkeypatch):
    structured = MagicMock()
    structured.invoke.side_effect = ValueError("not json")
    replace, state, ctx = _replacement_context(tmp_path, monkeypatch, structured)
    (finding,) = replace.ReplacementDuty().run(ctx)
    assert finding.action is None and finding.severity == "notable" and "opens nothing" in finding.headline
    assert "FIS" not in state.book          # a person is told; the best score is never opened unasked


# --- settings per call ----------------------------------------------------------

@pytest.mark.unit
def test_a_call_can_have_its_own_model_window_and_limit_on_the_same_connection():
    llm = _llm(idle_timeout=600, call_timeout=900)
    own = llm.with_call(model="qwen3.5:122b", num_ctx=65536, timeout=1200)
    assert (own.model, own.num_ctx, own.call_timeout, own.idle_timeout) == ("qwen3.5:122b", 65536, 1200, 600)
    assert own._client is llm._client and (llm.model, llm.num_ctx, llm.call_timeout) == ("qwen3.8:27b", 32768, 900)
    assert llm.with_call() is llm
    assert own._chat_params([], None)["model"] == "qwen3.5:122b" and own._chat_params([], None)["options"]["num_ctx"] == 65536


@pytest.mark.unit
def test_an_agents_settings_are_its_tiers_with_its_own_on_top():
    llm = _llm(call_timeout=900)
    config = {"agent_calls": {"deep": {"timeout": 1200}, "Portfolio Manager": {"model": "qwen3.5:122b"}}}
    pm = for_agent(llm, config, "Portfolio Manager", "deep")
    assert (pm.model, pm.call_timeout) == ("qwen3.5:122b", 1200)
    assert for_agent(llm, config, "Risk Manager", "deep").call_timeout == 1200
    assert for_agent(llm, config, "Trader", "quick") is llm and for_agent(llm, {}, "Trader", "quick") is llm
    with pytest.raises(ValueError, match="unknown setting"):
        for_agent(llm, {"agent_calls": {"Trader": {"temperature": 0}}}, "Trader", "quick")
    with pytest.raises(ValueError, match="cannot vary its calls"):
        for_agent(object(), config, "Portfolio Manager", "deep")


@pytest.mark.unit
def test_agent_calls_can_be_given_in_the_environment_as_json():
    assert _coerce('{"deep": {"timeout": 1200}}', {}) == {"deep": {"timeout": 1200}}
    with pytest.raises(ValueError):
        _coerce("[1, 2]", {})


# --- PAIR -------------------------------------------------------------------------

def _pair_dir(tmp_path, rows):
    folder = tmp_path / "pair"
    (folder / "cluster").mkdir(parents=True)
    (folder / "cluster" / "members.json").write_text(json.dumps([
        {"nodeUuid": "u-dev", "name": "InfinitiDev"}, {"nodeUuid": "u-col", "name": "colossus"}, {"nodeUuid": "u-pgx", "name": "pgx"}]))
    (folder / "node-id.json").write_text(json.dumps({"node_uuid": "u-dev"}))
    (folder / "workloads-history.json").write_text(json.dumps(rows))
    return folder


def _row(origin, node, created_s, completed_s, model="qwen3.8:27b", state="completed"):
    return {"id": "1", "model": model, "state": state, "originatedFrom": origin, "scheduledOn": node,
            "createdAt": created_s * 1000, "startedAt": created_s * 1000, "completedAt": completed_s * 1000}


@pytest.mark.unit
def test_a_call_is_matched_to_the_node_that_served_it_and_never_guessed(tmp_path):
    t = 1_800_000_000
    folder = _pair_dir(tmp_path, [
        _row("u-col", "pgx-other", t, t + 30),                       # another machine's call, same times
        _row("u-dev", "u-pgx", t, t + 30),                           # ours
        _row("u-dev", "u-col", t + 1, t + 55),                       # ours too, running beside it: ends later
        _row("u-dev", "u-dev", t, t + 30, model="gemma4:26b"),       # another model
        _row("u-dev", "u-dev", t, t + 30, state="cancelled"),
    ])
    assert pair.served_by("qwen3.8:27b", t + 0.2, t + 30.4, folder) == "pgx"
    assert pair.served_by("qwen3.8:27b", t + 1.1, t + 55.2, folder) == "colossus"
    assert pair.served_by("qwen3.8:27b", t + 100, t + 130, folder) == ""       # nothing fits: no node, not a guess
    assert pair.served_by("qwen3.8:27b", t, t + 30, tmp_path / "no-pair") == ""


@pytest.mark.unit
def test_a_pair_machine_refuses_an_endpoint_that_is_not_its_own_loopback(tmp_path, monkeypatch):
    check_endpoint("http://192.168.0.98:11434")                       # no PAIR here: any Ollama will do
    monkeypatch.setenv("TRADINGAGENTS_PAIR_DIR", str(_pair_dir(tmp_path, [])))
    check_endpoint("http://127.0.0.1:11435")
    check_endpoint("http://localhost:11434")
    with pytest.raises(ValueError, match="nvpair-proxy"):
        check_endpoint("http://192.168.0.98:11434")
    with pytest.raises(ValueError, match="own.*loopback"):
        create_tier_client({**DEFAULT_CONFIG, "llm_provider": "ollama", "quick_think_llm": "qwen3.8:27b",
                            "backend_url": "http://192.168.0.62:11435"}, "quick").get_llm()


# --- one way in -------------------------------------------------------------------

def _imports(path: Path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.add(node.module.split(".")[0])
    return names


@pytest.mark.unit
def test_only_the_client_package_talks_to_a_model_server():
    """No agent, duty or report builds its own client or posts to the server itself."""
    clients = ROOT / "tradingagents" / "llm_clients"
    libraries = {"langchain_ollama", "ollama", "langchain_openai", "langchain_anthropic", "openai", "anthropic"}
    offenders = sorted(
        str(path.relative_to(ROOT))
        for package in ("tradingagents", "cli")
        for path in (ROOT / package).rglob("*.py")
        if clients not in path.parents
        and (_imports(path) & libraries or "/api/chat" in path.read_text(encoding="utf-8")
             or "/api/generate" in path.read_text(encoding="utf-8"))
    )
    assert offenders == []


@pytest.mark.unit
def test_a_calls_node_is_added_to_its_usage_record_once_pairs_history_has_caught_up(tmp_path, monkeypatch):
    import time

    from tradingagents import usage
    from tradingagents.dataflows.config import run_config

    t = time.time() - 30
    folder = _pair_dir(tmp_path, [])
    monkeypatch.setenv("TRADINGAGENTS_PAIR_DIR", str(folder))
    monkeypatch.setattr(usage, "_pending", [])
    response = MagicMock(usage_metadata={"input_tokens": 10, "output_tokens": 2},
                         response_metadata={"model": "qwen3.8:27b", "total_duration": 20e9})
    cfg = {"results_dir": str(tmp_path), "usage_dir": ""}
    with run_config(cfg):
        rec = usage.record_response(response, "qwen3.8:27b", started=t, ended=t + 20)
        assert "node" not in rec and usage.read_day(rec["ts"][:10], cfg)[0].get("node") is None   # the history is behind
        (folder / "workloads-history.json").write_text(json.dumps([_row("u-dev", "u-col", t, t + 20)]))
        assert usage.resolve_nodes() == 1 and usage.resolve_nodes() == 0
    assert rec["node"] == "colossus"
    (row,) = usage.read_day(rec["ts"][:10], cfg)                 # the node line is folded into its record
    assert row["node"] == "colossus" and row["id"] == rec["id"]
    assert usage.summarize([row])["by_node"]["colossus"]["calls"] == 1
