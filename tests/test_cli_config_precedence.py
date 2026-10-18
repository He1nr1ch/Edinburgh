"""CLI config precedence (#976, #977).

An explicit environment override for the debate/risk round counts, or the
checkpoint flag, must win over the interactive research-depth selection — the CLI
must not clobber an env-configured value back to a prompt/flag default.
"""

from unittest import mock

import pytest

import cli.main as m
import cli.run as cli_run

# Minimal selections dict shaped like get_user_selections()'s return value.
SELECTIONS = {
    "research_depth": 5,
    "quick_think_llm": "gpt-5.4-mini",
    "deep_think_llm": "gpt-5.5",
    "backend_url": None,
    "llm_provider": "openai",
    "openai_reasoning_effort": None,
    "anthropic_effort": None,
}


def test_research_depth_sets_both_rounds_without_env(monkeypatch):
    for var in ("TRADINGAGENTS_MAX_DEBATE_ROUNDS", "TRADINGAGENTS_MAX_RISK_ROUNDS"):
        monkeypatch.delenv(var, raising=False)
    cfg = cli_run._build_run_config(SELECTIONS, checkpoint=None)
    assert cfg["max_debate_rounds"] == 5
    assert cfg["max_risk_discuss_rounds"] == 5


def test_env_round_counts_win_over_selection(monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_MAX_DEBATE_ROUNDS", "2")
    monkeypatch.setenv("TRADINGAGENTS_MAX_RISK_ROUNDS", "4")
    # DEFAULT_CONFIG already reflects the env (applied at import); emulate that.
    patched = dict(cli_run.DEFAULT_CONFIG, max_debate_rounds=2, max_risk_discuss_rounds=4)
    with mock.patch.object(cli_run, "DEFAULT_CONFIG", patched):
        cfg = cli_run._build_run_config(SELECTIONS, checkpoint=None)
    assert cfg["max_debate_rounds"] == 2  # env value, not research_depth=5
    assert cfg["max_risk_discuss_rounds"] == 4


def test_partial_env_only_overrides_that_count(monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_MAX_DEBATE_ROUNDS", "2")
    monkeypatch.delenv("TRADINGAGENTS_MAX_RISK_ROUNDS", raising=False)
    patched = dict(cli_run.DEFAULT_CONFIG, max_debate_rounds=2)
    with mock.patch.object(cli_run, "DEFAULT_CONFIG", patched):
        cfg = cli_run._build_run_config(SELECTIONS, checkpoint=None)
    assert cfg["max_debate_rounds"] == 2  # env wins
    assert cfg["max_risk_discuss_rounds"] == 5  # falls through to research_depth


def test_checkpoint_none_preserves_env_default():
    patched = dict(cli_run.DEFAULT_CONFIG, checkpoint_enabled=True)  # e.g. env-enabled
    with mock.patch.object(cli_run, "DEFAULT_CONFIG", patched):
        cfg = cli_run._build_run_config(SELECTIONS, checkpoint=None)
    assert cfg["checkpoint_enabled"] is True  # not clobbered back to False


@pytest.mark.parametrize("flag", [True, False])
def test_checkpoint_flag_overrides_env(flag):
    patched = dict(cli_run.DEFAULT_CONFIG, checkpoint_enabled=not flag)
    with mock.patch.object(cli_run, "DEFAULT_CONFIG", patched):
        cfg = cli_run._build_run_config(SELECTIONS, checkpoint=flag)
    assert cfg["checkpoint_enabled"] is flag


@pytest.mark.unit
def test_a_half_set_round_count_says_which_value_won(capsys, monkeypatch):
    """With only one of the two round-count variables set, the depth prompt is
    still shown but half the answer is discarded; the user was never told."""

    monkeypatch.setenv("TRADINGAGENTS_MAX_DEBATE_ROUNDS", "1")
    monkeypatch.delenv("TRADINGAGENTS_MAX_RISK_ROUNDS", raising=False)
    printed = []
    monkeypatch.setattr(m.console, "print", lambda *a, **k: printed.append(str(a[0]) if a else ""))

    config = cli_run._build_run_config({
        "ticker": "NVDA", "analysis_date": "2026-09-01", "asset_type": "stock",
        "analysts": [], "research_depth": 5, "llm_provider": "openai",
        "quick_think_llm": "gpt-5.6-luna", "deep_think_llm": "gpt-5.6",
        "backend_url": None,
    }, None)

    assert config["max_risk_discuss_rounds"] == 5
    assert any("TRADINGAGENTS_MAX_DEBATE_ROUNDS" in line for line in printed), printed


@pytest.mark.unit
def test_a_tier_s_thinking_setting_survives_a_main_provider_that_has_none(monkeypatch):
    """A run on Ollama with an OpenAI tier: the run's selections carry no
    OpenAI reasoning effort, so the configured one must reach that tier."""
    monkeypatch.setitem(cli_run.DEFAULT_CONFIG, "openai_reasoning_effort", "high")
    monkeypatch.setitem(cli_run.DEFAULT_CONFIG, "anthropic_effort", "low")
    cfg = cli_run._build_run_config({**SELECTIONS, "llm_provider": "ollama"}, checkpoint=None)
    assert (cfg["openai_reasoning_effort"], cfg["anthropic_effort"]) == ("high", "low")
