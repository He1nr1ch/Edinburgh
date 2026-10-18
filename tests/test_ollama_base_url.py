"""Tests for OLLAMA_BASE_URL env-var override across CLI and client paths."""

from __future__ import annotations

import re

# Rich colorizes console output and highlights numbers and URLs, which splits
# asserted substrings with escape codes ("port \x1b[1;33m11434"). Whether it
# does so depends on the ambient terminal, so strip the codes to keep these
# assertions independent of where the suite runs.
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _console_out(capsys) -> str:
    return _ANSI.sub("", capsys.readouterr().out)


# ---- ollama_client: base_url resolution ------------------------------------


def _base_url(**kwargs) -> str:
    from tradingagents.llm_clients.ollama_client import OllamaClient
    return str(OllamaClient(model="m", **kwargs).get_llm().base_url)


def test_resolver_returns_default_when_env_unset(monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    assert _base_url() == "http://localhost:11434"


def test_resolver_returns_env_when_set(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://remote-ollama:11434")
    assert _base_url() == "http://remote-ollama:11434"


def test_an_openai_style_v1_suffix_is_dropped(monkeypatch):
    """Earlier releases took the OpenAI-compatible URL; the native API wants the host root."""
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://remote-ollama:11434/v1")
    assert _base_url() == "http://remote-ollama:11434"


def test_resolver_evaluation_is_call_time(monkeypatch):
    """Setting the env AFTER module import must still take effect."""
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    import tradingagents.llm_clients.ollama_client  # noqa: F401
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://late-set:11434")
    assert _base_url() == "http://late-set:11434"


def test_explicit_base_url_overrides_env(monkeypatch):
    """An explicit base_url passed to the client wins over the env var."""
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://env-set:11434")
    assert _base_url(base_url="http://explicit:11434/v1") == "http://explicit:11434"


def test_the_factory_routes_ollama_to_the_native_client():
    from tradingagents.llm_clients import create_llm_client
    from tradingagents.llm_clients.ollama_client import OllamaClient

    assert isinstance(create_llm_client("ollama", "qwen3.8:27b"), OllamaClient)


def test_the_context_window_and_token_cap_reach_the_model():
    from tradingagents.llm_clients import build_llm_kwargs, create_llm_client

    kwargs = build_llm_kwargs({"llm_provider": "ollama", "ollama_num_ctx": "32768", "max_tokens": 4096,
                               "temperature": 0.1, "ollama_thinking": False})
    llm = create_llm_client("ollama", "qwen3.8:27b", **kwargs).get_llm()

    assert (llm.num_ctx, llm.num_predict, llm.temperature, llm.reasoning) == (32768, 4096, 0.1, False)


def test_thinking_is_off_by_default_and_switchable():
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.llm_clients import build_llm_kwargs

    assert DEFAULT_CONFIG["ollama_thinking"] is False
    assert build_llm_kwargs({"llm_provider": "ollama", "ollama_thinking": True})["reasoning"] is True
    assert "reasoning" not in build_llm_kwargs({"llm_provider": "ollama", "ollama_thinking": None})


def test_the_default_config_asks_for_a_large_window():
    from tradingagents.default_config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["ollama_num_ctx"] >= 16384


# ---- confirm_ollama_endpoint UX -------------------------------------------


def test_confirm_endpoint_shows_default(monkeypatch, capsys):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    from cli import prompts
    prompts.confirm_ollama_endpoint("http://localhost:11434")
    out = _console_out(capsys)
    assert "http://localhost:11434" in out
    assert "OLLAMA_BASE_URL" not in out  # not from env
    assert "Note" not in out  # no warnings for the canonical default


def test_confirm_endpoint_marks_env_origin(monkeypatch, capsys):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://remote-host:11434/v1")
    from cli import prompts
    prompts.confirm_ollama_endpoint("http://remote-host:11434/v1")
    out = _console_out(capsys)
    assert "http://remote-host:11434/v1" in out
    assert "OLLAMA_BASE_URL" in out


def test_confirm_endpoint_warns_on_missing_scheme(monkeypatch, capsys):
    """If user sets OLLAMA_BASE_URL=0.0.0.128, advise on the expected shape."""
    monkeypatch.setenv("OLLAMA_BASE_URL", "0.0.0.128")
    from cli import prompts
    prompts.confirm_ollama_endpoint("0.0.0.128")
    out = _console_out(capsys)
    assert "missing a scheme" in out
    assert "http://<host>:11434" in out


def test_confirm_endpoint_warns_on_non_default_port_remote(monkeypatch, capsys):
    """A remote host with no :11434 gets a soft hint about port mismatch."""
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://remote-host/v1")
    from cli import prompts
    prompts.confirm_ollama_endpoint("http://remote-host/v1")
    out = _console_out(capsys)
    assert "port 11434" in out


def test_confirm_endpoint_quiet_on_local_no_port(monkeypatch, capsys):
    """Local host without port shouldn't trigger the remote-port hint."""
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost/v1")
    from cli import prompts
    prompts.confirm_ollama_endpoint("http://localhost/v1")
    out = _console_out(capsys)
    assert "Note" not in out  # localhost is fine without explicit port


def test_ollama_model_labels_no_local_suffix():
    """Labels should no longer claim '(local)' since the endpoint is dynamic."""
    from tradingagents.llm_clients.model_catalog import get_model_options
    for mode in ("quick", "deep"):
        labels = [label for label, _ in get_model_options("ollama", mode)]
        assert all("local" not in label for label in labels), labels


def test_ollama_offers_custom_model_id():
    """Ollama users with custom-pulled models can pick 'Custom model ID'."""
    from tradingagents.llm_clients.model_catalog import get_model_options
    for mode in ("quick", "deep"):
        entries = get_model_options("ollama", mode)
        values = [v for _, v in entries]
        assert "custom" in values, f"Ollama {mode!r} missing 'custom' option: {entries}"
        # Custom option is last so it doesn't push the curated defaults off-screen
        assert values[-1] == "custom", f"'custom' should be last entry: {values}"
