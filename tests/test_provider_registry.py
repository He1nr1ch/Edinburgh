"""The OpenAI-compatible provider registry is the single source of truth for the
family; this guards each provider's resolved config (base URL, subclass, auth,
Responses API) so a future edit can't silently break one.
"""
import pytest

from tradingagents.llm_clients.openai_client import (
    OPENAI_COMPATIBLE_PROVIDERS,
    LocalCompatibleChatOpenAI,
    NormalizedChatOpenAI,
    is_openai_compatible,
)


@pytest.mark.unit
def test_registry_membership():
    assert is_openai_compatible("openai")
    assert not is_openai_compatible("ollama")      # its own client, on the native API
    assert is_openai_compatible("openai_compatible")  # the generic endpoint
    # Anthropic has its own API and client.
    assert not is_openai_compatible("anthropic")
    # Providers removed from the fork are gone, not silently served.
    for gone in ("google", "azure", "bedrock", "chatgpt", "deepseek", "openrouter", "xai"):
        assert not is_openai_compatible(gone)


@pytest.mark.unit
@pytest.mark.parametrize("provider,base_url,chat_class,responses", [
    ("openai", None, NormalizedChatOpenAI, True),
    ("openai_compatible", None, LocalCompatibleChatOpenAI, False),
])
def test_registry_spec(provider, base_url, chat_class, responses):
    spec = OPENAI_COMPATIBLE_PROVIDERS[provider]
    assert spec.base_url == base_url
    assert spec.chat_class is chat_class
    assert spec.use_responses_api is responses


@pytest.mark.unit
def test_key_optionality():
    # The generic endpoint is key-optional; the hosted API requires a key.
    assert OPENAI_COMPATIBLE_PROVIDERS["openai_compatible"].key_optional is True
    assert OPENAI_COMPATIBLE_PROVIDERS["openai_compatible"].require_base_url is True
    assert OPENAI_COMPATIBLE_PROVIDERS["openai"].key_optional is False


@pytest.mark.unit
def test_an_unsupported_provider_is_refused_by_the_factory():
    from tradingagents.llm_clients.factory import create_llm_client

    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        create_llm_client("google", "gemini")
