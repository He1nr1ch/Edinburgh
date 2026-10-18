"""Canonical provider -> API-key env-var mapping.

A single source of truth for which environment variable holds the API key for
each supported LLM provider. Used by the CLI's interactive key prompt and by
the clients themselves.
"""

from __future__ import annotations

PROVIDER_API_KEY_ENV: dict[str, str | None] = {
    "openai":     "OPENAI_API_KEY",
    "anthropic":  "ANTHROPIC_API_KEY",
    # Local runtimes do not authenticate.
    "ollama":     None,
    # Generic OpenAI-compatible endpoint: the client reads this when set (keyed
    # relays), but it is marked key-optional in the provider registry so the CLI
    # never forces a prompt and keyless local servers still work.
    "openai_compatible": "OPENAI_COMPATIBLE_API_KEY",
}


def get_api_key_env(provider: str) -> str | None:
    """Return the env var name for `provider`'s API key, or None if not applicable.

    Unknown providers also return None — callers should treat that as
    "no key check possible" rather than as "no key required".
    """
    return PROVIDER_API_KEY_ENV.get(provider.lower())
