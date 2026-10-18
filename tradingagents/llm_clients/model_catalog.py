"""Shared model catalog for CLI selections and validation."""

from __future__ import annotations

ModelOption = tuple[str, str]
ProviderModeOptions = dict[str, dict[str, list[ModelOption]]]

# Providers that serve whatever the user has deployed: offer only "Custom
# model ID" rather than a list that goes stale.
_CUSTOM_ONLY: dict[str, list[ModelOption]] = {
    "quick": [("Custom model ID", "custom")],
    "deep": [("Custom model ID", "custom")],
}


MODEL_OPTIONS: ProviderModeOptions = {
    "openai": {
        "quick": [
            ("GPT-6 Luna - Fast, high-volume and cost-efficient", "gpt-6-luna"),
            ("GPT-5.6 Luna - Previous-gen fast tier", "gpt-5.6-luna"),
            ("GPT-5.6 Terra - Balances intelligence and cost", "gpt-5.6-terra"),
            ("Custom model ID", "custom"),
        ],
        "deep": [
            ("GPT-6 Sol - Complex reasoning and analysis", "gpt-6-sol"),
            ("GPT-6 Astra - Hardest multi-step reasoning", "gpt-6-astra"),
            ("GPT-5.6 - Previous-gen reasoning (Sol)", "gpt-5.6"),
            ("GPT-5.5 - Earlier frontier, 1M context", "gpt-5.5"),
            ("Custom model ID", "custom"),
        ],
    },
    "anthropic": {
        "quick": [
            ("Claude Sonnet 5 - Best speed and intelligence balance", "claude-sonnet-5"),
            ("Claude Haiku 4.5 - Fastest with near-frontier intelligence", "claude-haiku-4-5"),
            ("Custom model ID", "custom"),
        ],
        "deep": [
            ("Claude Opus 5.5 - Frontier agentic and enterprise work", "claude-opus-5-5"),
            ("Claude Fable 5.1 - Most capable, demanding long-horizon reasoning", "claude-fable-5-1"),
            ("Claude Sonnet 5 - Near-frontier intelligence at Sonnet cost", "claude-sonnet-5"),
            ("Custom model ID", "custom"),
        ],
    },
    # Ollama: whatever has been pulled on the server named by OLLAMA_BASE_URL.
    # The first entry is the model on the PGX host; "Custom model ID" takes any
    # other pulled model.
    "ollama": {
        "quick": [
            ("Qwen 3.8 27B (qwen3.8:27b)", "qwen3.8:27b"),
            ("Qwen3:latest (8B)", "qwen3:latest"),
            ("GPT-OSS:latest (20B)", "gpt-oss:latest"),
            ("Custom model ID", "custom"),
        ],
        "deep": [
            ("Qwen 3.8 27B (qwen3.8:27b)", "qwen3.8:27b"),
            ("GPT-OSS:latest (20B)", "gpt-oss:latest"),
            ("Custom model ID", "custom"),
        ],
    },
    # Generic OpenAI-compatible endpoint: the model is whatever the user's
    # server serves, so only "Custom model ID" is offered.
    "openai_compatible": _CUSTOM_ONLY,
}


def get_model_options(provider: str, mode: str) -> list[ModelOption]:
    """Return shared model options for a provider and selection mode."""
    return MODEL_OPTIONS[provider.lower()][mode]


# Served by the provider but not offered in the picker: models retired from the
# menu, and the explicit ID of a model listed under a shorter name. Known to
# validation so a config naming one runs without an unknown-model warning.
LEGACY_MODELS: dict[str, list[str]] = {
    "openai": ["gpt-5.4", "gpt-5.4-mini", "gpt-5.6-sol"],
    "anthropic": ["claude-opus-5", "claude-fable-5", "claude-opus-4-8", "claude-opus-4-7"],
}


def get_known_models() -> dict[str, list[str]]:
    """Build known model names from the shared CLI catalog plus legacy IDs."""
    return {
        provider: sorted(
            {
                value
                for options in mode_options.values()
                for _, value in options
            }
            | set(LEGACY_MODELS.get(provider, []))
        )
        for provider, mode_options in MODEL_OPTIONS.items()
    }
