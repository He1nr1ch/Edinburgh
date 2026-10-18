"""Declarative per-model capability table for OpenAI-compatible providers.

The single place that knows which model IDs reject which API parameters or
need which structured-output method. Every model currently served (OpenAI,
Ollama, custom endpoints) takes the default; add a row to ``_BY_ID`` or
``_BY_PATTERN`` when a model needs a quirk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

StructuredMethod = Literal[
    "function_calling",  # uses tools; respects supports_tool_choice
    "json_mode",         # uses response_format={"type":"json_object"}
    "json_schema",       # uses response_format={"type":"json_schema",...}
    "none",              # no structured output available; caller falls back to free-text
]


@dataclass(frozen=True)
class ModelCapabilities:
    """What an OpenAI-compatible model accepts at the API level."""

    supports_tool_choice: bool
    supports_json_mode: bool
    supports_json_schema: bool
    preferred_structured_method: StructuredMethod


_DEFAULT = ModelCapabilities(
    supports_tool_choice=True,
    supports_json_mode=True,
    supports_json_schema=True,
    preferred_structured_method="function_calling",
)

# Exact-ID matches take precedence over pattern matches.
_BY_ID: dict[str, ModelCapabilities] = {}
_BY_PATTERN: list[tuple[re.Pattern[str], ModelCapabilities]] = []


def get_capabilities(model_name: str) -> ModelCapabilities:
    """Resolve capabilities by exact ID, then pattern, then default."""
    if model_name in _BY_ID:
        return _BY_ID[model_name]
    for pattern, caps in _BY_PATTERN:
        if pattern.match(model_name):
            return caps
    return _DEFAULT
