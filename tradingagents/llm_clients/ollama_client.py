"""Ollama through its native API, so the context window is set per request.

Ollama's OpenAI-compatible endpoint takes no ``num_ctx``, and a server left at
its default context (4,096 tokens on the PGX box) silently truncates the prompt
once the analysts' tool results arrive, dropping the user message and failing
with "no user query found in messages". The native API accepts ``num_ctx`` on
every call, so the window is a config setting rather than a server setting.
"""

from __future__ import annotations

import os
from typing import Any

from langchain_ollama import ChatOllama

from .base_client import BaseLLMClient, normalize_content
from .validators import validate_model

DEFAULT_BASE_URL = "http://localhost:11434"
BASE_URL_ENV = "OLLAMA_BASE_URL"

# Kwargs forwarded from the run config to ChatOllama. ``max_tokens`` is the
# cross-provider name; Ollama calls it ``num_predict``.
_PASSTHROUGH_KWARGS = ("temperature", "num_ctx", "callbacks", "keep_alive", "reasoning")
_RENAMED_KWARGS = {"max_tokens": "num_predict"}


def host_url(url: str | None) -> str:
    """The server root ChatOllama wants: the configured URL without an OpenAI ``/v1`` suffix."""
    url = (url or "").strip().rstrip("/")
    if url.endswith("/v1"):
        url = url[: -len("/v1")]
    return url or DEFAULT_BASE_URL


class NormalizedChatOllama(ChatOllama):
    """ChatOllama with content normalized to a string."""

    def invoke(self, input, config=None, **kwargs):
        return normalize_content(super().invoke(input, config, **kwargs))


class OllamaClient(BaseLLMClient):
    """Client for a local or LAN Ollama server."""

    provider = "ollama"

    def get_llm(self) -> Any:
        """A ChatOllama for the model, at the explicit URL, else OLLAMA_BASE_URL, else localhost."""
        self.warn_if_unknown_model()
        llm_kwargs: dict[str, Any] = {
            "model": self.model,
            "base_url": host_url(self.base_url or os.environ.get(BASE_URL_ENV)),
        }
        for key in _PASSTHROUGH_KWARGS:
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]
        for key, renamed in _RENAMED_KWARGS.items():
            if key in self.kwargs:
                llm_kwargs[renamed] = self.kwargs[key]
        return NormalizedChatOllama(**llm_kwargs)

    def validate_model(self) -> bool:
        """Any pulled model is valid; the server says if it is not there."""
        return validate_model("ollama", self.model)
