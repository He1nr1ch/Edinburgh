"""Ollama through its native API, so the context window is set per request.

Ollama's OpenAI-compatible endpoint takes no ``num_ctx``, and a server left at
its default context (4,096 tokens on the PGX box) silently truncates the prompt
once the analysts' tool results arrive, dropping the user message and failing
with "no user query found in messages". The native API accepts ``num_ctx`` on
every call, so the window is a config setting rather than a server setting.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from langchain_ollama import ChatOllama

from .base_client import BaseLLMClient, normalize_content
from .validators import validate_model

logger = logging.getLogger(__name__)

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


# Ollama answers a prompt longer than the context window by truncating it; once
# the user message is gone it fails with this message, which says nothing about
# the cause.
_TRUNCATED = "no user query found in messages"


class ContextWindowExceeded(RuntimeError):
    """The prompt did not fit the context window asked of Ollama."""


class NormalizedChatOllama(ChatOllama):
    """ChatOllama with content normalized to a string, and the context window watched.

    A prompt that fills most of the window is logged, and one that overflows it
    is reported as what it is, so the fix (a larger ``ollama_num_ctx``, or
    shorter reports) is named instead of an opaque server error.
    """

    def invoke(self, input, config=None, **kwargs):
        try:
            response = super().invoke(input, config, **kwargs)
        except Exception as exc:
            if _TRUNCATED in str(exc):
                raise ContextWindowExceeded(
                    f"The prompt did not fit Ollama's context window of {self.num_ctx} tokens on "
                    f"{self.model}; raise ollama_num_ctx (TRADINGAGENTS_OLLAMA_NUM_CTX) or shorten the reports."
                ) from exc
            raise
        used = (getattr(response, "usage_metadata", None) or {}).get("input_tokens")
        if used and self.num_ctx and used > 0.8 * self.num_ctx:
            logger.warning("Prompt used %d of the %d-token context window on %s", used, self.num_ctx, self.model)
        return normalize_content(response)


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
