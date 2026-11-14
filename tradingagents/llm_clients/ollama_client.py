"""The desk's one inference client: Ollama through its native API, behind PAIR.

Every model call the desk makes goes through :class:`NormalizedChatOllama`,
built by :class:`OllamaClient`; nothing else talks to the server. Each call
has, as settings rather than server state:

- its **model** and **context window**. Ollama's OpenAI-compatible endpoint
  takes no ``num_ctx``, and a server left at its default context (4,096 tokens
  on the PGX box) silently truncates the prompt once the analysts' tool results
  arrive, dropping the user message and failing with "no user query found in
  messages". The native API accepts ``num_ctx`` on every call.
- two **time limits**. Every call is streamed, so the idle limit (the longest
  silence between chunks, which includes the wait in PAIR's queue) catches a
  server that hangs, and the total limit catches a call that never ends. Past
  either, the call is dropped, which frees the node, and :class:`InferenceTimeout`
  is raised.

:meth:`NormalizedChatOllama.with_call` gives a copy with a different model,
window or limit on the same connection, which is how an agent gets settings of
its own (``agent_calls`` in the config, see ``factory.for_agent``).

**Fail closed.** A call that times out or cannot reach the server raises an
:class:`InferenceError`; it never returns something a caller could take for an
answer. What each caller does with that is its own rule, and the rule is the
same everywhere: an error is not a confirmation. A ticker's run fails and is
tried again later; the Watcher leaves the news unread and opens nothing.

**PAIR.** NVIDIA PAIR answers on this machine's loopback and schedules each
call on one of the cluster's nodes. The response does not say which; the
router's own history does, a few seconds later, so each answered call is
looked up there once the history has caught up and its usage record is given
the node (see ``usage.resolve_nodes`` and ``pair.served_by``). :func:`check_endpoint`
refuses a PAIR machine pointed at anything but its own loopback.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any
from urllib.parse import urlparse

import httpx
from langchain_ollama import ChatOllama
from pydantic import PrivateAttr

from tradingagents import errors, lane, pair, usage

from .base_client import BaseLLMClient, normalize_content
from .validators import validate_model

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:11434"
BASE_URL_ENV = "OLLAMA_BASE_URL"

# Kwargs forwarded from the run config to ChatOllama. ``max_tokens`` is the
# cross-provider name; Ollama calls it ``num_predict``.
_PASSTHROUGH_KWARGS = ("temperature", "num_ctx", "callbacks", "keep_alive", "reasoning", "idle_timeout", "call_timeout")
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


class InferenceError(RuntimeError):
    """The model gave no answer: it could not be reached, or it ran out of time.

    Never a ``ValueError``: an answer that does not fit its schema is one, and
    the two must not be confused (see ``agents.structured.settle``).
    """


class InferenceTimeout(InferenceError):
    """The call passed its idle or its total time limit and was dropped."""


class InferenceUnavailable(InferenceError):
    """The server could not be reached, or dropped the connection."""


CONNECT_SECONDS = 10.0
LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")


def is_loopback(url: str) -> bool:
    return (urlparse(url).hostname or "") in LOOPBACK_HOSTS


def check_endpoint(url: str, config: dict | None = None) -> None:
    """Refuse a PAIR machine whose client points anywhere but its own loopback.

    PAIR's proxy takes plaintext only from the machine it runs on; another
    node's address refuses the call, and plain Ollama on another host would
    bypass the router. Where there is no PAIR (no router folder), any Ollama
    address is fine.
    """
    if pair.available(pair.pair_dir(config)) and not is_loopback(url):
        raise ValueError(
            f"This machine runs PAIR, but the model endpoint is {url}. PAIR answers only on its own "
            "loopback: set OLLAMA_BASE_URL to http://127.0.0.1:<port of this machine's nvpair-proxy> "
            "(find it with: ss -ltnp | grep nvpair-proxy)."
        )


class NormalizedChatOllama(ChatOllama):
    """ChatOllama with content normalized to a string, time limits, and the context window watched.

    A prompt that fills most of the window is logged, and one that overflows it
    is reported as what it is, so the fix (a larger ``ollama_num_ctx``, or
    shorter reports) is named instead of an opaque server error. Every answered
    call is recorded in the usage log (tokens, seconds, the agent that asked,
    the PAIR node that served it).
    """

    idle_timeout: float | None = None     # seconds of silence allowed between chunks (the queue wait included)
    call_timeout: float | None = None     # seconds allowed for the whole call
    _lane_node: str = PrivateAttr(default="")   # set on the copy that sends down the direct lane: the machine at its end

    def _down_the_lane(self, url: str, node: str):
        """A copy of this model whose calls go to the lane's gateway instead of PAIR (see ``tradingagents.lane``)."""
        from ollama import Client

        twin = self.model_copy(update={"base_url": url})
        twin._client = Client(host=url, **{**(self.client_kwargs or {}), **(self.sync_client_kwargs or {})})
        twin._lane_node = node
        return twin

    def with_call(self, model: str | None = None, num_ctx: int | None = None, timeout: float | None = None):
        """A copy for calls with their own model, context window or total time limit, on the same connection."""
        update = {k: v for k, v in (("model", model), ("num_ctx", num_ctx), ("call_timeout", timeout)) if v}
        return self.model_copy(update=update) if update else self

    def _create_chat_stream(self, messages, stop=None, **kwargs):
        deadline = time.monotonic() + self.call_timeout if self.call_timeout else None
        stream = super()._create_chat_stream(messages, stop, **kwargs)
        try:
            for part in stream:
                yield part
                if deadline is not None and time.monotonic() > deadline:
                    raise InferenceTimeout(
                        f"{self.model} was still answering after {self.call_timeout:g} s, its limit for one call; "
                        "the call was dropped (raise ollama_call_timeout, or the agent's own in agent_calls, if this was a healthy call)."
                    )
        except httpx.TimeoutException as exc:
            raise InferenceTimeout(
                f"{self.model} sent nothing for {self.idle_timeout:g} s ({type(exc).__name__}); the call was dropped. "
                "Either the server hangs or PAIR's queue is longer than ollama_idle_timeout."
            ) from exc
        except (httpx.TransportError, ConnectionError) as exc:
            raise InferenceUnavailable(f"The model server at {self.base_url} could not be reached: {exc}") from exc
        finally:
            stream.close()      # a dropped call must leave the server too, so PAIR frees the node

    def invoke(self, input, config=None, **kwargs):
        if not self._lane_node:
            from tradingagents.dataflows.config import get_config

            with lane.claim(get_config()) as place:
                if place:
                    url, node = place
                    try:
                        return self._down_the_lane(url, node).invoke(input, config, **kwargs)
                    except Exception as exc:
                        # The lane's fault is never the call's: it goes through PAIR.
                        # A 503 is the gateway saying its owner is back; anything
                        # else it answers (the model is not on that machine) or
                        # fails to answer keeps the lane out of use for longer.
                        status = getattr(exc, "status_code", None)
                        if status == 503:
                            lane.shut(url, lane.SHUT_AFTER_REFUSAL)
                        elif status is not None or isinstance(exc, InferenceUnavailable):
                            lane.shut(url, lane.SHUT_AFTER_FAULT, f"{type(exc).__name__}: {exc}")
                        else:
                            raise
        started = time.time()
        try:
            response = super().invoke(input, config, **kwargs)
        except Exception as exc:
            if self._lane_node and (getattr(exc, "status_code", None) is not None or isinstance(exc, InferenceUnavailable)):
                raise           # the caller above sends it through PAIR; not an error of the desk's model
            agent = usage.agent_from(config)
            if _TRUNCATED in str(exc):
                error = ContextWindowExceeded(
                    f"The prompt did not fit Ollama's context window of {self.num_ctx} tokens on "
                    f"{self.model}; raise ollama_num_ctx (TRADINGAGENTS_OLLAMA_NUM_CTX) or shorten the reports."
                )
                errors.record("context", error, agent, self.model)
                raise error from exc
            if isinstance(exc, InferenceError):
                errors.record("timeout" if isinstance(exc, InferenceTimeout) else "unavailable", exc, agent, self.model)
            raise
        usage.record_response(response, self.model, config, started=started, ended=time.time(), node=self._lane_node)
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
        check_endpoint(llm_kwargs["base_url"])
        # The read limit is the idle limit: the stream's longest allowed silence.
        llm_kwargs["client_kwargs"] = {"timeout": httpx.Timeout(
            connect=CONNECT_SECONDS, read=llm_kwargs.get("idle_timeout"), write=60.0, pool=CONNECT_SECONDS)}
        return NormalizedChatOllama(**llm_kwargs)

    def validate_model(self) -> bool:
        """Any pulled model is valid; the server says if it is not there."""
        return validate_model("ollama", self.model)
