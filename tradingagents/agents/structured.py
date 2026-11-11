"""The typed contract every agent answers to, and what happens when an answer does not meet it.

Every agent follows the same pattern:

1. At agent creation, wrap the LLM with ``with_structured_output(Schema)``
   so the model returns a typed Pydantic instance (see ``schemas.py``).
2. At invocation, :func:`settle` runs the structured call. An answer that
   fails validation is asked for once more, with the error quoted back. The
   answer is stamped into an :class:`AgentVerdict` (call, thesis, confidence,
   evidence, the model that answered, how long it took), which the node puts
   in the run's ``verdicts`` for the feedback store and anything else that
   needs the call itself rather than the prose around it.
3. If the second attempt fails too, the contract is not met. By default the
   run is closed (``contract_fallback: fail``): the verdict is marked failed,
   ``contract_failure`` is set, the graph routes straight to the Portfolio
   Manager, and the ticker is rated REVIEW. With ``contract_fallback:
   free_text`` the agent answers in prose instead, as it did before, for a
   provider that offers no structured output.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from tradingagents import errors, usage
from tradingagents.agents.schemas import ENVELOPE_FIELDS, AgentVerdict, AnalystVerdict, Envelope

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Schema-only structured output binds exactly one tool (the schema itself), so a
# model that reaches for a search tool emits an unknown tool call and the whole
# structured attempt is discarded for a free-text retry. Agents on this path
# state the constraint explicitly rather than relying on the binding alone
# (#1130).
NO_EXTERNAL_TOOLS = (
    "Use only the evidence provided in this prompt. Do not call external tools "
    "or search the web; if something is missing, say so explicitly."
)


def bind_structured(llm: Any, schema: type[T], agent_name: str) -> Any | None:
    """Return ``llm.with_structured_output(schema)`` or ``None`` if unsupported.

    Logs a warning when the binding fails: the agent then cannot meet its
    contract at all, and every call takes the configured fallback.
    """
    try:
        return llm.with_structured_output(schema)
    except (NotImplementedError, AttributeError) as exc:
        logger.warning(
            "%s: provider does not support with_structured_output (%s); "
            "its contract cannot be met",
            agent_name, exc,
        )
        return None


ATTEMPTS = 2


def contract_fallback() -> str:
    """``fail`` (close the run) or ``free_text`` (answer in prose), from the run's config."""
    from tradingagents.dataflows.config import get_config

    return str(get_config().get("contract_fallback") or "fail").strip().lower()


def _corrected(prompt: Any, error: str) -> Any:
    """The prompt again, with what was wrong with the last answer."""
    note = (f"Your previous answer did not fit the required schema ({error}). Answer again: "
            "fill every required field, with numbers where numbers are asked for.")
    if isinstance(prompt, str):
        return f"{prompt}\n\n{note}"
    return [*prompt, {"role": "user", "content": note}]


def decide(structured_llm: Any | None, prompt: Any, agent_name: str, call: str | None = None,
           catch: type[BaseException] | tuple = Exception) -> tuple[T | None, AgentVerdict]:
    """Ask for the typed answer, once more if the first does not validate; the answer and its verdict.

    The answer is ``None`` when both attempts failed or the provider offers no
    structured output; the verdict then has ``ok`` False and the reason.
    ``call`` names the call for an agent whose schema does not carry one (a
    debater's stance is its role). ``catch`` is what counts as a failed
    answer; anything else (the provider unreachable, say) is raised.
    """
    result, error, attempts, answered = None, "", 0, False
    start = time.monotonic()
    with usage.capture() as records:
        if structured_llm is None:
            error = "the provider offers no structured output"
        else:
            ask = prompt
            for attempts in range(1, ATTEMPTS + 1):
                try:
                    with usage.as_agent(agent_name):
                        result = structured_llm.invoke(ask)
                    if not isinstance(result, BaseModel):
                        # A thinking model can answer in plain text instead of
                        # filling the schema, leaving the parser nothing to return.
                        result = None
                        raise ValueError("structured output returned no parsed result")
                    error = ""
                    break
                except catch as exc:
                    result = None
                    answered = isinstance(exc, ValueError)      # it answered, and the answer did not fit
                    error = f"{type(exc).__name__}: {' '.join(str(exc).split())[:300]}"
                    logger.warning("%s: answer %d of %d did not meet its contract (%s)", agent_name, attempts, ATTEMPTS, error)
                    ask = _corrected(prompt, error)
    verdict = AgentVerdict(
        agent=agent_name, ok=result is not None, error=error, attempts=attempts,
        latency_s=round(time.monotonic() - start, 2),
        model=(records[-1].get("model") if records else "") or "",
        tokens_in=sum(r.get("tokens_in") or 0 for r in records) if records else None,
        tokens_out=sum(r.get("tokens_out") or 0 for r in records) if records else None,
    )
    if result is None:
        if answered:    # no answer at all is the client's to record, as a timeout or an outage
            errors.record("contract", error, agent_name, verdict.model)
    else:
        verdict.fields = result.model_dump(mode="json", exclude=set(ENVELOPE_FIELDS))
        if isinstance(result, Envelope):
            verdict.call, verdict.thesis = call or result.call(), result.thesis()
            verdict.confidence, verdict.evidence = result.confidence, list(result.evidence)
    return result, verdict


def invoke_structured(structured_llm: Any | None, prompt: Any, agent_name: str) -> T | None:
    """The typed answer alone, or ``None`` when the contract was not met."""
    return decide(structured_llm, prompt, agent_name)[0]


@dataclass
class Outcome(Generic[T]):
    """What asking an agent gave: its typed answer, or prose under the free-text opt-out, or neither."""

    result: T | None
    verdict: AgentVerdict
    text: str | None = None

    @property
    def closed(self) -> bool:
        """No answer of either kind: the run closes here."""
        return self.result is None and self.text is None

    def markdown(self, render: Callable[[T], str]) -> str:
        """What the next agent and the report read: the rendered answer, the prose, or why there is none."""
        if self.result is not None:
            return render(self.result)
        return self.text if self.text is not None else closed_note(self.verdict)

    def update(self, key: str) -> dict:
        """The state update every node adds to its own: its verdict, and the closure when there is one."""
        out: dict = {"verdicts": {key: self.verdict.model_dump(mode="json")}}
        if self.closed:
            out["contract_failure"] = f"{self.verdict.agent}: {self.verdict.error}"
        return out


def closed_note(verdict: AgentVerdict) -> str:
    return (f"**No valid answer from the {verdict.agent}.** Its answer did not meet its contract after "
            f"{verdict.attempts or 'no'} attempt(s) ({verdict.error}); the run is closed without a call.")


def settle(structured_llm: Any | None, plain_llm: Any, prompt: Any, agent_name: str, call: str | None = None) -> Outcome:
    """Ask the agent for its typed answer and apply the configured fallback when the contract is not met.

    A closed run is for an answer that was given and did not fit (a parse or
    validation error, which is a ``ValueError``). The model being unreachable
    is not an answer: that is raised, so the run fails and can be tried again,
    instead of being recorded as a decision nobody made. Under the free-text
    opt-out any failure of the structured call leads to prose, as before.
    """
    free_text = contract_fallback() == "free_text"
    result, verdict = decide(structured_llm, prompt, agent_name, call, catch=Exception if free_text else ValueError)
    text = None
    if result is None and free_text:
        logger.warning("%s: answering in free text instead (contract_fallback=free_text)", agent_name)
        with usage.as_agent(agent_name):
            text = plain_llm.invoke(prompt).content
    return Outcome(result=result, verdict=verdict, text=text)


def analyst_verdict(llm: Any, agent_name: str, key: str, report: str, instrument_context: str = "") -> dict:
    """An analyst's report reduced to its typed verdict, as a state update; empty when switched off.

    One more short call on the analyst's own model. The report stands whatever
    this call gives: a verdict that cannot be had is recorded as failed and
    the run goes on, since the report is what the other agents read.
    """
    from tradingagents.dataflows.config import get_config

    config = get_config()
    if not report or not config.get("analyst_verdicts", True):
        return {}
    days = config.get("holding_period_days", 5)
    prompt = (
        f"You are the {agent_name}. Below is the report you have just written. {instrument_context}\n\n"
        f"State what it implies for the price over the next {days} trading days: Bullish, Neutral or Bearish, "
        "how sure you are, and the facts in the report that decide it. Use only the report; add nothing to it. "
        f"{NO_EXTERNAL_TOOLS}\n\n---\n\n{report}"
    )
    _, verdict = decide(bind_structured(llm, AnalystVerdict, agent_name), prompt, agent_name, catch=ValueError)
    return {"verdicts": {key: verdict.model_dump(mode="json")}}


def failed_upstream(state: dict) -> str:
    """The contract failure that closed this run, or ''."""
    return state.get("contract_failure") or ""


def merge_verdicts(left: dict | None, right: dict | None) -> dict:
    """The run's verdicts with a node's added; analysts working side by side each add their own."""
    return {**(left or {}), **(right or {})}
