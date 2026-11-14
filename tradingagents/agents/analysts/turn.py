"""One turn of a tool-using analyst, and its last turn once its tool rounds are spent."""

import json
import logging

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

logger = logging.getLogger(__name__)

WRAP_UP = (
    "You have used every tool round this report allows. Write your final report "
    "now from the tool results above, and say which data you could not retrieve."
)


def take_turn(prompt, llm, tools, messages, briefing=None):
    """The analyst's reply and its report, which stays empty while it calls tools.

    With a ``briefing`` (its data, fetched for it: see ``briefing.py``) it
    writes the report in this one turn, without tools.

    Told to wrap up, it answers without tools: a model offered tools may call one
    whatever it is told. Its earlier calls and their results reach it as text,
    since a provider may refuse a history holding tool calls when none are bound.
    """
    if messages and isinstance(messages[-1], HumanMessage) and messages[-1].content == WRAP_UP:
        # The prompt lists the tools it offers; a model told of tools it cannot
        # call may still try, and some providers then answer with nothing.
        prompt = prompt.partial(tool_names="none; your tool rounds are spent")
        result = (prompt | llm).invoke(_as_text(messages))
        return result, result.content
    if briefing:
        prompt = prompt.partial(tool_names="none in this turn; their results are in the last message")
        result = (prompt | llm).invoke([*messages, HumanMessage(content=briefing)])
        return result, result.content
    result = (prompt | llm.bind_tools(tools)).invoke(messages)
    return result, "" if result.tool_calls else result.content


def _as_text(messages):
    """The history with its tool calls and tool results written out as plain messages."""
    written = []
    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            said = message.content if isinstance(message.content, str) else ""
            calls = "; ".join(f"{c['name']}({json.dumps(c['args'], default=str)})" for c in message.tool_calls)
            written.append(AIMessage(content=f"{said}\n[Called {calls}]".strip()))
        elif isinstance(message, ToolMessage):
            written.append(HumanMessage(content=f"[{message.name or 'Tool'} returned]\n{message.content}"))
        else:
            written.append(message)
    return written


VERDICT_TOO = (
    "\n\nAnswer in the required form. `report` is the report in full. `signal` is what it implies for the "
    "price over the next {days} trading days (Bullish, Neutral or Bearish), `summary` the reason in a "
    "sentence or two, and `confidence` and `evidence` how sure you are and the facts in the report that "
    "decide it. The signal follows from the report: add nothing to it that the report does not say."
)


def report_and_verdict(prompt, llm, messages, handed, agent_name, key, instrument_context=""):
    """A briefed analyst's turn: its reply, its report and its verdict as a state update.

    One model call gives both the report and the typed verdict, which was a
    call of its own after every report: a short one, but with the report to
    read again and the model's thinking to do again, three times a ticker.
    When that one answer does not fit its form the analyst writes its report
    in prose and the verdict is asked for separately, as before; the report
    is what the other agents read and is never lost to a form.
    """
    from tradingagents.agents.schemas import AnalystReport
    from tradingagents.agents.structured import analyst_verdict, bind_structured, decide
    from tradingagents.dataflows.config import get_config

    config = get_config()
    if config.get("analyst_verdicts", True):
        ask = handed + VERDICT_TOO.format(days=config.get("holding_period_days", 5))
        asked = prompt.partial(tool_names="none in this turn; their results are in the last message").invoke(
            {"messages": [*messages, HumanMessage(content=ask)]}).to_messages()
        typed, verdict = decide(bind_structured(llm, AnalystReport, agent_name), asked, agent_name, catch=ValueError)
        if typed is not None and typed.report.strip():
            verdict.fields.pop("report", None)          # the report has its own place in the run's state
            return AIMessage(content=typed.report), typed.report, {"verdicts": {key: verdict.model_dump(mode="json")}}
        logger.warning("%s: its report and verdict did not come in one answer (%s); asking for them one after the other",
                       agent_name, verdict.error or "the report was empty")
    result, report = take_turn(prompt, llm, (), messages, handed)
    return result, report, analyst_verdict(llm, agent_name, key, report, instrument_context)
