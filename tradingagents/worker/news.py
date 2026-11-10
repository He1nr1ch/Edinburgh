"""The Watcher: news on a held position that the desk did not account for.

Every pass reads what was published about each position since the last look
and asks the model to judge it against the desk's own case for the position:
the Portfolio Manager's thesis and the news the News Analyst had already read
when the call was made. News the thesis covers is not a surprise; news it
does not is, and material adverse news of that kind closes or trims the
position on the same pass, before a loss becomes a disaster. The model
returns a structured judgement; the policy turns it into an order or an alert.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from tradingagents.agents.structured import NO_EXTERNAL_TOOLS, bind_structured, invoke_structured
from tradingagents.dataflows.vendors.alpaca import articles_since, get_asset_profile
from tradingagents.dataflows.vendors.alpaca.trading import Holding
from tradingagents.lessons import desk_lessons
from tradingagents.web import data
from tradingagents.worker.duty import Finding, WorkerContext
from tradingagents.worker.policy import action_for

logger = logging.getLogger(__name__)

# How much of the desk's own news reading goes into the prompt.
_KNOWN_NEWS_CHARS = 2500


class NewsAssessment(BaseModel):
    materiality: Literal["none", "low", "medium", "high"] = Field(
        description=(
            "How much the news could move this stock over the next few days. high: changes the "
            "investment case (guidance cut, fraud, M&A, a regulator's action, a major contract); "
            "medium: a real but bounded effect (an analyst rating change, a product delay); "
            "low: routine coverage; none: not about this company."
        ))
    direction: Literal["favorable", "adverse", "mixed"] = Field(
        description="Whether the news helps or hurts THIS position as held, long or short.")
    accounted_for: bool = Field(
        description=(
            "True when the desk's case for the position already covers this: the thesis names the "
            "risk or the event, the news the desk read at the time reported it, or it is a scheduled "
            "event landing as expected. False when it is new information the desk did not have."
        ))
    summary: str = Field(description="One or two sentences: what happened, with the numbers.")
    rationale: str = Field(description="Why it matters, or does not, for the position as held, against the desk's case.")


def _name(symbol: str) -> str:
    """The instrument's name, so news tagged with its ticker about another company is caught."""
    try:
        return get_asset_profile(symbol).get("name") or ""
    except Exception:
        return ""


def _field(text: str, name: str) -> str:
    m = re.search(rf"\*\*{name}\*\*:\s*(.+?)(?=\n\n\*\*|\Z)", text or "", re.S)
    return m.group(1).strip() if m else ""


def desk_case(symbol: str, decisions: dict[str, data.DecisionRef]) -> dict:
    """What the desk decided and knew about ``symbol``: thesis, levels, and the news it read."""
    ref = decisions.get(symbol.upper())
    if ref is None:
        return {}
    state = data.load_decision(ref)
    pm = state.get("final_trade_decision") or ""
    summary = data.decision_summary(state, None, 0)
    known = (state.get("news_report") or "").strip()
    return {
        "date": ref.date, "rating": summary["rating"], "target": summary["target"], "stop": summary["stop"],
        "horizon": summary["horizon"], "summary": summary["summary"],
        "thesis": _field(pm, "Investment Thesis"),
        "known_news": known[:_KNOWN_NEWS_CHARS] + (" …" if len(known) > _KNOWN_NEWS_CHARS else ""),
    }


def _prompt(h: Holding, articles: list[dict], name: str = "", case: dict | None = None) -> str:
    lines = [f"- {a['created_at']:%Y-%m-%d %H:%M} UTC · {a['source']} · {a['headline']}"
             + (f"\n  {a['summary']}" if a["summary"] else "") for a in articles]
    text = (
        f"You are the Watcher on a trading desk, guarding an open position during the market day.\n\n"
        f"Position: {h.side.upper()} {abs(h.qty):g} shares of {h.symbol}"
        + (f" ({name})" if name else "") + ", market value "
        f"{abs(h.market_value):,.0f}. A {'fall' if h.side == 'short' else 'rise'} in the price "
        f"makes money for this position.\n\n"
    )
    if case:
        text += (
            f"The desk's case, decided on {case['date']} ({case['rating']}"
            + (f", target {case['target']:,.2f}" if case.get("target") else "")
            + (f", stop {case['stop']:,.2f}" if case.get("stop") else "")
            + (f", {case['horizon']}" if case.get("horizon") else "") + "):\n"
            + (f"{case['summary']}\n" if case.get("summary") else "")
            + (f"{case['thesis']}\n" if case.get("thesis") else "")
            + (f"\nNews the desk had already read when it decided:\n{case['known_news']}\n" if case.get("known_news") else "")
            + "\n"
        )
    else:
        text += "The desk has no saved case for this position.\n\n"
    text += (
        f"News published since the last check:\n" + "\n".join(lines) + "\n\n"
        "Judge the news as a whole for this position, against the desk's case. What the thesis "
        "already names, or the desk had already read, is accounted for; what it did not have is "
        "not. A feed sometimes tags an article with this ticker when it is about a different "
        "company or fund: such news has materiality none. Most news is routine: say so plainly "
        "rather than inflating it. " + desk_lessons("Watcher", before="\n\n", after="\n\n") + NO_EXTERNAL_TOOLS
    )
    return text


def _severity(a: NewsAssessment) -> str:
    if a.materiality == "high" and not a.accounted_for:
        return "urgent"
    if a.materiality in ("medium", "high"):
        return "notable"
    return "info"


class WatcherDuty:
    name = "watcher"

    def run(self, context: WorkerContext) -> list[Finding]:
        structured = bind_structured(context.llm, NewsAssessment, "Watcher")
        lookback = timedelta(hours=context.config["worker_news_lookback_hours"])
        decisions = data.find_decisions(Path(context.config["results_dir"]))
        findings = []
        for h in context.watched():
            since_text = context.state.last_check.get(h.symbol)
            since = datetime.fromisoformat(since_text) if since_text else context.now - lookback
            try:
                articles = [a for a in articles_since(h.symbol, since)
                            if a["id"] and context.state.is_new(h.symbol, a["id"])]
            except Exception as exc:     # one symbol's feed failing must not stop the pass
                logger.warning("News for %s could not be read: %s", h.symbol, exc)
                continue
            context.state.last_check[h.symbol] = context.now.astimezone(UTC).isoformat()
            if not articles:
                continue
            case = desk_case(h.symbol, decisions)
            assessment = invoke_structured(structured, _prompt(h, articles, _name(h.symbol), case), "Watcher")
            context.state.mark_seen(h.symbol, [a["id"] for a in articles])
            if assessment is None:
                findings.append(Finding(self.name, h.symbol, "notable",
                                        f"{len(articles)} new article(s); the model could not judge them",
                                        "\n".join(a["headline"] for a in articles)))
                continue
            surprise = not assessment.accounted_for
            reason = f"{assessment.materiality}, {assessment.direction}, " \
                     f"{'not accounted for' if surprise else 'accounted for'}: {assessment.summary}"
            action = action_for(h, assessment.materiality, assessment.direction, surprise,
                                context.account.equity, context.config, reason)
            detail = (f"{assessment.summary}\n\n{assessment.rationale}\n\nHeadlines:\n"
                      + "\n".join(f"- {a['headline']} ({a['source']}) {a['url']}" for a in articles))
            findings.append(Finding(self.name, h.symbol, _severity(assessment),
                                    f"{h.side} {h.symbol}: {assessment.materiality} {assessment.direction} news"
                                    + (" (accounted for)" if assessment.accounted_for else ", not accounted for"),
                                    detail, action))
        return findings


NewsDuty = WatcherDuty    # the duty's earlier name
