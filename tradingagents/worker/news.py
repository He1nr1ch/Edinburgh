"""The news duty: read what was published about each position since the last look.

New articles about a held symbol go to the model with the position's side, so
"good" and "bad" are judged for the holder: a downgrade helps a short. The
model returns a structured judgement; the policy turns it into an order or an
alert.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Literal

from pydantic import BaseModel, Field

from tradingagents.agents.structured import NO_EXTERNAL_TOOLS, bind_structured, invoke_structured
from tradingagents.dataflows.vendors.alpaca import articles_since, get_asset_profile
from tradingagents.dataflows.vendors.alpaca.trading import Holding
from tradingagents.worker.duty import Finding, WorkerContext
from tradingagents.worker.policy import action_for

logger = logging.getLogger(__name__)


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
    surprise: bool = Field(
        description=(
            "True when the market did not expect it. Scheduled events that land as forecast, "
            "repeats of known stories and opinion pieces are not surprises."
        ))
    summary: str = Field(description="One or two sentences: what happened, with the numbers.")
    rationale: str = Field(description="Why it matters, or does not, for the position as held.")


def _name(symbol: str) -> str:
    """The instrument's name, so news tagged with its ticker about another company is caught."""
    try:
        return get_asset_profile(symbol).get("name") or ""
    except Exception:
        return ""


def _prompt(h: Holding, articles: list[dict], name: str = "") -> str:
    lines = [f"- {a['created_at']:%Y-%m-%d %H:%M} UTC · {a['source']} · {a['headline']}"
             + (f"\n  {a['summary']}" if a["summary"] else "") for a in articles]
    return (
        f"You watch a trading desk's open position during the market day.\n\n"
        f"Position: {h.side.upper()} {abs(h.qty):g} shares of {h.symbol}"
        + (f" ({name})" if name else "") + ", market value "
        f"{abs(h.market_value):,.0f}. A {'fall' if h.side == 'short' else 'rise'} in the price "
        f"makes money for this position.\n\n"
        f"News published since the last check:\n" + "\n".join(lines) + "\n\n"
        "Judge the news as a whole for this position. A feed sometimes tags an article with this "
        "ticker when it is about a different company or fund: such news has materiality none. "
        "Most news is routine: say so plainly rather than inflating it. " + NO_EXTERNAL_TOOLS
    )


def _severity(a: NewsAssessment) -> str:
    if a.materiality == "high" and a.surprise:
        return "urgent"
    if a.materiality in ("medium", "high"):
        return "notable"
    return "info"


class NewsDuty:
    name = "news"

    def run(self, context: WorkerContext) -> list[Finding]:
        structured = bind_structured(context.llm, NewsAssessment, "News watch")
        lookback = timedelta(hours=context.config["worker_news_lookback_hours"])
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
            assessment = invoke_structured(structured, _prompt(h, articles, _name(h.symbol)), "News watch")
            context.state.mark_seen(h.symbol, [a["id"] for a in articles])
            if assessment is None:
                findings.append(Finding(self.name, h.symbol, "notable",
                                        f"{len(articles)} new article(s); the model could not judge them",
                                        "\n".join(a["headline"] for a in articles)))
                continue
            reason = f"{assessment.materiality}, {assessment.direction}, " \
                     f"{'unexpected' if assessment.surprise else 'expected'}: {assessment.summary}"
            action = action_for(h, assessment.materiality, assessment.direction, assessment.surprise,
                                context.account.equity, context.config, reason)
            detail = (f"{assessment.summary}\n\n{assessment.rationale}\n\nHeadlines:\n"
                      + "\n".join(f"- {a['headline']} ({a['source']}) {a['url']}" for a in articles))
            findings.append(Finding(self.name, h.symbol, _severity(assessment),
                                    f"{h.side} {h.symbol}: {assessment.materiality} {assessment.direction} news"
                                    + ("" if assessment.surprise else " (expected)"),
                                    detail, action))
        return findings
