"""An analyst's data, fetched for it and handed over in one message.

Left to its tools, an analyst asks for its data a piece at a time: the price
history, then an indicator, then another, each a round trip to the model that
re-reads everything fetched so far and thinks again about what to ask next.
In the sweep the Market Analyst took four model calls a ticker, the News
Analyst three and the Fundamentals Analyst nearly three, and the three of
them 58% of the model time, though what they ask for hardly differs from one
ticker to the next.

So the data each analyst is known to use is fetched before it is asked
anything, through the same vendor routes its tools use and no later than the
run's date, and the analyst writes its report from that in one call. What
does not depend on the ticker (the market-wide news, the macro series) is
fetched once a day for the whole sweep and kept in the cache folder.

``analyst_briefing`` turns this off; the analyst then calls its tools as
before. It does so too when nothing at all could be fetched for it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path

from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.files import replace_file
from tradingagents.dataflows.router import route_to_vendor

logger = logging.getLogger(__name__)

# Every indicator the Market Analyst may choose from (its prompt names them).
INDICATORS = ("close_50_sma", "close_200_sma", "close_10_ema", "macd", "macds", "macdh", "rsi",
              "boll", "boll_ub", "boll_lb", "atr", "vwma", "mfi")
# The macro series a news report on a US equity draws on.
MACRO = ("fed_funds_rate", "10y_treasury", "yield_curve", "cpi", "unemployment", "vix")
PRICE_HISTORY_DAYS = 120
INDICATOR_DAYS = 30
NEWS_DAYS = 7

CLOSING = (
    "All the data this report may use is above: it was retrieved for you, as of {date}, through the same "
    "tools you would have called. No tool is available in this turn, so do not ask for more; write your "
    "final report now. Where a section says its data could not be retrieved, say so in the report rather "
    "than fill the gap."
)


def _back(day: str, days: int) -> str:
    return (date.fromisoformat(day) - timedelta(days=days)).isoformat()


def _shared(name: str, day: str, fetch: Callable[[], str]) -> str:
    """``fetch()``, kept for the day: the same for every ticker, so the sweep asks once, not five hundred times."""
    folder = get_config().get("data_cache_dir")
    if not folder:
        return fetch()
    path = Path(folder) / "briefing" / day / f"{name}.txt"
    if path.is_file():
        return path.read_text(encoding="utf-8")
    text = fetch()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        replace_file(path, lambda temp: Path(temp).write_text(text, encoding="utf-8"))
    except OSError as exc:
        logger.warning("The day's %s could not be kept for the next ticker: %s", name, exc)
    return text


def _compose(sections: list[tuple[str, Callable[[], str]]], day: str) -> str | None:
    """The sections fetched and laid out, each under the tool call it stands for; None when none could be had."""
    limit = int(get_config().get("max_tool_output_chars") or 16_000)
    parts, got = [], 0
    for title, fetch in sections:
        try:
            text = str(fetch())
            got += 1
        except Exception as exc:
            logger.warning("Briefing: %s could not be retrieved: %s", title, exc)
            text = f"Could not be retrieved ({type(exc).__name__}: {exc})."
        if len(text) > limit:
            text = text[:limit] + f"\n[... cut at {limit:,} characters]"
        parts.append(f"[{title} returned]\n{text}")
    if not got:
        return None
    return "\n\n".join(parts) + "\n\n" + CLOSING.format(date=day)


def market(symbol: str, day: str) -> str | None:
    """Price history, the verified snapshot and every indicator the analyst may choose from."""
    sections = [
        (f"get_stock_data({_back(day, PRICE_HISTORY_DAYS)} to {day})",
         lambda: route_to_vendor("get_stock_data", symbol, _back(day, PRICE_HISTORY_DAYS), day)),
        ("get_verified_market_snapshot", lambda: route_to_vendor("get_market_snapshot", symbol, day, INDICATOR_DAYS)),
    ]
    sections += [(f"get_indicators({name}, {INDICATOR_DAYS} days)",
                  lambda name=name: route_to_vendor("get_indicators", symbol, name, day, INDICATOR_DAYS))
                 for name in INDICATORS]
    return _compose(sections, day)


def news(symbol: str, day: str) -> str | None:
    """The instrument's news for the week, the market-wide news and the macro series (the last two once a day)."""
    sections = [
        (f"get_news({_back(day, NEWS_DAYS)} to {day})",
         lambda: route_to_vendor("get_news", symbol, _back(day, NEWS_DAYS), day)),
        ("get_global_news", lambda: _shared("global_news", day, lambda: route_to_vendor("get_global_news", day, None, None))),
    ]
    sections += [(f"get_macro_indicators({name})",
                  lambda name=name: _shared(f"macro_{name}", day,
                                            lambda: route_to_vendor("get_macro_indicators", name, day, None)))
                 for name in MACRO]
    return _compose(sections, day)


def fundamentals(symbol: str, day: str) -> str | None:
    """The three statements as filed, quarterly and annual."""
    sections = [(f"{method}({freq})", lambda method=method, freq=freq: route_to_vendor(method, symbol, freq, day))
                for freq in ("quarterly", "annual")
                for method in ("get_balance_sheet", "get_cashflow", "get_income_statement")]
    return _compose(sections, day)


def brief(build: Callable[[str, str], str | None], state: dict) -> str | None:
    """The analyst's briefing for this run, or None when it is to use its tools.

    None when briefings are off, when the analyst has already begun calling
    tools (a run resumed from a checkpoint), or when nothing could be fetched.
    """
    if not get_config().get("analyst_briefing", True):
        return None
    if any(getattr(m, "tool_calls", None) or type(m).__name__ == "ToolMessage" for m in state.get("messages") or []):
        return None
    try:
        return build(state["company_of_interest"], state["trade_date"])
    except Exception as exc:        # a briefing that cannot be built is not the run's failure: the tools remain
        logger.warning("No briefing for %s: %s", state.get("company_of_interest"), exc)
        return None
