"""Alpaca news (Benzinga-backed): articles about a symbol, and the market at large, in a date window."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.symbols import is_crypto_symbol, normalize_symbol
from tradingagents.dataflows.vendors.alpaca import common

logger = logging.getLogger(__name__)

# Alpaca serves at most this many articles per page.
_PAGE_SIZE = 50


def _news_symbol(canonical: str) -> str:
    """How Alpaca's news feed tags the symbol: a pair without its slash (``BTCUSD``)."""
    return canonical.replace("/", "") if is_crypto_symbol(canonical) else canonical


def _window(start_date: str, end_date: str) -> tuple[datetime, datetime]:
    """``[start 00:00, end 24:00)`` in UTC."""
    start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=UTC)
    end = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=UTC) + timedelta(days=1)
    return start, end


def fetch_articles(symbols: str | None, start: datetime, end: datetime, limit: int) -> list:
    """Up to ``limit`` articles in the window, newest first, across pages."""
    from alpaca.common.enums import Sort
    from alpaca.data.requests import NewsRequest

    articles: list = []
    page_token = None
    while len(articles) < limit:
        request = NewsRequest(
            symbols=symbols, start=start, end=end, limit=min(_PAGE_SIZE, limit - len(articles)),
            sort=Sort.DESC, include_content=False, exclude_contentless=False, page_token=page_token,
        )
        what = f"news for {symbols}" if symbols else "market news"
        page = common.call(lambda req=request: common.news_client().get_news(req), what)
        if page is None:
            break
        # alpaca-py's NewsSet keeps the articles under ``data["news"]``.
        data = getattr(page, "data", None)
        items = data.get("news", []) if isinstance(data, dict) else getattr(page, "news", None)
        if items is None:
            items = page.get("news", []) if isinstance(page, dict) else []
        articles.extend(items)
        page_token = getattr(page, "next_page_token", None)
        if not items or not page_token:
            break
    return articles[:limit]


def format_articles(articles) -> str:
    out = ""
    for article in articles:
        headline = getattr(article, "headline", None) or "No title"
        source = getattr(article, "source", None) or "Unknown"
        published = getattr(article, "created_at", None)
        when = f" {published:%Y-%m-%d}" if isinstance(published, datetime) else ""
        out += f"### {headline} (source: {source}{when})\n"
        summary = getattr(article, "summary", None)
        if summary:
            out += f"{summary}\n"
        url = getattr(article, "url", None)
        if url:
            out += f"Link: {url}\n"
        symbols = getattr(article, "symbols", None)
        if symbols:
            out += f"Symbols: {', '.join(symbols)}\n"
        out += "\n"
    return out


def get_news(ticker: str, start_date: str, end_date: str) -> str:
    """News about ``ticker`` published in the window, from Alpaca.

    An empty answer is a real absence: Alpaca's archive covers the window asked
    for, so no article in it means none was published, not that the feed did
    not look.
    """
    canonical = normalize_symbol(ticker)
    start, end = _window(start_date, end_date)
    limit = get_config()["news_article_limit"]
    articles = fetch_articles(_news_symbol(canonical), start, end, limit)
    header = f"## {canonical} News, from {start_date} to {end_date}:\n\n"
    if not articles:
        return header + f"No articles about {canonical} were published between {start_date} and {end_date}.\n"
    return header + format_articles(articles)


def get_global_news(curr_date: str, look_back_days: int | None = None, limit: int | None = None) -> str:
    """Market-wide news over the ``look_back_days`` ending at ``curr_date``, from Alpaca.

    Alpaca's feed is not tagged by topic, so the articles are the most recent
    across the whole market; the macro indicators tool grounds the macro view.
    """
    config = get_config()
    look_back_days = look_back_days if look_back_days is not None else config["global_news_lookback_days"]
    limit = limit if limit is not None else config["global_news_article_limit"]
    end_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    start_date = (end_dt - timedelta(days=look_back_days)).strftime("%Y-%m-%d")
    start, end = _window(start_date, curr_date)
    articles = fetch_articles(None, start, end, limit)
    header = f"## Market news, from {start_date} to {curr_date}:\n\n"
    if not articles:
        return header + f"No market news was published between {start_date} and {curr_date}.\n"
    return header + format_articles(articles)
