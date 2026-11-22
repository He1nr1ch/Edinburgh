"""The market around a stock, to its close: the S&P 500, the ten-year Treasury yield and oil, day by day.

FRED has all three, but late: read on a Sunday for Friday's close it had the
index and the yield to Thursday and the spot price of oil to Tuesday, and on
2026-11-20 the index fell 3.7% on the day it left out. An analyst reading a
stock against the market needs the market of the same day.

So the daily closes come from Yahoo's public chart, which has the session
that has just closed: the index itself (``^GSPC``), the ten-year yield
(``^TNX``) and the front-month WTI crude future (``CL=F``, the nearest
traded price to spot; FRED's spot series is the one that runs days behind).
Nothing after ``as_of_date`` is returned, so a past run sees what was known
then; these are market prices and are not revised. The report has the shape
of FRED's (``fred.get_macro_data``), so an analyst reads both alike.

When the source does not answer the caller falls back to FRED, late but
there (``agents/analysts/briefing.py``).
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import requests

ET = ZoneInfo("America/New_York")
CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
LOOKBACK_DAYS = 365
MAX_ROWS = 40

# alias (as the macro tool names it) -> symbol, title, units, decimals
SERIES = {
    "sp500": ("^GSPC", "S&P 500 index", "Index", 2),
    "10y_treasury": ("^TNX", "10-Year Treasury yield", "%", 2),
    "wti_crude": ("CL=F", "WTI crude oil, front-month future (the nearest traded price to spot)", "$ per barrel", 2),
}


def daily_closes(symbol: str, start: str, end: str) -> list[tuple[str, float]]:
    """The symbol's daily closes from ``start`` to ``end``, both included, oldest first."""
    first = datetime.combine(date.fromisoformat(start), datetime.min.time(), ET)
    last = datetime.combine(date.fromisoformat(end) + timedelta(days=1), datetime.min.time(), ET)
    response = requests.get(CHART_URL.format(symbol=symbol),
                            params={"interval": "1d", "period1": int(first.timestamp()), "period2": int(last.timestamp())},
                            headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    response.raise_for_status()
    chart = response.json()["chart"]
    if chart.get("error") or not chart.get("result"):
        raise ValueError(f"no chart for {symbol}: {chart.get('error')}")
    result = chart["result"][0]
    closes = result["indicators"]["quote"][0]["close"]
    days = [(datetime.fromtimestamp(t, ET).strftime("%Y-%m-%d"), float(c))
            for t, c in zip(result.get("timestamp") or [], closes, strict=False) if c is not None]
    return [(d, c) for d, c in days if start <= d <= end]


def get_market_series(name: str, as_of_date: str) -> str:
    """A year of one series to ``as_of_date`` as a markdown report; raises when it cannot be had."""
    symbol, title, units, places = SERIES[name]
    start = (date.fromisoformat(as_of_date) - timedelta(days=LOOKBACK_DAYS)).isoformat()
    points = daily_closes(symbol, start, as_of_date)
    if not points:
        raise ValueError(f"no closes for {symbol} from {start} to {as_of_date}")
    (first_day, first), (last_day, last) = points[0], points[-1]
    delta = last - first
    moved = f"{delta * 100:+.0f} bp" if units == "%" else f"{delta:+,.{places}f} ({delta / first * 100:+.2f}%)"
    lines = [f"## {title} ({symbol})", f"- Units: {units}", "- Frequency: Daily, close", f"- Window: {start} to {as_of_date}", "",
             f"**Latest:** {last:.{places}f} ({last_day}) | **Change from {first_day} to {last_day}:** {moved}, from {first:.{places}f}"]
    if len(points) >= 2:
        prev_day, prev = points[-2]
        step = last - prev
        lines.append(f"**On the day ({last_day} against {prev_day}):** "
                     + (f"{step * 100:+.0f} bp" if units == "%" else f"{step / prev * 100:+.2f}%"))
    shown = points[-MAX_ROWS:]
    if len(points) > MAX_ROWS:
        lines += ["", f"_(showing the most recent {MAX_ROWS} of {len(points)} closes)_"]
    lines += ["", "| Date | Value |", "| --- | --- |"] + [f"| {d} | {c:.{places}f} |" for d, c in shown]
    return "\n".join(lines) + "\n"
