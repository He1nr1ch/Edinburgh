"""Intraday bars for the P and B model: 15-minute bars, regular session, New York time, cached once the day is done."""

from __future__ import annotations

import logging
import os
from datetime import datetime, time, timedelta

import pandas as pd

from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.errors import NoMarketDataError
from tradingagents.dataflows.files import replace_file
from tradingagents.dataflows.symbols import (
    is_crypto_symbol,
    normalize_symbol,
    safe_ticker_component,
)
from tradingagents.dataflows.vendors.alpaca import intraday_bars as alpaca_intraday_bars

logger = logging.getLogger(__name__)

SESSION_OPEN, SESSION_CLOSE = time(9, 30), time(16, 0)


def intraday_bars(symbol: str, start: str, end: str, minutes: int = 15,
                  regular_session: bool = True) -> pd.DataFrame:
    """``minutes``-minute bars from ``start`` through ``end`` (dates, inclusive), New York wall-clock.

    Stocks keep the regular session only, 09:30 to 16:00, so an overnight gap
    is a gap between bars and not a bar of its own. Crypto trades round the
    clock and keeps everything. A window of finished days is cached under the
    data cache; one that reaches today is fetched every time.
    """
    canonical = normalize_symbol(symbol)
    end_day = datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)
    cache_dir = get_config()["data_cache_dir"]
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f"{safe_ticker_component(canonical)}-alpaca-{minutes}min-{start}-{end}.csv")
    finished = end_day <= datetime.now()
    if finished and os.path.exists(cache):
        frame = pd.read_csv(cache, encoding="utf-8")
        if not frame.empty and "Close" in frame.columns:
            frame["Date"] = pd.to_datetime(frame["Date"])
            return frame

    frame = alpaca_intraday_bars(canonical, start, end, minutes)
    if regular_session and not is_crypto_symbol(canonical):
        frame = regular_session_only(frame).reset_index(drop=True)
    if frame.empty:
        raise NoMarketDataError(symbol, canonical, f"no regular-session bars between {start} and {end}")
    if finished:
        replace_file(cache, lambda temp: frame.to_csv(temp, index=False, encoding="utf-8"))
    return frame


def regular_session_only(frame: pd.DataFrame) -> pd.DataFrame:
    """The bars that start within the regular session, 09:30 to 16:00 New York."""
    clock = pd.to_datetime(frame["Date"]).dt.time
    return frame[(clock >= SESSION_OPEN) & (clock < SESSION_CLOSE)]
