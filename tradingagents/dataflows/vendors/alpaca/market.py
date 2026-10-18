"""Alpaca daily bars: raw OHLCV, technical indicators, closes for settlement, the verified snapshot."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Annotated

import pandas as pd

from tradingagents.dataflows import indicators as ind, ohlcv
from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.errors import NoMarketDataError
from tradingagents.dataflows.files import replace_file
from tradingagents.dataflows.symbols import (
    is_crypto_symbol,
    normalize_symbol,
    safe_ticker_component,
)
from tradingagents.dataflows.vendors.alpaca import common

logger = logging.getLogger(__name__)

# Alpaca stamps a daily bar at the start of its session in New York time; the
# bar's date is that day's date in New York, for stocks and coins alike.
MARKET_TZ = "America/New_York"

_COLUMNS = {"timestamp": "Date", "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume"}


def _bars(canonical: str, start: datetime, end: datetime | None) -> pd.DataFrame:
    """Daily bars for ``canonical`` from ``start`` through ``end`` (inclusive), as an OHLCV frame.

    Raises NoMarketDataError when Alpaca has no bars for the symbol.
    """
    from alpaca.data.enums import Adjustment
    from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    if is_crypto_symbol(canonical):
        request = CryptoBarsRequest(symbol_or_symbols=canonical, timeframe=TimeFrame.Day,
                                    start=start, end=end)
        bars = common.call(lambda: common.crypto_client().get_crypto_bars(request), f"bars for {canonical}")
    else:
        request = StockBarsRequest(symbol_or_symbols=canonical, timeframe=TimeFrame.Day,
                                   start=start, end=end, adjustment=Adjustment.ALL,
                                   feed=common.data_feed())
        bars = common.call(lambda: common.stock_client().get_stock_bars(request), f"bars for {canonical}")

    frame = None if bars is None else bars.df
    if frame is None or frame.empty:
        raise NoMarketDataError(canonical, canonical, "no price bars")
    frame = frame.reset_index()
    if "symbol" in frame.columns:
        frame = frame[frame["symbol"] == canonical].drop(columns=["symbol"])
    frame = frame.rename(columns=_COLUMNS)[list(_COLUMNS.values())]
    stamps = pd.to_datetime(frame["Date"], utc=True).dt.tz_convert(MARKET_TZ)
    frame["Date"] = stamps.dt.tz_localize(None).dt.normalize()
    return frame.sort_values("Date").reset_index(drop=True)


def load_ohlcv(symbol: str, as_of_date: str, fill_gaps: bool = True) -> pd.DataFrame:
    """OHLCV up to ``as_of_date`` with caching, filtered to prevent look-ahead bias.

    Downloads five years of daily bars up to today and caches them per symbol;
    rows after ``as_of_date`` are dropped so a backtest never sees future prices.
    ``fill_gaps`` carries prices forward over gaps so indicators compute on a
    continuous series; pass ``False`` to read the values as reported.
    """
    canonical = normalize_symbol(symbol)
    safe_symbol = safe_ticker_component(canonical)
    config = get_config()
    as_of_dt = pd.to_datetime(as_of_date).normalize()
    now = pd.Timestamp.today()

    os.makedirs(config["data_cache_dir"], exist_ok=True)
    data_file = os.path.join(config["data_cache_dir"], f"{safe_symbol}-alpaca-daily.csv")

    # A cached file may be empty if a prior fetch failed. Treat an empty or
    # columnless cache as a miss rather than serving the poisoned file forever.
    data = None
    if os.path.exists(data_file):
        cached = pd.read_csv(data_file, on_bad_lines="skip", encoding="utf-8")
        if not cached.empty and "Close" in cached.columns and ohlcv.cache_is_fresh(data_file, as_of_dt, now):
            data = cached

    if data is None:
        start = (now - pd.DateOffset(years=5)).to_pydatetime()
        downloaded = _bars(canonical, start, None)
        replace_file(data_file, lambda temp: downloaded.to_csv(temp, index=False, encoding="utf-8"))
        data = downloaded

    return ohlcv.as_of_frame(data, symbol, canonical, as_of_date, fill_gaps=fill_gaps)


def get_stock_data(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """Daily OHLCV between the dates, inclusive, as CSV with a header comment."""
    datetime.strptime(start_date, "%Y-%m-%d")
    datetime.strptime(end_date, "%Y-%m-%d")
    canonical = normalize_symbol(symbol)

    data = load_ohlcv(symbol, end_date, fill_gaps=False)
    data = data[data["Date"] >= pd.to_datetime(start_date)]
    if data.empty:
        raise NoMarketDataError(symbol, canonical, f"no rows between {start_date} and {end_date}")

    data = data.copy()
    for col in ("Open", "High", "Low", "Close"):
        data[col] = data[col].round(2)
    data["Date"] = data["Date"].dt.strftime("%Y-%m-%d")

    label = canonical if canonical == symbol.upper() else f"{canonical} (from {symbol})"
    header = f"# Stock data for {label} from {start_date} to {end_date}\n"
    header += f"# Total records: {len(data)}\n\n"
    return header + data.to_csv(index=False)


def get_indicators(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    as_of_date: Annotated[str, "The current trading date you are trading on, YYYY-mm-dd"],
    look_back_days: Annotated[int, "how many days to look back"],
) -> str:
    """One technical indicator over the window ending at ``as_of_date``."""
    if indicator not in ind.INDICATOR_DESCRIPTIONS:
        raise ValueError(
            f"Indicator {indicator} is not supported. Please choose from: {list(ind.INDICATOR_DESCRIPTIONS)}"
        )
    data = load_ohlcv(symbol, as_of_date)
    return ind.render_indicator_window(data, indicator, as_of_date, look_back_days)


def get_market_snapshot(
    symbol: Annotated[str, "ticker symbol of the company"],
    as_of_date: Annotated[str, "the analysis date, YYYY-mm-dd"],
    look_back_days: Annotated[int, "recent rows to include"] = 30,
) -> str:
    """The deterministic verification snapshot: latest bar, indicators, recent closes."""
    data = load_ohlcv(symbol, as_of_date, fill_gaps=False)
    return ind.render_snapshot(normalize_symbol(symbol), data, as_of_date, look_back_days)


def get_closes(symbol: str, start_date: str, end_date: str) -> pd.Series:
    """Daily closes from ``start_date`` up to, not including, ``end_date``, indexed by date.

    Fetched directly rather than through the cache: settlement needs the bars
    after a decision date, which the cache cut at that date does not hold.
    """
    canonical = normalize_symbol(symbol)
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=1)
    if end < start:
        return pd.Series(dtype=float)
    try:
        frame = _bars(canonical, start, end)
    except NoMarketDataError:
        return pd.Series(dtype=float)
    frame = frame[(frame["Date"] >= pd.to_datetime(start_date)) & (frame["Date"] < pd.to_datetime(end_date))]
    return pd.Series(frame["Close"].to_numpy(dtype=float), index=pd.DatetimeIndex(frame["Date"]), name="Close")
