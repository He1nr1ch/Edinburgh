"""OHLCV frames as the indicators and the snapshot read them: dated, numeric, gap-free.

Vendor-neutral: a vendor module downloads bars into a frame with a ``Date``
column and ``Open``/``High``/``Low``/``Close``/``Volume`` columns, and the
helpers here normalize it, cut it at the analysis date, and refuse a stale one.
"""

from __future__ import annotations

import logging
import os

import pandas as pd

from tradingagents.dataflows.errors import NoMarketDataError

logger = logging.getLogger(__name__)

PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

# A vendor's latest OHLCV row this many calendar days before the requested date
# is treated as stale. Generous enough to span long holiday weekends, tight
# enough to catch a frame that stopped updating.
MAX_OHLCV_STALE_DAYS = 10

# How long a same-day cache that does not yet reach the requested day may be
# reused before it is refetched. Short enough that an intraday run picks up
# today's close soon after it publishes, long enough that a day with no bar at
# all (weekend, holiday) cannot trigger a download on every call.
OHLCV_CACHE_TTL_SECONDS = 900


def _local_midnight(value) -> pd.Timestamp:
    """A single timestamp as its naive, midnight-normalized local date (or NaT)."""
    if pd.isna(value):
        return pd.NaT
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError):
        return pd.NaT
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)  # drop tz, keep the local wall-clock date
    return ts.normalize()


def normalize_dates(dates) -> pd.Series:
    """Parse to naive, midnight-normalized dates so tz-aware or intraday
    timestamps compare correctly against the naive ``as_of_date`` cutoff.

    Normalized per element: a cache CSV round-trips offsets as strings and the
    series can carry mixed UTC offsets across daylight-saving changes, which
    ``pd.to_datetime`` cannot unify without ``utc=True``.
    """
    return pd.to_datetime(pd.Series(dates).map(_local_midnight))


def clean_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize a price frame for stockstats: parse/normalize dates and coerce
    prices to numeric (NaN where invalid). Dropping incomplete rows and filling
    gaps is left to ``fill_price_gaps`` so the caller can first inspect the
    latest in-range bar."""
    data = data.copy()
    data["Date"] = normalize_dates(data["Date"])
    data = data.dropna(subset=["Date"]).copy()

    price_cols = [c for c in PRICE_COLUMNS if c in data.columns]
    data[price_cols] = data[price_cols].apply(pd.to_numeric, errors="coerce")
    return data


def fill_price_gaps(data: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with no close and forward/back-fill remaining price gaps so
    indicators compute on a continuous series."""
    price_cols = [c for c in PRICE_COLUMNS if c in data.columns]
    data = data.dropna(subset=["Close"]).copy()
    data[price_cols] = data[price_cols].ffill().bfill()
    return data


def assert_not_stale(
    data: pd.DataFrame,
    as_of_date: str,
    symbol: str,
    canonical: str | None = None,
    *,
    max_stale_days: int = MAX_OHLCV_STALE_DAYS,
) -> None:
    """Reject OHLCV whose latest row is far older than as_of_date.

    Raises NoMarketDataError (with a stale-specific detail) so the router treats
    it like any other "no usable data from this vendor". Empty frames are left
    to the caller's existing no-data handling; this guards only the dangerous
    case of present-but-stale rows.
    """
    if data is None or data.empty:
        return
    requested = pd.to_datetime(as_of_date, errors="coerce")
    if pd.isna(requested):
        return
    requested = requested.normalize()
    dates = pd.to_datetime(data["Date"], errors="coerce").dropna()
    if dates.empty:
        return
    latest = dates.max().normalize()
    stale_days = (requested - latest).days
    if stale_days > max_stale_days:
        raise NoMarketDataError(
            symbol,
            canonical,
            f"latest row is {latest.date()}, {stale_days} days before the "
            f"requested {requested.date()} (stale) — refusing to use it",
        )


def cache_is_fresh(data_file, as_of_dt, now) -> bool:
    """Whether the symbol's cached download can serve this request.

    The file holds the download made on the day it was written, so it serves
    only that day. A current-day request also refetches once the file is older
    than the TTL, so an intraday run picks up the close once it publishes.
    """
    written = pd.Timestamp.fromtimestamp(os.path.getmtime(data_file))
    if written.date() != now.date():
        return False
    return as_of_dt.date() < now.date() or (now - written).total_seconds() <= OHLCV_CACHE_TTL_SECONDS


def as_of_frame(data: pd.DataFrame, symbol: str, canonical: str, as_of_date: str,
                fill_gaps: bool = True) -> pd.DataFrame:
    """The frame cut at ``as_of_date``, settled, continuous when asked, and not stale.

    Rows after the date are dropped so a backtest never sees future prices. A
    closeless newest bar is an unsettled session, not a symbol without data: it
    is dropped so the frame ends at the last settled bar. ``fill_gaps`` carries
    prices forward over gaps for the indicators; a caller that reports the
    numbers themselves passes ``False`` to read them as the vendor gave them.
    """
    as_of_dt = pd.to_datetime(as_of_date).normalize()
    data = clean_dataframe(data)
    data = data[data["Date"] <= as_of_dt]

    if not data.empty and pd.isna(data["Close"].iloc[-1]):
        settled = data["Close"].notna().to_numpy().nonzero()[0]
        if settled.size == 0:
            raise NoMarketDataError(symbol, canonical, "no bar in range has a closing price")
        logger.warning(
            "%s: %d trailing bar(s) through %s have no closing price; using %s "
            "as the latest close.", canonical, len(data) - settled[-1] - 1,
            data["Date"].iloc[-1].date(), data["Date"].iloc[settled[-1]].date(),
        )

    data = fill_price_gaps(data) if fill_gaps else data.dropna(subset=["Close"]).copy()
    assert_not_stale(data, as_of_date, symbol, canonical)
    return data
