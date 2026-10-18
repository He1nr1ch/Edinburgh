"""Indicators and the verified snapshot are computed from a frame, whatever vendor filled it."""

import pandas as pd
import pytest

from tradingagents.dataflows import indicators, ohlcv
from tradingagents.dataflows.errors import NoMarketDataError


def _frame(end="2026-01-09", periods=60):
    dates = pd.bdate_range(end=end, periods=periods)
    closes = [100.0 + i * 0.5 for i in range(periods)]
    return pd.DataFrame({
        "Date": dates, "Open": closes, "High": [c + 1 for c in closes],
        "Low": [c - 1 for c in closes], "Close": closes, "Volume": 1_000_000,
    })


@pytest.mark.unit
def test_an_indicator_window_lists_every_calendar_day_and_describes_the_indicator():
    out = indicators.render_indicator_window(_frame(), "rsi", "2026-01-09", 6)

    assert out.startswith("## rsi values from 2026-01-03 to 2026-01-09:")
    assert "2026-01-09:" in out and "2026-01-03:" in out
    assert "2026-01-04: N/A: Not a trading day" in out      # a Sunday
    assert "RSI: Measures momentum" in out


@pytest.mark.unit
def test_an_unknown_indicator_is_refused_by_name():
    with pytest.raises(ValueError, match="stochrsi"):
        indicators.indicator_by_date(_frame(), "stochrsi")


@pytest.mark.unit
def test_the_snapshot_uses_the_last_bar_on_or_before_the_date():
    frame = _frame(end="2026-01-16")   # bars after the analysis date must be ignored

    out = indicators.render_snapshot("NVDA", frame, "2026-01-09", look_back_days=5)

    assert "Verified market data snapshot for NVDA" in out
    assert "Latest trading row used: 2026-01-09" in out
    assert "| rsi |" in out and "| close_50_sma |" in out
    assert "Recent verified closes (last 5 rows)" in out
    assert "2026-01-16" not in out


@pytest.mark.unit
def test_a_snapshot_with_no_bar_by_the_date_is_no_data():
    with pytest.raises(NoMarketDataError):
        indicators.render_snapshot("NVDA", _frame(end="2026-01-09"), "2025-01-01")


@pytest.mark.unit
def test_as_of_frame_cuts_fills_and_rejects_a_stale_frame():
    frame = _frame(end="2026-01-16")
    frame.loc[frame.index[-8], "Close"] = None      # a gap mid-series

    cut = ohlcv.as_of_frame(frame, "NVDA", "NVDA", "2026-01-09")

    assert cut["Date"].max() == pd.Timestamp("2026-01-09")
    assert cut["Close"].notna().all()
    with pytest.raises(NoMarketDataError, match="stale"):
        ohlcv.as_of_frame(frame, "NVDA", "NVDA", "2026-03-01")


@pytest.mark.unit
def test_a_closeless_newest_bar_is_dropped_not_filled():
    frame = _frame(end="2026-01-09")
    frame.loc[frame.index[-1], "Close"] = None

    cut = ohlcv.as_of_frame(frame, "NVDA", "NVDA", "2026-01-09")

    assert cut["Date"].max() == pd.Timestamp("2026-01-08")
