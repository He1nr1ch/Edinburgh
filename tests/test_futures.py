"""The overnight futures: the move since the cash close, put on SPY and judged as the guard judges."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from tradingagents import futures, market_guard
from tradingagents.default_config import DEFAULT_CONFIG

ET = futures.ET
CLOSES = pd.Series([700.0, 698.0, 701.0, 699.0, 702.0, 700.0],
                   index=["2026-11-13", "2026-11-16", "2026-11-17", "2026-11-18", "2026-11-19", "2026-11-20"])


def bars(last: float, at: datetime) -> list[tuple[datetime, float]]:
    friday = datetime(2026, 11, 20, 15, 50, tzinfo=ET)
    return [(friday, 6990.0), (friday + timedelta(minutes=10), 7000.0), (friday + timedelta(minutes=15), 6995.0), (at, last)]


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setattr(market_guard, "closes", lambda config, today: CLOSES)
    monkeypatch.setattr(futures, "_looked_at", -1e9)
    return {**DEFAULT_CONFIG, "futures_state_path": str(tmp_path / "futures.json")}


@pytest.mark.unit
def test_the_future_is_measured_from_the_cash_close_and_put_on_spy(cfg, monkeypatch):
    now = datetime(2026, 11, 22, 20, 0, tzinfo=ET)                       # Sunday evening
    monkeypatch.setattr(futures, "fetch", lambda symbol: bars(6790.0, now - timedelta(minutes=5)))
    r = futures.read_overnight(cfg, now)
    assert r.base == 7000.0 and r.close_day == "2026-11-20" and r.live   # the 16:00 bar, not the 16:05 one
    assert round(r.pct, 1) == -3.0 and round(r.tape.price, 2) == 679.0   # SPY's 700 moved by as much
    assert r.tape.direction == -1 and r.tape.swing == -1                 # past 1.5% and past 2% on the day
    assert "since Friday's close" in r.describe() and "implied SPY 679.00" in r.describe()
    assert "no new longs would be opened and the longs held would be closed" in r.means()


@pytest.mark.unit
def test_a_future_that_has_not_traded_since_the_close_or_cannot_be_read_gives_no_reading(cfg, monkeypatch):
    now = datetime(2026, 11, 21, 12, 0, tzinfo=ET)
    monkeypatch.setattr(futures, "fetch", lambda symbol: bars(7000.0, datetime(2026, 11, 20, 16, 0, tzinfo=ET))[:2])
    assert futures.read_overnight(cfg, now) is None

    def down(symbol):
        raise OSError("no answer")
    monkeypatch.setattr(futures, "fetch", down)
    assert futures.read_overnight(cfg, now) is None                      # logged, and the desk is as it was
    assert futures.read_overnight({**cfg, "futures_watch": False}, now) is None


@pytest.mark.unit
def test_a_saturday_reading_is_kept_but_said_to_be_of_a_market_that_is_shut(cfg, monkeypatch):
    now = datetime(2026, 11, 21, 12, 0, tzinfo=ET)
    monkeypatch.setattr(futures, "fetch", lambda symbol: bars(6979.0, datetime(2026, 11, 20, 16, 59, tzinfo=ET)))
    r = futures.read_overnight(cfg, now)
    assert not r.live and "(not trading now)" in r.describe() and r.tape.direction == 0


@pytest.mark.unit
def test_the_worker_emails_when_what_the_futures_imply_changes_and_only_then(cfg, monkeypatch):
    sent = []
    monkeypatch.setattr("tradingagents.worker.notify.send_email", lambda subject, body, config: sent.append((subject, body)) or True)
    now = datetime(2026, 11, 22, 20, 0, tzinfo=ET)
    price = [6993.0]
    monkeypatch.setattr(futures, "fetch", lambda symbol: bars(price[0], now - timedelta(minutes=5)))

    assert futures.watch(cfg, now).state == "+0/+0" and sent == []        # as at the close: nothing to say
    assert futures.last_reading(cfg)["pct"] == -0.1
    assert futures.watch(cfg, now) is None                                # not asked again within the quarter hour

    price[0] = 6790.0
    monkeypatch.setattr(futures, "_looked_at", -1e9)
    futures.watch(cfg, now)
    assert len(sent) == 1 and sent[0][0] == "Edinburgh: the futures say the market is falling, a swing down"
    assert "Nothing is traded on this reading" in sent[0][1]
    monkeypatch.setattr(futures, "_looked_at", -1e9)
    futures.watch(cfg, now)
    assert len(sent) == 1                                                 # the same reading is not sent twice

    price[0] = 6995.0
    monkeypatch.setattr(futures, "_looked_at", -1e9)
    futures.watch(cfg, now)
    assert len(sent) == 2 and "neither falling nor rising" in sent[1][0]  # it came back
    assert futures.last_reading(cfg)["told"] == "+0/+0"


@pytest.mark.unit
def test_a_source_that_fails_is_never_the_loops_failure(cfg, monkeypatch):
    def down(symbol):
        raise OSError("no answer")
    monkeypatch.setattr(futures, "fetch", down)
    assert futures.watch(cfg, datetime(2026, 11, 22, 20, 0, tzinfo=ET)) is None
    assert futures.last_reading(cfg) == {}
