"""Nill's P and B model: the zones it draws, the three trades, the stop-first sizing, and the backtest's fills."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

import cli.main as m
from tradingagents.pb_range import (
    BREAKOUT,
    FAKE_BREAKOUT,
    RANGE,
    Params,
    Scanner,
    Setup,
    Tracker,
    Zone,
    four_hour_bias,
    partials,
    run as run_backtest,
    size_units,
    target_levels,
    value_area,
    week_of,
)
from tradingagents.pb_range.backtest import Trade, _manage
from tradingagents.pb_range.data import regular_session_only

WICK = 0.3


def make_bars(closes, start="2026-10-19", wick=WICK, volume=1_000) -> pd.DataFrame:
    """Fifteen-minute bars, 26 a session, one per close; each opens at the last close."""
    stamps, day = [], pd.Timestamp(start)
    while len(stamps) < len(closes):
        if day.weekday() < 5:
            stamps += [day + pd.Timedelta(hours=9, minutes=30 + 15 * k) for k in range(26)]
        day += pd.Timedelta(days=1)
    closes = np.asarray(closes, dtype=float)
    opens = np.concatenate(([closes[0]], closes[:-1]))
    return pd.DataFrame({
        "Date": stamps[:len(closes)], "Open": opens,
        "High": np.maximum(opens, closes) + wick, "Low": np.minimum(opens, closes) - wick,
        "Close": closes, "Volume": volume,
    })


CHOP = [100 + 0.5 * (-1) ** k for k in range(40)]           # a flat, choppy start: ATR about one point
IMPULSE_UP = list(np.linspace(100, 112, 7)[1:])             # twelve points up in six bars
RANGE_CYCLE = [111.9, 111.5, 111.1, 111.5]                  # a small range under the high, one touch of each edge a cycle


def p_scenario(after: list[float], cycles: int = 4) -> pd.DataFrame:
    return make_bars(CHOP + IMPULSE_UP + RANGE_CYCLE * cycles + after)


def the_zone(bars) -> Zone:
    zones = Tracker(bars).run()
    assert zones, "no zone was validated"
    return zones[0]


# -- the zones ---------------------------------------------------------------

@pytest.mark.unit
def test_an_impulse_then_a_range_with_three_touches_is_a_valid_p():
    zone = the_zone(p_scenario([]))

    assert zone.kind == "P"
    assert zone.origin == pytest.approx(min(CHOP) - WICK)       # where the move started
    assert zone.impulse == pytest.approx(112 + WICK - zone.origin)
    assert zone.valid and zone.live
    assert zone.validated - zone.start + 1 >= Params.range_min_bars    # half a session, at least
    assert zone.touches_top >= 3 and zone.touches_bottom >= 3
    assert zone.high == pytest.approx(112 + WICK) and zone.low == pytest.approx(111.1 - WICK)


@pytest.mark.unit
def test_a_down_impulse_makes_a_b_with_its_origin_at_the_top():
    closes = CHOP + list(np.linspace(100, 88, 7)[1:]) + [88.1, 88.5, 88.9, 88.5] * 4
    zone = the_zone(make_bars(closes))

    assert zone.kind == "B"
    assert zone.origin == pytest.approx(max(CHOP) + WICK)


@pytest.mark.unit
def test_two_touches_do_not_make_a_range():
    bars = p_scenario([], cycles=1)        # four bars: one touch of each edge
    assert Tracker(bars).run() == []


@pytest.mark.unit
def test_a_close_outside_ends_the_range_and_a_close_back_inside_is_a_reclaim():
    bars = p_scenario([110.2, 111.5])      # below the range's low, then back inside
    zone = the_zone(bars)

    assert [e.kind for e in zone.events] == ["validated", "break", "reclaim"]
    reclaim = zone.events[-1]
    assert reclaim.direction == "down" and reclaim.extreme == pytest.approx(110.2 - WICK)
    assert zone.live and zone.reclaims == 1


@pytest.mark.unit
def test_a_break_that_does_not_come_back_ends_the_zone_for_good():
    bars = p_scenario([110.2, 109.5, 109.0])
    zone = the_zone(bars)

    assert [e.kind for e in zone.events] == ["validated", "break", "ended"]
    assert not zone.live and zone.status() == "broke down"


@pytest.mark.unit
def test_a_range_too_tall_for_its_impulse_is_not_one():
    bars = make_bars(CHOP + IMPULSE_UP + [112, 111, 106, 107] * 4)    # swings of six after a move of twelve
    assert Tracker(bars).run() == []
    assert Tracker(bars, Params(range_max_fraction=0.7)).run() != []


# -- the three trades ------------------------------------------------------------

@pytest.mark.unit
def test_a_fake_breakdown_and_reclaim_is_a_long_with_the_stop_under_the_fake_low():
    bars = p_scenario([110.2, 111.5])
    setups = Scanner(bars, equity=100_000).run()
    fake = [s for s in setups if s.kind == FAKE_BREAKOUT]

    assert len(fake) == 1
    s = fake[0]
    assert s.side == "long"
    assert s.entry == pytest.approx(s.zone.low)                  # a limit at the edge it reclaimed
    assert s.stop == pytest.approx(110.2 - WICK - Params.stop_buffer_atr * s.zone.atr)
    assert s.targets == (pytest.approx(s.zone.high + s.zone.impulse),)   # nothing above: the impulse, one to one
    assert s.rr >= Params.min_rr
    assert s.units > 0 and s.units % Params.lot == 0
    assert s.risk == pytest.approx(s.units * (s.entry - s.stop))
    assert "sellers failed" in s.notes[0]


@pytest.mark.unit
def test_a_fake_breakout_against_its_impulse_is_taken_only_when_asked():
    bars = p_scenario([113.0, 111.5])      # above a P, then back inside: a short against the impulse
    assert not any(s.kind == FAKE_BREAKOUT for s in Scanner(bars).run())
    against = Params(fake_breakout_against_impulse=True)
    assert any(s.kind == FAKE_BREAKOUT and s.side == "short" for s in Scanner(bars, against).run())
    no_shorts = Params(fake_breakout_against_impulse=True, fake_breakout_shorts=False)
    assert not any(s.kind == FAKE_BREAKOUT for s in Scanner(bars, no_shorts).run())


@pytest.mark.unit
def test_a_breakout_needs_a_close_outside_a_pullback_and_a_close_beyond_the_breakout_bar():
    broke, pulled_back, confirmed = 113.0, 112.6, 113.8
    bars = p_scenario([broke, pulled_back, confirmed])
    setups = Scanner(bars).run()
    breakouts = [s for s in setups if s.kind == BREAKOUT]

    assert len(breakouts) == 1
    s = breakouts[0]
    assert s.side == "long"
    assert s.bar == len(bars) - 1                              # only on the confirming bar
    assert s.entry == pytest.approx(broke + WICK)              # a limit at the breakout bar's high
    assert s.stop == pytest.approx(s.zone.low - Params.stop_buffer_atr * s.zone.atr)
    assert s.notes[0].startswith("small range")

    without_pullback = p_scenario([broke, 113.5, 113.8])        # never came back to the edge
    assert not any(s.kind == BREAKOUT for s in Scanner(without_pullback).run())


@pytest.mark.unit
def test_a_small_range_is_not_range_traded_and_a_big_one_is():
    small = p_scenario([])
    assert not any(s.kind == RANGE for s in Scanner(small).run())

    big = make_bars(CHOP + IMPULSE_UP + [111.9, 110.5, 109.1, 110.5] * 4)      # three points tall
    ranges = [s for s in Scanner(big).run() if s.kind == RANGE]
    assert {s.side for s in ranges} == {"long", "short"}
    long = next(s for s in ranges if s.side == "long")
    assert long.entry == pytest.approx(long.zone.low) and long.targets == (pytest.approx(long.zone.high),)
    assert long.stop == pytest.approx(long.zone.low - Params.range_stop_fraction * long.zone.height)
    assert long.notes[0] == "edge to edge, no partials"


@pytest.mark.unit
def test_a_range_completed_in_the_evening_is_not_range_traded():
    big = make_bars(CHOP + IMPULSE_UP + [111.9, 110.5, 109.1, 110.5] * 4)
    assert any(s.kind == RANGE for s in Scanner(big, Params(evening_hour=None)).run())
    assert not any(s.kind == RANGE for s in Scanner(big, Params(evening_hour=0)).run())


@pytest.mark.unit
def test_a_setup_under_the_minimum_reward_to_risk_is_not_taken():
    bars = p_scenario([110.2, 111.5])
    assert any(s.kind == FAKE_BREAKOUT for s in Scanner(bars).run())
    assert not any(s.kind == FAKE_BREAKOUT for s in Scanner(bars, Params(min_rr=50)).run())


# -- targets and sizing -------------------------------------------------------------

def zone(kind, low, high, origin, impulse) -> Zone:
    return Zone(kind, high, low, origin, impulse, start=0, atr=1.0, validated=1)


@pytest.mark.unit
def test_against_the_impulse_the_targets_run_to_the_origin_of_the_move():
    p = zone("P", 110, 112, origin=100, impulse=12)
    between = zone("P", 104, 106, origin=98, impulse=6)
    levels = target_levels(p, "short", entry=112, prior=[between])

    assert levels == [110, 106, 104, 100]          # the range's far edge, the fair value between, the origin
    assert partials(levels) == (110, 104, 100)


@pytest.mark.unit
def test_with_the_impulse_the_target_is_the_next_range_or_the_impulse_projected():
    p = zone("P", 110, 112, origin=100, impulse=12)
    above = zone("B", 118, 121, origin=130, impulse=12)

    assert target_levels(p, "long", entry=110, prior=[above]) == [118, 121]
    assert partials(target_levels(p, "long", entry=110, prior=[above])) == (118, 119.5, 121)
    assert target_levels(p, "long", entry=110, prior=[]) == [124]       # 112 + 12


@pytest.mark.unit
def test_the_stop_decides_the_size_in_lots_of_three_within_the_account():
    assert size_units(100_000, 1.0, entry=100, stop=99) == 999        # 1,000 units risk 1%; 999 divides by three
    assert size_units(100_000, 1.0, entry=100, stop=99.5) == 999      # 2,000 would be twice the account
    assert size_units(100_000, 1.0, entry=100, stop=50) == 18         # 20, less the odd two
    assert size_units(100, 1.0, entry=100, stop=99.5, max_position_pct=1_000) == 2   # under a lot: in one piece
    assert size_units(100_000, 1.0, entry=100, stop=100) == 0
    assert size_units(100_000, 1.0, entry=100, stop=99, lot=1) == 1_000


# -- the backtest ----------------------------------------------------------------

def trade(side="long", entry=100.0, stop=98.0, targets=(104.0,), units=3) -> Trade:
    s = Setup(FAKE_BREAKOUT, side, 0, pd.Timestamp("2026-10-20 10:00"), entry, stop, targets,
              zone("P", 99, 101, 90, 10), units, units * abs(entry - stop))
    return Trade(s, s.at, entry, units, stop)


@pytest.mark.unit
def test_a_bar_that_reaches_the_stop_and_the_target_is_a_stop():
    t = trade()
    _manage(t, pd.Timestamp("2026-10-20 10:15"), open_=100, high=105, low=97)
    assert [f.reason for f in t.fills] == ["stop"] and t.r_multiple == pytest.approx(-1)


@pytest.mark.unit
def test_a_gap_through_the_stop_fills_at_the_open():
    t = trade()
    _manage(t, pd.Timestamp("2026-10-20 10:15"), open_=96, high=97, low=95)
    assert t.fills[0].price == 96 and t.r_multiple == pytest.approx(-2)


@pytest.mark.unit
def test_partials_take_a_third_each_and_move_the_stop_to_the_entry():
    t = trade(targets=(102.0, 104.0, 106.0), units=9)
    when = pd.Timestamp("2026-10-20 10:15")
    _manage(t, when, open_=100, high=102.5, low=99.5)
    assert [(f.reason, f.units) for f in t.fills] == [("target 1", 3)] and t.stop == 100
    _manage(t, when, open_=102, high=106.5, low=101.5)
    assert [(f.reason, f.units) for f in t.fills] == [("target 1", 3), ("target 2", 3), ("target 3", 3)]
    assert t.closed and t.pnl == pytest.approx(3 * 2 + 3 * 4 + 3 * 6)


@pytest.mark.unit
def test_the_backtest_fills_the_reclaim_limit_and_runs_it_to_the_target():
    dip_to_fill = 111.1 - WICK + 0.2                 # a bar whose low reaches the zone's low
    to_target = list(np.linspace(112, 112 + WICK + 12 + 2 * WICK + 1, 8))
    bars = p_scenario([110.2, 111.5, dip_to_fill] + to_target)
    result = run_backtest(bars)

    assert [t.setup.kind for t in result.trades] == [FAKE_BREAKOUT]
    t = result.trades[0]
    assert t.entry == pytest.approx(t.setup.entry)
    assert [f.reason for f in t.fills] == ["target 1"]
    assert t.r_multiple > Params.min_rr
    assert result.stats()["win_rate"] == 1.0


@pytest.mark.unit
def test_a_range_order_does_not_fill_in_the_evening():
    """The four worst trades of the S&P 100 sweep were range shorts filled in the
    last hour and gapped against overnight, to -11R."""
    bars = make_bars(CHOP + IMPULSE_UP + [111.9, 110.5, 109.1, 110.5] * 4 + [109.1 - WICK + 0.2] * 10)
    # The range is valid at 10:45; every bar that reaches its low comes after 11:00.
    evening = run_backtest(bars, Params(evening_hour=11, max_losses_per_day=99))
    assert evening.trades == []
    anytime = run_backtest(bars, Params(evening_hour=None, max_losses_per_day=99))
    assert [t.setup.kind for t in anytime.trades] == [RANGE]


@pytest.mark.unit
def test_a_range_trade_still_open_at_the_close_is_closed_there():
    low = 109.1 - WICK
    bars = make_bars(CHOP + IMPULSE_UP + [111.9, 110.5, 109.1, 110.5] * 4 + [low + 0.2] * 40)   # drifts, never reaches an edge
    result = run_backtest(bars, Params(max_losses_per_day=99))

    assert len(result.trades) == 1
    t = result.trades[0]
    assert t.setup.kind == RANGE and [f.reason for f in t.fills] == ["close of day"]
    assert t.fills[0].at.strftime("%H:%M") == "15:45" and t.fills[0].at.date() == t.entry_at.date()


@pytest.mark.unit
def test_after_the_days_losses_in_a_row_no_more_orders_that_day():
    # A big range: the range long fills and is stopped by the fake breakdown,
    # the reclaim then offers a long that fills on the dip.
    big_cycle = [111.9, 110.5, 109.1, 110.5]
    low = 109.1 - WICK
    after = [low - 1.0, 111.0, low + 0.2, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125]
    bars = make_bars(CHOP + IMPULSE_UP + big_cycle * 4 + after)

    both = run_backtest(bars, Params(max_losses_per_day=3))
    assert [t.setup.kind for t in both.trades] == [RANGE, FAKE_BREAKOUT]
    assert both.trades[0].r_multiple == pytest.approx(-1)

    one = run_backtest(bars, Params(max_losses_per_day=1))
    assert [t.setup.kind for t in one.trades] == [RANGE]


# -- the big picture ---------------------------------------------------------

@pytest.mark.unit
def test_the_four_hour_bias_follows_the_blocks():
    assert four_hour_bias(make_bars(np.linspace(100, 140, 16 * 30))) == "bullish"
    assert four_hour_bias(make_bars(np.linspace(140, 100, 16 * 30))) == "bearish"
    assert four_hour_bias(make_bars([100, 101] * (16 * 15))) == "unclear"
    assert four_hour_bias(make_bars([100] * 20)) == "unclear"        # too few blocks to say


@pytest.mark.unit
def test_the_value_area_holds_most_of_the_volume_around_the_point_of_control():
    bars = make_bars([100, 100.2, 100.1, 100.3, 100.2, 100.1, 103, 106, 109])
    bars.loc[bars["Close"] > 102, "Volume"] = 10        # the climb traded almost nothing
    area = value_area(bars, 0.70, bin_size=0.5)

    assert 99.5 <= area.poc <= 101
    assert area.low <= 100 and 100.5 <= area.high <= 102
    assert value_area(bars.assign(Volume=0)) is None


@pytest.mark.unit
def test_the_weeks_bars_start_on_monday():
    bars = make_bars([100] * 26 * 7, start="2026-10-15")       # Thursday to the next Friday
    week = week_of(bars, pd.Timestamp("2026-10-21 12:00"))
    assert week["Date"].min() == pd.Timestamp("2026-10-19 09:30") and week["Date"].max() <= pd.Timestamp("2026-10-21 12:00")


@pytest.mark.unit
def test_only_regular_session_bars_are_kept():
    frame = pd.DataFrame({"Date": pd.to_datetime(["2026-10-20 09:15", "2026-10-20 09:30", "2026-10-20 15:45", "2026-10-20 16:00"])})
    assert list(regular_session_only(frame)["Date"].dt.strftime("%H:%M")) == ["09:30", "15:45"]


# -- the command -------------------------------------------------------------

@pytest.mark.unit
def test_the_pb_range_command_reports_the_zones_and_the_backtest(monkeypatch):
    import tradingagents.pb_range.data as data

    bars = p_scenario([110.2, 111.5, 111.1 - WICK + 0.2] + list(np.linspace(112, 126, 8)))
    monkeypatch.setattr(data, "intraday_bars", lambda *a, **kw: bars)
    monkeypatch.setattr(m, "run_analysis", lambda **kw: pytest.fail("the analysis must not run"))

    result = CliRunner().invoke(m.app, ["pb-range", "nvda", "--date", "2026-10-22", "--backtest"])

    assert result.exit_code == 0, result.output
    assert "NVDA" in result.output and "P and B ranges" in result.output
    assert "Backtest: every setup traded" in result.output and "win rate 100%" in result.output
