"""Patrick Nill's P and B: an impulse, then a range, valid on its third touch.

The model behind Nill's Robbins World Cup results, as he describes it in
interviews (condensed in "Everyone Laughed at His Strategy", Trading Notes,
youtube.com/watch?v=kJyooWDFXyk):

- Markets range about 70% of the time. Big players fill inside a range, then
  push price to the next pool of liquidity, where a new range forms. The job
  is to trade from one area of fair value to the next, not to predict.
- A **P** is a strong move up followed by a range; a **B** a strong move down
  followed by one. Decisions are made on the 15-minute chart, the 4-hour chart
  gives the big picture.
- A range is valid only after **three touches** of its edges, and on 15-minute
  bars needs at least half a session to form. Waiting that long misses moves;
  the longer the range, the better its breakout works.
- Before drawing, the week's volume profile: a P or B that forms on the value
  area high or low is a stronger setup.

Nill does not share his exact rules for drawing and resizing the zones, so the
drawing rule here is this module's own, stated in :class:`Params`: an impulse
is a move of ``impulse_atr`` ATRs within ``impulse_max_bars`` bars that gives
back little on the way; the range is the box of the bars after it, while the
box stays under ``range_max_fraction`` of the impulse; a close outside the box
ends it. The defaults were checked on four months of the S&P 100 (July to
October 2026, split in two): a cap of 0.35 lost less than 0.5 in every period
and combination, and the fake breakout paid only with its impulse. The tracker is incremental, one bar at a time, so a scan and a
backtest see exactly what was known at each bar.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MARKET_TZ = "America/New_York"


@dataclass(frozen=True)
class Params:
    """The strategy's numbers. Bars are 15-minute bars unless said otherwise."""

    atr_bars: int = 14
    # An impulse travels at least this many ATRs within this many bars (four
    # hours of 15-minute bars), giving back no more than this share of itself.
    impulse_atr: float = 3.0
    impulse_max_bars: int = 16
    impulse_max_retrace: float = 0.35
    # The range after it: at most this share of the impulse tall while forming,
    # at least this many bars (half a session) and this many ATRs tall before it
    # can be valid, and valid on this many touches of its edges. A bar within
    # ``touch_tolerance`` of the range's height of an edge touches it.
    range_max_fraction: float = 0.35
    range_min_bars: int = 13
    range_min_atr: float = 0.3
    touches_to_validate: int = 3
    touch_tolerance: float = 0.15
    # A fake breakout closes back inside within this many bars of the break.
    reclaim_bars: int = 1
    # A confirmed breakout needs its pullback and confirming close within this
    # many bars of the break.
    breakout_window: int = 12
    # Stops sit this many ATRs beyond the structure they lean on. A range
    # trade's stop sits at least this share of the range beyond its edge: the
    # edges are where wicks poke through.
    stop_buffer_atr: float = 0.1
    range_stop_fraction: float = 0.2
    # Reward to risk, to the final target, below which a setup is not taken.
    min_rr: float = 3.0
    # A range at least this many ATRs tall is traded edge to edge; a smaller
    # one is traded for its breakout, where the stop is close and the target far.
    big_range_atr: float = 1.5
    # A range completed at or after this hour (New York) is not range-traded:
    # the night is too long to hold a range trade unseen. None for a market
    # without a close.
    evening_hour: int | None = 15
    # Losses in a row in one day after which the day is over.
    max_losses_per_day: int = 3
    # Positions are multiples of this many units so the partials come out even,
    # and risk this share of equity between entry and stop.
    lot: int = 3
    risk_pct: float = 1.0
    # A position's value is at most this share of equity: a tight stop must
    # not size a cash account into leverage it does not have.
    max_position_pct: float = 100.0
    # Share of the week's volume that defines its value area.
    value_area: float = 0.70
    # Whether the short side of the fake breakout is traded at all. Nill trades
    # it much less: when there is power to the upside, more usually comes.
    fake_breakout_shorts: bool = True
    # Whether a fake breakout is traded against its impulse too: a short on a P
    # after a failed break above, a long on a B after a failed break below.
    # Nill takes those, less often; the one with the impulse is the favourite,
    # and the only one that paid in the S&P 100 sweep (see the README).
    fake_breakout_against_impulse: bool = False
    # Where a breakout or fake breakout is entered: "edge", a limit back at the
    # level the setup completed on (the range's edge, the breakout bar's
    # extreme), as Nill does; or "close", at the close of the bar that
    # completed it, which fills every setup but pays a worse price.
    entry: str = "edge"


@dataclass
class Event:
    kind: str                   # validated, break, reclaim, ended
    bar: int
    zone: Zone
    direction: str | None = None    # up or down, for a break
    extreme: float | None = None    # the break's furthest price, for a break or a reclaim


@dataclass
class Zone:
    """One P or B: the impulse that made it and the range it left."""

    kind: str                   # P after a move up, B after a move down
    high: float
    low: float
    origin: float               # where the impulse started: the far target of a trade against it
    impulse: float              # the impulse's size in price
    start: int                  # bar index of the impulse's extreme, the range's first bar
    atr: float                  # ATR when the range began
    validated: int | None = None    # bar index of the third touch
    end: int | None = None          # bar index of the first close outside; None while live
    break_direction: str | None = None
    break_extreme: float | None = None
    touches_top: int = 0
    touches_bottom: int = 0
    bars: int = 1
    events: list[Event] = field(default_factory=list)

    @property
    def height(self) -> float:
        return self.high - self.low

    @property
    def mid(self) -> float:
        return (self.high + self.low) / 2

    @property
    def valid(self) -> bool:
        return self.validated is not None

    @property
    def touches(self) -> int:
        return self.touches_top + self.touches_bottom

    @property
    def live(self) -> bool:
        return self.end is None

    @property
    def reclaims(self) -> int:
        return sum(1 for e in self.events if e.kind == "reclaim")

    def contains(self, price: float) -> bool:
        return self.low <= price <= self.high

    def overlaps(self, other: Zone) -> bool:
        """Whether the two ranges share prices: two ranges at one level work better."""
        return self.low <= other.high and other.low <= self.high

    def status(self) -> str:
        if not self.valid:
            return "forming"
        if self.live:
            return "valid"
        return f"broke {self.break_direction}"


def true_range_atr(bars: pd.DataFrame, n: int) -> np.ndarray:
    """Wilder's ATR over ``n`` bars, NaN until there are ``n`` of them."""
    high, low, close = (bars[c].to_numpy(dtype=float) for c in ("High", "Low", "Close"))
    prev = np.concatenate(([np.nan], close[:-1]))
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev), np.abs(low - prev)))
    tr[0] = high[0] - low[0]
    return pd.Series(tr).ewm(alpha=1 / n, min_periods=n, adjust=False).mean().to_numpy()


class Tracker:
    """Reads bars one at a time and keeps the zone they are forming, with its events.

    ``bars`` has ``Date``, ``Open``, ``High``, ``Low``, ``Close``, ``Volume``
    columns, one row per 15-minute bar, in order. Call :meth:`step` with each
    bar index in turn; it returns what that bar changed. ``zones`` holds every
    zone that was validated, the live one last.
    """

    def __init__(self, bars: pd.DataFrame, params: Params | None = None):
        self.p = params or Params()
        self.bars = bars.reset_index(drop=True)
        self.high = self.bars["High"].to_numpy(dtype=float)
        self.low = self.bars["Low"].to_numpy(dtype=float)
        self.close = self.bars["Close"].to_numpy(dtype=float)
        self.atr = true_range_atr(self.bars, self.p.atr_bars)
        self.zones: list[Zone] = []
        self.live: Zone | None = None     # the candidate or valid zone being tracked

    # -- the state machine -------------------------------------------------

    def step(self, i: int) -> list[Event]:
        z = self.live
        if z is None:
            self.live = self._impulse_ending_at(i)
            return []
        if z.end is not None:
            return self._after_break(z, i)
        if not z.valid:
            return self._forming(z, i)
        return self._valid(z, i)

    def run(self) -> list[Zone]:
        """Every bar, in turn; the zones as of the last one."""
        for i in range(len(self.bars)):
            self.step(i)
        return self.zones

    def _forming(self, z: Zone, i: int) -> list[Event]:
        p = self.p
        close = self.close[i]
        if (close > z.high) if z.kind == "P" else (close < z.low):
            # A close beyond the extreme: the impulse is still going. The range
            # starts from its new extreme, with the origin of the whole move.
            extended = self._impulse_ending_at(i)
            if extended is not None:
                self.live = extended
                return []
        high, low = max(z.high, self.high[i]), min(z.low, self.low[i])
        if high - low > p.range_max_fraction * z.impulse:
            # Too tall for a range after that impulse: the move is not done, or
            # it is reversing. Look for an impulse ending here instead.
            self.live = self._impulse_ending_at(i)
            return []
        z.high, z.low, z.bars = high, low, i - z.start + 1
        self._count_touches(z, i)
        if (z.touches >= p.touches_to_validate and z.bars >= p.range_min_bars
                and z.height >= p.range_min_atr * z.atr):
            z.validated = i
            self.zones.append(z)
            event = Event("validated", i, z)
            z.events.append(event)
            return [event]
        return []

    def _valid(self, z: Zone, i: int) -> list[Event]:
        close = self.close[i]
        if close > z.high or close < z.low:
            z.end = i
            z.break_direction = "up" if close > z.high else "down"
            z.break_extreme = self.high[i] if close > z.high else self.low[i]
            event = Event("break", i, z, z.break_direction, z.break_extreme)
            z.events.append(event)
            return [event]
        # A wick beyond the edge with a close inside grows the box: a touch.
        z.high, z.low, z.bars = max(z.high, self.high[i]), min(z.low, self.low[i]), i - z.start + 1
        self._count_touches(z, i)
        return []

    def _after_break(self, z: Zone, i: int) -> list[Event]:
        """The bars after a close outside: a reclaim makes it a fake breakout."""
        close = self.close[i]
        if z.contains(close) and i - z.end <= self.p.reclaim_bars:
            if z.break_direction == "up":
                extreme = float(self.high[z.end:i + 1].max())
            else:
                extreme = float(self.low[z.end:i + 1].min())
            event = Event("reclaim", i, z, z.break_direction, extreme)
            z.events.append(event)
            z.end = z.break_direction = z.break_extreme = None
            z.bars = i - z.start + 1
            return [event]
        if i - z.end >= self.p.reclaim_bars:
            event = Event("ended", i, z)
            z.events.append(event)
            self.live = self._impulse_ending_at(i)
            return [event]
        return []

    # -- pieces --------------------------------------------------------------

    def _count_touches(self, z: Zone, i: int) -> None:
        """Touches of each edge from the range's first bar to bar ``i``.

        Consecutive bars at an edge are one touch: price came there once.
        """
        tol = self.p.touch_tolerance * z.height
        top = bottom = 0
        on_top = on_bottom = False
        for m in range(z.start, i + 1):
            at_top = self.high[m] >= z.high - tol
            at_bottom = self.low[m] <= z.low + tol
            top += at_top and not on_top
            bottom += at_bottom and not on_bottom
            on_top, on_bottom = at_top, at_bottom
        z.touches_top, z.touches_bottom = top, bottom

    def _impulse_ending_at(self, i: int) -> Zone | None:
        """A candidate zone whose impulse has its extreme at bar ``i``, if there is one.

        The longest window back from ``i`` in which the move is big enough and
        clean enough wins, so the origin is the start of the whole move. The
        range begins where the push ends: at the extreme bar's close and the
        wick beyond it, not at the body that was still the impulse.
        """
        p = self.p
        atr = self.atr[i]
        if np.isnan(atr) or atr <= 0:
            return None
        best: Zone | None = None
        for k in range(2, p.impulse_max_bars + 1):
            j0 = i - k
            if j0 < 0:
                break
            highs, lows = self.high[j0:i + 1], self.low[j0:i + 1]
            if self.high[i] >= highs.max():
                j = j0 + int(lows.argmin())
                move = self.high[i] - self.low[j]
                if (j < i and move >= p.impulse_atr * atr and self._retrace(j, i, "up") <= p.impulse_max_retrace
                        and (best is None or move > best.impulse)):
                    best = Zone("P", self.high[i], self.close[i], float(self.low[j]), float(move), i, float(atr))
            if self.low[i] <= lows.min():
                j = j0 + int(highs.argmax())
                move = self.high[j] - self.low[i]
                if (j < i and move >= p.impulse_atr * atr and self._retrace(j, i, "down") <= p.impulse_max_retrace
                        and (best is None or move > best.impulse)):
                    best = Zone("B", self.close[i], self.low[i], float(self.high[j]), float(move), i, float(atr))
        return best

    def _retrace(self, j: int, i: int, direction: str) -> float:
        """The deepest pullback along the move from bar ``j`` to bar ``i``, as a share of it."""
        highs, lows = self.high[j:i + 1], self.low[j:i + 1]
        if direction == "up":
            move = highs[-1] - lows[0]
            drawdown = (np.maximum.accumulate(highs) - lows).max()
        else:
            move = highs[0] - lows[-1]
            drawdown = (highs - np.minimum.accumulate(lows)).max()
        return float(drawdown / move) if move > 0 else np.inf


# -- the big picture ---------------------------------------------------------

def four_hour_bias(bars: pd.DataFrame, bars_per_block: int = 16, ema: int = 20) -> str:
    """The 4-hour chart's direction: bullish, bearish or unclear.

    Blocks of ``bars_per_block`` 15-minute bars stand in for 4-hour bars.
    Bullish when the last block closes above a rising EMA, bearish when below a
    falling one, unclear otherwise. Nill looks for counter trades when a push
    has stopped and the picture is unclear.
    """
    closes = bars["Close"].to_numpy(dtype=float)
    blocks = closes[bars_per_block - 1::bars_per_block]
    if len(blocks) < ema + 4:
        return "unclear"
    avg = pd.Series(blocks).ewm(span=ema, adjust=False).mean().to_numpy()
    if blocks[-1] > avg[-1] and avg[-1] > avg[-4]:
        return "bullish"
    if blocks[-1] < avg[-1] and avg[-1] < avg[-4]:
        return "bearish"
    return "unclear"


@dataclass(frozen=True)
class ValueArea:
    low: float      # VAL
    poc: float      # point of control, the price with the most volume
    high: float     # VAH

    def near(self, price: float, tolerance: float) -> bool:
        return abs(price - self.low) <= tolerance or abs(price - self.high) <= tolerance


def value_area(bars: pd.DataFrame, share: float = 0.70, bin_size: float | None = None) -> ValueArea | None:
    """The volume profile's value area: the prices around the point of control
    holding ``share`` of the volume. Each bar's volume is spread over its range.
    """
    if bars.empty or bars["Volume"].fillna(0).sum() <= 0:
        return None
    high, low, vol = (bars[c].to_numpy(dtype=float) for c in ("High", "Low", "Volume"))
    vol = np.nan_to_num(vol)
    if bin_size is None or bin_size <= 0:
        bin_size = max((high.max() - low.min()) / 50, 1e-9)
    edges = np.arange(low.min(), high.max() + 2 * bin_size, bin_size)
    hist = np.zeros(len(edges) - 1)
    for bar_high, bar_low, v in zip(high, low, vol, strict=True):
        first = np.searchsorted(edges, bar_low, "right") - 1
        last = np.searchsorted(edges, bar_high, "right") - 1
        first, last = max(first, 0), min(max(last, first), len(hist) - 1)
        hist[first:last + 1] += v / (last - first + 1)
    poc = int(hist.argmax())
    lo = hi = poc
    total, inside = hist.sum(), hist[poc]
    while inside < share * total and (lo > 0 or hi < len(hist) - 1):
        below = hist[lo - 1] if lo > 0 else -1
        above = hist[hi + 1] if hi < len(hist) - 1 else -1
        if above >= below:
            hi += 1
            inside += above
        else:
            lo -= 1
            inside += below
    centre = (edges[:-1] + edges[1:]) / 2
    return ValueArea(float(edges[lo]), float(centre[poc]), float(edges[hi + 1]))


def week_of(bars: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """The bars of ``as_of``'s week up to ``as_of``, for the week's value area."""
    dates = pd.to_datetime(bars["Date"])
    monday = (as_of - pd.Timedelta(days=as_of.weekday())).normalize()
    return bars[(dates >= monday) & (dates <= as_of)]
