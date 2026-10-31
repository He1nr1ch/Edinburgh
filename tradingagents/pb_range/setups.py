"""The three ways Nill trades a valid range, with the stop first and the size after.

1. **Range trade**: a limit at the bottom of a big range for the top, or at the
   top for the bottom. In and out, no partials. Not opened when the range
   completed in the evening.
2. **Breakout with confirmation**: a close outside the range, a pullback to its
   edge, then a close beyond the breakout bar's extreme. Small ranges after a
   big move are the best: a close stop, a far target.
3. **Fake breakout**: a close outside, then the next bar closes back inside.
   Entry at the edge it reclaimed, the stop just beyond the fake break's
   extreme, and the trade is held past the far edge: a market that wanted to go
   one way and could not tends to go the other way with force. Nill's favourite,
   on the long side.

The stop leans on structure (a low, the zone, a value-area level) and is set
before the size; the size is what risks ``risk_pct`` of equity to it, in
multiples of ``lot`` so partials come out even. Targets go from fair value to
fair value: the next range's edge first, the origin of the whole move last; at
a new high with nothing above, the impulse projected one to one. Below
``min_rr`` to the final target, no trade.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from tradingagents.pb_range.model import Event, Params, Tracker, ValueArea, Zone

RANGE, BREAKOUT, FAKE_BREAKOUT = "range", "breakout", "fake_breakout"


@dataclass(frozen=True)
class Setup:
    kind: str                       # range, breakout or fake_breakout
    side: str                       # long or short
    bar: int                        # the bar that completed the setup; the order is live from the next
    at: pd.Timestamp
    entry: float                    # the limit price
    stop: float
    targets: tuple[float, ...]      # one target, or three for partials
    zone: Zone
    units: int
    risk: float                     # money between entry and stop for ``units``
    bias: str = "unclear"
    notes: tuple[str, ...] = ()

    @property
    def target(self) -> float:
        return self.targets[-1]

    @property
    def per_unit_risk(self) -> float:
        return abs(self.entry - self.stop)

    @property
    def rr(self) -> float:
        """Reward to risk, to the final target."""
        return abs(self.target - self.entry) / self.per_unit_risk if self.per_unit_risk > 0 else 0.0

    @property
    def direction(self) -> int:
        return 1 if self.side == "long" else -1

    def describe(self) -> str:
        targets = " / ".join(f"{t:,.2f}" for t in self.targets)
        return (f"{self.kind} {self.side} {self.units} @ {self.entry:,.2f}, stop {self.stop:,.2f}, "
                f"target {targets} ({self.rr:.1f}R)")


def size_units(equity: float, risk_pct: float, entry: float, stop: float, lot: int = 3,
               max_position_pct: float = 100.0) -> int:
    """How many units risk ``risk_pct`` of ``equity`` between ``entry`` and ``stop``.

    The stop is decided first and the size follows from it, never the other
    way round. Rounded down to a multiple of ``lot`` when there is at least one
    lot, so the partials divide evenly; fewer units than a lot are taken as
    they are, in one piece. The position's value stays within
    ``max_position_pct`` of equity whatever the stop allows.
    """
    per_unit = abs(entry - stop)
    if per_unit <= 0 or equity <= 0 or risk_pct <= 0 or entry <= 0:
        return 0
    units = math.floor(equity * risk_pct / 100 / per_unit)
    units = min(units, math.floor(equity * max_position_pct / 100 / entry))
    if lot > 1 and units >= lot:
        units -= units % lot
    return units


def target_levels(zone: Zone, side: str, entry: float, prior: list[Zone]) -> list[float]:
    """Price levels beyond ``entry`` in the trade's direction, nearest first.

    Against the impulse (a short on a P, a long on a B): the range's far edge,
    then any fair value between, then the origin of the move, which is where
    price usually goes back to. With the impulse: the next range beyond, its
    near edge then its far one; with nothing beyond, the impulse projected one
    to one from the range's far edge.
    """
    direction = 1 if side == "long" else -1
    with_impulse = (side == "long") == (zone.kind == "P")
    far_edge = zone.high if direction > 0 else zone.low

    def beyond(level: float, floor: float) -> bool:
        return direction * (level - floor) > 1e-9

    if not with_impulse:
        levels = [far_edge] if beyond(far_edge, entry) else []
        between = sorted(
            {edge for z in prior for edge in (z.low, z.high)
             if beyond(edge, far_edge) and beyond(zone.origin, edge)},
            key=lambda x: direction * x,
        )
        levels += between
        if beyond(zone.origin, levels[-1] if levels else entry):
            levels.append(zone.origin)
        return levels

    ahead = sorted((z for z in prior if beyond(z.low if direction > 0 else z.high, far_edge)),
                   key=lambda z: direction * (z.low if direction > 0 else z.high))
    if ahead:
        nxt = ahead[0]
        near, far = (nxt.low, nxt.high) if direction > 0 else (nxt.high, nxt.low)
        return [near, far] if beyond(far, near) else [near]
    return [far_edge + direction * zone.impulse]


def partials(levels: list[float]) -> tuple[float, ...]:
    """One target, or three: the nearest level, one decided between, the last."""
    if not levels:
        return ()
    if len(levels) == 1:
        return (levels[0],)
    if len(levels) == 2:
        return (levels[0], (levels[0] + levels[1]) / 2, levels[1])
    return (levels[0], levels[len(levels) // 2], levels[-1])


class Scanner:
    """Turns a bar stream into setups: the tracker's events, plus the pending
    breakouts waiting for their pullback and confirmation."""

    def __init__(self, bars: pd.DataFrame, params: Params | None = None, *,
                 equity: float = 100_000, value_area: ValueArea | None = None, bias: str = "unclear"):
        self.p = params or Params()
        self.bars = bars.reset_index(drop=True)
        self.tracker = Tracker(self.bars, self.p)
        self.equity = equity
        self.value_area = value_area
        self.bias = bias
        self.dates = pd.to_datetime(self.bars["Date"])
        self.pending: list[_PendingBreak] = []
        self.setups: list[Setup] = []
        self.last_events: list[Event] = []

    @property
    def zones(self) -> list[Zone]:
        return self.tracker.zones

    def run(self) -> list[Setup]:
        for i in range(len(self.bars)):
            self.step(i)
        return self.setups

    def step(self, i: int) -> list[Setup]:
        events = self.tracker.step(i)
        self.last_events = events
        found: list[Setup] = []
        for event in events:
            if event.kind == "validated":
                found += self._range_trades(event.zone, i)
            elif event.kind == "break":
                self.pending.append(_PendingBreak(event.zone, event.direction, event.extreme, i))
            elif event.kind == "reclaim":
                self.pending = [b for b in self.pending if b.zone is not event.zone]
                setup = self._fake_breakout(event, i)
                if setup:
                    found.append(setup)
        found += self._confirm_breakouts(i)
        self.setups += found
        return found

    # -- the three trades --------------------------------------------------

    def _range_trades(self, zone: Zone, i: int) -> list[Setup]:
        p = self.p
        if zone.height < p.big_range_atr * zone.atr:
            return []      # a small range is for its breakout
        if p.evening_hour is not None and self.dates[i].hour >= p.evening_hour:
            return []      # completed in the evening: too long unseen
        buffer = max(p.stop_buffer_atr * zone.atr, p.range_stop_fraction * zone.height)
        notes = ("edge to edge, no partials",)
        return [s for s in (
            self._setup(RANGE, "long", i, zone, zone.low, zone.low - buffer, (zone.high,), notes),
            self._setup(RANGE, "short", i, zone, zone.high, zone.high + buffer, (zone.low,), notes),
        ) if s]

    def _confirm_breakouts(self, i: int) -> list[Setup]:
        p, found = self.p, []
        high, low, close = self.tracker.high[i], self.tracker.low[i], self.tracker.close[i]
        for b in list(self.pending):
            if b.at == i:
                continue
            z = b.zone
            if i - b.at > p.breakout_window or z.contains(close):
                self.pending.remove(b)      # too slow, or it fell back inside: no breakout
                continue
            tol = p.touch_tolerance * z.height
            if b.direction == "up":
                if b.pulled_back is None and low <= z.high + tol:
                    b.pulled_back = i
                if b.pulled_back is not None and i > b.pulled_back and close > b.extreme:
                    found.append(self._breakout(b, i, "long"))
            else:
                if b.pulled_back is None and high >= z.low - tol:
                    b.pulled_back = i
                if b.pulled_back is not None and i > b.pulled_back and close < b.extreme:
                    found.append(self._breakout(b, i, "short"))
        return [s for s in found if s]

    def _breakout(self, b: _PendingBreak, i: int, side: str) -> Setup | None:
        self.pending.remove(b)
        z, buffer = b.zone, self.p.stop_buffer_atr * b.zone.atr
        # The stop is below the range for a long, above it for a short: the
        # structure the breakout left. Small ranges keep it close.
        stop = z.low - buffer if side == "long" else z.high + buffer
        entry = b.extreme    # a limit at the level the confirming bar closed beyond
        notes = ("small range: close stop",) if z.height < self.p.big_range_atr * z.atr else \
                ("big range: wide stop",)
        return self._setup(BREAKOUT, side, i, z, entry, stop, partials(target_levels(z, side, entry, self._prior(z))), notes)

    def _fake_breakout(self, event: Event, i: int) -> Setup | None:
        z, buffer = event.zone, self.p.stop_buffer_atr * event.zone.atr
        if event.direction == "down":
            side, entry, stop = "long", z.low, event.extreme - buffer
            notes = ("sellers failed; hold past the top",)
        else:
            if not self.p.fake_breakout_shorts:
                return None
            side, entry, stop = "short", z.high, event.extreme + buffer
            notes = ("buyers failed; the less preferred side",)
        return self._setup(FAKE_BREAKOUT, side, i, z, entry, stop, partials(target_levels(z, side, entry, self._prior(z))), notes)

    # -- shared ------------------------------------------------------------

    def _prior(self, zone: Zone) -> list[Zone]:
        return [z for z in self.tracker.zones if z is not zone]

    def _setup(self, kind: str, side: str, i: int, zone: Zone, entry: float, stop: float,
               targets: tuple[float, ...], notes: tuple[str, ...]) -> Setup | None:
        if not targets:
            return None
        direction = 1 if side == "long" else -1
        if direction * (entry - stop) <= 0 or direction * (targets[-1] - entry) <= 0:
            return None
        units = size_units(self.equity, self.p.risk_pct, entry, stop, self.p.lot, self.p.max_position_pct)
        setup = Setup(kind, side, i, self.dates[i], float(entry), float(stop), tuple(float(t) for t in targets),
                      zone, units, units * abs(entry - stop), self.bias, notes + self._strength(zone))
        return setup if setup.rr >= self.p.min_rr else None

    def _strength(self, zone: Zone) -> tuple[str, ...]:
        notes = []
        if self.value_area and self.value_area.near(zone.low, self.p.touch_tolerance * zone.height):
            notes.append("on the week's value area low")
        elif self.value_area and self.value_area.near(zone.high, self.p.touch_tolerance * zone.height):
            notes.append("on the week's value area high")
        if any(z.overlaps(zone) for z in self._prior(zone)):
            notes.append("stacked on an earlier range")
        return tuple(notes)


@dataclass
class _PendingBreak:
    zone: Zone
    direction: str
    extreme: float
    at: int
    pulled_back: int | None = None
