"""Walk a bar series with the scanner and trade its setups by Nill's rules.

Orders are limits, live from the bar after the setup; a bar whose range
reaches the price fills it, at the open when the bar opens past it. Stops and
targets are checked on every bar's high and low, the stop first when a bar
reaches both, and at the open when the bar opens past the stop: the test is
against the trader, never for them. Partials take a
third at each target when the position divides in three; after the first, the
stop moves to the entry. A newly validated range in the trade's direction
trails the stop under it (over it for a short); the stop is never moved the
other way. After ``max_losses_per_day`` losses in a row, no new orders that
day. A range order does not fill in the evening and a range trade still open
at the session's last bar is closed there, so none is held overnight unseen.
One position at a time.

Fills at the limit price with no slippage or commission: a test of whether the
rules have an edge on the bars, not a broker statement.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from tradingagents.pb_range.model import Params, ValueArea, four_hour_bias
from tradingagents.pb_range.setups import RANGE, Scanner, Setup


@dataclass
class Fill:
    at: pd.Timestamp
    price: float
    units: int
    reason: str         # target 1..3, stop, or end


@dataclass
class Trade:
    setup: Setup
    entry_at: pd.Timestamp
    entry: float
    units: int
    stop: float
    fills: list[Fill] = field(default_factory=list)
    next_target: int = 0

    @property
    def side(self) -> str:
        return self.setup.side

    @property
    def open_units(self) -> int:
        return self.units - sum(f.units for f in self.fills)

    @property
    def closed(self) -> bool:
        return self.open_units <= 0

    @property
    def pnl(self) -> float:
        return sum(self.setup.direction * (f.price - self.entry) * f.units for f in self.fills)

    @property
    def initial_risk(self) -> float:
        return self.setup.per_unit_risk * self.units

    @property
    def r_multiple(self) -> float:
        return self.pnl / self.initial_risk if self.initial_risk > 0 else 0.0

    @property
    def exit_at(self) -> pd.Timestamp | None:
        return self.fills[-1].at if self.fills else None

    def _partial_units(self) -> list[int]:
        n = len(self.setup.targets)
        if n == 1 or self.units % n:
            return [0] * (n - 1) + [self.units]
        return [self.units // n] * n


@dataclass
class Result:
    trades: list[Trade]
    setups: list[Setup]
    scanner: Scanner

    @property
    def closed(self) -> list[Trade]:
        return [t for t in self.trades if t.closed]

    def stats(self) -> dict[str, float | int]:
        trades = self.closed
        if not trades:
            return {"trades": 0}
        rs = [t.r_multiple for t in trades]
        wins = [r for r in rs if r > 0]
        losses = [r for r in rs if r <= 0]
        streak = worst = 0
        for r in rs:
            streak = streak + 1 if r <= 0 else 0
            worst = max(worst, streak)
        return {
            "trades": len(trades),
            "win_rate": len(wins) / len(trades),
            "total_r": sum(rs),
            "mean_r": sum(rs) / len(rs),
            "profit_factor": (sum(wins) / -sum(losses)) if losses and sum(losses) < 0 else float("inf"),
            "pnl": sum(t.pnl for t in trades),
            "worst_losing_streak": worst,
            "by_kind": {k: sum(1 for t in trades if t.setup.kind == k) for k in {t.setup.kind for t in trades}},
        }


def run(bars: pd.DataFrame, params: Params | None = None, *, equity: float = 100_000,
        fill_window: int = 8, value_area: ValueArea | None = None) -> Result:
    """Trade ``bars`` (15-minute, in order) from the first bar to the last."""
    p = params or Params()
    bars = bars.reset_index(drop=True)
    scanner = Scanner(bars, p, equity=equity, value_area=value_area,
                      bias=four_hour_bias(bars))
    dates = pd.to_datetime(bars["Date"])
    opens, highs, lows, closes = (bars[c].to_numpy(dtype=float) for c in ("Open", "High", "Low", "Close"))

    trades: list[Trade] = []
    pending: list[Setup] = []
    position: Trade | None = None
    losses_today, day = 0, None

    for i in range(len(bars)):
        when = dates[i]
        if day != when.date():
            day, losses_today = when.date(), 0

        # 1. the open position against this bar; a range trade ends with the session
        if position is not None:
            _manage(position, when, opens[i], highs[i], lows[i])
            last_of_day = i == len(bars) - 1 or dates[i + 1].date() != when.date()
            if (not position.closed and position.setup.kind == RANGE
                    and p.evening_hour is not None and last_of_day):
                position.fills.append(Fill(when, float(closes[i]), position.open_units, "close of day"))
            if position.closed:
                losses_today = losses_today + 1 if position.r_multiple <= 0 else 0
                position = None

        # 2. pending orders: expire, or fill when no position is open
        still: list[Setup] = []
        for s in pending:
            expired = (not s.zone.live) if s.kind == RANGE else (i - s.bar > fill_window)
            if expired:
                continue
            fill = None if position is not None else _fill(s, p, when, opens[i], highs[i], lows[i])
            if fill is None:
                still.append(s)
                continue
            position = Trade(s, when, fill, s.units, s.stop)
            trades.append(position)
            _manage(position, when, opens[i], highs[i], lows[i])
            if position.closed:
                losses_today = losses_today + 1 if position.r_multiple <= 0 else 0
                position = None
        pending = still

        # 3. this bar's setups and events
        new = scanner.step(i)
        if position is not None:
            _trail(position, scanner, p, closes[i])
        if losses_today < p.max_losses_per_day:     # otherwise the day is over
            pending += [s for s in new if s.units > 0]

    if position is not None and not position.closed:
        position.fills.append(Fill(dates.iloc[-1], float(closes[-1]), position.open_units, "end"))
    return Result(trades, scanner.setups, scanner)


def _fill(s: Setup, p: Params, when: pd.Timestamp, open_: float, high: float, low: float) -> float | None:
    """The fill price on this bar, or None: a market order takes the open, a limit waits for its price.

    A range order does not fill in the evening: a range trade is not held
    through the night unseen, and the order waits for the next session.
    """
    if s.kind == RANGE and p.evening_hour is not None and when.hour >= p.evening_hour:
        return None
    if p.entry == "close" and s.kind != RANGE:
        return open_
    return _limit_fill(s, open_, high, low)


def _limit_fill(s: Setup, open_: float, high: float, low: float) -> float | None:
    if s.side == "long":
        if open_ <= s.entry:
            return open_
        return s.entry if low <= s.entry else None
    if open_ >= s.entry:
        return open_
    return s.entry if high >= s.entry else None


def _manage(t: Trade, when: pd.Timestamp, open_: float, high: float, low: float) -> None:
    """Stop and targets against one bar, the stop first.

    A bar that opens past the stop fills it at the open: a gap is paid in full.
    """
    d = t.setup.direction
    if (low <= t.stop) if d > 0 else (high >= t.stop):
        price = min(t.stop, open_) if d > 0 else max(t.stop, open_)
        t.fills.append(Fill(when, price, t.open_units, "stop"))
        return
    pieces = t._partial_units()
    while not t.closed and t.next_target < len(t.setup.targets):
        target = t.setup.targets[t.next_target]
        if not ((high >= target) if d > 0 else (low <= target)):
            break
        units = pieces[t.next_target] or (t.open_units if t.next_target == len(pieces) - 1 else 0)
        if units:
            t.fills.append(Fill(when, target, min(units, t.open_units), f"target {t.next_target + 1}"))
            if t.next_target == 0 and not t.closed:
                t.stop = t.entry       # to break even after the first partial
        t.next_target += 1


def _trail(t: Trade, scanner: Scanner, p: Params, close: float) -> None:
    """A new valid range in the trade's direction: the stop goes under it, never back."""
    d = t.setup.direction
    for event in scanner.last_events:
        if event.kind != "validated" or event.zone is t.setup.zone:
            continue
        z = event.zone
        buffer = p.stop_buffer_atr * z.atr
        new_stop = z.low - buffer if d > 0 else z.high + buffer
        if d * (new_stop - t.stop) > 0 and d * (close - new_stop) > 0:
            t.stop = new_stop
