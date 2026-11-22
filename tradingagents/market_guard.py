"""The market guard: which way the whole market is running, read from SPY.

Every other guard of the desk looks at one name: its stop, its target, its
news. None of them sees that twenty names are falling because the market is,
and so a long stopped out in a sell-off was replaced with another long, which
was stopped out in turn. This reads the market itself, by arithmetic, with no
model asked: SPY's price now against its last close, and against its close
``market_guard_days`` sessions back.

A market past either threshold is running, down or up. While it runs the desk
opens nothing against it: no new long in a falling market, no new short in a
rising one, from the book or from the bench. What is held stays with its own
stop; the best call still wins among those that may be opened; and the two
sides are not balanced by rule. A falling market is also when a net-long book
is hedged with SPY puts (hedge.py).

A market further out still, past ``market_swing_day_pct`` or
``market_swing_period_pct``, has swung, and the desk is married to no
position: the Watcher closes what is held against it, the shorts in a swing
up and the longs in a swing down, at a profit or a loss (worker/market.py).

A sharp day against the run of the week (a bounce in a sell-off) is neither:
the two readings disagree, and nothing is held back or closed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
BENCHMARK = "SPY"


@dataclass(frozen=True)
class Tape:
    """SPY now, against where it closed."""
    price: float
    day_pct: float | None       # from the last close
    period_pct: float | None    # from the close ``days`` sessions back
    days: int
    direction: int              # -1 falling, +1 rising, 0 neither
    swing: int = 0              # -1 swung down, +1 swung up, 0 no swing: past the larger limits, the way it runs

    @property
    def falling(self) -> bool:
        return self.direction < 0

    @property
    def barred(self) -> str | None:
        """The side nothing new is opened on while the market runs: long, short, or None."""
        return {-1: "long", 1: "short"}.get(self.direction)

    @property
    def closes(self) -> str | None:
        """The side closed out while the market has swung: long, short, or None."""
        return {-1: "long", 1: "short"}.get(self.swing)

    def describe(self) -> str:
        parts = [f"{self.day_pct:+.1f}% on the day" if self.day_pct is not None else "",
                 f"{self.period_pct:+.1f}% over {self.days} sessions" if self.period_pct is not None else ""]
        moves = ", ".join(p for p in parts if p) or "no closes to compare"
        state = {-1: "falling", 1: "rising", 0: "neither falling nor rising"}[self.direction]
        state += {-1: ", a swing down", 1: ", a swing up", 0: ""}[self.swing]
        return f"{BENCHMARK} {self.price:,.2f}, {moves}: {state}"

    def why_not(self, side: str) -> str | None:
        """Why a new position on ``side`` is not opened now, or None when it may be."""
        if side != self.barred:
            return None
        return f"the market is {'falling' if self.falling else 'rising'} ({self.describe()}); no new {side}"


def _run(pct: float | None, limit: float) -> int:
    if pct is None or limit <= 0:
        return 0
    return -1 if pct <= -limit else 1 if pct >= limit else 0


def judge(price: float, closes: list[float], day_limit_pct: float, days: int, period_limit_pct: float,
          swing_day_pct: float = 0.0, swing_period_pct: float = 0.0) -> Tape:
    """Read ``price`` against ``closes``, the completed sessions before today, oldest first.

    Falling or rising when the day's move or the move over ``days`` sessions
    (today the last of them) is past its limit; neither when the two point
    opposite ways. A limit of 0 turns that reading off. A swing is the same
    question asked of the larger ``swing_`` limits, and only ever the way the
    market is running: a day past its swing limit against a week running the
    other way is the bounce again, and no swing.
    """
    day = (price / closes[-1] - 1) * 100 if closes else None
    period = (price / closes[-days] - 1) * 100 if days > 0 and len(closes) >= days else None
    runs = {_run(day, day_limit_pct), _run(period, period_limit_pct)} - {0}
    direction = runs.pop() if len(runs) == 1 else 0
    swung = {_run(day, swing_day_pct), _run(period, swing_period_pct)} - {0}
    swing = direction if direction and swung == {direction} else 0
    return Tape(price, day, period, days, direction, swing)


def closes(config: dict, today):
    """The benchmark's closes of the sessions before ``today``, oldest first, far enough back for the guard."""
    from tradingagents.dataflows.vendors.alpaca import get_closes

    days = int(config.get("market_guard_days") or 0)
    # Up to, not including, today: the sessions that have closed.
    return get_closes(BENCHMARK, (today - timedelta(days=2 * days + 14)).isoformat(), today.isoformat())


def judged(config: dict, price: float, history: list[float]) -> Tape:
    """``price`` read against ``history`` by the desk's own limits."""
    return judge(float(price), history, float(config.get("market_guard_day_pct") or 0),
                 int(config.get("market_guard_days") or 0), float(config.get("market_guard_period_pct") or 0),
                 float(config.get("market_swing_day_pct") or 0), float(config.get("market_swing_period_pct") or 0))


def read_tape(config: dict, now: datetime | None = None) -> Tape | None:
    """The market as it stands, or None when the guard is off or SPY cannot be read.

    None bars nothing: a desk that cannot see the market trades as it did
    before it had a guard, and says so in the log.
    """
    if not config.get("market_guard", True):
        return None
    from tradingagents.dataflows.vendors.alpaca import latest_price

    today = (now or datetime.now(ET)).astimezone(ET).date()
    try:
        history = closes(config, today)
        price = latest_price(BENCHMARK)
    except Exception as exc:
        logger.warning("The market guard could not read %s (%s); nothing is held back", BENCHMARK, exc)
        return None
    if not price or history.empty:
        logger.warning("The market guard has no %s price or closes; nothing is held back", BENCHMARK)
        return None
    return judged(config, price, [float(c) for c in history])
