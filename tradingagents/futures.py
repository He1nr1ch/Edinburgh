"""The overnight market: what the S&P 500 futures say of the open, while SPY does not trade.

The desk does not trade futures, and nothing here is traded on. The S&P
future, the ten-year Treasury yield and the price of oil are background: what
the market did while the desk was not looking, for whoever reads the book.

The market guard (market_guard.py) reads SPY, and SPY is silent from Friday's
close until the pre-market on Monday: a market that falls three percent on a
Sunday night was first seen by the desk at nine the next morning. The E-mini
future (ES) trades from Sunday 18:00 New York time and nearly round the clock
after it, so its move since the cash close is the market's move while the
desk was not looking.

The reading is that move put on SPY's last close and judged as the guard
judges SPY itself: the same limits, the same words. It is a reading and
nothing else. No order follows from it: the desk still acts on SPY, from the
open. The worker looks while the market is shut (``futures_look_minutes``),
keeps the last reading in ``futures_state_path`` for the status page, says it
in the morning's book email, and emails when what the futures imply changes:
a market that was running at the close and no longer is, or one that has
begun to run or to swing since.

The quotes are Yahoo's public chart of ``ES=F``, delayed by some minutes and
asked for without a key; Alpaca has no index futures. A source that does not
answer leaves the desk as it was before it had this: the reading is missing
and the log says so.

Beside the future stand the two other things a desk looks at before the open
(:func:`background`): the ten-year Treasury yield (``^TNX``, quoted only in
the American day) and oil (``CL=F``, the front-month WTI future, the nearest
live price to spot). They are stated and judged by nothing: no limit, no
email, no order.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

from tradingagents import market_guard
from tradingagents.market_guard import Tape

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
CASH_CLOSE = (16, 0)        # New York; a half day closes at 13:00, and its reading is from 16:00 all the same
LIVE_MINUTES = 30           # a last trade older than this is a market that is shut, not one that is still
# symbol, what it is called, and how its move is said: a yield in basis points, a price in percent.
BACKGROUND = (("^TNX", "10-year Treasury yield", "yield"), ("CL=F", "WTI crude oil", "price"))


@dataclass(frozen=True)
class Overnight:
    """The future now against where it stood at the cash close, and SPY as that implies it."""
    symbol: str
    price: float
    at: datetime                # the future's last trade
    base: float                 # the future at the cash close
    close_day: str              # the session whose close it is measured from
    live: bool                  # the future is trading now
    tape: Tape                  # SPY's last close moved by as much, judged as the market guard judges

    @property
    def pct(self) -> float:
        return (self.price / self.base - 1) * 100

    @property
    def state(self) -> str:
        """Which way the implied market runs, as one word for comparing two readings."""
        return f"{self.tape.direction:+d}/{self.tape.swing:+d}"

    def describe(self) -> str:
        day = date.fromisoformat(self.close_day).strftime("%A")
        when = self.at.astimezone(ET).strftime("%a %H:%M ET")
        return (f"{self.symbol} {self.price:,.2f} at {when}{'' if self.live else ' (not trading now)'}, "
                f"{self.pct:+.1f}% since {day}'s close; implied {self.tape.describe()}")

    def means(self) -> str:
        """What the desk would do at the open were SPY to open where the future implies."""
        t = self.tape
        if not t.barred:
            return "Read as it stands, nothing is held back at the open."
        return (f"Read as it stands, no new {t.barred}s would be opened"
                + (f" and the {t.closes}s held would be closed from the open" if t.closes else "") + ".")

    def to_dict(self) -> dict:
        return {"symbol": self.symbol, "price": self.price, "at": self.at.isoformat(timespec="seconds"), "base": self.base,
                "close_day": self.close_day, "live": self.live, "pct": round(self.pct, 2), "state": self.state,
                "implied_spy": round(self.tape.price, 2), "text": self.describe(), "means": self.means()}


def fetch(symbol: str) -> list[tuple[datetime, float]]:
    """The future's five-minute closes over the last five days, oldest first."""
    response = requests.get(CHART_URL.format(symbol=symbol), params={"interval": "5m", "range": "5d", "includePrePost": "true"},
                            headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    response.raise_for_status()
    chart = response.json()["chart"]
    if chart.get("error") or not chart.get("result"):
        raise ValueError(f"no chart for {symbol}: {chart.get('error')}")
    result = chart["result"][0]
    closes = result["indicators"]["quote"][0]["close"]
    return [(datetime.fromtimestamp(t, ET), float(c)) for t, c in zip(result.get("timestamp") or [], closes, strict=False) if c]


def measure(bars: list[tuple[datetime, float]], close_day: str) -> tuple[float, float, datetime] | None:
    """The future at the cash close of ``close_day``, its last price and when; None when the bars do not reach."""
    close_at = _close_at(close_day)
    before = [(t, p) for t, p in bars if close_at - timedelta(hours=1) <= t <= close_at]
    if not before or not bars or bars[-1][0] <= close_at:
        return None
    return before[-1][1], bars[-1][1], bars[-1][0]


def _close_at(close_day: str) -> datetime:
    return datetime.combine(date.fromisoformat(close_day), datetime.min.time(), ET).replace(
        hour=CASH_CLOSE[0], minute=CASH_CLOSE[1])


def background(config: dict, close_day: str) -> list[dict]:
    """The ten-year yield and oil as they stand, and their move since the cash close when they have traded since.

    Background only. One that cannot be read is left out.
    """
    if not config.get("futures_watch", True):
        return []
    close_at, day, out = _close_at(close_day), date.fromisoformat(close_day).strftime("%A"), []
    for symbol, name, kind in BACKGROUND:
        try:
            bars = fetch(symbol)
        except Exception as exc:
            logger.warning("%s could not be read (%s)", name, exc)
            continue
        if not bars:
            continue
        at, value = bars[-1]
        before = [p for t, p in bars if t <= close_at and t.date() == close_at.date()]
        change = None
        if before and at > close_at:
            change = (value - before[-1]) * 100 if kind == "yield" else (value / before[-1] - 1) * 100
        level = f"{value:.2f}%" if kind == "yield" else f"{value:,.2f}"
        moved = "" if change is None else (f", {change:+.0f} bp" if kind == "yield" else f", {change:+.1f}%") + f" since {day}'s close"
        out.append({"symbol": symbol, "name": name, "value": round(value, 3), "at": at.isoformat(timespec="seconds"),
                    "change": None if change is None else round(change, 2), "unit": "bp" if kind == "yield" else "%",
                    "text": f"{name} {level} at {at.strftime('%a %H:%M ET')}{moved}"})
    return out


def read_overnight(config: dict, now: datetime | None = None) -> Overnight | None:
    """What the futures imply now, or None when they are not watched or cannot be read."""
    if not config.get("futures_watch", True):
        return None
    now = (now or datetime.now(ET)).astimezone(ET)
    symbol = str(config.get("futures_symbol") or "ES=F")
    try:
        closes = market_guard.closes(config, now.date())
        if closes.empty:
            raise ValueError(f"no {market_guard.BENCHMARK} closes")
        close_day = str(closes.index[-1])[:10]
        measured = measure(fetch(symbol), close_day)
    except Exception as exc:
        logger.warning("The futures could not be read (%s); the desk sees the market at the pre-market as before", exc)
        return None
    if measured is None:
        return None             # nothing has traded since the close
    base, price, at = measured
    history = [float(c) for c in closes]
    tape = market_guard.judged(config, history[-1] * price / base, history)
    return Overnight(symbol, price, at, base, close_day, now - at <= timedelta(minutes=LIVE_MINUTES), tape)


def at_close(config: dict, now: datetime | None = None) -> str | None:
    """The state of the market as the guard read it at the last close, to compare a first reading with."""
    now = (now or datetime.now(ET)).astimezone(ET)
    history = [float(c) for c in market_guard.closes(config, now.date())]
    if len(history) < 2:
        return None
    tape = market_guard.judged(config, history[-1], history[:-1])
    return f"{tape.direction:+d}/{tape.swing:+d}"


_looked_at = -1e9


def watch(config: dict, now: datetime | None = None) -> Overnight | None:
    """The worker's look while the market is shut: keep the reading, and email when what it implies changes.

    Called on every turn of the worker's loop; it asks the source only every
    ``futures_look_minutes``. A failure is logged and never the loop's.
    """
    global _looked_at
    from tradingagents.worker.notify import send_email

    minutes = float(config.get("futures_look_minutes") or 0)
    if not config.get("futures_watch", True) or minutes <= 0 or time.monotonic() - _looked_at < minutes * 60:
        return None
    _looked_at = time.monotonic()
    try:
        reading = read_overnight(config, now)
        if reading is None:
            return None
        path = Path(config["futures_state_path"])
        try:
            before = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        except ValueError:
            before = {}
        # What the desk last told its owner of this close; at first, what the
        # guard read at the close itself, so an unchanged market sends nothing.
        told = before.get("told") if before.get("close_day") == reading.close_day else at_close(config, now)
        also = background(config, reading.close_day)
        if reading.live and reading.state != told:
            logger.info("Futures (background): %s", reading.describe())
            way = {-1: "falling", 1: "rising", 0: "neither falling nor rising"}[reading.tape.direction]
            swing = {-1: ", a swing down", 1: ", a swing up", 0: ""}[reading.tape.swing]
            body = (f"{reading.describe()}\n" + "".join(f"{b['text']}\n" for b in also)
                    + f"\n{reading.means()}\n\nBackground only. The desk does not trade futures, and nothing is "
                    f"traded on this reading: it acts on {market_guard.BENCHMARK} itself, from the open.")
            if send_email(f"Edinburgh, background: the futures say the market is {way}{swing}", body, config):
                told = reading.state
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({**reading.to_dict(), "told": told, "background": also}, indent=2) + "\n", encoding="utf-8")
        return reading
    except Exception as exc:
        logger.warning("The futures could not be watched: %s", exc)
        return None


def last_reading(config: dict) -> dict:
    """The reading the worker last kept, for the status page; {} when there is none."""
    path = Path(config.get("futures_state_path") or "")
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, ValueError):
        return {}
