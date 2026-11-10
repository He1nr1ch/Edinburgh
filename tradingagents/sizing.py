"""Turn a list run's calls into a sized long/short book for a given equity.

Each ticker is analysed alone, so the position sizes the agents give are not
sized against one another: across the S&P 100 they summed to 25% for the Buys
and over 150% with the Holds. This step sizes the whole list at once, by
arithmetic rather than a model, so the same calls always give the same book.

Buys and Overweights go long, Sells and Underweights go short, Holds get no
position. Each position's weight is in proportion to its Risk Manager ceiling,
scaled so longs plus shorts reach the target share of equity (the gross
exposure), with no position above the cap. A position the cap stops is held
there and the rest is spread over the others; when every position is at the
cap, the book falls short of the target and says by how much.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from tradingagents.feedback.extract import labelled_number

# Which way each rating puts money; a Hold takes no position.
SIDE = {"Buy": 1, "Overweight": 1, "Sell": -1, "Underweight": -1}


@dataclass(frozen=True)
class Call:
    ticker: str
    rating: str
    ceiling_pct: float | None       # the Risk Manager's maximum position, % of equity
    risk_level: str | None = None
    target: float | None = None     # the Portfolio Manager's price target
    stop: float | None = None       # the Risk Manager's stop, else the Trader's
    stop_from: str = ""             # who gave it: "Risk Manager" or "Trader"
    decided: str = ""               # the analysis date of the decision, YYYY-MM-DD

    def target_move_pct(self, close: float | None) -> float | None:
        """The move from ``close`` to the target in the position's direction, as a percentage."""
        side = SIDE.get(self.rating)
        if not side or not self.target or not close:
            return None
        return side * (self.target / close - 1) * 100


def stop_for(call: Call, close: float, default_pct: float) -> tuple[float, str]:
    """The stop to place, and where it came from.

    The desk's stop when it gave one on the losing side of the close; else, by
    the user's rule, ``default_pct`` from the close against the position (below
    for a long, above for a short). A stop on the profit side would never
    protect anything, so it counts as none.
    """
    side = SIDE.get(call.rating, 0)
    if call.stop and side and (call.stop - close) * side < 0:
        return call.stop, call.stop_from or "desk"
    return round(close * (1 - side * default_pct / 100), 2), f"default {default_pct:g}%"


def worth_trading(call: Call, close: float | None, min_target_pct: float) -> str | None:
    """Why ``call`` is not taken, or None when it is.

    The user's rule: a trade needs a target more than ``min_target_pct`` from
    the close in its own direction. A call with no target, or one on the wrong
    side, has no expected profit to show and is not taken either.
    """
    if min_target_pct <= 0:
        return None
    move = call.target_move_pct(close)
    if move is None:
        return "no price target" if not call.target else "no closing price"
    if move <= 0:
        return f"target on the wrong side of the close ({move:+.1f}%)"
    if move < min_target_pct:
        return f"target {move:+.1f}% is under the {min_target_pct:g}% minimum"
    return None


@dataclass
class Order:
    ticker: str
    rating: str
    risk_level: str | None
    ceiling_pct: float
    weight_pct: float               # signed: negative is short
    price: float
    shares: int                     # signed: negative is short
    stop: float | None = None
    stop_from: str = ""
    target: float | None = None
    score: float | None = None      # see ranking.py; set when the book is built

    @property
    def side(self) -> str:
        return "long" if self.shares > 0 else "short"

    @property
    def value(self) -> float:
        return abs(self.shares) * self.price


@dataclass
class Book:
    equity: float
    target_gross_pct: float
    cap_pct: float
    orders: list[Order] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)   # (ticker, why)

    @property
    def long_value(self) -> float:
        return sum(o.value for o in self.orders if o.shares > 0)

    @property
    def short_value(self) -> float:
        return sum(o.value for o in self.orders if o.shares < 0)

    @property
    def gross_pct(self) -> float:
        return 100 * (self.long_value + self.short_value) / self.equity

    @property
    def net_pct(self) -> float:
        return 100 * (self.long_value - self.short_value) / self.equity

    @property
    def cash(self) -> float:
        """Cash left after buying the longs; a short's proceeds are held as margin, not spent."""
        return self.equity - self.long_value


def target_weights(calls: list[Call], gross_pct: float, cap_pct: float) -> dict[str, float]:
    """Signed weights (% of equity) in proportion to each call's ceiling, summing to ``gross_pct``.

    Weights over ``cap_pct`` are held at the cap and the remainder goes to the
    rest, until the target is met or every position is at the cap.
    """
    active = [c for c in calls if SIDE.get(c.rating) and (c.ceiling_pct or 0) > 0]
    weights: dict[str, float] = {}
    remaining = gross_pct
    while active and remaining > 1e-9:
        scale = remaining / sum(c.ceiling_pct for c in active)
        capped = [c for c in active if c.ceiling_pct * scale > cap_pct]
        if not capped:
            weights |= {c.ticker: c.ceiling_pct * scale for c in active}
            break
        for c in capped:
            weights[c.ticker] = cap_pct
            remaining -= cap_pct
        active = [c for c in active if c not in capped]
    return {c.ticker: SIDE[c.rating] * weights[c.ticker] for c in calls if c.ticker in weights}


def top_per_side(calls: list[Call], prices: dict[str, float], n: int) -> list[Call]:
    """The ``n`` longs and ``n`` shorts with the largest expected move to their target.

    The user's rule: a book of the ten best of each side, not every call that
    clears the bar. Calls without a measurable move rank last; 0 keeps every call.
    """
    if n <= 0:
        return list(calls)
    kept = []
    for side in (1, -1):
        group = [c for c in calls if SIDE.get(c.rating) == side]
        group.sort(key=lambda c: (-(c.target_move_pct(prices.get(c.ticker)) or 0.0), c.ticker))
        kept.extend(group[:n])
    return kept


def bench_for(calls: list[Call], prices: dict[str, float], chosen: set[str],
              min_target_pct: float, n: int) -> list[Call]:
    """The ``n`` best calls after the book, long or short together, by expected move.

    The user's rule: a short list of replacements ready for the day, for the
    Watcher to pick from when a book position sells out. Only calls that clear
    the floor and are not in the book qualify.
    """
    pool = [c for c in calls if SIDE.get(c.rating) and c.ticker not in chosen
            and prices.get(c.ticker) and worth_trading(c, prices[c.ticker], min_target_pct) is None]
    pool.sort(key=lambda c: (-(c.target_move_pct(prices[c.ticker]) or 0.0), c.ticker))
    return pool[:n]


def size_book(calls: list[Call], prices: dict[str, float], equity: float,
              gross_pct: float = 75.0, cap_pct: float = 10.0, min_target_pct: float = 0.0,
              names_per_side: int = 0, default_stop_pct: float = 2.0) -> Book:
    """The book for ``equity`` in whole shares at ``prices``.

    Calls whose target is under ``min_target_pct`` from the close are left out
    (see :func:`worth_trading`); 0 takes every call. Of those left, only the
    ``names_per_side`` longs and shorts with the largest expected move are
    sized (see :func:`top_per_side`); 0 sizes them all.

    Each position is rounded down to whole shares, then single shares go back
    to the positions furthest below their weight, while the book stays within
    the gross target and each position within the cap. Rounding down alone left
    a list of small positions several percent short of the target.
    """
    book = Book(equity=equity, target_gross_pct=gross_pct, cap_pct=cap_pct)
    taken = []
    for c in calls:
        if not SIDE.get(c.rating):
            continue
        why_not = worth_trading(c, prices.get(c.ticker), min_target_pct)
        if why_not:
            book.skipped.append((c.ticker, why_not))
        elif not (c.ceiling_pct or 0) > 0:
            book.skipped.append((c.ticker, f"{c.rating} with a Risk Manager ceiling of 0%"))
        elif not prices.get(c.ticker):
            book.skipped.append((c.ticker, "no closing price"))
        else:
            taken.append(c)
    # A chosen name whose weight is under one share (a $6,000 stock on a small
    # book) gives up its place to the next on its side, so the book keeps its
    # count. Each pass drops such names and refills; the pool only shrinks.
    pool = [c for c in taken if prices.get(c.ticker)]
    while True:
        chosen = top_per_side(pool, prices, names_per_side)
        weights = target_weights(chosen, gross_pct, cap_pct)
        counts = {t: math.floor(equity * abs(w) / 100 / prices[t]) for t, w in weights.items()}
        unbuyable = [t for t, n in counts.items() if n == 0]
        if not unbuyable:
            break
        for t in unbuyable:
            book.skipped.append((t, f"{abs(weights[t]):.2f}% is less than one share at {prices[t]:,.2f}"))
        pool = [c for c in pool if c.ticker not in unbuyable]
    for c in taken:
        if c not in chosen and c.ticker not in dict(book.skipped):
            move = c.target_move_pct(prices.get(c.ticker))
            book.skipped.append((c.ticker, f"outside the top {names_per_side} on its side"
                                           + (f" ({move:+.1f}% to target)" if move is not None else "")))
    priced = chosen
    by_ticker = {c.ticker: c for c in priced}

    budget = equity * gross_pct / 100 - sum(n * prices[t] for t, n in counts.items())
    while True:
        def shortfall(t):
            return equity * abs(weights[t]) / 100 - counts[t] * prices[t]
        fits = [t for t in counts if prices[t] <= budget
                and (counts[t] + 1) * prices[t] <= equity * cap_pct / 100]
        if not fits:
            break
        best = max(fits, key=shortfall)
        if shortfall(best) <= 0:
            break
        counts[best] += 1
        budget -= prices[best]

    for ticker, weight in weights.items():
        c, price, n = by_ticker[ticker], prices[ticker], counts[ticker]
        stop, stop_from = stop_for(c, price, default_stop_pct)
        book.orders.append(Order(ticker, c.rating, c.risk_level, c.ceiling_pct, weight, price,
                                 int(math.copysign(n, weight)), stop, stop_from, c.target))
    book.orders.sort(key=lambda o: (o.shares < 0, -o.value))
    return book


def call_from_state(state: dict, ticker: str = "", decided: str = "") -> Call:
    """The desk's call in a saved run state."""
    rm_stop = labelled_number(state.get("risk_assessment") or "", "Stop Loss")
    tr_stop = labelled_number(state.get("trader_investment_plan") or "", "Stop Loss")
    return Call(
        ticker=state.get("company_of_interest") or ticker,
        rating=state.get("final_rating") or "",
        ceiling_pct=labelled_number(state.get("risk_assessment") or "", "Maximum Position"),
        risk_level=state.get("final_risk_level") or state.get("risk_level"),
        target=labelled_number(state.get("final_trade_decision") or "", "Price Target"),
        stop=rm_stop or tr_stop,
        stop_from="Risk Manager" if rm_stop else ("Trader" if tr_stop else ""),
        decided=decided or state.get("trade_date") or "",
    )


def load_calls(run_dir: Path, trade_date: str) -> list[Call]:
    """Every ticker's final call on ``trade_date`` from a run's saved states."""
    return [call_from_state(json.loads(path.read_text(encoding="utf-8")), path.parts[-3], trade_date)
            for path in sorted(Path(run_dir).glob(f"*/TradingAgentsStrategy_logs/full_states_log_{trade_date}.json"))]


def last_closes(tickers: list[str], trade_date: str) -> dict[str, float]:
    """Each ticker's close on ``trade_date``, or the last one before it."""
    from tradingagents.dataflows.vendors.alpaca import get_closes

    end = (datetime.strptime(trade_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    start = (datetime.strptime(trade_date, "%Y-%m-%d") - timedelta(days=10)).strftime("%Y-%m-%d")
    prices = {}
    for ticker in tickers:
        closes = get_closes(ticker, start, end)
        if not closes.empty:
            prices[ticker] = float(closes.iloc[-1])
    return prices
