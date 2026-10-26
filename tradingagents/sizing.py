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


@dataclass(frozen=True)
class Order:
    ticker: str
    rating: str
    risk_level: str | None
    ceiling_pct: float
    weight_pct: float               # signed: negative is short
    price: float
    shares: int                     # signed: negative is short

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


def size_book(calls: list[Call], prices: dict[str, float], equity: float,
              gross_pct: float = 75.0, cap_pct: float = 10.0) -> Book:
    """The book for ``equity`` in whole shares at ``prices``.

    Each position is rounded down to whole shares, then single shares go back
    to the positions furthest below their weight, while the book stays within
    the gross target and each position within the cap. Rounding down alone left
    a list of small positions several percent short of the target.
    """
    book = Book(equity=equity, target_gross_pct=gross_pct, cap_pct=cap_pct)
    for c in calls:
        if not SIDE.get(c.rating):
            continue
        if not (c.ceiling_pct or 0) > 0:
            book.skipped.append((c.ticker, f"{c.rating} with a Risk Manager ceiling of 0%"))
        elif not prices.get(c.ticker):
            book.skipped.append((c.ticker, "no closing price"))
    priced = [c for c in calls if prices.get(c.ticker)]
    by_ticker = {c.ticker: c for c in priced}
    weights = target_weights(priced, gross_pct, cap_pct)
    counts = {t: math.floor(equity * abs(w) / 100 / prices[t]) for t, w in weights.items()}

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
        if n == 0:
            book.skipped.append((ticker, f"{abs(weight):.2f}% is less than one share at {price:,.2f}"))
            continue
        book.orders.append(Order(ticker, c.rating, c.risk_level, c.ceiling_pct, weight, price,
                                 int(math.copysign(n, weight))))
    book.orders.sort(key=lambda o: (o.shares < 0, -o.value))
    return book


def load_calls(run_dir: Path, trade_date: str) -> list[Call]:
    """Every ticker's final call on ``trade_date`` from a run's saved states."""
    calls = []
    for path in sorted(Path(run_dir).glob(f"*/TradingAgentsStrategy_logs/full_states_log_{trade_date}.json")):
        state = json.loads(path.read_text(encoding="utf-8"))
        calls.append(Call(
            ticker=state.get("company_of_interest") or path.parts[-3],
            rating=state.get("final_rating") or "",
            ceiling_pct=labelled_number(state.get("risk_assessment") or "", "Maximum Position"),
            risk_level=state.get("final_risk_level") or state.get("risk_level"),
        ))
    return calls


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
