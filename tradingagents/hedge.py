"""Hedge a net-short book against a market rally with SPY calls.

The book's market exposure is beta-weighted: each position's value times its
beta to SPY over the past year, so a short in a stock that barely follows the
market counts for little. A book that is net short by that measure is hedged
with at-the-money SPY calls, enough contracts to cover the exposure.

Which expiry: the one-week call, unless the one-month call costs less than
``monthly_ratio`` times the weekly. A month is about 4.3 weeks, so rolling
weekly calls costs more per week of cover; the ratio says how much cheaper
per week the month must be before it wins. Either can win: it depends on the
quotes of the day.

Once hedged, no more calls are bought: while a recorded hedge has not
expired, the plan buys nothing and reports how much of the exposure the
held calls cover.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from tradingagents.dataflows.vendors.alpaca import OptionQuote


@dataclass(frozen=True)
class HeldHedge:
    """SPY calls already bought, from the hedge ledger."""
    symbol: str
    contracts: int
    expiry: date
    bought_on: str
    cost: float


@dataclass(frozen=True)
class Hedge:
    exposure: float                 # beta-weighted net value; negative is net short
    spy_price: float
    contracts: int
    choice: str | None              # "weekly", "monthly", or None when no hedge is bought
    quote: OptionQuote | None
    weekly: OptionQuote | None
    monthly: OptionQuote | None
    reason: str
    held: tuple[HeldHedge, ...] = ()

    @property
    def cost(self) -> float:
        return self.contracts * self.quote.contract_cost if self.quote else 0.0

    @property
    def coverage(self) -> float:
        """Share notional of the calls held and bought, over the exposure (1.0 is exact)."""
        contracts = self.contracts + sum(h.contracts for h in self.held)
        return contracts * 100 * self.spy_price / abs(self.exposure) if self.exposure else 0.0


def beta(stock: pd.Series, market: pd.Series) -> float:
    """Beta of daily returns of ``stock`` to ``market`` over the dates both trade."""
    joined = pd.concat([stock.pct_change(), market.pct_change()], axis=1, join="inner").dropna()
    if len(joined) < 20:
        return 1.0      # too little history to say; count it as moving with the market
    return float(np.cov(joined.iloc[:, 0], joined.iloc[:, 1])[0, 1] / np.var(joined.iloc[:, 1], ddof=1))


def choose_expiry(weekly: OptionQuote | None, monthly: OptionQuote | None,
                  monthly_ratio: float) -> tuple[OptionQuote | None, str | None, str]:
    """The weekly call unless the monthly costs less than ``monthly_ratio`` times it."""
    if weekly is None and monthly is None:
        return None, None, "no SPY call quotes"
    if weekly is None:
        return monthly, "monthly", "no weekly quote"
    if monthly is None:
        return weekly, "weekly", "no monthly quote"
    ratio = monthly.ask / weekly.ask
    if ratio < monthly_ratio:
        return monthly, "monthly", f"monthly is {ratio:.2f}x the weekly, under {monthly_ratio:g}x"
    return weekly, "weekly", f"monthly is {ratio:.2f}x the weekly, not under {monthly_ratio:g}x"


def contracts_for(exposure: float, spy_price: float) -> int:
    """Calls to cover a net-short exposure, to the nearest contract; none for a net-long book."""
    if exposure >= 0:
        return 0
    return max(1, round(-exposure / (100 * spy_price)))


def plan_hedge(exposure: float, spy_price: float, weekly: OptionQuote | None,
               monthly: OptionQuote | None, monthly_ratio: float = 2.1,
               held: tuple[HeldHedge, ...] = ()) -> Hedge:
    contracts = contracts_for(exposure, spy_price)
    if contracts == 0:
        return Hedge(exposure, spy_price, 0, None, None, weekly, monthly,
                     "the book is not net short, so a rally does not hurt it", held)
    if held:
        names = ", ".join(f"{h.contracts} {h.symbol} until {h.expiry}" for h in held)
        return Hedge(exposure, spy_price, 0, None, None, weekly, monthly, f"already hedged: {names}", held)
    quote, choice, reason = choose_expiry(weekly, monthly, monthly_ratio)
    return Hedge(exposure, spy_price, contracts if quote else 0, choice, quote, weekly, monthly, reason)


def open_hedges(ledger: Path, as_of: str) -> tuple[HeldHedge, ...]:
    """The recorded hedges that have not expired by ``as_of``."""
    if not Path(ledger).is_file():
        return ()
    day = datetime.strptime(as_of, "%Y-%m-%d").date()
    held = [HeldHedge(e["symbol"], int(e["contracts"]), date.fromisoformat(e["expiry"]), e["bought_on"], float(e["cost"]))
            for e in json.loads(Path(ledger).read_text(encoding="utf-8"))]
    return tuple(h for h in held if h.expiry >= day)


def record_hedge(ledger: Path, plan: Hedge, bought_on: str) -> HeldHedge:
    """Add the plan's calls to the ledger as bought."""
    if not plan.contracts:
        raise ValueError("the plan buys no calls, so there is nothing to record")
    held = HeldHedge(plan.quote.symbol, plan.contracts, plan.quote.expiry, bought_on, round(plan.cost, 2))
    path = Path(ledger)
    entries = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
    entries.append({**asdict(held), "expiry": held.expiry.isoformat()})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    return held


def hedge_book(book, trade_date: str, monthly_ratio: float = 2.1,
               weekly_days: int = 7, monthly_days: int = 28,
               held: tuple[HeldHedge, ...] = ()) -> Hedge:
    """Price the hedge for a sized book: betas over the past year, SPY calls quoted now."""
    from tradingagents.dataflows.vendors.alpaca import atm_call, get_closes

    day = datetime.strptime(trade_date, "%Y-%m-%d")
    start, end = (day - timedelta(days=365)).strftime("%Y-%m-%d"), (day + timedelta(days=1)).strftime("%Y-%m-%d")
    spy = get_closes("SPY", start, end)
    exposure = sum(math.copysign(o.value, o.shares) * beta(get_closes(o.ticker, start, end), spy)
                   for o in book.orders)
    spy_price = float(spy.iloc[-1])
    weekly = atm_call("SPY", spy_price, (day + timedelta(days=weekly_days)).date())
    monthly = atm_call("SPY", spy_price, (day + timedelta(days=monthly_days)).date())
    return plan_hedge(exposure, spy_price, weekly, monthly, monthly_ratio, held)
