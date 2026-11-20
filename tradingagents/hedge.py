"""Hedge a book against the market: SPY calls for a net-short one, SPY puts for a net-long one.

The book's market exposure is beta-weighted: each position's value times its
beta to SPY over the past year, so a short in a stock that barely follows the
market counts for little. A book that is net short by that measure is hedged
with at-the-money SPY calls, enough contracts to cover the exposure. A book
that is net long is hedged the same way with at-the-money SPY puts, but only
while the market is falling (market_guard.py): the desk's books lean long as
a matter of course, and puts bought every week of a rising market would be a
standing cost, where calls for the rarer net-short book are not.

Which expiry: the one-week call, unless the one-month call costs less than
``monthly_ratio`` times the weekly. A month is about 4.3 weeks, so rolling
weekly calls costs more per week of cover; the ratio says how much cheaper
per week the month must be before it wins. Either can win: it depends on the
quotes of the day.

Once hedged, no more are bought: while a recorded hedge of the same kind
has not expired, the plan buys nothing and reports how much of the exposure
the options held cover.
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
    """SPY calls or puts already bought, from the hedge ledger."""
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
    kind: str = "call"              # "call" against a rally, "put" against a fall

    @property
    def cost(self) -> float:
        return self.contracts * self.quote.contract_cost if self.quote else 0.0

    @property
    def coverage(self) -> float:
        """Share notional of the options held and bought, over the exposure (1.0 is exact)."""
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


def kind_of(symbol: str) -> str:
    """Whether an OCC option symbol is a call or a put (SPY261120P00710000 is a put)."""
    return "put" if symbol[-9:-8].upper() == "P" else "call"


def kind_for(exposure: float) -> str:
    """What hedges a book of this exposure: calls a net-short one, puts a net-long one."""
    return "call" if exposure < 0 else "put"


def contracts_for(exposure: float, spy_price: float, falling: bool = False) -> int:
    """Calls to cover a net-short exposure, or puts a net-long one in a
    ``falling`` market, to the nearest whole contract.

    None for a net-long book while the market is not falling, and none when
    the exposure is under half a contract's notional: the option then adds
    more exposure the other way than it covers.
    """
    if exposure >= 0 and not falling:
        return 0
    return round(abs(exposure) / (100 * spy_price))


def plan_hedge(exposure: float, spy_price: float, weekly: OptionQuote | None,
               monthly: OptionQuote | None, monthly_ratio: float = 2.1,
               held: tuple[HeldHedge, ...] = (), falling: bool = False) -> Hedge:
    kind = kind_for(exposure)
    held = tuple(h for h in held if kind_of(h.symbol) == kind)
    contracts = contracts_for(exposure, spy_price, falling)
    if contracts == 0:
        if exposure < 0:
            why = (f"the net-short exposure ({exposure:,.0f}) is under half a contract's {100 * spy_price:,.0f}, "
                   "so a call would add more long exposure than it covers")
        elif not falling:
            why = "the book is not net short, so a rally does not hurt it, and the market is not falling"
        else:
            why = (f"the market is falling, but the net-long exposure ({exposure:,.0f}) is under half a contract's "
                   f"{100 * spy_price:,.0f}, so a put would add more short exposure than it covers")
        return Hedge(exposure, spy_price, 0, None, None, weekly, monthly, why, held, kind)
    if held:
        names = ", ".join(f"{h.contracts} {h.symbol} until {h.expiry}" for h in held)
        return Hedge(exposure, spy_price, 0, None, None, weekly, monthly, f"already hedged: {names}", held, kind)
    quote, choice, reason = choose_expiry(weekly, monthly, monthly_ratio)
    if kind == "put":
        reason = reason.replace("no SPY call quotes", "no SPY put quotes") + "; the market is falling and the book is net long"
    return Hedge(exposure, spy_price, contracts if quote else 0, choice, quote, weekly, monthly, reason, kind=kind)


def open_hedges(ledger: Path, as_of: str) -> tuple[HeldHedge, ...]:
    """The recorded hedges that have not expired by ``as_of``."""
    if not Path(ledger).is_file():
        return ()
    day = datetime.strptime(as_of, "%Y-%m-%d").date()
    held = [HeldHedge(e["symbol"], int(e["contracts"]), date.fromisoformat(e["expiry"]), e["bought_on"], float(e["cost"]))
            for e in json.loads(Path(ledger).read_text(encoding="utf-8"))]
    return tuple(h for h in held if h.expiry >= day)


def record_held(ledger: Path, held: HeldHedge) -> HeldHedge:
    """Add calls or puts to the ledger as bought."""
    path = Path(ledger)
    entries = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
    entries.append({**asdict(held), "expiry": held.expiry.isoformat()})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    return held


def record_hedge(ledger: Path, plan: Hedge, bought_on: str) -> HeldHedge:
    """Add the plan's options to the ledger as bought."""
    if not plan.contracts:
        raise ValueError("the plan buys no options, so there is nothing to record")
    return record_held(ledger, HeldHedge(plan.quote.symbol, plan.contracts, plan.quote.expiry, bought_on, round(plan.cost, 2)))


HEDGE_SIDES = ("call (", "put (")    # a book row whose side starts so is the hedge: "call (weekly hedge)"


def hedge_row(plan: Hedge) -> list | None:
    """The hedge as a row of the book CSV, in the book's columns: the OCC symbol
    as the ticker, the ask as the price, contracts as shares, cost as value.
    None when the plan buys nothing."""
    if not plan.contracts:
        return None
    return [plan.quote.symbol, f"{plan.kind} ({plan.choice} hedge)", "", "", "", "",
            plan.quote.ask, plan.contracts, round(plan.cost, 2), "", "", ""]


def is_hedge_row(row: dict) -> bool:
    return str(row.get("side", "")).startswith(HEDGE_SIDES)


def held_from_row(row: dict, bought_on: str) -> HeldHedge:
    """The ledger entry for a hedge row of the book, once its calls are bought."""
    from tradingagents.dataflows.vendors.alpaca.options import expiry_of

    contracts = abs(int(float(row["shares"])))
    return HeldHedge(row["ticker"], contracts, expiry_of(row["ticker"]), bought_on, round(float(row["value"]), 2))


def describe_hedge(plan: Hedge | None) -> str:
    """One line on the hedge, for the log and the morning email."""
    if plan is None:
        return "not priced"
    if plan.contracts:
        return (f"buy {plan.contracts} {plan.quote.symbol} ({plan.choice}) at {plan.quote.ask:.2f}, "
                f"{plan.cost:,.0f} in all: {plan.reason}; covers {plan.coverage:.0%} of the "
                f"{plan.exposure:+,.0f} beta-weighted exposure")
    return f"none: {plan.reason}" + (f"; the {plan.kind}s held cover {plan.coverage:.0%}" if plan.held else "") \
        + f" (exposure {plan.exposure:+,.0f}, SPY {plan.spy_price:,.2f})"


def hedge_book(book, trade_date: str, monthly_ratio: float = 2.1,
               weekly_days: int = 7, monthly_days: int = 28,
               held: tuple[HeldHedge, ...] = (), falling: bool = False) -> Hedge:
    """Price the hedge for a sized book: betas over the past year, SPY options quoted now.

    Calls for a net-short book; puts for a net-long one when the market is
    ``falling``. The strike and contract count follow SPY's live price when
    Alpaca has one, since the option quotes are live; the analysis date's
    close is the fallback.
    """
    from tradingagents.dataflows.vendors.alpaca import atm_call, atm_put, get_closes, latest_price

    day = datetime.strptime(trade_date, "%Y-%m-%d")
    start, end = (day - timedelta(days=365)).strftime("%Y-%m-%d"), (day + timedelta(days=1)).strftime("%Y-%m-%d")
    spy = get_closes("SPY", start, end)
    exposure = sum(math.copysign(o.value, o.shares) * beta(get_closes(o.ticker, start, end), spy)
                   for o in book.orders)
    spy_price = latest_price("SPY") or float(spy.iloc[-1])
    quoted = atm_put if exposure >= 0 and falling else atm_call
    weekly = quoted("SPY", spy_price, (day + timedelta(days=weekly_days)).date())
    monthly = quoted("SPY", spy_price, (day + timedelta(days=monthly_days)).date())
    return plan_hedge(exposure, spy_price, weekly, monthly, monthly_ratio, held, falling)
