"""Option quotes from Alpaca: the at-the-money call for an expiry, for hedging a book."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from tradingagents.dataflows.vendors.alpaca import common


@dataclass(frozen=True)
class OptionQuote:
    symbol: str             # OCC symbol, e.g. SPY261030C00760000
    expiry: date
    strike: float
    bid: float
    ask: float
    quoted_at: datetime | None

    @property
    def contract_cost(self) -> float:
        """What one contract costs to buy at the ask (100 shares)."""
        return self.ask * 100


def _strike(symbol: str) -> float:
    return int(symbol[-8:]) / 1000


def _expiry(symbol: str) -> date:
    return datetime.strptime(symbol[-15:-9], "%y%m%d").date()


def atm_call(underlying: str, price: float, expiring_on_or_after: date,
             search_days: int = 14, strike_band: float = 0.02) -> OptionQuote | None:
    """The call nearest ``price`` on the first expiry on or after ``expiring_on_or_after``.

    Only quotes with a live ask count; None when Alpaca lists no such call.
    Quotes are the latest Alpaca has, not as of a past date.
    """
    from datetime import timedelta

    from alpaca.data.requests import OptionChainRequest

    request = OptionChainRequest(
        underlying_symbol=underlying.upper(), type="call",
        expiration_date_gte=expiring_on_or_after,
        expiration_date_lte=expiring_on_or_after + timedelta(days=search_days),
        strike_price_gte=price * (1 - strike_band), strike_price_lte=price * (1 + strike_band),
    )
    chain = common.call(lambda: common.option_client().get_option_chain(request),
                        f"{underlying} calls from {expiring_on_or_after}") or {}
    quoted = [(sym, snap.latest_quote) for sym, snap in chain.items()
              if snap.latest_quote is not None and (snap.latest_quote.ask_price or 0) > 0]
    if not quoted:
        return None
    first = min(_expiry(sym) for sym, _ in quoted)
    sym, q = min(((s, q) for s, q in quoted if _expiry(s) == first), key=lambda sq: abs(_strike(sq[0]) - price))
    return OptionQuote(sym, first, _strike(sym), float(q.bid_price or 0), float(q.ask_price), q.timestamp)
