"""The Alpaca brokerage account: what is held, whether the market is open, and placing orders.

The account is paper or live by the key pair and ``alpaca_paper``; nothing here
chooses one. Only the daily worker calls these.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from tradingagents.dataflows.errors import VendorUnavailableError
from tradingagents.dataflows.vendors.alpaca import common


@dataclass(frozen=True)
class Holding:
    symbol: str
    qty: float              # signed: negative is short
    market_value: float     # signed, as Alpaca reports it
    avg_entry_price: float
    asset_class: str        # us_equity, us_option, crypto

    @property
    def side(self) -> str:
        return "long" if self.qty > 0 else "short"


@dataclass(frozen=True)
class Account:
    equity: float
    cash: float
    paper: bool
    holdings: tuple[Holding, ...]


@dataclass(frozen=True)
class Clock:
    is_open: bool
    now: datetime
    next_open: datetime
    next_close: datetime


def _client():
    return common.trading_client(common.paper_trading())


def get_account() -> Account:
    """The account's equity, cash and every open position."""
    client = _client()
    account = common.call(client.get_account, "the trading account")
    positions = common.call(client.get_all_positions, "open positions") or []
    if account is None:
        raise VendorUnavailableError("Alpaca returned no trading account")
    holdings = tuple(
        Holding(p.symbol, float(p.qty), float(p.market_value or 0), float(p.avg_entry_price or 0),
                str(getattr(p.asset_class, "value", p.asset_class)))
        for p in positions
    )
    return Account(float(account.equity), float(account.cash), common.paper_trading(), holdings)


def get_clock() -> Clock:
    clock = common.call(_client().get_clock, "the market clock")
    if clock is None:
        raise VendorUnavailableError("Alpaca returned no market clock")
    return Clock(bool(clock.is_open), clock.timestamp, clock.next_open, clock.next_close)


def open_orders() -> dict[str, str]:
    """Every order sent but not yet filled or cancelled, by order id: its symbol."""
    from alpaca.trading.enums import QueryOrderStatus
    from alpaca.trading.requests import GetOrdersRequest

    request = GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500, nested=True)
    orders = common.call(lambda: _client().get_orders(filter=request), "open orders") or []
    found = {}
    for o in orders:
        for order in (o, *(getattr(o, "legs", None) or ())):
            found[str(order.id)] = order.symbol
    return found


def cancel_orders_for(symbol: str, open_now: dict[str, str]) -> int:
    """Cancel every open order for ``symbol`` (a bracket's legs included); returns how many."""
    ids = [order_id for order_id, sym in open_now.items() if sym == symbol]
    for order_id in ids:
        common.call(lambda oid=order_id: _client().cancel_order_by_id(oid), f"cancelling order {order_id}")
    return len(ids)


def submit_market_order(symbol: str, qty: int, side: str) -> str:
    """Send a day market order (``side`` is buy or sell); returns Alpaca's order id."""
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    request = MarketOrderRequest(symbol=symbol, qty=qty, side=OrderSide(side), time_in_force=TimeInForce.DAY)
    order = common.call(lambda: _client().submit_order(request), f"a {side} order for {qty} {symbol}")
    if order is None:
        raise VendorUnavailableError(f"Alpaca did not accept the {side} order for {qty} {symbol}")
    return str(order.id)
