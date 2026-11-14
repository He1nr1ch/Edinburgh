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


def _client(client=None):
    """The configured account's client, or the one given (a second account, see mirror.py)."""
    return client if client is not None else common.trading_client(common.paper_trading())


def get_account(client=None, paper: bool | None = None) -> Account:
    """The account's equity, cash and every open position (of the client given, when one is)."""
    client = _client(client)
    account = common.call(client.get_account, "the trading account")
    positions = common.call(client.get_all_positions, "open positions") or []
    if account is None:
        raise VendorUnavailableError("Alpaca returned no trading account")
    holdings = tuple(
        Holding(p.symbol, float(p.qty), float(p.market_value or 0), float(p.avg_entry_price or 0),
                str(getattr(p.asset_class, "value", p.asset_class)))
        for p in positions
    )
    return Account(float(account.equity), float(account.cash), common.paper_trading() if paper is None else paper, holdings)


def get_clock() -> Clock:
    clock = common.call(_client().get_clock, "the market clock")
    if clock is None:
        raise VendorUnavailableError("Alpaca returned no market clock")
    return Clock(bool(clock.is_open), clock.timestamp, clock.next_open, clock.next_close)


def last_trading_day(before: datetime) -> str:
    """The most recent session that closed before ``before`` (a date, YYYY-MM-DD), by Alpaca's calendar."""
    from datetime import timedelta

    from alpaca.trading.requests import GetCalendarRequest

    request = GetCalendarRequest(start=(before - timedelta(days=10)).date(), end=before.date())
    days = common.call(lambda: _client().get_calendar(request), "the market calendar") or []
    # A session counts once its close has passed: any earlier date, or today
    # after the close. Calendar times are Eastern; ``before`` is local time.
    local = before.astimezone()
    closed = [d for d in days if d.date < local.date()
              or (d.date == local.date() and local.time() > d.close.time())]
    if not closed:
        raise VendorUnavailableError("Alpaca's calendar shows no session before " + before.isoformat())
    return closed[-1].date.isoformat()


def fills_on(day: str) -> dict[str, dict]:
    """Alpaca's record of every order placed on ``day`` (a date), by order id: fill price, quantity, status."""
    from zoneinfo import ZoneInfo

    from alpaca.trading.enums import QueryOrderStatus
    from alpaca.trading.requests import GetOrdersRequest

    start = datetime.fromisoformat(day).replace(tzinfo=ZoneInfo("America/New_York"))
    request = GetOrdersRequest(status=QueryOrderStatus.ALL, after=start, until=start.replace(hour=23, minute=59),
                               limit=500, nested=True)
    orders = common.call(lambda: _client().get_orders(filter=request), f"orders on {day}") or []
    found = {}
    for o in orders:
        for order in (o, *(getattr(o, "legs", None) or ())):
            price = getattr(order, "filled_avg_price", None)
            found[str(order.id)] = {"filled_avg_price": float(price) if price else None,
                                    "filled_qty": float(order.filled_qty or 0),
                                    "status": str(getattr(order.status, "value", order.status))}
    return found


def open_orders(client=None) -> dict[str, str]:
    """Every order sent but not yet filled or cancelled, by order id: its symbol."""
    from alpaca.trading.enums import QueryOrderStatus
    from alpaca.trading.requests import GetOrdersRequest

    request = GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500, nested=True)
    orders = common.call(lambda: _client(client).get_orders(filter=request), "open orders") or []
    found = {}
    for o in orders:
        for order in (o, *(getattr(o, "legs", None) or ())):
            found[str(order.id)] = order.symbol
    return found


def cancel_orders_for(symbol: str, open_now: dict[str, str], client=None) -> int:
    """Cancel every open order for ``symbol`` (a bracket's legs included); returns how many.

    Returns once Alpaca shows them cancelled (or a few seconds have passed): it
    holds the shares a waiting stop has claimed until the cancel is through,
    and refuses a market order for them sent before that.
    """
    ids = [order_id for order_id, sym in open_now.items() if sym == symbol]
    for order_id in ids:
        cancel_order(order_id, client)
    _await_closed(ids, client)
    return len(ids)


def cancel_order(order_id: str, client=None) -> None:
    common.call(lambda: _client(client).cancel_order_by_id(order_id), f"cancelling order {order_id}")


_OPEN_STATUSES = {"new", "accepted", "pending_new", "partially_filled", "pending_cancel", "pending_replace",
                  "accepted_for_bidding", "held"}
CANCEL_WAIT_SECONDS = 3.0


def _await_closed(ids: list[str], client=None) -> None:
    """Wait, briefly, until none of the orders is open any more; a failure to ask is not a failure to cancel."""
    import time

    deadline = time.monotonic() + CANCEL_WAIT_SECONDS
    waiting = list(ids)
    while waiting and time.monotonic() < deadline:
        try:
            waiting = [oid for oid in waiting if order_result(oid, client)["status"] in _OPEN_STATUSES]
        except Exception:
            return
        if waiting:
            time.sleep(0.25)


@dataclass(frozen=True)
class StopOrder:
    """A stop order waiting at Alpaca: ``side`` is the order's (sell closes a long, buy covers a short)."""
    id: str
    symbol: str
    side: str
    qty: int
    stop_price: float


def _value(x) -> str:
    return str(getattr(x, "value", x))


def open_stops(client=None) -> list[StopOrder]:
    """Every plain stop order waiting at Alpaca (a bracket's legs are not among them)."""
    from alpaca.trading.enums import QueryOrderStatus
    from alpaca.trading.requests import GetOrdersRequest

    request = GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500, nested=True)
    orders = common.call(lambda: _client(client).get_orders(filter=request), "open orders") or []
    return [StopOrder(str(o.id), o.symbol, _value(o.side), int(float(o.qty or 0)), float(o.stop_price))
            for o in orders
            if _value(getattr(o, "order_type", None) or getattr(o, "type", "")) == "stop" and o.stop_price]


def submit_stop_order(symbol: str, qty: int, side: str, stop_price: float, client=None) -> str:
    """Leave a day stop order at Alpaca (``side`` is buy or sell); returns its order id.

    Alpaca holds it and sends a market order when the price trades through
    ``stop_price``, whether or not the worker is running. It lapses at the close.
    """
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import StopOrderRequest

    request = StopOrderRequest(symbol=symbol, qty=qty, side=OrderSide(side), time_in_force=TimeInForce.DAY,
                               stop_price=stop_price)
    order = common.call(lambda: _client(client).submit_order(request), f"a {side} stop for {qty} {symbol}")
    if order is None:
        raise VendorUnavailableError(f"Alpaca did not accept the {side} stop for {qty} {symbol}")
    return str(order.id)


def replace_stop_order(order_id: str, qty: int, stop_price: float, client=None) -> str:
    """Change a waiting stop's size and price in one request; returns the id of the order that replaces it."""
    from alpaca.trading.requests import ReplaceOrderRequest

    request = ReplaceOrderRequest(qty=qty, stop_price=stop_price)
    order = common.call(lambda: _client(client).replace_order_by_id(order_id, request), f"replacing stop {order_id}")
    if order is None:
        raise VendorUnavailableError(f"Alpaca did not replace the stop {order_id}")
    return str(order.id)


def order_result(order_id: str, client=None) -> dict:
    """What became of an order: its status, how much of it filled, at what price and when."""
    order = common.call(lambda: _client(client).get_order_by_id(order_id), f"order {order_id}")
    if order is None:
        raise VendorUnavailableError(f"Alpaca has no order {order_id}")
    price = getattr(order, "filled_avg_price", None)
    return {"status": _value(order.status), "filled_qty": float(order.filled_qty or 0),
            "filled_avg_price": float(price) if price else None, "filled_at": getattr(order, "filled_at", None)}


def submit_market_order(symbol: str, qty: int, side: str, client=None) -> str:
    """Send a day market order (``side`` is buy or sell); returns Alpaca's order id."""
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    request = MarketOrderRequest(symbol=symbol, qty=qty, side=OrderSide(side), time_in_force=TimeInForce.DAY)
    order = common.call(lambda: _client(client).submit_order(request), f"a {side} order for {qty} {symbol}")
    if order is None:
        raise VendorUnavailableError(f"Alpaca did not accept the {side} order for {qty} {symbol}")
    return str(order.id)
