"""From a news assessment to an order, and the guardrails every order passes.

The model judges the news; this module decides what that is allowed to do, by
fixed rules, so the same judgement always leads to the same order:

- news that hurts the position, material and unexpected: urgent exits, notable trims;
- news that helps it, urgent and unexpected: adds, up to the position cap;
- anything expected, mixed or minor: an alert, no order.
"""

from __future__ import annotations

import math
from datetime import datetime
from zoneinfo import ZoneInfo

from tradingagents.dataflows.vendors.alpaca.trading import Holding
from tradingagents.worker.duty import Action


def _closing_side(h: Holding) -> str:
    return "sell" if h.qty > 0 else "buy"


def _opening_side(h: Holding) -> str:
    return "buy" if h.qty > 0 else "sell"


def action_for(holding: Holding, materiality: str, direction: str, surprise: bool,
               equity: float, config: dict, reason: str) -> Action | None:
    """The order the news calls for, or None when it calls for an alert only."""
    if not surprise or direction == "mixed" or materiality not in ("medium", "high"):
        return None
    qty = abs(int(holding.qty))
    if direction == "adverse":
        if materiality == "high":
            return Action(holding.symbol, "exit", _closing_side(holding), qty, reason)
        trim = math.floor(qty * config["worker_trim_fraction"])
        return Action(holding.symbol, "trim", _closing_side(holding), trim, reason) if trim >= 1 else None
    if direction == "favorable" and materiality == "high":
        price = abs(holding.market_value) / qty if qty else 0
        room = equity * config["worker_position_cap_pct"] / 100 - abs(holding.market_value)
        add = min(math.floor(qty * config["worker_add_fraction"]), math.floor(room / price) if price else 0)
        return Action(holding.symbol, "add", _opening_side(holding), add, reason) if add >= 1 else None
    return None


ET = ZoneInfo("America/New_York")


def _now() -> datetime:
    return datetime.now(ET)


def too_early(config: dict, now: datetime | None = None) -> str | None:
    """Why no order may go out yet today, or None once ``worker_trade_from`` (ET) has come."""
    start = str(config.get("worker_trade_from") or "").strip()
    if not start:
        return None
    hour, minute = (int(x) for x in start.split(":"))
    local = (now or _now()).astimezone(ET)
    if (local.hour, local.minute) < (hour, minute):
        return f"no trades before {hour:02d}:{minute:02d} ET (worker_trade_from)"
    return None


def refusal(action: Action, *, market_open: bool, paper: bool, pending: set[str],
            config: dict, now: datetime | None = None) -> str | None:
    """Why the guardrails stop ``action``, or None when it may be sent.

    There is no limit on orders a day: a symbol trades again whenever new news
    calls for it. What stops it is an earlier order for it still unfilled, since
    the position read this pass does not yet show that order, and sizing a new
    one from it could sell the same shares twice.
    """
    if not config["worker_trading_enabled"]:
        return "trading is turned off (worker_trading_enabled)"
    if not paper and not config["worker_allow_live"]:
        return "the account is live and worker_allow_live is off"
    if not market_open:
        return "the market is closed"
    early = too_early(config, now)
    if early:
        return early
    if action.symbol in pending:
        return f"an earlier order for {action.symbol} has not filled yet"
    return None
