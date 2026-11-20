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


def stale_levels(side: str, price: float | None, stop: float | None, target: float | None,
                 min_target_pct: float, min_stop_pct: float = 0.0, close: float | None = None,
                 max_run_pct: float = 0.0) -> str | None:
    """Why a call sized at an earlier price is not opened at ``price`` now, or None when it still stands.

    The book and the bench are built on the last close and opened hours, and
    on a Monday days, later, while the stop and the target stay where the
    analysis put them. By then the price may be through the call's stop; so
    near it that ordinary noise takes the position out (the desk's own stop,
    now also an order at the broker, would fill within minutes); so near its
    target that less than the desk's minimum move is left; or so far its own
    way from the ``close`` it was sized on that the move the desk called has
    largely happened without it.
    """
    from tradingagents.sizing import through_stop

    if not price:
        return "no live price"
    d = 1 if side == "long" else -1
    if through_stop(d, price, stop):
        return f"the price {price:,.2f} is already through its stop {stop:,.2f}"
    if stop and min_stop_pct > 0:
        room = d * (price / stop - 1) * 100
        if room < min_stop_pct:
            return (f"its stop {stop:,.2f} is only {room:.1f}% from the price {price:,.2f}; "
                    f"a new position needs {min_stop_pct:g}%")
    if target and min_target_pct > 0:
        move = d * (target / price - 1) * 100
        if move <= min_target_pct:
            return f"only {move:+.1f}% is left to its target {target:,.2f} from {price:,.2f}"
    if close and max_run_pct > 0:
        run = d * (price / close - 1) * 100
        if run > max_run_pct:
            return (f"the price {price:,.2f} has already run {run:+.1f}% its way from the {close:,.2f} "
                    f"close it was sized on; the limit is {max_run_pct:g}%")
    return None


def _number(text) -> float | None:
    try:
        return float(text) if text not in (None, "") else None
    except (TypeError, ValueError):
        return None


def newest_levels(row: dict, decisions: dict, default_stop_pct: float) -> tuple[str | None, float | None, float | None, str | None]:
    """A book or bench row read against the ticker's newest decision: why it is not opened, its stop, target and date.

    The book is built once, before the open; the sweep goes on analysing. A
    name the desk has since turned on (a long now rated Hold or lower, a short
    now Hold or higher) is not opened: the next morning's review would only
    close it. One it still backs is judged, and opened, on the newest
    decision's stop and target, which are the ones the Watcher will hold it to.
    """
    from tradingagents.web import data

    stop, target, decided = _number(row.get("stop")), _number(row.get("target")), row.get("decided") or None
    ref = decisions.get(row["ticker"].upper())
    if ref is None:
        return None, stop, target, decided
    summary = data.decision_summary(data.load_decision(ref), None, default_stop_pct)
    if summary["side"] != row["side"]:
        return (f"its newest decision ({ref.date}, rated {summary['rating'] or 'nothing'}) no longer backs a {row['side']}",
                stop, target, decided)
    return None, summary["stop"] or stop, summary["target"] or target, ref.date


ET = ZoneInfo("America/New_York")


def _now() -> datetime:
    return datetime.now(ET)


def too_early(config: dict, now: datetime | None = None) -> str | None:
    """Why no order the desk chooses to make may go out yet today, or None once ``worker_trade_from`` (ET) has come.

    A stop is not such an order (see ``refusal``): it acts from the open.
    """
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
    # The window keeps the desk out of the dear first minutes when it chooses
    # to trade: the book, a review exit, a target, a replacement. A stop is not
    # a choice. A position that opens through its stop has already lost what
    # the stop allowed, and each minute it is held for the clock can add to
    # that: on 2026-11-20 APH, GEV and KKR waited the hour and went out 1.2% to
    # 1.4% below their stops, against the 0.3% a side the first minutes cost.
    early = None if action.at_stop else too_early(config, now)
    if early:
        return early
    if action.symbol in pending:
        return f"an earlier order for {action.symbol} has not filled yet"
    return None
