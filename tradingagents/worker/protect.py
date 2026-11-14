"""The desk's stops, also held at Alpaca: a day stop order behind every watched position.

The stop guard (stops.py) only acts when a pass runs: every few minutes, and
not at all while the worker is down. So each position's stop is left at the
broker as well, as a stop order for the whole position, and Alpaca acts on it
whatever happens here. The guard stays: it closes what a stop order did not
(one Alpaca refused, a position already through its stop).

This is a reconciliation, run after every duty of a pass, not a step of
opening a position: it reads the account and the stops waiting at Alpaca and
brings the second in line with the first. A position with no stop gets one; a
stop for the wrong size (after a trim or an add) or the wrong price (the desk
decided again) is replaced; a stop whose position has gone is cancelled, since
it would open the position the other way round. A stop that Alpaca filled is
recorded as the exit it was.

The orders are day orders, placed from ``worker_trade_from`` on: nothing of
the desk's trades in the first minutes after the open, its stops included.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from tradingagents.dataflows.vendors.alpaca import trading
from tradingagents.web import data
from tradingagents.worker.duty import Action, Finding, WorkerContext
from tradingagents.worker.policy import refusal
from tradingagents.worker.stops import breached, standing_stop

logger = logging.getLogger(__name__)

NAME = "protect"


def order_price(stop: float) -> float:
    """The stop as Alpaca takes it: cents from a dollar up, hundredths of a cent below."""
    return round(stop, 2 if stop >= 1 else 4)


def _settle(context: WorkerContext, open_ids: set[str], held: dict[str, trading.Holding]) -> list[Finding]:
    """Stops the worker left that are no longer waiting: record the ones Alpaca filled, forget the rest."""
    state, findings = context.state, []
    for symbol, rec in list(state.stops.items()):
        if rec["order_id"] in open_ids:
            continue
        try:
            result = trading.order_result(rec["order_id"])
        except Exception as exc:
            findings.append(Finding(NAME, symbol, "notable", f"{symbol}: the stop left at Alpaca could not be read", str(exc)))
            continue
        filled = int(result["filled_qty"])
        if filled >= 1:
            h = held.get(symbol)
            still_held = h is not None and h.side == rec["held"]
            at = result["filled_at"] if isinstance(result["filled_at"], datetime) else context.now
            price = result["filled_avg_price"]
            reason = (f"the stop {rec['stop']:,.2f} held at Alpaca filled"
                      + (f" at {price:,.2f}" if price else ""))
            state.record_order(symbol, "trim" if still_held else "exit", rec["side"], filled, rec["order_id"],
                               at.astimezone(), reason, entry_price=rec.get("entry_price"))
            findings.append(Finding(NAME, symbol, "urgent",
                                    f"{rec['held']} {symbol}: stop hit at {rec['stop']:,.2f}, closed by Alpaca",
                                    f"{reason}; {filled} share(s), order {rec['order_id']}"))
        del state.stops[symbol]
    return findings


def protect(context: WorkerContext, market_open: bool, dry_run: bool = False) -> list[Finding]:
    """Bring the stop orders at Alpaca in line with the positions held; what changed, and what failed."""
    config = context.config
    if not config.get("worker_broker_stops", True):
        return []
    account = trading.get_account()
    if refusal(Action("", NAME, "", 0, ""), market_open=market_open, paper=account.paper, pending=set(), config=config):
        return []
    state = context.state
    open_now = trading.open_orders()
    waiting = trading.open_stops()
    watched = {h.symbol.upper(): h for h in replace(context, account=account).watched()}
    # A name with an order of the worker's own still unfilled is left for the
    # next look: its position is about to change, and Alpaca has its shares.
    mine = {o["order_id"] for o in state.orders}
    pending = {sym.upper() for order_id, sym in open_now.items() if order_id in mine}
    tracked = {rec["order_id"] for rec in state.stops.values()}

    findings = [] if dry_run else _settle(context, {s.id for s in waiting}, watched)

    # The stop that stands for each position; any other for it, and any of the
    # worker's own whose position has gone, is cancelled.
    standing: dict[str, trading.StopOrder] = {}
    for s in waiting:
        symbol, h = s.symbol.upper(), watched.get(s.symbol.upper())
        fits = h is not None and s.side == ("sell" if h.qty > 0 else "buy")
        if fits and symbol not in standing:
            standing[symbol] = s
        elif (fits or s.id in tracked) and not dry_run:
            try:
                trading.cancel_order(s.id)
                findings.append(Finding(NAME, symbol, "info", f"{symbol}: a stop nothing needs any more cancelled",
                                        f"{s.side} {s.qty} at {s.stop_price:,.2f}, order {s.id}"))
            except Exception as exc:
                findings.append(Finding(NAME, symbol, "notable", f"{symbol}: a stray stop could not be cancelled", str(exc)))

    decisions = data.find_decisions(Path(config["results_dir"]))
    default_pct = float(config.get("default_stop_pct") or 0)
    for symbol, h in watched.items():
        qty = abs(int(h.qty))
        if qty < 1 or symbol in pending or (default_pct <= 0 and symbol not in decisions):
            continue
        level, source = standing_stop(h, decisions, default_pct)
        stop, side = order_price(level), "sell" if h.qty > 0 else "buy"
        if breached(h, stop):
            continue        # the stop guard closes it; Alpaca would refuse a stop on the wrong side of the price
        have = standing.get(symbol)
        record = {"side": side, "held": h.side, "qty": qty, "stop": stop, "entry_price": h.avg_entry_price}
        if have is not None and have.qty == qty and abs(have.stop_price - stop) < 0.005:
            state.stops[symbol] = {**record, "order_id": have.id}
            continue
        what = (f"stop {stop:,.2f} ({source}) for {qty} share(s)" if have is None else
                f"stop moved from {have.stop_price:,.2f} for {have.qty} to {stop:,.2f} ({source}) for {qty} share(s)")
        if dry_run:
            findings.append(Finding(NAME, symbol, "info", f"{h.side} {symbol}: {what}", "dry run: not left at Alpaca"))
            continue
        try:
            order_id = (trading.submit_stop_order(h.symbol, qty, side, stop) if have is None
                        else trading.replace_stop_order(have.id, qty, stop))
        except Exception as exc:
            findings.append(Finding(NAME, symbol, "notable", f"{h.side} {symbol}: no stop could be left at Alpaca",
                                    f"{what}: {exc}"))
            continue
        state.stops[symbol] = {**record, "order_id": order_id}
        findings.append(Finding(NAME, symbol, "info", f"{h.side} {symbol}: {what}", f"held at Alpaca, order {order_id}"))
    if not dry_run:
        state.save()
    return findings
