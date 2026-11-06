"""The Watcher's stop guard: a position through its stop is closed, news or no news.

The desk's stops are levels in a file, not orders at the broker, so nothing
would act on them by itself. Each pass checks every watched position's price
against the stop that stands for it: the Risk Manager's or the Trader's from
its decision, else the 2% default measured from the entry price (the sizing
rules, so the page and the Watcher agree). No model is asked: a breach is
arithmetic, and the exit goes out on the same pass.
"""

from __future__ import annotations

import logging
from pathlib import Path

from tradingagents.dataflows.vendors.alpaca.trading import Holding
from tradingagents.sizing import Call, stop_for
from tradingagents.web import data
from tradingagents.worker.duty import Action, Finding, WorkerContext

logger = logging.getLogger(__name__)


def standing_stop(h: Holding, decisions: dict[str, data.DecisionRef], default_stop_pct: float) -> tuple[float, str]:
    """The stop for a held position, and where it came from."""
    ref = decisions.get(h.symbol.upper())
    rating = "Buy" if h.qty > 0 else "Sell"
    if ref is not None:
        summary = data.decision_summary(data.load_decision(ref), h.avg_entry_price, default_stop_pct)
        if summary["stop"] and summary["side"] in ("long", "short"):
            return summary["stop"], summary["stop_from"]
    return stop_for(Call(h.symbol, rating, None), h.avg_entry_price, default_stop_pct)


def breached(h: Holding, stop: float) -> bool:
    """Whether the position's price is through its stop on the losing side."""
    price = abs(h.market_value) / abs(h.qty) if h.qty else None
    if not price:
        return False
    return price <= stop if h.qty > 0 else price >= stop


class StopDuty:
    name = "stops"

    def run(self, context: WorkerContext) -> list[Finding]:
        decisions = data.find_decisions(Path(context.config["results_dir"]))
        default_pct = float(context.config.get("default_stop_pct") or 0)
        findings = []
        for h in context.watched():
            if default_pct <= 0 and h.symbol.upper() not in decisions:
                continue
            stop, source = standing_stop(h, decisions, default_pct)
            if not breached(h, stop):
                continue
            price = abs(h.market_value) / abs(h.qty)
            loss = (price / h.avg_entry_price - 1) * 100 * (1 if h.qty > 0 else -1) if h.avg_entry_price else 0.0
            reason = f"price {price:,.2f} is through the stop {stop:,.2f} ({source}); {loss:+.1f}% on the position"
            findings.append(Finding(
                self.name, h.symbol, "urgent",
                f"{h.side} {h.symbol}: stop hit at {stop:,.2f} ({source})",
                reason,
                Action(h.symbol, "exit", "sell" if h.qty > 0 else "buy", abs(int(h.qty)), reason),
            ))
        return findings
