"""The Watcher's market guard on what is held: a market that swings takes the side against it out.

The market guard (market_guard.py) stops the desk opening anything against a
market that is running. This is the other half: the desk is married to no
position, and when SPY has swung, past ``market_swing_day_pct`` on the day or
``market_swing_period_pct`` over ``market_guard_days`` sessions, every position
held against it is closed on the pass that sees it: the shorts in a swing up,
the longs in a swing down, whether that takes a profit or cuts a loss. No
model is asked: it is arithmetic on SPY.

It is an order the desk chooses to make, so it waits for the trading window
(``worker_trade_from``); a position through its own stop before then still
goes at the open. What it closes the replacement guard refills from the
bench, from the side the market is not running against.
"""

from __future__ import annotations

from tradingagents.market_guard import read_tape
from tradingagents.worker.duty import Action, Finding, WorkerContext


class MarketDuty:
    name = "market"

    def run(self, context: WorkerContext) -> list[Finding]:
        tape = read_tape(context.config, context.now)
        if not tape or not tape.closes:
            return []
        # A position another guard closed a moment ago on this pass (its stop,
        # its target) is still in the pass's picture of the account.
        at = context.now.isoformat()
        leaving = {o["symbol"].upper() for o in context.state.orders if o["kind"] == "exit" and o["at"] == at}
        way = "up" if tape.swing > 0 else "down"
        findings = []
        for h in context.watched():
            if h.side != tape.closes or h.symbol.upper() in leaving or not h.qty:
                continue
            price = abs(h.market_value) / abs(h.qty)
            gain = (price / h.avg_entry_price - 1) * 100 * (1 if h.qty > 0 else -1) if h.avg_entry_price else 0.0
            reason = (f"the market has swung {way} ({tape.describe()}); the {h.side} is closed at {price:,.2f}, "
                      f"{'taking' if gain >= 0 else 'cutting'} {gain:+.1f}% on the position")
            findings.append(Finding(
                self.name, h.symbol, "urgent", f"{h.side} {h.symbol}: closed against a market swinging {way}", reason,
                Action(h.symbol, "exit", "sell" if h.qty > 0 else "buy", abs(int(h.qty)), reason),
            ))
        return findings
