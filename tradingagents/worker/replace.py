"""The Watcher's replacement call: a book position that sells out is replaced from the bench.

The daily job leaves a bench of the next-best calls after the book, long or
short together. When a position the worker opened is no longer held (its
stop, its target, or news closed it) the Watcher picks a replacement from
the bench: the model chooses among the candidates that still clear the
desk's rules at today's prices, and the freed capital opens it on the same
pass. A candidate is used once; a slot is refilled once.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from tradingagents.agents.structured import NO_EXTERNAL_TOOLS, bind_structured, invoke_structured
from tradingagents.book import latest_bench
from tradingagents.dataflows.vendors.alpaca import latest_price, trading
from tradingagents.worker.duty import Action, Finding, WorkerContext

logger = logging.getLogger(__name__)


class ReplacementChoice(BaseModel):
    ticker: str = Field(description="The candidate to open, exactly as listed; or NONE to leave the slot empty.")
    reason: str = Field(description="One or two sentences: why this one over the others, for the position that left.")


def candidates(bench: list[dict], held: set[str], used: set[str], min_target_pct: float) -> list[dict]:
    """Bench rows still worth opening now: not held, not used, target still over the floor from the live price."""
    out = []
    for row in bench:
        t = row["ticker"].upper()
        if t in held or t in used:
            continue
        try:
            target = float(row["target"])
            stop = float(row["stop"]) if row.get("stop") else None
        except (TypeError, ValueError):
            continue
        price = latest_price(t)
        if not price:
            continue
        side = 1 if row["side"] == "long" else -1
        move = side * (target / price - 1) * 100
        if move <= min_target_pct:
            continue
        if stop and (stop - price) * side >= 0:
            continue        # already through its stop
        out.append({**row, "live_price": price, "live_move_pct": round(move, 2)})
    return out


def _prompt(departed: dict, options: list[dict]) -> str:
    lines = [f"- {o['ticker']}: {o['side']} ({o['rating']}, {o['risk_level'] or 'risk n/a'}), price {o['live_price']:,.2f}, "
             f"target {float(o['target']):,.2f} ({o['live_move_pct']:+.1f}% to go), stop {float(o['stop']):,.2f}"
             for o in options]
    return (
        "You are the Watcher on a long/short trading desk. A book position has just closed and its "
        "capital is free; choose its replacement from the desk's bench, the next-best calls the desk "
        "made this morning.\n\n"
        f"The position that left: {departed['side']} {departed['ticker']}, {departed['value']:,.0f} of equity, "
        f"closed because {departed['why']}.\n\n"
        "Candidates (all still clear the desk's rules at today's prices):\n" + "\n".join(lines) + "\n\n"
        "Prefer the larger move still to go, a stop not too near the price, and a side that keeps the "
        "book balanced after this one left. Answer with one ticker from the list, or NONE if none is "
        "worth opening. " + NO_EXTERNAL_TOOLS
    )


class ReplacementDuty:
    name = "replace"

    def run(self, context: WorkerContext) -> list[Finding]:
        state = context.state
        if not state.book:
            return []
        held = {h.symbol.upper() for h in context.account.holdings}
        pending = set(trading.open_orders().values())
        gone = [t for t in list(state.book) if t not in held and t not in pending]
        if not gone:
            return []
        today = context.now.strftime("%Y-%m-%d")
        day = state.day(today)
        bench_path, bench = latest_bench(Path(context.config["results_dir"]))
        used = {r.upper() for r in day["replaced"].values()} | set(state.book)
        findings = []
        for ticker in gone:
            departed = state.book.pop(ticker)
            closed_by = next((o for o in reversed(state.orders) if o["symbol"] == ticker and o["kind"] == "exit"), None)
            departed = {**departed, "ticker": ticker, "why": (closed_by or {}).get("reason", "it is no longer held")}
            options = candidates(bench, held | set(state.book), used, float(context.config["min_target_pct"]))
            if not options:
                findings.append(Finding(self.name, ticker, "notable", f"{ticker} left the book; no replacement on the bench",
                                        f"bench {bench_path.name if bench_path else 'missing'}; none still clears the rules"))
                continue
            choice = invoke_structured(bind_structured(context.llm, ReplacementChoice, "Watcher"), _prompt(departed, options), "Watcher")
            pick = None
            if choice and choice.ticker.upper() != "NONE":
                pick = next((o for o in options if o["ticker"].upper() == choice.ticker.upper()), None)
            if pick is None and choice is None:
                pick = options[0]       # the model did not answer: the bench's own order stands
            if pick is None:
                findings.append(Finding(self.name, ticker, "notable", f"{ticker} left the book; the Watcher opens nothing",
                                        choice.reason if choice else "no answer"))
                continue
            cap_value = context.account.equity * float(context.config["worker_position_cap_pct"]) / 100
            value = min(float(departed.get("value") or 0) or cap_value, cap_value)
            qty = math.floor(value / pick["live_price"])
            if qty < 1:
                findings.append(Finding(self.name, ticker, "notable", f"{ticker} left the book; {pick['ticker']} is too dear for its capital",
                                        f"{value:,.0f} buys no whole share at {pick['live_price']:,.2f}"))
                continue
            reason = (f"replaces {ticker}: {choice.reason if choice else 'first on the bench'} "
                      f"({pick['live_move_pct']:+.1f}% to target)")
            used.add(pick["ticker"].upper())
            day["replaced"][ticker] = pick["ticker"]
            findings.append(Finding(
                self.name, pick["ticker"], "urgent",
                f"{pick['side']} {pick['ticker']} replaces {ticker}",
                reason + f"\n\nCandidates: " + ", ".join(o["ticker"] for o in options),
                Action(pick["ticker"], "replace", "buy" if pick["side"] == "long" else "sell", qty, reason,
                       value=qty * pick["live_price"]),
            ))
        return findings
