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
from functools import partial
from pathlib import Path
from typing import Literal

from pydantic import Field

from tradingagents.agents.schemas import Envelope, envelope_lines
from tradingagents.agents.structured import NO_EXTERNAL_TOOLS, bind_structured, decide
from tradingagents.book import latest_bench
from tradingagents.dataflows.vendors.alpaca import latest_price, trading
from tradingagents.feedback.store import keep_desk_verdict
from tradingagents.lessons import desk_lessons
from tradingagents.ranking import age_days, balances_book, levels_pct, score_call
from tradingagents.worker.duty import Action, Finding, WorkerContext

logger = logging.getLogger(__name__)


class ReplacementChoice(Envelope):
    ticker: str = Field(description="The candidate to open, exactly as listed; or NONE to leave the slot empty.")
    reason: str = Field(description="One or two sentences: why this one over the others, for the position that left.")

    def call(self) -> str:
        return self.ticker.strip().upper()[:32]

    def thesis(self) -> str:
        return self.reason


def candidates(bench: list[dict], held: set[str], used: set[str], min_target_pct: float,
               net_value: float = 0.0, today: str | None = None, default_stop_pct: float = 2.0) -> list[dict]:
    """Bench rows still worth opening now, best first.

    Not held, not used, the target still over the floor from the live price and
    the stop not yet reached; then ranked by the score at live prices (see
    ranking.py), with the balance bonus against the book's current net.
    """
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
        reward, risk = levels_pct(side, price, target, stop, default_stop_pct)
        s = score_call(reward, risk, row.get("rating", ""), row.get("risk_level"),
                       age_days(row.get("decided"), today), balances_book(side, net_value), min_risk_pct=default_stop_pct)
        out.append({**row, "live_price": price, "live_move_pct": round(move, 2), "live_score": s.score, "live_parts": s.parts()})
    out.sort(key=lambda o: (-o["live_score"], o["ticker"]))
    return out


def _prompt(departed: dict, options: list[dict]) -> str:
    lines = [f"- {o['ticker']}: score {o['live_score']:.2f} — {o['side']} ({o['rating']}, {o['risk_level'] or 'risk n/a'}), "
             f"price {o['live_price']:,.2f}, target {float(o['target']):,.2f} ({o['live_move_pct']:+.1f}% to go), "
             f"stop {float(o['stop']):,.2f}; {o['live_parts']}"
             for o in options]
    return (
        "You are the Watcher on a long/short trading desk. A book position has just closed and its "
        "capital is free; choose its replacement from the desk's bench, the next-best calls the desk "
        "made this morning.\n\n"
        f"The position that left: {departed['side']} {departed['ticker']}, {departed['value']:,.0f} of equity, "
        f"closed because {departed['why']}.\n\n"
        "Candidates, best score first (all still clear the desk's rules at today's prices). The score is "
        "reward to risk, times conviction, stance, freshness and a bonus for the side that balances the "
        "book; it already prefers the larger move, the wider stop and the balancing side, so the first "
        "is the default:\n" + "\n".join(lines) + "\n\n"
        "Take the first unless you see a reason in the numbers to prefer another (say which and why), "
        "or NONE if none is worth opening. The side of the departed position does not matter: the desk "
        "takes the most attractive call, long or short, and an uneven book is fine. " + desk_lessons("Watcher", before="\n\n", after="\n\n") + NO_EXTERNAL_TOOLS
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
            entry = state.book.pop(ticker)
            departed = entry
            closed_by = next((o for o in reversed(state.orders) if o["symbol"] == ticker and o["kind"] == "exit"), None)
            departed = {**departed, "ticker": ticker, "why": (closed_by or {}).get("reason", "it is no longer held")}
            net = sum(h.market_value for h in context.account.holdings if h.asset_class == "us_equity")
            options = candidates(bench, held | set(state.book), used, float(context.config["min_target_pct"]),
                                 net, today, float(context.config.get("default_stop_pct") or 2.0))
            if not options:
                findings.append(Finding(self.name, ticker, "notable", f"{ticker} left the book; no replacement on the bench",
                                        f"bench {bench_path.name if bench_path else 'missing'}; none still clears the rules"))
                continue
            try:
                choice, verdict = decide(bind_structured(context.llm, ReplacementChoice, "Watcher"), _prompt(departed, options),
                                   "Watcher", catch=ValueError)
            except Exception as exc:
                # No answer is not a confirmation: nothing is opened on the score
                # alone. The name goes back in the book, so the next pass asks again.
                state.book[ticker] = entry
                logger.warning("No replacement for %s yet, the model gave no answer (%s); asking again next pass", ticker, exc)
                findings.append(Finding(self.name, ticker, "info", f"{ticker} left the book; its replacement waits for the model",
                                        f"{type(exc).__name__}: {exc}"))
                continue
            pick = None
            if choice and choice.ticker.upper() != "NONE":
                pick = next((o for o in options if o["ticker"].upper() == choice.ticker.upper()), None)
            asked = {"departed": {k: departed.get(k) for k in ("ticker", "side", "value", "why")},
                     "candidates": [{k: o.get(k) for k in ("ticker", "side", "rating", "risk_level", "live_price", "target", "stop",
                                                           "live_move_pct", "live_score")} for o in options]}
            keep = partial(keep_desk_verdict, context.config, verdict, "replacement", today, ticker, context=asked)
            if pick is None:
                keep(action="none")
                findings.append(Finding(self.name, ticker, "notable", f"{ticker} left the book; the Watcher opens nothing",
                                        choice.reason if choice else "the model's answer did not meet its contract"))
                continue
            cap_value = context.account.equity * float(context.config["worker_position_cap_pct"]) / 100
            value = min(float(departed.get("value") or 0) or cap_value, cap_value)
            qty = math.floor(value / pick["live_price"])
            if qty < 1:
                keep(action=f"none: {pick['ticker']} too dear")
                findings.append(Finding(self.name, ticker, "notable", f"{ticker} left the book; {pick['ticker']} is too dear for its capital",
                                        f"{value:,.0f} buys no whole share at {pick['live_price']:,.2f}"))
                continue
            keep(action=f"open {pick['side']} {pick['ticker']} {qty}")
            reason = (f"replaces {ticker}: {choice.reason} "
                      f"(score {pick['live_score']:.2f}, {pick['live_move_pct']:+.1f}% to target)")
            used.add(pick["ticker"].upper())
            day["replaced"][ticker] = pick["ticker"]
            findings.append(Finding(
                self.name, pick["ticker"], "urgent",
                f"{pick['side']} {pick['ticker']} replaces {ticker}",
                reason + "\n".join(envelope_lines(choice)) + "\n\nCandidates by score: " + ", ".join(f"{o['ticker']} {o['live_score']:.2f}" for o in options),
                Action(pick["ticker"], "replace", "buy" if pick["side"] == "long" else "sell", qty, reason,
                       value=qty * pick["live_price"]),
            ))
        return findings
