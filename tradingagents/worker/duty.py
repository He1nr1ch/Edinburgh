"""What a duty of the daily worker sees, and what it hands back.

A duty looks at the account during the day and reports findings; a finding may
carry an action. The runner applies the guardrails to every action, whichever
duty proposed it, so a new duty adds a check, not a new way to trade.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from tradingagents.dataflows.vendors.alpaca.trading import Account, Holding


@dataclass(frozen=True)
class Action:
    """An order a duty wants placed. ``kind`` is exit, trim or add."""
    symbol: str
    kind: str
    side: str           # buy or sell
    qty: int
    reason: str


@dataclass
class Finding:
    duty: str
    symbol: str
    severity: str       # info, notable, urgent
    headline: str
    detail: str
    action: Action | None = None
    outcome: str = ""   # what the runner did with the action


@dataclass
class WorkerContext:
    account: Account
    now: datetime
    config: dict
    state: Any                          # WorkerState
    llm: Any = None
    findings: list[Finding] = field(default_factory=list)

    def watched(self) -> list[Holding]:
        """The equity positions the worker looks after (options and crypto are left alone)."""
        allow = self.config.get("worker_symbols")
        names = {s.strip().upper() for s in allow.split(",")} if allow else None
        return [h for h in self.account.holdings
                if h.asset_class == "us_equity" and (names is None or h.symbol.upper() in names)]


class Duty(Protocol):
    name: str

    def run(self, context: WorkerContext) -> list[Finding]: ...
