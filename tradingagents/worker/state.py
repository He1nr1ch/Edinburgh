"""What the worker remembers between passes: news already read, orders already placed."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Article ids remembered per symbol; enough to cover a day of passes.
_SEEN_LIMIT = 300


@dataclass
class WorkerState:
    path: Path
    seen: dict[str, list[str]] = field(default_factory=dict)        # symbol -> article ids
    last_check: dict[str, str] = field(default_factory=dict)        # symbol -> ISO time
    orders: list[dict] = field(default_factory=list)                # every order placed

    @classmethod
    def load(cls, path: str | Path) -> WorkerState:
        path = Path(path)
        if not path.is_file():
            return cls(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(path, data.get("seen", {}), data.get("last_check", {}), data.get("orders", []))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {"seen": self.seen, "last_check": self.last_check, "orders": self.orders}
        self.path.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")

    def is_new(self, symbol: str, article_id: str) -> bool:
        return article_id not in self.seen.get(symbol, [])

    def mark_seen(self, symbol: str, article_ids: list[str]) -> None:
        ids = self.seen.setdefault(symbol, [])
        ids.extend(i for i in article_ids if i not in ids)
        del ids[:-_SEEN_LIMIT]

    def orders_on(self, day: str) -> list[dict]:
        return [o for o in self.orders if str(o.get("at", "")).startswith(day)]

    def record_order(self, symbol: str, kind: str, side: str, qty: int, order_id: str, at: datetime) -> None:
        self.orders.append({"symbol": symbol, "kind": kind, "side": side, "qty": qty,
                            "order_id": order_id, "at": at.isoformat()})
