"""What the worker remembers between passes: news already read, orders already placed."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
OPENS = ("open", "replace")


def held_side(kind: str, side: str) -> str:
    """The side of the position an order belongs to: a sell that opens is a short, a buy that closes was one."""
    if kind in (*OPENS, "add"):
        return "long" if side == "buy" else "short"
    return "long" if side == "sell" else "short"


def entry_day(at: str | datetime) -> str:
    """The trading date (Eastern) of an order's time."""
    when = datetime.fromisoformat(at) if isinstance(at, str) else at
    return (when.astimezone(ET) if when.tzinfo else when).strftime("%Y-%m-%d")

# Article ids remembered per symbol; enough to cover a day of passes.
_SEEN_LIMIT = 300


@dataclass
class WorkerState:
    path: Path
    seen: dict[str, list[str]] = field(default_factory=dict)        # symbol -> article ids
    last_check: dict[str, str] = field(default_factory=dict)        # symbol -> ISO time
    orders: list[dict] = field(default_factory=list)                # every order placed
    book: dict[str, dict] = field(default_factory=dict)             # ticker -> the book position the worker opened
    days: dict[str, dict] = field(default_factory=dict)             # date -> {"analysis", "book_path", "placed", "replaced"}
    positions: dict[str, dict] = field(default_factory=dict)        # symbol -> {"id", "entered", "side"} of the open position

    @classmethod
    def load(cls, path: str | Path) -> WorkerState:
        path = Path(path)
        if not path.is_file():
            return cls(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(path, data.get("seen", {}), data.get("last_check", {}), data.get("orders", []),
                   data.get("book", {}), data.get("days", {}), data.get("positions", {}))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {"seen": self.seen, "last_check": self.last_check, "orders": self.orders,
                "book": self.book, "days": self.days, "positions": self.positions}
        self.path.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")

    def is_new(self, symbol: str, article_id: str) -> bool:
        return article_id not in self.seen.get(symbol, [])

    def mark_seen(self, symbol: str, article_ids: list[str]) -> None:
        ids = self.seen.setdefault(symbol, [])
        ids.extend(i for i in article_ids if i not in ids)
        del ids[:-_SEEN_LIMIT]

    def day(self, date: str) -> dict:
        return self.days.setdefault(date, {"analysis": "", "book_path": "", "placed": [], "replaced": {}})

    def orders_on(self, day: str) -> list[dict]:
        return [o for o in self.orders if str(o.get("at", "")).startswith(day)]

    def record_order(self, symbol: str, kind: str, side: str, qty: int, order_id: str, at: datetime,
                     reason: str = "", entry_price: float | None = None, stop: float | None = None,
                     decided: str | None = None) -> None:
        """Record an order sent. An exit or a trim carries the position's average entry price
        at the time, so its result can be stated without reconstructing the entry later. An
        order that opens a position carries the stop it was opened with, so its result can
        be stated in R whatever the desk decides about the ticker afterwards, and the
        analysis date of the decision it came from (``decided``), so the lists can say
        how old the desk's call was when the position was opened."""
        record = {"symbol": symbol, "kind": kind, "side": side, "qty": qty,
                  "order_id": order_id, "at": at.isoformat(), "reason": reason}
        record.update(self._tag(symbol, kind, side, at))
        if entry_price:
            record["entry_price"] = float(entry_price)
        if stop:
            record["stop"] = float(stop)
        if decided:
            record["decided"] = str(decided)
        self.orders.append(record)

    # --- positions ---------------------------------------------------------
    #
    # A ticker is not a position: the desk can be short a name in the morning
    # and long it by the afternoon, or hold it across days. Each position gets
    # an ID and its entry date when it is opened, and every later order on it
    # (add, trim, exit) carries both, so the lists can tell two positions in
    # one ticker apart and a carried-over position from one opened today.

    def _new_id(self, symbol: str, side: str, entered: str) -> str:
        """``FDX-20261113-S``; a second position in the name, on the side, on the day, ends ``-2``."""
        base = f"{symbol.upper()}-{entered.replace('-', '')}-{'L' if side == 'long' else 'S'}"
        taken = {o.get("position_id") for o in self.orders if o["kind"] in OPENS}
        n, pid = 1, base
        while pid in taken:
            n += 1
            pid = f"{base}-{n}"
        return pid

    def position_before(self, symbol: str, side: str, at: str | None = None) -> dict | None:
        """The position on ``side`` that the recorded orders show open in ``symbol`` (before ``at``), or None.

        For orders from before positions were tagged: the last order that
        opened the name, when it opened this side and no exit followed it.
        A position the worker has no opening order for has no known entry.
        """
        for o in reversed(self.orders):
            if o["symbol"].upper() != symbol.upper() or (at and o["at"] >= at):
                continue
            if o["kind"] == "exit":
                return None
            if o["kind"] in OPENS:
                if held_side(o["kind"], o["side"]) != side:
                    return None
                entered = o.get("entered") or entry_day(o["at"])
                return {"id": o.get("position_id") or f"{symbol.upper()}-{entered.replace('-', '')}-{'L' if side == 'long' else 'S'}",
                        "entered": entered, "side": side}
        return None

    def _tag(self, symbol: str, kind: str, side: str, at: datetime) -> dict:
        """The position this order belongs to, as the fields its record carries; keeps the open positions current."""
        key, held = symbol.upper(), held_side(kind, side)
        if kind in OPENS:
            entered = entry_day(at)
            position = {"id": self._new_id(key, held, entered), "entered": entered, "side": held}
        else:
            position = self.positions.get(key)
            if not position or position.get("side") != held:
                position = self.position_before(key, held)
            if not position:
                # Held, but not through an order of the worker's: no entry on record.
                position = {"id": f"{key}-unrecorded-{'L' if held == 'long' else 'S'}", "entered": None, "side": held}
        if kind == "exit":
            self.positions.pop(key, None)
        else:
            self.positions[key] = position
        return {"position_id": position["id"], "entered": position["entered"]}
