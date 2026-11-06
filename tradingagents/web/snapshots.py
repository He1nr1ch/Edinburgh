"""A record of the account each day: "on this day, I held these positions".

Alpaca reports only what is held now, so the worker writes a snapshot on every
pass to ``<results_dir>/positions/<date>.json``; the last write of a day is
its close-of-day picture. Each row carries the decision that stood for the
ticker on that day (rating, stop, target), frozen, so the page for a past day
does not change when a later run decides differently.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from tradingagents.web import data


def folder(results_dir: Path) -> Path:
    return Path(results_dir) / "positions"


def write_snapshot(results_dir: Path, account, default_stop_pct: float, now: datetime | None = None) -> Path:
    """Write today's snapshot (overwriting an earlier one today); returns its path."""
    now = now or datetime.now().astimezone()
    day = now.strftime("%Y-%m-%d")
    decisions = data.decisions_as_of(results_dir, day)
    rows = data.positions_view(account.holdings, decisions, default_stop_pct)
    record = {
        "date": day, "as_of": now.isoformat(timespec="seconds"), "paper": account.paper,
        "equity": account.equity, "cash": account.cash, "positions": rows,
    }
    path = folder(results_dir) / f"{day}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def list_days(results_dir: Path) -> list[dict]:
    """Every recorded day, newest first: date, when it was written, equity, position count."""
    days = []
    for path in sorted(folder(results_dir).glob("????-??-??.json"), reverse=True):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        days.append({"date": path.stem, "as_of": record.get("as_of", ""), "equity": record.get("equity"),
                     "positions": len(record.get("positions", [])), "paper": record.get("paper")})
    return days


def read_day(results_dir: Path, day: str) -> dict | None:
    path = folder(results_dir) / f"{day}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
