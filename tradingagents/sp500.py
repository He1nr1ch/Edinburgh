"""
sp500.py — S&P 500 constituent universe → watchlists/sp500.txt

Adapted from Dublin's sp500.py, which keeps the list in its database; here the
list is the watchlist file the sweeps read. Fetches the current S&P 500
component list from Wikipedia and writes it as one comma-separated line, in
the dot spelling the rest of the project uses (BRK.B), so index membership
changes (adds/drops) flow through without hand edits.

The refresh fails closed: if the fetch errors out or returns suspiciously few
rows, the existing file is left untouched so a sweep never loses its universe
to a transient Wikipedia problem.

Standalone:
    tradingagents sp500     # force a refresh now, regardless of age
"""

from __future__ import annotations

from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# The index holds ~503 symbols (multiple share classes). A fetch returning
# fewer than this is a parse/outage artifact, not a real membership change —
# refuse to overwrite the file with it.
MIN_EXPECTED_SYMBOLS = 450

MAX_AGE_DAYS = 30

WATCHLIST = Path(__file__).resolve().parents[1] / "watchlists" / "sp500.txt"


def fetch_sp500_from_wikipedia() -> list[tuple[str, str, str]]:
    """
    Return the current constituents as (symbol, company_name, gics_sector)
    tuples. Symbols come back in dot notation (BRK.B), matching Alpaca.
    """
    response = requests.get(
        WIKIPEDIA_URL,
        headers={"User-Agent": "Edinburgh-trading-desk/1.0 (S&P 500 constituent refresh)"},
        timeout=30,
    )
    response.raise_for_status()

    constituents = None
    for table in pd.read_html(StringIO(response.text)):
        if "Symbol" in table.columns and "Security" in table.columns:
            constituents = table
            break
    if constituents is None:
        raise ValueError("No constituents table with Symbol/Security columns found on Wikipedia page")

    rows = []
    for _, row in constituents.iterrows():
        symbol = str(row["Symbol"]).strip().upper().replace("-", ".")
        if not symbol or symbol == "NAN":
            continue
        rows.append((
            symbol,
            str(row["Security"]).strip(),
            str(row.get("GICS Sector", "")).strip(),
        ))
    return rows


def refresh_sp500_watchlist(path: Path = WATCHLIST) -> int:
    """
    Fetch constituents and replace the watchlist file. The file is written
    whole, so a reader never sees a half-written universe. Raises (leaving
    the existing file intact) on fetch failure or an implausibly small result.
    """
    rows = fetch_sp500_from_wikipedia()
    if len(rows) < MIN_EXPECTED_SYMBOLS:
        raise ValueError(
            f"Fetched only {len(rows)} symbols (expected ≥ {MIN_EXPECTED_SYMBOLS}) — "
            f"refusing to overwrite {path.name}"
        )
    symbols = sorted({symbol for symbol, _, _ in rows})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(",".join(symbols) + "\n", encoding="utf-8")
    print(f"[sp500] Refreshed {path.name} with {len(symbols)} constituents")
    return len(symbols)


def _file_status(path: Path) -> tuple[int, datetime | None]:
    """Return (symbol count, last modified) for the watchlist file."""
    if not path.is_file():
        return 0, None
    symbols = [s for s in path.read_text(encoding="utf-8").replace("\n", "").split(",") if s.strip()]
    return len(symbols), datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def refresh_if_stale(max_age_days: int = MAX_AGE_DAYS, path: Path = WATCHLIST) -> None:
    """
    Refresh the file if it is missing or older than max_age_days; otherwise
    no-op. A failed refresh is fatal only when there is no file — with a
    populated (if stale) universe a sweep can still run, so just warn.
    """
    count, last_updated = _file_status(path)
    if count and last_updated:
        age_days = (datetime.now(timezone.utc) - last_updated).days
        if age_days < max_age_days:
            print(f"[sp500] {count} symbols, refreshed {age_days} day(s) ago — still fresh")
            return
        print(f"[sp500] Universe is {age_days} day(s) old — refreshing …")

    try:
        refresh_sp500_watchlist(path)
    except Exception as e:
        if count:
            print(f"[sp500] Refresh failed ({e}) — keeping existing {count} symbols")
        else:
            raise


def get_sp500_symbols(path: Path = WATCHLIST) -> list[str]:
    """
    Return the universe, sorted. If there is no file yet (first run),
    populate it first.
    """
    count, _ = _file_status(path)
    if not count:
        print(f"[sp500] {path.name} is missing — fetching constituents …")
        refresh_sp500_watchlist(path)
    return sorted(s.strip() for s in path.read_text(encoding="utf-8").replace("\n", "").split(",") if s.strip())


if __name__ == "__main__":
    refresh_sp500_watchlist()
