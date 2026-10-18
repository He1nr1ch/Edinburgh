"""Reading the typed calls out of the agents' rendered markdown.

The Trader and Risk Manager render their structured output to labelled lines
(``**Stop Loss**: 209.0``); the free-text fallback is asked for the same lines.
These readers take the number after a label, or None when there is none.
"""

from __future__ import annotations

import re
import unicodedata

from tradingagents.agents.rating import extract_rating

_NUMBER = r"[-+]?\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)"


def labelled_number(text: str, label: str) -> float | None:
    """The number following ``**label**:`` (bold optional), or None."""
    if not text:
        return None
    norm = unicodedata.normalize("NFKC", text)
    pattern = re.compile(r"[*_]*" + re.escape(label) + r"[*_]*\s*[:\-‐-―]\s*[*_]*" + _NUMBER, re.IGNORECASE)
    m = pattern.search(norm)
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return None


_PROPOSAL_RE = re.compile(r"FINAL TRANSACTION PROPOSAL:\s*[*_]*\s*(BUY|SELL|HOLD)", re.IGNORECASE)
_ACTION_RE = re.compile(r"[*_]*Action[*_]*\s*:\s*[*_]*\s*(Buy|Sell|Hold)\b", re.IGNORECASE)


def trader_action(text: str) -> str | None:
    """Buy / Sell / Hold from the Trader's proposal, or None."""
    if not text:
        return None
    m = _PROPOSAL_RE.search(text) or _ACTION_RE.search(text)
    return m.group(1).capitalize() if m else None


def trader_fields(text: str) -> dict:
    return {
        "trader_action": trader_action(text),
        "trader_entry_price": labelled_number(text, "Entry Price"),
        "trader_stop_loss": labelled_number(text, "Stop Loss"),
    }


def risk_fields(text: str) -> dict:
    return {
        "rm_max_position_pct": labelled_number(text, "Maximum Position"),
        "rm_stop_loss": labelled_number(text, "Stop Loss"),
    }


def research_recommendation(text: str) -> str | None:
    """The Research Manager's recommendation, which it labels like a rating."""
    if not text:
        return None
    norm = unicodedata.normalize("NFKC", text)
    m = re.search(r"[*_]*Recommendation[*_]*\s*[:\-‐-―]\s*[*_]*\s*(\w+)", norm, re.IGNORECASE)
    if m and m.group(1).capitalize() in ("Buy", "Overweight", "Hold", "Underweight", "Sell"):
        return m.group(1).capitalize()
    return extract_rating(text)
