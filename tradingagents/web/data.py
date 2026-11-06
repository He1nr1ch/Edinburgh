"""What the positions page shows: the account's open positions, each joined to the desk's latest decision.

Decisions are the saved run states (``full_states_log_<date>.json``) under the
results directory, from list runs (``backtest/<run>/<TICKER>/…``) and single
runs (``<TICKER>/…``) alike. A position's decision is the newest one for its
ticker. The stop and target follow the sizing rules, so the page shows the
same levels the orders carried.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from tradingagents.feedback.extract import labelled_number
from tradingagents.sizing import SIDE, Call, stop_for

# The agents' outputs, in the order the desk produced them, with the state
# key each is stored under.
AGENT_SECTIONS = (
    ("Market Analyst", "market_report"),
    ("News Analyst", "news_report"),
    ("Fundamentals Analyst", "fundamentals_report"),
    ("Bull Researcher", ("investment_debate_state", "bull_history")),
    ("Bear Researcher", ("investment_debate_state", "bear_history")),
    ("Research Manager", "investment_plan"),
    ("Trader", "trader_investment_plan"),
    ("Aggressive Risk Analyst", ("risk_debate_state", "aggressive_history")),
    ("Conservative Risk Analyst", ("risk_debate_state", "conservative_history")),
    ("Neutral Risk Analyst", ("risk_debate_state", "neutral_history")),
    ("Risk Manager", "risk_assessment"),
    ("Portfolio Manager", "final_trade_decision"),
)


@dataclass(frozen=True)
class DecisionRef:
    ticker: str
    date: str
    path: Path
    run: str            # the run folder's name, or "single"


def _field(text: str, name: str) -> str:
    m = re.search(rf"\*\*{name}\*\*:\s*(.+)", text or "")
    return m.group(1).strip() if m else ""


def all_decisions(results_dir: Path) -> dict[str, list[DecisionRef]]:
    """Every saved decision per ticker, newest first (by analysis date, then file time)."""
    found: dict[str, list[DecisionRef]] = {}
    patterns = ("backtest/*/*/TradingAgentsStrategy_logs/full_states_log_*.json",
                "*/TradingAgentsStrategy_logs/full_states_log_*.json")
    for pattern in patterns:
        for path in Path(results_dir).glob(pattern):
            if "backtest" in path.parts and pattern.startswith("*/"):
                continue        # the first pattern already took it
            date = path.name[len("full_states_log_"):-len(".json")]
            run = path.parts[-4] if "backtest" in path.parts else "single"
            found.setdefault(path.parts[-3], []).append(DecisionRef(path.parts[-3], date, path, run))
    for refs in found.values():
        refs.sort(key=lambda r: (r.date, r.path.stat().st_mtime), reverse=True)
    return found


def find_decisions(results_dir: Path) -> dict[str, DecisionRef]:
    """The newest saved decision per ticker."""
    return {ticker: refs[0] for ticker, refs in all_decisions(results_dir).items()}


def decisions_as_of(results_dir: Path, day: str) -> dict[str, DecisionRef]:
    """The decision that stood for each ticker on ``day``: the newest dated on or before it."""
    found = {}
    for ticker, refs in all_decisions(results_dir).items():
        standing = next((r for r in refs if r.date <= day), None)
        if standing is not None:
            found[ticker] = standing
    return found


def load_decision(ref: DecisionRef) -> dict:
    return json.loads(ref.path.read_text(encoding="utf-8"))


def decision_summary(state: dict, close: float | None, default_stop_pct: float) -> dict:
    """The decision's numbers: rating, risk, size, stop (with its source), target, expected move."""
    pm = state.get("final_trade_decision") or ""
    ra = state.get("risk_assessment") or ""
    tr = state.get("trader_investment_plan") or ""
    rm_stop = labelled_number(ra, "Stop Loss")
    tr_stop = labelled_number(tr, "Stop Loss")
    call = Call(
        ticker=state.get("company_of_interest") or "",
        rating=state.get("final_rating") or _field(pm, "Rating"),
        ceiling_pct=labelled_number(ra, "Maximum Position"),
        risk_level=state.get("final_risk_level") or state.get("risk_level"),
        target=labelled_number(pm, "Price Target"),
        stop=rm_stop or tr_stop,
        stop_from="Risk Manager" if rm_stop else ("Trader" if tr_stop else ""),
    )
    stop, stop_from = stop_for(call, close, default_stop_pct) if close and SIDE.get(call.rating) else (call.stop, call.stop_from)
    return {
        "rating": call.rating,
        "side": {1: "long", -1: "short"}.get(SIDE.get(call.rating, 0), "none"),
        "risk_level": call.risk_level or "",
        "size_pct": state.get("final_position_size_pct"),
        "ceiling_pct": call.ceiling_pct,
        "stop": stop,
        "stop_from": stop_from,
        "target": call.target,
        "expected_move_pct": call.target_move_pct(close),
        "horizon": _field(pm, "Time Horizon"),
        "summary": _field(pm, "Executive Summary"),
    }


def agent_outputs(state: dict) -> list[tuple[str, str]]:
    """Each agent's text, in desk order; an agent that did not run is left out."""
    out = []
    for name, key in AGENT_SECTIONS:
        text = (state.get(key[0]) or {}).get(key[1]) if isinstance(key, tuple) else state.get(key)
        if isinstance(text, str) and text.strip():
            out.append((name, text))
    return out


def latest_book(results_dir: Path, day: str | None = None) -> tuple[Path | None, list[dict]]:
    """The newest orders CSV the size command wrote, as rows; for ``day``, the newest for that date."""
    folder = Path(results_dir) / "backtest" / "sizing"
    pattern = f"*_{day}.csv" if day else "*.csv"
    files = sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime) if folder.is_dir() else []
    if not files:
        return None, []
    with files[-1].open(encoding="utf-8") as f:
        return files[-1], list(csv.DictReader(f))


def positions_view(holdings, decisions: dict[str, DecisionRef], default_stop_pct: float) -> list[dict]:
    """One row per open position, with its decision's numbers and where it stands against them."""
    rows = []
    for h in holdings:
        ref = decisions.get(h.symbol.upper())
        price = abs(h.market_value) / abs(h.qty) if h.qty else None
        row = {
            "ticker": h.symbol, "side": h.side, "qty": h.qty, "market_value": h.market_value,
            "avg_entry_price": h.avg_entry_price, "price": price,
            "pnl_pct": ((price / h.avg_entry_price - 1) * 100 * (1 if h.qty > 0 else -1)
                        if price and h.avg_entry_price else None),
            "asset_class": h.asset_class, "decision_date": None, "run": None, "decision": None,
            "to_stop_pct": None, "to_target_pct": None,
        }
        if ref is not None:
            summary = decision_summary(load_decision(ref), price, default_stop_pct)
            row.update(decision_date=ref.date, run=ref.run, decision=summary)
            d = 1 if h.qty > 0 else -1
            if price and summary["stop"]:
                row["to_stop_pct"] = (summary["stop"] / price - 1) * 100 * d
            if price and summary["target"]:
                row["to_target_pct"] = (summary["target"] / price - 1) * 100 * d
        rows.append(row)
    return rows


def as_json(ref: DecisionRef) -> dict:
    return {**asdict(ref), "path": str(ref.path)}


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")
