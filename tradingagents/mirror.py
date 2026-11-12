"""The mirror account: Valya's book is the exact opposite of Heinrich's.

A second Alpaca paper account (``API_KEY_VALYA`` / ``API_SECRET_VALYA``, the
names Dublin's ``.env`` uses) holds every equity position of the main
account the other way round, in the same number of shares: where the desk is
long, Valya is short, and where it is short, she is long. Her levels are the
desk's swapped: the desk's target is her stop and the desk's stop is her
target, which the geometry gives for free (a long's target is above the
price, which is where a short's stop belongs), and which the exits honour by
construction, since she leaves a name the moment the desk does, for whatever
reason it did.

Nothing here decides anything. After each of the Watcher's passes, and after
the book is placed at the open, :func:`follow` reads both accounts and sends
the market orders that make Valya's equity positions the negation of
Heinrich's: a name the desk opened is opened the other way, a name it closed
is closed, a trim is trimmed. A name whose desk order has not filled yet
waits for the next pass, as does one with a mirror order still open, and a
name that must change side is first flattened and opened the other way on
the pass after, since one order may not cross zero. Options (the SPY hedge)
and crypto are not mirrored. A name Alpaca will not let her short is noted
in the ledger and shown on her page as unmirrored.

The ledger (``mirror_state_path``) keeps the orders sent and the names
refused; no transaction list or P/L report is kept for her, by design.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

from tradingagents.dataflows.vendors.alpaca import common, trading
from tradingagents.dataflows.vendors.alpaca.trading import Account, Holding

logger = logging.getLogger(__name__)

KEY_VARS = ("API_KEY_VALYA", "ALPACA_MIRROR_API_KEY")
SECRET_VARS = ("API_SECRET_VALYA", "ALPACA_MIRROR_SECRET_KEY")
MIRRORED_CLASSES = {"us_equity"}


def credentials() -> tuple[str, str] | None:
    key = next((os.environ[v] for v in KEY_VARS if os.environ.get(v)), None)
    secret = next((os.environ[v] for v in SECRET_VARS if os.environ.get(v)), None)
    return (key, secret) if key and secret else None


def configured(config: dict) -> bool:
    return bool(config.get("mirror_enabled", True)) and credentials() is not None


def client():
    """The mirror account's trading client. It is always a paper client: the mirror never trades live."""
    creds = credentials()
    if creds is None:
        raise common.VendorNotConfiguredError("The mirror account is not configured: set API_KEY_VALYA and API_SECRET_VALYA")
    return common.trading_client(True, *creds)


def get_account(cl=None) -> Account:
    return trading.get_account(cl or client(), paper=True)


# --- the ledger -----------------------------------------------------------------

class MirrorState:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.orders: list[dict] = []
        self.refused: dict[str, dict] = {}      # symbol -> {"at", "why", "wanted"}
        self.last_follow = ""
        if self.path.is_file():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self.orders = list(data.get("orders", []))
                self.refused = dict(data.get("refused", {}))
                self.last_follow = data.get("last_follow", "")
            except (OSError, ValueError):
                pass

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"orders": self.orders[-2000:], "refused": self.refused, "last_follow": self.last_follow},
                                        indent=2) + "\n", encoding="utf-8")

    def record(self, symbol: str, side: str, qty: int, order_id: str, at: datetime, reason: str) -> None:
        self.orders.append({"symbol": symbol, "side": side, "qty": qty, "order_id": order_id,
                            "at": at.isoformat(), "reason": reason})
        self.refused.pop(symbol.upper(), None)

    def refuse(self, symbol: str, wanted: str, why: str, at: datetime) -> None:
        self.refused[symbol.upper()] = {"at": at.isoformat(), "wanted": wanted, "why": str(why)[:300]}


# --- what to send -----------------------------------------------------------------

def target_positions(main: Account) -> dict[str, int]:
    """Valya's wanted shares per symbol: the negation of every equity position of the main account."""
    return {h.symbol.upper(): -int(round(h.qty)) for h in main.holdings
            if h.asset_class in MIRRORED_CLASSES and int(round(h.qty)) != 0}


def current_positions(mirror: Account) -> dict[str, int]:
    return {h.symbol.upper(): int(round(h.qty)) for h in mirror.holdings
            if h.asset_class in MIRRORED_CLASSES and int(round(h.qty)) != 0}


def orders_to_send(wanted: dict[str, int], have: dict[str, int], main_pending: set[str], mirror_pending: set[str]) -> list[dict]:
    """The market orders that take ``have`` to ``wanted``: symbol, side, qty, why; skipping names still in motion.

    A name that must change side is only flattened here; the opening order
    follows on the pass after, once the close has filled.
    """
    out = []
    for symbol in sorted(set(wanted) | set(have)):
        want, has = wanted.get(symbol, 0), have.get(symbol, 0)
        if want == has:
            continue
        if symbol in main_pending:
            out.append({"symbol": symbol, "side": None, "qty": 0, "why": "the desk's own order has not filled yet"})
            continue
        if symbol in mirror_pending:
            out.append({"symbol": symbol, "side": None, "qty": 0, "why": "an earlier mirror order has not filled yet"})
            continue
        if has and want and (has > 0) != (want > 0):
            delta, why = -has, f"flatten {'long' if has > 0 else 'short'} {abs(has)} before going the other way"
        else:
            delta = want - has
            if want == 0:
                why = f"the desk closed {'its short' if has > 0 else 'its long'}: close {'long' if has > 0 else 'short'} {abs(has)}"
            elif has == 0:
                why = f"the desk opened {'a long' if want < 0 else 'a short'} of {abs(want)}: open {'short' if want < 0 else 'long'} {abs(want)}"
            elif abs(want) > abs(has):
                why = f"the desk added {abs(want) - abs(has)}: add to the {'short' if want < 0 else 'long'}"
            else:
                why = f"the desk trimmed {abs(has) - abs(want)}: trim the {'short' if want < 0 else 'long'}"
        out.append({"symbol": symbol, "side": "buy" if delta > 0 else "sell", "qty": abs(delta), "why": why})
    return out


# --- the pass -----------------------------------------------------------------

def _log(config: dict, now: datetime, text: str) -> None:
    path = Path(config["results_dir"]) / "worker" / f"{now:%Y-%m-%d}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"{now:%H:%M:%S} mirror: {text}\n")


def follow(config: dict, dry_run: bool = False, main: Account | None = None, market_open: bool | None = None) -> list[dict]:
    """Bring the mirror account to the opposite of the main one; returns the orders sent (or, dry, wanted).

    Never raises for a single name: a refused order (a stock Alpaca will not
    lend) is noted in the ledger and the rest go through. Called after every
    Watcher pass and after the book is placed; safe to call by hand
    (``tradingagents mirror``) to sync the accounts the first time.
    """
    if not configured(config):
        return []
    now = datetime.now(UTC).astimezone()
    name = config.get("mirror_name", "Valya")
    state = MirrorState(config["mirror_state_path"])
    cl = client()
    main = main or trading.get_account()
    if market_open is None:
        market_open = trading.get_clock().is_open
    mirror = get_account(cl)
    wanted, have = target_positions(main), current_positions(mirror)
    main_pending = set(trading.open_orders().values())
    mirror_pending = set(trading.open_orders(cl).values())
    plan = orders_to_send(wanted, have, main_pending, mirror_pending)
    todo = [o for o in plan if o["side"]]
    for o in plan:
        if not o["side"]:
            _log(config, now, f"{o['symbol']} waits: {o['why']}")
    if not todo:
        state.last_follow = now.isoformat()
        state.save()
        return []
    if not market_open:
        _log(config, now, f"{len(todo)} order(s) wanted for {name} but the market is closed; next pass")
        return []
    sent = []
    for o in todo:
        label = f"{o['side']} {o['qty']} {o['symbol']} ({o['why']})"
        if dry_run:
            _log(config, now, f"dry run: would send {label}")
            sent.append({**o, "order_id": None})
            continue
        try:
            order_id = trading.submit_market_order(o["symbol"], o["qty"], o["side"], cl)
        except Exception as exc:
            state.refuse(o["symbol"], label, str(exc), now)
            _log(config, now, f"refused: {label}: {exc}")
            logger.warning("Mirror (%s): %s refused: %s", name, label, exc)
            continue
        state.record(o["symbol"], o["side"], o["qty"], order_id, now, o["why"])
        sent.append({**o, "order_id": order_id})
        _log(config, now, f"sent {label}, order {order_id}")
    state.last_follow = now.isoformat()
    state.save()
    logger.info("Mirror (%s): %d order(s) sent, %d refused.", name, len(sent), len(todo) - len(sent))
    return sent


# --- the page's view -------------------------------------------------------------

def positions_view(mirror: Account, main: Account, decisions: dict, default_stop_pct: float, refused: dict | None = None) -> list[dict]:
    """Valya's positions with her levels: the desk's target as her stop, the desk's stop as her target."""
    from tradingagents.web import data

    desk = {h.symbol.upper(): h for h in main.holdings}
    rows = []
    for h in mirror.holdings:
        if h.asset_class not in MIRRORED_CLASSES:
            continue
        price = abs(h.market_value) / abs(h.qty) if h.qty else None
        row = {"ticker": h.symbol, "side": h.side, "qty": h.qty, "market_value": h.market_value,
               "avg_entry_price": h.avg_entry_price, "price": price,
               "pnl_pct": ((price / h.avg_entry_price - 1) * 100 * (1 if h.qty > 0 else -1)
                           if price and h.avg_entry_price else None),
               "stop": None, "target": None, "to_stop_pct": None, "to_target_pct": None,
               "desk_side": None, "decision_date": None, "matched": False}
        d = desk.get(h.symbol.upper())
        if d is not None:
            row["desk_side"] = d.side
            row["matched"] = int(round(d.qty)) == -int(round(h.qty))
        ref = decisions.get(h.symbol.upper())
        if ref is not None:
            s = data.decision_summary(data.load_decision(ref), price, default_stop_pct)
            row["decision_date"] = ref.date
            row["stop"], row["target"] = s.get("target"), s.get("stop")      # swapped: hers are the desk's the other way
            sign = 1 if h.qty > 0 else -1
            if price and row["stop"]:
                row["to_stop_pct"] = (row["stop"] / price - 1) * 100 * sign
            if price and row["target"]:
                row["to_target_pct"] = (row["target"] / price - 1) * 100 * sign
        rows.append(row)
    held = {r["ticker"].upper() for r in rows}
    for symbol, want in target_positions(main).items():
        if symbol not in held:
            why = (refused or {}).get(symbol, {}).get("why", "not yet mirrored")
            rows.append({"ticker": symbol, "side": "long" if want > 0 else "short", "qty": 0, "market_value": 0.0,
                         "avg_entry_price": None, "price": None, "pnl_pct": None, "stop": None, "target": None,
                         "to_stop_pct": None, "to_target_pct": None, "desk_side": "short" if want > 0 else "long",
                         "decision_date": None, "matched": False, "missing": why})
    rows.sort(key=lambda r: (r.get("missing") is not None, r["ticker"]))
    return rows
