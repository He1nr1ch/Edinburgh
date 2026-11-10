"""The daily worker's loop: a pass every few minutes while the market is open.

Each pass reads the account, runs every duty, sends the actions that pass the
guardrails, logs everything, and emails what needs attention. After the close
it emails the day's summary and sleeps until the next open: it runs for good,
as the container's process.
"""

from __future__ import annotations

import logging
import math
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from tradingagents import build
from tradingagents.dataflows.config import run_config
from tradingagents.dataflows.vendors.alpaca import trading
from tradingagents.worker.duty import Action, Duty, Finding, WorkerContext
from tradingagents.worker.notify import send_email
from tradingagents.worker.policy import refusal
from tradingagents.worker.state import WorkerState
from tradingagents.web.snapshots import write_snapshot

logger = logging.getLogger(__name__)


ET = ZoneInfo("America/New_York")


def default_duties() -> list[Duty]:
    """The Watcher's guards, in order: stop and target (arithmetic, no model), news, then replacements."""
    from tradingagents.worker.news import WatcherDuty
    from tradingagents.worker.replace import ReplacementDuty
    from tradingagents.worker.stops import StopDuty, TargetDuty

    return [StopDuty(), TargetDuty(), WatcherDuty(), ReplacementDuty()]


def _log(config: dict, now: datetime, text: str) -> None:
    path = Path(config["results_dir"]) / "worker" / f"{now:%Y-%m-%d}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"{now:%H:%M:%S} {text}\n")


def _act(f: Finding, context: WorkerContext, market_open: bool, pending: set[str],
         open_now: dict[str, str], dry_run: bool) -> None:
    a = f.action
    why_not = refusal(a, market_open=market_open, paper=context.account.paper,
                      pending=pending, config=context.config)
    # A close or trim first cancels the symbol's waiting orders: Alpaca holds the
    # shares a waiting sell has claimed, and a take-profit left behind after the
    # position is closed would fill later and open it the other way round.
    waiting = sum(1 for sym in open_now.values() if sym == a.symbol)
    if why_not:
        f.outcome = f"not sent: {why_not}"
    elif dry_run:
        f.outcome = "dry run: not sent" + (f"; would first cancel {waiting} waiting order(s)"
                                           if a.kind in ("exit", "trim") and waiting else "")
    else:
        try:
            cancelled = trading.cancel_orders_for(a.symbol, open_now) if a.kind in ("exit", "trim") else 0
            order_id = trading.submit_market_order(a.symbol, a.qty, a.side)
            held = next((h for h in context.account.holdings if h.symbol.upper() == a.symbol.upper()), None)
            context.state.record_order(a.symbol, a.kind, a.side, a.qty, order_id, context.now, a.reason,
                                       entry_price=held.avg_entry_price if held and a.kind in ("exit", "trim") else None)
            if a.kind == "replace":
                context.state.book[a.symbol.upper()] = {"placed_on": context.now.strftime("%Y-%m-%d"),
                                                        "side": "long" if a.side == "buy" else "short",
                                                        "value": a.value}
            pending.add(a.symbol)
            f.outcome = f"sent, order {order_id}" + (
                f"; cancelled {cancelled} waiting order(s) first"
                + ("; the shares left have no bracket now" if a.kind == "trim" else "")
                if cancelled else "")
        except Exception as exc:
            f.outcome = f"failed: {exc}"


def run_pass(config: dict, llm, duties: list[Duty], dry_run: bool = False) -> list[Finding]:
    """One look at the account: every duty, the guardrails, the log and the alert."""
    with run_config(config):
        now = datetime.now(UTC).astimezone()
        state = WorkerState.load(config["worker_state_path"])
        account = trading.get_account()
        market_open = trading.get_clock().is_open
        context = WorkerContext(account, now, config, state, llm)
        findings: list[Finding] = []
        for duty in duties:
            try:
                findings.extend(duty.run(context))
            except Exception as exc:     # a failing duty is reported, not fatal to the others
                findings.append(Finding(duty.name, "-", "notable", f"the {duty.name} duty failed", str(exc)))
        open_now = trading.open_orders() if any(f.action for f in findings) else {}
        # Only the worker's own orders, sent and not yet filled, hold a symbol back;
        # orders placed by anything else (a bracket's take-profit) do not.
        mine = {o["order_id"] for o in state.orders}
        pending = {sym for order_id, sym in open_now.items() if order_id in mine}
        for f in findings:
            if f.action:
                _act(f, context, market_open, pending, open_now, dry_run)
        state.save()
        # The day's record for the positions page: the last pass of the day
        # leaves the close-of-day picture.
        try:
            write_snapshot(Path(config["results_dir"]), trading.get_account(), config.get("default_stop_pct") or 0, now)
        except Exception as exc:
            logger.warning("The positions snapshot could not be written: %s", exc)

    mode = "paper" if account.paper else "LIVE"
    _log(config, now, f"pass ({mode}, {len(context.watched())} positions watched): {len(findings)} finding(s)")
    logger.info("Pass (%s): %d positions watched, %d finding(s).", mode, len(context.watched()), len(findings))
    for f in findings:
        order = f" -> {f.action.kind} {f.action.side} {f.action.qty} [{f.outcome}]" if f.action else ""
        _log(config, now, f"  [{f.severity}] {f.symbol} {f.headline}{order}")
        logger.info("[%s] %s %s%s", f.severity, f.symbol, f.headline, order)
    alerts = [f for f in findings if f.severity != "info" or f.action]
    if alerts:
        body = "\n\n".join(
            f"[{f.severity.upper()}] {f.headline}\n"
            + (f"Order: {f.action.kind} — {f.action.side} {f.action.qty} {f.symbol} — {f.outcome}\n" if f.action else "")
            + f"\n{f.detail}" for f in alerts)
        sent = send_email(f"Edinburgh worker ({mode}): {len(alerts)} alert(s) at {now:%H:%M}", body, config)
        if not sent:
            _log(config, now, "  (alert email not sent: email is not set up or failed)")
    return findings


def review_exits(holdings, decisions) -> list[tuple]:
    """Held positions whose newest decision no longer backs their side, with the rating that says so.

    The nightly sweep reviews the holdings first; a long now rated Hold, Underweight
    or Sell, or a short now rated Hold, Overweight or Buy, is closed at the open.
    """
    from tradingagents.sizing import SIDE
    from tradingagents.web.data import load_decision

    out = []
    for h in holdings:
        if h.asset_class != "us_equity":
            continue
        ref = decisions.get(h.symbol.upper())
        if ref is None:
            continue
        rating = load_decision(ref).get("final_rating") or ""
        if SIDE.get(rating, 0) != (1 if h.qty > 0 else -1):
            out.append((h, rating or "no rating", ref.date))
    return out


def place_book(config: dict, dry_run: bool = False) -> list[str]:
    """Open the day's book at the open, once per day, after the holdings review.

    First the review: a held position whose newest decision no longer backs its
    side is closed. Then the book's orders (the one the daily job built this
    morning, or the newest saved, with a note) are opened, skipping names held,
    and only while the gross exposure, counting what is still held, stays within
    ``gross_pct`` of equity.
    """
    from tradingagents.book import load_rows
    from tradingagents.hedge import held_from_row, is_hedge_row, open_hedges, record_held
    from tradingagents.web.data import latest_book

    with run_config(config):
        now = datetime.now(UTC).astimezone()
        today = now.strftime("%Y-%m-%d")
        state = WorkerState.load(config["worker_state_path"])
        day = state.day(today)
        if day["placed"] or not config.get("worker_place_book", True):
            return []
        ledger = Path(config["hedge_ledger_path"])
        hedged = open_hedges(ledger, today)
        book_path = Path(day["book_path"]) if day.get("book_path") else None
        if book_path is None or not book_path.is_file():
            book_path, _ = latest_book(Path(config["results_dir"]))
            if book_path is None:
                _log(config, now, "no book to place")
                return []
            _log(config, now, f"today's job left no book; placing the newest, {book_path.name}")
        from tradingagents.web.data import find_decisions

        account = trading.get_account()
        market_open = trading.get_clock().is_open
        open_now = trading.open_orders()
        # Only the worker's own unfilled orders hold a name back (a bracket's
        # take-profit does not); the review cancels those before it closes.
        mine = {o["order_id"] for o in state.orders}
        pending = {sym for oid, sym in open_now.items() if oid in mine}
        closed = set()
        for h, rating, decided in review_exits(account.holdings, find_decisions(Path(config["results_dir"]))):
            qty, side = abs(int(h.qty)), "sell" if h.qty > 0 else "buy"
            action = Action(h.symbol, "exit", side, qty, f"review on {decided}: rated {rating}, no longer {h.side}")
            why_not = refusal(action, market_open=market_open, paper=account.paper, pending=pending, config=config)
            if why_not:
                _log(config, now, f"  review exit not sent: {h.symbol}: {why_not}")
                continue
            if dry_run:
                _log(config, now, f"  dry run: would close {h.side} {h.symbol} ({action.reason})")
                closed.add(h.symbol.upper())
                continue
            try:
                trading.cancel_orders_for(h.symbol, open_now)
                order_id = trading.submit_market_order(h.symbol, qty, side)
            except Exception as exc:
                _log(config, now, f"  review exit failed: {h.symbol}: {exc}")
                continue
            state.record_order(h.symbol, "exit", side, qty, order_id, now, action.reason, entry_price=h.avg_entry_price)
            state.book.pop(h.symbol.upper(), None)     # the book refills it, not the bench
            closed.add(h.symbol.upper())
            _log(config, now, f"  closed {h.side} {h.symbol}: {action.reason} (order {order_id})")
        held = {h.symbol.upper() for h in account.holdings} - closed
        gross = sum(abs(h.market_value) for h in account.holdings
                    if h.symbol.upper() in held and h.asset_class == "us_equity")
        gross_cap = account.equity * float(config["gross_pct"]) / 100
        placed = []
        waiting = set(open_now.values()) - closed
        rows = load_rows(book_path)
        # Best first: when the gross target leaves room for only some of the
        # book, the highest-scored calls take it (see ranking.py).
        rows.sort(key=lambda o: -float(o["score"]) if o.get("score") not in (None, "") else 0.0)
        for o in rows:
            if o["ticker"].upper() in held | waiting:
                continue
            if is_hedge_row(o):
                # The SPY calls the job priced for a net-short book. They are the
                # hedge ledger's business, not the Watcher's: it never adds them
                # to its book, and its guards look at equities only.
                contracts = abs(int(float(o["shares"])))
                if hedged:
                    _log(config, now, "  hedge not bought: still hedged with "
                         + ", ".join(f"{h.contracts} {h.symbol} until {h.expiry}" for h in hedged))
                    continue
                action = Action(o["ticker"], "open", "buy", contracts, f"the day's book ({book_path.name}): its SPY call hedge")
                why_not = refusal(action, market_open=market_open, paper=account.paper, pending=pending, config=config)
                if why_not:
                    _log(config, now, f"  hedge not bought: {why_not}")
                    continue
                if dry_run:
                    _log(config, now, f"  dry run: would buy {contracts} {o['ticker']} (hedge)")
                    placed.append(o["ticker"])
                    continue
                try:
                    order_id = trading.submit_market_order(o["ticker"], contracts, "buy")
                except Exception as exc:
                    _log(config, now, f"  hedge failed: {o['ticker']}: {exc}")
                    continue
                state.record_order(o["ticker"], "open", "buy", contracts, order_id, now, action.reason)
                record_held(ledger, held_from_row(o, today))
                placed.append(o["ticker"])
                _log(config, now, f"  bought {contracts} {o['ticker']} (hedge, order {order_id}); recorded in {ledger.name}")
                continue
            if o["side"] not in ("long", "short"):
                continue
            if gross + float(o["value"]) > gross_cap:
                _log(config, now, f"  not placed: {o['ticker']}: the book is at its {config['gross_pct']:g}% gross "
                                  f"({gross:,.0f} of {gross_cap:,.0f}) with what is still held")
                continue
            qty = abs(int(float(o["shares"])))
            action = Action(o["ticker"], "open", "buy" if o["side"] == "long" else "sell", qty,
                            f"the day's book ({book_path.name})")
            why_not = refusal(action, market_open=market_open, paper=account.paper, pending=pending, config=config)
            if why_not:
                _log(config, now, f"  not placed: {o['ticker']}: {why_not}")
                continue
            if dry_run:
                _log(config, now, f"  dry run: would open {o['side']} {qty} {o['ticker']}")
                placed.append(o["ticker"])
                gross += float(o["value"])
                continue
            try:
                order_id = trading.submit_market_order(o["ticker"], qty, action.side)
            except Exception as exc:
                _log(config, now, f"  failed: {o['ticker']}: {exc}")
                continue
            state.record_order(o["ticker"], "open", action.side, qty, order_id, now, action.reason)
            state.book[o["ticker"].upper()] = {"placed_on": today, "side": o["side"], "value": float(o["value"])}
            gross += float(o["value"])
            placed.append(o["ticker"])
            _log(config, now, f"  opened {o['side']} {qty} {o['ticker']} (order {order_id})")
        day["placed"] = placed
        state.save()
        _log(config, now, f"book placed: {len(placed)} order(s) from {book_path.name}")
        if placed:
            send_email(f"Edinburgh: the day's book, {len(placed)} order(s)",
                       "\n".join(f"- {t}" for t in placed) + f"\n\nfrom {book_path}", config)
        return placed


def build_book_job(config: dict) -> None:
    """The morning book, once per trading day, from the decisions the sweep has left; recorded whatever happens."""
    from tradingagents.daily import build_morning_book, describe_timing
    from tradingagents.hedge import describe_hedge

    with run_config(config):
        today = datetime.now(ET).strftime("%Y-%m-%d")
        state = WorkerState.load(config["worker_state_path"])
        day = state.day(today)
        if day["analysis"].startswith("book built"):
            return
        try:
            equity = trading.get_account().equity
            built = build_morning_book(config, equity)
            day.update(analysis=f"book built {datetime.now(ET).strftime('%H:%M')}", book_path=str(built.book_path))
            send_email(f"Edinburgh: the book for {today}",
                       f"{len(built.book.orders)} orders and a bench of {len(built.bench)} on {built.date}'s close.\n"
                       f"Hedge: {describe_hedge(built.hedge)}\n"
                       f"{describe_timing(built.timing or {})}\n"
                       + "\n".join(f"- {o.side} {abs(o.shares)} {o.ticker} @ {o.price:,.2f}, stop {o.stop}, target {o.target}"
                                   for o in built.book.orders), config)
        except Exception as exc:
            logger.exception("The morning book could not be built")
            day["analysis"] = f"failed: {exc}"
        state.save()


def book_due(config: dict, now: datetime, state: WorkerState, next_open: datetime) -> bool:
    """Whether today's book should be built now: its time (ET) has come on a day the market opens
    (by Alpaca's clock, so not a weekend or a holiday) and it is not built yet."""
    local = now.astimezone(ET)
    if next_open.astimezone(ET).date() != local.date():
        return False
    hour, minute = (int(x) for x in str(config["daily_book_time"]).split(":"))
    due_at = local.replace(hour=hour, minute=minute, second=0, microsecond=0)
    # Anything but a built book counts as not built: a failure is retried, and
    # the old nightly job's "running" or "done" marker (a record from before the
    # sweep ran around the clock) does not stand in for today's book.
    status = state.day(local.strftime("%Y-%m-%d"))["analysis"]
    return local >= due_at and not status.startswith("book built")


def tick_sweep(sweep, config: dict, market_open: bool) -> None:
    """Advance the continuous sweep; a failure here is logged, never the loop's."""
    from tradingagents.web.data import all_decisions

    try:
        with run_config(config):
            now = datetime.now().astimezone()
            date = trading.last_trading_day(now)
            holdings = [h.symbol for h in trading.get_account().holdings if h.asset_class == "us_equity"]
        last = {t: refs[0].date for t, refs in all_decisions(Path(config["results_dir"])).items()}
        sweep.tick(date, market_open, holdings, last)
    except Exception as exc:
        logger.warning("The sweep could not be advanced: %s", exc)


def summarize_day(config: dict, day: str) -> str:
    """The close-of-day email: the transactions with their reasons, then the day's notable findings."""
    from tradingagents.web.transactions import summarize, transactions_on

    state = WorkerState.load(config["worker_state_path"])
    try:
        with run_config(config):
            fills = trading.fills_on(day)
    except Exception as exc:      # the list still goes out, without fill prices
        logger.warning("Fill prices for %s could not be read: %s", day, exc)
        fills = {}
    lines = [summarize(transactions_on(state, day, fills), day)]
    log = Path(config["results_dir"]) / "worker" / f"{day}.log"
    if log.is_file():
        notable = [line for line in log.read_text(encoding="utf-8").splitlines() if "[urgent]" in line or "[notable]" in line]
        lines += ["", f"{len(notable)} notable or urgent finding(s):"] + notable
    return "\n".join(lines)


def _sleep_until(target: datetime, now: datetime, chunk_minutes: int = 15) -> None:
    """Sleep toward ``target`` at most ``chunk_minutes`` at a time, so the clock is asked again."""
    time.sleep(max(1.0, min((target - now).total_seconds(), chunk_minutes * 60)))


def run_day(config: dict, llm, duties: list[Duty], interval_minutes: int, dry_run: bool = False, sweep=None) -> None:
    """Passes from now until the close, then the day's summary. Call while the market is open.

    The Watcher's pass runs first on every turn and the sweep is advanced after
    it; during market hours the sweep runs on one node, so the Watcher's calls
    find the others free."""
    # The build again at the top of the trading day: this process may have been
    # running since a deploy days ago, and the day's record is what later answers
    # "which build traded on this date".
    today = datetime.now(ET).strftime("%Y-%m-%d")
    _log(config, datetime.now(UTC).astimezone(), build.log_banner(f"trading day {today}"))
    state = WorkerState.load(config["worker_state_path"])
    state.day(today)["build"] = build.describe()
    state.save()
    place_book(config, dry_run)
    while True:
        run_pass(config, llm, duties, dry_run)
        if sweep is not None:
            tick_sweep(sweep, config, market_open=True)
        with run_config(config):
            clock = trading.get_clock()
        if not clock.is_open:
            break
        time.sleep(interval_minutes * 60)
    day = datetime.now().strftime("%Y-%m-%d")
    send_email(f"Edinburgh worker: summary for {day}", summarize_day(config, day), config)


def run_forever(config: dict, llm, duties: list[Duty], interval_minutes: int, dry_run: bool = False) -> None:
    """The container's loop: the sweep around the clock, the book at its time, the Watcher through each session.

    Weekends and holidays follow Alpaca's clock. A failure (the broker or the
    model unreachable) is logged and retried a minute later, never fatal.
    """
    from tradingagents.sweep import Sweep

    build.log_banner("worker start")
    sweep = Sweep(config)
    logger.info("Sweep: %s", sweep.status())
    while True:
        try:
            with run_config(config):
                clock = trading.get_clock()
                state = WorkerState.load(config["worker_state_path"])
            tick_sweep(sweep, config, clock.is_open)
            if not clock.is_open and book_due(config, clock.now, state, clock.next_open):
                logger.info("Building the morning book.")
                build_book_job(config)
                continue
            if clock.is_open:
                logger.info("The market is open; watching until the close.")
                run_day(config, llm, duties, interval_minutes, dry_run, sweep)
                continue
            # Closed: sleep a short while so the sweep's rounds keep turning and
            # the book is built on the minute.
            time.sleep(min(300, max(1.0, (clock.next_open - clock.now).total_seconds())))
        except Exception:
            logger.exception("The worker hit an error; trying again in a minute.")
            time.sleep(60)
