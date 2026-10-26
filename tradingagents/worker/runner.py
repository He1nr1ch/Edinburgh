"""The daily worker's loop: a pass every few minutes while the market is open.

Each pass reads the account, runs every duty, sends the actions that pass the
guardrails, logs everything, and emails what needs attention. After the close
it emails the day's summary and sleeps until the next open: it runs for good,
as the container's process.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from pathlib import Path

from tradingagents.dataflows.config import run_config
from tradingagents.dataflows.vendors.alpaca import trading
from tradingagents.worker.duty import Duty, Finding, WorkerContext
from tradingagents.worker.notify import send_email
from tradingagents.worker.policy import refusal
from tradingagents.worker.state import WorkerState

logger = logging.getLogger(__name__)


def default_duties() -> list[Duty]:
    from tradingagents.worker.news import NewsDuty

    return [NewsDuty()]


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
            context.state.record_order(a.symbol, a.kind, a.side, a.qty, order_id, context.now)
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


def summarize_day(config: dict, day: str) -> str:
    state = WorkerState.load(config["worker_state_path"])
    orders = state.orders_on(day)
    log = Path(config["results_dir"]) / "worker" / f"{day}.log"
    lines = [f"Edinburgh worker, {day}: {len(orders)} order(s)."]
    lines += [f"- {o['at'][11:16]} {o['kind']} {o['side']} {o['qty']} {o['symbol']} (order {o['order_id']})" for o in orders]
    if log.is_file():
        notable = [line for line in log.read_text(encoding="utf-8").splitlines() if "[urgent]" in line or "[notable]" in line]
        lines += ["", f"{len(notable)} notable or urgent finding(s):"] + notable
    return "\n".join(lines)


def _sleep_until(target: datetime, now: datetime, chunk_minutes: int = 15) -> None:
    """Sleep toward ``target`` at most ``chunk_minutes`` at a time, so the clock is asked again."""
    time.sleep(max(1.0, min((target - now).total_seconds(), chunk_minutes * 60)))


def run_day(config: dict, llm, duties: list[Duty], interval_minutes: int, dry_run: bool = False) -> None:
    """Passes from now until the close, then the day's summary. Call while the market is open."""
    while True:
        run_pass(config, llm, duties, dry_run)
        with run_config(config):
            clock = trading.get_clock()
        if not clock.is_open:
            break
        time.sleep(interval_minutes * 60)
    day = datetime.now().strftime("%Y-%m-%d")
    send_email(f"Edinburgh worker: summary for {day}", summarize_day(config, day), config)


def run_forever(config: dict, llm, duties: list[Duty], interval_minutes: int, dry_run: bool = False) -> None:
    """The container's loop: each market day from the open to the close, asleep in between.

    Weekends and holidays follow Alpaca's clock. A failure (the broker or the
    model unreachable) is logged and retried a minute later, never fatal.
    """
    while True:
        try:
            with run_config(config):
                clock = trading.get_clock()
            if clock.is_open:
                logger.info("The market is open; watching until the close.")
                run_day(config, llm, duties, interval_minutes, dry_run)
                continue
            logger.info("The market is closed; the next open is %s.", clock.next_open)
            _sleep_until(clock.next_open, clock.now)
        except Exception:
            logger.exception("The worker hit an error; trying again in a minute.")
            time.sleep(60)
