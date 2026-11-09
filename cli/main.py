import sys

import typer

from cli.display import console
from cli.models import AnalystType, AssetType
from cli.prompts import filter_analysts_for_asset_type, parse_analysts
from cli.run import run_analysis
from tradingagents.backtest import iter_grid, run_backtest, summarize
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.portfolio import load_portfolio

# prompt_toolkit's win32 output module is importable only on Windows (it asserts
# the platform at import time), so gate on the platform rather than catching the
# failure — that way a genuinely broken prompt_toolkit on Windows still surfaces
# instead of silently disabling the handler below. Off Windows this stays an
# empty tuple, which `except` accepts and never matches (#1138).
if sys.platform == "win32":  # pragma: no cover - platform dependent
    from prompt_toolkit.output.win32 import NoConsoleScreenBufferError

    _NO_CONSOLE_ERRORS: tuple[type[BaseException], ...] = (NoConsoleScreenBufferError,)
else:
    _NO_CONSOLE_ERRORS = ()

app = typer.Typer(
    name="edinburgh",
    help="Edinburgh: a multi-agent trading desk on Alpaca data",
    add_completion=True,  # Enable shell completion
)


@app.callback(invoke_without_command=True)
def analyze(
    ctx: typer.Context,
    checkpoint: bool | None = typer.Option(
        None,
        "--checkpoint/--no-checkpoint",
        help="Enable/disable checkpoint-resume (save state after each node so a "
        "crashed run can resume). Omit to honor TRADINGAGENTS_CHECKPOINT_ENABLED.",
    ),
    clear_checkpoints: bool = typer.Option(
        False,
        "--clear-checkpoints",
        help="Delete all saved checkpoints before running (force fresh start).",
    ),
    portfolio: str = typer.Option(
        None,
        "--portfolio",
        help="JSON file with current holdings and cash, so the trader, risk and "
        "portfolio agents size against your actual position.",
    ),
    ticker: str = typer.Option(None, "--ticker", help="Ticker to analyze, e.g. NVDA or BTC/USD; skips the prompt"),
    date: str = typer.Option(None, "--date", help="Analysis date, YYYY-MM-DD; skips the prompt"),
    analysts: str = typer.Option(
        None, "--analysts", help="Comma-separated analysts, e.g. market,news; skips the prompt"
    ),
    save: bool | None = typer.Option(
        None, "--save/--no-save", help="Save the report under results_dir without asking"
    ),
    show: bool | None = typer.Option(
        None, "--show/--no-show", help="Show the full report at the end without asking"
    ),
):
    """Run an analysis. This is what a bare `tradingagents` does.

    Flags answer their questions; with provider, models and depth also
    set through TRADINGAGENTS_* variables, the run asks nothing.
    """
    if ctx.invoked_subcommand is not None:
        return
    if clear_checkpoints:
        from tradingagents.graph.checkpointer import clear_all_checkpoints
        n = clear_all_checkpoints(DEFAULT_CONFIG["data_cache_dir"])
        console.print(f"[yellow]Cleared {n} checkpoint(s).[/yellow]")
    portfolio_context = None
    if portfolio:
        try:
            portfolio_context = load_portfolio(portfolio)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from None

    try:
        flags = {"ticker": ticker, "date": date, "analysts": analysts, "save": save, "show": show}
        run_analysis(checkpoint=checkpoint, portfolio=portfolio_context, flags=flags)
    except _NO_CONSOLE_ERRORS:
        # A terminal with no console buffer cannot host the interactive prompts.
        # Emit one actionable line on stderr instead of a prompt_toolkit
        # traceback; plain text, since rich may not render here either (#1138).
        typer.echo(
            "Error: no Windows console available. The interactive CLI needs a real "
            "console buffer — run it from Windows Terminal, PowerShell, or cmd.exe "
            "rather than a piped or embedded terminal.",
            err=True,
        )
        raise typer.Exit(code=1) from None


@app.command()
def backtest(
    tickers: str = typer.Argument(..., help="Comma-separated tickers, e.g. NVDA,AAPL"),
    start: str = typer.Option(..., "--start", help="First analysis date, YYYY-MM-DD"),
    end: str = typer.Option(..., "--end", help="Last analysis date, YYYY-MM-DD"),
    every: int = typer.Option(7, "--every", help="Days between analysis dates"),
    analysts: str = typer.Option(
        None, "--analysts", help="Comma-separated analysts to run: market, news, fundamentals; omit for all the asset type allows"
    ),
    asset_type: str = typer.Option("stock", "--asset-type", help="stock or crypto"),
    portfolio: str = typer.Option(
        None, "--portfolio", help="JSON file with holdings and cash, held constant across the grid"
    ),
    run_id: str = typer.Option(
        None, "--run-id", help="Continue an earlier sweep: its cells are skipped and its log reused"
    ),
):
    """Score past decisions over a grid of tickers and dates."""

    try:
        dates = iter_grid(start, end, every)
        book = load_portfolio(portfolio) if portfolio else None
        kind = AssetType(asset_type.strip().lower())
        # The analysts are named and checked as for an analysis; without a
        # choice, every analyst the asset type allows runs.
        chosen = (parse_analysts(analysts, kind) if analysts
                  else filter_analysts_for_asset_type(list(AnalystType), kind))
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None

    names = [t.strip() for t in tickers.split(",") if t.strip()]
    if not names:
        console.print("[red]No ticker to analyze; pass them comma-separated, e.g. NVDA,AAPL[/red]")
        raise typer.Exit(code=1)

    def show_progress(done, total, ticker, date):
        console.print(f"[dim][{done}/{total}] {ticker} {date}[/dim]")

    kwargs = {"asset_type": kind.value, "portfolio": book, "run_id": run_id, "progress": show_progress,
              "selected_analysts": [a.value for a in chosen]}

    try:
        result = run_backtest(names, dates, DEFAULT_CONFIG, **kwargs)
    except Exception as exc:  # a missing key or an unknown analyst is a setup error
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None
    console.print(summarize(result).render())
    console.print(f"\nRan {result.cells_run} cells, skipped {result.skipped}. Log: {result.log_path}")
    console.print(f"Continue or settle this sweep: --run-id {result.run_id}")
    for ticker, date, reason in result.failures:
        console.print(f"[yellow]failed:[/yellow] {ticker} {date}: {reason}")
    for ticker, reason in result.settlement_failures:
        console.print(f"[yellow]unsettled:[/yellow] {ticker}: {reason}")


@app.command()
def scorecard(
    ticker: str = typer.Option(None, "--ticker", help="Also list this ticker's settled decisions"),
    as_of: str = typer.Option(None, "--as-of", help="Only outcomes known by this date, YYYY-MM-DD"),
):
    """Show the feedback loop's track record: each agent's hit rate and settled decisions."""
    from tradingagents.feedback import FeedbackStore

    store = FeedbackStore(DEFAULT_CONFIG)
    if not store.enabled:
        console.print("[yellow]No feedback store is configured (TRADINGAGENTS_FEEDBACK_DB_URL).[/yellow]")
        raise typer.Exit(code=1)
    pending = store.pending()
    text = store.render_context(ticker or "", as_of=as_of)
    if text:
        console.print(text)
    else:
        console.print("No settled decisions yet.")
    if pending:
        console.print(f"\n{len(pending)} decision(s) pending settlement: "
                      + ", ".join(f"{p['ticker']} {p['trade_date']}" for p in pending))


@app.command()
def sp500():
    """Refresh watchlists/sp500.txt from Wikipedia's constituent list (fails closed on a bad fetch)."""
    from tradingagents.sp500 import WATCHLIST, refresh_sp500_watchlist

    try:
        count = refresh_sp500_watchlist()
    except Exception as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None
    console.print(f"{count} symbols written to {WATCHLIST}")


@app.command()
def size(
    run_ids: list[str] = typer.Argument(..., help="Backtest run IDs to size together, e.g. sp100_20261023_t0_a sp100_20261023_t0_b"),
    date: str = typer.Option(..., "--date", help="The analysis date whose calls to size, YYYY-MM-DD"),
    equity: float = typer.Option(100_000, "--equity", help="Portfolio equity to size against"),
    gross: float = typer.Option(75.0, "--gross", help="Target gross exposure, longs plus shorts, as % of equity"),
    cap: float = typer.Option(10.0, "--cap", help="Most any one position may be, as % of equity"),
    min_target: float = typer.Option(None, "--min-target", help="Take a call only when its target is more than this % from the close (default min_target_pct)"),
    top: int = typer.Option(None, "--top", help="Size only this many longs and this many shorts, by expected move (default names_per_side; 0 for all)"),
    hedge: bool = typer.Option(True, "--hedge/--no-hedge", help="Price SPY calls to cover a net-short book"),
    record_hedge: bool = typer.Option(False, "--record-hedge", help="Record the calls this plan buys as bought, so later runs do not hedge again"),
):
    """Size a list run's calls into a long/short book: shares and dollars per ticker, its bench and its hedge."""
    from pathlib import Path

    from rich.table import Table

    from tradingagents.book import build_book, save_book

    try:
        built = build_book(run_ids, date, DEFAULT_CONFIG, equity, gross, cap, min_target, top)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None
    book = built.book
    floor = DEFAULT_CONFIG["min_target_pct"] if min_target is None else min_target
    per_side = DEFAULT_CONFIG["names_per_side"] if top is None else top
    outside = sum(1 for _, why in book.skipped if why.startswith("outside the top"))
    if outside:
        console.print(f"[dim]{outside} call(s) cleared the floor but sit outside the top {per_side} on their side[/dim]")
    under = [t for t, why in book.skipped if "minimum" in why or "no price target" in why or "wrong side" in why]
    if under:
        console.print(f"[dim]{len(under)} call(s) left out for a target under {floor:g}% or missing: "
                      f"{', '.join(under[:12])}{' …' if len(under) > 12 else ''}[/dim]")

    table = Table(title=f"Book for {equity:,.0f} on {date}" + (f": top {per_side} longs and shorts" if per_side else ""))
    for column in ("Ticker", "Side", "Rating", "Risk", "Ceiling", "Weight", "Price", "Shares", "Value", "Stop", "Stop from", "Target"):
        table.add_column(column, justify="right" if column not in ("Ticker", "Side", "Rating", "Risk", "Stop from") else "left")
    for o in book.orders:
        table.add_row(o.ticker, o.side, o.rating, o.risk_level or "", f"{o.ceiling_pct:g}%",
                      f"{o.weight_pct:+.2f}%", f"{o.price:,.2f}", f"{o.shares:+,d}", f"{o.value:,.0f}",
                      f"{o.stop:,.2f}" if o.stop else "", o.stop_from, f"{o.target:,.2f}" if o.target else "")
    defaulted = sum(1 for o in book.orders if o.stop_from.startswith("default"))
    if defaulted:
        console.print(f"[dim]{defaulted} stop(s) set by default: the desk gave none, or one on the wrong side[/dim]")
    console.print(table)
    console.print(
        f"Long {book.long_value:,.0f} · short {book.short_value:,.0f} · "
        f"gross {book.gross_pct:.1f}% (target {gross:g}%) · net {book.net_pct:+.1f}% · "
        f"cash after longs {book.cash:,.0f}"
    )
    if book.gross_pct < gross - 1:
        at_cap = all(abs(o.weight_pct) >= cap - 1e-9 for o in book.orders)
        why = (f"every position is at the {cap:g}% cap; more calls or a higher --cap would fill it"
               if at_cap else "whole shares of these prices do not divide it more finely")
        console.print(f"[yellow]Short of the {gross:g}% target: {why}.[/yellow]")
    # The floor and the top-N cuts are counted above; a line each for a hundred
    # names is noise. What is left is the odd case worth a look: a stock too
    # dear for one share, a zero ceiling, a missing price.
    counted = ("outside the top", "target ", "no price target")
    for ticker, why in book.skipped:
        if not why.startswith(counted):
            console.print(f"[dim]no position: {ticker}: {why}[/dim]")
    if built.bench:
        console.print("[bold]Bench[/bold] (replacements, by move to target): "
                      + ", ".join(f"{b['ticker']} {b['side']} {b['expected_move_pct']:+.1f}%" for b in built.bench))

    plan = None
    if hedge:
        from tradingagents.hedge import hedge_book, hedge_row, open_hedges
        from tradingagents.hedge import record_hedge as record

        ledger = Path(DEFAULT_CONFIG["hedge_ledger_path"])
        plan = hedge_book(book, date, monthly_ratio=DEFAULT_CONFIG["hedge_monthly_ratio"],
                          held=open_hedges(ledger, date))
        console.print(f"\n[bold]Hedge[/bold] · beta-weighted exposure {plan.exposure:+,.0f} · SPY {plan.spy_price:,.2f}")
        for label, q in (("weekly", plan.weekly), ("monthly", plan.monthly)):
            if q:
                console.print(f"  {label:<8} {q.symbol}  strike {q.strike:g}  expires {q.expiry}  "
                              f"bid {q.bid:.2f} ask {q.ask:.2f}  = {q.contract_cost:,.0f} a contract")
        if plan.contracts:
            console.print(f"  [green]Buy {plan.contracts} {plan.quote.symbol} ({plan.choice}) for "
                          f"{plan.cost:,.0f}[/green]: {plan.reason}; covers {plan.coverage:.0%} of the exposure")
            stamp = plan.quote.quoted_at
            if stamp is not None and stamp.date().isoformat() != date:
                console.print(f"  [yellow]Quoted {stamp:%Y-%m-%d %H:%M} UTC, not on {date}: "
                              "option prices are the latest Alpaca has.[/yellow]")
            if record_hedge:
                record(ledger, plan, date)
                console.print(f"  Recorded as bought in {ledger}; later runs will not hedge again until it expires.")
            else:
                console.print("  Once bought, run again with --record-hedge so later runs do not buy more.")
        else:
            console.print(f"  No new calls: {plan.reason}"
                          + (f"; they cover {plan.coverage:.0%} of the exposure" if plan.held else "") + ".")
            if plan.held and plan.coverage < 1:
                console.print("  [yellow]The calls held cover less than the exposure now; "
                              "none are added while they are open.[/yellow]")

    save_book(built, DEFAULT_CONFIG, hedge_row(plan) if plan else None)
    console.print(f"Orders saved to {built.book_path}; bench to {built.bench_path.name}")


@app.command(name="pb-range")
def pb_range(
    ticker: str = typer.Argument(..., help="Ticker, e.g. NVDA or BTC/USD"),
    date: str = typer.Option(None, "--date", help="Last session to read, YYYY-MM-DD (default today)"),
    days: int = typer.Option(10, "--days", help="Calendar days of 15-minute bars to read before it"),
    equity: float = typer.Option(100_000, "--equity", help="Equity the positions are sized against"),
    risk: float = typer.Option(1.0, "--risk", help="Share of equity risked between entry and stop, in %"),
    backtest: bool = typer.Option(False, "--backtest", help="Also trade every setup in the window by the rules and report"),
):
    """Patrick Nill's P and B ranges on the 15-minute chart: the zones, the week's value area, the 4-hour bias, and the setups.

    An impulse then a range, valid on its third touch; traded edge to edge, on a
    confirmed breakout, or on a fake breakout that closes back inside. The stop
    comes from structure and the size from the stop.
    """
    from datetime import datetime, timedelta

    from rich.table import Table

    from tradingagents.dataflows.errors import NoMarketDataError, VendorError
    from tradingagents.dataflows.symbols import is_crypto_symbol, normalize_symbol
    from tradingagents.pb_range import (
        Params,
        Scanner,
        four_hour_bias,
        run as run_pb,
        value_area,
        week_of,
    )
    from tradingagents.pb_range.data import intraday_bars

    symbol = normalize_symbol(ticker)
    end = date or datetime.now().strftime("%Y-%m-%d")
    try:
        start = (datetime.strptime(end, "%Y-%m-%d") - timedelta(days=days)).strftime("%Y-%m-%d")
    except ValueError:
        console.print("[red]--date must be YYYY-MM-DD[/red]")
        raise typer.Exit(code=1) from None
    params = Params(risk_pct=risk, evening_hour=None if is_crypto_symbol(symbol) else Params.evening_hour)
    try:
        bars = intraday_bars(symbol, start, end)
    except (NoMarketDataError, VendorError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None

    last = bars["Date"].iloc[-1]
    area = value_area(week_of(bars, last), params.value_area)
    bias = four_hour_bias(bars)
    scanner = Scanner(bars, params, equity=equity, value_area=area, bias=bias)
    scanner.run()
    console.print(f"[bold]{symbol}[/bold] · {len(bars)} fifteen-minute bars {bars['Date'].iloc[0]:%Y-%m-%d %H:%M} to {last:%Y-%m-%d %H:%M} New York "
                  f"· 4-hour bias {bias}"
                  + (f" · week's value area {area.low:,.2f} / POC {area.poc:,.2f} / {area.high:,.2f}" if area else ""))

    zones = Table(title="P and B ranges")
    for column in ("Kind", "From", "Status", "Low", "High", "Height", "Touches", "Origin", "Impulse"):
        zones.add_column(column, justify="left" if column in ("Kind", "From", "Status") else "right")
    for z in scanner.zones:
        zones.add_row(z.kind, f"{bars['Date'].iloc[z.start]:%m-%d %H:%M}", z.status() + (f", {z.reclaims} fake" if z.reclaims else ""),
                      f"{z.low:,.2f}", f"{z.high:,.2f}", f"{z.height / z.atr:.1f} ATR",
                      f"{z.touches_top}+{z.touches_bottom}", f"{z.origin:,.2f}", f"{z.impulse:,.2f}")
    console.print(zones if scanner.zones else "No impulse followed by a range in these bars.")

    session = last.normalize()
    recent = [s for s in scanner.setups if s.at >= session] or scanner.setups[-3:]
    if recent:
        table = Table(title="Setups" + (" (last session)" if recent[0].at >= session else " (latest)"))
        for column in ("When", "Trade", "Side", "Entry", "Stop", "Targets", "R:R", "Units", "Risk", "Notes"):
            table.add_column(column, justify="left" if column in ("When", "Trade", "Side", "Notes") else "right")
        for s in recent:
            table.add_row(f"{s.at:%m-%d %H:%M}", s.kind, s.side, f"{s.entry:,.2f}", f"{s.stop:,.2f}",
                          " / ".join(f"{t:,.2f}" for t in s.targets), f"{s.rr:.1f}", f"{s.units:,d}",
                          f"{s.risk:,.0f}", "; ".join(s.notes))
        console.print(table)
    else:
        console.print("No setup: no valid range has offered an edge, a confirmed breakout or a fake breakout.")

    if backtest:
        result = run_pb(bars, params, equity=equity, value_area=area)
        stats = result.stats()
        if not stats["trades"]:
            console.print("Backtest: no order filled in this window.")
            return
        trades = Table(title="Backtest: every setup traded by the rules")
        for column in ("Entered", "Trade", "Side", "Entry", "Units", "Exits", "P&L", "R"):
            trades.add_column(column, justify="left" if column in ("Entered", "Trade", "Side", "Exits") else "right")
        for t in result.closed:
            exits = ", ".join(f"{f.reason} {f.price:,.2f}" for f in t.fills)
            trades.add_row(f"{t.entry_at:%m-%d %H:%M}", t.setup.kind, t.side, f"{t.entry:,.2f}", f"{t.units:,d}",
                           exits, f"{t.pnl:+,.0f}", f"{t.r_multiple:+.2f}")
        console.print(trades)
        console.print(f"{stats['trades']} trades · win rate {stats['win_rate']:.0%} · {stats['total_r']:+.1f}R in all, "
                      f"{stats['mean_r']:+.2f}R each · profit factor {stats['profit_factor']:.2f} · "
                      f"P&L {stats['pnl']:+,.0f} on {equity:,.0f} · worst run {stats['worst_losing_streak']} losses")


@app.command()
def worker(
    once: bool = typer.Option(False, "--once", help="One pass now, then stop, market open or not"),
    today: bool = typer.Option(False, "--today", help="Watch until today's close, then stop, instead of running for good"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Judge the news and log the orders it would send, without sending them"),
    interval: int = typer.Option(None, "--interval", help="Minutes between passes (default worker_interval_minutes)"),
):
    """Watch the held positions every market day: read their news and act within the guardrails.

    Runs for good, asleep outside market hours: the process for a container.
    """
    import logging
    from tradingagents.dataflows.vendors.alpaca.common import paper_trading
    from tradingagents.llm_clients.factory import create_tier_client
    from tradingagents.worker.notify import email_configured
    from tradingagents.worker.runner import default_duties, run_day, run_forever, run_pass

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = DEFAULT_CONFIG
    llm = create_tier_client(config, "quick").get_llm()
    mode = "paper" if paper_trading() else "LIVE"
    trading = ("off" if not config["worker_trading_enabled"] else
               "dry run" if dry_run else
               "on" if mode == "paper" or config["worker_allow_live"] else "blocked (live account)")
    console.print(f"Worker on the {mode} account · trading {trading} · "
                  f"email {'on' if email_configured() else 'off (set TRADINGAGENTS_SMTP_USER and _PASSWORD)'}")
    if once:
        findings = run_pass(config, llm, default_duties(), dry_run)
        for f in findings:
            order = f" -> {f.action.kind} {f.action.side} {f.action.qty} [{f.outcome}]" if f.action else ""
            console.print(f"[{f.severity}] {f.symbol}: {f.headline}{order}")
        if not findings:
            console.print("Nothing new about the positions watched.")
        return
    minutes = interval or config["worker_interval_minutes"]
    if today:
        run_day(config, llm, default_duties(), minutes, dry_run)
    else:
        run_forever(config, llm, default_duties(), minutes, dry_run)


@app.command()
def daily(
    date: str = typer.Option(None, "--date", help="Analysis date, YYYY-MM-DD (default: the last close)"),
    parts: int = typer.Option(None, "--parts", help="Processes to run at once (default daily_parts)"),
    equity: float = typer.Option(None, "--equity", help="Equity to size against (default: the account's)"),
):
    """The 4 AM job by hand: analyse the watchlist on the last close, size the book, bench the next-best."""
    import logging

    from tradingagents.daily import run_daily
    from tradingagents.dataflows.vendors.alpaca import trading

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from tradingagents.daily import describe_timing

    built = run_daily(DEFAULT_CONFIG, equity or trading.get_account().equity, date, parts)
    console.print(f"{len(built.book.orders)} orders, bench {len(built.bench)}; saved to {built.book_path}")
    console.print(f"Timed: {describe_timing(built.timing or {})} · {built.timing_path}")


@app.command()
def snapshot():
    """Record today's open positions for the positions page's day-by-day view."""
    from pathlib import Path

    from tradingagents.dataflows.vendors.alpaca import trading
    from tradingagents.web.snapshots import write_snapshot

    path = write_snapshot(Path(DEFAULT_CONFIG["results_dir"]), trading.get_account(), DEFAULT_CONFIG["default_stop_pct"])
    console.print(f"Snapshot written to {path}")


@app.command()
def web(
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8010, "--port"),
):
    """Serve the positions page: every open position, each linked to what the agents decided."""
    import uvicorn

    uvicorn.run("tradingagents.web.app:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    app()
