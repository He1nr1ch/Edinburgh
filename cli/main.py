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
def size(
    run_ids: list[str] = typer.Argument(..., help="Backtest run IDs to size together, e.g. sp100_20261023_t0_a sp100_20261023_t0_b"),
    date: str = typer.Option(..., "--date", help="The analysis date whose calls to size, YYYY-MM-DD"),
    equity: float = typer.Option(100_000, "--equity", help="Portfolio equity to size against"),
    gross: float = typer.Option(75.0, "--gross", help="Target gross exposure, longs plus shorts, as % of equity"),
    cap: float = typer.Option(10.0, "--cap", help="Most any one position may be, as % of equity"),
    hedge: bool = typer.Option(True, "--hedge/--no-hedge", help="Price SPY calls to cover a net-short book"),
    record_hedge: bool = typer.Option(False, "--record-hedge", help="Record the calls this plan buys as bought, so later runs do not hedge again"),
):
    """Size a list run's calls into a long/short book: shares and dollars per ticker, and its hedge."""
    import csv
    from pathlib import Path

    from rich.table import Table

    from tradingagents.sizing import last_closes, load_calls, size_book

    root = Path(DEFAULT_CONFIG["results_dir"]) / "backtest"
    calls = [c for run_id in run_ids for c in load_calls(root / run_id, date)]
    if not calls:
        console.print(f"[red]No calls for {date} under {', '.join(run_ids)} in {root}[/red]")
        raise typer.Exit(code=1)
    book = size_book(calls, last_closes([c.ticker for c in calls], date), equity, gross, cap)

    table = Table(title=f"Book for {equity:,.0f} on {date}")
    for column in ("Ticker", "Side", "Rating", "Risk", "Ceiling", "Weight", "Price", "Shares", "Value"):
        table.add_column(column, justify="right" if column not in ("Ticker", "Side", "Rating", "Risk") else "left")
    for o in book.orders:
        table.add_row(o.ticker, o.side, o.rating, o.risk_level or "", f"{o.ceiling_pct:g}%",
                      f"{o.weight_pct:+.2f}%", f"{o.price:,.2f}", f"{o.shares:+,d}", f"{o.value:,.0f}")
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
    for ticker, why in book.skipped:
        console.print(f"[dim]no position: {ticker}: {why}[/dim]")

    plan = None
    if hedge:
        from tradingagents.hedge import hedge_book, open_hedges
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

    out = root / "sizing" / f"{'+'.join(run_ids)}_{date}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ticker", "side", "rating", "risk_level", "ceiling_pct", "weight_pct", "price", "shares", "value"])
        for o in book.orders:
            writer.writerow([o.ticker, o.side, o.rating, o.risk_level or "", o.ceiling_pct,
                             round(o.weight_pct, 4), o.price, o.shares, round(o.value, 2)])
        if plan and plan.contracts:
            writer.writerow([plan.quote.symbol, f"call ({plan.choice} hedge)", "", "", "", "",
                             plan.quote.ask, plan.contracts, round(plan.cost, 2)])
    console.print(f"Orders saved to {out}")


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


if __name__ == "__main__":
    app()
