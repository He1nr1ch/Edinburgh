"""The positions page: every open position today, each linked to what the desk's agents decided.

A small read-only FastAPI app, in the shape of Dublin's logbook viewer. It
reads the account from Alpaca and the decisions from the results directory
the worker also uses, so on Colossus it runs as its own container beside the
worker with the same data folder mounted. Nothing here places an order.

Routes: ``/`` the positions and the day's book; ``/report/<ticker>`` the
newest decision for a ticker, ``/report/<ticker>/<date>`` a given day's;
``/api/positions`` and ``/api/report/<ticker>/<date>`` the same as JSON;
``/health``.

Local run: tradingagents web --port 8010
"""

from __future__ import annotations

import html
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from markdown_it import MarkdownIt

from tradingagents import build
from tradingagents.dataflows.config import get_config
from tradingagents.web import data, snapshots, transactions
from tradingagents.worker.state import WorkerState

app = FastAPI(title="Edinburgh positions", docs_url=None, redoc_url=None)

# Raw HTML in a model's text is not rendered: the text is the model's, not ours.
_md = MarkdownIt("commonmark", {"html": False, "linkify": False}).enable("table")

# Dark slate and amber, as Dublin's internal pages.
_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  body {{ background: #1e2129; color: #e8e8e8; font: 15px/1.55 system-ui, sans-serif;
         max-width: 1200px; margin: 0 auto; padding: 1.5rem 1rem; }}
  a {{ color: #ffc107; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  h1, h2, h3, h4 {{ color: #fff; line-height: 1.25; }}
  h2 {{ border-bottom: 1px solid #3a3f4b; padding-bottom: .3rem; margin-top: 2rem; }}
  .meta {{ color: #9aa3b2; font-size: .9rem; }}
  table {{ border-collapse: collapse; margin: 1rem 0; width: 100%; font-variant-numeric: tabular-nums; }}
  .wrap {{ overflow-x: auto; }}
  th, td {{ border: 1px solid #3a3f4b; padding: .3rem .6rem; white-space: nowrap; text-align: right; }}
  th:first-child, td:first-child, td.l, th.l {{ text-align: left; }}
  th {{ background: #2a2f3a; }}
  tr:hover td {{ background: #262b36; }}
  .long {{ color: #7ed67e; }} .short {{ color: #f28b82; }} .none {{ color: #9aa3b2; }}
  .pos {{ color: #7ed67e; }} .neg {{ color: #f28b82; }}
  .dflt {{ color: #9aa3b2; font-style: italic; }}
  .agent {{ background: #242833; border: 1px solid #3a3f4b; border-radius: 6px; padding: .2rem 1rem 1rem; margin: 1rem 0; }}
  .agent h3 {{ margin: .8rem 0 .4rem; color: #ffc107; }}
  .agent table {{ width: auto; }}
  .agent td, .agent th {{ white-space: normal; text-align: left; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: .6rem; margin: 1rem 0; }}
  .card {{ background: #242833; border: 1px solid #3a3f4b; border-radius: 6px; padding: .5rem .8rem; }}
  .card .k {{ color: #9aa3b2; font-size: .8rem; }} .card .v {{ font-size: 1.15rem; }}
  code {{ background: #2a2f3a; padding: .1rem .3rem; border-radius: 3px; }}
  footer {{ margin-top: 3rem; color: #9aa3b2; font-size: .8rem; border-top: 1px solid #3a3f4b; padding-top: .6rem; }}
</style>
</head>
<body>
{body}
<footer>Edinburgh build {build}</footer>
</body>
</html>"""


def _results_dir() -> Path:
    return Path(get_config()["results_dir"])


def _default_stop() -> float:
    return float(get_config().get("default_stop_pct") or 0)


def _page(title: str, body: str, status_code: int = 200) -> HTMLResponse:
    return HTMLResponse(_PAGE.format(title=html.escape(title), body=body, build=html.escape(build.describe())),
                        status_code=status_code)


def _num(value, digits=2, suffix="", signed=False) -> str:
    if value is None:
        return "—"
    text = f"{value:+,.{digits}f}" if signed else f"{value:,.{digits}f}"
    return text + suffix


def _pct(value) -> str:
    if value is None:
        return "—"
    cls = "pos" if value > 0 else "neg" if value < 0 else ""
    return f'<span class="{cls}">{value:+.1f}%</span>'


def _holdings():
    from tradingagents.dataflows.vendors.alpaca import trading

    return trading.get_account()


def _positions_table(rows: list[dict]) -> str:
    out = ['<div class="wrap"><table><tr><th>Ticker</th><th>Side</th><th>Qty</th><th>Value</th>'
           '<th>Entry</th><th>Price</th><th>P&amp;L</th><th>Rating</th><th>Risk</th>'
           '<th>Stop</th><th>To stop</th><th>Target</th><th>To target</th><th>Decided</th><th>Report</th></tr>']
    for r in rows:
        d = r["decision"] or {}
        stop_cls = ' class="dflt"' if str(d.get("stop_from", "")).startswith("default") else ""
        stop = (f'<span{stop_cls} title="{html.escape(str(d.get("stop_from", "")))}">{_num(d.get("stop"))}</span>'
                if d.get("stop") else "—")
        report = (f'<a href="/report/{html.escape(r["ticker"])}/{r["decision_date"]}">agents</a>'
                  if r.get("decision_date") else '<span class="none">none</span>')
        out.append(
            f'<tr><td><b>{html.escape(r["ticker"])}</b></td><td class="{r["side"]}">{r["side"]}</td>'
            f'<td>{_num(r["qty"], 0, signed=True)}</td><td>{_num(abs(r["market_value"]), 0)}</td>'
            f'<td>{_num(r["avg_entry_price"])}</td><td>{_num(r["price"])}</td><td>{_pct(r["pnl_pct"])}</td>'
            f'<td class="l">{html.escape(d.get("rating") or "—")}</td><td class="l">{html.escape(d.get("risk_level") or "—")}</td>'
            f'<td>{stop}</td><td>{_pct(r["to_stop_pct"])}</td><td>{_num(d.get("target"))}</td>'
            f'<td>{_pct(r["to_target_pct"])}</td><td class="l">{r.get("decision_date") or "—"}</td><td class="l">{report}</td></tr>')
    out.append("</table></div><p class=\"meta\">A stop in italics is the 2% default: the desk gave none, "
               "or one on the wrong side. \"To stop\" and \"to target\" are from the price shown, "
               "in the position's direction.</p>")
    return "\n".join(out)


def _book_table(orders: list[dict], held: set[str]) -> str:
    out = ['<div class="wrap"><table><tr><th>Ticker</th><th>Side</th><th>Rating</th><th>Risk</th>'
           '<th>Shares</th><th>Price</th><th>Value</th><th>Stop</th><th>From</th><th>Target</th><th>Held</th><th>Report</th></tr>']
    for o in orders:
        if o["side"] not in ("long", "short"):
            out.append(f'<tr><td><b>{html.escape(o["ticker"])}</b></td><td class="l" colspan="7">{html.escape(o["side"])}</td>'
                       f'<td colspan="2">{html.escape(o["value"])}</td><td></td><td></td></tr>')
            continue
        out.append(
            f'<tr><td><b>{html.escape(o["ticker"])}</b></td><td class="{o["side"]}">{o["side"]}</td>'
            f'<td class="l">{html.escape(o["rating"])}</td><td class="l">{html.escape(o["risk_level"])}</td>'
            f'<td>{html.escape(o["shares"])}</td><td>{html.escape(o["price"])}</td><td>{html.escape(o["value"])}</td>'
            f'<td>{html.escape(o.get("stop") or "")}</td><td class="l">{html.escape(o.get("stop_from") or "")}</td>'
            f'<td>{html.escape(o.get("target") or "")}</td><td class="l">{"yes" if o["ticker"] in held else "no"}</td>'
            f'<td class="l"><a href="/report/{html.escape(o["ticker"])}">agents</a></td></tr>')
    out.append("</table></div>")
    return "\n".join(out)


def _transactions_rows(day: str) -> list[dict]:
    state = WorkerState.load(get_config()["worker_state_path"])
    try:
        from tradingagents.dataflows.vendors.alpaca import trading

        fills = trading.fills_on(day) if state.orders_on(day) else {}
    except Exception:
        fills = {}
    return transactions.transactions_on(state, day, fills)


def _transactions_table(rows: list[dict]) -> str:
    if not rows:
        return "<p>No orders were sent.</p>"
    out = ['<div class="wrap"><table><tr><th>Time</th><th>Action</th><th>Ticker</th><th>Side</th><th>Shares</th>'
           '<th>Fill</th><th>Value</th><th class="l">Why</th><th>Report</th></tr>']
    for r in rows:
        cls = "neg" if r["flow"] == "out" else "pos"
        out.append(
            f'<tr><td class="l">{r["time"]}</td><td class="l {cls}">{html.escape(r["action"])}</td>'
            f'<td><b>{html.escape(r["ticker"])}</b></td><td class="{r["position"]}">{r["position"]}</td>'
            f'<td>{r["qty"]:,}</td><td>{_num(r["price"])}</td><td>{_num(r["value"], 0)}</td>'
            f'<td class="l" style="white-space:normal;max-width:48rem">{html.escape(r["reason"])}</td>'
            f'<td class="l"><a href="/report/{html.escape(r["ticker"])}">agents</a></td></tr>')
    out.append("</table></div>")
    return "\n".join(out)


@app.get("/transactions", response_class=HTMLResponse)
def transactions_index():
    state = WorkerState.load(get_config()["worker_state_path"])
    days = transactions.days_with_transactions(state)
    body = ['<p class="meta"><a href="/">← positions</a></p><h1>Transactions</h1>']
    if not days:
        body.append("<p>No orders sent yet.</p>")
    else:
        body.append('<div class="wrap"><table><tr><th>Day</th><th>Exits</th><th>Entries</th></tr>')
        body += [f'<tr><td class="l"><a href="/transactions/{d}">{d}</a></td><td>{outs}</td><td>{ins}</td></tr>'
                 for d, outs, ins in days]
        body.append("</table></div>")
    return _page("Transactions", "\n".join(body))


@app.get("/transactions/{day}", response_class=HTMLResponse)
def transactions_day(day: str):
    rows = _transactions_rows(day)
    outs, ins = sum(r["flow"] == "out" for r in rows), sum(r["flow"] == "in" for r in rows)
    body = [f'<p class="meta"><a href="/transactions">← transactions</a> · <a href="/day/{html.escape(day)}">positions that day</a></p>',
            f"<h1>Transactions on {html.escape(day)}</h1>",
            f'<p class="meta">{outs} exit(s), {ins} entr{"y" if ins == 1 else "ies"}. Fill prices are Alpaca\'s; '
            "the reason is the Watcher's own record of why it sent the order.</p>",
            _transactions_table(rows)]
    return _page(f"Transactions {day}", "\n".join(body))


@app.get("/health")
def health():
    decisions = data.find_decisions(_results_dir())
    return {"ok": True, "build": build.BUILD_VERSION, "git_sha": build.git_sha(), "decisions": len(decisions),
            "results_dir": str(_results_dir())}


@app.get("/api/positions")
def api_positions():
    account = _holdings()
    rows = data.positions_view(account.holdings, data.find_decisions(_results_dir()), _default_stop())
    return {"as_of": data.now_text(), "paper": account.paper, "equity": account.equity,
            "cash": account.cash, "positions": rows}


@app.get("/api/report/{ticker}/{date}")
def api_report(ticker: str, date: str):
    ref = _find(ticker, date)
    state = data.load_decision(ref)
    return {"ticker": ref.ticker, "date": ref.date, "run": ref.run,
            "summary": data.decision_summary(state, None, _default_stop()),
            "agents": [{"agent": name, "text": text} for name, text in data.agent_outputs(state)]}


def _find(ticker: str, date: str | None) -> data.DecisionRef:
    decisions = data.find_decisions(_results_dir())
    ticker = ticker.upper()
    if date is None:
        ref = decisions.get(ticker)
        if ref is None:
            raise HTTPException(404, f"no decision saved for {ticker}")
        return ref
    for pattern in (f"backtest/*/{ticker}/TradingAgentsStrategy_logs/full_states_log_{date}.json",
                    f"{ticker}/TradingAgentsStrategy_logs/full_states_log_{date}.json"):
        for path in sorted(_results_dir().glob(pattern)):
            run = path.parts[-4] if "backtest" in path.parts else "single"
            return data.DecisionRef(ticker, date, path, run)
    raise HTTPException(404, f"no decision saved for {ticker} on {date}")


@app.get("/", response_class=HTMLResponse)
def index():
    try:
        account = _holdings()
        rows = data.positions_view(account.holdings, data.find_decisions(_results_dir()), _default_stop())
        account_line = (f"{'Paper' if account.paper else 'LIVE'} account · equity {account.equity:,.0f} · "
                        f"cash {account.cash:,.0f} · {len(rows)} open position(s)")
    except Exception as exc:   # the page must open even when Alpaca does not answer
        rows, account_line = [], f"Alpaca did not answer: {html.escape(str(exc))}"

    days = snapshots.list_days(_results_dir())
    today = days[0] if days and days[0]["date"] == data.now_text()[:10] else None
    recorded = (f' · today recorded at {today["as_of"][11:16]}' if today else " · no record of today yet")
    body = [f"<h1>Edinburgh positions</h1><p class=\"meta\">{account_line} · as of {data.now_text()}{recorded} · "
            f'<a href="/days">past days</a> · <a href="/transactions">transactions</a></p>']
    body.append("<h2>Open positions</h2>")
    body.append(_positions_table(rows) if rows else "<p>No open positions.</p>")

    today_rows = _transactions_rows(data.now_text()[:10])
    body.append(f'<h2>Today\'s transactions <span class="meta">· <a href="/transactions">all days</a></span></h2>')
    body.append(_transactions_table(today_rows))

    book_path, orders = data.latest_book(_results_dir())
    body.append("<h2>The day's book</h2>")
    if orders:
        body.append(f'<p class="meta">From <code>{html.escape(book_path.name)}</code></p>')
        body.append(_book_table(orders, {r["ticker"] for r in rows}))
    else:
        body.append("<p>No sized book yet: run <code>tradingagents size</code>.</p>")

    from tradingagents.book import latest_bench

    bench_path, bench = latest_bench(_results_dir())
    body.append("<h2>Waiting in the wings</h2>")
    if bench:
        body.append(f'<p class="meta">The bench, best score first, from <code>{html.escape(bench_path.name)}</code>. '
                    "Score = reward to risk × conviction × stance × freshness × balance; the Watcher re-scores "
                    "at live prices when a slot opens.</p>")
        body.append('<div class="wrap"><table><tr><th>#</th><th>Ticker</th><th>Side</th><th>Rating</th><th>Risk</th>'
                    '<th>Decided</th><th>Price</th><th>Stop</th><th>Target</th><th>Move</th><th>Score</th><th>Report</th></tr>')
        for i, b in enumerate(bench, 1):
            body.append(
                f'<tr><td>{i}</td><td><b>{html.escape(b["ticker"])}</b></td><td class="{b["side"]}">{b["side"]}</td>'
                f'<td class="l">{html.escape(b["rating"])}</td><td class="l">{html.escape(b.get("risk_level") or "")}</td>'
                f'<td class="l">{html.escape(b.get("decided") or "")}</td><td>{html.escape(b["price"])}</td>'
                f'<td>{html.escape(b.get("stop") or "")}</td><td>{html.escape(b.get("target") or "")}</td>'
                f'<td>{html.escape(b.get("expected_move_pct") or "")}%</td>'
                f'<td title="{html.escape(b.get("score_parts") or "")}"><b>{html.escape(b.get("score") or "")}</b></td>'
                f'<td class="l"><a href="/report/{html.escape(b["ticker"])}">agents</a></td></tr>')
        body.append("</table></div>")
    else:
        body.append("<p>No bench yet.</p>")
    return _page("Edinburgh positions", "\n".join(body))


@app.get("/days", response_class=HTMLResponse)
def days():
    rows = snapshots.list_days(_results_dir())
    body = ['<p class="meta"><a href="/">← positions</a></p><h1>Past days</h1>']
    if not rows:
        body.append("<p>No days recorded yet. The worker records one on every pass; "
                    "<code>tradingagents snapshot</code> records one now.</p>")
    else:
        body.append('<div class="wrap"><table><tr><th>Day</th><th>Recorded at</th><th>Positions</th><th>Equity</th><th>Account</th></tr>')
        for d in rows:
            body.append(f'<tr><td class="l"><a href="/day/{d["date"]}">{d["date"]}</a></td><td class="l">{html.escape(d["as_of"][11:16])}</td>'
                        f'<td>{d["positions"]}</td><td>{_num(d["equity"], 0)}</td>'
                        f'<td class="l">{"paper" if d["paper"] else "LIVE"}</td></tr>')
        body.append("</table></div>")
    return _page("Past days", "\n".join(body))


@app.get("/day/{day}", response_class=HTMLResponse)
def day(day: str):
    record = snapshots.read_day(_results_dir(), day)
    if record is None:
        raise HTTPException(404, f"no record of {day}")
    rows = record.get("positions", [])
    body = [f'<p class="meta"><a href="/days">← past days</a> · <a href="/">today</a></p>',
            f"<h1>On {html.escape(day)}</h1>",
            f'<p class="meta">Recorded at {html.escape(record.get("as_of", "")[11:16])} · '
            f'{"paper" if record.get("paper") else "LIVE"} account · equity {_num(record.get("equity"), 0)} · '
            f'cash {_num(record.get("cash"), 0)} · {len(rows)} position(s)</p>']
    body.append("<h2>Positions held</h2>")
    body.append(_positions_table(rows) if rows else "<p>No positions were held.</p>")
    body.append(f'<h2>Transactions that day <span class="meta">· <a href="/transactions/{html.escape(day)}">full page</a></span></h2>')
    body.append(_transactions_table(_transactions_rows(day)))
    book_path, orders = data.latest_book(_results_dir(), day)
    if orders:
        body.append(f"<h2>The book sized for {html.escape(day)}</h2>")
        body.append(f'<p class="meta">From <code>{html.escape(book_path.name)}</code></p>')
        body.append(_book_table(orders, {r["ticker"] for r in rows}))
    return _page(f"On {day}", "\n".join(body))


@app.get("/report/{ticker}", response_class=HTMLResponse)
def report_latest(ticker: str):
    return _report(_find(ticker, None))


@app.get("/report/{ticker}/{date}", response_class=HTMLResponse)
def report(ticker: str, date: str):
    return _report(_find(ticker, date))


def _report(ref: data.DecisionRef) -> HTMLResponse:
    state = data.load_decision(ref)
    summary = data.decision_summary(state, None, _default_stop())
    cards = [("Rating", summary["rating"] or "—"), ("Side", summary["side"]), ("Risk", summary["risk_level"] or "—"),
             ("Size", f'{summary["size_pct"]:g}% of equity' if summary["size_pct"] is not None else "—"),
             ("Ceiling", f'{summary["ceiling_pct"]:g}%' if summary["ceiling_pct"] is not None else "—"),
             ("Stop", f'{_num(summary["stop"])} <span class="meta">{html.escape(summary["stop_from"])}</span>' if summary["stop"] else "—"),
             ("Target", _num(summary["target"])), ("Horizon", summary["horizon"] or "—")]
    body = [f'<p class="meta"><a href="/">← positions</a></p>',
            f"<h1>{html.escape(ref.ticker)} · {ref.date}</h1>",
            f'<p class="meta">run <code>{html.escape(ref.run)}</code> · file <code>{html.escape(ref.path.name)}</code></p>',
            '<div class="cards">' + "".join(f'<div class="card"><div class="k">{k}</div><div class="v">{v}</div></div>'
                                            for k, v in cards) + "</div>"]
    if summary["summary"]:
        body.append(f"<p><b>Executive summary.</b> {html.escape(summary['summary'])}</p>")
    body.append("<h2>What each agent decided</h2>")
    for name, text in data.agent_outputs(state):
        body.append(f'<div class="agent"><h3>{html.escape(name)}</h3>{_md.render(text)}</div>')
    return _page(f"{ref.ticker} {ref.date}", "\n".join(body))


@app.exception_handler(HTTPException)
async def _not_found(_, exc: HTTPException):
    if exc.status_code == 404:
        return _page("Not found", f'<p class="meta"><a href="/">← positions</a></p><h1>Not found</h1><p>{html.escape(str(exc.detail))}</p>',
                     status_code=404)
    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
