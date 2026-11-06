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

from tradingagents.dataflows.config import get_config
from tradingagents.web import data

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
</style>
</head>
<body>
{body}
</body>
</html>"""


def _results_dir() -> Path:
    return Path(get_config()["results_dir"])


def _default_stop() -> float:
    return float(get_config().get("default_stop_pct") or 0)


def _page(title: str, body: str, status_code: int = 200) -> HTMLResponse:
    return HTMLResponse(_PAGE.format(title=html.escape(title), body=body), status_code=status_code)


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


@app.get("/health")
def health():
    decisions = data.find_decisions(_results_dir())
    return {"ok": True, "decisions": len(decisions), "results_dir": str(_results_dir())}


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

    body = [f"<h1>Edinburgh positions</h1><p class=\"meta\">{account_line} · as of {data.now_text()}</p>"]
    body.append("<h2>Open positions</h2>")
    if rows:
        body.append('<div class="wrap"><table><tr><th>Ticker</th><th>Side</th><th>Qty</th><th>Value</th>'
                    '<th>Entry</th><th>Price</th><th>P&amp;L</th><th>Rating</th><th>Risk</th>'
                    '<th>Stop</th><th>To stop</th><th>Target</th><th>To target</th><th>Decided</th><th>Report</th></tr>')
        for r in rows:
            d = r["decision"] or {}
            stop_cls = ' class="dflt"' if str(d.get("stop_from", "")).startswith("default") else ""
            stop = (f'<span{stop_cls} title="{html.escape(d.get("stop_from", ""))}">{_num(d.get("stop"))}</span>'
                    if d.get("stop") else "—")
            report = (f'<a href="/report/{html.escape(r["ticker"])}/{r["decision_date"]}">agents</a>'
                      if r["decision_date"] else '<span class="none">none</span>')
            body.append(
                f'<tr><td><b>{html.escape(r["ticker"])}</b></td><td class="{r["side"]}">{r["side"]}</td>'
                f'<td>{_num(r["qty"], 0, signed=True)}</td><td>{_num(abs(r["market_value"]), 0)}</td>'
                f'<td>{_num(r["avg_entry_price"])}</td><td>{_num(r["price"])}</td><td>{_pct(r["pnl_pct"])}</td>'
                f'<td class="l">{html.escape(d.get("rating") or "—")}</td><td class="l">{html.escape(d.get("risk_level") or "—")}</td>'
                f'<td>{stop}</td><td>{_pct(r["to_stop_pct"])}</td><td>{_num(d.get("target"))}</td>'
                f'<td>{_pct(r["to_target_pct"])}</td><td class="l">{r["decision_date"] or "—"}</td><td class="l">{report}</td></tr>')
        body.append("</table></div><p class=\"meta\">A stop in italics is the 2% default: the desk gave none, "
                    "or one on the wrong side. \"To stop\" and \"to target\" are from the current price, "
                    "in the position's direction.</p>")
    else:
        body.append("<p>No open positions.</p>")

    book_path, orders = data.latest_book(_results_dir())
    body.append("<h2>The day's book</h2>")
    if orders:
        held = {r["ticker"] for r in rows}
        body.append(f'<p class="meta">From <code>{html.escape(book_path.name)}</code></p>')
        body.append('<div class="wrap"><table><tr><th>Ticker</th><th>Side</th><th>Rating</th><th>Risk</th>'
                    '<th>Shares</th><th>Price</th><th>Value</th><th>Stop</th><th>From</th><th>Target</th><th>Held</th><th>Report</th></tr>')
        for o in orders:
            if o["side"] not in ("long", "short"):
                body.append(f'<tr><td><b>{html.escape(o["ticker"])}</b></td><td class="l" colspan="7">{html.escape(o["side"])}</td>'
                            f'<td colspan="2">{html.escape(o["value"])}</td><td></td><td></td></tr>')
                continue
            body.append(
                f'<tr><td><b>{html.escape(o["ticker"])}</b></td><td class="{o["side"]}">{o["side"]}</td>'
                f'<td class="l">{html.escape(o["rating"])}</td><td class="l">{html.escape(o["risk_level"])}</td>'
                f'<td>{html.escape(o["shares"])}</td><td>{html.escape(o["price"])}</td><td>{html.escape(o["value"])}</td>'
                f'<td>{html.escape(o.get("stop") or "")}</td><td class="l">{html.escape(o.get("stop_from") or "")}</td>'
                f'<td>{html.escape(o.get("target") or "")}</td><td class="l">{"yes" if o["ticker"] in held else "no"}</td>'
                f'<td class="l"><a href="/report/{html.escape(o["ticker"])}">agents</a></td></tr>')
        body.append("</table></div>")
    else:
        body.append("<p>No sized book yet: run <code>tradingagents size</code>.</p>")
    return _page("Edinburgh positions", "\n".join(body))


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
