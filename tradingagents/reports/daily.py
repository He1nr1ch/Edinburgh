"""The daily report: what the desk held, did and learned today, reviewed by the deep model.

After the close the worker gathers the day from the desk's own records (the
positions snapshot, the transactions with their results, the Watcher's
findings, the book, the sweep's coverage, the feedback scorecard), has the
deep-tier model review it in the voice of the desk's reflections, and saves
``reports/daily/<date>.md`` and ``.json``. The review's lessons go to the
lessons store, where every agent and the Watcher read them before
their next call. ``tradingagents report --date`` runs it by hand.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from tradingagents.agents.schemas import Envelope, envelope_lines
from tradingagents.agents.structured import NO_EXTERNAL_TOOLS, bind_structured, decide
from tradingagents.feedback.store import keep_desk_verdict
from tradingagents.lessons import AGENTS, Lessons

logger = logging.getLogger(__name__)

AgentName = Literal["Market Analyst", "News Analyst", "Fundamentals Analyst", "Bull Researcher", "Bear Researcher",
                    "Research Manager", "Trader",
                    "Aggressive Risk Analyst", "Conservative Risk Analyst", "Neutral Risk Analyst",
                    "Risk Manager", "Portfolio Manager", "Watcher", "Desk"]


class Lesson(BaseModel):
    agent: AgentName = Field(description="Who should act on it: the agent whose part of the job it concerns, or Desk for a rule or a parameter.")
    lesson: str = Field(description="One sentence an agent can apply next time, specific to how it decides.")
    evidence: str = Field(description="The fact from today that taught it: a ticker, a level, a number.")


class DailyReview(Envelope):
    """The day's review. Its confidence is how far one day's evidence supports its lessons."""

    what_happened: str = Field(description="Two to four sentences: the day's result and what drove it, with the numbers.")
    what_worked: str = Field(description="One to three sentences on the decisions and guards that did their job today, with evidence.")
    what_did_not: str = Field(description="One to three sentences on what cost money or missed, with evidence; say so plainly if nothing did.")
    lessons: list[Lesson] = Field(description="Up to five lessons, each addressed to one agent or the Desk. Fewer is better than padding.")
    rule_suggestions: list[str] = Field(description="Up to three changes to the desk's rules or parameters the day argues for, each with the evidence; empty when none.")

    def thesis(self) -> str:
        return self.what_happened


# --- the facts ---------------------------------------------------------------

def _cause(reason: str) -> str:
    r = reason.lower()
    if "through the stop" in r:
        return "stop"
    if "through the target" in r:
        return "target"
    if r.startswith("review on"):
        return "review"
    if "replaces " in r:
        return "replacement"
    if "the day's book" in r:
        return "book"
    if "not accounted for" in r or "accounted for" in r:
        return "news"
    return "other"


def gather(config: dict, day: str) -> dict:
    """Everything the review reads, from the desk's own records; nothing is asked of a model here."""
    from tradingagents.book import load_rows
    from tradingagents.feedback import FeedbackStore
    from tradingagents.web import snapshots
    from tradingagents.web.transactions import realised, realised_r, transactions_on
    from tradingagents.worker.state import WorkerState

    results = Path(config["results_dir"])
    state = WorkerState.load(config["worker_state_path"])
    snap = snapshots.read_day(results, day) or {}
    prior = next((snapshots.read_day(results, d["date"]) for d in snapshots.list_days(results) if d["date"] < day), None) or {}

    cache: dict[str, dict] = {}

    def fills_for(d: str) -> dict:
        if d not in cache:
            try:
                from tradingagents.dataflows.vendors.alpaca import trading

                cache[d] = trading.fills_on(d)
            except Exception:
                cache[d] = {}
        return cache[d]

    txns = transactions_on(state, day, fills_for(day) if state.orders_on(day) else {}, fills_for, config=config)
    for t in txns:
        t["cause"] = _cause(t["reason"])
    exits = [t for t in txns if t["flow"] == "out"]
    positions = snap.get("positions", [])
    unrealised = sum((p["pnl_pct"] or 0) / 100 * abs(p["market_value"]) for p in positions if p.get("pnl_pct") is not None)

    log = results / "worker" / f"{day}.log"
    findings = []
    if log.is_file():
        findings = [line.strip() for line in log.read_text(encoding="utf-8").splitlines()
                    if "[urgent]" in line or "[notable]" in line]

    day_rec = state.days.get(day, {})
    book_rows = load_rows(Path(day_rec["book_path"])) if day_rec.get("book_path") and Path(day_rec["book_path"]).is_file() else []
    held = {p["ticker"] for p in positions}

    sweep_path = Path(config.get("sweep_state_path", ""))
    sweep = json.loads(sweep_path.read_text(encoding="utf-8")) if sweep_path.is_file() else {}
    from tradingagents import usage as usage_log
    from tradingagents.sweep import progress as sweep_progress

    try:
        spend = usage_log.summarize(usage_log.read_day(day, config))
    except Exception as exc:
        logger.warning("The usage log could not be read: %s", exc)
        spend = {}
    try:
        from tradingagents import errors as error_log

        model_errors = error_log.summarize(error_log.read_day(day, config))
    except Exception as exc:
        logger.warning("The error log could not be read: %s", exc)
        model_errors = {}
    # The sweep dates a decision by the close it analyses, the last one before
    # the session, so none carries the day's own date until after its close.
    # Counted on that date: counted on the day's, the figure was always 0 and
    # read as a sweep that decides nothing.
    decided_on = sweep.get("date") or day
    decided_today = len(list(results.glob(f"backtest/*/*/TradingAgentsStrategy_logs/full_states_log_{decided_on}.json")))

    try:
        store = FeedbackStore(config)
        card, decisions_r = store.scorecard(), store.r_record()
    except Exception as exc:
        logger.warning("The feedback scorecard could not be read: %s", exc)
        card, decisions_r = {}, {}

    return {
        "day": day,
        "build": day_rec.get("build", ""),
        "equity": snap.get("equity"), "cash": snap.get("cash"), "prior_equity": prior.get("equity"),
        "equity_change": (snap.get("equity") - prior.get("equity")) if snap.get("equity") and prior.get("equity") else None,
        "realised": realised(txns), "unrealised": round(unrealised, 2) if positions else None,
        # The day in R: each exit's result over the risk to the stop it was opened with.
        "r": realised_r(txns),
        "r_by_side": {side: s for side in ("long", "short") if (s := realised_r([t for t in txns if t["position"] == side]))},
        "decisions_r": decisions_r,
        "positions": positions, "transactions": txns,
        "exits_by_cause": dict(Counter(t["cause"] for t in exits)),
        "entries": sum(1 for t in txns if t["flow"] == "in"),
        "findings": findings,
        "book": book_rows, "book_held": sum(1 for r in book_rows if r["ticker"] in held),
        "sweep": {"cycle": sweep.get("cycle"), "date": sweep.get("date"), "queued": len(sweep.get("queue", [])),
                  "analysed_this_cycle": sweep.get("analysed_this_cycle"), "decided_today": decided_today, "decided_on": decided_on,
                  "progress": sweep_progress(sweep)},
        "usage": {"total": spend.get("total", {}), "by_agent": spend.get("by_agent", {}), "by_host": spend.get("by_host", {})},
        "errors": model_errors,
        "scorecard": card,
    }


# --- the review --------------------------------------------------------------

def _facts_text(f: dict) -> str:
    lines = [f"Day: {f['day']}" + (f"; build {f['build']}" if f["build"] else "")]
    if f["equity"] is not None:
        change = f" ({f['equity_change']:+,.0f} on the day)" if f["equity_change"] is not None else ""
        lines.append(f"Equity {f['equity']:,.0f}{change}; cash {f['cash']:,.0f}; realised today "
                     f"{f['realised']:+,.2f}" if f["realised"] is not None else
                     f"Equity {f['equity']:,.0f}{change}; cash {f['cash']:,.0f}; nothing realised today")
        if f["unrealised"] is not None:
            lines.append(f"Unrealised across {len(f['positions'])} open positions: {f['unrealised']:+,.0f}")
    if f.get("r"):
        lines.append(_r_line(f) + ". R is an exit's result over the risk it was opened with (entry to stop): "
                     "a stop-out is about -1R, and the mean R per exit is the desk's edge")
    lines.append(f"Entries {f['entries']}; exits by cause {f['exits_by_cause'] or 'none'}")
    if f["book"]:
        lines.append(f"Book this morning: {len(f['book'])} orders, {f['book_held']} held at the close")
    if f["transactions"]:
        lines.append("Transactions:")
        for t in f["transactions"]:
            in_r = f", {t['r']:+.2f}R on a {t['risk_pct']:.1f}% stop" if t.get("r") is not None else ""
            res = f", P/L {t['pnl']:+,.2f} ({t['pnl_pct']:+.1f}%{in_r})" if t["pnl"] is not None else ""
            lines.append(f"- {t['time']} {t['action']} {t['position']} {t['ticker']} {t['qty']}{res}: {t['reason'][:220]}")
    if f["positions"]:
        lines.append("Open positions at the close (P/L %, distance to stop, distance to target, rating):")
        for p in sorted(f["positions"], key=lambda p: p.get("pnl_pct") or 0):
            d = p.get("decision") or {}
            lines.append(f"- {p['side']} {p['ticker']}: {p['pnl_pct']:+.1f}%" if p.get("pnl_pct") is not None else f"- {p['side']} {p['ticker']}:"
                         + (f", stop {p['to_stop_pct']:+.1f}% away" if p.get("to_stop_pct") is not None else "")
                         + (f", target {p['to_target_pct']:+.1f}% away" if p.get("to_target_pct") is not None else "")
                         + (f", {d.get('rating')} ({d.get('stop_from')})" if d else ", no decision"))
    if f["findings"]:
        lines.append("Watcher findings (urgent and notable):")
        lines += [f"- {x[:220]}" for x in f["findings"][:25]]
    s = f["sweep"]
    if s.get("cycle"):
        lines.append(f"Sweep: cycle {s['cycle']} on {s['date']}'s close, {s['analysed_this_cycle']} analysed this cycle, "
                     f"{s['queued']} queued; {s['decided_today']} decisions on record for {s.get('decided_on') or s['date']}'s close "
                     "(the sweep dates a decision by the close it analyses, so none is dated today until tonight)"
                     + (f"; pace {s['progress']['minutes_per_ticker']:.1f} min per ticker" if (s.get('progress') or {}).get('minutes_per_ticker') else ""))
    u = f.get("usage") or {}
    if (u.get("total") or {}).get("calls"):
        from tradingagents.usage import describe as describe_usage

        lines.append("Model usage today: " + describe_usage(u) + "; by agent (minutes): "
                     + ", ".join(f"{a} {r['seconds'] / 60:.0f}" for a, r in list(u["by_agent"].items())[:9]))
    e = f.get("errors") or {}
    if e.get("total") or e.get("failed_tickers"):
        from tradingagents.errors import describe as describe_errors

        lines.append("Model errors today: " + describe_errors(e)
                     + ("; by agent: " + ", ".join(f"{a} {n}" for a, n in list(e["by_agent"].items())[:9]) if e.get("by_agent") else "")
                     + ". A contract error is an answer that failed its schema twice (the run closed as REVIEW); "
                       "a timeout or unavailable is no answer at all (the run failed and was queued again).")
    if f["scorecard"]:
        lines.append("Feedback scorecard over settled decisions (agent: right/judged, mean return on its calls):")
        for agent, st in f["scorecard"].items():
            if st.get("n"):
                mean = f", {st['mean']:+.1%}" if st.get("mean") is not None else ""
                lines.append(f"- {agent}: {st['hits']}/{st['n']}{mean}")
    if f.get("decisions_r"):
        lines.append("Settled decisions in R (each call trading its own stop and target over its window): " + _decisions_r_text(f["decisions_r"]))
    return "\n".join(lines)


def _r_line(f: dict) -> str:
    """The day's exits in R, with each side's share."""
    from tradingagents.web.transactions import r_text

    sides = "; ".join(f"{side} {s['total']:+.2f}R over {s['n']}" for side, s in (f.get("r_by_side") or {}).items())
    return f"In R today: {r_text(f['r'])}" + (f" ({sides})" if sides else "")


def _decisions_r_text(record: dict) -> str:
    return "; ".join(f"{name} {s['mean']:+.2f}R each over {s['n']}" for name, s in record.items())


def _prompt(facts: dict, prior_lessons: str) -> str:
    return (
        "You are the desk's end-of-day reviewer on a long/short equity desk run by agents: three analysts "
        "(Market, News, Fundamentals), a Bull and a Bear Researcher who argue each side from the analysts' reports, "
        "a Research Manager who settles their debate, a Trader who sets "
        "entry and stop, three Risk Analysts (Aggressive, Conservative, Neutral) who argue the Trader's plan from "
        "their fixed stances, a Risk Manager who settles that debate and sets the stance, the position ceiling and the stop, a Portfolio "
        "Manager who sets the rating, size and target, and a Watcher that guards positions through the day "
        "(stop and target guards, news judged against the desk's case, replacements from a scored bench). "
        "The desk's rules: Sell means short; a call needs a target over 5% away; the morning book opens with "
        "the top ten each side by expected move, but after that the most attractive call wins regardless of "
        "side (the desk is here to make money, not to stay balanced; an uneven book is not a fault); a missing "
        "stop defaults to 2%; the sweep analyses the index around the clock and the book is rebuilt each "
        "morning from decisions under five days old.\n\n"
        f"Today's record:\n{_facts_text(facts)}\n\n"
        + (f"Lessons already on record (do not repeat them; refine or contradict them if today's evidence warrants):\n{prior_lessons}\n\n" if prior_lessons else "")
        + "Review the day as the desk's own reflections do: specific, terse, every figure cited, no praise and no "
        "padding. Address each lesson to the one agent whose decision it would change, or to the Desk when it is "
        "a rule or a parameter; a lesson must be something that agent can do differently next time, stated as an "
        "instruction. A lesson for a Risk Analyst sharpens how it argues its own side (what to weigh, what it "
        "overstated or missed); it never asks it to give up its stance, since the debate needs all three. "
        "A one-day window proves little about a thesis, so lessons about direction need more than "
        "one day's move; lessons about stops, targets, sizing and the Watcher's calls can rest on today. "
        + NO_EXTERNAL_TOOLS
    )


def review(config: dict, facts: dict, llm=None) -> DailyReview | None:
    """The deep model's review of the day; None when it does not answer in form."""
    from tradingagents.llm_clients.factory import create_tier_client

    llm = llm or create_tier_client(config, "deep").get_llm()
    lessons = Lessons(config["lessons_path"])
    prior = "\n".join(f"- ({l.get('since') or l.get('date')}) {l['agent']}: {l['lesson']}"
                      for l in (lessons.standing + lessons.daily[-10:]))
    rev, verdict = decide(bind_structured(llm, DailyReview, "Daily review"), _prompt(facts, prior), "Daily review")
    keep_desk_verdict(config, verdict, "daily_review", facts["day"],
                      action=f"{len(rev.lessons)} lesson(s)" if rev else "none")
    return rev


# --- the report --------------------------------------------------------------

def render(facts: dict, rev: DailyReview | None) -> str:
    f = facts
    out = [f"# Edinburgh daily report, {f['day']}", ""]
    if f["build"]:
        out.append(f"*Build {f['build']}*  ")
    if f["equity"] is not None:
        change = f" ({f['equity_change']:+,.0f})" if f["equity_change"] is not None else ""
        out.append(f"**Equity** {f['equity']:,.0f}{change} · **cash** {f['cash']:,.0f} · "
                   f"**realised** {f['realised']:+,.2f}" if f["realised"] is not None else
                   f"**Equity** {f['equity']:,.0f}{change} · **cash** {f['cash']:,.0f} · nothing realised")
        if f["unrealised"] is not None:
            out.append(f"**Unrealised** {f['unrealised']:+,.0f} across {len(f['positions'])} positions")
    if f.get("r"):
        out.append("  \n**" + _r_line(f).replace("In R today:", "In R today**", 1)
                   + " · *R is an exit's result over the risk to the stop it was opened with*")
    out.append("")
    if rev:
        out += ["## What happened", rev.what_happened, "", "## What worked", rev.what_worked, "",
                "## What did not", rev.what_did_not, ""]
        if envelope_lines(rev):
            out += ["## How sure the review is", *envelope_lines(rev)[1:], ""]
        if rev.lessons:
            out += ["## Lessons"] + [f"- **{l.agent}**: {l.lesson} *({l.evidence})*" for l in rev.lessons] + [""]
        if rev.rule_suggestions:
            out += ["## Rule suggestions (for you to decide, not applied)"] + [f"- {s}" for s in rev.rule_suggestions] + [""]
    else:
        out += ["## Review", "The model did not return a review in form; the facts below stand on their own.", ""]
    out.append("## Transactions")
    if f["transactions"]:
        out += ["| Time | Action | Ticker | Side | Shares | Fill | P/L | P/L % | R | Cause | Why |", "|---|---|---|---|---:|---:|---:|---:|---:|---|---|"]
        for t in f["transactions"]:
            out.append(f"| {t['time']} | {t['action']} | {t['ticker']} | {t['position']} | {t['qty']} | "
                       f"{t['price'] if t['price'] else '—'} | {f'{t['pnl']:+,.2f}' if t['pnl'] is not None else '—'} | "
                       f"{f'{t['pnl_pct']:+.1f}%' if t['pnl_pct'] is not None else '—'} | {f'{t['r']:+.2f}R' if t.get('r') is not None else '—'} | {t['cause']} | {t['reason'][:140].replace('|', '/')} |")
    else:
        out.append("No orders were sent.")
    out.append("")
    out.append("## Open positions at the close")
    if f["positions"]:
        out += ["| Ticker | Side | P/L % | To stop | To target | Rating | Stop from |", "|---|---|---:|---:|---:|---|---|"]
        for p in sorted(f["positions"], key=lambda p: p.get("pnl_pct") or 0):
            d = p.get("decision") or {}
            pct = lambda v: f"{v:+.1f}%" if v is not None else "—"  # noqa: E731
            out.append(f"| {p['ticker']} | {p['side']} | {pct(p.get('pnl_pct'))} | {pct(p.get('to_stop_pct'))} | "
                       f"{pct(p.get('to_target_pct'))} | {d.get('rating') or '—'} | {d.get('stop_from') or '—'} |")
    else:
        out.append("None.")
    out.append("")
    if f["findings"]:
        out += ["## Watcher findings"] + [f"- {x}" for x in f["findings"]] + [""]
    s = f["sweep"]
    if s.get("cycle"):
        out += ["## Sweep", f"Cycle {s['cycle']} on {s['date']}'s close: {s['analysed_this_cycle']} analysed this cycle, "
                f"{s['queued']} queued, {s['decided_today']} decisions on record for {s.get('decided_on') or s['date']}'s close."
                + (f" Pace {s['progress']['minutes_per_ticker']:.1f} min per ticker." if (s.get('progress') or {}).get('minutes_per_ticker') else ""), ""]
    u = f.get("usage") or {}
    if (u.get("total") or {}).get("calls"):
        from tradingagents.usage import describe as describe_usage

        out += ["## Model usage", describe_usage(u) + ".", "", "| Agent | Calls | Tokens in | Tokens out | Minutes |", "|---|---:|---:|---:|---:|"]
        out += [f"| {a} | {r['calls']} | {r['tokens_in']:,} | {r['tokens_out']:,} | {r['seconds'] / 60:.0f} |" for a, r in u["by_agent"].items()]
        if len(u.get("by_host") or {}) > 1:
            out.append("\nBy host: " + ", ".join(f"{h} {r['seconds'] / 3600:.1f} h" for h, r in u["by_host"].items()))
        out.append("")
    e = f.get("errors") or {}
    if e:
        from tradingagents.errors import describe as describe_errors

        out += ["## Model errors", describe_errors(e).capitalize() + "."]
        if e.get("total"):
            out += ["", "| Kind | Count |", "|---|---:|"] + [f"| {k} | {n} |" for k, n in e["by_kind"].items()]
            out += ["", "| Agent | Count |", "|---|---:|"] + [f"| {a} | {n} |" for a, n in e["by_agent"].items()]
            out += ["", "Most recent:"] + [f"- {line}" for line in e["recent"]]
        if e.get("failed_tickers"):
            out += ["", "Ticker runs that failed (each is queued again once per sweep cycle):"]
            out += [f"- **{t}**: {why[:200]}" for t, why in e["failed_tickers"].items()]
        out.append("")
    if f["scorecard"]:
        out += ["## Feedback scorecard (settled decisions)", "| Agent | Right / judged | Mean return |", "|---|---:|---:|"]
        for agent, st in f["scorecard"].items():
            if st.get("n"):
                out.append(f"| {agent} | {st['hits']}/{st['n']} | {f'{st['mean']:+.1%}' if st.get('mean') is not None else '—'} |")
        out.append("")
    if f.get("decisions_r"):
        out += ["## Settled decisions in R", "Each call trading its own stop and target over its holding window.", "",
                "| Calls | Scored | Total | Mean per call |", "|---|---:|---:|---:|"]
        out += [f"| {name} | {s['n']} | {s['total']:+.2f}R | {s['mean']:+.2f}R |" for name, s in f["decisions_r"].items()]
        out.append("")
    return "\n".join(out)


def report_dir(config: dict) -> Path:
    return Path(config["results_dir"]) / "reports" / "daily"


def run(config: dict, day: str | None = None, llm=None) -> tuple[Path, DailyReview | None]:
    """Gather, review, save, and teach: the day's report and its lessons in the store."""
    day = day or date.today().isoformat()
    facts = gather(config, day)
    rev = None
    try:
        rev = review(config, facts, llm)
    except Exception as exc:
        logger.warning("The daily review could not be made: %s", exc)
    folder = report_dir(config)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{day}.md").write_text(render(facts, rev), encoding="utf-8")
    (folder / f"{day}.json").write_text(json.dumps({"facts": {k: v for k, v in facts.items() if k != "positions"} | {"positions": facts["positions"]},
                                                     "review": rev.model_dump() if rev else None}, indent=2, default=str) + "\n", encoding="utf-8")
    if rev and rev.lessons:
        Lessons(config["lessons_path"]).add_daily(day, [l.model_dump() for l in rev.lessons])
    logger.info("Daily report for %s: %s, %d lesson(s)", day, folder / f"{day}.md", len(rev.lessons) if rev else 0)
    return folder / f"{day}.md", rev


def is_week_end(day: str, next_open) -> bool:
    """Whether ``day`` is the last session of its week: the next open falls in a later ISO week."""
    d = date.fromisoformat(day)
    nxt = next_open.date() if hasattr(next_open, "date") else next_open
    return nxt.isocalendar()[:2] != d.isocalendar()[:2]


def week_days(day: str) -> list[str]:
    """The ISO week's dates up to and including ``day``."""
    d = date.fromisoformat(day)
    monday = d - timedelta(days=d.weekday())
    return [(monday + timedelta(days=i)).isoformat() for i in range((d - monday).days + 1)]


__all__ = ["AGENTS", "DailyReview", "Lesson", "gather", "review", "render", "run", "is_week_end", "week_days", "report_dir"]
