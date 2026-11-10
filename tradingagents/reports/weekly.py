"""The weekly report: the week's results by day, side and cause, the agents' record, and the standing lessons.

On the week's last session, after the daily report, the worker gathers the
week from the daily reports' facts, has the deep model review it, and saves
``reports/weekly/<week-ending>.md`` and ``.json``. The review consolidates the
week's daily lessons with the standing ones into a short standing list, which
replaces the old one in the lessons store: that is how the desk's learning
accumulates without the agents' prompts growing without bound.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

from pydantic import BaseModel, Field

from tradingagents.agents.structured import NO_EXTERNAL_TOOLS, bind_structured, invoke_structured
from tradingagents.lessons import MAX_STANDING, Lessons
from tradingagents.reports.daily import Lesson, report_dir, week_days

logger = logging.getLogger(__name__)


class WeeklyReview(BaseModel):
    summary: str = Field(description="Three to five sentences: the week's result and what drove it, with the numbers.")
    by_agent: str = Field(description="One short paragraph per agent whose decisions shaped the week (skip the rest): what its calls got right and wrong, with evidence.")
    standing_lessons: list[Lesson] = Field(description=f"The consolidated standing lessons for the desk going forward, at most {MAX_STANDING}: merge the week's daily lessons with the prior standing ones, keep what the evidence still supports, drop what it does not, sharpen wording. Each addressed to one agent or the Desk.")
    rule_suggestions: list[str] = Field(description="Up to three changes to the desk's rules or parameters the week argues for, each with the evidence; empty when none.")


def gather(config: dict, week_ending: str) -> dict:
    """The week from the daily reports' saved facts (days without a report are skipped)."""
    days = []
    for day in week_days(week_ending):
        path = report_dir(config) / f"{day}.json"
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            days.append({"day": day, **data.get("facts", {}), "review": data.get("review")})
    exits = [t for d in days for t in d.get("transactions", []) if t.get("flow") == "out"]
    by_cause: dict[str, dict] = defaultdict(lambda: {"n": 0, "pnl": 0.0})
    by_side: dict[str, dict] = defaultdict(lambda: {"n": 0, "pnl": 0.0, "wins": 0})
    for t in exits:
        by_cause[t.get("cause", "other")]["n"] += 1
        by_side[t["position"]]["n"] += 1
        if t.get("pnl") is not None:
            by_cause[t.get("cause", "other")]["pnl"] += t["pnl"]
            by_side[t["position"]]["pnl"] += t["pnl"]
            by_side[t["position"]]["wins"] += int(t["pnl"] > 0)
    realised = [d.get("realised") for d in days if d.get("realised") is not None]
    first, last = (days[0], days[-1]) if days else ({}, {})
    lessons = Lessons(config["lessons_path"])
    return {
        "week_ending": week_ending, "days": days,
        "equity_start": first.get("prior_equity") or first.get("equity"), "equity_end": last.get("equity"),
        "realised_week": round(sum(realised), 2) if realised else None,
        "by_day": [(d["day"], d.get("realised"), d.get("equity")) for d in days],
        "exits": len(exits), "entries": sum(d.get("entries", 0) for d in days),
        "by_cause": dict(by_cause), "by_side": dict(by_side),
        "daily_lessons": [l for d in days for l in ((d.get("review") or {}).get("lessons") or [])],
        "standing": lessons.standing,
        "scorecard": last.get("scorecard") or {},
        "agent_counts": dict(Counter(l["agent"] for d in days for l in ((d.get("review") or {}).get("lessons") or []))),
    }


def _facts_text(w: dict) -> str:
    lines = [f"Week ending {w['week_ending']}: {len(w['days'])} session(s) with a daily report."]
    if w["equity_start"] and w["equity_end"]:
        lines.append(f"Equity {w['equity_start']:,.0f} → {w['equity_end']:,.0f} ({w['equity_end'] - w['equity_start']:+,.0f})")
    if w["realised_week"] is not None:
        lines.append(f"Realised over the week: {w['realised_week']:+,.2f}")
    lines.append("By day (realised, equity at close): " + "; ".join(
        f"{d} {r:+,.0f}" if r is not None else f"{d} —" for d, r, _ in w["by_day"]))
    lines.append(f"Entries {w['entries']}, exits {w['exits']}. Exits by cause: "
                 + ", ".join(f"{c} {v['n']} ({v['pnl']:+,.0f})" for c, v in w["by_cause"].items()))
    lines.append("By side: " + ", ".join(f"{s} {v['n']} exits, {v['wins']} winners, {v['pnl']:+,.0f}" for s, v in w["by_side"].items()))
    if w["scorecard"]:
        lines.append("Feedback scorecard (settled decisions): " + "; ".join(
            f"{a} {st['hits']}/{st['n']}" + (f" ({st['mean']:+.1%})" if st.get("mean") is not None else "")
            for a, st in w["scorecard"].items() if st.get("n")))
    for d in w["days"]:
        rev = d.get("review") or {}
        if rev:
            lines.append(f"\n{d['day']} — what happened: {rev.get('what_happened', '')}\n  worked: {rev.get('what_worked', '')}\n  did not: {rev.get('what_did_not', '')}")
    if w["daily_lessons"]:
        lines.append("\nThe week's daily lessons:")
        lines += [f"- {l['agent']}: {l['lesson']} [{l.get('evidence', '')}]" for l in w["daily_lessons"]]
    if w["standing"]:
        lines.append("\nStanding lessons before this week:")
        lines += [f"- ({l.get('since')}) {l['agent']}: {l['lesson']}" for l in w["standing"]]
    return "\n".join(lines)


def _prompt(w: dict) -> str:
    return (
        "You are the desk's end-of-week reviewer on a long/short equity desk run by agents (Market, News and "
        "Fundamentals Analysts; Research Manager; Trader; Risk Manager; Portfolio Manager; the Watcher that "
        "guards positions intraday). The desk's rules: Sell means short; a call needs a target over 5% away; "
        "the book is the top ten each side by expected move; a missing stop defaults to 2%; the index is "
        "analysed around the clock and the book rebuilt each morning from decisions under five days old.\n\n"
        f"The week's record:\n{_facts_text(w)}\n\n"
        "Review the week as the desk's own reflections do: specific, terse, every figure cited. Then write the "
        f"standing lessons the desk carries forward, at most {MAX_STANDING}: merge the week's daily lessons with "
        "the prior standing ones, keep what the week's evidence still supports, drop what it contradicts or "
        "what one day's noise produced, and sharpen each into an instruction the named agent can apply. "
        "Prefer lessons about stops, targets, sizing, the Watcher's calls and the rules, which a week can judge, "
        "over lessons about direction, which it mostly cannot. " + NO_EXTERNAL_TOOLS
    )


def render(w: dict, rev: WeeklyReview | None) -> str:
    out = [f"# Edinburgh weekly report, week ending {w['week_ending']}", ""]
    if w["equity_start"] and w["equity_end"]:
        out.append(f"**Equity** {w['equity_start']:,.0f} → {w['equity_end']:,.0f} ({w['equity_end'] - w['equity_start']:+,.0f})"
                   + (f" · **realised** {w['realised_week']:+,.2f}" if w["realised_week"] is not None else ""))
    out += ["", "| Day | Realised | Equity at close |", "|---|---:|---:|"]
    out += [f"| {d} | {f'{r:+,.2f}' if r is not None else '—'} | {f'{e:,.0f}' if e else '—'} |" for d, r, e in w["by_day"]]
    out += ["", f"**Entries** {w['entries']} · **exits** {w['exits']}", "", "| Exit cause | Count | P/L |", "|---|---:|---:|"]
    out += [f"| {c} | {v['n']} | {v['pnl']:+,.2f} |" for c, v in w["by_cause"].items()]
    out += ["", "| Side | Exits | Winners | P/L |", "|---|---:|---:|---:|"]
    out += [f"| {s} | {v['n']} | {v['wins']} | {v['pnl']:+,.2f} |" for s, v in w["by_side"].items()]
    out.append("")
    if rev:
        out += ["## The week", rev.summary, "", "## By agent", rev.by_agent, ""]
        out += ["## Standing lessons (carried forward)"] + [f"- **{l.agent}**: {l.lesson} *({l.evidence})*" for l in rev.standing_lessons] + [""]
        if rev.rule_suggestions:
            out += ["## Rule suggestions (for you to decide, not applied)"] + [f"- {s}" for s in rev.rule_suggestions] + [""]
    else:
        out += ["## Review", "The model did not return a review in form; the standing lessons are unchanged.", ""]
    if w["scorecard"]:
        out += ["## Feedback scorecard (settled decisions)", "| Agent | Right / judged | Mean return |", "|---|---:|---:|"]
        out += [f"| {a} | {st['hits']}/{st['n']} | {f'{st['mean']:+.1%}' if st.get('mean') is not None else '—'} |"
                for a, st in w["scorecard"].items() if st.get("n")]
        out.append("")
    return "\n".join(out)


def weekly_dir(config: dict) -> Path:
    return Path(config["results_dir"]) / "reports" / "weekly"


def run(config: dict, week_ending: str, llm=None) -> tuple[Path, WeeklyReview | None]:
    """Gather the week, review it, save the report, and set the standing lessons."""
    from tradingagents.llm_clients.factory import create_tier_client

    w = gather(config, week_ending)
    rev = None
    try:
        llm = llm or create_tier_client(config, "deep").get_llm()
        rev = invoke_structured(bind_structured(llm, WeeklyReview, "Weekly review"), _prompt(w), "Weekly review")
    except Exception as exc:
        logger.warning("The weekly review could not be made: %s", exc)
    folder = weekly_dir(config)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{week_ending}.md").write_text(render(w, rev), encoding="utf-8")
    (folder / f"{week_ending}.json").write_text(json.dumps({"facts": {k: v for k, v in w.items() if k != "days"},
                                                             "review": rev.model_dump() if rev else None}, indent=2, default=str) + "\n", encoding="utf-8")
    if rev and rev.standing_lessons:
        Lessons(config["lessons_path"]).set_standing(week_ending, [l.model_dump() for l in rev.standing_lessons])
    logger.info("Weekly report for the week ending %s: %s", week_ending, folder / f"{week_ending}.md")
    return folder / f"{week_ending}.md", rev
