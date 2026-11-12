"""The desk's lessons: what the daily and weekly reviews taught, read back by the agents that decide.

The feedback store scores each agent on settled outcomes; the memory log keeps
per-ticker reflections. This is the third strand: the reviews' lessons, in
words an agent can act on, kept in ``lessons.json``:

- ``daily``: each day's review adds a few lessons, each addressed to an agent
  (or to the desk's rules), with the evidence that taught it. Kept 30 days.
- ``standing``: the weekly review consolidates the week's daily lessons with
  the standing ones into a short list that holds until a later week rewrites
  it, so the set the agents read stays small and current.

:func:`render_for` gives an agent its block: the standing lessons addressed
to it or to the desk, then the last week's daily lessons for it, capped, so
every agent (analysts, researchers, deciders) and the Watcher start each call knowing what the desk
has already learned about their own part of the job.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

AGENTS = ("Market Analyst", "News Analyst", "Fundamentals Analyst", "Bull Researcher", "Bear Researcher",
          "Research Manager", "Trader",
          "Aggressive Risk Analyst", "Conservative Risk Analyst", "Neutral Risk Analyst",
          "Risk Manager", "Portfolio Manager", "Watcher", "Desk")
KEEP_DAILY_DAYS = 30
MAX_STANDING = 12
MAX_PER_AGENT = 8


class Lessons:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.daily: list[dict] = []
        self.standing: list[dict] = []
        self.updated = ""
        if self.path.is_file():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self.daily, self.standing = list(data.get("daily", [])), list(data.get("standing", []))
                self.updated = data.get("updated", "")
            except (OSError, ValueError):
                pass

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"daily": self.daily, "standing": self.standing, "updated": self.updated},
                                        indent=2) + "\n", encoding="utf-8")

    def add_daily(self, day: str, lessons: list[dict]) -> None:
        """Replace ``day``'s lessons with these (a rerun of the review is idempotent) and prune old ones."""
        self.daily = [d for d in self.daily if d.get("date") != day]
        self.daily += [{"date": day, "agent": l["agent"], "lesson": l["lesson"], "evidence": l.get("evidence", "")}
                       for l in lessons if l.get("lesson")]
        cutoff = (date.fromisoformat(day) - timedelta(days=KEEP_DAILY_DAYS)).isoformat()
        self.daily = sorted((d for d in self.daily if d["date"] >= cutoff), key=lambda d: d["date"])
        self.updated = day
        self.save()

    def set_standing(self, week_ending: str, lessons: list[dict]) -> None:
        """The weekly review's consolidated list replaces the standing lessons."""
        seen: set[tuple] = set()
        self.standing = []
        for l in lessons:
            key = (l.get("agent"), (l.get("lesson") or "").strip().lower())
            if not key[1] or key in seen:
                continue
            seen.add(key)
            self.standing.append({"agent": l["agent"], "lesson": l["lesson"].strip(), "evidence": l.get("evidence", ""),
                                  "since": l.get("since") or week_ending})
        self.standing = self.standing[:MAX_STANDING]
        self.updated = week_ending
        self.save()

    def for_agent(self, agent: str, as_of: str | None = None, days: int = 7) -> list[dict]:
        """Standing lessons for ``agent`` or the desk, then the last ``days`` of daily ones for ``agent``."""
        as_of = as_of or date.today().isoformat()
        since = (date.fromisoformat(as_of) - timedelta(days=days)).isoformat()
        standing = [l for l in self.standing if l["agent"] in (agent, "Desk")]
        recent = [l for l in reversed(self.daily) if l["agent"] == agent and since <= l["date"] <= as_of]
        # The agent's own recent lessons always get room; standing ones fill the rest.
        recent = recent[:MAX_PER_AGENT // 2]
        return standing[:MAX_PER_AGENT - len(recent)] + recent

    def render_for(self, agent: str, as_of: str | None = None) -> str:
        """The block an agent's prompt carries; empty when the desk has learned nothing for it yet."""
        rows = self.for_agent(agent, as_of)
        if not rows:
            return ""
        lines = [f"Lessons from the desk's own daily and weekly reviews, for the {agent}"
                 + (" and the desk" if agent != "Desk" else "") + " (apply them where they fit; they came from real outcomes):"]
        for l in rows:
            when = l.get("since") or l.get("date") or ""
            evidence = f" [{l['evidence']}]" if l.get("evidence") else ""
            lines.append(f"- ({when}{', standing' if 'since' in l else ''}) {l['lesson']}{evidence}")
        return "\n".join(lines)


def desk_lessons(agent: str, before: str = "", after: str = "\n\n") -> str:
    """The lessons block for ``agent`` from the run's config, wrapped only when there is one."""
    from tradingagents.dataflows.config import get_config

    path = get_config().get("lessons_path")
    if not path:
        return ""
    text = Lessons(path).render_for(agent)
    return f"{before}{text}{after}" if text else ""
