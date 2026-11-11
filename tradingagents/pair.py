"""Which PAIR node did the work: read from the router's own workload history.

NVIDIA PAIR (Personal AI Router) answers every Ollama call through a local
proxy and schedules it on one of the cluster's nodes, but the response says
nothing about which. The router keeps that record itself, in its config
folder (``~/.config/Nvidia Corporation/Personal AI Router`` on Linux): a
``workloads-history.json`` (rotated to ``.1``, ``.2``, ...) of every call
with the node it was scheduled on, the node it came from, the model and the
start and end times, and ``cluster/members.json`` naming the nodes. The
history is synced between the nodes, so any one of them has the cluster's.

:func:`node_shares` turns it into the table the positions page and
``tradingagents nodes`` show: calls and busy hours per node and model over
a window. The folder is ``pair_dir`` in the config (``TRADINGAGENTS_PAIR_DIR``);
on Colossus the web container has it mounted read-only.
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from pathlib import Path

DEFAULT_DIR = Path.home() / ".config" / "Nvidia Corporation" / "Personal AI Router"


def pair_dir(config: dict | None = None) -> Path:
    path = (config or {}).get("pair_dir") or os.environ.get("TRADINGAGENTS_PAIR_DIR") or ""
    return Path(path).expanduser() if path else DEFAULT_DIR


def available(folder: Path | None = None) -> bool:
    folder = folder or pair_dir()
    return (folder / "workloads-history.json").is_file()


def members(folder: Path | None = None) -> dict[str, str]:
    """Node uuid → name."""
    path = (folder or pair_dir()) / "cluster" / "members.json"
    try:
        return {m["nodeUuid"]: m.get("name") or m.get("id") or m["nodeUuid"] for m in json.loads(path.read_text(encoding="utf-8"))}
    except (OSError, ValueError, KeyError, TypeError):
        return {}


def history(folder: Path | None = None) -> list[dict]:
    """Every workload record on file, the rotated files included, deduplicated."""
    folder = folder or pair_dir()
    seen: dict[tuple, dict] = {}
    for path in sorted(folder.glob("workloads-history.json*")):
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for r in rows if isinstance(rows, list) else []:
            if isinstance(r, dict) and r.get("createdAt"):
                seen[(r.get("id"), r["createdAt"])] = r
    return sorted(seen.values(), key=lambda r: r["createdAt"])


def node_shares(rows: list[dict], names: dict[str, str], since_ms: float = 0, until_ms: float | None = None) -> dict:
    """Calls and busy hours by serving node (and by node and model), and where the calls came from."""
    by_node: dict[str, dict] = defaultdict(lambda: {"calls": 0, "busy_hours": 0.0, "cancelled": 0, "models": defaultdict(lambda: {"calls": 0, "busy_hours": 0.0})})
    origins: dict[str, int] = defaultdict(int)
    first = last = None
    for r in rows:
        t = r["createdAt"]
        if t < since_ms or (until_ms is not None and t > until_ms):
            continue
        first = t if first is None else min(first, t)
        last = t if last is None else max(last, t)
        node = names.get(r.get("scheduledOn"), r.get("scheduledOn") or "?")
        origins[names.get(r.get("originatedFrom"), r.get("originatedFrom") or "?")] += 1
        row = by_node[node]
        row["calls"] += 1
        if r.get("state") == "cancelled":
            row["cancelled"] += 1
        hours = ((r.get("completedAt") or 0) - (r.get("startedAt") or 0)) / 3.6e6 if r.get("completedAt") and r.get("startedAt") else 0.0
        row["busy_hours"] += max(0.0, hours)
        m = row["models"][r.get("model") or "?"]
        m["calls"] += 1
        m["busy_hours"] += max(0.0, hours)
    nodes = []
    for name, row in by_node.items():
        nodes.append({"node": name, "calls": row["calls"], "busy_hours": round(row["busy_hours"], 2), "cancelled": row["cancelled"],
                      "models": {k: {"calls": v["calls"], "busy_hours": round(v["busy_hours"], 2)} for k, v in row["models"].items()}})
    nodes.sort(key=lambda n: -n["busy_hours"])
    total_calls = sum(n["calls"] for n in nodes)
    for n in nodes:
        n["share_pct"] = round(100 * n["calls"] / total_calls, 1) if total_calls else 0.0
    return {"nodes": nodes, "origins": dict(origins), "calls": total_calls,
            "busy_hours": round(sum(n["busy_hours"] for n in nodes), 2),
            "first": _iso(first), "last": _iso(last)}


def shares(hours: float = 24, folder: Path | None = None, now_ms: float | None = None) -> dict:
    """The last ``hours`` of the router's history as node shares; empty when there is no history."""
    folder = folder or pair_dir()
    if not available(folder):
        return {}
    now_ms = now_ms if now_ms is not None else time.time() * 1000
    out = node_shares(history(folder), members(folder), since_ms=now_ms - hours * 3.6e6, until_ms=now_ms)
    out["hours"] = hours
    out["source"] = str(folder)
    return out


def _iso(ms: float | None) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ms / 1000)) if ms else ""


def render(s: dict) -> str:
    """A plain-text table for the terminal."""
    if not s:
        return "No PAIR history found (set TRADINGAGENTS_PAIR_DIR to the router's config folder)."
    if not s["calls"]:
        return f"No calls in the last {s.get('hours', '?')} h of PAIR's history."
    lines = [f"PAIR, last {s.get('hours', '?')} h ({s['first']} → {s['last']}): {s['calls']} calls, {s['busy_hours']:.1f} busy hours",
             f"{'node':24s} {'calls':>7s} {'share':>6s} {'busy h':>7s}  models"]
    for n in s["nodes"]:
        models = ", ".join(f"{m} {v['calls']} ({v['busy_hours']:.1f} h)" for m, v in sorted(n["models"].items(), key=lambda kv: -kv[1]["busy_hours"]))
        lines.append(f"{n['node']:24s} {n['calls']:7d} {n['share_pct']:5.0f}% {n['busy_hours']:7.1f}  {models}")
    lines.append("originated from: " + ", ".join(f"{k} {v}" for k, v in sorted(s["origins"].items(), key=lambda kv: -kv[1])))
    return "\n".join(lines)
