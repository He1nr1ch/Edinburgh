"""Build identity, as Dublin does it: the one place that knows which build is running.

The version is stamped at **build** time, never computed at runtime: a version
derived from the clock when a process happens to start would just re-report
the clock, which is the confusion this exists to end. The scheme is Dublin's:

    1.2026.11.09.15
    │ │    │  │  └─ hour of the build (24h, builder's local time)
    │ └────┴──┴──── build date
    └─────────────── major: 1 = Edinburgh (Dublin is 4)

so one build per hour is distinguishable, which is plenty for a hand-run
deploy. `deploy.sh` derives the string, passes it into the image as a build
arg, tags the image with it (`docker ps` on Colossus therefore names the
build it is running) and exports it as ``EDINBURGH_BUILD`` inside the
container. Everything that reports a build reads it from here: the worker's
startup and per-day banners, the day's record, the page's footer and health.

A checkout run outside Docker is not a build and says so: ``1.dev`` plus the
git sha and whether the tree is dirty.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

MAJOR = 1                       # Edinburgh
DEV_VERSION = f"{MAJOR}.dev"    # not a build: running from a source checkout

#: The build this process is: the stamp baked into the image, or 1.dev locally.
BUILD_VERSION = os.getenv("EDINBURGH_BUILD") or DEV_VERSION

#: True when running a stamped image rather than a checkout.
IS_BUILD = BUILD_VERSION != DEV_VERSION

_REPO = Path(__file__).resolve().parents[1]
_git_cache: dict[tuple, str | None] = {}

logger = logging.getLogger(__name__)


def _git(*args: str) -> str | None:
    """``git *args`` in the repo, or None (no git, no checkout, any failure)."""
    if args in _git_cache:
        return _git_cache[args]
    try:
        out = subprocess.run(("git", *args), cwd=_REPO, capture_output=True, text=True, timeout=5)
        value = (out.stdout.strip() or None) if out.returncode == 0 else None
    except Exception:
        value = None
    _git_cache[args] = value
    return value


def git_sha() -> str | None:
    """Short commit the build was made from, or None if unknown.

    Baked into the image by deploy.sh (which refuses to deploy a dirty tree,
    so the sha identifies the source exactly); a checkout run asks git.
    """
    return os.getenv("EDINBURGH_GIT_SHA") or _git("rev-parse", "--short", "HEAD")


def describe() -> str:
    """Human-readable build identity, e.g. ``1.2026.11.09.15 (ca1c4d1)``.

    A dev run is marked as such and flagged when the tree has uncommitted
    changes; the sha alone would misrepresent what is running.
    """
    out = BUILD_VERSION
    sha = git_sha()
    if sha:
        dirty = "" if IS_BUILD or not _git("status", "--porcelain") else "-dirty"
        out += f" ({sha}{dirty})"
    if not IS_BUILD:
        out += " [checkout, not a build]"
    return out


def log_banner(what: str) -> str:
    """Log and return the build line, e.g. at process start and at each day's open."""
    line = f"Edinburgh build {describe()} — {what}"
    logger.info(line)
    return line
