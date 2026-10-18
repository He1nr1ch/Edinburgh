"""Point-in-time rules shared by every dated path: data is served as of the run's date.

- ``as_of`` / ``as_of_window`` clamp a date or window the model asks for to the
  trade date, so no tool reaches a vendor with a later one.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta


def get_current_date() -> str:
    """Today's date, YYYY-MM-DD."""
    return date.today().strftime("%Y-%m-%d")


def is_historical(run_date) -> bool:
    """Whether a run is dated before today, so live-only data would come from after it."""
    return bool(run_date) and str(run_date) < get_current_date()


def _parse(date: str | None) -> datetime | None:
    try:
        return datetime.strptime(date, "%Y-%m-%d")
    except (TypeError, ValueError):
        return None


def as_of(requested: str | None, trade_date: str) -> str | None:
    """The date a tool serves: the model's date, but never later than the run's.

    A model can omit the date or pass today's instead of the analysis date, which
    would walk past every point-in-time guard behind the tool. An empty
    ``trade_date`` (a direct call outside a graph run) passes the request through.
    """
    if not trade_date:
        return requested
    parsed = _parse(requested)
    return requested if parsed is not None and parsed <= _parse(trade_date) else trade_date


def as_of_window(start_date: str, end_date: str, trade_date: str) -> tuple[str, str]:
    """``[start, end]`` with its end clamped to the run date.

    A window wholly after the run date keeps its length and moves back to end there.
    An unreadable end reads as the run date, but no window can be guessed from an
    unreadable start, so it raises ValueError for the caller to send back (#1476).
    """
    start, old_end = _parse(start_date), _parse(end_date)
    if start is None:
        raise ValueError(f"start_date {start_date!r} is not a date; give it as YYYY-MM-DD")
    end = as_of(end_date, trade_date)
    if end == end_date or start <= _parse(end):
        return start_date, end
    span = (old_end - start) if old_end is not None and old_end >= start else timedelta(0)
    return f"{_parse(end) - span:%Y-%m-%d}", end
