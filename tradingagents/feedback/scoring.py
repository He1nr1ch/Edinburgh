"""The rules that turn a settled price window into verdicts and per-agent scores.

Pure functions, so the rules can be read and tested apart from the database.

A directional call (Buy, Overweight, Sell, Underweight) wins when the price
moved its way. A Hold is right when the price stayed inside a band, missed a
move when it rose past it, and avoided one when it fell past it. A risk stance
is right when the outcome justified it: Aggressive on a clear win, Conservative
on a loss or a stop hit, Neutral when there was no meaningful loss.

A result is also stated in R: the return divided by the trade's own risk, the
distance from its entry to its stop. A stop-out is about -1R, a 6% gain on a
2% stop is +3R, and trades with different levels can be compared and averaged.
"""

from __future__ import annotations

from dataclasses import dataclass

from tradingagents.agents.rating import RATING_REVIEW

# Direction of every call the agents can make: ratings and trader actions.
DIRECTION: dict[str, int] = {
    "Buy": 1, "Overweight": 1, "Hold": 0, "Underweight": -1, "Sell": -1,
}

AGENTS = ("research_manager", "trader", "risk_manager", "portfolio_manager")


def direction_of(call: str | None) -> int | None:
    """+1, -1 or 0 for a directional call; None when it is not one (REVIEW, missing)."""
    if not call or call == RATING_REVIEW:
        return None
    return DIRECTION.get(call.strip().capitalize())


def exposure_return(direction: int | None, raw_return: float) -> float:
    """The return the call was exposed to: signed for a directional call, long for a Hold.

    A Hold keeps whatever is held, which this desk reads as long exposure.
    """
    return raw_return * (direction if direction else 1)


def verdict(direction: int | None, raw_return: float, hold_band_pct: float) -> str | None:
    """win / loss / flat for a directional call; flat / missed / avoided for a Hold."""
    if direction is None:
        return None
    if direction == 0:
        if abs(raw_return) * 100 <= hold_band_pct:
            return "flat"
        return "missed" if raw_return > 0 else "avoided"
    signed = direction * raw_return
    if signed > 0:
        return "win"
    if signed < 0:
        return "loss"
    return "flat"


def call_correct(direction: int | None, raw_return: float, hold_band_pct: float) -> bool | None:
    """Whether a directional call was right; None when it cannot be judged."""
    v = verdict(direction, raw_return, hold_band_pct)
    if v is None:
        return None
    return v in ("win", "flat", "avoided")


def stance_correct(level: str | None, exposure: float, stop_hit: bool | None, hold_band_pct: float) -> bool | None:
    """Whether a risk stance was justified by the outcome."""
    if not level or level == RATING_REVIEW:
        return None
    level = level.strip().capitalize()
    moved = exposure * 100
    if level == "Aggressive":
        return moved > hold_band_pct
    if level == "Conservative":
        return exposure < 0 or bool(stop_hit)
    if level == "Neutral":
        return not stop_hit and moved >= -hold_band_pct
    return None


def stop_was_hit(direction: int | None, stop: float | None, low: float, high: float) -> bool | None:
    """Whether the window's closes crossed the stop: below it for a long, above it for a short."""
    if stop is None:
        return None
    if direction is not None and direction < 0:
        return high >= stop
    return low <= stop


def risk_fraction(direction: int, entry: float, stop: float | None, default_stop_pct: float) -> float | None:
    """The trade's risk as a fraction of its entry: the distance to its stop.

    The desk's stop when it lies on the losing side of the entry; else, by the
    user's rule, ``default_stop_pct`` (the sizing rules' ``stop_for``). None
    when there is no direction, no entry, or no stop and no default.
    """
    if not direction or not entry or entry <= 0:
        return None
    if stop and (stop - entry) * direction < 0:
        return abs(stop / entry - 1)
    return default_stop_pct / 100 if default_stop_pct and default_stop_pct > 0 else None


def r_multiple(directional_return: float | None, risk: float | None) -> float | None:
    """A result in R: the signed return over the risk taken; None when either is unknown."""
    if directional_return is None or not risk:
        return None
    return directional_return / risk


def window_r(direction: int | None, closes, stop: float | None, target: float | None,
             default_stop_pct: float) -> float | None:
    """The R a call earned over its window, trading its own levels.

    ``closes`` run from the entry to the window's end. The position is closed
    at the first close through its stop or its target, as the Watcher would
    close it, else at the window's end. A close that gaps through the stop
    costs more than 1R, and is scored as what it cost.
    """
    prices = [float(c) for c in closes]
    if not direction or not prices:
        return None
    entry = prices[0]
    risk = risk_fraction(direction, entry, stop, default_stop_pct)
    if not risk:
        return None
    stop_level = entry * (1 - direction * risk)
    exit_price = prices[-1]
    for price in prices[1:]:
        through_stop = (price - stop_level) * direction <= 0
        through_target = bool(target) and (target - entry) * direction > 0 and (price - target) * direction >= 0
        if through_stop or through_target:
            exit_price = price
            break
    return r_multiple((exit_price / entry - 1) * direction, risk)


def r_summary(values) -> dict | None:
    """R over a set of results: how many, the total and the mean (the edge per trade); None when none is known."""
    known = [v for v in values if v is not None]
    if not known:
        return None
    return {"n": len(known), "total": round(sum(known), 2), "mean": round(sum(known) / len(known), 2)}


@dataclass(frozen=True)
class Scored:
    """What a settled decision earned, and each agent's mark for it."""

    direction: int
    directional_return: float
    weighted_return: float | None
    stop_hit: bool | None
    r_multiple: float | None          # the result over the risk to the stop; None for a Hold
    verdict: str
    agent_scores: list[dict]          # rows for the agent_scores table, minus decision_id


def score(decision: dict, raw_return: float, low: float, high: float, hold_band_pct: float,
          closes=None, default_stop_pct: float = 2.0) -> Scored:
    """Score a decision row against its window: ``low``/``high`` are the window's closes.

    With the window's ``closes`` (entry first) the result is also scored in R.
    """
    pm_direction = direction_of(decision.get("rating"))
    direction = pm_direction if pm_direction is not None else 0
    directional = raw_return * direction
    size = decision.get("position_size_pct")
    weighted = directional * size / 100 if size is not None else None
    stop = decision.get("rm_stop_loss") if decision.get("rm_stop_loss") is not None else decision.get("trader_stop_loss")
    stop_hit = stop_was_hit(direction, stop, low, high)
    exposure = exposure_return(direction, raw_return)

    rows = []
    for agent, call in (("research_manager", decision.get("research_recommendation")),
                        ("trader", decision.get("trader_action")),
                        ("portfolio_manager", decision.get("rating"))):
        d = direction_of(call)
        if call and call != RATING_REVIEW:
            rows.append({"agent": agent, "call": call, "correct": call_correct(d, raw_return, hold_band_pct),
                         "score": raw_return * d if d is not None else None})
    stance = decision.get("rm_risk_level")
    if stance and stance != RATING_REVIEW:
        rows.append({"agent": "risk_manager", "call": stance,
                     "correct": stance_correct(stance, exposure, stop_hit, hold_band_pct), "score": exposure})

    return Scored(direction=direction, directional_return=directional, weighted_return=weighted,
                  stop_hit=stop_hit,
                  r_multiple=(window_r(direction, closes, stop, decision.get("price_target"), default_stop_pct)
                              if closes is not None else None),
                  verdict=verdict(pm_direction, raw_return, hold_band_pct) or "flat",
                  agent_scores=rows)
