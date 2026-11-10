"""One score for a call waiting in the wings, so the best opportunity is taken first.

The bench, the morning placement and the Watcher's replacement all rank by
this, recomputed at the prices of the moment, so they agree on what "best"
means and none of them has to weigh it in prose.

    score = reward-to-risk × conviction × stance × freshness × balance

- reward-to-risk: the expected move to the target over the distance to the
  stop, both in percent of the current price. A stop nearer than the desk's
  default stop distance (2%) counts as that distance: that close, it is noise
  waiting to fire, not less risk, and must not outrank a stop placed sensibly.
- conviction: 1.25 for a Buy or a Sell, 1.0 for an Overweight or Underweight.
- stance: 1.1 Aggressive, 1.0 Neutral, 0.9 Conservative.
- freshness: minus 10% a day since the decision, down to half.
- balance: 1.15 when the candidate's side pulls the book's net exposure
  toward zero.

The hard rules (the target floor, not through its stop, not held, not used)
are filters before the score, not part of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

STRONG = {"Buy", "Sell"}
STANCE = {"Aggressive": 1.1, "Neutral": 1.0, "Conservative": 0.9}
MIN_RISK_PCT = 2.0      # the default stop distance; a nearer stop earns no more
FRESHNESS_PER_DAY = 0.10
FRESHNESS_FLOOR = 0.5
BALANCE_BONUS = 1.15
STRONG_BONUS = 1.25


@dataclass(frozen=True)
class Score:
    score: float
    reward_pct: float
    risk_pct: float
    reward_to_risk: float
    conviction: float
    stance: float
    freshness: float
    balance: float

    def parts(self) -> str:
        return (f"{self.reward_pct:+.1f}% to target over {self.risk_pct:.1f}% to stop = {self.reward_to_risk:.2f}"
                f" × conviction {self.conviction:g} × stance {self.stance:g} × fresh {self.freshness:.2f}"
                f" × balance {self.balance:g}")


def age_days(decided: str | None, as_of: str | None) -> float:
    """Whole days from the decision date to ``as_of``; 0 when either is unknown."""
    if not decided or not as_of:
        return 0.0
    try:
        return max(0.0, float((date.fromisoformat(as_of[:10]) - date.fromisoformat(decided[:10])).days))
    except ValueError:
        return 0.0


def score_call(reward_pct: float, risk_pct: float | None, rating: str, risk_level: str | None,
               days_old: float = 0.0, balances: bool = False, min_risk_pct: float = MIN_RISK_PCT) -> Score:
    """The score of a call with ``reward_pct`` to its target and ``risk_pct`` to its stop."""
    risk = max(float(risk_pct or 0.0), min_risk_pct)
    rr = max(0.0, reward_pct) / risk
    conviction = STRONG_BONUS if rating in STRONG else 1.0
    stance = STANCE.get(risk_level or "", 1.0)
    freshness = max(FRESHNESS_FLOOR, 1.0 - FRESHNESS_PER_DAY * days_old)
    balance = BALANCE_BONUS if balances else 1.0
    return Score(round(rr * conviction * stance * freshness * balance, 3), round(reward_pct, 2), round(risk, 2),
                 round(rr, 3), conviction, stance, round(freshness, 2), balance)


def levels_pct(side: int, price: float, target: float | None, stop: float | None,
               default_stop_pct: float) -> tuple[float, float]:
    """Reward and risk from ``price`` in percent, in the position's direction.

    A missing stop is the default distance; a stop on the profit side counts
    as the default too (it protects nothing, as the sizing rules say).
    """
    reward = side * (target / price - 1) * 100 if target and price else 0.0
    risk = -side * (stop / price - 1) * 100 if stop and price else 0.0
    if risk <= 0:
        risk = default_stop_pct
    return reward, risk


def balances_book(side: int, net_value: float) -> bool:
    """Whether opening on ``side`` (+1 long, -1 short) pulls a book with ``net_value`` toward zero."""
    return net_value != 0 and (side > 0) != (net_value > 0)
