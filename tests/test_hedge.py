"""The hedge: weekly SPY calls unless the monthly costs under the ratio times the weekly."""

from datetime import date

import pandas as pd
import pytest

from tradingagents.dataflows.vendors.alpaca import OptionQuote
from tradingagents.hedge import beta, choose_expiry, contracts_for, open_hedges, plan_hedge, record_hedge


def _call(expiry, ask):
    return OptionQuote(f"SPY{expiry:%y%m%d}C00760000", expiry, 760.0, ask - 0.05, ask, None)


WEEKLY = _call(date(2026, 10, 30), 7.52)


@pytest.mark.unit
@pytest.mark.parametrize("monthly_ask, choice", [
    (14.64, "monthly"),    # 1.95x: under 2.1x, so the month wins
    (15.79, "monthly"),    # 2.0997x: still under
    (15.80, "weekly"),     # 2.1011x: not under
    (18.00, "weekly"),
])
def test_the_weekly_call_unless_the_monthly_is_under_the_ratio(monthly_ask, choice):
    quote, chosen, _ = choose_expiry(WEEKLY, _call(date(2026, 11, 20), monthly_ask), 2.1)
    assert chosen == choice
    assert quote.expiry == (date(2026, 11, 20) if choice == "monthly" else date(2026, 10, 30))


@pytest.mark.unit
def test_a_missing_quote_falls_to_the_other_expiry():
    assert choose_expiry(None, _call(date(2026, 11, 20), 14.64), 2.1)[1] == "monthly"
    assert choose_expiry(WEEKLY, None, 2.1)[1] == "weekly"
    assert choose_expiry(None, None, 2.1)[0] is None


@pytest.mark.unit
def test_contracts_cover_the_net_short_exposure_and_none_for_a_net_long_book():
    assert contracts_for(-44_809, 759.97) == 1        # 0.59 of a contract
    assert contracts_for(-120_000, 759.97) == 2       # 1.58
    assert contracts_for(-5_000, 759.97) == 1         # any net short gets at least one
    assert contracts_for(30_000, 759.97) == 0


@pytest.mark.unit
def test_the_plan_prices_the_chosen_calls():
    plan = plan_hedge(-44_809, 759.97, WEEKLY, _call(date(2026, 11, 20), 14.64))
    assert (plan.contracts, plan.choice) == (1, "monthly")
    assert plan.cost == pytest.approx(1_464)
    assert plan.coverage == pytest.approx(75_997 / 44_809)

    long_book = plan_hedge(10_000, 759.97, WEEKLY, None)
    assert long_book.contracts == 0 and long_book.cost == 0


@pytest.mark.unit
def test_beta_is_the_return_slope_against_the_market():
    days = pd.bdate_range("2026-01-01", periods=60)
    market = pd.Series([100 * (1.01 if i % 2 else 0.995) ** i for i in range(60)], index=days)
    levered = market.pct_change().fillna(0).mul(2).add(1).cumprod() * 50
    assert beta(levered, market) == pytest.approx(2.0, rel=1e-6)
    assert beta(market.iloc[:5], market) == 1.0      # too short to measure


@pytest.mark.unit
def test_once_hedged_no_more_calls_are_bought_until_the_hedge_expires(tmp_path):
    ledger = tmp_path / "hedges.json"
    monthly = _call(date(2026, 11, 20), 14.64)
    first = plan_hedge(-44_809, 759.97, WEEKLY, monthly, held=open_hedges(ledger, "2026-10-23"))
    assert first.contracts == 1
    record_hedge(ledger, first, "2026-10-23")

    again = plan_hedge(-60_000, 759.97, WEEKLY, monthly, held=open_hedges(ledger, "2026-10-30"))
    assert again.contracts == 0 and again.cost == 0
    assert "already hedged" in again.reason
    assert again.coverage == pytest.approx(75_997 / 60_000)

    assert open_hedges(ledger, "2026-11-21") == ()           # expired: the next plan hedges anew
    assert plan_hedge(-44_809, 759.97, WEEKLY, monthly, held=()).contracts == 1


@pytest.mark.unit
def test_a_plan_that_buys_nothing_cannot_be_recorded(tmp_path):
    with pytest.raises(ValueError):
        record_hedge(tmp_path / "hedges.json", plan_hedge(10_000, 759.97, WEEKLY, None), "2026-10-23")
