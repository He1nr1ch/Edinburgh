"""Sizing a list run's calls into a book: weights by ceiling, a gross target, a cap."""

import json

import pytest

from tradingagents.sizing import Call, load_calls, size_book, target_weights


@pytest.mark.unit
def test_weights_follow_the_ceilings_and_reach_the_gross_target():
    calls = [Call("AAA", "Buy", 5), Call("BBB", "Overweight", 2.5), Call("CCC", "Sell", 5),
             Call("DDD", "Underweight", 2.5), Call("EEE", "Hold", 5)]
    weights = target_weights(calls, gross_pct=75, cap_pct=50)

    assert weights == pytest.approx({"AAA": 25, "BBB": 12.5, "CCC": -25, "DDD": -12.5})
    assert sum(abs(w) for w in weights.values()) == pytest.approx(75)


@pytest.mark.unit
def test_a_capped_position_passes_its_excess_to_the_rest():
    calls = [Call("BIG", "Buy", 15), Call("A", "Buy", 5), Call("B", "Sell", 5)]
    weights = target_weights(calls, gross_pct=30, cap_pct=10)

    assert weights == pytest.approx({"BIG": 10, "A": 10, "B": -10})


@pytest.mark.unit
def test_with_every_position_at_the_cap_the_book_falls_short_rather_than_breaking_it():
    calls = [Call("A", "Buy", 5), Call("B", "Sell", 5)]
    assert target_weights(calls, gross_pct=75, cap_pct=10) == pytest.approx({"A": 10, "B": -10})


@pytest.mark.unit
def test_holds_and_zero_ceilings_take_no_position():
    calls = [Call("H", "Hold", 5), Call("Z", "Sell", 0), Call("A", "Buy", 5)]
    book = size_book(calls, {"H": 10.0, "Z": 10.0, "A": 10.0}, equity=100_000, gross_pct=5, cap_pct=10)

    assert [o.ticker for o in book.orders] == ["A"]
    assert ("Z", "Sell with a Risk Manager ceiling of 0%") in book.skipped


@pytest.mark.unit
def test_the_book_is_whole_shares_with_shorts_negative():
    calls = [Call("A", "Buy", 5), Call("B", "Sell", 5)]
    book = size_book(calls, {"A": 300.0, "B": 70.0}, equity=100_000, gross_pct=20, cap_pct=10)

    by = {o.ticker: o for o in book.orders}
    assert by["A"].shares == 33 and by["A"].side == "long"      # 10,000 / 300, rounded down
    assert by["B"].shares == -142 and by["B"].side == "short"   # 10,000 / 70
    assert book.long_value == pytest.approx(9_900)
    assert book.cash == pytest.approx(90_100)
    assert book.net_pct == pytest.approx((9_900 - 9_940) / 1_000)


@pytest.mark.unit
def test_rounding_to_whole_shares_is_topped_up_toward_the_target():
    """Rounding each small position down left a long list several percent short."""
    calls = [Call(t, "Sell", 5) for t in ("A", "B", "C", "D")]
    prices = {"A": 880.0, "B": 870.0, "C": 860.0, "D": 850.0}   # 1.5% of 100k is under two shares
    book = size_book(calls, prices, equity=100_000, gross_pct=6, cap_pct=10)

    # Rounded down: one share each, 3.46%. The two cheapest get a second share;
    # the 830 left buys neither of the others, so the book stops under 6%.
    assert {o.ticker: abs(o.shares) for o in book.orders} == {"A": 1, "B": 1, "C": 2, "D": 2}
    assert 5.0 < book.gross_pct <= 6.0


@pytest.mark.unit
def test_a_ticker_without_a_price_is_left_out_and_named():
    book = size_book([Call("A", "Buy", 5)], {}, equity=100_000)
    assert book.orders == [] and book.skipped == [("A", "no closing price")]


@pytest.mark.unit
def test_calls_are_read_from_a_runs_saved_states(tmp_path):
    folder = tmp_path / "NVDA" / "TradingAgentsStrategy_logs"
    folder.mkdir(parents=True)
    (folder / "full_states_log_2026-10-23.json").write_text(json.dumps({
        "company_of_interest": "NVDA", "final_rating": "Underweight", "final_risk_level": "Neutral",
        "risk_assessment": "**Risk Level**: Neutral\n\n**Maximum Position**: 2.5\n",
    }))
    assert load_calls(tmp_path, "2026-10-23") == [Call("NVDA", "Underweight", 2.5, "Neutral")]
    assert load_calls(tmp_path, "2026-10-16") == []
