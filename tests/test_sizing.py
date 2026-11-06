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


@pytest.mark.unit
def test_a_call_needs_a_target_more_than_the_minimum_away_in_its_own_direction():
    from tradingagents.sizing import worth_trading

    long = Call("A", "Overweight", 5, target=106.0)
    assert worth_trading(long, 100.0, 5) is None                      # +6%
    assert "under the 5% minimum" in worth_trading(Call("A", "Overweight", 5, target=104.0), 100.0, 5)
    assert "wrong side" in worth_trading(Call("A", "Overweight", 5, target=95.0), 100.0, 5)
    short = Call("B", "Underweight", 5, target=93.0)
    assert worth_trading(short, 100.0, 5) is None                     # -7% for a short
    assert "under the 5% minimum" in worth_trading(Call("B", "Sell", 5, target=97.0), 100.0, 5)
    assert "wrong side" in worth_trading(Call("B", "Sell", 5, target=108.0), 100.0, 5)
    assert "no price target" in worth_trading(Call("C", "Sell", 5), 100.0, 5)
    assert worth_trading(Call("C", "Sell", 5), 100.0, 0) is None      # 0 turns the rule off


@pytest.mark.unit
def test_the_book_leaves_out_small_target_calls_and_says_why():
    calls = [Call("BIG", "Buy", 5, target=120.0), Call("TINY", "Buy", 5, target=101.0), Call("SH", "Sell", 5, target=90.0)]
    prices = {"BIG": 100.0, "TINY": 100.0, "SH": 100.0}
    book = size_book(calls, prices, equity=100_000, gross_pct=20, cap_pct=10, min_target_pct=5)

    assert sorted(o.ticker for o in book.orders) == ["BIG", "SH"]
    assert ("TINY", "target +1.0% is under the 5% minimum") in book.skipped


@pytest.mark.unit
def test_calls_carry_the_portfolio_managers_target(tmp_path):
    folder = tmp_path / "NVDA" / "TradingAgentsStrategy_logs"
    folder.mkdir(parents=True)
    (folder / "full_states_log_2026-11-04.json").write_text(json.dumps({
        "company_of_interest": "NVDA", "final_rating": "Underweight", "final_risk_level": "Neutral",
        "risk_assessment": "**Maximum Position**: 2\n", "final_trade_decision": "**Price Target**: 198.77\n",
    }))
    (call,) = load_calls(tmp_path, "2026-11-04")
    assert call.target == 198.77 and call.target_move_pct(200.13) == pytest.approx(0.68, abs=0.01)


@pytest.mark.unit
def test_the_book_is_the_top_n_longs_and_shorts_by_expected_move():
    from tradingagents.sizing import top_per_side

    longs = [Call(f"L{i}", "Overweight", 5, target=100 + 6 + i) for i in range(4)]     # +6% .. +9%
    shorts = [Call(f"S{i}", "Underweight", 5, target=100 - 6 - i) for i in range(4)]  # -6% .. -9%
    prices = {c.ticker: 100.0 for c in longs + shorts}

    kept = top_per_side(longs + shorts, prices, 2)
    assert [c.ticker for c in kept] == ["L3", "L2", "S3", "S2"]
    assert top_per_side(longs + shorts, prices, 0) == longs + shorts

    book = size_book(longs + shorts, prices, equity=100_000, gross_pct=40, cap_pct=10,
                     min_target_pct=5, names_per_side=2)
    assert sorted(o.ticker for o in book.orders) == ["L2", "L3", "S2", "S3"]
    assert ("L0", "outside the top 2 on its side (+6.0% to target)") in book.skipped


@pytest.mark.unit
def test_a_top_name_too_dear_for_one_share_gives_its_place_to_the_next():
    """NVR at $6,377 ranked first among the shorts and got no shares, leaving nine."""
    shorts = [Call("NVR", "Underweight", 2, target=4_864.0)] + \
             [Call(f"S{i}", "Underweight", 2, target=100 - 8 + i * 0.5) for i in range(4)]   # -8%, -7.5%, -7%, -6.5%
    prices = {"NVR": 6_376.82, "S0": 100.0, "S1": 100.0, "S2": 100.0, "S3": 100.0}
    book = size_book(shorts, prices, equity=100_000, gross_pct=6, cap_pct=10, min_target_pct=5, names_per_side=3)

    assert sorted(o.ticker for o in book.orders) == ["S0", "S1", "S2"]
    assert any(t == "NVR" and "less than one share" in why for t, why in book.skipped)
    assert ("S3", "outside the top 3 on its side (+6.5% to target)") in book.skipped   # a short's profit, in its own direction
