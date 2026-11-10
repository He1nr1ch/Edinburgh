"""The score for a call waiting in the wings, and the places that rank by it."""

from __future__ import annotations

import pytest

from tradingagents.ranking import age_days, balances_book, levels_pct, score_call


@pytest.mark.unit
def test_the_score_is_reward_to_risk_scaled_by_conviction_stance_freshness_and_balance():
    base = score_call(16.0, 8.0, "Overweight", "Neutral")
    assert (base.reward_to_risk, base.score) == (2.0, 2.0)
    assert score_call(16.0, 8.0, "Buy", "Neutral").score == 2.5                  # a strong rating: ×1.25
    assert score_call(16.0, 8.0, "Overweight", "Aggressive").score == 2.2        # ×1.1
    assert score_call(16.0, 8.0, "Overweight", "Conservative").score == 1.8      # ×0.9
    assert score_call(16.0, 8.0, "Overweight", "Neutral", days_old=3).score == 1.4   # -10% a day
    assert score_call(16.0, 8.0, "Overweight", "Neutral", days_old=9).score == 1.0   # floor at half
    assert score_call(16.0, 8.0, "Overweight", "Neutral", balances=True).score == 2.3  # ×1.15


@pytest.mark.unit
def test_a_stop_under_one_percent_away_counts_as_one_percent():
    """A stop that close is noise waiting to fire, not less risk, so it must not inflate the score."""
    assert score_call(10.0, 0.2, "Overweight", "Neutral").reward_to_risk == 10.0
    assert score_call(10.0, 1.0, "Overweight", "Neutral").reward_to_risk == 10.0
    assert score_call(10.0, 1.8, "Overweight", "Neutral").reward_to_risk == pytest.approx(5.556, abs=0.001)


@pytest.mark.unit
def test_levels_in_percent_follow_the_side_and_a_missing_stop_is_the_default():
    assert levels_pct(1, 100.0, 120.0, 92.0, 2.0) == (pytest.approx(20.0), pytest.approx(8.0))
    assert levels_pct(-1, 100.0, 85.0, 105.0, 2.0) == (pytest.approx(15.0), pytest.approx(5.0))
    assert levels_pct(1, 100.0, 120.0, None, 2.0)[1] == 2.0
    assert levels_pct(1, 100.0, 120.0, 103.0, 2.0)[1] == 2.0      # a stop on the profit side protects nothing
    assert age_days("2026-11-06", "2026-11-09") == 3.0 and age_days("", "2026-11-09") == 0.0
    assert balances_book(-1, 20_000) and not balances_book(1, 20_000) and not balances_book(1, 0)


@pytest.mark.unit
def test_the_bench_is_best_score_first_and_the_book_rows_carry_a_score(tmp_path, monkeypatch):
    import json
    from pathlib import Path

    from tradingagents import book as bk

    def decision(run, ticker, date, rating, stop, target, risk="Neutral"):
        folder = tmp_path / "backtest" / run / ticker / "TradingAgentsStrategy_logs"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"full_states_log_{date}.json").write_text(json.dumps({
            "company_of_interest": ticker, "final_rating": rating, "final_risk_level": risk,
            "risk_assessment": f"**Maximum Position**: 5\n\n**Stop Loss**: {stop}\n",
            "final_trade_decision": f"**Rating**: {rating}\n\n**Price Target**: {target}\n"}))

    decision("r", "TOP", "2026-11-09", "Overweight", 94, 130)         # +30% / 6%: r:r 5
    decision("r", "WIDE", "2026-11-09", "Overweight", 92, 116)        # +16% / 8%: r:r 2
    decision("r", "TIGHT", "2026-11-09", "Overweight", 99.5, 110)     # +10% / 1% floor: r:r 10 ... but the floor caps it
    decision("r", "SHORT", "2026-11-09", "Sell", 104, 90, "Aggressive")   # -10% / 4%: 2.5 ×1.25 ×1.1 ×1.15 balance
    from tradingagents.default_config import DEFAULT_CONFIG
    cfg = {**DEFAULT_CONFIG, "results_dir": str(tmp_path), "names_per_side": 1, "bench_size": 10}
    monkeypatch.setattr(bk, "last_closes", lambda tickers, date: {t: 100.0 for t in tickers})

    built = bk.build_book(["r"], "2026-11-09", cfg, 100_000)
    assert sorted(o.ticker for o in built.book.orders) == ["SHORT", "TOP"]       # the top long and the only short
    assert all(o.score for o in built.book.orders)
    assert [b["ticker"] for b in built.bench] == ["TIGHT", "WIDE"]                # 10 over the 1% floor beats 2
    assert built.bench[0]["score"] == 10.0 and "over 1.0% to stop" in built.bench[0]["score_parts"]


@pytest.mark.unit
def test_replacement_candidates_come_best_score_first_with_the_balance_bonus(monkeypatch):
    from tradingagents.worker import replace

    bench = [{"ticker": "L1", "side": "long", "rating": "Overweight", "risk_level": "Neutral", "stop": "92", "target": "116", "decided": "2026-11-09"},
             {"ticker": "S1", "side": "short", "rating": "Underweight", "risk_level": "Neutral", "stop": "108", "target": "84", "decided": "2026-11-09"}]
    monkeypatch.setattr(replace, "latest_price", lambda t: 100.0)
    long_book = replace.candidates(bench, set(), set(), 5.0, net_value=50_000, today="2026-11-09")
    assert [c["ticker"] for c in long_book] == ["S1", "L1"]            # same r:r; the short balances a long book
    assert long_book[0]["live_score"] == pytest.approx(2.3) and long_book[1]["live_score"] == 2.0
    short_book = replace.candidates(bench, set(), set(), 5.0, net_value=-50_000, today="2026-11-09")
    assert [c["ticker"] for c in short_book] == ["L1", "S1"]


@pytest.mark.unit
def test_the_opening_placement_takes_the_best_scored_rows_first(tmp_path, monkeypatch):
    from datetime import datetime
    from unittest.mock import MagicMock

    from tradingagents.dataflows.vendors.alpaca.trading import Account
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.worker import runner

    sizing = tmp_path / "backtest" / "sizing"
    sizing.mkdir(parents=True)
    (sizing / "night_2026-11-09.csv").write_text(
        "ticker,side,rating,risk_level,ceiling_pct,weight_pct,price,shares,value,stop,stop_from,target,score\n"
        "LOW,long,Overweight,Neutral,10,10,100,400,40000,95,Risk Manager,110,1.5\n"
        "HIGH,long,Overweight,Neutral,10,10,100,400,40000,95,Risk Manager,130,4.2\n"
        "MID,short,Underweight,Neutral,10,-10,100,-200,20000,105,Risk Manager,80,2.8\n")
    sent = []
    monkeypatch.setattr(runner.trading, "get_account", lambda: Account(100_000, 100_000, True, ()))
    monkeypatch.setattr(runner.trading, "get_clock", lambda: MagicMock(is_open=True))
    monkeypatch.setattr(runner.trading, "open_orders", lambda: {})
    monkeypatch.setattr(runner.trading, "submit_market_order", lambda s, q, side: sent.append(s) or f"ord-{s}")
    monkeypatch.setattr(runner, "send_email", lambda *a, **k: True)
    cfg = {**DEFAULT_CONFIG, "results_dir": str(tmp_path), "worker_state_path": str(tmp_path / "s.json"),
           "worker_trading_enabled": True, "gross_pct": 75.0}

    runner.place_book(cfg)
    assert sent == ["HIGH", "MID"]       # 40,000 + 20,000 fit the 75,000; LOW, the weakest, is the one left out
