"""Moving the feedback store to a new database."""

from __future__ import annotations

from datetime import datetime

import pytest
from sqlalchemy import create_engine

from tradingagents.feedback import FeedbackStore
from tradingagents.feedback.migrate import copy_store
from tradingagents.feedback.schema import agent_scores, decisions

STATE = {"final_rating": "Buy", "final_trade_decision": "**Rating**: Buy", "final_position_size_pct": 3.0,
         "verdicts": {"trader": {"agent": "Trader", "ok": True, "call": "Buy", "confidence": 0.7, "evidence": ["RSI 55"],
                                 "fields": {"entry_price": 10.0}}}}


def _url(tmp_path, name):
    return "sqlite:///" + str(tmp_path / name)


@pytest.mark.unit
def test_the_store_is_copied_whole_and_the_new_one_carries_on_from_it(tmp_path):
    source = FeedbackStore({"feedback_db_url": _url(tmp_path, "old.db")})
    first = source.record("GD", "2026-11-09", STATE)
    source.record("JPM", "2026-11-10", STATE)
    source.record_desk_verdict({"agent": "Watcher", "ok": True, "call": "high adverse", "fields": {"a": 1}}, "news", "2026-11-10", "JPM", "exit 40")
    with source.engine.begin() as conn:      # a score whose decision is gone: SQLite allows it, Postgres would not
        conn.execute(agent_scores.insert().values(decision_id=999, agent="trader", call="Buy"))
        conn.execute(agent_scores.insert().values(decision_id=first, agent="trader", call="Buy", correct=True, score=0.02))

    report = copy_store(_url(tmp_path, "old.db"), _url(tmp_path, "new.db"))
    assert report["decisions"] == {"copied": 2, "skipped": 0} and report["agent_verdicts"]["copied"] == 2
    assert report["agent_scores"] == {"copied": 1, "skipped": 1} and report["desk_verdicts"]["copied"] == 1

    target = FeedbackStore({"feedback_db_url": _url(tmp_path, "new.db")})
    assert [p["ticker"] for p in target.pending()] == ["GD", "JPM"]
    assert target.verdicts("GD", "2026-11-09")[0]["evidence"] == ["RSI 55"]
    assert target.desk_verdicts(kind="news")[0]["fields"] == {"a": 1}
    assert target.record("KIM", "2026-11-11", STATE) == 3          # ids kept, so the next one follows them

    with pytest.raises(ValueError, match="already holds rows"):     # a move into a new database, not a merge
        copy_store(_url(tmp_path, "old.db"), _url(tmp_path, "new.db"))


@pytest.mark.unit
def test_a_copy_that_cannot_be_made_says_why_and_changes_nothing(tmp_path):
    with pytest.raises(ValueError, match="no database at"):
        copy_store(_url(tmp_path, "missing.db"), _url(tmp_path, "new.db"))
    FeedbackStore({"feedback_db_url": _url(tmp_path, "old.db")}).record("GD", "2026-11-09", STATE)
    with pytest.raises(ValueError, match="same database"):
        copy_store(_url(tmp_path, "old.db"), _url(tmp_path, "old.db"))

    # A store from before the verdict tables existed is copied with what it has.
    old = create_engine(_url(tmp_path, "older.db"))
    decisions.create(old)
    with old.begin() as conn:
        conn.execute(decisions.insert().values(ticker="GD", trade_date="2026-11-01", asset_type="stock", recorded_at=datetime(2026, 11, 1),
                                               status="pending", rating="Hold", holding_days=5, benchmark="SPY"))
    report = copy_store(_url(tmp_path, "older.db"), _url(tmp_path, "newer.db"))
    assert report["decisions"]["copied"] == 1 and report["agent_verdicts"] == {"copied": 0, "skipped": 0}
