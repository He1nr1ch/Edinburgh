"""Tests for the run timer: total time and each stage's share of it."""

from cli.display import RunTimer, format_duration


def _run(report_keys=("market_report", "news_report")):
    return RunTimer(report_keys, now=100.0)


def test_each_stage_runs_from_the_end_of_the_one_before():
    timer = _run()
    timer.mark("Setup", now=102.0)
    timer.observe({"news_report": "n"}, now=130.0)
    timer.observe({"market_report": "m"}, now=164.0)
    timer.observe({"investment_plan": "p"}, now=184.0)
    timer.observe({"investment_plan": "p", "trader_investment_plan": "t"}, now=190.0)
    timer.observe({"risk_assessment": "r"}, now=215.0)
    timer.observe({"final_trade_decision": "d"}, now=225.0)
    timer.finish(now=226.5)

    assert timer.stages() == {
        "Setup": 2.0, "Analysts": 62.0, "Research": 20.0, "Trader": 6.0,
        "Risk": 25.0, "Portfolio": 10.0, "Wrap-up": 1.5,
    }
    assert timer.total() == 126.5


def test_the_analysts_finish_only_when_every_selected_report_has_landed():
    timer = _run()
    timer.observe({"news_report": "n"}, now=130.0)
    assert "Analysts" not in timer.stages()
    timer.observe({"market_report": "m"}, now=140.0)
    assert timer.stages()["Analysts"] == 40.0


def test_a_stage_keeps_the_time_it_first_finished():
    """The state carries every output to the end; a later chunk restating one does not move it."""
    timer = _run(("market_report",))
    timer.observe({"market_report": "m", "investment_plan": "p"}, now=150.0)
    timer.observe({"market_report": "m", "investment_plan": "p"}, now=190.0)
    assert timer.stages() == {"Analysts": 50.0, "Research": 0.0}


def test_the_summary_and_the_saved_record():
    timer = _run(("market_report",))
    timer.mark("Setup", now=101.0)
    timer.observe({"market_report": "m"}, now=165.0)
    timer.finish(now=227.0)

    assert timer.format_summary() == "Run time: 2m 07s (Setup 1.0s | Analysts 1m 04s | Wrap-up 1m 02s)"
    record = timer.to_dict(ticker="NVDA")
    assert record["ticker"] == "NVDA"
    assert record["total_seconds"] == 127.0
    assert record["stages"] == {"Setup": 1.0, "Analysts": 64.0, "Wrap-up": 62.0}


def test_durations_read_at_a_glance():
    assert format_duration(4.26) == "4.3s"
    assert format_duration(59.94) == "59.9s"
    assert format_duration(127) == "2m 07s"
    assert format_duration(3725) == "1h 02m"
