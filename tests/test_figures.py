"""The ratios handed to the Fundamentals Analyst, and the statements cut to their recent periods."""

import pytest

from tradingagents.agents.analysts import figures

INCOME = (
    "# Income Statement for ACME (quarterly), USD in millions unless the row says otherwise\n"
    "# SEC EDGAR facts filed on or before 2026-11-24, at the values filed then\n\n"
    ",2025-03-31,2025-06-30,2025-09-28,2026-03-31,2026-06-30,2026-09-27\n"
    "Revenue,1000,1100,1200,1100,1210,1500\n"
    "Cost of Revenue,600,660,720,660,,900\n"
    "Gross Profit,unavailable (not tagged by this filer),unavailable (not tagged by this filer),"
    "unavailable (not tagged by this filer),unavailable (not tagged by this filer),"
    "unavailable (not tagged by this filer),unavailable (not tagged by this filer)\n"
    "Operating Income,100,110,120,110,121,300\n"
    "Net Income,50,55,-60,55,60,150\n"
    "Diluted EPS (USD/shares),0.50,0.55,-0.60,0.55,0.60,1.50\n"
)
BALANCE = (
    "# Balance Sheet for ACME (quarterly)\n\n"
    ",2025-09-28,2026-06-30,2026-09-27\n"
    "Current Assets,900,950,1000\n"
    "Cash and Equivalents,100,150,200\n"
    "Total Liabilities,3000,3100,3200\n"
    "Current Liabilities,1000,1000,800\n"
    "Stockholders Equity,1000,1000,0\n"
)
CASH = (
    "# Cash Flow Statement for ACME (quarterly)\n\n"
    ",2025-09-28 (9 months),2026-03-31,2026-09-27 (9 months)\n"
    "Operating Cash Flow,400,165,500\n"
    "Capital Expenditure,100,55,-150\n"
)


def rows(text: str, heading: str) -> dict[str, list[str]]:
    block = text.split(heading + "\n", 1)[1].split("\n\n", 1)[0].splitlines()
    return {line.split(",")[0]: line.split(",")[1:] for line in block[1:]}


@pytest.mark.unit
def test_a_statement_is_cut_to_its_latest_periods_and_says_so():
    cut = figures.recent(INCOME, 2)
    assert ",2026-06-30,2026-09-27\n" in cut and "2025-03-31" not in cut
    assert "Revenue,1210,1500\n" in cut and "Diluted EPS (USD/shares),0.60,1.50\n" in cut
    assert "# The latest 2 of the 6 periods filed" in cut
    assert figures.recent(INCOME, 6) == INCOME                                 # nothing to cut
    assert figures.recent("Could not be retrieved (RuntimeError).", 2) == "Could not be retrieved (RuntimeError)."


@pytest.mark.unit
def test_margins_and_growth_come_from_the_same_period_and_the_one_a_year_before():
    income = rows(figures.key_figures({("get_income_statement", "quarterly"): INCOME}), "Income (quarterly):")
    # A 52-week calendar: the quarter ended 2026-09-27 is compared with the one ended 2025-09-28.
    assert income["2026-09-27"] == ["1500", "+25.0%", "40.0%", "20.0%", "10.0%", "1.50", "n/a"]   # no growth from a loss
    assert income["2026-03-31"] == ["1100", "+10.0%", "40.0%", "10.0%", "5.0%", "0.55", "+10.0%"]
    assert income["2026-06-30"][2] == "n/a"                                    # no cost of revenue filed: no gross margin
    assert income["2025-09-28"][1] == "n/a"                                    # nothing filed a year before it


@pytest.mark.unit
def test_leverage_and_cash_conversion_are_given_only_where_both_terms_were_filed():
    text = figures.key_figures({("get_income_statement", "quarterly"): INCOME, ("get_balance_sheet", "quarterly"): BALANCE,
                                ("get_cashflow", "quarterly"): CASH})
    balance = rows(text, "Balance sheet:")
    assert balance["2025-09-28"] == ["0.90x", "10.0%", "3.00x"]
    assert balance["2026-09-27"] == ["1.25x", "25.0%", "n/a"]                  # no equity to divide by
    cash = rows(text, "Cash flow (quarterly; a period marked with months is the year to date):")
    # Capital expenditure counts as an outflow whichever sign it was filed with.
    assert cash["2026-09-27 (9 months)"] == ["500", "+25.0%", "150", "350", "+16.7%", "n/a", "n/a"]
    # Only the first quarter is the same period in both statements: the rest are year to date against a quarter.
    assert cash["2026-03-31"] == ["165", "n/a", "55", "110", "n/a", "10.0%", "3.00x"]


@pytest.mark.unit
def test_nothing_is_computed_from_sections_that_are_not_statements():
    assert figures.key_figures({("get_cashflow", "quarterly"): "Could not be retrieved (RuntimeError: EDGAR is down)."}) is None
    assert figures.key_figures({}) is None


@pytest.mark.unit
def test_the_briefing_hands_over_the_recent_periods_and_the_figures_from_all_of_them(monkeypatch):
    from tradingagents.agents.analysts import briefing

    def route(method, symbol, freq, day):
        if freq == "annual":
            raise RuntimeError("EDGAR is down")
        return {"get_income_statement": INCOME, "get_balance_sheet": BALANCE, "get_cashflow": CASH}[method]

    monkeypatch.setattr(briefing, "route_to_vendor", route)
    monkeypatch.setattr(figures, "QUARTERS_KEPT", 3)
    text = briefing.fundamentals("ACME", "2026-11-24")
    statements, computed = text.split("[key figures]\n")
    assert ",2026-03-31,2026-06-30,2026-09-27\n" in statements and "2025-09-28\n" not in statements.split("[get_income")[1]
    assert computed.startswith(figures.NOTE)
    assert "2026-09-27,1500,+25.0%" in computed                               # the year before was cut from the statement, not from the sum
    assert "Income (annual)" not in computed and computed.rstrip().endswith("rather than fill the gap.")
