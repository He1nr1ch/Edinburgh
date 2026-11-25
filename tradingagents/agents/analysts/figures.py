"""The ratios a fundamentals report is built on, worked out before the analyst is asked.

Handed the statements as filed, the Fundamentals Analyst found the latest
columns in eighteen years of quarters and worked every margin, growth rate
and leverage ratio out by hand in its thinking: about 6,500 of the 7,500
tokens it wrote for a 440-word report, and the longest call of a ticker's
run. Arithmetic is not what the model is for. So the statements it is handed
are cut to their recent periods (:func:`recent`), and the ratios it used to
derive are computed here from the full statements (:func:`key_figures`) and
handed over beside them; the analyst is left with what they mean.

A figure is given only when both of its terms were filed for the same
period; anything else reads ``n/a`` and is never estimated. A year-on-year
change compares a period with the one of the same length ending a year
earlier, and is not given against a base of zero or less.
"""

from __future__ import annotations

from datetime import date

QUARTERS_KEPT = 8
YEARS_KEPT = 5
# How far a period's end may sit from a year before another's and still be its
# year-ago period: a 52/53-week fiscal calendar moves its quarter ends by days.
_YEAR_AGO_DAYS = (350, 380)

NOTE = (
    "Computed from the statements above, USD in millions; use these as given and do not work them out "
    "again. n/a means a term was not filed for that period. Compute by hand only a figure that is not here."
)


class Table:
    """One statement as the vendor writes it: comment lines, a row of period labels, a row per line item."""

    def __init__(self, head: list[str], periods: list[str], rows: dict[str, list[str]]):
        self.head, self.periods, self.rows = head, periods, rows

    def value(self, label: str, period: str) -> float | None:
        """The filed figure, or None when the row or the period is missing or the cell is not a number."""
        row = next((cells for name, cells in self.rows.items() if name == label or name.startswith(label + " (")), None)
        if row is None or period not in self.periods:
            return None
        try:
            return float(row[self.periods.index(period)])
        except (ValueError, IndexError):
            return None

    def year_before(self, period: str) -> str | None:
        """The period of the same length that ended a year before ``period``, if it was filed."""
        end, _, span = period.partition(" ")
        try:
            day = date.fromisoformat(end)
        except ValueError:
            return None
        for other in self.periods:
            other_end, _, other_span = other.partition(" ")
            try:
                gap = (day - date.fromisoformat(other_end)).days
            except ValueError:
                continue
            if other_span == span and _YEAR_AGO_DAYS[0] <= gap <= _YEAR_AGO_DAYS[1]:
                return other
        return None


def parse(text: str) -> Table | None:
    """The statement in ``text``, or None when it is not one (a section that could not be retrieved)."""
    lines = str(text).splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith(",")), None)
    if start is None:
        return None
    periods = lines[start].split(",")[1:]
    rows = {}
    for line in lines[start + 1:]:
        cells = line.split(",")
        if len(cells) == len(periods) + 1:
            rows[cells[0]] = cells[1:]
    if not periods or not rows:
        return None
    return Table(lines[:start], periods, rows)


def recent(text: str, keep: int) -> str:
    """The statement with only its latest ``keep`` periods; anything that is not a statement, unchanged."""
    table = parse(text)
    if table is None or len(table.periods) <= keep:
        return text
    head = [*[line for line in table.head if line.strip()],
            f"# The latest {keep} of the {len(table.periods)} periods filed", ""]
    body = [",".join(["", *table.periods[-keep:]])]
    body += [",".join([name, *cells[-keep:]]) for name, cells in table.rows.items()]
    return "\n".join(head + body) + "\n"


def _ratio(top: float | None, bottom: float | None) -> float | None:
    return None if top is None or bottom is None or bottom == 0 else top / bottom


def _pct(top: float | None, bottom: float | None) -> str:
    share = _ratio(top, bottom)
    return "n/a" if share is None else f"{share * 100:.1f}%"


def _times(top: float | None, bottom: float | None) -> str:
    multiple = _ratio(top, bottom)
    return "n/a" if multiple is None else f"{multiple:.2f}x"


def _change(now: float | None, then: float | None) -> str:
    if now is None or then is None or then <= 0:
        return "n/a"
    return f"{(now / then - 1) * 100:+.1f}%"


def _amount(value: float | None, digits: int = 0) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _free_cash_flow(cash: Table, period: str) -> float | None:
    operating, capex = cash.value("Operating Cash Flow", period), cash.value("Capital Expenditure", period)
    return None if operating is None or capex is None else operating - abs(capex)


def _income(table: Table, last: int) -> list[str]:
    rows = ["Period,Revenue,Revenue YoY,Gross margin,Operating margin,Net margin,Diluted EPS,EPS YoY"]
    for period in table.periods[-last:]:
        before = table.year_before(period)
        revenue = table.value("Revenue", period)
        gross = table.value("Gross Profit", period)
        if gross is None and revenue is not None and table.value("Cost of Revenue", period) is not None:
            gross = revenue - table.value("Cost of Revenue", period)
        eps = table.value("Diluted EPS", period)
        rows.append(",".join([
            period, _amount(revenue), _change(revenue, table.value("Revenue", before) if before else None),
            _pct(gross, revenue), _pct(table.value("Operating Income", period), revenue),
            _pct(table.value("Net Income", period), revenue), _amount(eps, 2),
            _change(eps, table.value("Diluted EPS", before) if before else None),
        ]))
    return rows


def _balance(table: Table) -> list[str]:
    latest = table.periods[-1]
    rows = ["Balance at,Current ratio,Cash / current liabilities,Total liabilities / equity"]
    for period in (table.year_before(latest), table.periods[-2] if len(table.periods) > 1 else None, latest):
        if period is None or any(row.startswith(period + ",") for row in rows):
            continue
        current = table.value("Current Liabilities", period)
        rows.append(",".join([
            period, _times(table.value("Current Assets", period), current),
            _pct(table.value("Cash and Equivalents", period), current),
            _times(table.value("Total Liabilities", period), table.value("Stockholders Equity", period)),
        ]))
    return rows


def _cash(table: Table, income: Table | None, last: int) -> list[str]:
    """Cash flow by period; the income statement, when its periods are the same ones, gives the ratios to it."""
    rows = ["Period,Operating cash flow,OCF YoY,Capital expenditure,Free cash flow,FCF YoY,FCF margin,OCF / net income"]
    for period in table.periods[-last:]:
        before = table.year_before(period)
        operating, free = table.value("Operating Cash Flow", period), _free_cash_flow(table, period)
        capex = table.value("Capital Expenditure", period)
        rows.append(",".join([
            period, _amount(operating), _change(operating, table.value("Operating Cash Flow", before) if before else None),
            _amount(None if capex is None else abs(capex)), _amount(free),
            _change(free, _free_cash_flow(table, before) if before else None),
            _pct(free, income.value("Revenue", period) if income else None),
            _times(operating, income.value("Net Income", period) if income else None),
        ]))
    return rows


def key_figures(statements: dict[tuple[str, str], str]) -> str | None:
    """The ratios the statements imply, as one section; None when no statement could be read.

    ``statements`` is the text of each statement by (tool name, ``quarterly``
    or ``annual``), in full: a year-on-year change needs the year before,
    which the cut handed to the analyst may not hold.
    """
    tables = {key: parse(text) for key, text in statements.items()}
    parts = []
    for freq, last in (("quarterly", 4), ("annual", 3)):
        income, balance, cash = (tables.get((name, freq)) for name in
                                 ("get_income_statement", "get_balance_sheet", "get_cashflow"))
        if income:
            parts.append(f"Income ({freq}):\n" + "\n".join(_income(income, last)))
        if balance and freq == "quarterly":
            parts.append("Balance sheet:\n" + "\n".join(_balance(balance)))
        if cash:
            # A quarterly cash flow is filed year to date, the income statement by quarter: a ratio
            # across the two is only given where both report the very same period.
            parts.append(f"Cash flow ({freq}; a period marked with months is the year to date):\n"
                         + "\n".join(_cash(cash, income, last)))
    if not parts:
        return None
    return NOTE + "\n\n" + "\n\n".join(parts)
