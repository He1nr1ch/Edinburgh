"""Company statements as they were filed, from SEC EDGAR.

Every other fundamentals vendor serves a period's current value and cuts the
statement at the fiscal period end. That is two claims a run should not make: a
period that has ended is not public until the company files, weeks later, and a
figure that was later restated is not what investors saw at the time.

EDGAR reports every fact with the date it was filed, so a run dated ``as_of_date``
serves exactly what was on file by then, restatements included at the vintage
that was current: Apple's 2008 total assets read 39.6B until the 2010 amendment
restated them to 36.2B.

Access needs no key or account, only a User-Agent identifying the caller, which
SEC requires and refuses requests without. US filers only: anything absent from
EDGAR's ticker map falls through to the next configured vendor.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import date, datetime
from pathlib import Path

import requests

from tradingagents import __version__
from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.errors import NoMarketDataError, VendorUnavailableError
from tradingagents.dataflows.files import replace_file

logger = logging.getLogger(__name__)

_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"

# The ticker table is fetched once a day.
_CACHE_TTL_SECONDS = 24 * 60 * 60

# A company's facts change only when it files, a dozen or so times a year, and
# SEC answers no "changed since?" request (no Last-Modified, no ETag). So the
# kept facts are served until the company's filing index, a tenth the size,
# shows a newer filing with financial data than the one they were fetched at.
# The index is asked at most this often per company: a run asks for six
# statements at once, and they share one look.
_CHECK_TTL_SECONDS = 15 * 60
# SEC adds a filing to the facts shortly after the index shows it. Facts
# fetched in between do not hold it yet, so they are fetched again on a later
# look until they do, or until the filing is this old: one that is not in the
# facts by then (an 8-K's cover page) never will be.
_FACTS_LAG_SECONDS = 60 * 60
# Prospectus supplements and free-writing prospectuses: offerings off a shelf.
_SHELF_FORMS = ("424", "FWP")

# Line items, each with the tags filers use for it, best first. First match wins
# and values are never summed across tags: a company reporting revenue under two
# tags would otherwise be counted twice.
_STATEMENTS: dict[str, list[tuple[str, tuple[str, ...]]]] = {
    "balance_sheet": [
        ("Total Assets", ("Assets",)),
        ("Current Assets", ("AssetsCurrent",)),
        ("Cash and Equivalents", ("CashAndCashEquivalentsAtCarryingValue",)),
        ("Total Liabilities", ("Liabilities",)),
        ("Current Liabilities", ("LiabilitiesCurrent",)),
        ("Stockholders Equity", ("StockholdersEquity",
                                 "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest")),
    ],
    "income_statement": [
        ("Revenue", ("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues",
                     "SalesRevenueNet")),
        ("Cost of Revenue", ("CostOfRevenue", "CostOfGoodsAndServicesSold")),
        ("Gross Profit", ("GrossProfit",)),
        ("Operating Income", ("OperatingIncomeLoss",)),
        ("Net Income", ("NetIncomeLoss",)),
        ("Diluted EPS", ("EarningsPerShareDiluted",)),
    ],
    "cashflow": [
        ("Operating Cash Flow", ("NetCashProvidedByUsedInOperatingActivities",
                                 "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations")),
        ("Investing Cash Flow", ("NetCashProvidedByUsedInInvestingActivities",)),
        ("Financing Cash Flow", ("NetCashProvidedByUsedInFinancingActivities",)),
        ("Capital Expenditure", ("PaymentsToAcquirePropertyPlantAndEquipment",
                                 "PaymentsToAcquireProductiveAssets")),
    ],
}

# A statement's figures cover a span: a quarter is about 90 days, a year about
# 365. One filing reports both the quarter and the year to date under the same
# end date, so a match on the end date alone can report half a year as a quarter.
_SPANS = {"quarterly": (60, 115), "annual": (300, 400)}

# A 10-Q's cash flows are often filed only year to date. A quarterly table takes
# the quarter where filed, else the span to date, named in the column; it never
# subtracts one filing from another, which would give a figure no filing states.
_YEAR_TO_DATE = ((150, 200, 6), (240, 290, 9))

# A fiscal year is a period an annual report covers. A 10-Q balance has no span
# to reject, and some filers' 10-Qs report twelve-month totals that pass the span
# check, so either would read as a fiscal year. The value is still the latest
# filing of any form: a recast after a split or spin-off counts from its filing.
_ANNUAL_FORMS = ("10-K", "20-F", "40-F")


def _user_agent() -> str:
    """Who SEC sees. No account or key exists; callers identify themselves.

    www.sec.gov, which serves the ticker map, refuses a User-Agent carrying no
    contact address: a client name alone or with a project URL gets 403, one
    with an address gets 200. So the default carries a placeholder address and
    the package version. Set SEC_EDGAR_USER_AGENT to your own name and address
    so SEC can reach you about your traffic rather than the project.
    """
    configured = os.getenv("SEC_EDGAR_USER_AGENT", "").strip()
    return configured or f"TradingAgents/{__version__} (contact@example.com)"




def _fetch_json(url: str) -> dict:
    """Read a public EDGAR document, respecting SEC's identification rule."""
    try:
        response = requests.get(url, headers={"User-Agent": _user_agent()}, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        # Every failure here is "this vendor cannot serve it now", so the router
        # moves on instead of seeing a transport exception it has no rule for.
        raise VendorUnavailableError(f"SEC EDGAR request failed ({status or type(exc).__name__})") from exc
    except ValueError as exc:
        raise VendorUnavailableError("SEC EDGAR returned an unreadable response") from exc


def _cached_json(url: str, name: str) -> dict:
    path = Path(get_config()["data_cache_dir"]) / "sec_edgar" / name
    if path.exists() and time.time() - path.stat().st_mtime < _CACHE_TTL_SECONDS:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            pass  # a truncated file is a miss, not a failure
    try:
        data = _fetch_json(url)
    except VendorUnavailableError as exc:
        kept = _read_json(path)
        if kept is None:
            raise
        logger.warning("%s; using the copy of %s kept on %s", exc, name, date.fromtimestamp(path.stat().st_mtime))
        return kept
    _keep_json(path, data)
    return data


def _read_json(path: Path) -> dict | None:
    """The kept file, or None when there is none or it cannot be read."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _keep_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    replace_file(path, lambda temp: Path(temp).write_text(json.dumps(data), encoding="utf-8"))


def _newest_filing(cik: str) -> tuple[str, str]:
    """(accession number, time accepted) of the filer's newest filing with XBRL data; empty when it has none.

    Prospectus supplements are passed over while the filer has anything else:
    a bank files dozens a day (JPMorgan, 10,653 in a year), each with a fee
    table in XBRL and no statement line, and counting them would have its facts
    fetched every day for nothing.
    """
    recent = (_fetch_json(_SUBMISSIONS_URL.format(cik=cik)).get("filings") or {}).get("recent") or {}
    accessions = recent.get("accessionNumber") or []
    forms = recent.get("form") or [""] * len(accessions)
    filings = [(accepted or "", accession, form or "")
               for accession, accepted, xbrl, form in zip(accessions, recent.get("acceptanceDateTime") or [],
                                                          recent.get("isXBRL") or [], forms, strict=False)
               if xbrl]
    reports = [f for f in filings if not f[2].startswith(_SHELF_FORMS)]
    accepted, accession, _ = max(reports or filings, default=("", "", ""))
    return accession, accepted


def _holds_filing(facts: dict, accession: str) -> bool:
    return any(fact.get("accn") == accession
               for taxonomy in (facts.get("facts") or {}).values() for tag in taxonomy.values()
               for values in (tag.get("units") or {}).values() for fact in values)


def _settled(accepted: str) -> bool:
    """Whether a filing accepted then has had time to reach the facts."""
    try:
        age = datetime.now().astimezone() - datetime.fromisoformat(accepted)
    except (TypeError, ValueError):
        return True
    return age.total_seconds() > _FACTS_LAG_SECONDS


def _company_facts(cik: str) -> dict:
    """The filer's facts: the kept copy while it is current, fetched anew once the filer has filed since.

    Beside the facts is a note of the filing they were fetched at. Within
    ``_CHECK_TTL_SECONDS`` of the last look the copy is served unasked;
    after it the filing index is read, and the facts are fetched only when it
    shows a newer filing (or the last fetch came before SEC had added the
    filing to them). When SEC cannot be reached, a kept copy is served rather
    than no fundamentals at all.
    """
    folder = Path(get_config()["data_cache_dir"]) / "sec_edgar"
    path, note_path = folder / f"CIK{cik}.json", folder / f"CIK{cik}.filing.json"
    note = _read_json(note_path) if path.exists() else None
    if note is not None and time.time() - note_path.stat().st_mtime < _CHECK_TTL_SECONDS:
        kept = _read_json(path)
        if kept is not None:
            return kept
    try:
        newest, accepted = _newest_filing(cik)
        if note is not None and note.get("complete") and note.get("filing") == newest:
            kept = _read_json(path)
            if kept is not None:
                os.utime(note_path)        # looked just now
                return kept
        facts = _fetch_json(_FACTS_URL.format(cik=cik))
    except VendorUnavailableError as exc:
        kept = _read_json(path)
        if kept is None:
            raise
        logger.warning("%s; using the facts for CIK %s kept on %s", exc, cik, date.fromtimestamp(path.stat().st_mtime))
        return kept
    _keep_json(path, facts)
    complete = not newest or _holds_filing(facts, newest) or _settled(accepted)
    _keep_json(note_path, {"filing": newest, "accepted": accepted, "complete": complete})
    return facts


# Filers SEC's ticker file leaves out, by the exchange ticker Alpaca uses.
# Verify any entry against data.sec.gov/submissions before adding it.
_CIK_OVERRIDES: dict[str, str] = {}


def cik_for(ticker: str) -> str | None:
    """The filer's CIK, or None when the ticker is not a US filer.

    SEC spells a share class with a dash (``BRK-B``) where the exchanges and
    Alpaca use a dot (``BRK.B``), so both spellings are looked up.
    """
    wanted = ticker.strip().upper()
    if wanted in _CIK_OVERRIDES:
        return _CIK_OVERRIDES[wanted]
    spellings = {wanted, wanted.replace(".", "-")}
    table = _cached_json(_TICKERS_URL, "company_tickers.json")
    for entry in table.values():
        if entry.get("ticker", "").upper() in spellings:
            return f"{int(entry['cik_str']):010d}"
    return None


def _span_index(fact: dict, spans: tuple[tuple[int, int], ...]) -> int | None:
    """Which of ``spans`` a duration fact covers (0 for an instant fact), or None."""
    if "start" not in fact:
        return 0
    days = (date.fromisoformat(fact["end"]) - date.fromisoformat(fact["start"])).days
    return next((i for i, (low, high) in enumerate(spans) if low <= days <= high), None)


def _as_of(facts: dict, tags: tuple[str, ...], as_of_date: str, spans: tuple[tuple[int, int], ...],
           forms: tuple[str, ...] = ()) -> tuple[dict, str]:
    """({(period end, span index): value}, unit) for the tags the filer reports, as known then.

    A period reported more than once takes its latest filing on or before the
    date, so an amendment counts from the day it was filed and not before. The
    unit comes from the filing: most lines are USD, earnings per share are
    USD/shares, and scaling those alike would print a real figure as zero. A
    duration fact (revenue, cash flow) must cover one of ``spans``; an instant
    fact (a balance) has no span and serves any.
    """
    values: dict[tuple[str, int], float] = {}
    chosen_unit = "USD"
    # Tags are tried in order and a period keeps the first one that reports it:
    # filers renamed lines over the years, so one tag covers only part of the
    # history. Values are never added across tags, which would double count.
    for tag in tags:
        for unit, unit_values in ((facts.get(tag) or {}).get("units", {})).items():
            latest: dict[tuple[str, int], dict] = {}
            covered: set[tuple[str, int]] = set()   # periods a filing of ``forms`` reports
            for fact in unit_values:
                index = _span_index(fact, spans)
                key = (fact["end"], index)
                if fact["filed"] > as_of_date or index is None or key in values:
                    continue
                if not forms or fact.get("form", "").startswith(forms):
                    covered.add(key)
                seen = latest.get(key)
                if seen is None or fact["filed"] >= seen["filed"]:
                    latest[key] = fact
            latest = {key: fact for key, fact in latest.items() if key in covered}
            if latest:
                chosen_unit = unit
                values.update({key: fact["val"] for key, fact in latest.items()})
    return dict(sorted(values.items())), chosen_unit


def _statement(kind: str, ticker: str, freq: str, as_of_date: str, title: str) -> str:
    as_of_date = as_of_date or datetime.now().strftime("%Y-%m-%d")
    cik = cik_for(ticker)
    if cik is None:
        raise NoMarketDataError(ticker, ticker, "not a US SEC filer")

    facts = _company_facts(cik)
    us_gaap = (facts.get("facts") or {}).get("us-gaap")
    if not us_gaap:
        raise NoMarketDataError(ticker, ticker, "US filer with no us-gaap facts")

    quarterly = freq.lower() == "quarterly"
    if quarterly:
        spans = (_SPANS["quarterly"], *((low, high) for low, high, _ in _YEAR_TO_DATE))
        names = ["", *(f" ({months} months)" for _, _, months in _YEAR_TO_DATE)]
    else:
        spans, names = (_SPANS["annual"],), [""]
    forms = () if quarterly else _ANNUAL_FORMS
    lines = {label: _as_of(us_gaap, tags, as_of_date, spans, forms) for label, tags in _STATEMENTS[kind]}
    # Each row takes the shortest span it reports for a period, and a column
    # holds one span of one period, so a row filed only to date keeps its figure
    # beside a row filed by quarter.
    chosen = {label: {} for label in lines}
    for label, (values, _) in lines.items():
        for end, index in values:
            chosen[label][end] = min(index, chosen[label].get(end, index))
    periods = sorted({(end, index) for spans_of in chosen.values() for end, index in spans_of.items()})
    if not periods:
        raise NoMarketDataError(ticker, ticker, f"no {freq} {title.lower()} filed by {as_of_date}")

    header = (
        f"# {title} for {ticker.upper()} ({freq}), USD in millions unless the row says otherwise\n"
        f"# SEC EDGAR facts filed on or before {as_of_date}, at the values filed then\n\n"
    )
    rows = [",".join([""] + [end + names[index] for end, index in periods])]
    for label, (values, unit) in lines.items():
        # Every row spans the same columns, or a reader lines the table up wrong.
        if not values:
            rows.append(",".join([label] + ["unavailable (not tagged by this filer)"] * len(periods)))
            continue
        name = label if unit == "USD" else f"{label} ({unit})"
        # Plain numbers: a thousands separator would split the CSV field.
        cells = []
        for end, index in periods:
            value = values.get((end, index)) if chosen[label].get(end) == index else None
            cells.append("" if value is None else f"{value / 1e6:.0f}" if unit == "USD" else f"{value:.2f}")
        rows.append(",".join([name] + cells))
    return header + "\n".join(rows) + "\n"


def get_balance_sheet(ticker: str, freq: str = "quarterly", as_of_date: str | None = None) -> str:
    """Balance sheet as filed on or before ``as_of_date``."""
    return _statement("balance_sheet", ticker, freq, as_of_date, "Balance Sheet")


def get_income_statement(ticker: str, freq: str = "quarterly", as_of_date: str | None = None) -> str:
    """Income statement as filed on or before ``as_of_date``.

    A fourth quarter is never derived: filers report it only inside the annual
    figure, and subtracting three separately filed quarters would invent a number
    with no filing date behind it.
    """
    return _statement("income_statement", ticker, freq, as_of_date, "Income Statement")


def get_cashflow(ticker: str, freq: str = "quarterly", as_of_date: str | None = None) -> str:
    """Cash flow statement as filed on or before ``as_of_date``."""
    return _statement("cashflow", ticker, freq, as_of_date, "Cash Flow Statement")
