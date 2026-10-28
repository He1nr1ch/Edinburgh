"""The S&P 500 watchlist refresh, adapted from Dublin: fails closed, dot spelling, stale check."""

from __future__ import annotations

import os
import time

import pytest

from tradingagents import sp500


def _rows(n):
    return [(f"S{i}", f"Company {i}", "Sector") for i in range(n)]


@pytest.mark.unit
def test_a_short_fetch_leaves_the_existing_file_alone(tmp_path, monkeypatch):
    path = tmp_path / "sp500.txt"
    path.write_text("AAPL,MSFT\n")
    monkeypatch.setattr(sp500, "fetch_sp500_from_wikipedia", lambda: _rows(10))

    with pytest.raises(ValueError, match="refusing to overwrite"):
        sp500.refresh_sp500_watchlist(path)
    assert path.read_text() == "AAPL,MSFT\n"


@pytest.mark.unit
def test_a_good_fetch_is_written_sorted_and_deduplicated(tmp_path, monkeypatch):
    path = tmp_path / "sp500.txt"
    rows = _rows(sp500.MIN_EXPECTED_SYMBOLS) + [("BRK.B", "Berkshire", "Financials"), ("S1", "dup", "x")]
    monkeypatch.setattr(sp500, "fetch_sp500_from_wikipedia", lambda: rows)

    assert sp500.refresh_sp500_watchlist(path) == sp500.MIN_EXPECTED_SYMBOLS + 1
    symbols = path.read_text().strip().split(",")
    assert symbols == sorted(set(symbols)) and "BRK.B" in symbols


@pytest.mark.unit
def test_wikipedias_dash_spelling_becomes_the_projects_dot_spelling(monkeypatch):
    import pandas as pd

    class Response:
        text = "<html></html>"

        def raise_for_status(self):
            pass

    monkeypatch.setattr(sp500.requests, "get", lambda *a, **k: Response())
    monkeypatch.setattr(sp500.pd, "read_html", lambda _: [pd.DataFrame(
        {"Symbol": ["BRK-B", "aapl "], "Security": ["Berkshire", "Apple"], "GICS Sector": ["F", "IT"]})])

    assert [r[0] for r in sp500.fetch_sp500_from_wikipedia()] == ["BRK.B", "AAPL"]


@pytest.mark.unit
def test_a_fresh_file_is_kept_and_a_stale_one_refreshed(tmp_path, monkeypatch, capsys):
    path = tmp_path / "sp500.txt"
    path.write_text("AAPL\n")
    calls = []
    monkeypatch.setattr(sp500, "refresh_sp500_watchlist", lambda p: calls.append(p) or 1)

    sp500.refresh_if_stale(path=path)
    assert calls == [] and "still fresh" in capsys.readouterr().out

    old = time.time() - 40 * 86400
    os.utime(path, (old, old))
    sp500.refresh_if_stale(path=path)
    assert calls == [path]


@pytest.mark.unit
def test_a_failed_refresh_of_a_stale_file_warns_but_keeps_it(tmp_path, monkeypatch, capsys):
    path = tmp_path / "sp500.txt"
    path.write_text("AAPL\n")
    old = time.time() - 40 * 86400
    os.utime(path, (old, old))

    def boom(p):
        raise RuntimeError("wikipedia down")

    monkeypatch.setattr(sp500, "refresh_sp500_watchlist", boom)
    sp500.refresh_if_stale(path=path)
    assert "keeping existing 1 symbols" in capsys.readouterr().out
    assert sp500.get_sp500_symbols(path) == ["AAPL"]
