"""The Alpaca vendor: credentials, bars into frames, caching, news, and how failures are reported."""

from datetime import UTC, datetime
from types import SimpleNamespace

import pandas as pd
import pytest
from alpaca.common.exceptions import APIError

from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.errors import (
    NoMarketDataError,
    VendorNotConfiguredError,
    VendorUnavailableError,
)
from tradingagents.dataflows.vendors.alpaca import common, identity, market, news


@pytest.fixture(autouse=True)
def _fresh_clients(monkeypatch):
    for name in ("stock_client", "crypto_client", "news_client", "trading_client"):
        getattr(common, name).cache_clear()
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")


def _bars_df(symbol, end="2026-01-09", periods=60):
    """Alpaca's ``BarSet.df``: a (symbol, timestamp) MultiIndex of lower-case columns."""
    stamps = pd.bdate_range(end=end, periods=periods, tz="America/New_York").tz_convert("UTC")
    closes = [100.0 + i for i in range(periods)]
    frame = pd.DataFrame({
        "open": closes, "high": [c + 1 for c in closes], "low": [c - 1 for c in closes],
        "close": closes, "volume": 1000, "trade_count": 10, "vwap": closes,
    }, index=pd.MultiIndex.from_product([[symbol], stamps], names=["symbol", "timestamp"]))
    return SimpleNamespace(df=frame)


class _Stock:
    def __init__(self, response):
        self.response = response
        self.requests = []

    def get_stock_bars(self, request):
        self.requests.append(request)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


# --- credentials ---------------------------------------------------------------

@pytest.mark.unit
def test_missing_credentials_are_a_configuration_error(monkeypatch):
    for name in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "APCA_API_KEY_ID", "APCA_API_SECRET_KEY"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(VendorNotConfiguredError, match="ALPACA_API_KEY"):
        common.credentials()


@pytest.mark.unit
def test_alpacas_own_variable_names_are_accepted(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY")
    monkeypatch.delenv("ALPACA_SECRET_KEY")
    monkeypatch.setenv("APCA_API_KEY_ID", "k2")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "s2")
    assert common.credentials() == ("k2", "s2")


# --- how a failure is reported -------------------------------------------------

def _api_error(status, message="boom"):
    """An APIError as alpaca-py raises it: the HTTP status lives on the wrapped error's response."""
    http_error = SimpleNamespace(response=SimpleNamespace(status_code=status), request=None)
    return APIError(f'{{"code": 40010001, "message": "{message}"}}', http_error=http_error)


@pytest.mark.unit
@pytest.mark.parametrize("status, expected", [
    (401, VendorNotConfiguredError), (403, VendorNotConfiguredError),
    (429, VendorUnavailableError), (500, VendorUnavailableError),
])
def test_refusals_and_outages_are_typed(status, expected):
    def request():
        raise _api_error(status)
    with pytest.raises(expected):
        common.call(request, "bars for X")


@pytest.mark.unit
def test_an_unknown_symbol_is_an_empty_answer():
    def request():
        raise _api_error(404, "symbol not found")
    assert common.call(request, "bars for X") is None


@pytest.mark.unit
def test_a_transport_failure_is_an_outage():
    import requests

    def request():
        raise requests.ConnectionError("down")
    with pytest.raises(VendorUnavailableError, match="unreachable"):
        common.call(request, "bars for X")


# --- bars ---------------------------------------------------------------------

@pytest.mark.unit
def test_bars_become_a_dated_ohlcv_frame_in_new_york_dates(monkeypatch):
    stock = _Stock(_bars_df("NVDA"))
    monkeypatch.setattr(common, "stock_client", lambda: stock)

    frame = market._bars("NVDA", datetime(2025, 1, 1, tzinfo=UTC), None)

    assert list(frame.columns) == ["Date", "Open", "High", "Low", "Close", "Volume"]
    assert frame["Date"].iloc[-1] == pd.Timestamp("2026-01-09")
    assert frame["Date"].dt.tz is None
    assert stock.requests[0].feed.value == "sip"


@pytest.mark.unit
def test_the_configured_feed_is_requested(monkeypatch):
    stock = _Stock(_bars_df("NVDA"))
    monkeypatch.setattr(common, "stock_client", lambda: stock)
    set_config({"alpaca_feed": "iex"})
    try:
        market._bars("NVDA", datetime(2025, 1, 1, tzinfo=UTC), None)
    finally:
        set_config({"alpaca_feed": "sip"})
    assert stock.requests[0].feed.value == "iex"


@pytest.mark.unit
def test_no_bars_is_no_data(monkeypatch):
    monkeypatch.setattr(common, "stock_client", lambda: _Stock(SimpleNamespace(df=pd.DataFrame())))
    with pytest.raises(NoMarketDataError):
        market._bars("ZZZZ", datetime(2025, 1, 1, tzinfo=UTC), None)


@pytest.mark.unit
def test_a_crypto_pair_goes_to_the_crypto_client(monkeypatch):
    seen = []

    class Crypto:
        def get_crypto_bars(self, request):
            seen.append(request)
            return _bars_df("BTC/USD")

    monkeypatch.setattr(common, "crypto_client", lambda: Crypto())
    monkeypatch.setattr(common, "stock_client", lambda: pytest.fail("stock client used for a coin"))

    frame = market._bars("BTC/USD", datetime(2025, 1, 1, tzinfo=UTC), None)

    assert seen and seen[0].symbol_or_symbols == "BTC/USD"
    assert len(frame) == 60


@pytest.mark.unit
def test_load_ohlcv_caches_cuts_at_the_date_and_fills(monkeypatch, tmp_path):
    stock = _Stock(_bars_df("NVDA", end="2026-01-16"))
    monkeypatch.setattr(common, "stock_client", lambda: stock)
    set_config({"data_cache_dir": str(tmp_path)})

    first = market.load_ohlcv("nvda", "2026-01-09")
    second = market.load_ohlcv("NVDA", "2026-01-08")

    assert first["Date"].max() == pd.Timestamp("2026-01-09")
    assert second["Date"].max() == pd.Timestamp("2026-01-08")
    assert len(stock.requests) == 1                 # the second read came from the cache
    assert (tmp_path / "NVDA-alpaca-daily.csv").exists()


@pytest.mark.unit
def test_get_stock_data_renders_the_window_as_csv(monkeypatch, tmp_path):
    monkeypatch.setattr(common, "stock_client", lambda: _Stock(_bars_df("NVDA", end="2026-01-16")))
    set_config({"data_cache_dir": str(tmp_path)})

    out = market.get_stock_data("NVDA", "2026-01-05", "2026-01-09")

    assert out.startswith("# Stock data for NVDA from 2026-01-05 to 2026-01-09\n# Total records: 5\n")
    assert "2026-01-09," in out and "2026-01-16" not in out


@pytest.mark.unit
def test_get_indicators_and_snapshot_read_the_cached_frame(monkeypatch, tmp_path):
    monkeypatch.setattr(common, "stock_client", lambda: _Stock(_bars_df("NVDA", end="2026-01-16")))
    set_config({"data_cache_dir": str(tmp_path)})

    rsi = market.get_indicators("NVDA", "rsi", "2026-01-09", 5)
    snap = market.get_market_snapshot("NVDA", "2026-01-09", 5)

    assert rsi.startswith("## rsi values from 2026-01-04 to 2026-01-09")
    assert "Latest trading row used: 2026-01-09" in snap
    with pytest.raises(ValueError, match="not supported"):
        market.get_indicators("NVDA", "stochrsi", "2026-01-09", 5)


@pytest.mark.unit
def test_get_closes_is_a_series_ending_before_the_end_date(monkeypatch):
    monkeypatch.setattr(common, "stock_client", lambda: _Stock(_bars_df("SPY", end="2026-01-16")))

    closes = market.get_closes("SPY", "2026-01-05", "2026-01-12")

    assert isinstance(closes, pd.Series) and len(closes) == 5
    assert closes.index.max() < pd.Timestamp("2026-01-12")
    assert closes.index.min() >= pd.Timestamp("2026-01-05")


@pytest.mark.unit
def test_get_closes_for_an_unknown_symbol_is_empty(monkeypatch):
    monkeypatch.setattr(common, "stock_client", lambda: _Stock(SimpleNamespace(df=pd.DataFrame())))
    assert market.get_closes("ZZZZ", "2026-01-05", "2026-01-12").empty


# --- news ---------------------------------------------------------------------

def _article(headline, day, symbols=("NVDA",)):
    return SimpleNamespace(headline=headline, source="benzinga", url=f"https://x/{headline}",
                           summary=f"{headline} summary", created_at=datetime(2026, 1, day, 12, tzinfo=UTC),
                           symbols=list(symbols))


class _News:
    def __init__(self, pages):
        self.pages = list(pages)
        self.requests = []

    def get_news(self, request):
        self.requests.append(request)
        return self.pages.pop(0)


@pytest.mark.unit
def test_news_is_fetched_across_pages_up_to_the_limit(monkeypatch):
    pages = [SimpleNamespace(news=[_article("a", 8), _article("b", 7)], next_page_token="t2"),
             SimpleNamespace(news=[_article("c", 6)], next_page_token=None)]
    client = _News(pages)
    monkeypatch.setattr(common, "news_client", lambda: client)
    set_config({"news_article_limit": 3})

    out = news.get_news("NVDA", "2026-01-05", "2026-01-09")

    assert out.startswith("## NVDA News, from 2026-01-05 to 2026-01-09:")
    assert "### a (source: benzinga 2026-01-08)" in out and "### c" in out
    assert client.requests[0].symbols == "NVDA"
    assert client.requests[1].page_token == "t2"
    assert client.requests[0].end == datetime(2026, 1, 10, tzinfo=UTC)   # the end day, inclusive


@pytest.mark.unit
def test_a_coin_is_tagged_without_its_slash(monkeypatch):
    client = _News([SimpleNamespace(news=[], next_page_token=None)])
    monkeypatch.setattr(common, "news_client", lambda: client)

    out = news.get_news("BTC/USD", "2026-01-05", "2026-01-09")

    assert client.requests[0].symbols == "BTCUSD"
    assert "No articles about BTC/USD" in out


@pytest.mark.unit
def test_global_news_asks_for_the_whole_market(monkeypatch):
    client = _News([SimpleNamespace(news=[_article("macro", 8, symbols=())], next_page_token=None)])
    monkeypatch.setattr(common, "news_client", lambda: client)

    out = news.get_global_news("2026-01-09", look_back_days=7, limit=5)

    assert client.requests[0].symbols is None
    assert out.startswith("## Market news, from 2026-01-02 to 2026-01-09:")
    assert "### macro" in out


# --- identity -----------------------------------------------------------------

@pytest.mark.unit
def test_asset_profile_names_the_instrument(monkeypatch):
    class Trading:
        def get_asset(self, symbol):
            return SimpleNamespace(name="NVIDIA Corporation Common Stock", exchange=SimpleNamespace(value="NASDAQ"),
                                   asset_class=SimpleNamespace(value="us_equity"), status=None,
                                   tradable=True, shortable=True, fractionable=True)

    monkeypatch.setattr(common, "trading_client", lambda paper: Trading())

    profile = identity.get_asset_profile("nvda")

    assert profile["name"] == "NVIDIA Corporation Common Stock"
    assert profile["exchange"] == "NASDAQ" and profile["asset_class"] == "us_equity"


@pytest.mark.unit
def test_an_unknown_asset_is_an_empty_profile(monkeypatch):
    class Trading:
        def get_asset(self, symbol):
            raise _api_error(404, "asset not found")

    monkeypatch.setattr(common, "trading_client", lambda paper: Trading())
    assert identity.get_asset_profile("ZZZZ") == {}


@pytest.mark.unit
def test_the_instrument_identity_reads_alpaca_and_fails_open(monkeypatch):
    from tradingagents.agents import context

    context._identity.cache_clear()
    monkeypatch.setattr(context, "get_asset_profile", lambda t: {"name": "NVIDIA", "exchange": "NASDAQ", "asset_class": "us_equity"})
    assert context.resolve_instrument_identity("NVDA") == {"company_name": "NVIDIA", "exchange": "NASDAQ", "quote_type": "us_equity"}

    context._identity.cache_clear()

    def down(t):
        raise VendorUnavailableError("down")
    monkeypatch.setattr(context, "get_asset_profile", down)
    assert context.resolve_instrument_identity("NVDA") == {}
