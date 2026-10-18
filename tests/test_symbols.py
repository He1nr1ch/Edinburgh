"""Symbols resolve to Alpaca's spelling, once, for every path that prices or caches them."""

import pytest

from tradingagents.dataflows.symbols import crypto_base, is_crypto_symbol, normalize_symbol


@pytest.mark.unit
@pytest.mark.parametrize("raw, canonical", [
    ("aapl", "AAPL"),
    (" NVDA ", "NVDA"),
    ("brk.b", "BRK.B"),
    ("BTC/USD", "BTC/USD"),
    ("btc/usd", "BTC/USD"),
    ("BTCUSD", "BTC/USD"),
    ("BTC-USD", "BTC/USD"),
    ("SOLUSDT", "SOL/USDT"),
    ("doge-usdc", "DOGE/USDC"),
    ("ETHBTC", "ETH/BTC"),
])
def test_normalize_symbol(raw, canonical):
    assert normalize_symbol(raw) == canonical


@pytest.mark.unit
def test_a_dash_in_a_stock_is_left_alone():
    """BRK-B is a share class, not a pair; only a known quote currency makes a pair."""
    assert normalize_symbol("BRK-B") == "BRK-B"
    assert normalize_symbol("ABC-XYZ") == "ABC-XYZ"


@pytest.mark.unit
def test_an_unknown_base_quoted_in_usd_stays_a_stock():
    """A ticker that happens to end in USD is not a coin unless its base is known."""
    assert normalize_symbol("FUSD") == "FUSD"


@pytest.mark.unit
def test_empty_input_is_returned_empty():
    assert normalize_symbol("") == ""
    assert normalize_symbol("   ") == ""


@pytest.mark.unit
def test_crypto_predicates():
    assert is_crypto_symbol("BTC/USD") and not is_crypto_symbol("AAPL")
    assert crypto_base("btcusd") == "BTC"
    assert crypto_base("AAPL") is None
