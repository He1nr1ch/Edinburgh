"""CLI symbol validation/classification must agree with the data path."""
import pytest

import cli.run as cli_run
from cli.models import AssetType
from cli.prompts import detect_asset_type, is_valid_ticker_input, normalize_ticker_symbol
from tradingagents.dataflows.symbols import normalize_symbol


@pytest.mark.parametrize("raw,expected", [
    ("BTCUSD", "BTC/USD"),
    ("BTC-USD", "BTC/USD"),
    ("btc/usd", "BTC/USD"),
    ("ethusdt", "ETH/USDT"),
    ("ETH-USDC", "ETH/USDC"),
    # non-crypto must be untouched
    ("AAPL", "AAPL"),
    ("brk.b", "BRK.B"),
    ("SPY", "SPY"),
])
def test_normalize_symbol_crypto_and_passthrough(raw, expected):
    assert normalize_symbol(raw) == expected


@pytest.mark.parametrize("value,ok", [
    ("AAPL", True),
    ("BRK.B", True),
    ("BTC/USD", True),
    ("BTC-USD", True),
    ("", True),                 # empty -> defaults to SPY downstream
    ("bad symbol!", False),     # space + '!' rejected
    ("GC=F", False),            # Yahoo futures syntax is not an Alpaca symbol
    ("^GSPC", False),
    ("A" * 40, False),          # too long
])
def test_ticker_input_validation(value, ok):
    assert is_valid_ticker_input(value) is ok


@pytest.mark.parametrize("raw,expected", [
    ("BTCUSD", AssetType.CRYPTO),
    ("BTC-USD", AssetType.CRYPTO),
    ("BTC/USD", AssetType.CRYPTO),
    ("ETHUSD", AssetType.CRYPTO),
    ("AAPL", AssetType.STOCK),
    ("BRK.B", AssetType.STOCK),
])
def test_detect_asset_type(raw, expected):
    assert detect_asset_type(raw) == expected


def test_cli_normalize_delegates_to_data_layer():
    # CLI must produce the same canonical symbol the data path will price.
    for raw in ("BTCUSD", "btc-usdt", "AAPL", "brk.b"):
        assert normalize_ticker_symbol(raw) == normalize_symbol(raw)


@pytest.mark.unit
def test_the_run_directory_cannot_escape_the_results_directory(tmp_path, monkeypatch):
    """Every other path that interpolates a ticker validates it first; the CLI's
    own results tree did not, so a ticker of '..' wrote a level up."""

    with pytest.raises(ValueError):
        cli_run._run_directory({"results_dir": str(tmp_path)}, "..", "2026-09-01")

    ok = cli_run._run_directory({"results_dir": str(tmp_path)}, "NVDA", "2026-09-01")
    assert str(ok).startswith(str(tmp_path))
    pair = cli_run._run_directory({"results_dir": str(tmp_path)}, "BTC/USD", "2026-09-01")
    assert pair == tmp_path / "BTC-USD" / "2026-09-01"
