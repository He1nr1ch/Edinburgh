"""Who a symbol is, from Alpaca's asset list: name, exchange and asset class."""

from __future__ import annotations

from tradingagents.dataflows.symbols import normalize_symbol
from tradingagents.dataflows.vendors.alpaca import common


def get_asset_profile(ticker: str) -> dict:
    """Alpaca's current record for ``ticker``: ``name``, ``exchange``, ``asset_class``.

    Empty when Alpaca does not list the symbol. Raises for a failed request, so
    a caller can tell an unknown symbol from an outage.
    """
    canonical = normalize_symbol(ticker)
    asset = common.call(lambda: common.trading_client(common.paper_trading()).get_asset(canonical),
                        f"asset {canonical}")
    if asset is None:
        return {}
    profile = {}
    for field in ("name", "exchange", "asset_class", "status", "tradable", "shortable", "fractionable"):
        value = getattr(asset, field, None)
        if value is not None:
            profile[field] = value.value if hasattr(value, "value") else value
    return profile
