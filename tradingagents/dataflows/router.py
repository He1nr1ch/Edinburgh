import logging

from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.errors import (
    NoMarketDataError,
    VendorNotConfiguredError,
    VendorUnavailableError,
)
from tradingagents.dataflows.vendors.alpaca import (
    get_global_news as get_alpaca_global_news,
    get_indicators as get_alpaca_indicators,
    get_market_snapshot as get_alpaca_market_snapshot,
    get_news as get_alpaca_news,
    get_stock_data as get_alpaca_stock_data,
)
from tradingagents.dataflows.vendors.fred import get_macro_data as get_fred_macro_data
from tradingagents.dataflows.vendors.sec_edgar import (
    get_balance_sheet as get_sec_edgar_balance_sheet,
    get_cashflow as get_sec_edgar_cashflow,
    get_income_statement as get_sec_edgar_income_statement,
)

logger = logging.getLogger(__name__)

# Tools organized by category
TOOLS_CATEGORIES = {
    "core_stock_apis": {
        "description": "OHLCV stock price data",
        "tools": [
            "get_stock_data"
        ]
    },
    "technical_indicators": {
        "description": "Technical analysis indicators and the verified market snapshot",
        "tools": [
            "get_indicators",
            "get_market_snapshot",
        ]
    },
    "fundamental_data": {
        "description": "Company financial statements",
        "tools": [
            "get_balance_sheet",
            "get_cashflow",
            "get_income_statement"
        ]
    },
    "news_data": {
        "description": "Company and market news",
        "tools": [
            "get_news",
            "get_global_news",
        ]
    },
    "macro_data": {
        "description": "Macroeconomic indicators (rates, inflation, labor, growth)",
        "tools": [
            "get_macro_indicators",
        ]
    },
}

# Optional enrichment categories. These add macro context to the news analyst
# but are not core to a decision, so a vendor failure here degrades to a
# sentinel instead of aborting the run. Core categories (prices, fundamentals,
# news) still raise so a broken primary is loud.
OPTIONAL_CATEGORIES = {"macro_data"}

# Mapping of methods to their vendor-specific implementations
VENDOR_METHODS = {
    # core_stock_apis
    "get_stock_data": {
        "alpaca": get_alpaca_stock_data,
    },
    # technical_indicators
    "get_indicators": {
        "alpaca": get_alpaca_indicators,
    },
    "get_market_snapshot": {
        "alpaca": get_alpaca_market_snapshot,
    },
    # fundamental_data
    "get_balance_sheet": {
        "sec_edgar": get_sec_edgar_balance_sheet,
    },
    "get_cashflow": {
        "sec_edgar": get_sec_edgar_cashflow,
    },
    "get_income_statement": {
        "sec_edgar": get_sec_edgar_income_statement,
    },
    # news_data
    "get_news": {
        "alpaca": get_alpaca_news,
    },
    "get_global_news": {
        "alpaca": get_alpaca_global_news,
    },
    # macro_data
    "get_macro_indicators": {
        "fred": get_fred_macro_data,
    },
}


def get_category_for_method(method: str) -> str:
    """Get the category that contains the specified method."""
    for category, info in TOOLS_CATEGORIES.items():
        if method in info["tools"]:
            return category
    raise ValueError(f"Method '{method}' not found in any category")


def get_vendor(category: str, method: str = None) -> str:
    """Get the configured vendor for a data category or specific tool method.
    Tool-level configuration takes precedence over category-level.
    """
    config = get_config()

    # Check tool-level configuration first (if method provided)
    if method:
        tool_vendors = config.get("tool_vendors", {})
        if method in tool_vendors:
            return tool_vendors[method]

    # Fall back to category-level configuration
    return config.get("data_vendors", {}).get(category, "default")


def vendor_unavailable(method: str, error: Exception) -> str:
    """What a call returns when every vendor was throttled or unreachable."""
    return (
        f"DATA_UNAVAILABLE: no configured vendor could serve {method} right now "
        f"({error}). This says nothing about the instrument; report the "
        f"data as unavailable and do not estimate or fabricate values."
    )


def no_data_available(error: NoMarketDataError) -> str:
    """What a call returns when every vendor that answered had no usable data."""
    resolved = "" if error.canonical == error.symbol else f" (resolved to '{error.canonical}')"
    reason = f" ({error.detail})" if error.detail else ""
    return (
        f"NO_DATA_AVAILABLE: No usable market data for '{error.symbol}'{resolved} from "
        f"any configured vendor{reason}. The symbol may be invalid, delisted, "
        f"not covered, or the vendor returned stale data. Do not estimate or "
        f"fabricate values — report that data is unavailable for this symbol."
    )


def route_to_vendor(method: str, *args, **kwargs):
    """Route method calls to appropriate vendor implementation with fallback support."""
    category = get_category_for_method(method)
    vendor_config = get_vendor(category, method)
    primary_vendors = [v.strip() for v in vendor_config.split(',')]

    if method not in VENDOR_METHODS:
        raise ValueError(f"Method '{method}' not supported")

    all_available_vendors = list(VENDOR_METHODS[method].keys())

    # The configured vendor list IS the chain: we do NOT silently fall back to
    # vendors the user did not choose. For multi-vendor fallback, list them in
    # order, e.g. data_vendors="alpaca,other". The "default" sentinel (no
    # explicit config) uses all available vendors.
    explicit = [v for v in primary_vendors if v and v != "default"]
    if explicit:
        vendor_chain = [v for v in explicit if v in VENDOR_METHODS[method]]
        if not vendor_chain:
            raise ValueError(
                f"Configured vendor(s) {explicit} not available for '{method}'. "
                f"Available: {all_available_vendors}."
            )
    else:
        vendor_chain = all_available_vendors

    last_no_data: NoMarketDataError | None = None
    last_unavailable: VendorUnavailableError | None = None
    failed: Exception | None = None     # a vendor that raised something untyped
    first_error: Exception | None = None
    for vendor in vendor_chain:
        vendor_impl = VENDOR_METHODS[method][vendor]
        impl_func = vendor_impl[0] if isinstance(vendor_impl, list) else vendor_impl

        try:
            return impl_func(*args, **kwargs)
        except VendorUnavailableError as e:
            logger.warning("Vendor %r unavailable for %s: %s; trying next vendor.", vendor, method, e)
            last_unavailable = e
            continue
        except VendorNotConfiguredError as e:
            logger.warning("Vendor %r not configured for %s; trying next vendor.", vendor, method)
            if first_error is None:
                first_error = e  # Surface it if no other vendor can serve the call.
            continue
        except NoMarketDataError as e:
            last_no_data = e  # No data here; another configured vendor may have it
            continue
        except Exception as e:
            # Don't let one vendor's failure crash the call when another can
            # serve it, but never swallow silently: a broken primary must be
            # visible in the logs, not hidden behind a fallback's verdict.
            logger.warning("Vendor %r failed for %s: %s", vendor, method, e)
            if first_error is None:
                first_error = e
            failed = e
            continue

    # A vendor that throttled or failed the request never said whether it has
    # the symbol, so no other vendor's "no data" can speak for the whole chain:
    # report the vendors as the problem, not the instrument.
    if last_unavailable is not None:
        return vendor_unavailable(method, last_unavailable)
    if failed is not None and last_no_data is not None:
        return vendor_unavailable(method, failed)

    # Every vendor that answered reported "no data": the symbol is genuinely unavailable.
    if last_no_data is not None:
        if first_error is not None:
            logger.warning(
                "Returning NO_DATA for %s, but a vendor errored earlier: %s",
                method, first_error,
            )
        return no_data_available(last_no_data)

    # No vendor returned data and none reported clean "no data" — surface the
    # first real error. Optional enrichment categories degrade to a sentinel
    # instead, so flavour data can't abort the run.
    if first_error is not None:
        if category in OPTIONAL_CATEGORIES:
            logger.warning("Optional %s unavailable for %s: %s", category, method, first_error)
            return (
                f"DATA_UNAVAILABLE: optional {category} could not be retrieved "
                f"({first_error}). Proceed without it; do not fabricate values."
            )
        raise first_error

    raise RuntimeError(f"No available vendor for '{method}'")
