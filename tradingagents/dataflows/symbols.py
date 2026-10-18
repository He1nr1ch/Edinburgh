"""Symbol normalization for Alpaca, and ticker values safe to use in a path.

Alpaca spells US equities as their plain exchange ticker (``AAPL``, ``BRK.B``)
and crypto pairs with a slash (``BTC/USD``). Users often type the broker or
chart style instead (``btcusd``, ``BTC-USD``), so every vendor entry point
resolves symbols here and the same canonical form reaches the cache, the
memory log and the agents.
"""

from __future__ import annotations

import re

# Crypto bases Alpaca quotes against USD; a bare ``BTCUSD`` or ``BTC-USD`` is one
# of these pairs. Extend by adding a row.
CRYPTO_BASES = frozenset(
    {"BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "LTC", "BCH", "DOT", "AVAX", "LINK",
     "UNI", "AAVE", "SHIB", "SUSHI", "YFI", "MKR", "GRT", "CRV", "BAT", "PEPE", "TRUMP", "USDT", "USDC"}
)

_QUOTES = ("USD", "USDT", "USDC", "BTC", "ETH")


def is_crypto_symbol(symbol: str) -> bool:
    """Whether a canonical symbol is a crypto pair (``BTC/USD``)."""
    return "/" in symbol


def crypto_base(symbol: str) -> str | None:
    """The base coin of a crypto pair (``BTC`` for ``BTC/USD``), or None for a stock."""
    canonical = normalize_symbol(symbol)
    return canonical.split("/", 1)[0] if is_crypto_symbol(canonical) else None


def normalize_symbol(raw: str) -> str:
    """Canonical Alpaca symbol for what the user typed.

        user types          Alpaca wants
        ---------------     ------------
        aapl                AAPL
        brk.b               BRK.B
        BTCUSD, BTC-USD     BTC/USD
        eth/usd             ETH/USD
    """
    symbol = (raw or "").strip().upper()
    if not symbol:
        return symbol
    if "/" in symbol:
        return symbol
    if "-" in symbol:
        base, _, quote = symbol.partition("-")
        if quote in _QUOTES and base.isalnum():
            return f"{base}/{quote}"
        return symbol
    for quote in _QUOTES:
        if symbol.endswith(quote) and symbol[: -len(quote)] in CRYPTO_BASES:
            return f"{symbol[: -len(quote)]}/{quote}"
    return symbol


# Tickers can contain letters, digits, dot (share classes), dash, underscore and,
# for a crypto pair, a slash. The slash is the one character that cannot stand in
# a path, so it is written as a dash; nothing else enables directory traversal.
_TICKER_PATH_RE = re.compile(r"^[A-Za-z0-9._\-/]+$")


def safe_ticker_component(value: str, *, max_len: int = 32) -> str:
    """``value`` as a single filesystem path component, or ValueError.

    Tickers come from user CLI input or from LLM tool calls, both of which
    can be influenced by attacker-controlled content (e.g. prompt injection
    embedded in fetched news). Without validation, a value like
    ``"../../../etc/foo"`` flows into ``os.path.join`` / ``Path /`` and
    escapes the configured cache, checkpoint, or results directory.

    Returns ``value`` with a crypto pair's slash written as a dash
    (``BTC/USD`` -> ``BTC-USD``); raises ``ValueError`` for anything else
    that is not a plain ticker.
    """
    if not isinstance(value, str) or not value:
        raise ValueError(f"ticker must be a non-empty string, got {value!r}")
    if len(value) > max_len:
        raise ValueError(f"ticker exceeds {max_len} chars: {value!r}")
    if not _TICKER_PATH_RE.fullmatch(value):
        raise ValueError(
            f"ticker contains characters not allowed in a filesystem path: {value!r}"
        )
    # The regex allows '.', so '.', '..' and '../x' would pass, and as a path
    # component they traverse the parent directory. Each slash-separated part (a
    # pair has two) must be a real name: non-empty and not only dots or dashes.
    parts = value.split("/")
    if len(parts) > 2 or any(not part or set(part) <= {".", "-"} for part in parts):
        raise ValueError(f"ticker is not a plain symbol or pair: {value!r}")
    return "-".join(parts)
