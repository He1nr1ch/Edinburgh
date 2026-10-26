"""What every Alpaca request shares: credentials, clients, and how a failure is reported."""

from __future__ import annotations

import functools
import logging
import os
from collections.abc import Callable
from typing import TypeVar

import requests
from alpaca.common.exceptions import APIError

from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.errors import VendorNotConfiguredError, VendorUnavailableError

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Alpaca's own variable names are accepted alongside the project's.
_KEY_VARS = ("ALPACA_API_KEY", "APCA_API_KEY_ID")
_SECRET_VARS = ("ALPACA_SECRET_KEY", "APCA_API_SECRET_KEY")


def credentials() -> tuple[str, str]:
    """The Alpaca key pair from the environment, or VendorNotConfiguredError."""
    key = next((os.environ[v] for v in _KEY_VARS if os.environ.get(v)), None)
    secret = next((os.environ[v] for v in _SECRET_VARS if os.environ.get(v)), None)
    if not key or not secret:
        raise VendorNotConfiguredError(
            "Alpaca is not configured: set ALPACA_API_KEY and ALPACA_SECRET_KEY "
            "(or APCA_API_KEY_ID and APCA_API_SECRET_KEY) in the environment or .env"
        )
    return key, secret


def paper_trading() -> bool:
    """Whether the key pair belongs to a paper account (config ``alpaca_paper``)."""
    return bool(get_config().get("alpaca_paper", True))


def data_feed():
    """The equity data feed to request (config ``alpaca_feed``: sip, iex, delayed_sip)."""
    from alpaca.data.enums import DataFeed

    return DataFeed(str(get_config().get("alpaca_feed", "sip")).lower())


@functools.lru_cache(maxsize=1)
def stock_client():
    from alpaca.data.historical import StockHistoricalDataClient

    key, secret = credentials()
    return StockHistoricalDataClient(key, secret)


@functools.lru_cache(maxsize=1)
def crypto_client():
    from alpaca.data.historical import CryptoHistoricalDataClient

    key, secret = credentials()
    return CryptoHistoricalDataClient(key, secret)


@functools.lru_cache(maxsize=1)
def news_client():
    from alpaca.data.historical import NewsClient

    key, secret = credentials()
    return NewsClient(key, secret)


@functools.lru_cache(maxsize=1)
def option_client():
    from alpaca.data.historical.option import OptionHistoricalDataClient

    key, secret = credentials()
    return OptionHistoricalDataClient(key, secret)


@functools.lru_cache(maxsize=2)
def trading_client(paper: bool):
    from alpaca.trading.client import TradingClient

    key, secret = credentials()
    return TradingClient(key, secret, paper=paper)


def _status(exc: APIError) -> int | None:
    code = getattr(exc, "status_code", None)
    if code is None:
        response = getattr(exc, "response", None)
        code = getattr(response, "status_code", None)
    return int(code) if code is not None else None


def call(request: Callable[[], T], what: str) -> T | None:
    """Make an Alpaca request, reporting a failure as what it is.

    Returns ``None`` when Alpaca answers that it has nothing (an unknown
    symbol); raises ``VendorUnavailableError`` for a throttle, an outage or a
    transport failure, which say nothing about the instrument; and
    ``VendorNotConfiguredError`` when the credentials or the subscription are
    refused, so the operator sees the account problem rather than a market one.
    """
    try:
        return request()
    except APIError as exc:
        status = _status(exc)
        message = str(exc)
        if status in (401, 403):
            raise VendorNotConfiguredError(f"Alpaca refused the request ({status}) while fetching {what}: {message}") from exc
        if status in (404, 422) or "not found" in message.lower() or "invalid symbol" in message.lower():
            logger.info("Alpaca has no %s: %s", what, message)
            return None
        if status == 429:
            raise VendorUnavailableError(f"Alpaca rate limited the request for {what}") from exc
        raise VendorUnavailableError(f"Alpaca request for {what} failed ({status}): {message}") from exc
    except requests.RequestException as exc:
        raise VendorUnavailableError(f"Alpaca is unreachable while fetching {what}: {type(exc).__name__}") from exc
