"""Tests for the ticker path-component validator that blocks directory traversal."""

import os
import unittest

import pytest

from tradingagents.dataflows.symbols import safe_ticker_component


@pytest.mark.unit
class TestSafeTickerComponent(unittest.TestCase):
    def test_accepts_common_ticker_formats(self):
        for ticker in ("AAPL", "BRK-B", "BRK.A", "SPY", "QQQ"):
            self.assertEqual(safe_ticker_component(ticker), ticker)

    def test_a_crypto_pair_becomes_a_single_component(self):
        self.assertEqual(safe_ticker_component("BTC/USD"), "BTC-USD")
        self.assertEqual(safe_ticker_component("ETH/USDT"), "ETH-USDT")

    def test_rejects_path_separators(self):
        for bad in (".", "..", "../etc", "a\\b", "/abs", "..\\..\\x", "/", "a//b"):
            with self.assertRaises(ValueError):
                safe_ticker_component(bad)

    def test_rejects_null_byte_and_whitespace(self):
        for bad in ("AAP L", "AAPL\x00", "AAPL\n", "\tAAPL"):
            with self.assertRaises(ValueError):
                safe_ticker_component(bad)

    def test_rejects_yahoo_only_syntax(self):
        for bad in ("GC=F", "^GSPC", "XAUUSD+"):
            with self.assertRaises(ValueError):
                safe_ticker_component(bad)

    def test_rejects_empty_or_non_string(self):
        for bad in ("", None, 123, b"AAPL"):
            with self.assertRaises(ValueError):
                safe_ticker_component(bad)

    def test_rejects_overlong_input(self):
        with self.assertRaises(ValueError):
            safe_ticker_component("A" * 33)

    def test_rejects_dot_only_values(self):
        # '.' and '..' pass the regex but traverse when used as a path
        # component (e.g. ``Path(results_dir) / ticker / "logs"``).
        for bad in (".", "..", "...", "....", "-", "./."):
            with self.assertRaises(ValueError):
                safe_ticker_component(bad)

    def test_traversal_string_does_not_escape_join(self):
        """Sanity: sanitized values stay within base when joined."""
        base = os.path.realpath("/tmp/cache")
        for ticker in ("AAPL", "BTC/USD"):
            component = safe_ticker_component(ticker)
            joined = os.path.realpath(os.path.join(base, f"{component}.csv"))
            self.assertTrue(joined.startswith(base + os.sep))


if __name__ == "__main__":
    unittest.main()
