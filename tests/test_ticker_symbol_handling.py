import unittest

import pytest

from cli.prompts import normalize_ticker_symbol
from tradingagents.agents.context import build_instrument_context


@pytest.mark.unit
class TickerSymbolHandlingTests(unittest.TestCase):
    def test_normalize_ticker_symbol_upper_cases_and_keeps_share_classes(self):
        self.assertEqual(normalize_ticker_symbol(" brk.b "), "BRK.B")

    def test_build_instrument_context_mentions_exact_symbol(self):
        context = build_instrument_context("BRK.B")
        self.assertIn("BRK.B", context)
        self.assertIn("exact spelling", context)

    def test_single_get_ticker_no_shadow(self):
        # A second get_ticker with an empty prompt (a bare "?") once shadowed
        # the descriptive one; the selection flow must use the one in prompts.
        import cli.prompts
        import cli.selections
        self.assertIs(cli.selections.get_ticker, cli.prompts.get_ticker)


if __name__ == "__main__":
    unittest.main()
