# Edinburgh

A multi-agent trading desk on Alpaca market data, forked from
[TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents) v0.7.0
and cut down to the parts that matter here.

## The pipeline

```
Market Analyst ─┐
News Analyst   ─┼─► Bull vs Bear debate ─► Research Manager ─► Trader ─► Aggressive / Neutral / Conservative risk debate ─► Portfolio Manager
Fundamentals  ─┘
```

1. **Analysts** gather evidence. Prices, indicators and news come from Alpaca;
   financial statements come from SEC EDGAR as filed; macro series from FRED.
2. **Bull and Bear researchers** argue the case; the **Research Manager** rules.
3. The **Trader** proposes a transaction (direction, entry, stop, sizing).
4. **Risk Management** debates it from aggressive, neutral and conservative stances.
5. The **Portfolio Manager** makes the final call on direction and size.
6. A memory log settles each decision against the market after its holding
   window and feeds the lesson back to the Portfolio Manager. This is the seed of
   the feedback loop; a scored, database-backed loop is the next piece of work.

## Setup

Python 3.11 or later.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env     # add ALPACA_API_KEY / ALPACA_SECRET_KEY and the LLM you use
```

Run the interactive desk:

```bash
tradingagents
```

Or unattended, with every question answered by a flag or a `TRADINGAGENTS_*`
variable (see `.env.example`):

```bash
TRADINGAGENTS_LLM_PROVIDER=ollama OLLAMA_BASE_URL=http://192.168.0.98:11434/v1 \
TRADINGAGENTS_QUICK_THINK_LLM=qwen3.8:27b TRADINGAGENTS_DEEP_THINK_LLM=qwen3.8:27b \
TRADINGAGENTS_MAX_DEBATE_ROUNDS=1 TRADINGAGENTS_MAX_RISK_ROUNDS=1 \
tradingagents --ticker NVDA --date 2026-10-16 --analysts market,news,fundamentals --save --no-show
```

Programmatic use:

```python
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

graph = TradingAgentsGraph(config=DEFAULT_CONFIG)
state, rating = graph.propagate("NVDA", "2026-10-16")
```

## Data

| Category | Vendor | Notes |
|---|---|---|
| Daily bars, indicators, verified snapshot | Alpaca | `alpaca_feed` picks `sip` (paid) or `iex` (free) |
| Company and market news | Alpaca | Benzinga-backed feed, windowed by publish date |
| Balance sheet, cash flow, income statement | SEC EDGAR | as filed, US filers; no key, set `SEC_EDGAR_USER_AGENT` |
| Macro series | FRED | needs `FRED_API_KEY` |
| Instrument identity | Alpaca assets | name, exchange, asset class |

Alpaca has no fundamentals feed, so there is no market-cap or valuation-multiple
tool; the Fundamentals Analyst works from the statements.

Crypto pairs are spelled Alpaca's way, `BTC/USD`; `BTCUSD` and `BTC-USD` are
accepted and normalized.

## LLM providers

`ollama` (local or LAN, no key), `anthropic`, `openai`, or any OpenAI-compatible
endpoint via `openai_compatible` with `TRADINGAGENTS_LLM_BACKEND_URL`. The quick
tier (analysts, researchers, debaters, trader) and the deep tier (the two
managers) can each run on their own provider.

## Tests

```bash
pytest -q
ruff check .
```

The suite runs offline; every vendor is mocked.

## Removed from the upstream project

Yahoo Finance, Alpha Vantage, Reddit, StockTwits and Polymarket vendors; the
Sentiment Analyst; Google, Azure, Bedrock, ChatGPT-plan, DeepSeek, Qwen, GLM,
MiniMax, OpenRouter, Mistral, Kimi, Groq, NVIDIA and xAI providers; translated
output; the start-up update check; the HTML report; remembered CLI selections.

> Research software. Not investment advice.
