# Edinburgh

A multi-agent trading desk on Alpaca market data, forked from
[TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents) v0.7.0
and cut down to the parts that matter here.

## The pipeline

```
Market Analyst ─┐
News Analyst   ─┼─► Bull vs Bear debate ─► Research Manager ─► Trader ─► Aggressive / Neutral / Conservative debate ─► Risk Manager ─► Portfolio Manager
Fundamentals  ─┘
```

1. **Analysts** gather evidence. Prices, indicators and news come from Alpaca;
   financial statements come from SEC EDGAR as filed; macro series from FRED.
2. **Bull and Bear researchers** argue the case; the **Research Manager** rules.
3. The **Trader** proposes a transaction (direction, entry, stop, sizing).
4. **Risk Management** debates it from aggressive, neutral and conservative
   stances; the **Risk Manager** settles on a risk level (Aggressive, Neutral or
   Conservative), a position ceiling as a percentage of equity, and a stop.
5. The **Portfolio Manager** makes the final call: a rating for direction and a
   numeric position size, within the ceiling unless it says why not. Both the
   risk level and the size are typed fields, not prose, so they can be scored.
6. The **feedback loop** records every run's typed calls in a database, settles
   each decision against the market after its holding window, scores every
   agent's call, and shows the track record back to the Risk Manager and the
   Portfolio Manager on the next run. See below.

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
TRADINGAGENTS_LLM_PROVIDER=ollama OLLAMA_BASE_URL=http://127.0.0.1:11435 \
TRADINGAGENTS_QUICK_THINK_LLM=gemma4:26b TRADINGAGENTS_DEEP_THINK_LLM=gemma4:26b \
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

## The feedback loop

Every finished run writes a row to `decisions`: the Portfolio Manager's rating,
position size and risk level, the Risk Manager's stance, ceiling and stop, the
Trader's action, entry and stop, and the Research Manager's recommendation.
Once the holding window has traded (5 trading days by default), the next run
settles it: `outcomes` gets entry and exit prices, raw return, alpha against
SPY, drawdown, whether the stop was hit, and a verdict; `agent_scores` gets one
row per agent saying whether its call was right.

The rules are plain and live in `tradingagents/feedback/scoring.py`: a Buy wins
when the price rises, a Sell when it falls, a Hold is right within a 2% band,
and a risk stance is right when the outcome justified it (Aggressive on a clear
win, Conservative on a loss or a stop hit, Neutral otherwise).

The track record, hit rate and mean return per agent and per call plus the
ticker's own settled history with its lessons, is injected into the Risk Manager
and Portfolio Manager prompts, point-in-time filtered so a backtest only sees
what was known by its date.

```bash
tradingagents scorecard --ticker NVDA
```

The store is a SQLAlchemy URL (`TRADINGAGENTS_FEEDBACK_DB_URL`): SQLite under
`~/.tradingagents/memory/feedback.db` by default, Postgres when it lands
(`postgresql+psycopg://user:pass@host/db`, with `psycopg` installed). The
markdown memory log still writes the per-decision reflections; the store keeps
each lesson with its outcome.

## Sizing and the daily worker

`tradingagents size <run IDs> --date YYYY-MM-DD` turns a list run's calls into
a long/short book in whole shares (`--equity`, `--gross`, `--cap`), weighted by
each call's Risk Manager ceiling, and prices a SPY call hedge for a net-short
book: the weekly at-the-money call unless the monthly costs less than
`hedge_monthly_ratio` (2.1) times it. The hedge goes into the book CSV as a
row of its own; `--record-hedge` records calls bought by hand in the hedge
ledger, so later runs do not hedge again until they expire.

`tradingagents worker` is the always-on desk. At `daily_analysis_time` (1 AM ET)
on days the market opens (by Alpaca's clock, so weekends and holidays are skipped) it runs the daily job (`tradingagents daily` by hand): refresh
the watchlist (`daily_watchlist`, `~/.tradingagents/watchlists/sp500.txt`, seeded from the repo's copy and refreshed from Wikipedia when a month old) and analyse on the last close, qwen thinking on, in `daily_parts`
processes: the held positions first, then the watchlist least recently analysed first, until
`daily_deadline` (9 AM), so each night picks up where the last stopped (~11 minutes a ticker
thinking; the S&P 500 is about four nights). Size the book from every decision under
`decision_max_age_days` old and write a bench of the `bench_size` best-scored calls after it, long or short together
(the score, in `ranking.py`, is reward to risk × conviction × stance × freshness × balance,
recomputed at live prices wherever a waiting call is chosen: the bench, the opening placement
when the gross target has room for only some, and the Watcher's replacement), and price the SPY call hedge for a net-short book into the same
CSV (`daily_hedge`; a hedge that cannot be priced leaves the book unhedged, and
the email says so); the job times itself, each part and the whole, into a timing file
beside the book and the morning email. At the open it first closes any holding whose newest decision no longer backs its side, then
places the book's orders within the gross target, counting what is still held (`worker_place_book`),
buys the hedge's calls unless the ledger shows calls still held, and records the
ones it buys in the ledger; the Watcher's guards look at equities only. Then
the Watcher runs over the account's equity positions until the close, with
four guards on every pass. The stop guard and the target guard close a
position through its stop (the Risk Manager's, the Trader's, or the 2% default
from entry) or its target, no model asked. The news guard reads what was
published since the last pass and has the model judge it against the desk's
own case, the Portfolio Manager's thesis and the news the desk had read when it
decided: adverse, material news the case did not account for exits (high) or
halves (medium) the position; favorable, material, unaccounted-for news adds a
quarter, up to `worker_position_cap_pct`. The replacement guard sees a book
position that has closed and asks the model to pick its replacement from the
bench, among the candidates that still clear the rules at live prices, and
opens it with the freed capital. A close or trim first cancels the symbol's
waiting orders. It trades a paper account only,
unless `worker_allow_live` is on; `--dry-run` sends nothing, `--once` makes one
pass. Alerts go by email when `TRADINGAGENTS_SMTP_USER` and
`TRADINGAGENTS_SMTP_PASSWORD` are set (for Gmail, an app password).

It is deployed to Colossus as an always-on container, the same way as Dublin:
`./deploy.sh` builds the image, pushes it to Docker Hub and, over SSH, pulls
and restarts `edinburgh-worker` on Colossus with `--env-file ~/edinburgh/.env`,
host networking (PAIR answers only on each machine's own loopback) and
`~/edinburgh/data` mounted as the worker's data folder. It refuses to deploy
uncommitted work. Builds are numbered as Dublin's are, `1.YYYY.MM.DD.HH` (major 1 =
Edinburgh, then the build's date and hour on the builder's clock), stamped into the image
and used as its tag, so `docker ps` on Colossus names the running build; a checkout reports
`1.dev` plus its git sha (`tradingagents/build.py`). The page's footer and `/health` show it. Provision the env once with
`scp .env heinrich@192.168.0.98:edinburgh/.env`, with `OLLAMA_BASE_URL` set to
PAIR's proxy on Colossus (`http://127.0.0.1:11434`).

`tradingagents web --port 8010` serves the positions page: every open position
in the Alpaca account with the desk's rating, stop (the Risk Manager's, the
Trader's, or the 2% default), target and distance to each, a link per ticker
to what every agent decided, and the day's sized book. The worker records the
account on every pass to `positions/<date>.json`, so `/days` lists each day
and `/day/<date>` shows the positions held that day, frozen with the decision
that stood for each; `tradingagents snapshot` records a day by hand. Decisions are read
from the results directory, so `scripts/publish_runs.sh` copies the dev box's
run states and books to Colossus after a list run. `deploy.sh` runs it there
as `edinburgh-web` on port 8010, read-only on the worker's data folder:
http://192.168.0.98:8010/. Colossus runs firewalld and a host-networked
container gets no port opened for it, so once, with sudo on Colossus:
`sudo firewall-cmd --permanent --add-port=8010/tcp && sudo firewall-cmd --reload`.

To run it on this machine instead, `docker compose --profile worker up -d worker`
does the same from the compose file; set `TRADINGAGENTS_DATA_DIR` in `.env` to
the full path of `~/.tradingagents` (Compose does not expand `~`) so the container
shares its state, logs and hedge record with runs on the host.

## The dev box as a PAIR node

PAIR spreads the model calls over every node that is up, and this box
(InfinitiDev, the RTX 5090) takes about half of them when it is. The 3 AM
analysis runs on Colossus, so for it to use this box the box must be awake,
with PAIR running, at 3 AM ET. PAIR here is the desktop app (`/opt/PAIR/nvpair`),
which survives suspend and rejoins the mesh within seconds of a resume, but it
starts only with the desktop session and the box is usually suspended by hand
at night. `scripts/pair-node/install.sh` fixes both: it puts a PAIR entry in
`~/.config/autostart`, and installs `edinburgh-wake.timer`, a system timer with
`WakeSystem` that wakes the box from suspend at 02:50 ET on weekdays (sudo
once). GNOME's idle suspend on AC is off on this box, so after the wake it stays
up until suspended again; a market holiday just means a wake for nothing. A
reboot still needs a login before PAIR is up.

## The P and B range model

`tradingagents pb-range NVDA` reads the 15-minute chart the way Patrick Nill
does (the Robbins World Cup trader; the model is condensed in Trading Notes'
[video](https://www.youtube.com/watch?v=kJyooWDFXyk)). It is arithmetic on
bars, no model, and lives in `tradingagents/pb_range/`:

- **Zones.** A **P** is a strong move up followed by a range, a **B** a strong
  move down followed by one. A range is valid after **three touches** of its
  edges and at least half a session of bars. The impulse is a move of 3 ATRs
  within 16 bars that gives back little; the range is the box of the bars
  after it, at most 35% of the impulse tall; a close outside ends it. Nill does
  not share his drawing rules, so these are the module's own, all in `Params`.
- **Three trades.** The *range trade*, a limit at one edge of a big range for
  the other, in and out, not opened when the range completes in the evening.
  The *breakout with confirmation*: a close outside, a pullback to the edge, a
  close beyond the breakout bar; best on small ranges after a big move. The
  *fake breakout*: a close outside and the next bar back inside, bought at the
  edge with the stop just beyond the fake low, held past the top of the range.
  By default only in the direction of the impulse (a long on a P, a short on a
  B), Nill's favourite; `fake_breakout_against_impulse` turns the other on.
- **Stop first, then size.** The stop leans on structure and is never widened.
  The size is what risks `risk_pct` of equity to it, in lots of three so the
  partials come out even, and never more than the account. Targets run from
  fair value to fair value: the next range's edge, the origin of the move, or
  the impulse projected one to one at a new high. Under 3R to the final
  target, no trade. After three losses in a row the day is over.
- **The big picture.** The 4-hour bias (blocks of 16 bars) and the week's
  volume-profile value area; a range on the value area high or low is noted
  as the stronger setup.

The command prints the zones, the setups of the last session with entry, stop,
targets, R:R and units, and with `--backtest` trades every setup in the window
by the rules (limits, stops first, a third off at each target, break-even
after the first, the stop trailed under each new range, range trades closed
with the session) and reports the R multiples. Intraday bars come from Alpaca;
the free IEX feed carries little volume, so the value area is only as good as
the feed.

**What four months of the S&P 100 said.** The defaults were checked on
15-minute bars of the 102 names in `watchlists/sp100.txt` from July to October
2026, split at 1 September, one parameter at a time and then in combination
(fills at the limit, no costs). Two changes held in every period and
combination and are now the defaults: a range cap of 35% of the impulse
instead of 50%, and fake breakouts with the impulse only. Two defects the
sweep exposed were fixed: a range order filled in the last hour and held
overnight was the worst trade of every run, so range orders no longer fill in
the evening and a range trade still open at the last bar is closed there.
With those defaults the sweep made +56R on July–August and −5R on
September–October over 314 trades, winning 20% of them; five trades supplied
all of the profit and without them it lost 45R. Read that as a method with a
small hit rate and rare ten-to-forty R winners whose edge, if any, this sample
cannot confirm, not as a system to size up. The breakout with confirmation
fired 27 times and lost; the range trade was flat; the against-impulse fake
breakout lost in every cut.

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
