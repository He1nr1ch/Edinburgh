# From Dublin to Edinburgh: YouTube episode list

36 episodes in six seasons, drawn from the commit histories of Dublin
(April–July 2026) and Edinburgh (October 2026 onward).

## Season 1: Dublin, the gate machine

1. **Why I let an AI trade for me (on paper).** The premise, the paper accounts, and what the series will cover.
2. **From Catalan to Dublin.** What carried over from the earlier project and why it moved to Colossus.
3. **The gate chain.** News sentiment → RSI → VWAP → LLM confirmation, and why a trade needs every gate to agree.
4. **Every decision goes in the database.** The Postgres trading log and what it makes possible later.
5. **Scanning the S&P 500 without timing out.** Batching stock data, and the weekend bug.
6. **Choosing a local model.** qwen3 on Colossus, Nemotron on PGX, and the LLM timeouts.
7. **Letterman.** Ranking stocks 1–100 instead of a yes/no gate, and growing it to 15 stocks.
8. **One source of truth.** Pulling duplicated indicator and database code into one module.
9. **Indicators as plugins.** One file per indicator: VWAP, RSI, Bollinger, TD Sequential, K-reversal, the Fractal Strength Index.

## Season 2: Dublin's hard lessons

10. **Shipping it.** Docker, the deploy script, and the 19.9 GB image caused by a baked-in virtualenv.
11. **The day I committed my `.env`.** What leaked, untracking it twice, and rotating keys.
12. **Shorting was a leak.** Removing shorts, building a backtester, and making the LLM more independent.
13. **Hedging with SPY.** The regime hedger and news-catalyst shorts.
14. **Asking the AI where the money went.** P&L analysis from the database, then by LLM.
15. **VIX, MACD as a percentage, and the daily analysis page.**
16. **Meta-labeling.** Capturing features at entry and labeling outcomes nightly.
17. **When the scheduler lies.** End-of-day jobs that didn't fire, missed slots, and pinning Eastern Time.
18. **What Dublin taught me.** Where gates run out, and why I wanted a desk of agents instead.

## Season 3: The road to Edinburgh

19. **Finding TradingAgents.** The open-source multi-agent desk: analysts, bull and bear researchers, trader, risk, portfolio manager.
20. **Forking it.** Rebuilding around Alpaca data and a trimmed desk; what I removed and why.
21. **Talking to Ollama natively.** Hidden thinking, context overflows, and bounding tool results to fit the window.
22. **The first live run.** The two defects it found, and the SEC filer and BNY ticker oddities.
23. **Closing the loop.** Risk Manager verdicts, numeric sizing, and a scored feedback store.

## Season 4: The hardware and the cluster

24. **Three machines, one desk.** InfinitiDev, Colossus and PGX as a PAIR cluster running gemma4.
25. **Timing everything.** Stage timings, short reports, and what a full sweep costs.
26. **The sweep as a pool of processes.** Around-the-clock analysis, the direct lane to the fast desktop, one-call analysts.
27. **Deploying to Fedora.** SELinux labels, firewalld, image pruning, and Dublin-style build numbers.

## Season 5: The desk learns to trade

28. **Sell means short.** Telling the desk it runs both ways, with the stop above.
29. **The rules of the book.** Targets over 5%, top 10 per side, a stop on every order, nothing before 10:30.
30. **The Watcher.** The self-running desk: the overnight job, the book at the open, stop, target and news guards, and the bench.
31. **Patrick Nill's P and B ranges.** Trading ranges on the 15-minute chart and tuning on four months of the S&P 100.
32. **The positions page.** Every open position linked to what each agent decided, plus the daily transaction list.
33. **Valya, the mirror account.** A second account that holds the exact opposite of the desk.

## Season 6: Teaching the machine

34. **Reports that teach.** Daily and weekly reviews whose lessons flow back to the agents.
35. **Typed contracts.** Every answer carrying confidence, evidence, model and latency; one inference client; Postgres for feedback.
36. **Results in R, and what comes next.** Scoring each call against its own risk, the honest scoreboard so far, and the road ahead.

## Notes for filming

- Rotate the Dublin credentials listed in Dublin's `TODO.md` before episode 11 shows that history on screen.
- Episodes 1, 18 and 36 work as stand-alone entry points for trailers or a shorter cut.
