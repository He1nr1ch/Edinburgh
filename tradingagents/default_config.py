import os

_TRADINGAGENTS_HOME = os.path.join(os.path.expanduser("~"), ".tradingagents")

# Single source of truth for env-var → config-key overrides. To expose
# a new config key for environment-based override, add a row here — no
# entry-point script changes required. Coercion is driven by the type
# of the existing default, so users can keep writing plain strings in
# their .env file.
_ENV_OVERRIDES = {
    "TRADINGAGENTS_LLM_PROVIDER":         "llm_provider",
    "TRADINGAGENTS_DEEP_THINK_LLM":       "deep_think_llm",
    "TRADINGAGENTS_QUICK_THINK_LLM":      "quick_think_llm",
    "TRADINGAGENTS_DEEP_THINK_PROVIDER":       "deep_think_provider",
    "TRADINGAGENTS_QUICK_THINK_PROVIDER":      "quick_think_provider",
    "TRADINGAGENTS_DEEP_THINK_BACKEND_URL":    "deep_think_backend_url",
    "TRADINGAGENTS_QUICK_THINK_BACKEND_URL":   "quick_think_backend_url",
    "TRADINGAGENTS_LLM_BACKEND_URL":      "backend_url",
    "TRADINGAGENTS_MAX_DEBATE_ROUNDS":    "max_debate_rounds",
    "TRADINGAGENTS_MAX_RISK_ROUNDS":      "max_risk_discuss_rounds",
    "TRADINGAGENTS_MAX_TOOL_ROUNDS":      "max_tool_rounds",
    "TRADINGAGENTS_MAX_TOOL_OUTPUT_CHARS": "max_tool_output_chars",
    "TRADINGAGENTS_MAX_TOOL_HISTORY_CHARS": "max_tool_history_chars",
    "TRADINGAGENTS_CHECKPOINT_ENABLED":   "checkpoint_enabled",
    "TRADINGAGENTS_BENCHMARK_TICKER":     "benchmark_ticker",
    "TRADINGAGENTS_ALPACA_FEED":          "alpaca_feed",
    "TRADINGAGENTS_ALPACA_PAPER":         "alpaca_paper",
    "TRADINGAGENTS_FEEDBACK_HOLD_BAND_PCT": "feedback_hold_band_pct",
    "TRADINGAGENTS_TEMPERATURE":          "temperature",
    "TRADINGAGENTS_LLM_MAX_RETRIES":      "llm_max_retries",
    "TRADINGAGENTS_MAX_TOKENS":           "max_tokens",
    # Provider-specific reasoning/thinking knobs (None = each provider's own
    # default). Settable here for non-interactive runs; the CLI also offers an
    # interactive choice, which is skipped when the matching var is set.
    "TRADINGAGENTS_OPENAI_REASONING_EFFORT": "openai_reasoning_effort",
    "TRADINGAGENTS_ANTHROPIC_EFFORT":        "anthropic_effort",
    "TRADINGAGENTS_OLLAMA_NUM_CTX":          "ollama_num_ctx",
    "TRADINGAGENTS_OLLAMA_THINKING":         "ollama_thinking",
    "TRADINGAGENTS_REPORT_STYLE":            "report_style",
    "TRADINGAGENTS_HEDGE_MONTHLY_RATIO":     "hedge_monthly_ratio",
    "TRADINGAGENTS_WORKER_TRADING_ENABLED":  "worker_trading_enabled",
    "TRADINGAGENTS_WORKER_ALLOW_LIVE":       "worker_allow_live",
    "TRADINGAGENTS_WORKER_SYMBOLS":          "worker_symbols",
    "TRADINGAGENTS_WORKER_ALERT_EMAIL":      "worker_alert_email",
}


_BOOL_TRUE = ("true", "1", "yes", "on")
_BOOL_FALSE = ("false", "0", "no", "off")


def _coerce(value: str, reference):
    """Coerce env-var string to the type of the existing default value.

    Invalid values raise ``ValueError`` rather than silently falling back to a
    default — a misspelled boolean (e.g. ``treu``) or non-numeric int should fail
    loudly at startup, not quietly misconfigure an unattended run.
    """
    if isinstance(reference, bool):
        normalized = value.strip().lower()
        if normalized in _BOOL_TRUE:
            return True
        if normalized in _BOOL_FALSE:
            return False
        raise ValueError(
            f"expected a boolean ({'/'.join(_BOOL_TRUE + _BOOL_FALSE)}), got {value!r}"
        )
    if isinstance(reference, int) and not isinstance(reference, bool):
        return int(value)
    if isinstance(reference, float):
        return float(value)
    return value


def _apply_env_overrides(config: dict) -> dict:
    """Apply TRADINGAGENTS_* env vars to the config dict in-place."""
    for env_var, key in _ENV_OVERRIDES.items():
        raw = os.environ.get(env_var)
        if raw is None or raw == "":
            continue
        try:
            config[key] = _coerce(raw, config.get(key))
        except ValueError as exc:
            raise ValueError(f"Invalid value for {env_var}: {exc}") from exc
    return config


def build_default_config() -> dict:
    """The built-in defaults with the TRADINGAGENTS_* environment folded in.

    Read when the package is imported, as DEFAULT_CONFIG; call it again to see
    the environment as it is now.
    """
    return _apply_env_overrides({
        "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR") or os.path.join(_TRADINGAGENTS_HOME, "logs"),
        "data_cache_dir": os.getenv("TRADINGAGENTS_CACHE_DIR") or os.path.join(_TRADINGAGENTS_HOME, "cache"),
        "memory_log_path": os.getenv("TRADINGAGENTS_MEMORY_LOG_PATH") or os.path.join(_TRADINGAGENTS_HOME, "memory", "trading_memory.md"),
        # The SPY calls bought to hedge a book (tradingagents size --record-hedge),
        # so a book that is already hedged is not hedged again.
        "hedge_ledger_path": os.getenv("TRADINGAGENTS_HEDGE_LEDGER_PATH") or os.path.join(_TRADINGAGENTS_HOME, "memory", "hedges.json"),
        "worker_state_path": os.getenv("TRADINGAGENTS_WORKER_STATE_PATH") or os.path.join(_TRADINGAGENTS_HOME, "memory", "worker_state.json"),
        # Optional cap on the number of resolved memory log entries. When set,
        # the oldest resolved entries are pruned once this limit is exceeded.
        # Pending entries are never pruned. None disables rotation entirely.
        "memory_log_max_entries": None,
        # The feedback loop's database: decisions, settled outcomes and per-agent
        # scores, shown back to the deciding agents as a track record. A SQLAlchemy
        # URL; SQLite by default, Postgres when it lands
        # (postgresql+psycopg://user:pass@host/db). Empty disables the store.
        "feedback_db_url": os.getenv("TRADINGAGENTS_FEEDBACK_DB_URL") or "sqlite:///" + os.path.join(_TRADINGAGENTS_HOME, "memory", "feedback.db"),
        # A Hold, or a Neutral stance, is right when the price stays within this
        # band (percent) over the holding window; an Aggressive stance needs a
        # move beyond it.
        "feedback_hold_band_pct": 2.0,
        # How many settled decisions the track record summarises.
        "feedback_context_decisions": 30,
        # LLM settings
        "llm_provider": "openai",
        "deep_think_llm": "gpt-6-sol",
        "quick_think_llm": "gpt-6-luna",
        # When None, each provider's client falls back to its own default endpoint
        # (api.openai.com for OpenAI, generativelanguage.googleapis.com for Gemini, ...).
        # The CLI overrides this per provider when the user picks one. Keeping a
        # provider-specific URL here would leak (e.g. OpenAI's /v1 was previously
        # being forwarded to Gemini, producing malformed request URLs).
        "backend_url": None,
        # A tier may name its own provider and endpoint (#1440): the quick tier serves
        # the analysts, researchers, debaters and trader, the deep tier the managers.
        # None means the tier uses llm_provider and backend_url.
        "quick_think_provider": None,
        "deep_think_provider": None,
        "quick_think_backend_url": None,
        "deep_think_backend_url": None,
        # Provider-specific thinking configuration
        "openai_reasoning_effort": None,    # "medium", "high", "low"
        "anthropic_effort": None,           # "high", "medium", "low"
        # Context window asked of an Ollama server per request. The server's own
        # default (often 4,096) truncates the prompt once tool results arrive,
        # which drops the user message and fails the call; the analysts need
        # tens of thousands of tokens. Lower it if the model does not fit.
        "ollama_num_ctx": 32768,
        # Whether an Ollama model that can think (gemma4, qwen3.x) does so. Off by default:
        # the agents' work is reading reports and writing them, where hidden
        # reasoning adds minutes per call for little gain. Turn it on for the
        # deep tier by running that tier on its own provider settings.
        "ollama_thinking": False,
        # Sampling temperature, forwarded to every provider when set. None leaves
        # each provider at its own default. Lower values reduce run-to-run
        # variation on models that honor it; reasoning models largely ignore it
        # and no setting makes LLM output bit-identical across runs (see README).
        "temperature": None,
        # SDK retry budget forwarded to every provider chat client. None leaves each
        # provider/SDK at its own default (usually 2). Raise it to ride out bursty
        # 429 throttling on rate-limited deployments instead of aborting a run (#1091).
        "llm_max_retries": None,
        # Cap on output tokens forwarded to every provider chat client. None leaves
        # each provider at its own default. Set it to bound a model that emits
        # unbounded reasoning/output and hangs or trips a gateway idle timeout
        # (e.g. some deepseek-v4-flash deployments, #1204).
        "max_tokens": None,
        # Checkpoint/resume: when True, LangGraph saves state after each node
        # so a crashed run can resume from the last successful step.
        "checkpoint_enabled": False,
        # How long the agents write: "concise" gives each report, debate turn and
        # decision a length to keep to; "detailed" leaves it to the model, as the
        # prompts did before. Later agents read every earlier report, so concise
        # output is also shorter input and a faster run.
        "report_style": "concise",
        # Debate and discussion settings
        "max_debate_rounds": 1,
        "max_risk_discuss_rounds": 1,
        "max_recur_limit": 100,
        # Rounds of tool calls an analyst may make before it is asked for its report.
        "max_tool_rounds": 20,
        # Each tool result is cut at this many characters (about 4,000 tokens), and
        # an analyst whose tool results together pass the history budget (about
        # 40,000 tokens) is asked for its report: tool results are what fills a
        # model's context window, and a call past it fails.
        "max_tool_output_chars": 16_000,
        "max_tool_history_chars": 160_000,
        # News / data fetching parameters
        # Increase for longer lookback strategies or to broaden macro coverage;
        # decrease to reduce token usage in agent prompts.
        "news_article_limit": 20,             # max articles per ticker (ticker-news)
        "global_news_article_limit": 10,      # max articles for global/macro news
        "global_news_lookback_days": 7,       # macro news lookback window
        # Search queries used by get_global_news for macro headlines. Extend or
        # replace to broaden geographic / sector coverage.
        "global_news_queries": [
            "Federal Reserve interest rates inflation",
            "S&P 500 earnings GDP economic outlook",
            "geopolitical risk trade war sanctions",
            "ECB Bank of England BOJ central bank policy",
            "oil commodities supply chain energy",
        ],
        # Data vendor configuration
        # Category-level configuration (default for all tools in category).
        # The configured value is the exact vendor chain — requests are NOT silently
        # routed to vendors you didn't choose. For ordered fallback, list several,
        # e.g. "alpaca,other". "default" uses all available vendors.
        "data_vendors": {
            "core_stock_apis": "alpaca",        # Options: alpaca
            "technical_indicators": "alpaca",   # Options: alpaca
            "fundamental_data": "sec_edgar",    # Options: sec_edgar (statements as filed, US filers)
            "news_data": "alpaca",              # Options: alpaca
            "macro_data": "fred",               # Options: fred (needs FRED_API_KEY)
        },
        # Tool-level configuration (takes precedence over category-level)
        "tool_vendors": {
            # Example: "get_news": "alpaca",  # Override category default
        },
        # Alpaca: the equity data feed (sip needs a paid market-data plan; iex is
        # free) and whether the key pair is a paper-trading account's.
        "alpaca_feed": "sip",
        "alpaca_paper": True,
        # Trading days from the first close after the analysis date over which a
        # decision's outcome is measured, for reflection and for the backtest figures.
        "holding_period_days": 5,
        # A net-short book is hedged with the one-week at-the-money SPY call,
        # unless the one-month call costs less than this many times the weekly.
        "hedge_monthly_ratio": 2.1,
        # The daily worker (tradingagents worker). It trades by itself within these
        # limits: on a paper account unless worker_allow_live is turned on.
        "worker_trading_enabled": True,
        "worker_allow_live": False,
        "worker_interval_minutes": 5,
        "worker_trim_fraction": 0.5,        # adverse, notable, unexpected news: sell this share of the position
        "worker_add_fraction": 0.25,        # favorable, urgent, unexpected news: add this share of it
        "worker_position_cap_pct": 10.0,    # an add never takes a position past this % of equity
        "worker_news_lookback_hours": 18,   # the first pass reads back this far, to the previous close
        "worker_symbols": None,             # comma-separated symbols to watch; None watches every equity position
        "worker_alert_email": None,         # None sends alerts to TRADINGAGENTS_SMTP_USER
        "worker_smtp_host": "smtp.gmail.com",
        "worker_smtp_port": 587,
        # Benchmark for alpha calculation in the reflection layer; None means SPY.
        "benchmark_ticker": None,
    })


DEFAULT_CONFIG = build_default_config()
