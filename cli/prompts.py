import datetime
import os
import sys
from pathlib import Path

import questionary
import typer
from dotenv import find_dotenv, set_key

from cli.display import console
from cli.models import AnalystType, AssetType
from tradingagents.llm_clients.api_key_env import get_api_key_env
from tradingagents.llm_clients.model_catalog import get_model_options
from tradingagents.llm_clients.openai_client import OPENAI_COMPATIBLE_PROVIDERS

TICKER_INPUT_EXAMPLES = "SPY, NVDA, BTC/USD"

ANALYST_CHOICES = [
    ("Market Analyst", AnalystType.MARKET),
    ("News Analyst", AnalystType.NEWS),
    ("Fundamentals Analyst", AnalystType.FUNDAMENTALS),
]


# The style of every selection menu.
_MENU_STYLE = questionary.Style([
    ("selected", "fg:magenta noinherit"),
    ("highlighted", "fg:magenta noinherit"),
    ("pointer", "fg:magenta noinherit"),
])


def is_valid_ticker_input(value: str) -> bool:
    """Whether a ticker entry is acceptable (charset + length).

    Allows the characters Alpaca symbols use: letters, digits, ``.`` for share
    classes (``BRK.B``), and ``/`` or ``-`` for crypto pairs (``BTC/USD``).
    Empty input is allowed (it defaults to SPY downstream).
    """
    v = value.strip()
    return not v or (all(ch.isalnum() or ch in "./-" for ch in v) and len(v) <= 32)


def get_ticker() -> str:
    """Prompt the user to enter a ticker symbol."""
    ticker = questionary.text(
        f"Enter ticker symbol (e.g. {TICKER_INPUT_EXAMPLES}):",
        validate=lambda x: (
            is_valid_ticker_input(x)
            or "Please enter a valid ticker symbol, e.g. AAPL, BRK.B, BTC/USD."
        ),
        style=questionary.Style(
            [
                ("text", "fg:green"),
                ("highlighted", "noinherit"),
            ]
        ),
    ).ask()

    if ticker is None:
        console.print("\n[red]No ticker symbol provided. Exiting...[/red]")
        exit(1)

    return normalize_ticker_symbol(ticker) if ticker.strip() else "SPY"


def parse_ticker(value: str) -> str:
    """A ticker given on the command line, canonical; empty or malformed is refused."""
    if not value.strip() or not is_valid_ticker_input(value):
        raise ValueError(f"not a ticker symbol: {value!r} (e.g. {TICKER_INPUT_EXAMPLES})")
    return normalize_ticker_symbol(value)


def parse_analysis_date(value: str) -> str:
    """An analysis date as YYYY-MM-DD, today or earlier."""
    try:
        day = datetime.datetime.strptime(value.strip(), "%Y-%m-%d").date()
    except ValueError:
        raise ValueError(f"not a date: {value!r}; use YYYY-MM-DD") from None
    if day > datetime.date.today():
        raise ValueError(f"{value} is in the future")
    return day.isoformat()


def parse_analysts(value: str, asset_type: AssetType) -> list[AnalystType]:
    """Comma-separated analyst names, in the canonical order, checked against the asset."""
    names = [n for n in (n.strip().lower() for n in value.split(",")) if n]
    if not names:
        raise ValueError("name at least one analyst")
    known = {a.value: a for a in AnalystType}
    unknown = [n for n in names if n not in known]
    if unknown:
        raise ValueError(f"unknown analyst {', '.join(unknown)}; choose from {', '.join(known)}")
    available = filter_analysts_for_asset_type(list(known.values()), asset_type)
    unavailable = [n for n in names if known[n] not in available]
    if unavailable:
        raise ValueError(f"{', '.join(unavailable)} is not available for {asset_type.value}")
    return [a for a in available if a.value in names]


def normalize_ticker_symbol(ticker: str) -> str:
    """Resolve user input to the symbol the data layer prices (single source of truth)."""
    from tradingagents.dataflows.symbols import normalize_symbol

    return normalize_symbol(ticker)


def detect_asset_type(ticker: str) -> AssetType:
    """Classify on the canonical symbol: a crypto pair reads as crypto, anything else as stock."""
    from tradingagents.dataflows.symbols import is_crypto_symbol

    return AssetType.CRYPTO if is_crypto_symbol(normalize_ticker_symbol(ticker)) else AssetType.STOCK


def filter_analysts_for_asset_type(
    analysts: list[AnalystType], asset_type: AssetType
) -> list[AnalystType]:
    if asset_type != AssetType.CRYPTO:
        return analysts
    return [
        analyst
        for analyst in analysts
        if analyst != AnalystType.FUNDAMENTALS
    ]


def select_analysts(asset_type: AssetType = AssetType.STOCK) -> list[AnalystType]:
    """Select analysts using an interactive checkbox."""
    available_analysts = filter_analysts_for_asset_type(
        [value for _, value in ANALYST_CHOICES],
        asset_type,
    )
    choices = questionary.checkbox(
        "Select Your [Analysts Team]:",
        choices=[
            questionary.Choice(display, value=value)
            for display, value in ANALYST_CHOICES
            if value in available_analysts
        ],
        instruction="\n- Press Space to select/unselect analysts\n- Press 'a' to select/unselect all\n- Press Enter when done",
        validate=lambda x: len(x) > 0 or "You must select at least one analyst.",
        style=questionary.Style(
            [
                ("checkbox-selected", "fg:green"),
                ("selected", "fg:green noinherit"),
                ("highlighted", "noinherit"),
                ("pointer", "noinherit"),
            ]
        ),
    ).ask()

    if not choices:
        console.print("\n[red]No analysts selected. Exiting...[/red]")
        exit(1)

    return choices


def select_research_depth() -> int:
    """Select research depth using an interactive selection."""

    DEPTH_OPTIONS = [
        ("Shallow - Quick research, few debate and strategy discussion rounds", 1),
        ("Medium - Middle ground, moderate debate rounds and strategy discussion", 3),
        ("Deep - Comprehensive research, in depth debate and strategy discussion", 5),
    ]

    choice = questionary.select(
        "Select Your [Research Depth]:",
        choices=[
            questionary.Choice(display, value=value) for display, value in DEPTH_OPTIONS
        ],
        instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
        style=questionary.Style(
            [
                ("selected", "fg:yellow noinherit"),
                ("highlighted", "fg:yellow noinherit"),
                ("pointer", "fg:yellow noinherit"),
            ]
        ),
    ).ask()

    if choice is None:
        console.print("\n[red]No research depth selected. Exiting...[/red]")
        exit(1)

    return choice


def _require_text(message: str, hint: str) -> str:
    """Prompt for a required value; exit cleanly if the user cancels."""
    response = questionary.text(
        message,
        validate=lambda x: len(x.strip()) > 0 or hint,
    ).ask()
    if response is None:
        console.print("\n[red]Cancelled. Exiting...[/red]")
        exit(1)
    return response.strip()


def _prompt_custom_model_id() -> str:
    """Prompt user to type a custom model ID."""
    return _require_text("Enter model ID:", "Please enter a model ID.")


def _select_model(provider: str, mode: str) -> str:
    """Select a model for the given provider and mode (quick/deep)."""
    choice = questionary.select(
        f"Select Your [{mode.title()}-Thinking LLM Engine]:",
        choices=[
            questionary.Choice(display, value=value)
            for display, value in get_model_options(provider, mode)
        ],
        instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
        style=_MENU_STYLE,
    ).ask()

    if choice is None:
        console.print(f"\n[red]No {mode} thinking llm engine selected. Exiting...[/red]")
        exit(1)

    if choice == "custom":
        return _prompt_custom_model_id()

    return choice


def select_shallow_thinking_agent(provider) -> str:
    """Select shallow thinking llm engine using an interactive selection."""
    return _select_model(provider, "quick")


def select_deep_thinking_agent(provider) -> str:
    """Select deep thinking llm engine using an interactive selection."""
    return _select_model(provider, "deep")


def _llm_provider_table() -> list[tuple[str, str, str | None]]:
    """(display_name, provider_key, base_url) for every supported provider.

    Shared by the interactive picker and by env-driven configuration so an
    env-set provider resolves to the same default endpoint the menu uses.
    Ollama users point at a remote server via OLLAMA_BASE_URL.
    """
    ollama_url = os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434/v1"
    return [
        ("Ollama (local or LAN server)", "ollama", ollama_url),
        ("Anthropic", "anthropic", "https://api.anthropic.com/"),
        ("OpenAI", "openai", "https://api.openai.com/v1"),
        ("OpenAI-compatible (vLLM, LM Studio, llama.cpp, custom relay)", "openai_compatible", None),
    ]


def provider_default_url(provider_key: str) -> str | None:
    """Return the default backend URL for a provider key, or None if unknown."""
    key = provider_key.lower()
    for _, pk, url in _llm_provider_table():
        if pk == key:
            return url
    return None


def resolve_backend_url(
    provider: str, menu_url: str | None = None, env_url: str | None = None
) -> str | None:
    """Resolve the backend URL with the correct precedence.

    An explicit env override (``env_url``, from ``TRADINGAGENTS_LLM_BACKEND_URL``
    via ``DEFAULT_CONFIG['backend_url']``) is honored regardless of how the
    provider was chosen. Otherwise the menu URL, then the provider's default.
    """
    return env_url or menu_url or provider_default_url(provider)


def prompt_openai_compatible_url() -> str:
    """Prompt for a custom OpenAI-compatible endpoint base URL."""
    url = questionary.text(
        "Enter the OpenAI-compatible base URL "
        "(e.g. http://localhost:8000/v1 for vLLM, http://localhost:1234/v1 for LM Studio):",
        validate=lambda x: x.strip().startswith(("http://", "https://"))
        or "Enter a URL starting with http:// or https://",
    ).ask()
    if not url:
        console.print("\n[red]No endpoint URL provided. Exiting...[/red]")
        exit(1)
    return url.strip()


def select_llm_provider() -> tuple[str, str | None]:
    """Select the LLM provider and its API endpoint."""
    choice = questionary.select(
        "Select your LLM Provider:",
        choices=[questionary.Choice(display, value=(provider_key, url))
                 for display, provider_key, url in _llm_provider_table()],
        instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
        style=_MENU_STYLE,
    ).ask()

    if choice is None:
        console.print("\n[red]No LLM provider selected. Exiting...[/red]")
        exit(1)

    provider, url = choice
    return provider, url


def ask_openai_reasoning_effort() -> str:
    """Ask for OpenAI reasoning effort level."""
    choices = [
        questionary.Choice("Medium (Default)", "medium"),
        questionary.Choice("High (More thorough)", "high"),
        questionary.Choice("Low (Faster)", "low"),
    ]
    return questionary.select(
        "Select Reasoning Effort:",
        choices=choices,
        style=questionary.Style([
            ("selected", "fg:cyan noinherit"),
            ("highlighted", "fg:cyan noinherit"),
            ("pointer", "fg:cyan noinherit"),
        ]),
    ).ask()


def ask_anthropic_effort() -> str | None:
    """Ask for Anthropic effort level."""
    return questionary.select(
        "Select Effort Level:",
        choices=[
        questionary.Choice("High (recommended)", "high"),
        questionary.Choice("Medium (balanced)", "medium"),
        questionary.Choice("Low (faster, cheaper)", "low"),
        ],
        style=questionary.Style([
            ("selected", "fg:cyan noinherit"),
            ("highlighted", "fg:cyan noinherit"),
            ("pointer", "fg:cyan noinherit"),
        ]),
    ).ask()


def confirm_ollama_endpoint(url: str) -> None:
    """Show the resolved Ollama endpoint after provider selection."""
    from_env = os.environ.get("OLLAMA_BASE_URL")
    origin = " (from OLLAMA_BASE_URL)" if from_env and from_env == url else ""
    console.print(f"[green]✓ Using Ollama at {url}{origin}[/green]")

    if not url.startswith(("http://", "https://")):
        console.print(
            f"[yellow]Note: {url!r} is missing a scheme. "
            f"Ollama-serve typically expects a URL like "
            f"http://<host>:11434/v1.[/yellow]"
        )
    elif ":11434" not in url and "://localhost" not in url and "://127.0.0.1" not in url:
        console.print(
            f"[yellow]Note: {url!r} doesn't include port 11434. "
            f"Make sure your remote ollama-serve listens on the port "
            f"shown above.[/yellow]"
        )


def ensure_api_key(provider: str) -> str | None:
    """Make sure the API key for `provider` is available in the environment.

    If the env var is already set, returns its value untouched. Otherwise
    interactively prompts the user, persists the value to the project's
    .env file via python-dotenv's set_key (creating .env if needed), and
    exports it into os.environ so the current process picks it up.

    Returns None for providers that do not require a key (e.g. ollama)
    and for providers not found in the canonical mapping.
    """
    env_var = get_api_key_env(provider)
    if env_var is None:
        return None  # ollama / unknown — no key check possible

    # Key-optional providers (generic OpenAI-compatible / local servers) read the
    # key when present but must never force an interactive prompt.
    spec = OPENAI_COMPATIBLE_PROVIDERS.get(provider.lower())
    if spec is not None and spec.key_optional:
        return os.environ.get(env_var)

    existing = os.environ.get(env_var)
    if existing:
        return existing

    if not sys.stdin.isatty():
        console.print(f"[red]{env_var} is not set; there is no terminal to ask for it.[/red]")
        raise typer.Exit(code=1)

    console.print(
        f"\n[yellow]{env_var} is not set in your environment.[/yellow]"
    )
    key = questionary.password(
        f"Paste your {env_var} (will be saved to .env):",
        style=questionary.Style([
            ("text", "fg:cyan"),
            ("highlighted", "noinherit"),
        ]),
    ).ask()
    if not key:
        console.print(
            f"[red]Skipped. API calls will fail until {env_var} is set.[/red]"
        )
        return None

    env_path = find_dotenv(usecwd=True) or str(Path.cwd() / ".env")
    # The file holds credentials, so make it owner-only before writing: create
    # it 0600 when absent, and tighten an existing one (set_key keeps the mode).
    if not os.path.exists(env_path):
        os.close(os.open(env_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
    os.chmod(env_path, 0o600)
    set_key(env_path, env_var, key)
    os.environ[env_var] = key
    console.print(f"[green]Saved {env_var} to {env_path}[/green]")
    return key
