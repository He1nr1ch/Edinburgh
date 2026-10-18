"""The interactive choices for a run: ticker, date, analysts, depth, provider and models."""

import datetime
import os
from pathlib import Path

import typer
from rich.align import Align
from rich.panel import Panel

from cli.display import (
    console,
)
from cli.prompts import (
    ask_anthropic_effort,
    ask_openai_reasoning_effort,
    confirm_ollama_endpoint,
    detect_asset_type,
    ensure_api_key,
    get_ticker,
    parse_analysis_date,
    parse_analysts,
    parse_ticker,
    prompt_openai_compatible_url,
    resolve_backend_url,
    select_analysts,
    select_deep_thinking_agent,
    select_llm_provider,
    select_research_depth,
    select_shallow_thinking_agent,
)
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.llm_clients.factory import REASONING_EFFORT_PROVIDERS


def get_user_selections(flags=None):
    """Ask for the run's settings."""
    return _prompt_selections(flags or {})


def depth_from_env() -> bool:
    """Both round counts come from the environment, so the depth question is skipped."""
    return bool(os.environ.get("TRADINGAGENTS_MAX_DEBATE_ROUNDS")
                and os.environ.get("TRADINGAGENTS_MAX_RISK_ROUNDS"))


def unattended_gaps(flags) -> list[str]:
    """The flags and environment variables a run with no terminal still needs."""
    env = os.environ.get
    gaps = [f"--{name}" for name in ("ticker", "date", "analysts") if flags.get(name) is None]
    gaps += [f"--{name} or --no-{name}" for name in ("save", "show") if flags.get(name) is None]
    if not depth_from_env():
        gaps.append("TRADINGAGENTS_MAX_DEBATE_ROUNDS and TRADINGAGENTS_MAX_RISK_ROUNDS")
    if not env("TRADINGAGENTS_LLM_PROVIDER"):
        gaps.append("TRADINGAGENTS_LLM_PROVIDER")
    if not (env("TRADINGAGENTS_QUICK_THINK_LLM") or env("TRADINGAGENTS_DEEP_THINK_LLM")):
        gaps.append("TRADINGAGENTS_QUICK_THINK_LLM or TRADINGAGENTS_DEEP_THINK_LLM")
    return gaps


def _check_tier_providers(main_provider: str) -> None:
    """Check each provider the model tiers use before the run.

    A tier on another provider needs its model from its variable, since the
    model question offers the main provider's models. Each provider a tier uses
    has its key checked now, prompting for a missing key, rather than at its
    first call, which for the deep tier comes after everything else has been
    paid for; a provider no tier uses needs no key.
    """
    used = []
    for tier in ("quick", "deep"):
        provider = (DEFAULT_CONFIG.get(f"{tier}_think_provider") or main_provider).lower()
        if provider != main_provider.lower():
            variable = f"TRADINGAGENTS_{tier.upper()}_THINK_LLM"
            if not os.environ.get(variable):
                console.print(f"[red]The {tier} tier runs on {provider}; set {variable} to one of its models.[/red]")
                raise typer.Exit(code=1)
        if provider not in used:
            used.append(provider)
    for provider in used:
        ensure_api_key(provider)


def _from_flag(parse, value, *args):
    """A flag's value through the same check its prompt applies; a bad one ends the run."""
    try:
        return parse(value, *args)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None


def _prompt_selections(flags):
    """Walk the selection steps. Flags and the environment skip their questions."""
    with open(Path(__file__).parent / "static" / "welcome.txt", encoding="utf-8") as f:
        welcome_ascii = f.read()

    welcome_content = f"{welcome_ascii}\n"
    welcome_content += "[bold green]Edinburgh: multi-agent trading desk - CLI[/bold green]\n\n"
    welcome_content += "[bold]Workflow Steps:[/bold]\n"
    welcome_content += "I. Analyst Team → II. Research Team → III. Trader → IV. Risk Management → V. Portfolio Management\n"

    welcome_box = Panel(
        welcome_content,
        border_style="green",
        padding=(1, 2),
        title="Welcome to Edinburgh",
        subtitle="Multi-agent trading desk",
    )
    console.print(Align.center(welcome_box))
    console.print()
    console.print()

    def create_question_box(title, prompt, default=None):
        box_content = f"[bold]{title}[/bold]\n"
        box_content += f"[dim]{prompt}[/dim]"
        if default:
            box_content += f"\n[dim]Default: {default}[/dim]"
        return Panel(box_content, border_style="blue", padding=(1, 2))

    def thinking_value_or_prompt(env_var, config_key, label, box_title, box_body, prompt_fn):
        """Return the env-configured reasoning/thinking value, or prompt for it."""
        if os.environ.get(env_var):
            value = DEFAULT_CONFIG[config_key]
            console.print(f"[green]✓ {label} from environment:[/green] {value}")
            return value
        console.print(create_question_box(box_title, box_body))
        return prompt_fn()

    # Step 1: Ticker symbol
    if flags.get("ticker") is not None:
        selected_ticker = _from_flag(parse_ticker, flags["ticker"])
        console.print(f"[green]✓ Ticker from --ticker:[/green] {selected_ticker}")
    else:
        console.print(
            create_question_box(
                "Step 1: Ticker Symbol",
                "Enter the ticker (e.g. SPY, NVDA, BTC/USD)",
                "SPY",
            )
        )
        selected_ticker = get_ticker()
    asset_type = detect_asset_type(selected_ticker)
    if asset_type.value != "stock":
        console.print(
            f"[green]Detected asset type:[/green] {asset_type.value}"
        )

    # Step 2: Analysis date
    if flags.get("date") is not None:
        analysis_date = _from_flag(parse_analysis_date, flags["date"])
        console.print(f"[green]✓ Analysis date from --date:[/green] {analysis_date}")
    else:
        default_date = datetime.datetime.now().strftime("%Y-%m-%d")
        console.print(
            create_question_box(
                "Step 2: Analysis Date",
                "Enter the analysis date (YYYY-MM-DD)",
                default_date,
            )
        )
        analysis_date = get_analysis_date()

    # Step 3: Select analysts
    if flags.get("analysts") is not None:
        selected_analysts = _from_flag(parse_analysts, flags["analysts"], asset_type)
    else:
        console.print(
            create_question_box(
                "Step 3: Analysts Team", "Select your LLM analyst agents for the analysis"
            )
        )
        selected_analysts = select_analysts(asset_type)
    console.print(
        f"[green]Selected analysts:[/green] {', '.join(analyst.value for analyst in selected_analysts)}"
    )

    # Step 4: Research depth (skipped when both round counts are set via env).
    if depth_from_env():
        selected_research_depth = DEFAULT_CONFIG["max_debate_rounds"]
        console.print(
            f"[green]✓ Research depth from environment:[/green] "
            f"{DEFAULT_CONFIG['max_debate_rounds']} debate / "
            f"{DEFAULT_CONFIG['max_risk_discuss_rounds']} risk rounds"
        )
    else:
        console.print(
            create_question_box(
                "Step 4: Research Depth", "Select your research depth level"
            )
        )
        selected_research_depth = select_research_depth()

    # Step 5: LLM Provider (skipped when set via TRADINGAGENTS_LLM_PROVIDER).
    provider_from_env = bool(os.environ.get("TRADINGAGENTS_LLM_PROVIDER"))
    if provider_from_env:
        selected_llm_provider = DEFAULT_CONFIG["llm_provider"].lower()
        backend_url = resolve_backend_url(
            selected_llm_provider, env_url=DEFAULT_CONFIG["backend_url"]
        )
        console.print(f"[green]✓ LLM provider from environment:[/green] {selected_llm_provider}")
        console.print(f"[green]✓ Backend URL:[/green] {backend_url}")
    else:
        console.print(
            create_question_box(
                "Step 5: LLM Provider", "Select your LLM provider"
            )
        )
        selected_llm_provider, backend_url = select_llm_provider()

        # Honor an explicit env backend URL even when the provider was chosen
        # interactively, so it isn't overwritten by the menu default.
        backend_url = resolve_backend_url(
            selected_llm_provider, backend_url, env_url=DEFAULT_CONFIG["backend_url"]
        )

        # The generic OpenAI-compatible endpoint has no default; ask for it if
        # neither the menu nor the environment supplied one.
        if selected_llm_provider == "openai_compatible" and not backend_url:
            backend_url = prompt_openai_compatible_url()

        if selected_llm_provider == "ollama":
            confirm_ollama_endpoint(backend_url)

    _check_tier_providers(selected_llm_provider)

    # Step 6: Thinking agents (skipped when either model is set via environment)
    if os.environ.get("TRADINGAGENTS_QUICK_THINK_LLM") or os.environ.get("TRADINGAGENTS_DEEP_THINK_LLM"):
        selected_shallow_thinker = DEFAULT_CONFIG["quick_think_llm"]
        selected_deep_thinker = DEFAULT_CONFIG["deep_think_llm"]
        console.print(
            f"[green]✓ Thinking agents from environment:[/green] "
            f"quick={selected_shallow_thinker}, deep={selected_deep_thinker}"
        )
    else:
        console.print(
            create_question_box(
                "Step 6: Thinking Agents", "Select your thinking agents for analysis"
            )
        )
        selected_shallow_thinker = select_shallow_thinking_agent(selected_llm_provider)
        selected_deep_thinker = select_deep_thinking_agent(selected_llm_provider)

    # Step 7: Provider-specific reasoning configuration. Each knob is settable
    # via its TRADINGAGENTS_* env var; when that var is set (or the provider
    # itself came from env) the prompt is skipped. None = the provider's default.
    reasoning_effort = None
    anthropic_effort = None

    provider_lower = selected_llm_provider.lower()
    if provider_from_env:
        reasoning_effort = DEFAULT_CONFIG["openai_reasoning_effort"]
        anthropic_effort = DEFAULT_CONFIG["anthropic_effort"]
    elif provider_lower in REASONING_EFFORT_PROVIDERS:
        reasoning_effort = thinking_value_or_prompt(
            "TRADINGAGENTS_OPENAI_REASONING_EFFORT", "openai_reasoning_effort",
            "Reasoning effort", "Step 7: Reasoning Effort",
            "Configure OpenAI reasoning effort level", ask_openai_reasoning_effort,
        )
    elif provider_lower == "anthropic":
        anthropic_effort = thinking_value_or_prompt(
            "TRADINGAGENTS_ANTHROPIC_EFFORT", "anthropic_effort",
            "Claude effort", "Step 7: Effort Level",
            "Configure Claude effort level", ask_anthropic_effort,
        )

    return {
        "ticker": selected_ticker,
        "asset_type": asset_type.value,
        "analysis_date": analysis_date,
        "analysts": selected_analysts,
        "research_depth": selected_research_depth,
        "llm_provider": selected_llm_provider.lower(),
        "backend_url": backend_url,
        "quick_think_llm": selected_shallow_thinker,
        "deep_think_llm": selected_deep_thinker,
        "openai_reasoning_effort": reasoning_effort,
        "anthropic_effort": anthropic_effort,
    }


def get_analysis_date():
    """Get the analysis date from user input."""
    while True:
        date_str = typer.prompt(
            "", default=datetime.datetime.now().strftime("%Y-%m-%d")
        )
        try:
            return parse_analysis_date(date_str)
        except ValueError as exc:
            console.print(f"[red]Error: {exc}[/red]")
