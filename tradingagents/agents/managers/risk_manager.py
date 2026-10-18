"""Risk Manager: settles the risk debate into a stance and a position ceiling.

The aggressive, neutral and conservative analysts argue the trader's proposal;
this node decides the level of risk the desk takes on it, as a typed
``RiskAssessment`` (risk level, maximum position, stop). The Portfolio Manager
then sizes within that ceiling, and the feedback loop can score the stance.
"""

from __future__ import annotations

from tradingagents.agents.context import (
    get_instrument_context_from_state,
    get_portfolio_context_from_state,
)
from tradingagents.agents.rating import parse_risk_level
from tradingagents.agents.schemas import RiskAssessment, render_risk_assessment
from tradingagents.agents.structured import NO_EXTERNAL_TOOLS, bind_structured, invoke_structured


def create_risk_manager(llm):
    structured_llm = bind_structured(llm, RiskAssessment, "Risk Manager")

    def risk_manager_node(state) -> dict:
        instrument_context = get_instrument_context_from_state(state)
        portfolio_context = get_portfolio_context_from_state(state)
        history = state["risk_debate_state"]["history"]
        trader_plan = state["trader_investment_plan"]
        market_report = (state.get("market_report") or "").strip()
        market_section = f"**Technical market report** (for price levels):\n{market_report}\n\n" if market_report else ""
        past_context = (state.get("past_context") or "").strip()
        record_section = f"**Track record and lessons from settled decisions:**\n{past_context}\n\n" if past_context else ""

        prompt = f"""As the Risk Manager, settle the risk analysts' debate: decide the level of risk the desk takes on the trader's proposal, and the ceiling on the position.

{instrument_context}

{portfolio_context}

---

**Risk levels** (use exactly one):
- **Aggressive**: the upside case is strong enough to carry a full-size position with wide stops
- **Neutral**: a standard position with ordinary stops
- **Conservative**: the downside case warrants a small position, tight stops, or no position at all

**Trader's transaction proposal:**
{trader_plan}

{market_section}{record_section}**Risk analysts' debate:**
{history}

---

Weigh the three stances on their evidence, independent of speaking order. The debate always contains conflicting stances; choosing between them is the job. Set the maximum position as a percentage of total portfolio equity, one number from 0 to 100, and a stop-loss as an absolute price when the market report gives the levels to ground one.

## Output

Write these sections, in this order, starting with the risk level on its own line:

- **Risk Level**: exactly one of Aggressive / Neutral / Conservative
- **Maximum Position**: the ceiling as a percentage of portfolio equity, e.g. 5
- **Stop Loss**: an absolute price, or not provided
- **Rationale**: the risks that decided the stance and bound the size

{NO_EXTERNAL_TOOLS}"""

        assessment = invoke_structured(structured_llm, prompt, "Risk Manager")
        if assessment is not None:
            rendered = render_risk_assessment(assessment)
            level = assessment.risk_level.value
        else:
            rendered = llm.invoke(prompt).content
            level = parse_risk_level(rendered)

        return {"risk_assessment": rendered, "risk_level": level}

    return risk_manager_node
