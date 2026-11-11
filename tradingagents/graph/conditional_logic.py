from tradingagents.agents.state import AgentState


class ConditionalLogic:
    """Handles conditional logic for determining graph flow."""

    def __init__(self, max_debate_rounds=1, max_risk_discuss_rounds=1):
        """Initialize with configuration parameters."""
        self.max_debate_rounds = max_debate_rounds
        self.max_risk_discuss_rounds = max_risk_discuss_rounds

    @staticmethod
    def closed(state: AgentState) -> bool:
        """Whether an agent's answer failed its contract: the run then goes straight to its end."""
        return bool(state.get("contract_failure"))

    def after_research_manager(self, state: AgentState) -> str:
        return "Portfolio Manager" if self.closed(state) else "Trader"

    def after_trader(self, state: AgentState) -> str:
        return "Portfolio Manager" if self.closed(state) else "Aggressive Analyst"

    def should_continue_debate(self, state: AgentState) -> str:
        """Determine if debate should continue."""
        if self.closed(state):
            return "Portfolio Manager"

        if (
            state["investment_debate_state"]["count"] >= 2 * self.max_debate_rounds
        ):  # max_debate_rounds turns each for bull and bear
            return "Research Manager"
        if state["investment_debate_state"]["current_response"].startswith("Bull"):
            return "Bear Researcher"
        return "Bull Researcher"

    def should_continue_risk_analysis(self, state: AgentState) -> str:
        """Determine if risk analysis should continue."""
        if self.closed(state):
            return "Portfolio Manager"
        if (
            state["risk_debate_state"]["count"] >= 3 * self.max_risk_discuss_rounds
        ):  # max_risk_discuss_rounds turns each for the three risk analysts
            return "Risk Manager"
        if state["risk_debate_state"]["latest_speaker"].startswith("Aggressive"):
            return "Conservative Analyst"
        if state["risk_debate_state"]["latest_speaker"].startswith("Conservative"):
            return "Neutral Analyst"
        return "Aggressive Analyst"
