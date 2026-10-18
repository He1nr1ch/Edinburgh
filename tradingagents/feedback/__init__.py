"""The feedback loop: every decision recorded with its typed calls, settled against
the market, scored per agent, and shown back to the deciding agents as a track record.

``schema`` is the Postgres-shaped table layout (SQLite until Postgres lands),
``scoring`` the pure rules that turn a price window into verdicts, ``extract``
reads the typed calls out of the agents' rendered text, and ``store`` is the
database behind it all.
"""

from tradingagents.feedback.store import FeedbackStore

__all__ = ["FeedbackStore"]
