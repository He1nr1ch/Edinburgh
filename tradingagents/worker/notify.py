"""Email the worker's findings, over SMTP.

The account is in the environment: TRADINGAGENTS_SMTP_USER and
TRADINGAGENTS_SMTP_PASSWORD (for Gmail, an app password, not the account's).
Without them nothing is sent and the log is the only record.
"""

from __future__ import annotations

import logging
import os
import smtplib
from email.message import EmailMessage

logger = logging.getLogger(__name__)


def email_configured() -> bool:
    return bool(os.environ.get("TRADINGAGENTS_SMTP_USER") and os.environ.get("TRADINGAGENTS_SMTP_PASSWORD"))


def send_email(subject: str, body: str, config: dict) -> bool:
    """Send one message to ``worker_alert_email``; False when email is not set up or fails."""
    user, password = os.environ.get("TRADINGAGENTS_SMTP_USER"), os.environ.get("TRADINGAGENTS_SMTP_PASSWORD")
    to = config.get("worker_alert_email") or user
    if not (user and password and to):
        return False
    message = EmailMessage()
    message["Subject"], message["From"], message["To"] = subject, user, to
    message.set_content(body)
    try:
        with smtplib.SMTP(config["worker_smtp_host"], config["worker_smtp_port"], timeout=30) as smtp:
            smtp.starttls()
            smtp.login(user, password)
            smtp.send_message(message)
        return True
    except (smtplib.SMTPException, OSError) as exc:
        logger.warning("The alert email could not be sent: %s", exc)
        return False
