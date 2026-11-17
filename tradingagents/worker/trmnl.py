"""Show the open positions on a TRMNL e-ink display.

TRMNL's servers draw the screen, and they cannot reach Colossus, so the worker
sends the positions out instead: after every pass it posts them to a TRMNL
Private Plugin's webhook, whose template (docs/trmnl/positions.liquid) lays
them out. The webhook's address is in the environment, TRMNL_WEBHOOK_URL;
without it nothing is sent.

TRMNL takes at most 2 kB a post and 12 posts an hour, so the rows are short
lists of ready-made text and an unchanged screen is not sent again.
"""

from __future__ import annotations

import json
import logging
import os

import requests

logger = logging.getLogger(__name__)

MAX_BYTES = 2000
_last_sent: str | None = None


def _pct(value) -> str:
    return "" if value is None else f"{value:+.1f}"


def payload(record: dict) -> dict:
    """The snapshot of a day as TRMNL's merge variables: one short row a position."""
    rows = [[r["ticker"], "L" if r["qty"] > 0 else "S", _pct(r.get("pnl_pct")),
             _pct(r.get("to_stop_pct")), _pct(r.get("to_target_pct"))]
            for r in record.get("positions", [])]
    return {"merge_variables": {
        "as_of": str(record.get("as_of", ""))[11:16], "date": record.get("date", ""),
        "account": "Paper" if record.get("paper") else "LIVE",
        "equity": f'{record.get("equity") or 0:,.0f}', "cash": f'{record.get("cash") or 0:,.0f}',
        "rows": rows,
    }}


def push(record: dict) -> bool:
    """Post the positions to the TRMNL webhook; False when not set up, unchanged or refused."""
    global _last_sent
    url = os.environ.get("TRMNL_WEBHOOK_URL")
    if not url:
        return False
    body = payload(record)
    # The time alone changing is not worth one of the hour's twelve posts.
    key = json.dumps({**body["merge_variables"], "as_of": ""}, sort_keys=True)
    if key == _last_sent:
        return False
    text = json.dumps(body, separators=(",", ":"))
    if len(text.encode()) > MAX_BYTES:
        logger.warning("The TRMNL screen is not sent: %d bytes is over TRMNL's limit of %d.", len(text.encode()), MAX_BYTES)
        return False
    try:
        response = requests.post(url, data=text, headers={"Content-Type": "application/json"}, timeout=20)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("The TRMNL screen could not be sent: %s", exc)
        return False
    _last_sent = key
    return True
