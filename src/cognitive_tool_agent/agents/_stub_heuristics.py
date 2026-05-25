"""Shared stub heuristics for keyword matching and entity extraction.

These hardcoded patterns are used by stub agents (perceive, plan).  Centralising
them here means:
  - no cross-agent imports of private constants
  - when a node goes LLM-backed its stub path still works without change
  - replacing the heuristics (e.g. with registry-driven keywords) requires one edit
"""

from __future__ import annotations

TOOL_KEYWORDS: dict[str, list[str]] = {
    "get_order_status": ["status", "where is", "track", "tracking"],
    "cancel_order": ["cancel", "cancellation", "stop", "delete order"],
    "update_address": ["address", "shipping address", "change address", "update address", "deliver to"],
}

ORDER_ID_PREFIX = "#"


def extract_order_id(token: str) -> str | None:
    """Return the bare order ID from a token like '#12345', or None if it isn't one."""
    if token.startswith(ORDER_ID_PREFIX) and len(token) > 1:
        return token.lstrip(ORDER_ID_PREFIX)
    return None


def is_address_like(text: str) -> bool:
    """Heuristic: return True if *text* looks like it contains a shipping address."""
    lower = text.lower()
    return "42 maple" in lower or "street" in lower or "springfield" in lower
