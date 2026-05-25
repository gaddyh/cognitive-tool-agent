from __future__ import annotations

import math
from typing import Any


def compute_extended_summary(data: dict[str, Any]) -> dict[str, Any]:
    """Extend the base conversion_summary with tool entropy and call-pattern stats."""
    registry: dict = data["tool_registry"]
    sequences: list = data["action_sequences"]
    base: dict = data["conversion_summary"]

    # --- tool entropy (Shannon, bits) ---
    total_calls = sum(e.get("usage_count", 0) for e in registry.values())
    if total_calls > 0:
        probs = [e["usage_count"] / total_calls for e in registry.values() if e.get("usage_count", 0) > 0]
        tool_entropy = -sum(p * math.log2(p) for p in probs)
    else:
        tool_entropy = 0.0

    # --- avg tool calls per simulation ---
    n_sims = max(base.get("simulations_count", 1), 1)
    avg_tools_per_sim = base.get("actual_tool_calls_count", 0) / n_sims

    # --- avg turns before first write action ---
    write_tools = {name for name, e in registry.items() if e.get("tool_type") == "write"}
    first_write_turns: list[int] = []
    for seq in sequences:
        candidates = [
            aa["actual_turn_idx"]
            for aa in seq.get("aligned_actions", [])
            if aa.get("actual_tool") in write_tools and aa.get("actual_turn_idx") is not None
        ]
        if candidates:
            first_write_turns.append(min(candidates))
    avg_turns_before_write = (
        sum(first_write_turns) / len(first_write_turns) if first_write_turns else 0.0
    )

    # --- read / write call ratio ---
    read_calls = sum(
        e.get("usage_count", 0) for e in registry.values() if e.get("tool_type") == "read"
    )
    write_calls = sum(
        e.get("usage_count", 0) for e in registry.values() if e.get("tool_type") == "write"
    )
    rw_ratio = read_calls / max(write_calls, 1)

    return {
        **base,
        "tool_entropy_bits": round(tool_entropy, 3),
        "avg_tools_per_simulation": round(avg_tools_per_sim, 2),
        "avg_turns_before_write_action": round(avg_turns_before_write, 1),
        "read_tool_calls": read_calls,
        "write_tool_calls": write_calls,
        "read_write_ratio": round(rw_ratio, 2),
    }
