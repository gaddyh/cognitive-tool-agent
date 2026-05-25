from __future__ import annotations

from collections import defaultdict
from typing import Any


def _classify_stage(tool_name: str) -> str:
    if "find_user_id" in tool_name:
        return "auth"
    if tool_name.startswith("get_"):
        return "lookup"
    if tool_name in {
        "exchange_delivered_order_items",
        "return_delivered_order_items",
        "cancel_pending_order",
        "modify_pending_order_items",
        "modify_pending_order_address",
        "modify_pending_order_payment",
    }:
        return "action"
    if tool_name == "transfer_to_human_agents":
        return "escalation"
    if tool_name == "calculate":
        return "reasoning"
    return "unknown"


def compute_failure_heatmap(data: dict[str, Any]) -> list[dict]:
    """
    Aggregate failures along four dimensions:
      tool      — which tool failed most
      stage     — auth / lookup / action / escalation / reasoning
      read_write — read / write / generic
      argument  — which arg had mismatches or was missing
    """
    failures: list[dict] = data["failure_rows"]
    registry: dict = data["tool_registry"]

    by_tool: dict[str, int] = defaultdict(int)
    by_stage: dict[str, int] = defaultdict(int)
    by_rw: dict[str, int] = defaultdict(int)
    by_arg: dict[str, int] = defaultdict(int)

    for row in failures:
        tool = row.get("expected_tool", "unknown")
        by_tool[tool] += 1

        stage = _classify_stage(tool)
        by_stage[stage] += 1

        tool_type = (
            row.get("tool_type")
            or registry.get(tool, {}).get("tool_type")
            or "unknown"
        )
        by_rw[tool_type] += 1

        delta: dict = row.get("argument_delta", {})
        for arg in delta.get("missing", {}):
            by_arg[arg] += 1
        for arg in delta.get("mismatched", {}):
            by_arg[arg] += 1

    rows: list[dict] = []
    for tool, count in sorted(by_tool.items(), key=lambda x: -x[1]):
        rows.append({"dimension": "tool", "value": tool, "count": count})
    for stage, count in sorted(by_stage.items(), key=lambda x: -x[1]):
        rows.append({"dimension": "stage", "value": stage, "count": count})
    for rw, count in sorted(by_rw.items(), key=lambda x: -x[1]):
        rows.append({"dimension": "read_write", "value": rw, "count": count})
    for arg, count in sorted(by_arg.items(), key=lambda x: -x[1]):
        rows.append({"dimension": "argument", "value": arg, "count": count})

    return rows
