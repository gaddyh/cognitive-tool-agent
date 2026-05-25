from __future__ import annotations

from collections import defaultdict
from typing import Any


def compute_argument_emergence(data: dict[str, Any]) -> list[dict]:
    """
    For each required argument (aggregated across all tools that use it),
    classify how it emerges deterministically:

      appears_explicitly    — value substring-matched in user entity hints before the call
      requires_tool_chaining — value found verbatim in a preceding tool result
      requires_grounding    — matched action but arg not explicit and not chained
                              (implies multi-turn context or NL → structured mapping)
      requires_inference    — not classified by the above (failed or ambiguous)

    These four counts sum to total_instances.
    """
    registry: dict = data["tool_registry"]
    sequences: list = data["action_sequences"]
    supervision: list = data["turn_supervision"]

    sup_by_sim: dict[str, list] = defaultdict(list)
    for row in supervision:
        sup_by_sim[row["simulation_id"]].append(row)

    stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "explicit": 0, "tool_chaining": 0, "grounding": 0, "inference": 0}
    )

    for seq in sequences:
        sim_id = seq["simulation_id"]
        sup_rows = sup_by_sim.get(sim_id, [])

        for aa in seq.get("aligned_actions", []):
            tool_name = aa.get("actual_tool")
            call_turn = aa.get("actual_turn_idx")
            if tool_name is None or call_turn is None:
                continue

            tool_entry = registry.get(tool_name, {})
            required_args: list[str] = tool_entry.get("required_args", [])
            actual_args: dict = aa.get("actual_arguments", {}) or {}
            action_match: bool = aa.get("action_match", False)

            pre_user_rows = [
                r for r in sup_rows
                if r["role"] == "user" and r["turn_idx"] < call_turn
            ]
            pre_tool_rows = [
                r for r in sup_rows
                if r["role"] == "tool" and r["turn_idx"] < call_turn
            ]

            pre_user_hints: set[str] = set()
            for r in pre_user_rows:
                pre_user_hints.update(
                    (r["cognitive_label"].get("perception_entity_hints") or {}).keys()
                )

            pre_tool_content = " ".join(
                (r.get("content") or "") for r in pre_tool_rows
            ).lower()

            for arg in required_args:
                s = stats[arg]
                s["total"] += 1

                if arg in pre_user_hints:
                    s["explicit"] += 1
                    continue

                arg_val = actual_args.get(arg)
                if arg_val is not None:
                    str_val = str(arg_val).lower()
                    if str_val and str_val in pre_tool_content:
                        s["tool_chaining"] += 1
                        continue

                if action_match:
                    s["grounding"] += 1
                else:
                    s["inference"] += 1

    rows = []
    for arg_name, s in sorted(stats.items()):
        total = s["total"]
        if total == 0:
            continue

        def pct(n: int) -> float:
            return round(100.0 * n / total, 1)

        rows.append({
            "arg_name": arg_name,
            "total_instances": total,
            "appears_explicitly": s["explicit"],
            "appears_explicitly_pct": pct(s["explicit"]),
            "requires_tool_chaining": s["tool_chaining"],
            "requires_tool_chaining_pct": pct(s["tool_chaining"]),
            "requires_grounding": s["grounding"],
            "requires_grounding_pct": pct(s["grounding"]),
            "requires_inference": s["inference"],
            "requires_inference_pct": pct(s["inference"]),
        })

    return rows
