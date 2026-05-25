from __future__ import annotations

from collections import defaultdict
from typing import Any


_BURDEN_LEVELS = ("low", "medium", "high", "very_high")


def _level(value: float, thresholds: tuple[float, float, float]) -> str:
    if value <= thresholds[0]:
        return "low"
    if value <= thresholds[1]:
        return "medium"
    if value <= thresholds[2]:
        return "high"
    return "very_high"


def _burden_score(level: str) -> int:
    return _BURDEN_LEVELS.index(level)


def compute_cognitive_burden(data: dict[str, Any]) -> dict[str, dict]:
    """
    For each tool, compute five heuristic burden dimensions and a complexity score.

    Dimensions:
      extraction_burden  — number of required args
      memory_burden      — avg turns between entity hint and tool call
      readiness_burden   — write tools require confirmation
      reasoning_burden   — avg depth (preceding tool calls) when this tool fires
      grounding_burden   — fraction of required args never seen in entity hints

    Complexity score (integer):
      required_args_count
      + memory_burden_score (0–3)
      + write_penalty (2 if write, else 0)
      + grounding_penalty (fraction × n_args, rounded)
      + confirmation_penalty (1 if write)
      + chain_depth_score (min(avg_depth, 3))
    """
    registry: dict = data["tool_registry"]
    sequences: list = data["action_sequences"]
    supervision: list = data["turn_supervision"]

    sup_by_sim: dict[str, list] = defaultdict(list)
    for row in supervision:
        sup_by_sim[row["simulation_id"]].append(row)

    results: dict[str, dict] = {}

    for tool_name, entry in registry.items():
        required_args: list[str] = entry.get("required_args", [])
        tool_type: str | None = entry.get("tool_type")
        usage_count: int = entry.get("usage_count", 0)

        turn_distances: list[float] = []
        chain_depths: list[int] = []
        arg_explicit: dict[str, int] = defaultdict(int)
        arg_total: dict[str, int] = defaultdict(int)

        for seq in sequences:
            sim_id = seq["simulation_id"]
            sup_rows = sup_by_sim.get(sim_id, [])
            aligned = seq.get("aligned_actions", [])

            for action_idx, aa in enumerate(aligned):
                if aa.get("actual_tool") != tool_name:
                    continue
                call_turn: int | None = aa.get("actual_turn_idx")
                if call_turn is None:
                    continue

                # memory burden: turns between last user hint and this call
                hint_turns = [
                    r["turn_idx"]
                    for r in sup_rows
                    if r["role"] == "user"
                    and r["turn_idx"] < call_turn
                    and any(
                        arg in (r["cognitive_label"].get("perception_entity_hints") or {})
                        for arg in required_args
                    )
                ]
                if hint_turns:
                    turn_distances.append(call_turn - max(hint_turns))
                else:
                    turn_distances.append(float(call_turn))

                # reasoning burden: how many tool calls preceded this one
                preceding = sum(
                    1 for prev in aligned[:action_idx]
                    if prev.get("actual_tool") is not None
                )
                chain_depths.append(preceding)

                # grounding: which required args appeared explicitly?
                pre_user_hints: set[str] = set()
                for r in sup_rows:
                    if r["role"] == "user" and r["turn_idx"] < call_turn:
                        pre_user_hints.update(
                            (r["cognitive_label"].get("perception_entity_hints") or {}).keys()
                        )

                for arg in required_args:
                    arg_total[arg] += 1
                    if arg in pre_user_hints:
                        arg_explicit[arg] += 1

        n_args = len(required_args)
        avg_turn_dist = sum(turn_distances) / len(turn_distances) if turn_distances else 0.0
        avg_chain_depth = sum(chain_depths) / len(chain_depths) if chain_depths else 0.0

        total_arg_instances = sum(arg_total[a] for a in required_args)
        total_explicit = sum(arg_explicit[a] for a in required_args)
        grounding_fraction = (
            1.0 - total_explicit / total_arg_instances
            if total_arg_instances > 0
            else 0.0
        )

        extraction_burden = _level(n_args, (1, 2, 3))
        memory_burden = _level(avg_turn_dist, (2.0, 4.0, 7.0))
        readiness_burden = "high" if tool_type == "write" else "low"
        reasoning_burden = _level(avg_chain_depth, (1.0, 3.0, 5.0))
        grounding_burden = _level(grounding_fraction, (0.25, 0.5, 0.75))

        complexity_score = (
            n_args
            + _burden_score(memory_burden)
            + (2 if tool_type == "write" else 0)
            + round(grounding_fraction * n_args)
            + (1 if tool_type == "write" else 0)
            + min(round(avg_chain_depth), 3)
        )

        results[tool_name] = {
            "tool_name": tool_name,
            "tool_type": tool_type or "unknown",
            "required_args_count": n_args,
            "usage_count": usage_count,
            "avg_turn_distance": round(avg_turn_dist, 1),
            "avg_chain_depth": round(avg_chain_depth, 1),
            "grounding_fraction": round(grounding_fraction, 3),
            "extraction_burden": extraction_burden,
            "memory_burden": memory_burden,
            "readiness_burden": readiness_burden,
            "reasoning_burden": reasoning_burden,
            "grounding_burden": grounding_burden,
            "complexity_score": complexity_score,
        }

    return results
