from __future__ import annotations

from ..schemas.graph_builder import FailureMap, GraphCandidate, GraphRevision


_FAILURE_TO_UPGRADE: dict[str, tuple[str, str, str]] = {
    "policy_blocked_execution": (
        "candidate_C",
        "add_judge",
        "Policy failures require a dedicated readiness/judge stage. Upgrade to full pipeline.",
    ),
    "missing_fields_blocked_execution": (
        "candidate_C",
        "add_judge",
        "Missing field failures need a readiness gate to block premature execution.",
    ),
    "premature_execution": (
        "candidate_C",
        "add_judge",
        "Agent executed without confirmation — add readiness judge to enforce policy.",
    ),
    "wrong_tool_selected": (
        "candidate_C",
        "split_node",
        "Wrong tool selection indicates weak intent reasoning. Full pipeline adds a dedicated reason stage.",
    ),
    "wrong_arguments": (
        "candidate_C",
        "split_node",
        "Argument extraction failures benefit from an explicit reason stage for entity grounding.",
    ),
    "rejection_missed": (
        "candidate_C",
        "add_judge",
        "Unsupported actions were not rejected. Readiness and reasoning stages catch these earlier.",
    ),
    "execution_not_planned": (
        "candidate_B",
        "split_node",
        "Execution not planned — adding perception stage improves intent detection.",
    ),
}

_DEFAULT_UPGRADE = (
    "candidate_C",
    "split_node",
    "General failures suggest the monolithic baseline lacks cognitive decomposition. Upgrade to full pipeline.",
)


class GraphOptimizer:
    def optimize(
        self,
        failure_map: FailureMap,
        candidates: list[GraphCandidate],
    ) -> tuple[GraphCandidate, GraphRevision | None]:
        candidate_map = {c.id: c for c in candidates}
        baseline_id = failure_map.candidate_id

        if failure_map.failure_count == 0:
            return candidate_map[baseline_id], None

        target_id, change_type, rationale = _FAILURE_TO_UPGRADE.get(
            failure_map.dominant_failure_type, _DEFAULT_UPGRADE
        )

        if target_id == baseline_id or target_id not in candidate_map:
            ordered = list(candidate_map.keys())
            current_idx = ordered.index(baseline_id) if baseline_id in ordered else -1
            next_idx = current_idx + 1
            if next_idx < len(ordered):
                target_id = ordered[next_idx]
            else:
                return candidate_map[baseline_id], None

        revision = GraphRevision(
            from_candidate_id=baseline_id,
            to_candidate_id=target_id,
            change_type=change_type,
            rationale=rationale,
        )
        return candidate_map[target_id], revision
