from __future__ import annotations

from collections import Counter

from ..schemas.dataset import DatasetRow
from ..schemas.graph_builder import FailureMap, RowFailure
from ..schemas.trace import CognitiveTrace


class FailureAnalyzer:
    def analyze(
        self,
        candidate_id: str,
        traces: list[CognitiveTrace],
        rows: list[DatasetRow],
    ) -> FailureMap:
        failures: list[RowFailure] = []

        for trace, row in zip(traces, rows):
            failure = _analyze_row(trace, row)
            if failure is not None:
                failures.append(failure)

        stage_counter: Counter[str] = Counter(f.failure_stage for f in failures)
        type_counter: Counter[str] = Counter(f.failure_type for f in failures)

        dominant_stage = stage_counter.most_common(1)[0][0] if stage_counter else "none"
        dominant_type = type_counter.most_common(1)[0][0] if type_counter else "none"

        return FailureMap(
            candidate_id=candidate_id,
            failures=failures,
            dominant_failure_stage=dominant_stage,
            dominant_failure_type=dominant_type,
            total_rows=len(rows),
            failure_count=len(failures),
        )


def _analyze_row(
    trace: CognitiveTrace, row: DatasetRow
) -> RowFailure | None:
    expected_action = row.expected.expected_action
    actual_action: str | None = None
    failure_stage = "none"
    failure_type = "none"
    explanation = ""

    if trace.action is None:
        return RowFailure(
            row_id=row.id,
            expected_action=expected_action,
            actual_action=None,
            failure_stage="acting",
            failure_type="no_action_produced",
            explanation="Executor produced no ActionResult",
        )

    action_type_map = {
        "tool_executed": "execute_tool",
        "followup_asked": "ask_followup",
        "answered_directly": "answer_directly",
        "abstained": "abstain",
        "rejected": "reject",
    }
    actual_action = action_type_map.get(trace.action.action_type, trace.action.action_type)

    if actual_action != expected_action:
        failure_stage, failure_type, explanation = _classify_mismatch(
            trace, expected_action, actual_action
        )
        return RowFailure(
            row_id=row.id,
            expected_action=expected_action,
            actual_action=actual_action,
            failure_stage=failure_stage,
            failure_type=failure_type,
            explanation=explanation,
        )

    if expected_action == "execute_tool" and row.expected.expected_tool is not None:
        if trace.action.tool_name != row.expected.expected_tool:
            return RowFailure(
                row_id=row.id,
                expected_action=expected_action,
                actual_action=actual_action,
                failure_stage="planning",
                failure_type="wrong_tool_selected",
                explanation=(
                    f"Expected tool '{row.expected.expected_tool}', "
                    f"got '{trace.action.tool_name}'"
                ),
            )

    if (
        expected_action == "execute_tool"
        and row.expected.expected_arguments is not None
        and trace.action.tool_arguments != row.expected.expected_arguments
    ):
        return RowFailure(
            row_id=row.id,
            expected_action=expected_action,
            actual_action=actual_action,
            failure_stage="acting",
            failure_type="wrong_arguments",
            explanation=(
                f"Expected args {row.expected.expected_arguments}, "
                f"got {trace.action.tool_arguments}"
            ),
        )

    return None


def _classify_mismatch(
    trace: CognitiveTrace, expected: str, actual: str
) -> tuple[str, str, str]:
    if expected == "execute_tool" and actual in ("ask_followup",):
        if trace.readiness and trace.readiness.policy_violations:
            return "readiness", "policy_blocked_execution", (
                "readiness stage blocked execution due to policy: "
                + "; ".join(trace.readiness.policy_violations)
            )
        if trace.readiness and trace.readiness.missing_required_fields:
            return "readiness", "missing_fields_blocked_execution", (
                "missing required fields: " + str(trace.readiness.missing_required_fields)
            )
        return "planning", "execution_not_planned", (
            f"expected execute_tool but planner chose {actual}"
        )

    if expected == "ask_followup" and actual == "execute_tool":
        return "readiness", "premature_execution", (
            "agent executed without required confirmation or clarification"
        )

    if expected == "reject" and actual != "reject":
        return "planning", "rejection_missed", (
            f"expected rejection of unsupported action, got {actual}"
        )

    if actual == "reject" and expected != "reject":
        return "planning", "false_rejection", (
            f"planner rejected a supported action (expected {expected})"
        )

    return "unknown", "action_type_mismatch", (
        f"expected '{expected}', got '{actual}'"
    )
