from __future__ import annotations

from ..schemas.act import NEXT_ACTION_TO_ACTION_TYPE
from ..schemas.dataset import DatasetRow
from ..schemas.trace import CognitiveTrace


def end_to_end_success(
    traces: list[CognitiveTrace], rows: list[DatasetRow]
) -> float:
    if not rows:
        return 0.0
    correct = 0
    for trace, row in zip(traces, rows):
        expected = row.expected.expected_action
        if trace.action is None:
            continue
        actual_type = trace.action.action_type
        if _action_type_matches(actual_type, expected):
            correct += 1
    return correct / len(rows)


def tool_name_accuracy(
    traces: list[CognitiveTrace], rows: list[DatasetRow]
) -> float:
    tool_rows = [r for r in rows if r.expected.expected_tool is not None]
    if not tool_rows:
        return 1.0
    correct = 0
    for trace, row in zip(traces, rows):
        if row.expected.expected_tool is None:
            continue
        if trace.action and trace.action.tool_name == row.expected.expected_tool:
            correct += 1
    return correct / len(tool_rows)


def argument_exact_match(
    traces: list[CognitiveTrace], rows: list[DatasetRow]
) -> float:
    arg_rows = [r for r in rows if r.expected.expected_arguments is not None]
    if not arg_rows:
        return 1.0
    correct = 0
    for trace, row in zip(traces, rows):
        if row.expected.expected_arguments is None:
            continue
        if trace.action and trace.action.tool_arguments == row.expected.expected_arguments:
            correct += 1
    return correct / len(arg_rows)


def policy_violation_rate(
    traces: list[CognitiveTrace], rows: list[DatasetRow]
) -> float:
    if not rows:
        return 0.0
    violations = sum(
        1
        for trace in traces
        if trace.readiness is not None and bool(trace.readiness.policy_violations)
    )
    return violations / len(rows)


def stage_failure_rate(
    traces: list[CognitiveTrace], rows: list[DatasetRow]
) -> float:
    if not rows:
        return 0.0
    failures = sum(
        1
        for trace in traces
        if trace.learning is not None
        and trace.learning.failure_analysis.failed_stage != "none"
    )
    return failures / len(rows)


def _action_type_matches(actual_type: str, expected_action: str) -> bool:
    return actual_type == NEXT_ACTION_TO_ACTION_TYPE.get(expected_action, expected_action)
