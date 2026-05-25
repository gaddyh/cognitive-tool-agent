from __future__ import annotations

from typing import Any

from ..schemas.simulation import RawSimulation
from ..schemas.trace_converter import ActionSequenceRow, FailureRow


def _argument_delta(expected: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    """
    Compute the diff between expected and actual arguments.
    Returns a dict with keys: missing, extra, mismatched.
    """
    missing = {k: v for k, v in expected.items() if k not in actual}
    extra = {k: v for k, v in actual.items() if k not in expected}
    mismatched = {
        k: {"expected": expected[k], "actual": actual[k]}
        for k in expected
        if k in actual and str(expected[k]) != str(actual[k])
    }
    delta: dict[str, Any] = {}
    if missing:
        delta["missing"] = missing
    if extra:
        delta["extra"] = extra
    if mismatched:
        delta["mismatched"] = mismatched
    return delta


def extract_failures(
    sim: RawSimulation,
    sequence: ActionSequenceRow,
) -> list[FailureRow]:
    """
    Build FailureRows for every failed action, merging:
    - action_checks (which expected action failed + reward)
    - ActionSequenceRow alignment (what the agent actually called and with what args)
    """
    tool_type_map: dict[str, str | None] = {
        ac.action.action_id: ac.tool_type
        for ac in sim.reward_info.action_checks
    }

    aligned_map: dict[str, "AlignedAction"] = {  # noqa: F821
        aa.expected_action_id: aa
        for aa in sequence.aligned_actions
    }

    rows: list[FailureRow] = []
    for ac in sim.reward_info.action_checks:
        if ac.action_match:
            continue

        aligned = aligned_map.get(ac.action.action_id)
        actual_tool = aligned.actual_tool if aligned else None
        actual_args = aligned.actual_arguments if aligned else {}
        actual_turn = aligned.actual_turn_idx if aligned else None

        delta = _argument_delta(ac.action.arguments, actual_args)

        rows.append(FailureRow(
            simulation_id=sim.id,
            task_id=sim.task_id,
            expected_action_id=ac.action.action_id,
            expected_tool=ac.action.name,
            expected_arguments=ac.action.arguments,
            actual_tool=actual_tool,
            actual_arguments=actual_args,
            actual_turn_idx=actual_turn,
            argument_delta=delta,
            action_reward=ac.action_reward,
            tool_type=tool_type_map.get(ac.action.action_id),
        ))

    return rows
