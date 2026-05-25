from __future__ import annotations

from ..schemas.simulation import RawSimulation, RawTask, SimulationMessage
from ..schemas.trace_converter import ActionSequenceRow, AlignedAction


def _actual_tool_calls(messages: list[SimulationMessage]) -> list[tuple[int, str, dict]]:
    """Return (turn_idx, tool_name, arguments) for every actual tool call in order."""
    calls = []
    for msg in messages:
        if msg.tool_calls:
            for tc in msg.tool_calls:
                calls.append((msg.turn_idx, tc.name, tc.arguments))
    return calls


def align_actions(sim: RawSimulation, task: RawTask) -> ActionSequenceRow:
    """
    Pair each expected action with the closest matching actual tool call.

    Strategy:
    1. Walk expected actions in order.
    2. For each expected action, look for the first unconsumed actual call
       with the same tool name.
    3. If found, mark as match and consume that call; otherwise leave actual as None.
    This is deterministic and handles extra/missing actual calls gracefully.
    """
    expected = task.expected_actions()
    actual_calls = _actual_tool_calls(sim.messages)

    action_check_map: dict[str, tuple[bool, float]] = {
        ac.action.action_id: (ac.action_match, ac.action_reward)
        for ac in sim.reward_info.action_checks
    }

    consumed: set[int] = set()
    aligned: list[AlignedAction] = []

    for exp in expected:
        match_idx: int | None = None
        for i, (turn_idx, tool_name, args) in enumerate(actual_calls):
            if i in consumed:
                continue
            if tool_name == exp.name:
                match_idx = i
                break

        match_flag, reward = action_check_map.get(exp.action_id, (False, 0.0))

        if match_idx is not None:
            consumed.add(match_idx)
            turn_idx, actual_tool, actual_args = actual_calls[match_idx]
            aligned.append(AlignedAction(
                expected_action_id=exp.action_id,
                expected_tool=exp.name,
                expected_arguments=exp.arguments,
                actual_tool=actual_tool,
                actual_arguments=actual_args,
                actual_turn_idx=turn_idx,
                action_match=match_flag,
                action_reward=reward,
            ))
        else:
            aligned.append(AlignedAction(
                expected_action_id=exp.action_id,
                expected_tool=exp.name,
                expected_arguments=exp.arguments,
                actual_tool=None,
                actual_arguments={},
                actual_turn_idx=None,
                action_match=False,
                action_reward=0.0,
            ))

    return ActionSequenceRow(
        simulation_id=sim.id,
        task_id=sim.task_id,
        aligned_actions=aligned,
    )
