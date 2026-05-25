from __future__ import annotations

from typing import Any

from ..schemas.simulation import RawSimulation, RawTask, SimulationMessage
from ..schemas.trace_converter import CognitiveLabel, TurnSupervisionRow


def _extract_entity_hints(
    message_text: str,
    expected_args: dict[str, Any],
) -> dict[str, Any]:
    """
    Deterministically match expected argument values against message text.
    If the string representation of a value appears verbatim in the message,
    include it as an entity hint.
    """
    hints: dict[str, Any] = {}
    if not message_text:
        return hints
    lower = message_text.lower()
    for key, value in expected_args.items():
        if value is None:
            continue
        str_val = str(value).lower()
        if str_val and str_val in lower:
            hints[key] = value
    return hints


def _next_expected_args_for_turn(
    turn_idx: int,
    messages: list[SimulationMessage],
    task: RawTask,
) -> dict[str, Any]:
    """
    Find the expected action most likely triggered after this turn by
    matching the upcoming assistant tool call name against the expected list.
    Returns the expected action's arguments if a match is found.
    """
    expected = task.expected_actions()
    expected_by_name: dict[str, dict[str, Any]] = {e.name: e.arguments for e in expected}

    for msg in messages:
        if msg.turn_idx <= turn_idx:
            continue
        if msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.name in expected_by_name:
                    return expected_by_name[tc.name]
            break

    return {}


def _label_turn(
    msg: SimulationMessage,
    messages: list[SimulationMessage],
    task: RawTask,
    is_last_turn: bool,
) -> CognitiveLabel:
    if msg.is_tool_result:
        return CognitiveLabel(
            perception_tool_result=msg.content,
        )

    if msg.is_tool_call and msg.tool_calls:
        tc = msg.tool_calls[0]
        return CognitiveLabel(
            plan_next_action="call_tool",
            plan_tool_name=tc.name,
            plan_arguments=tc.arguments,
        )

    if msg.is_user:
        next_args = _next_expected_args_for_turn(msg.turn_idx, messages, task)
        hints = _extract_entity_hints(msg.content or "", next_args)
        return CognitiveLabel(
            perception_message=msg.content,
            perception_entity_hints=hints,
        )

    if msg.is_assistant_text:
        next_action = "respond" if is_last_turn else "ask_followup"
        return CognitiveLabel(
            plan_next_action=next_action,
        )

    return CognitiveLabel()


def supervise_turns(sim: RawSimulation, task: RawTask) -> list[TurnSupervisionRow]:
    messages = sim.messages
    rows: list[TurnSupervisionRow] = []
    last_idx = max((m.turn_idx for m in messages), default=0)

    for msg in messages:
        is_last = msg.turn_idx == last_idx
        label = _label_turn(msg, messages, task, is_last)
        rows.append(TurnSupervisionRow(
            turn_id=f"{sim.id}_turn_{msg.turn_idx}",
            simulation_id=sim.id,
            task_id=sim.task_id,
            turn_idx=msg.turn_idx,
            role=msg.role,
            content=msg.content,
            cognitive_label=label,
        ))

    return rows
