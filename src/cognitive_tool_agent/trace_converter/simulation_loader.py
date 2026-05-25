from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..schemas.simulation import SimulationFile, SimulationMessage, ToolCall


def _parse_tool_calls(raw: list[dict[str, Any]] | None) -> list[ToolCall] | None:
    if not raw:
        return None
    calls = []
    for tc in raw:
        arguments = tc.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (json.JSONDecodeError, TypeError):
                arguments = {}
        calls.append(ToolCall(
            id=tc.get("id", ""),
            name=tc.get("name", ""),
            arguments=arguments,
            requestor=tc.get("requestor", "assistant"),
        ))
    return calls or None


def _parse_message(raw: dict[str, Any]) -> SimulationMessage:
    tool_calls = _parse_tool_calls(raw.get("tool_calls"))
    return SimulationMessage(
        role=raw.get("role", ""),
        content=raw.get("content"),
        tool_calls=tool_calls,
        turn_idx=raw.get("turn_idx", 0),
        id=raw.get("id"),
    )


def load_simulation_file(path: Path | str) -> SimulationFile:
    """Load a tau-bench-style simulation JSON file into a SimulationFile."""
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)

    raw_tasks = data.get("tasks", [])
    raw_simulations = data.get("simulations", [])

    from ..schemas.simulation import RawTask, RawSimulation, RewardInfo, ActionCheck, ExpectedAction

    tasks = [RawTask.model_validate(t) for t in raw_tasks]

    simulations: list[RawSimulation] = []
    for sim in raw_simulations:
        messages = [_parse_message(m) for m in (sim.get("messages") or [])]

        reward_raw = sim.get("reward_info") or {}
        action_checks = []
        for ac in (reward_raw.get("action_checks") or []):
            action_raw = ac.get("action", {})
            arguments = action_raw.get("arguments", {}) or {}
            exp_action = ExpectedAction(
                action_id=action_raw.get("action_id", ""),
                requestor=action_raw.get("requestor", "assistant"),
                name=action_raw.get("name", ""),
                arguments=arguments,
                tool_type=ac.get("tool_type"),
            )
            action_checks.append(ActionCheck(
                action=exp_action,
                action_match=ac.get("action_match", False),
                action_reward=ac.get("action_reward", 0.0),
                tool_type=ac.get("tool_type"),
            ))

        reward_info = RewardInfo(
            reward=reward_raw.get("reward", 0.0),
            action_checks=action_checks,
        )

        simulations.append(RawSimulation(
            id=sim.get("id", ""),
            task_id=str(sim.get("task_id", "")),
            messages=messages,
            reward_info=reward_info,
        ))

    return SimulationFile(tasks=tasks, simulations=simulations)
