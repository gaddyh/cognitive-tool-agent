from __future__ import annotations

from typing import Any
from pydantic import BaseModel


class ToolRegistryEntry(BaseModel):
    name: str
    required_args: list[str] = []
    seen_args: list[str] = []
    usage_count: int = 0
    tool_type: str | None = None


class AlignedAction(BaseModel):
    expected_action_id: str
    expected_tool: str
    expected_arguments: dict[str, Any] = {}
    actual_tool: str | None = None
    actual_arguments: dict[str, Any] = {}
    actual_turn_idx: int | None = None
    action_match: bool = False
    action_reward: float = 0.0


class ActionSequenceRow(BaseModel):
    simulation_id: str
    task_id: str
    aligned_actions: list[AlignedAction] = []


class CognitiveLabel(BaseModel):
    perception_message: str | None = None
    perception_tool_result: str | None = None
    perception_entity_hints: dict[str, Any] = {}
    plan_next_action: str | None = None
    plan_tool_name: str | None = None
    plan_arguments: dict[str, Any] = {}


class TurnSupervisionRow(BaseModel):
    turn_id: str
    simulation_id: str
    task_id: str
    turn_idx: int
    role: str
    content: str | None = None
    cognitive_label: CognitiveLabel
    split: str | None = None
    scenario_type: str | None = None
    primary_scenario: str | None = None
    is_multi_action: bool | None = None
    requires_grounding: bool | None = None
    difficulty_bucket: str | None = None


class FailureRow(BaseModel):
    simulation_id: str
    task_id: str
    expected_action_id: str
    expected_tool: str
    expected_arguments: dict[str, Any] = {}
    actual_tool: str | None = None
    actual_arguments: dict[str, Any] = {}
    actual_turn_idx: int | None = None
    argument_delta: dict[str, Any] = {}
    action_reward: float = 0.0
    tool_type: str | None = None


class ConversionSummary(BaseModel):
    tasks_count: int
    simulations_count: int
    messages_count: int
    expected_actions_count: int
    actual_tool_calls_count: int
    matched_actions_count: int
    failed_actions_count: int
