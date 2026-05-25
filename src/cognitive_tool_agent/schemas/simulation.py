from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    id: str = ""
    name: str
    arguments: dict[str, Any] = {}
    requestor: str = "assistant"


class SimulationMessage(BaseModel):
    role: str
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    turn_idx: int = 0
    id: str | None = None

    @property
    def is_tool_result(self) -> bool:
        return self.role == "tool"

    @property
    def is_tool_call(self) -> bool:
        return self.role == "assistant" and bool(self.tool_calls)

    @property
    def is_user(self) -> bool:
        return self.role == "user"

    @property
    def is_assistant_text(self) -> bool:
        return self.role == "assistant" and not self.tool_calls and self.content is not None


class ExpectedAction(BaseModel):
    action_id: str
    requestor: str = "assistant"
    name: str
    arguments: dict[str, Any] = {}
    tool_type: str | None = None


class ActionCheck(BaseModel):
    action: ExpectedAction
    action_match: bool
    action_reward: float
    tool_type: str | None = None


class RewardInfo(BaseModel):
    reward: float = 0.0
    action_checks: list[ActionCheck] = []


class UserInstructions(BaseModel):
    domain: str | None = None
    reason_for_call: str | None = None
    known_info: str | None = None
    unknown_info: str | None = None
    task_instructions: str | None = None


class UserScenario(BaseModel):
    instructions: UserInstructions | None = None


class RawTask(BaseModel):
    id: str
    evaluation_criteria: dict[str, Any] = {}
    user_scenario: UserScenario | None = None

    def expected_actions(self) -> list[ExpectedAction]:
        raw = self.evaluation_criteria.get("actions", [])
        return [ExpectedAction.model_validate(a) for a in raw]


class RawSimulation(BaseModel):
    id: str
    task_id: str
    messages: list[SimulationMessage] = []
    reward_info: RewardInfo = Field(default_factory=RewardInfo)


class SimulationFile(BaseModel):
    tasks: list[RawTask] = []
    simulations: list[RawSimulation] = []
