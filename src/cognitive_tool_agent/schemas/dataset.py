from typing import Any
from pydantic import BaseModel


class ExpectedBehavior(BaseModel):
    expected_action: str
    expected_tool: str | None = None
    expected_arguments: dict[str, Any] | None = None
    expected_followup_contains: list[str] = []
    expected_failure_stage: str | None = None


class DatasetRow(BaseModel):
    id: str
    user_message: str
    context: list[str] = []
    tools: list[str] = []
    world_state: dict[str, Any] = {}

    expected: ExpectedBehavior

    tags: list[str] = []
