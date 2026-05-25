from typing import Any, Literal
from pydantic import BaseModel, Field


class Confidence(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    reason: str = ""


class Evidence(BaseModel):
    source: Literal["user_message", "context", "tool_schema", "world_state", "tool_result"]
    text: str
    reason: str = ""


class ToolSchema(BaseModel):
    name: str
    description: str
    required_fields: list[str] = []
    optional_fields: list[str] = []
    properties: dict[str, Any] = {}


class UserInput(BaseModel):
    message: str
    conversation_context: list[str] = []
    available_tools: list[ToolSchema] = []
    world_state: dict[str, Any] = {}
