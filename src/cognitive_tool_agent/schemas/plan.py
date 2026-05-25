from typing import Any, Literal
from pydantic import BaseModel
from .common import Confidence


class ToolCallPlan(BaseModel):
    tool_name: str
    arguments: dict[str, Any]


class PlanResult(BaseModel):
    next_action: Literal[
        "execute_tool",
        "ask_followup",
        "answer_directly",
        "abstain",
        "reject",
    ]

    tool_call: ToolCallPlan | None = None
    followup_question: str | None = None
    direct_answer: str | None = None

    blocking_reasons: list[str] = []
    confidence: Confidence
