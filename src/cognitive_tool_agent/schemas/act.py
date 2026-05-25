from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel

NEXT_ACTION_TO_ACTION_TYPE: dict[str, str] = {
    "execute_tool": "tool_executed",
    "ask_followup": "followup_asked",
    "answer_directly": "answered_directly",
    "abstain": "abstained",
    "reject": "rejected",
}

ACTION_TYPE_TO_NEXT_ACTION: dict[str, str] = {
    v: k for k, v in NEXT_ACTION_TO_ACTION_TYPE.items()
}


class ActionResult(BaseModel):
    action_type: Literal[
        "tool_executed",
        "followup_asked",
        "answered_directly",
        "abstained",
        "rejected",
    ]

    success: bool
    tool_name: str | None = None
    tool_arguments: dict[str, Any] | None = None
    tool_result: Any | None = None
    user_facing_message: str | None = None
    error: str | None = None
