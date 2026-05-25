from typing import Any, Literal
from pydantic import BaseModel


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
