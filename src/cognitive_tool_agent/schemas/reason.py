from typing import Any, Literal
from pydantic import BaseModel
from .common import Confidence, Evidence


class ResolvedEntity(BaseModel):
    surface_text: str
    entity_type: str
    resolved_id: str | None = None
    resolved_value: Any | None = None
    status: Literal["resolved", "ambiguous", "missing", "not_needed"]
    candidates: list[Any] = []


class MissingRequirement(BaseModel):
    field_name: str
    reason: str
    can_infer: bool = False


class ReasoningResult(BaseModel):
    selected_intent: str | None
    selected_tool: str | None

    resolved_entities: list[ResolvedEntity] = []
    missing_requirements: list[MissingRequirement] = []

    reasoning_status: Literal[
        "ready",
        "needs_clarification",
        "unsupported",
        "unsafe",
        "low_confidence",
    ]

    confidence: Confidence
    evidence: list[Evidence] = []
