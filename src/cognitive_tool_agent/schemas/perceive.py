from typing import Any, Literal
from pydantic import BaseModel
from .common import Confidence, Evidence


class MentionedEntity(BaseModel):
    text: str
    entity_type: str | None = None


class RawFieldCandidate(BaseModel):
    name: str
    value: Any
    evidence_text: str


class PerceptionResult(BaseModel):
    intent_candidates: list[str]
    mentioned_entities: list[MentionedEntity] = []
    raw_field_candidates: list[RawFieldCandidate] = []

    ambiguity_detected: bool
    ambiguity_type: Literal[
        "none",
        "missing_referent",
        "multiple_intents",
        "vague_time",
        "vague_entity",
        "underspecified_action",
    ] = "none"

    candidate_tools: list[str] = []
    confidence: Confidence
    evidence: list[Evidence] = []
