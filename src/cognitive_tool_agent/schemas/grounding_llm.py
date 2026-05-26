from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from .grounding import GroundingResult


class GroundedField(BaseModel):
    value: Any | None = None
    status: Literal["grounded", "missing", "ambiguous", "not_applicable"]
    evidence: list[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class LLMGroundingOutput(BaseModel):
    fields: dict[str, GroundedField]
    unresolved_ids: list[str] = Field(default_factory=list)
    clarification_needed: bool = False
    clarification_question: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)

    def to_grounding_result(self) -> GroundingResult:
        resolved_args: dict[str, Any] = {
            name: f.value
            for name, f in self.fields.items()
            if f.status == "grounded" and f.value is not None
        }

        unresolved: list[str] = list(self.unresolved_ids)
        for name, f in self.fields.items():
            if f.status in {"missing", "ambiguous"}:
                unresolved.append(name)

        return GroundingResult(
            grounding_mode="llm",
            resolved_args=resolved_args,
            unresolved_ids=sorted(set(unresolved)),
            grounding_confidence=self.confidence,
            candidates_examined=len(self.fields),
        )
