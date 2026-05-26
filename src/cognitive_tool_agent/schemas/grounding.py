from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class GroundingResult(BaseModel):
    grounding_mode: Literal["stub", "oracle", "disabled", "deterministic", "llm"] = "stub"
    resolved_args: dict[str, Any] = {}
    unresolved_ids: list[str] = []
    grounding_confidence: float = 0.0
    candidates_examined: int = 0
