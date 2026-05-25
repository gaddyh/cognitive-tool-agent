from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class GraphRevisionSuggestion(BaseModel):
    target_capability: str
    failure_pattern: str
    suggestion: str
    rationale: str
    priority: Literal["low", "medium", "high"]
    evidence: list[str]


class GraphRevisionAdvisorReport(BaseModel):
    graph_id: str
    suggestions: list[GraphRevisionSuggestion]
