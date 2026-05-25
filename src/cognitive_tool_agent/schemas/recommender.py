from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from .graph_spec import GraphSpec


class CapabilityRequirement(BaseModel):
    required: bool
    strength: float
    evidence: list[str]


class CapabilityInferenceResult(BaseModel):
    required_capabilities: dict[str, CapabilityRequirement]
    raw_signals: dict[str, float]
    signal_sources: dict[str, str] = {}


class RecommendedGraph(BaseModel):
    graph_spec: GraphSpec
    memory_required: bool
    readiness_required: bool
    parallel_lookup_nodes: bool
    rationale: list[str]
    required_capabilities: dict[str, CapabilityRequirement]
    confidence: float
