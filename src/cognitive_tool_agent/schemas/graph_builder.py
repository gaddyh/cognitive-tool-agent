from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel

from .graph_spec import GraphSpec


TaskType = Literal[
    "tool_calling",
    "rag",
    "classification",
    "planning",
    "generation",
    "unknown",
]

ChangeType = Literal[
    "split_node",
    "add_judge",
    "add_memory",
    "add_retrieval",
    "parallelize",
    "change_model",
    "change_schema",
    "change_metric_weight",
    "change_fallback_path",
]


class DatasetProfile(BaseModel):
    task_type: TaskType
    input_space: str
    output_space: str
    label_set: list[str]
    ambiguity_rate: float
    contradiction_count: int
    row_count: int
    tool_count: int
    notes: list[str] = []


class BehaviorDecomposition(BaseModel):
    task_type: str
    stages: list[str]
    rationale: str


class GraphCandidate(BaseModel):
    id: str
    graph_spec: GraphSpec
    rationale: str
    latency_estimate: float = 1.0
    cost_estimate: float = 1.0


class StageMetric(BaseModel):
    name: str
    stage: str | None
    description: str


class EvaluationPlan(BaseModel):
    metrics: list[StageMetric]


class RowFailure(BaseModel):
    row_id: str
    expected_action: str
    actual_action: str | None
    failure_stage: str
    failure_type: str
    explanation: str = ""


class FailureMap(BaseModel):
    candidate_id: str
    failures: list[RowFailure]
    dominant_failure_stage: str
    dominant_failure_type: str
    total_rows: int
    failure_count: int


class GraphRevision(BaseModel):
    from_candidate_id: str
    to_candidate_id: str
    change_type: ChangeType
    rationale: str


class TradeoffEntry(BaseModel):
    candidate_id: str
    end_to_end_success: float
    latency_estimate: float
    cost_estimate: float
    failure_count: int
    recommendation: str


class LabReport(BaseModel):
    dataset_profile: DatasetProfile
    behavior_decomposition: BehaviorDecomposition
    candidates: list[GraphCandidate]
    evaluation_plan: EvaluationPlan
    baseline_scores: dict[str, dict[str, float]]
    optimized_scores: dict[str, dict[str, float]]
    failure_map: FailureMap
    revision: GraphRevision | None = None
    tradeoff_summary: list[TradeoffEntry]
