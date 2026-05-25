from __future__ import annotations

from pydantic import BaseModel


class GraphRunResult(BaseModel):
    graph_id: str
    rows_count: int
    traces_path: str
    scores: dict[str, float]
    failure_summary: dict[str, int]


class GraphEvaluationRow(BaseModel):
    graph_id: str
    node_count: int
    end_to_end_success: float
    tool_name_accuracy: float
    argument_exact_match: float
    policy_violation_rate: float
    stage_failure_rate: float
    failure_count: int
    grounding_mode: str = "n/a"


class GraphEvaluationReport(BaseModel):
    source_dataset: str
    rows_evaluated: int
    results: list[GraphEvaluationRow]
