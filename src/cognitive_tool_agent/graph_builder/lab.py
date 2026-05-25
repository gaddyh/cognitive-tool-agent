from __future__ import annotations

from ..schemas.dataset import DatasetRow
from ..schemas.graph_builder import (
    GraphCandidate,
    LabReport,
    TradeoffEntry,
)
from ..tools.registry import ToolRegistry
from ..tools.fake_tools import DEFAULT_REGISTRY
from .baseline_runner import BaselineRunner
from .behavior_decomposer import BehaviorDecomposer
from .dataset_profiler import DatasetProfiler
from .evaluation_designer import EvaluationDesigner
from .failure_analyzer import FailureAnalyzer
from .graph_candidate_generator import GraphCandidateGenerator
from .graph_optimizer import GraphOptimizer


class Lab:
    """
    Orchestrates the full Graph Builder loop:

    profile → decompose → generate → design → run all candidates →
    analyze baseline failures → optimize → produce LabReport
    """

    def __init__(self, registry: ToolRegistry | None = None) -> None:
        self._registry = registry or DEFAULT_REGISTRY

    def run(self, rows: list[DatasetRow]) -> LabReport:
        registry = self._registry

        profile = DatasetProfiler().profile(rows)
        decomposition = BehaviorDecomposer().decompose(profile)
        candidates = GraphCandidateGenerator().generate(decomposition)
        eval_plan = EvaluationDesigner().design(profile)

        runner = BaselineRunner()

        all_traces: dict[str, list] = {}
        all_scores: dict[str, dict[str, float]] = {}
        for candidate in candidates:
            traces, scores = runner.run(candidate, rows, registry)
            all_traces[candidate.id] = traces
            all_scores[candidate.id] = scores

        baseline_candidate = candidates[0]
        baseline_traces = all_traces[baseline_candidate.id]

        failure_map = FailureAnalyzer().analyze(
            candidate_id=baseline_candidate.id,
            traces=baseline_traces,
            rows=rows,
        )

        optimized_candidate, revision = GraphOptimizer().optimize(failure_map, candidates)

        baseline_scores = {baseline_candidate.id: all_scores[baseline_candidate.id]}
        optimized_scores = {optimized_candidate.id: all_scores[optimized_candidate.id]}

        tradeoff_summary = _build_tradeoff(candidates, all_scores, failure_map, optimized_candidate)

        return LabReport(
            dataset_profile=profile,
            behavior_decomposition=decomposition,
            candidates=candidates,
            evaluation_plan=eval_plan,
            baseline_scores=baseline_scores,
            optimized_scores=optimized_scores,
            failure_map=failure_map,
            revision=revision,
            tradeoff_summary=tradeoff_summary,
        )


def _build_tradeoff(
    candidates: list[GraphCandidate],
    all_scores: dict[str, dict[str, float]],
    failure_map,
    optimized_candidate: GraphCandidate,
) -> list[TradeoffEntry]:
    entries: list[TradeoffEntry] = []
    for candidate in candidates:
        scores = all_scores.get(candidate.id, {})
        success = scores.get("end_to_end_success", 0.0)

        if candidate.id == optimized_candidate.id:
            recommendation = "RECOMMENDED — best balance of success rate and failure coverage"
        elif success == max(
            all_scores.get(c.id, {}).get("end_to_end_success", 0.0) for c in candidates
        ):
            recommendation = "highest end_to_end_success but consider latency tradeoff"
        else:
            recommendation = "baseline — use as comparison point"

        entries.append(
            TradeoffEntry(
                candidate_id=candidate.id,
                end_to_end_success=round(success, 3),
                latency_estimate=candidate.latency_estimate,
                cost_estimate=candidate.cost_estimate,
                failure_count=failure_map.failure_count if candidate.id == failure_map.candidate_id else 0,
                recommendation=recommendation,
            )
        )
    return entries
