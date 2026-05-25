from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from ..evals.evaluator import Evaluator
from ..graph.cognitive_graph import GraphExecutor
from ..graph_builder.failure_analyzer import FailureAnalyzer
from ..graph_builder.graph_candidate_generator import (
    make_monolithic_baseline,
    make_perceive_plan_act,
)
from ..graph_runner.dataset_adapter import ActionSequenceAdapter
from ..graph_runner.trace_writer import TraceWriter
from ..schemas.experiment import ExperimentSpec, NodeRuntimeConfig
from ..schemas.graph_runner import GraphEvaluationReport, GraphEvaluationRow, GraphRunResult
from ..schemas.recommender import RecommendedGraph
from ..tools.registry import ToolRegistry


def _load_recommended_graph(path: Path) -> RecommendedGraph:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return RecommendedGraph.model_validate(raw)


class GraphEvaluationRunner:
    def __init__(self) -> None:
        self._evaluator = Evaluator()
        self._failure_analyzer = FailureAnalyzer()
        self._trace_writer = TraceWriter()
        self._adapter = ActionSequenceAdapter()

    def run(
        self,
        recommended_graph_path: Path,
        action_seq_path: Path,
        turn_sup_path: Path,
        tool_registry_path: Path,
        out_dir: Path,
    ) -> GraphEvaluationReport:
        rows, registry = self._adapter.load(
            action_seq_path, turn_sup_path, tool_registry_path
        )
        recommended = _load_recommended_graph(recommended_graph_path)

        monolithic_graph = make_monolithic_baseline().graph_spec
        minimal_graph = make_perceive_plan_act().graph_spec
        recommended_graph = recommended.graph_spec

        experiments: list[tuple[str, ExperimentSpec, str]] = [
            ("monolithic", ExperimentSpec(graph=monolithic_graph), "n/a"),
            ("minimal", ExperimentSpec(graph=minimal_graph), "n/a"),
            (
                "recommended_stub",
                ExperimentSpec(
                    graph=recommended_graph,
                    runtime=[NodeRuntimeConfig(node_id="grounding", mode="stub")],
                ),
                "stub",
            ),
            (
                "recommended_oracle",
                ExperimentSpec(
                    graph=recommended_graph,
                    runtime=[NodeRuntimeConfig(node_id="grounding", mode="oracle")],
                ),
                "oracle",
            ),
        ]

        executor = GraphExecutor()
        results: list[GraphEvaluationRow] = []
        run_results: list[GraphRunResult] = []

        for graph_id, experiment, grounding_mode in experiments:
            traces = [executor.run(experiment, row, registry) for row in rows]

            traces_path = out_dir / f"graph_traces_{graph_id}.jsonl"
            self._trace_writer.write(traces, traces_path)

            scores = self._evaluator.score(traces, rows)
            failure_map = self._failure_analyzer.analyze(graph_id, traces, rows)

            failure_summary = dict(
                Counter(f.failure_type for f in failure_map.failures)
            )
            failure_summary["_total"] = failure_map.failure_count

            run_results.append(
                GraphRunResult(
                    graph_id=graph_id,
                    rows_count=len(rows),
                    traces_path=str(traces_path),
                    scores=scores,
                    failure_summary=failure_summary,
                )
            )

            results.append(
                GraphEvaluationRow(
                    graph_id=graph_id,
                    node_count=len(experiment.graph.nodes),
                    end_to_end_success=scores["end_to_end_success"],
                    tool_name_accuracy=scores["tool_name_accuracy"],
                    argument_exact_match=scores["argument_exact_match"],
                    policy_violation_rate=scores["policy_violation_rate"],
                    stage_failure_rate=scores["stage_failure_rate"],
                    failure_count=failure_map.failure_count,
                    grounding_mode=grounding_mode,
                )
            )

        return GraphEvaluationReport(
            source_dataset=str(action_seq_path),
            rows_evaluated=len(rows),
            results=results,
        )
