"""Smoke tests: full Lab loop on the micro JSONL dataset."""
from pathlib import Path

import pytest

from cognitive_tool_agent.datasets.loader import load_jsonl
from cognitive_tool_agent.graph_builder.lab import Lab
from cognitive_tool_agent.schemas.graph_builder import LabReport

DATASET_PATH = Path(__file__).parent.parent / "data" / "dev" / "tool_calling_micro.jsonl"


@pytest.fixture(scope="module")
def rows():
    return load_jsonl(DATASET_PATH)


@pytest.fixture(scope="module")
def report(rows):
    return Lab().run(rows)


def test_dataset_loads(rows):
    assert len(rows) == 5


def test_report_is_lab_report(report):
    assert isinstance(report, LabReport)


def test_dataset_profile_populated(report):
    profile = report.dataset_profile
    assert profile.task_type == "tool_calling"
    assert profile.row_count == 5
    assert profile.tool_count == 3
    assert 0.0 <= profile.ambiguity_rate <= 1.0


def test_behavior_decomposition_populated(report):
    decomp = report.behavior_decomposition
    assert decomp.task_type == "tool_calling"
    assert len(decomp.stages) > 0
    assert "perceive" in decomp.stages


def test_three_candidates_generated(report):
    assert len(report.candidates) == 3
    candidate_ids = [c.id for c in report.candidates]
    assert "candidate_A" in candidate_ids
    assert "candidate_B" in candidate_ids
    assert "candidate_C" in candidate_ids


def test_candidate_graphs_have_nodes(report):
    for candidate in report.candidates:
        assert len(candidate.graph_spec.nodes) > 0


def test_evaluation_plan_has_metrics(report):
    assert len(report.evaluation_plan.metrics) > 0
    metric_names = [m.name for m in report.evaluation_plan.metrics]
    assert "end_to_end_success" in metric_names


def test_baseline_scores_keyed_by_candidate(report):
    assert isinstance(report.baseline_scores, dict)
    assert len(report.baseline_scores) == 1
    for candidate_id, scores in report.baseline_scores.items():
        assert isinstance(scores, dict)
        assert "end_to_end_success" in scores
        assert 0.0 <= scores["end_to_end_success"] <= 1.0


def test_optimized_scores_keyed_by_candidate(report):
    assert isinstance(report.optimized_scores, dict)
    assert len(report.optimized_scores) == 1
    for candidate_id, scores in report.optimized_scores.items():
        assert isinstance(scores, dict)
        assert "end_to_end_success" in scores


def test_failure_map_populated(report):
    fm = report.failure_map
    assert fm.candidate_id == "candidate_A"
    assert fm.total_rows == 5
    assert fm.failure_count >= 0
    assert fm.failure_count <= fm.total_rows
    assert fm.dominant_failure_stage != ""
    assert fm.dominant_failure_type != ""


def test_tradeoff_summary_has_three_entries(report):
    assert len(report.tradeoff_summary) == 3
    ids = [e.candidate_id for e in report.tradeoff_summary]
    assert "candidate_A" in ids


def test_tradeoff_summary_scores_in_range(report):
    for entry in report.tradeoff_summary:
        assert 0.0 <= entry.end_to_end_success <= 1.0
        assert entry.latency_estimate >= 1.0
        assert entry.cost_estimate >= 1.0


def test_executor_node_driven_monolithic(rows):
    from cognitive_tool_agent.graph.cognitive_graph import GraphExecutor
    from cognitive_tool_agent.graph_builder.graph_candidate_generator import make_monolithic
    from cognitive_tool_agent.schemas.experiment import ExperimentSpec
    from cognitive_tool_agent.tools.fake_tools import DEFAULT_REGISTRY

    candidate = make_monolithic()
    executor = GraphExecutor()
    trace = executor.run(ExperimentSpec(graph=candidate.graph_spec), rows[0], DEFAULT_REGISTRY)

    assert trace.input is not None
    assert trace.action is not None
    assert trace.perception is None
    assert trace.reasoning is None


def test_executor_node_driven_full_pipeline(rows):
    from cognitive_tool_agent.graph.cognitive_graph import GraphExecutor
    from cognitive_tool_agent.graph_builder.graph_candidate_generator import make_full_pipeline
    from cognitive_tool_agent.schemas.experiment import ExperimentSpec
    from cognitive_tool_agent.tools.fake_tools import DEFAULT_REGISTRY

    candidate = make_full_pipeline()
    executor = GraphExecutor()
    trace = executor.run(ExperimentSpec(graph=candidate.graph_spec), rows[0], DEFAULT_REGISTRY)

    assert trace.input is not None
    assert trace.perception is not None
    assert trace.reasoning is not None
    assert trace.readiness is not None
    assert trace.plan is not None
    assert trace.action is not None
    assert trace.learning is not None


def test_happy_path_row_executes_tool(rows):
    from cognitive_tool_agent.graph.cognitive_graph import GraphExecutor
    from cognitive_tool_agent.graph_builder.graph_candidate_generator import make_full_pipeline
    from cognitive_tool_agent.schemas.experiment import ExperimentSpec
    from cognitive_tool_agent.tools.fake_tools import DEFAULT_REGISTRY

    happy_row = next(r for r in rows if r.id == "tc-001")
    candidate = make_full_pipeline()
    executor = GraphExecutor()
    trace = executor.run(ExperimentSpec(graph=candidate.graph_spec), happy_row, DEFAULT_REGISTRY)

    assert trace.action is not None
    assert trace.action.action_type == "tool_executed"
    assert trace.action.tool_name == "get_order_status"


def test_unsupported_action_row_rejects(rows):
    from cognitive_tool_agent.graph.cognitive_graph import GraphExecutor
    from cognitive_tool_agent.graph_builder.graph_candidate_generator import make_full_pipeline
    from cognitive_tool_agent.schemas.experiment import ExperimentSpec
    from cognitive_tool_agent.tools.fake_tools import DEFAULT_REGISTRY

    reject_row = next(r for r in rows if r.id == "tc-005")
    candidate = make_full_pipeline()
    executor = GraphExecutor()
    trace = executor.run(ExperimentSpec(graph=candidate.graph_spec), reject_row, DEFAULT_REGISTRY)

    assert trace.action is not None
    assert trace.action.action_type == "rejected"
