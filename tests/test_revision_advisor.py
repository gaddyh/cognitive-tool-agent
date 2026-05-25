"""Tests for GraphRevisionAdvisor rule-based heuristics."""
from __future__ import annotations

import pytest

from cognitive_tool_agent.recommender.revision_advisor import GraphRevisionAdvisor
from cognitive_tool_agent.schemas.graph_runner import GraphEvaluationReport, GraphEvaluationRow
from cognitive_tool_agent.schemas.recommender import (
    CapabilityInferenceResult,
    CapabilityRequirement,
)


def _make_capability_inference(
    grounding_strength: float = 0.5,
    peak_grounding_strength: float = 0.85,
    peak_grounding_instances: float = 88.0,
    chaining_strength: float = 0.54,
    write_fraction: float = 0.19,
    write_failure_fraction: float = 0.43,
    avg_chain_depth: float = 3.04,
    peak_grounding_arg: str = "item_ids",
) -> CapabilityInferenceResult:
    return CapabilityInferenceResult(
        required_capabilities={
            "memory": CapabilityRequirement(required=True, strength=chaining_strength, evidence=[]),
            "grounding": CapabilityRequirement(required=True, strength=grounding_strength, evidence=[]),
            "readiness": CapabilityRequirement(required=True, strength=write_fraction, evidence=[]),
            "deep_planning": CapabilityRequirement(required=True, strength=avg_chain_depth / 5.0, evidence=[]),
        },
        raw_signals={
            "chaining_strength": chaining_strength,
            "grounding_strength": grounding_strength,
            "peak_grounding_strength": peak_grounding_strength,
            "peak_grounding_instances": peak_grounding_instances,
            "write_fraction": write_fraction,
            "avg_chain_depth": avg_chain_depth,
            "write_failure_fraction": write_failure_fraction,
        },
        signal_sources={"peak_grounding_arg": peak_grounding_arg},
    )


def _make_eval_report(
    stub_e2e: float = 0.55,
    oracle_e2e: float = 0.80,
    stub_arg_match: float = 0.40,
    stub_policy_viol: float = 0.10,
    minimal_e2e: float = 0.60,
    monolithic_e2e: float = 0.50,
) -> GraphEvaluationReport:
    return GraphEvaluationReport(
        source_dataset="data/out/action_sequence.jsonl",
        rows_evaluated=100,
        results=[
            GraphEvaluationRow(
                graph_id="monolithic",
                node_count=1,
                end_to_end_success=monolithic_e2e,
                tool_name_accuracy=0.70,
                argument_exact_match=0.45,
                policy_violation_rate=0.20,
                stage_failure_rate=0.30,
                failure_count=50,
                grounding_mode="n/a",
            ),
            GraphEvaluationRow(
                graph_id="minimal",
                node_count=3,
                end_to_end_success=minimal_e2e,
                tool_name_accuracy=0.75,
                argument_exact_match=0.50,
                policy_violation_rate=0.15,
                stage_failure_rate=0.25,
                failure_count=40,
                grounding_mode="n/a",
            ),
            GraphEvaluationRow(
                graph_id="recommended_stub",
                node_count=7,
                end_to_end_success=stub_e2e,
                tool_name_accuracy=0.78,
                argument_exact_match=stub_arg_match,
                policy_violation_rate=stub_policy_viol,
                stage_failure_rate=0.20,
                failure_count=45,
                grounding_mode="stub",
            ),
            GraphEvaluationRow(
                graph_id="recommended_oracle",
                node_count=7,
                end_to_end_success=oracle_e2e,
                tool_name_accuracy=0.92,
                argument_exact_match=0.85,
                policy_violation_rate=0.05,
                stage_failure_rate=0.10,
                failure_count=20,
                grounding_mode="oracle",
            ),
        ],
    )


# ── oracle gap rule ────────────────────────────────────────────────────────────

def test_oracle_gap_triggers_grounding_suggestion():
    report = _make_eval_report(stub_e2e=0.55, oracle_e2e=0.80)
    inference = _make_capability_inference()
    advisor = GraphRevisionAdvisor()
    result = advisor.advise(report, inference)

    targets = [s.target_capability for s in result.suggestions]
    assert "grounding" in targets


def test_oracle_gap_suggestion_is_high_priority():
    report = _make_eval_report(stub_e2e=0.55, oracle_e2e=0.80)
    inference = _make_capability_inference()
    advisor = GraphRevisionAdvisor()
    result = advisor.advise(report, inference)

    grounding_suggestions = [
        s for s in result.suggestions
        if s.target_capability == "grounding" and s.failure_pattern == "oracle_gap"
    ]
    assert len(grounding_suggestions) >= 1
    assert grounding_suggestions[0].priority == "high"


def test_no_oracle_gap_suggestion_when_gap_small():
    report = _make_eval_report(stub_e2e=0.75, oracle_e2e=0.80)
    inference = _make_capability_inference()
    advisor = GraphRevisionAdvisor()
    result = advisor.advise(report, inference)

    oracle_gap_suggestions = [
        s for s in result.suggestions if s.failure_pattern == "oracle_gap"
    ]
    assert len(oracle_gap_suggestions) == 0


# ── argument resolution rule ──────────────────────────────────────────────────

def test_low_arg_match_triggers_split_grounding_suggestion():
    report = _make_eval_report(stub_arg_match=0.30)
    inference = _make_capability_inference(peak_grounding_arg="new_item_ids")
    advisor = GraphRevisionAdvisor()
    result = advisor.advise(report, inference)

    split_suggestions = [
        s for s in result.suggestions
        if s.failure_pattern == "argument_resolution_failure"
    ]
    assert len(split_suggestions) >= 1
    assert "new_item_ids" in split_suggestions[0].suggestion


def test_high_arg_match_no_split_suggestion():
    report = _make_eval_report(stub_arg_match=0.80)
    inference = _make_capability_inference()
    advisor = GraphRevisionAdvisor()
    result = advisor.advise(report, inference)

    split_suggestions = [
        s for s in result.suggestions
        if s.failure_pattern == "argument_resolution_failure"
    ]
    assert len(split_suggestions) == 0


# ── policy violation rule ─────────────────────────────────────────────────────

def test_high_policy_violations_triggers_readiness_suggestion():
    report = _make_eval_report(stub_policy_viol=0.25)
    inference = _make_capability_inference()
    advisor = GraphRevisionAdvisor()
    result = advisor.advise(report, inference)

    readiness_suggestions = [
        s for s in result.suggestions if s.target_capability == "readiness"
    ]
    assert len(readiness_suggestions) >= 1


def test_low_policy_violations_no_readiness_suggestion():
    report = _make_eval_report(stub_policy_viol=0.05)
    inference = _make_capability_inference()
    advisor = GraphRevisionAdvisor()
    result = advisor.advise(report, inference)

    readiness_suggestions = [
        s for s in result.suggestions if s.target_capability == "readiness"
    ]
    assert len(readiness_suggestions) == 0


# ── recommended not beating minimal rule ─────────────────────────────────────

def test_recommended_not_beating_minimal_triggers_warning():
    report = _make_eval_report(stub_e2e=0.58, minimal_e2e=0.60)
    inference = _make_capability_inference()
    advisor = GraphRevisionAdvisor()
    result = advisor.advise(report, inference)

    topology_suggestions = [
        s for s in result.suggestions
        if s.failure_pattern == "recommended_not_beating_minimal"
    ]
    assert len(topology_suggestions) >= 1


def test_recommended_beating_minimal_no_topology_warning():
    report = _make_eval_report(stub_e2e=0.70, minimal_e2e=0.60)
    inference = _make_capability_inference()
    advisor = GraphRevisionAdvisor()
    result = advisor.advise(report, inference)

    topology_suggestions = [
        s for s in result.suggestions
        if s.failure_pattern == "recommended_not_beating_minimal"
    ]
    assert len(topology_suggestions) == 0


# ── report structure ──────────────────────────────────────────────────────────

def test_advisor_report_has_graph_id():
    report = _make_eval_report()
    inference = _make_capability_inference()
    advisor = GraphRevisionAdvisor()
    result = advisor.advise(report, inference)
    assert result.graph_id == "recommended_stub"


def test_advisor_report_suggestions_have_evidence():
    report = _make_eval_report(stub_e2e=0.55, oracle_e2e=0.80)
    inference = _make_capability_inference()
    advisor = GraphRevisionAdvisor()
    result = advisor.advise(report, inference)

    for s in result.suggestions:
        assert isinstance(s.evidence, list)
        assert len(s.evidence) > 0
