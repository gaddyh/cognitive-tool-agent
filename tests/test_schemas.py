"""Round-trip Pydantic parse/serialize tests for all schemas."""
import json

import pytest

from cognitive_tool_agent.schemas.common import Confidence, Evidence, ToolSchema, UserInput
from cognitive_tool_agent.schemas.perceive import MentionedEntity, PerceptionResult, RawFieldCandidate
from cognitive_tool_agent.schemas.reason import MissingRequirement, ReasoningResult, ResolvedEntity
from cognitive_tool_agent.schemas.readiness import ReadinessResult
from cognitive_tool_agent.schemas.plan import PlanResult, ToolCallPlan
from cognitive_tool_agent.schemas.act import ActionResult
from cognitive_tool_agent.schemas.learn import FailureAnalysis, LearningResult
from cognitive_tool_agent.schemas.trace import CognitiveTrace
from cognitive_tool_agent.schemas.dataset import DatasetRow, ExpectedBehavior
from cognitive_tool_agent.schemas.graph_spec import EdgeSpec, GraphSpec, NodeSpec
from cognitive_tool_agent.schemas.graph_builder import (
    BehaviorDecomposition,
    DatasetProfile,
    EvaluationPlan,
    FailureMap,
    GraphCandidate,
    GraphRevision,
    LabReport,
    RowFailure,
    StageMetric,
    TradeoffEntry,
)


def _roundtrip(model_instance):
    serialized = model_instance.model_dump_json()
    deserialized = type(model_instance).model_validate_json(serialized)
    assert deserialized == model_instance


def test_confidence_roundtrip():
    _roundtrip(Confidence(score=0.9, reason="test"))


def test_tool_schema_roundtrip():
    _roundtrip(ToolSchema(
        name="get_order_status",
        description="Gets status",
        required_fields=["order_id"],
    ))


def test_user_input_roundtrip():
    _roundtrip(UserInput(
        message="What is my order status?",
        conversation_context=["user: hello"],
        available_tools=[ToolSchema(name="get_order_status", description="desc")],
        world_state={"authenticated": True},
    ))


def test_perception_result_roundtrip():
    _roundtrip(PerceptionResult(
        intent_candidates=["use_get_order_status"],
        mentioned_entities=[MentionedEntity(text="#A1234", entity_type="order_id")],
        raw_field_candidates=[RawFieldCandidate(name="order_id", value="A1234", evidence_text="#A1234")],
        ambiguity_detected=False,
        ambiguity_type="none",
        candidate_tools=["get_order_status"],
        confidence=Confidence(score=0.85, reason="keyword match"),
        evidence=[Evidence(source="user_message", text="What is my order status?")],
    ))


def test_reasoning_result_roundtrip():
    _roundtrip(ReasoningResult(
        selected_intent="use_get_order_status",
        selected_tool="get_order_status",
        resolved_entities=[ResolvedEntity(
            surface_text="#A1234",
            entity_type="order_id",
            resolved_id="A1234",
            resolved_value="A1234",
            status="resolved",
        )],
        missing_requirements=[],
        reasoning_status="ready",
        confidence=Confidence(score=0.8, reason="resolved"),
    ))


def test_readiness_result_roundtrip():
    _roundtrip(ReadinessResult(
        ready=True,
        blocking_reasons=[],
        policy_violations=[],
        missing_required_fields=[],
        confidence=Confidence(score=0.9, reason="all clear"),
    ))


def test_plan_result_execute_tool_roundtrip():
    _roundtrip(PlanResult(
        next_action="execute_tool",
        tool_call=ToolCallPlan(tool_name="get_order_status", arguments={"order_id": "A1234"}),
        confidence=Confidence(score=0.9, reason="ready"),
    ))


def test_plan_result_ask_followup_roundtrip():
    _roundtrip(PlanResult(
        next_action="ask_followup",
        followup_question="Which order?",
        confidence=Confidence(score=0.7, reason="ambiguous"),
    ))


def test_action_result_roundtrip():
    _roundtrip(ActionResult(
        action_type="tool_executed",
        success=True,
        tool_name="get_order_status",
        tool_arguments={"order_id": "A1234"},
        tool_result={"status": "shipped"},
    ))


def test_learning_result_roundtrip():
    _roundtrip(LearningResult(
        should_add_to_dataset=False,
        dataset_split_suggestion="none",
        failure_analysis=FailureAnalysis(
            failed_stage="none",
            failure_type=None,
            explanation="no failure detected",
        ),
    ))


def test_learning_result_readiness_failure():
    result = LearningResult(
        should_add_to_dataset=True,
        dataset_split_suggestion="dev",
        failure_analysis=FailureAnalysis(
            failed_stage="readiness",
            failure_type="policy_violation",
            explanation="confirmation not provided",
        ),
    )
    _roundtrip(result)
    assert result.failure_analysis.failed_stage == "readiness"


def test_cognitive_trace_roundtrip():
    _roundtrip(CognitiveTrace(
        input=UserInput(message="hello"),
        perception=None,
        reasoning=None,
        readiness=None,
        plan=None,
        action=None,
        learning=None,
    ))


def test_dataset_row_roundtrip():
    _roundtrip(DatasetRow(
        id="tc-001",
        user_message="What is my order status?",
        tools=["get_order_status"],
        expected=ExpectedBehavior(
            expected_action="execute_tool",
            expected_tool="get_order_status",
            expected_arguments={"order_id": "A1234"},
        ),
        tags=["happy_path"],
    ))


def test_graph_spec_roundtrip():
    spec = GraphSpec(
        id="test_graph",
        nodes=[
            NodeSpec(id="perceive", role="perceive"),
            NodeSpec(id="plan", role="plan"),
            NodeSpec(id="act", role="act"),
        ],
        edges=[
            EdgeSpec(from_node="perceive", to_node="plan"),
            EdgeSpec(from_node="plan", to_node="act"),
        ],
    )
    _roundtrip(spec)
    order = spec.topological_order()
    assert [n.id for n in order] == ["perceive", "plan", "act"]


def test_graph_spec_topological_order_single_node():
    spec = GraphSpec(
        id="mono",
        nodes=[NodeSpec(id="monolithic", role="monolithic")],
        edges=[],
    )
    order = spec.topological_order()
    assert len(order) == 1
    assert order[0].role == "monolithic"


def test_graph_spec_topological_order_full_pipeline():
    roles = ["perceive", "reason", "readiness", "plan", "act", "learn"]
    nodes = [NodeSpec(id=r, role=r) for r in roles]
    edges = [EdgeSpec(from_node=roles[i], to_node=roles[i + 1]) for i in range(len(roles) - 1)]
    spec = GraphSpec(id="full", nodes=nodes, edges=edges)
    order = spec.topological_order()
    assert [n.role for n in order] == roles


def test_dataset_profile_roundtrip():
    _roundtrip(DatasetProfile(
        task_type="tool_calling",
        input_space="natural_language",
        output_space="tool_calls",
        label_set=["execute_tool", "ask_followup", "reject"],
        ambiguity_rate=0.2,
        contradiction_count=0,
        row_count=5,
        tool_count=3,
    ))


def test_lab_report_baseline_scores_structure():
    from cognitive_tool_agent.schemas.graph_spec import GraphSpec, NodeSpec
    graph = GraphSpec(id="g", nodes=[NodeSpec(id="monolithic", role="monolithic")])
    candidate = GraphCandidate(id="candidate_A", graph_spec=graph, rationale="test")
    profile = DatasetProfile(
        task_type="tool_calling",
        input_space="nl",
        output_space="tools",
        label_set=["execute_tool"],
        ambiguity_rate=0.0,
        contradiction_count=0,
        row_count=1,
        tool_count=1,
    )
    report = LabReport(
        dataset_profile=profile,
        behavior_decomposition=BehaviorDecomposition(
            task_type="tool_calling",
            stages=["perceive", "plan", "act"],
            rationale="test",
        ),
        candidates=[candidate],
        evaluation_plan=EvaluationPlan(metrics=[
            StageMetric(name="end_to_end_success", stage=None, description="test"),
        ]),
        baseline_scores={"candidate_A": {"end_to_end_success": 0.8}},
        optimized_scores={"candidate_C": {"end_to_end_success": 1.0}},
        failure_map=FailureMap(
            candidate_id="candidate_A",
            failures=[],
            dominant_failure_stage="none",
            dominant_failure_type="none",
            total_rows=1,
            failure_count=0,
        ),
        revision=None,
        tradeoff_summary=[
            TradeoffEntry(
                candidate_id="candidate_A",
                end_to_end_success=0.8,
                latency_estimate=1.0,
                cost_estimate=1.0,
                failure_count=0,
                recommendation="baseline",
            )
        ],
    )
    assert isinstance(report.baseline_scores, dict)
    assert "candidate_A" in report.baseline_scores
    assert isinstance(report.baseline_scores["candidate_A"], dict)
    assert isinstance(report.optimized_scores, dict)
    assert "candidate_C" in report.optimized_scores
    _roundtrip(report)
