"""Phase 5 verification: NodeInput narrowing invariants.

Confirms that _build_node_input constructs a NodeInput containing only the
slots that incoming edges provide (filtered by ROLE_INPUTS), and that the
learn invariant is not violated.

These tests are structural: they verify what GraphExecutor passes to each
agent, not what the agents return.
"""
from __future__ import annotations

import pytest

from cognitive_tool_agent.datasets.loader import load_jsonl
from cognitive_tool_agent.graph.cognitive_graph import GraphExecutor, RunContext, _row_to_user_input
from cognitive_tool_agent.graph.node_input import NodeInput
from cognitive_tool_agent.graph_builder.graph_candidate_generator import make_full_pipeline
from cognitive_tool_agent.schemas.experiment import ExperimentSpec
from cognitive_tool_agent.tools.fake_tools import DEFAULT_REGISTRY

from pathlib import Path

DATASET_PATH = Path(__file__).parent.parent / "data" / "dev" / "tool_calling_micro.jsonl"


@pytest.fixture(scope="module")
def rows():
    return load_jsonl(DATASET_PATH)


@pytest.fixture(scope="module")
def first_row(rows):
    return rows[0]


def test_learn_node_trace_so_far_is_never_edge_supplied(first_row):
    """_build_node_input for learn never includes edge-supplied slots —
    ROLE_INPUTS['learn'] is empty, so the slot guard rejects everything.
    trace_so_far is populated from RunContext.to_trace(), not from edges."""
    from cognitive_tool_agent.schemas.graph_spec import NodeSpec

    candidate = make_full_pipeline()
    graph = candidate.graph_spec
    registry = DEFAULT_REGISTRY

    user_input = _row_to_user_input(first_row, registry)
    executor = GraphExecutor()

    # Manually run the full pipeline to populate ctx
    experiment = ExperimentSpec(graph=graph)
    trace = executor.run(experiment, first_row, registry)

    # Reconstruct ctx at learn-dispatch time (after all prior nodes have run)
    ctx = RunContext(row=first_row, registry=registry, user_input=user_input)
    ctx.perception = trace.perception
    ctx.reasoning = trace.reasoning
    ctx.readiness = trace.readiness
    ctx.plan = trace.plan
    ctx.action = trace.action

    learn_node = graph.node_map["learn"]
    node_input = executor._build_node_input(learn_node, ctx, graph)

    # trace_so_far must be populated (from RunContext)
    assert node_input.trace_so_far is not None, (
        "learn node: trace_so_far should be set from RunContext.to_trace()"
    )
    # No edge-supplied data — learn has no inputs in ROLE_INPUTS
    assert node_input.perception is None, "learn: perception should not be edge-supplied"
    assert node_input.reasoning is None, "learn: reasoning should not be edge-supplied"
    assert node_input.plan is None, "learn: plan should not be edge-supplied"
    assert node_input.action is None, "learn: action should not be edge-supplied"


def test_perceive_node_gets_no_edge_inputs(first_row):
    """perceive has no incoming edges and ROLE_INPUTS['perceive'] = {}
    → its NodeInput carries only ambient context."""
    candidate = make_full_pipeline()
    graph = candidate.graph_spec
    registry = DEFAULT_REGISTRY

    user_input = _row_to_user_input(first_row, registry)
    ctx = RunContext(row=first_row, registry=registry, user_input=user_input)
    executor = GraphExecutor()

    perceive_node = graph.node_map["perceive"]
    node_input = executor._build_node_input(perceive_node, ctx, graph)

    assert node_input.user_input == user_input
    assert node_input.perception is None
    assert node_input.reasoning is None
    assert node_input.trace_so_far is None


def test_act_node_gets_only_plan_slot(first_row):
    """act's ROLE_INPUTS = {'plan': 'required'}
    → _build_node_input supplies plan and nothing else from edges."""
    from cognitive_tool_agent.schemas.plan import PlanResult

    candidate = make_full_pipeline()
    graph = candidate.graph_spec
    registry = DEFAULT_REGISTRY

    user_input = _row_to_user_input(first_row, registry)
    ctx = RunContext(row=first_row, registry=registry, user_input=user_input)
    from cognitive_tool_agent.schemas.common import Confidence
    fake_plan = PlanResult(
        next_action="execute_tool",
        confidence=Confidence(score=0.9, reason="test"),
    )
    ctx.plan = fake_plan
    ctx.reasoning = None
    executor = GraphExecutor()

    act_node = graph.node_map["act"]
    node_input = executor._build_node_input(act_node, ctx, graph)

    assert node_input.plan is fake_plan, "act: plan should be edge-supplied"
    assert node_input.reasoning is None, "act: reasoning is not in ROLE_INPUTS['act']"
    assert node_input.trace_so_far is None, "act: trace_so_far should be None (not learn)"


def test_plan_node_gets_reasoning_and_readiness(first_row):
    """plan's ROLE_INPUTS = {'reasoning': 'optional', 'readiness': 'optional'}
    → _build_node_input supplies both slots when available."""
    from cognitive_tool_agent.schemas.reason import ReasoningResult
    from cognitive_tool_agent.schemas.readiness import ReadinessResult
    from cognitive_tool_agent.schemas.common import Confidence

    candidate = make_full_pipeline()
    graph = candidate.graph_spec
    registry = DEFAULT_REGISTRY

    user_input = _row_to_user_input(first_row, registry)
    ctx = RunContext(row=first_row, registry=registry, user_input=user_input)
    ctx.reasoning = ReasoningResult(
        selected_intent="use_get_order_status",
        selected_tool="get_order_status",
        reasoning_status="ready",
        confidence=Confidence(score=0.9, reason="test"),
    )
    ctx.readiness = ReadinessResult(
        ready=True,
        confidence=Confidence(score=0.9, reason="test"),
    )
    executor = GraphExecutor()

    plan_node = graph.node_map["plan"]
    node_input = executor._build_node_input(plan_node, ctx, graph)

    assert node_input.reasoning is ctx.reasoning, "plan: reasoning should be edge-supplied"
    assert node_input.readiness is ctx.readiness, "plan: readiness should be edge-supplied"
    assert node_input.perception is None, "plan: perception is not in ROLE_INPUTS['plan']"
    assert node_input.trace_so_far is None, "plan: trace_so_far should be None (not learn)"
