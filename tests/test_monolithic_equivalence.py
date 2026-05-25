"""Monolithic equivalence test.

Verifies that make_monolithic_baseline() (plan→act two-node graph) produces
identical plan and action outputs to calling PlanAgent + ActAgent directly
with reasoning=None, readiness=None — the same starved-input behavior the
old _run_monolithic function implemented.

This is the oracle-gap safety net: if the baseline graph ever diverges from
the direct agent call, oracle-gap comparisons built against the baseline
become invalid.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from cognitive_tool_agent.datasets.loader import load_jsonl
from cognitive_tool_agent.tools.fake_tools import DEFAULT_REGISTRY

DATASET_PATH = Path(__file__).parent.parent / "data" / "dev" / "tool_calling_micro.jsonl"


@pytest.fixture(scope="module")
def rows():
    return load_jsonl(DATASET_PATH)


@pytest.mark.parametrize("row_index", range(5))
def test_monolithic_baseline_matches_direct_agent_call(rows, row_index):
    """Graph execution (plan→act) must produce identical plan and action to
    running PlanAgent + ActAgent directly with no upstream inputs."""
    from cognitive_tool_agent.agents.act_agent import ActAgent
    from cognitive_tool_agent.agents.plan_agent import PlanAgent
    from cognitive_tool_agent.graph.cognitive_graph import GraphExecutor, _row_to_user_input
    from cognitive_tool_agent.graph.node_input import NodeInput
    from cognitive_tool_agent.graph_builder.graph_candidate_generator import make_monolithic_baseline
    from cognitive_tool_agent.schemas.experiment import ExperimentSpec

    row = rows[row_index]
    registry = DEFAULT_REGISTRY
    user_input = _row_to_user_input(row, registry)

    plan_ctx = NodeInput(user_input=user_input, registry=registry, row=row)
    plan_agent = PlanAgent(mode="stub")
    expected_plan = plan_agent.run(plan_ctx)

    act_ctx = NodeInput(user_input=user_input, registry=registry, row=row, plan=expected_plan)
    act_agent = ActAgent(mode="stub")
    expected_action = act_agent.run(act_ctx)

    candidate = make_monolithic_baseline()
    executor = GraphExecutor()
    trace = executor.run(ExperimentSpec(graph=candidate.graph_spec), row, registry)

    assert trace.plan is not None, f"row {row.id}: trace.plan is None"
    assert trace.action is not None, f"row {row.id}: trace.action is None"
    assert trace.plan.next_action == expected_plan.next_action, (
        f"row {row.id}: plan.next_action mismatch: "
        f"{trace.plan.next_action!r} != {expected_plan.next_action!r}"
    )
    assert trace.action.action_type == expected_action.action_type, (
        f"row {row.id}: action_type mismatch: "
        f"{trace.action.action_type!r} != {expected_action.action_type!r}"
    )
    assert trace.action.tool_name == expected_action.tool_name, (
        f"row {row.id}: tool_name mismatch: "
        f"{trace.action.tool_name!r} != {expected_action.tool_name!r}"
    )
    assert trace.action.tool_arguments == expected_action.tool_arguments, (
        f"row {row.id}: tool_arguments mismatch: "
        f"{trace.action.tool_arguments!r} != {expected_action.tool_arguments!r}"
    )
    assert trace.perception is None, f"row {row.id}: perception should be None (no perceive node)"
    assert trace.reasoning is None, f"row {row.id}: reasoning should be None (no reason node)"
