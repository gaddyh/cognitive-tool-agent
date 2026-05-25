"""Tests for graph_runner: ActionSequenceAdapter, TraceWriter, and GraphEvaluationRunner."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from cognitive_tool_agent.graph_runner.dataset_adapter import (
    ActionSequenceAdapter,
    _is_write_action,
    _select_primary_action,
)
from cognitive_tool_agent.graph_runner.trace_writer import TraceWriter
from cognitive_tool_agent.schemas.trace_converter import AlignedAction, ActionSequenceRow


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_action(tool: str, expected_args: dict | None = None) -> AlignedAction:
    return AlignedAction(
        expected_action_id="a1",
        expected_tool=tool,
        expected_arguments=expected_args or {},
        action_match=True,
        action_reward=1.0,
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


# ── _is_write_action ──────────────────────────────────────────────────────────

def test_is_write_action_detects_modify():
    assert _is_write_action("modify_pending_order_items") is True


def test_is_write_action_detects_cancel():
    assert _is_write_action("cancel_order") is True


def test_is_write_action_read_is_false():
    assert _is_write_action("get_order_details") is False


# ── _select_primary_action ────────────────────────────────────────────────────

def test_select_primary_prefers_write_action():
    actions = [
        _make_action("get_user_details"),
        _make_action("get_order_details"),
        _make_action("cancel_order"),
    ]
    primary, reason = _select_primary_action(actions)
    assert primary.expected_tool == "cancel_order"
    assert reason == "last_write_action"


def test_select_primary_fallback_to_last_when_no_write():
    actions = [
        _make_action("get_user_details"),
        _make_action("get_order_details"),
    ]
    primary, reason = _select_primary_action(actions)
    assert primary.expected_tool == "get_order_details"
    assert reason == "last_action_fallback"


# ── ActionSequenceAdapter ─────────────────────────────────────────────────────

@pytest.fixture()
def synthetic_data(tmp_path):
    action_seq = [
        {
            "simulation_id": "sim-aabbccdd-0001",
            "task_id": "task_1",
            "aligned_actions": [
                {
                    "expected_action_id": "1_0",
                    "expected_tool": "get_user_details",
                    "expected_arguments": {"user_id": "u001"},
                    "action_match": True,
                    "action_reward": 1.0,
                },
                {
                    "expected_action_id": "1_1",
                    "expected_tool": "cancel_order",
                    "expected_arguments": {"order_id": "#W123", "reason": "user_request"},
                    "action_match": False,
                    "action_reward": 0.0,
                },
            ],
        }
    ]
    turn_sup = [
        {
            "turn_id": "t1",
            "simulation_id": "sim-aabbccdd-0001",
            "task_id": "task_1",
            "turn_idx": 0,
            "role": "user",
            "content": "Please cancel my order",
            "cognitive_label": {"plan_next_action": None, "plan_tool_name": None, "plan_arguments": {}},
        }
    ]
    tool_registry = {
        "get_user_details": {
            "name": "get_user_details",
            "required_args": ["user_id"],
            "seen_args": ["user_id"],
            "usage_count": 10,
            "tool_type": "read",
        },
        "cancel_order": {
            "name": "cancel_order",
            "required_args": ["order_id"],
            "seen_args": ["order_id", "reason"],
            "usage_count": 5,
            "tool_type": "write",
        },
    }

    action_seq_path = tmp_path / "action_sequence.jsonl"
    turn_sup_path = tmp_path / "turn_supervision.jsonl"
    tool_registry_path = tmp_path / "tool_registry.json"

    _write_jsonl(action_seq_path, action_seq)
    _write_jsonl(turn_sup_path, turn_sup)
    with open(tool_registry_path, "w") as f:
        json.dump(tool_registry, f)

    return action_seq_path, turn_sup_path, tool_registry_path


def test_adapter_produces_one_row_per_simulation(synthetic_data):
    action_seq_path, turn_sup_path, tool_registry_path = synthetic_data
    adapter = ActionSequenceAdapter()
    rows, registry = adapter.load(action_seq_path, turn_sup_path, tool_registry_path)
    assert len(rows) == 1


def test_adapter_row_uses_user_message(synthetic_data):
    action_seq_path, turn_sup_path, tool_registry_path = synthetic_data
    adapter = ActionSequenceAdapter()
    rows, _ = adapter.load(action_seq_path, turn_sup_path, tool_registry_path)
    assert rows[0].user_message == "Please cancel my order"


def test_adapter_row_expected_is_write_tool(synthetic_data):
    action_seq_path, turn_sup_path, tool_registry_path = synthetic_data
    adapter = ActionSequenceAdapter()
    rows, _ = adapter.load(action_seq_path, turn_sup_path, tool_registry_path)
    assert rows[0].expected.expected_tool == "cancel_order"
    assert rows[0].expected.expected_action == "tool_executed"


def test_adapter_row_world_state_has_selection_reason(synthetic_data):
    action_seq_path, turn_sup_path, tool_registry_path = synthetic_data
    adapter = ActionSequenceAdapter()
    rows, _ = adapter.load(action_seq_path, turn_sup_path, tool_registry_path)
    assert rows[0].world_state["primary_action_selection_reason"] == "last_write_action"


def test_adapter_registry_contains_tools(synthetic_data):
    action_seq_path, turn_sup_path, tool_registry_path = synthetic_data
    adapter = ActionSequenceAdapter()
    _, registry = adapter.load(action_seq_path, turn_sup_path, tool_registry_path)
    assert registry.lookup("cancel_order") is not None
    assert registry.lookup("get_user_details") is not None


def test_adapter_registry_has_required_fields(synthetic_data):
    action_seq_path, turn_sup_path, tool_registry_path = synthetic_data
    adapter = ActionSequenceAdapter()
    _, registry = adapter.load(action_seq_path, turn_sup_path, tool_registry_path)
    schema = registry.lookup("cancel_order")
    assert "order_id" in schema.required_fields


# ── TraceWriter ───────────────────────────────────────────────────────────────

def test_trace_writer_roundtrip(tmp_path):
    from cognitive_tool_agent.graph.cognitive_graph import GraphExecutor
    from cognitive_tool_agent.graph_builder.graph_candidate_generator import _make_monolithic
    from cognitive_tool_agent.tools.fake_tools import DEFAULT_REGISTRY
    from cognitive_tool_agent.datasets.loader import load_jsonl

    micro_path = (
        Path(__file__).parent.parent / "data" / "dev" / "tool_calling_micro.jsonl"
    )
    rows = load_jsonl(micro_path)
    candidate = _make_monolithic()
    executor = GraphExecutor()
    traces = [executor.run(candidate.graph_spec, row, DEFAULT_REGISTRY) for row in rows]

    out_path = tmp_path / "traces.jsonl"
    writer = TraceWriter()
    writer.write(traces, out_path)
    loaded = writer.load(out_path)

    assert len(loaded) == len(traces)
    for orig, loaded_trace in zip(traces, loaded):
        assert orig.input.message == loaded_trace.input.message


# ── GraphEvaluationRunner (smoke) ─────────────────────────────────────────────

def test_graph_evaluation_runner_produces_four_results(synthetic_data, tmp_path):
    from cognitive_tool_agent.graph_runner.run_recommended_graph import GraphEvaluationRunner
    from cognitive_tool_agent.recommender.graph_recommender import GraphRecommender
    from cognitive_tool_agent.recommender.capability_inference import CapabilityInferenceEngine

    action_seq_path, turn_sup_path, tool_registry_path = synthetic_data

    reports_dir = Path(__file__).parent.parent / "reports"
    report_path = reports_dir / "cognitive_dataset_report.json"
    if not report_path.exists():
        pytest.skip("cognitive_dataset_report.json not available")

    recommended_path = reports_dir / "recommended_graph.json"
    if not recommended_path.exists():
        pytest.skip("recommended_graph.json not available")

    runner = GraphEvaluationRunner()
    eval_report = runner.run(
        recommended_graph_path=recommended_path,
        action_seq_path=action_seq_path,
        turn_sup_path=turn_sup_path,
        tool_registry_path=tool_registry_path,
        out_dir=tmp_path,
    )

    assert len(eval_report.results) == 4
    graph_ids = {r.graph_id for r in eval_report.results}
    assert "monolithic" in graph_ids
    assert "minimal" in graph_ids
    assert "recommended_stub" in graph_ids
    assert "recommended_oracle" in graph_ids


def test_graph_evaluation_oracle_confidence_not_below_stub(synthetic_data, tmp_path):
    from cognitive_tool_agent.graph_runner.run_recommended_graph import GraphEvaluationRunner

    action_seq_path, turn_sup_path, tool_registry_path = synthetic_data
    recommended_path = (
        Path(__file__).parent.parent / "reports" / "recommended_graph.json"
    )
    if not recommended_path.exists():
        pytest.skip("recommended_graph.json not available")

    runner = GraphEvaluationRunner()
    eval_report = runner.run(
        recommended_graph_path=recommended_path,
        action_seq_path=action_seq_path,
        turn_sup_path=turn_sup_path,
        tool_registry_path=tool_registry_path,
        out_dir=tmp_path,
    )

    stub = next(r for r in eval_report.results if r.graph_id == "recommended_stub")
    oracle = next(r for r in eval_report.results if r.graph_id == "recommended_oracle")
    assert oracle.argument_exact_match >= stub.argument_exact_match


# ── GroundingAgent ────────────────────────────────────────────────────────────

def test_grounding_agent_stub_returns_grounding_result():
    from cognitive_tool_agent.agents.grounding_agent import GroundingAgent
    from cognitive_tool_agent.schemas.common import UserInput
    from cognitive_tool_agent.schemas.dataset import DatasetRow, ExpectedBehavior

    agent = GroundingAgent(mode="stub")
    user_input = UserInput(message="Cancel my order", available_tools=[])
    row = DatasetRow(
        id="r1",
        user_message="Cancel my order",
        expected=ExpectedBehavior(expected_action="tool_executed", expected_tool="cancel_order"),
    )
    result = agent.run(user_input, reasoning=None, row=row)
    assert result.grounding_mode == "stub"
    assert 0.0 <= result.grounding_confidence <= 1.0


def test_grounding_agent_oracle_uses_expected_args():
    from cognitive_tool_agent.agents.grounding_agent import GroundingAgent
    from cognitive_tool_agent.schemas.common import UserInput
    from cognitive_tool_agent.schemas.dataset import DatasetRow, ExpectedBehavior

    agent = GroundingAgent(mode="oracle")
    user_input = UserInput(message="Cancel order", available_tools=[])
    row = DatasetRow(
        id="r2",
        user_message="Cancel order",
        expected=ExpectedBehavior(
            expected_action="tool_executed",
            expected_tool="cancel_order",
            expected_arguments={"order_id": "#W999"},
        ),
    )
    result = agent.run(user_input, reasoning=None, row=row)
    assert result.grounding_mode == "oracle"
    assert result.resolved_args["order_id"] == "#W999"
    assert result.grounding_confidence == 1.0


def test_grounding_agent_disabled_returns_empty():
    from cognitive_tool_agent.agents.grounding_agent import GroundingAgent
    from cognitive_tool_agent.schemas.common import UserInput
    from cognitive_tool_agent.schemas.dataset import DatasetRow, ExpectedBehavior

    agent = GroundingAgent(mode="disabled")
    user_input = UserInput(message="x", available_tools=[])
    row = DatasetRow(
        id="r3",
        user_message="x",
        expected=ExpectedBehavior(expected_action="tool_executed"),
    )
    result = agent.run(user_input, reasoning=None, row=row)
    assert result.grounding_mode == "disabled"
    assert result.resolved_args == {}


def test_graph_executor_runs_grounding_node():
    from cognitive_tool_agent.graph.cognitive_graph import GraphExecutor
    from cognitive_tool_agent.schemas.graph_spec import EdgeSpec, GraphSpec, NodeSpec
    from cognitive_tool_agent.schemas.dataset import DatasetRow, ExpectedBehavior
    from cognitive_tool_agent.tools.fake_tools import DEFAULT_REGISTRY

    graph = GraphSpec(
        id="test_grounding",
        nodes=[
            NodeSpec(id="perceive", role="perceive"),
            NodeSpec(id="reason", role="reason"),
            NodeSpec(id="grounding", role="grounding"),
            NodeSpec(id="plan", role="plan"),
            NodeSpec(id="act", role="act"),
        ],
        edges=[
            EdgeSpec(from_node="perceive", to_node="reason"),
            EdgeSpec(from_node="reason", to_node="grounding"),
            EdgeSpec(from_node="grounding", to_node="plan"),
            EdgeSpec(from_node="plan", to_node="act"),
        ],
    )
    row = DatasetRow(
        id="g1",
        user_message="What is my order status?",
        tools=["get_order_status"],
        expected=ExpectedBehavior(
            expected_action="tool_executed",
            expected_tool="get_order_status",
        ),
    )
    from cognitive_tool_agent.graph.cognitive_graph import NodeConfig
    executor = GraphExecutor(node_configs={"grounding": NodeConfig(mode="stub")})
    trace = executor.run(graph, row, DEFAULT_REGISTRY)

    assert trace.grounding is not None
    assert trace.grounding.grounding_mode == "stub"
    assert trace.perception is not None
    assert trace.reasoning is not None
    assert trace.action is not None
