"""Unit tests for the grounding eval dataset builder.

Exercises _build_rows() in isolation — no disk I/O, no LLM calls.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from cognitive_tool_agent.schemas.common import ToolSchema
from cognitive_tool_agent.schemas.trace_converter import CognitiveLabel, TurnSupervisionRow
from cognitive_tool_agent.tools.registry import ToolRegistry

_null_label = CognitiveLabel()


def _tool_turn(
    sim_id: str,
    turn_idx: int,
    tool_name: str,
    arguments: dict,
    split: str = "dev",
    difficulty_bucket: str = "easy",
) -> TurnSupervisionRow:
    return TurnSupervisionRow(
        turn_id=f"{sim_id}:{turn_idx}",
        simulation_id=sim_id,
        task_id="task-001",
        turn_idx=turn_idx,
        role="assistant",
        content=None,
        cognitive_label=CognitiveLabel(
            plan_next_action="call_tool",
            plan_tool_name=tool_name,
            plan_arguments=arguments,
        ),
        split=split,
        difficulty_bucket=difficulty_bucket,
    )


def _user_turn(sim_id: str, turn_idx: int, content: str) -> TurnSupervisionRow:
    return TurnSupervisionRow(
        turn_id=f"{sim_id}:{turn_idx}",
        simulation_id=sim_id,
        task_id="task-001",
        turn_idx=turn_idx,
        role="user",
        content=content,
        cognitive_label=_null_label,
        split="dev",
    )


def _make_registry(*tool_specs: tuple[str, list[str]]) -> ToolRegistry:
    registry = ToolRegistry()
    for name, required in tool_specs:
        schema = ToolSchema(
            name=name,
            description=f"Tool: {name}",
            required_fields=required,
        )
        registry.register(schema, lambda **kwargs: {"status": "ok"})
    return registry


def _build(turns, registry, mode="required"):
    from build_grounding_eval_dataset import _build_rows
    return _build_rows(turns, registry, mode)


def test_basic_row_produced():
    sim_id = "sim-001"
    registry = _make_registry(("cancel_order", ["order_id"]))
    turns = [
        _user_turn(sim_id, 0, "Cancel my order #W999"),
        _tool_turn(sim_id, 1, "cancel_order", {"order_id": "#W999"}),
    ]
    result = _build(turns, registry)
    assert "dev" in result
    assert len(result["dev"]) == 1
    row = result["dev"][0]
    assert row["selected_tool"] == "cancel_order"
    assert row["target_args"] == {"order_id": "#W999"}
    assert "order_id" in row["target_fields"]


def test_target_args_filtered_to_required_by_default():
    sim_id = "sim-002"
    registry = _make_registry(("exchange_order", ["order_id", "item_ids"]))
    turns = [
        _user_turn(sim_id, 0, "Exchange item"),
        _tool_turn(sim_id, 1, "exchange_order", {
            "order_id": "#W1",
            "item_ids": ["ITEM-1"],
            "note": "fragile",
        }),
    ]
    result = _build(turns, registry, mode="required")
    row = result["dev"][0]
    assert "note" not in row["target_args"]
    assert "note" not in row["target_fields"]
    assert "order_id" in row["target_args"]
    assert "item_ids" in row["target_args"]


def test_target_args_all_expected_mode():
    sim_id = "sim-003"
    registry = _make_registry(("exchange_order", ["order_id"]))
    turns = [
        _user_turn(sim_id, 0, "Exchange item"),
        _tool_turn(sim_id, 1, "exchange_order", {
            "order_id": "#W1",
            "note": "fragile",
        }),
    ]
    result = _build(turns, registry, mode="all-expected")
    row = result["dev"][0]
    assert "note" in row["target_args"]
    assert "note" in row["target_fields"]


def test_conversation_context_populated():
    sim_id = "sim-004"
    registry = _make_registry(("cancel_order", ["order_id"]))
    turns = [
        _user_turn(sim_id, 0, "Hello"),
        _user_turn(sim_id, 1, "Cancel order #W1"),
        _tool_turn(sim_id, 2, "cancel_order", {"order_id": "#W1"}),
    ]
    result = _build(turns, registry)
    row = result["dev"][0]
    assert "Hello" in row["conversation_context"]
    assert row["user_message"] == "Cancel order #W1"


def test_prior_tool_calls_accumulated():
    sim_id = "sim-005"
    registry = _make_registry(
        ("get_order", ["order_id"]),
        ("cancel_order", ["order_id"]),
    )
    turns = [
        _user_turn(sim_id, 0, "What's my order?"),
        _tool_turn(sim_id, 1, "get_order", {"order_id": "#W1"}),
        _user_turn(sim_id, 2, "Cancel it"),
        _tool_turn(sim_id, 3, "cancel_order", {"order_id": "#W1"}),
    ]
    result = _build(turns, registry)
    rows = result["dev"]
    assert len(rows) == 2
    cancel_row = next(r for r in rows if r["selected_tool"] == "cancel_order")
    prior_calls = cancel_row["available_state"]["prior_tool_calls"]
    assert any(c["tool_name"] == "get_order" for c in prior_calls)


def test_current_deterministic_args_populated():
    sim_id = "sim-006"
    registry = _make_registry(("cancel_order", ["order_id"]))
    turns = [
        _user_turn(sim_id, 0, "Cancel order #W999"),
        _tool_turn(sim_id, 1, "cancel_order", {"order_id": "#W999"}),
    ]
    result = _build(turns, registry)
    row = result["dev"][0]
    assert "current_deterministic_args" in row
    assert isinstance(row["current_deterministic_args"], dict)


def test_split_routing():
    sim_id_train = "sim-train-001"
    sim_id_test = "sim-test-001"
    registry = _make_registry(("cancel_order", ["order_id"]))
    turns = [
        TurnSupervisionRow(
            turn_id=f"{sim_id_train}:1",
            simulation_id=sim_id_train,
            task_id="t",
            turn_idx=1,
            role="assistant",
            content=None,
            cognitive_label=CognitiveLabel(
                plan_next_action="call_tool",
                plan_tool_name="cancel_order",
                plan_arguments={"order_id": "#W1"},
            ),
            split="train",
            difficulty_bucket="easy",
        ),
        TurnSupervisionRow(
            turn_id=f"{sim_id_test}:1",
            simulation_id=sim_id_test,
            task_id="t",
            turn_idx=1,
            role="assistant",
            content=None,
            cognitive_label=CognitiveLabel(
                plan_next_action="call_tool",
                plan_tool_name="cancel_order",
                plan_arguments={"order_id": "#W2"},
            ),
            split="test",
            difficulty_bucket="easy",
        ),
    ]
    result = _build(turns, registry)
    assert len(result.get("train", [])) == 1
    assert len(result.get("test", [])) == 1
    assert result["train"][0]["target_args"]["order_id"] == "#W1"
    assert result["test"][0]["target_args"]["order_id"] == "#W2"


def test_unknown_tool_skipped_gracefully():
    sim_id = "sim-007"
    registry = _make_registry(("cancel_order", ["order_id"]))
    turns = [
        _user_turn(sim_id, 0, "Do something"),
        _tool_turn(sim_id, 1, "nonexistent_tool", {"foo": "bar"}),
    ]
    result = _build(turns, registry)
    all_rows = [r for rows in result.values() for r in rows]
    assert all_rows == []


def test_target_args_target_fields_consistent():
    sim_id = "sim-008"
    registry = _make_registry(("get_product", ["product_id"]))
    turns = [
        _user_turn(sim_id, 0, "Tell me about item X"),
        _tool_turn(sim_id, 1, "get_product", {"product_id": "PROD-42"}),
    ]
    result = _build(turns, registry)
    row = result["dev"][0]
    assert set(row["target_fields"]) == set(row["target_args"].keys())


def test_difficulty_bucket_propagated():
    sim_id = "sim-009"
    registry = _make_registry(("exchange_order", ["order_id", "item_ids"]))
    turns = [
        _tool_turn(
            sim_id, 1, "exchange_order",
            {"order_id": "#W1", "item_ids": ["ITEM-1"]},
            difficulty_bucket="hard",
        )
    ]
    result = _build(turns, registry)
    row = result["dev"][0]
    assert row["difficulty_bucket"] == "hard"


def test_target_args_empty_when_no_matching_required_fields():
    sim_id = "sim-010"
    registry = _make_registry(("get_order", ["order_id"]))
    turns = [
        _tool_turn(sim_id, 1, "get_order", {"unrelated_field": "value"}),
    ]
    result = _build(turns, registry)
    row = result["dev"][0]
    assert row["target_args"] == {}
    assert row["target_fields"] == []
