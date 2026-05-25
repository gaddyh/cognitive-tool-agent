"""Tests for TurnToolCallAdapter row construction."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from cognitive_tool_agent.graph_runner.turn_tool_call_adapter import TurnToolCallAdapter


def _write_turns(tmp_path: Path, turns: list[dict]) -> Path:
    p = tmp_path / "turns.jsonl"
    p.write_text("\n".join(json.dumps(t) for t in turns), encoding="utf-8")
    return p


def _write_registry(tmp_path: Path, tools: dict) -> Path:
    p = tmp_path / "registry.json"
    p.write_text(json.dumps(tools), encoding="utf-8")
    return p


_REGISTRY = {
    "cancel_pending_order": {
        "required_args": ["order_id"],
        "seen_args": ["order_id"],
    },
    "get_user_details": {
        "required_args": ["user_id"],
        "seen_args": ["user_id"],
    },
}

_SIM = "sim-abc123-0000-0000-0000-000000000000"
_TASK = "7"

_TURNS = [
    {
        "turn_id": f"{_SIM}_turn_0",
        "simulation_id": _SIM,
        "task_id": _TASK,
        "turn_idx": 0,
        "role": "assistant",
        "content": "Hi! How can I help?",
        "cognitive_label": {
            "plan_next_action": "ask_followup",
            "plan_tool_name": None,
            "plan_arguments": {},
        },
    },
    {
        "turn_id": f"{_SIM}_turn_1",
        "simulation_id": _SIM,
        "task_id": _TASK,
        "turn_idx": 1,
        "role": "user",
        "content": "Please cancel order #W123.",
        "cognitive_label": {
            "plan_next_action": None,
            "plan_tool_name": None,
            "plan_arguments": {},
        },
    },
    {
        "turn_id": f"{_SIM}_turn_2",
        "simulation_id": _SIM,
        "task_id": _TASK,
        "turn_idx": 2,
        "role": "assistant",
        "content": None,
        "cognitive_label": {
            "plan_next_action": "call_tool",
            "plan_tool_name": "cancel_pending_order",
            "plan_arguments": {"order_id": "#W123"},
        },
    },
]


def test_adapter_creates_one_row_per_call_tool_turn(tmp_path):
    turns_path = _write_turns(tmp_path, _TURNS)
    registry_path = _write_registry(tmp_path, _REGISTRY)

    rows, registry = TurnToolCallAdapter().load(turns_path, registry_path)

    assert len(rows) == 1
    row = rows[0]
    assert row.expected.expected_tool == "cancel_pending_order"
    assert row.expected.expected_arguments == {"order_id": "#W123"}
    assert row.expected.expected_action == "tool_executed"


def test_adapter_uses_last_user_message_before_tool_call_turn(tmp_path):
    turns_path = _write_turns(tmp_path, _TURNS)
    registry_path = _write_registry(tmp_path, _REGISTRY)

    rows, _ = TurnToolCallAdapter().load(turns_path, registry_path)

    assert rows[0].user_message == "Please cancel order #W123."


def test_adapter_world_state_metadata(tmp_path):
    turns_path = _write_turns(tmp_path, _TURNS)
    registry_path = _write_registry(tmp_path, _REGISTRY)

    rows, _ = TurnToolCallAdapter().load(turns_path, registry_path)
    ws = rows[0].world_state

    assert ws["simulation_id"] == _SIM
    assert ws["task_id"] == _TASK
    assert ws["turn_idx"] == 2
    assert ws["source"] == "turn_tool_call"
    assert ws["primary_tool"] == "cancel_pending_order"


def test_adapter_skips_non_call_tool_assistant_turns(tmp_path):
    turns_path = _write_turns(tmp_path, _TURNS)
    registry_path = _write_registry(tmp_path, _REGISTRY)

    rows, _ = TurnToolCallAdapter().load(turns_path, registry_path)

    ids = [r.id for r in rows]
    assert all("turn:2" in rid for rid in ids), "only turn_idx=2 should be included"


def test_adapter_empty_plan_arguments_becomes_none(tmp_path):
    turns_with_no_args = [
        *_TURNS[:2],
        {
            "turn_id": f"{_SIM}_turn_2",
            "simulation_id": _SIM,
            "task_id": _TASK,
            "turn_idx": 2,
            "role": "assistant",
            "content": None,
            "cognitive_label": {
                "plan_next_action": "call_tool",
                "plan_tool_name": "get_user_details",
                "plan_arguments": {},
            },
        },
    ]
    turns_path = _write_turns(tmp_path, turns_with_no_args)
    registry_path = _write_registry(tmp_path, _REGISTRY)

    rows, _ = TurnToolCallAdapter().load(turns_path, registry_path)

    assert rows[0].expected.expected_arguments is None


def test_adapter_registry_contains_all_tools(tmp_path):
    turns_path = _write_turns(tmp_path, _TURNS)
    registry_path = _write_registry(tmp_path, _REGISTRY)

    rows, registry = TurnToolCallAdapter().load(turns_path, registry_path)

    assert set(registry.names()) == set(_REGISTRY.keys())
    assert set(rows[0].tools) == set(_REGISTRY.keys())
