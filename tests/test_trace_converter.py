"""Tests for the Trace-to-Cognitive-Dataset converter pipeline."""
import json
import tempfile
from pathlib import Path

import pytest

from cognitive_tool_agent.schemas.simulation import (
    ActionCheck,
    ExpectedAction,
    RawSimulation,
    RawTask,
    RewardInfo,
    SimulationFile,
    SimulationMessage,
    ToolCall,
)
from cognitive_tool_agent.schemas.trace_converter import (
    ConversionSummary,
)
from cognitive_tool_agent.trace_converter.action_aligner import align_actions
from cognitive_tool_agent.trace_converter.failure_extractor import extract_failures
from cognitive_tool_agent.trace_converter.tool_registry_scanner import scan_tool_registry
from cognitive_tool_agent.trace_converter.turn_supervisor import supervise_turns
from cognitive_tool_agent.trace_converter.converter import TraceConverter


# ---------------------------------------------------------------------------
# Shared fixture data
# ---------------------------------------------------------------------------

EXPECTED_ACTIONS = [
    ExpectedAction(action_id="0_0", name="find_user_id_by_name_zip",
                   arguments={"first_name": "Yusuf", "last_name": "Rossi", "zip": "19122"},
                   tool_type="read"),
    ExpectedAction(action_id="0_1", name="get_order_details",
                   arguments={"order_id": "#W2378156"}, tool_type="read"),
    ExpectedAction(action_id="0_2", name="exchange_delivered_order_items",
                   arguments={"order_id": "#W2378156", "item_ids": ["A"], "new_item_ids": ["B"],
                               "payment_method_id": "cc_1"},
                   tool_type="write"),
]

MESSAGES = [
    SimulationMessage(role="assistant", content="Hi! How can I help?", turn_idx=0),
    SimulationMessage(role="user",
                      content="Hi! My name is Yusuf Rossi and zip is 19122.",
                      turn_idx=1),
    SimulationMessage(role="assistant", content=None,
                      tool_calls=[ToolCall(id="c1", name="find_user_id_by_name_zip",
                                           arguments={"first_name": "Yusuf", "last_name": "Rossi",
                                                      "zip": "19122"})],
                      turn_idx=2),
    SimulationMessage(role="tool", content="yusuf_rossi_9620", id="c1", turn_idx=3),
    SimulationMessage(role="assistant", content=None,
                      tool_calls=[ToolCall(id="c2", name="get_order_details",
                                           arguments={"order_id": "#W2378156"})],
                      turn_idx=4),
    SimulationMessage(role="tool", content='{"status": "delivered"}', id="c2", turn_idx=5),
    SimulationMessage(role="assistant", content="Let me confirm the exchange details.", turn_idx=6),
    SimulationMessage(role="user", content="Yes, go ahead.", turn_idx=7),
    SimulationMessage(role="assistant", content=None,
                      tool_calls=[ToolCall(id="c3", name="exchange_delivered_order_items",
                                           arguments={"order_id": "#W2378156",
                                                      "item_ids": ["A"], "new_item_ids": ["X"],
                                                      "payment_method_id": "cc_1"})],
                      turn_idx=8),
    SimulationMessage(role="tool", content='{"status": "exchange requested"}', id="c3", turn_idx=9),
    SimulationMessage(role="user", content="Thanks! ###STOP###", turn_idx=10),
]

ACTION_CHECKS = [
    ActionCheck(action=EXPECTED_ACTIONS[0], action_match=True, action_reward=1.0, tool_type="read"),
    ActionCheck(action=EXPECTED_ACTIONS[1], action_match=True, action_reward=1.0, tool_type="read"),
    ActionCheck(action=EXPECTED_ACTIONS[2], action_match=False, action_reward=0.0, tool_type="write"),
]


def _make_sim() -> RawSimulation:
    return RawSimulation(
        id="sim-001",
        task_id="0",
        messages=MESSAGES,
        reward_info=RewardInfo(reward=0.0, action_checks=ACTION_CHECKS),
    )


def _make_task() -> RawTask:
    return RawTask(
        id="0",
        evaluation_criteria={"actions": [a.model_dump() for a in EXPECTED_ACTIONS]},
    )


def _make_sim_file() -> SimulationFile:
    return SimulationFile(tasks=[_make_task()], simulations=[_make_sim()])


# ---------------------------------------------------------------------------
# tool_registry_scanner
# ---------------------------------------------------------------------------

def test_registry_contains_all_expected_tools():
    sim_file = _make_sim_file()
    registry = scan_tool_registry(sim_file)
    assert "find_user_id_by_name_zip" in registry
    assert "get_order_details" in registry
    assert "exchange_delivered_order_items" in registry


def test_registry_required_args_from_expected():
    sim_file = _make_sim_file()
    registry = scan_tool_registry(sim_file)
    entry = registry["find_user_id_by_name_zip"]
    assert "first_name" in entry.required_args
    assert "zip" in entry.required_args


def test_registry_usage_count_from_actual_calls():
    sim_file = _make_sim_file()
    registry = scan_tool_registry(sim_file)
    assert registry["get_order_details"].usage_count == 1
    assert registry["find_user_id_by_name_zip"].usage_count == 1


def test_registry_seen_args_from_actual_calls():
    sim_file = _make_sim_file()
    registry = scan_tool_registry(sim_file)
    assert "order_id" in registry["get_order_details"].seen_args


def test_registry_tool_type_propagated():
    sim_file = _make_sim_file()
    registry = scan_tool_registry(sim_file)
    assert registry["exchange_delivered_order_items"].tool_type == "write"
    assert registry["get_order_details"].tool_type == "read"


# ---------------------------------------------------------------------------
# action_aligner
# ---------------------------------------------------------------------------

def test_align_actions_all_expected_present():
    sim = _make_sim()
    task = _make_task()
    seq = align_actions(sim, task)
    assert len(seq.aligned_actions) == len(EXPECTED_ACTIONS)


def test_align_actions_matched_flags():
    sim = _make_sim()
    task = _make_task()
    seq = align_actions(sim, task)
    assert seq.aligned_actions[0].action_match is True
    assert seq.aligned_actions[1].action_match is True
    assert seq.aligned_actions[2].action_match is False


def test_align_actions_actual_tool_populated():
    sim = _make_sim()
    task = _make_task()
    seq = align_actions(sim, task)
    assert seq.aligned_actions[0].actual_tool == "find_user_id_by_name_zip"
    assert seq.aligned_actions[1].actual_tool == "get_order_details"
    assert seq.aligned_actions[2].actual_tool == "exchange_delivered_order_items"


def test_align_actions_missing_actual_tool():
    sim = RawSimulation(
        id="sim-002",
        task_id="0",
        messages=[],
        reward_info=RewardInfo(reward=0.0, action_checks=[]),
    )
    task = _make_task()
    seq = align_actions(sim, task)
    for aa in seq.aligned_actions:
        assert aa.actual_tool is None
        assert aa.action_match is False


# ---------------------------------------------------------------------------
# turn_supervisor
# ---------------------------------------------------------------------------

def test_supervise_turns_row_count():
    sim = _make_sim()
    task = _make_task()
    rows = supervise_turns(sim, task)
    assert len(rows) == len(MESSAGES)


def test_supervise_user_turn_has_perception_message():
    sim = _make_sim()
    task = _make_task()
    rows = supervise_turns(sim, task)
    user_rows = [r for r in rows if r.role == "user"]
    assert all(r.cognitive_label.perception_message is not None for r in user_rows)


def test_supervise_tool_call_turn_has_plan_call_tool():
    sim = _make_sim()
    task = _make_task()
    rows = supervise_turns(sim, task)
    call_rows = [r for r in rows if r.role == "assistant" and r.cognitive_label.plan_next_action == "call_tool"]
    assert len(call_rows) == 3


def test_supervise_tool_result_turn_has_perception_result():
    sim = _make_sim()
    task = _make_task()
    rows = supervise_turns(sim, task)
    tool_rows = [r for r in rows if r.role == "tool"]
    assert all(r.cognitive_label.perception_tool_result is not None for r in tool_rows)


def test_supervise_entity_hints_extracted():
    sim = _make_sim()
    task = _make_task()
    rows = supervise_turns(sim, task)
    user_turn_1 = next(r for r in rows if r.role == "user" and r.turn_idx == 1)
    hints = user_turn_1.cognitive_label.perception_entity_hints
    assert hints.get("first_name") == "Yusuf" or hints.get("zip") == "19122"


def test_supervise_last_assistant_text_is_respond():
    sim = _make_sim()
    task = _make_task()
    rows = supervise_turns(sim, task)
    assistant_text_rows = [
        r for r in rows
        if r.role == "assistant" and r.cognitive_label.plan_next_action in ("respond", "ask_followup")
    ]
    assert len(assistant_text_rows) >= 1


# ---------------------------------------------------------------------------
# failure_extractor
# ---------------------------------------------------------------------------

def test_failure_extractor_count():
    sim = _make_sim()
    task = _make_task()
    seq = align_actions(sim, task)
    failures = extract_failures(sim, seq)
    assert len(failures) == 1


def test_failure_row_expected_tool():
    sim = _make_sim()
    task = _make_task()
    seq = align_actions(sim, task)
    failures = extract_failures(sim, seq)
    assert failures[0].expected_tool == "exchange_delivered_order_items"


def test_failure_row_actual_tool_populated():
    sim = _make_sim()
    task = _make_task()
    seq = align_actions(sim, task)
    failures = extract_failures(sim, seq)
    assert failures[0].actual_tool == "exchange_delivered_order_items"


def test_failure_row_argument_delta_has_mismatched():
    sim = _make_sim()
    task = _make_task()
    seq = align_actions(sim, task)
    failures = extract_failures(sim, seq)
    delta = failures[0].argument_delta
    assert "mismatched" in delta
    assert "new_item_ids" in delta["mismatched"]


def test_failure_row_no_failures_when_all_match():
    all_match_checks = [
        ActionCheck(action=EXPECTED_ACTIONS[0], action_match=True, action_reward=1.0),
        ActionCheck(action=EXPECTED_ACTIONS[1], action_match=True, action_reward=1.0),
        ActionCheck(action=EXPECTED_ACTIONS[2], action_match=True, action_reward=1.0),
    ]
    sim = RawSimulation(
        id="sim-003",
        task_id="0",
        messages=MESSAGES,
        reward_info=RewardInfo(reward=1.0, action_checks=all_match_checks),
    )
    task = _make_task()
    seq = align_actions(sim, task)
    failures = extract_failures(sim, seq)
    assert failures == []


# ---------------------------------------------------------------------------
# simulation_loader
# ---------------------------------------------------------------------------

def test_simulation_loader_roundtrip(tmp_path):
    sim_json = {
        "tasks": [
            {
                "id": "0",
                "evaluation_criteria": {
                    "actions": [
                        {"action_id": "0_0", "requestor": "assistant",
                         "name": "find_user_id_by_name_zip",
                         "arguments": {"first_name": "Yusuf", "last_name": "Rossi", "zip": "19122"}}
                    ]
                }
            }
        ],
        "simulations": [
            {
                "id": "abc",
                "task_id": "0",
                "messages": [
                    {"role": "user", "content": "Hello", "turn_idx": 0},
                    {"role": "assistant", "content": None, "turn_idx": 1,
                     "tool_calls": [{"id": "c1", "name": "find_user_id_by_name_zip",
                                     "arguments": {"first_name": "Yusuf", "last_name": "Rossi",
                                                   "zip": "19122"}}]},
                ],
                "reward_info": {
                    "reward": 1.0,
                    "action_checks": [
                        {"action": {"action_id": "0_0", "requestor": "assistant",
                                    "name": "find_user_id_by_name_zip",
                                    "arguments": {"first_name": "Yusuf", "last_name": "Rossi",
                                                  "zip": "19122"}},
                         "action_match": True, "action_reward": 1.0, "tool_type": "read"}
                    ]
                }
            }
        ]
    }
    p = tmp_path / "sim.json"
    p.write_text(json.dumps(sim_json))

    from cognitive_tool_agent.trace_converter.simulation_loader import load_simulation_file
    sf = load_simulation_file(p)
    assert len(sf.tasks) == 1
    assert len(sf.simulations) == 1
    assert sf.simulations[0].messages[1].is_tool_call


# ---------------------------------------------------------------------------
# TraceConverter end-to-end
# ---------------------------------------------------------------------------

def test_converter_produces_all_five_files(tmp_path):
    sim_json = {
        "tasks": [{"id": "0", "evaluation_criteria": {"actions": [
            {"action_id": "0_0", "requestor": "assistant", "name": "get_order_details",
             "arguments": {"order_id": "#W1"}}
        ]}}],
        "simulations": [{
            "id": "s1", "task_id": "0",
            "messages": [
                {"role": "user", "content": "Check order W1", "turn_idx": 0},
                {"role": "assistant", "content": None, "turn_idx": 1,
                 "tool_calls": [{"id": "c1", "name": "get_order_details",
                                 "arguments": {"order_id": "#W1"}}]},
                {"role": "tool", "content": '{"status":"pending"}', "id": "c1", "turn_idx": 2},
                {"role": "assistant", "content": "Your order is pending.", "turn_idx": 3},
            ],
            "reward_info": {
                "reward": 1.0,
                "action_checks": [
                    {"action": {"action_id": "0_0", "requestor": "assistant",
                                "name": "get_order_details", "arguments": {"order_id": "#W1"}},
                     "action_match": True, "action_reward": 1.0, "tool_type": "read"}
                ]
            }
        }]
    }
    input_file = tmp_path / "input.json"
    input_file.write_text(json.dumps(sim_json))
    out_dir = tmp_path / "out"

    converter = TraceConverter()
    summary = converter.run(input_file, out_dir)

    for fname in ["tool_registry.json", "action_sequence.jsonl",
                  "turn_supervision.jsonl", "failure_rows.jsonl", "conversion_summary.json"]:
        assert (out_dir / fname).exists(), f"Missing {fname}"

    assert summary.tasks_count == 1
    assert summary.simulations_count == 1
    assert summary.messages_count == 4
    assert summary.expected_actions_count == 1
    assert summary.actual_tool_calls_count == 1
    assert summary.matched_actions_count == 1
    assert summary.failed_actions_count == 0


def test_converter_summary_counts_failures(tmp_path):
    sim_json = {
        "tasks": [{"id": "0", "evaluation_criteria": {"actions": [
            {"action_id": "0_0", "requestor": "assistant", "name": "get_order_details",
             "arguments": {"order_id": "#W1"}}
        ]}}],
        "simulations": [{
            "id": "s2", "task_id": "0",
            "messages": [
                {"role": "user", "content": "Check order", "turn_idx": 0},
            ],
            "reward_info": {
                "reward": 0.0,
                "action_checks": [
                    {"action": {"action_id": "0_0", "requestor": "assistant",
                                "name": "get_order_details", "arguments": {"order_id": "#W1"}},
                     "action_match": False, "action_reward": 0.0, "tool_type": "read"}
                ]
            }
        }]
    }
    input_file = tmp_path / "input2.json"
    input_file.write_text(json.dumps(sim_json))
    out_dir = tmp_path / "out2"

    converter = TraceConverter()
    summary = converter.run(input_file, out_dir)

    assert summary.failed_actions_count == 1
    assert summary.matched_actions_count == 0

    failure_lines = (out_dir / "failure_rows.jsonl").read_text().strip().splitlines()
    assert len(failure_lines) == 1
    row = json.loads(failure_lines[0])
    assert row["expected_tool"] == "get_order_details"
