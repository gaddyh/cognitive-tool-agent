"""Unit tests for scenario_profiler.profile_simulation()."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from cognitive_tool_agent.schemas.simulation import (
    ExpectedAction,
    RawSimulation,
    RawTask,
    RewardInfo,
)
from cognitive_tool_agent.trace_converter.scenario_profiler import profile_simulation


def _make_sim(task_id: str = "0") -> RawSimulation:
    return RawSimulation(id="sim-001", task_id=task_id, messages=[], reward_info=RewardInfo())


def _make_task(actions: list[dict]) -> RawTask:
    return RawTask(id="0", evaluation_criteria={"actions": actions})


def _action(name: str, args: dict | None = None) -> dict:
    return {"action_id": "x", "requestor": "assistant", "name": name, "arguments": args or {}}


class TestLookupOnly:
    def test_no_terminal_actions(self):
        task = _make_task([
            _action("find_user_id_by_name_zip"),
            _action("get_user_details"),
        ])
        p = profile_simulation(_make_sim(), task, num_tool_calls=3)
        assert p.primary_scenario == "lookup_only"
        assert p.scenario_type == "lookup_only|single_action"
        assert p.terminal_tool_fingerprint == "none"
        assert p.is_multi_action is False

    def test_empty_actions(self):
        task = _make_task([])
        p = profile_simulation(_make_sim(), task, num_tool_calls=0)
        assert p.primary_scenario == "lookup_only"
        assert p.terminal_tool_fingerprint == "none"


class TestSingleTerminal:
    def test_exchange_single(self):
        task = _make_task([
            _action("find_user_id_by_name_zip"),
            _action("get_order_details", {"order_id": "#W1"}),
            _action("exchange_delivered_order_items", {
                "order_id": "#W1", "item_ids": ["123"], "new_item_ids": ["456"],
                "payment_method_id": "pm_1",
            }),
        ])
        p = profile_simulation(_make_sim(), task, num_tool_calls=5)
        assert p.primary_scenario == "exchange"
        assert p.scenario_type == "exchange|single_action"
        assert p.terminal_tool_fingerprint == "exchange_delivered_order_items"
        assert p.is_multi_action is False
        assert p.has_item_ids is True
        assert p.has_order_id is True
        assert p.difficulty_bucket == "hard"
        assert p.requires_grounding is True

    def test_cancel_single(self):
        task = _make_task([
            _action("find_user_id_by_email"),
            _action("get_order_details", {"order_id": "#W2"}),
            _action("cancel_pending_order", {"order_id": "#W2", "reason": "no_longer_needed"}),
        ])
        p = profile_simulation(_make_sim(), task, num_tool_calls=4)
        assert p.primary_scenario == "cancel"
        assert p.scenario_type == "cancel|single_action"
        assert p.difficulty_bucket == "medium"
        assert p.has_item_ids is False
        assert p.has_order_id is True

    def test_return_no_product_id(self):
        task = _make_task([
            _action("find_user_id_by_name_zip"),
            _action("get_order_details", {"order_id": "#W3"}),
            _action("return_delivered_order_items", {
                "order_id": "#W3", "item_ids": ["789"], "payment_method_id": "pm_2",
            }),
        ])
        p = profile_simulation(_make_sim(), task, num_tool_calls=4)
        assert p.primary_scenario == "return"
        assert p.has_product_id is False
        assert p.difficulty_bucket == "hard"


class TestMultiAction:
    def test_two_different_terminal_families(self):
        task = _make_task([
            _action("find_user_id_by_email"),
            _action("get_order_details", {"order_id": "#W1"}),
            _action("cancel_pending_order", {"order_id": "#W1"}),
            _action("return_delivered_order_items", {
                "order_id": "#W2", "item_ids": ["a"], "payment_method_id": "pm_1",
            }),
        ])
        p = profile_simulation(_make_sim(), task, num_tool_calls=6)
        assert p.is_multi_action is True
        assert p.scenario_type.endswith("|multi_action")
        assert p.primary_scenario == "return"

    def test_same_family_twice_is_not_multi(self):
        task = _make_task([
            _action("find_user_id_by_name_zip"),
            _action("cancel_pending_order", {"order_id": "#W1"}),
            _action("cancel_pending_order", {"order_id": "#W2"}),
        ])
        p = profile_simulation(_make_sim(), task, num_tool_calls=4)
        assert p.is_multi_action is False
        assert p.scenario_type == "cancel|single_action"
        assert p.terminal_tool_fingerprint == "cancel_pending_order"

    def test_modify_items_and_address_is_multi(self):
        task = _make_task([
            _action("find_user_id_by_name_zip"),
            _action("get_order_details", {"order_id": "#W1"}),
            _action("modify_pending_order_items", {"order_id": "#W1", "item_ids": ["x"]}),
            _action("modify_pending_order_address", {"order_id": "#W1"}),
        ])
        p = profile_simulation(_make_sim(), task, num_tool_calls=5)
        assert p.is_multi_action is True
        assert p.terminal_tool_fingerprint == "modify_pending_order_address+modify_pending_order_items"


class TestGradingFlags:
    def test_product_id_medium_difficulty(self):
        task = _make_task([
            _action("get_product_details", {"product_id": "prod-1"}),
        ])
        p = profile_simulation(_make_sim(), task, num_tool_calls=2)
        assert p.has_product_id is True
        assert p.difficulty_bucket == "medium"
        assert p.requires_grounding is True

    def test_easy_difficulty_no_grounding_args(self):
        task = _make_task([
            _action("find_user_id_by_name_zip", {"first_name": "A", "last_name": "B", "zip": "12345"}),
            _action("transfer_to_human_agents", {}),
        ])
        p = profile_simulation(_make_sim(), task, num_tool_calls=3)
        assert p.has_item_ids is False
        assert p.has_order_id is False
        assert p.has_product_id is False
        assert p.requires_grounding is False
        assert p.difficulty_bucket == "easy"


class TestRequiresToolChaining:
    def test_chaining_when_3_or_more_expected(self):
        task = _make_task([_action("get_user_details")] * 3)
        p = profile_simulation(_make_sim(), task, num_tool_calls=3)
        assert p.requires_tool_chaining is True

    def test_no_chaining_with_2(self):
        task = _make_task([_action("get_user_details")] * 2)
        p = profile_simulation(_make_sim(), task, num_tool_calls=2)
        assert p.requires_tool_chaining is False


class TestSplitIsNone:
    def test_split_defaults_to_none(self):
        task = _make_task([_action("cancel_pending_order", {"order_id": "#W1"})])
        p = profile_simulation(_make_sim(), task, num_tool_calls=2)
        assert p.split is None
