"""Tests for GroundingAgent deterministic mode (Pass 1: scalar IDs).

All tests are label-clean: no expected_arguments or expected_tool is referenced
inside the code under test or in test setup.
"""
from __future__ import annotations

import json

import pytest

from cognitive_tool_agent.agents.grounding_agent import GroundingAgent
from cognitive_tool_agent.graph.node_input import NodeInput
from cognitive_tool_agent.schemas.common import ToolSchema, UserInput
from cognitive_tool_agent.schemas.dataset import DatasetRow, ExpectedBehavior
from cognitive_tool_agent.tools.fake_tools import DEFAULT_REGISTRY


def _make_ctx(
    message: str,
    tool_name: str,
    required_fields: list[str],
    prior_tool_calls: list[dict] | None = None,
    prior_tool_results: list[dict] | None = None,
) -> NodeInput:
    tool_schema = ToolSchema(
        name=tool_name,
        description=f"Tool: {tool_name}",
        required_fields=required_fields,
    )
    user_input = UserInput(message=message, available_tools=[tool_schema])
    row = DatasetRow(
        id="test-det-001",
        user_message=message,
        world_state={
            "primary_tool": tool_name,
            "prior_tool_calls": prior_tool_calls or [],
            "prior_tool_results": prior_tool_results or [],
        },
        expected=ExpectedBehavior(expected_action="tool_executed"),
    )
    return NodeInput(user_input=user_input, registry=DEFAULT_REGISTRY, row=row)


def test_deterministic_mode_dispatched_not_stub():
    ctx = _make_ctx("cancel order #W999", "cancel_pending_order", ["order_id"])
    agent = GroundingAgent(mode="deterministic")
    result = agent.run(ctx)
    assert result.grounding_mode == "deterministic", (
        "mode='deterministic' must not fall through to stub"
    )


def test_deterministic_resolves_order_id_from_prior_tool_call():
    ctx = _make_ctx(
        message="Yes, please cancel it.",
        tool_name="cancel_pending_order",
        required_fields=["order_id"],
        prior_tool_calls=[
            {"tool_name": "get_order_details", "arguments": {"order_id": "#W1234"}},
        ],
    )
    result = GroundingAgent(mode="deterministic").run(ctx)
    assert result.resolved_args.get("order_id") == "#W1234"
    assert result.grounding_mode == "deterministic"


def test_deterministic_resolves_order_id_from_prior_tool_result_json():
    order_json = json.dumps({"order_id": "#W5678", "status": "pending"})
    ctx = _make_ctx(
        message="Cancel it please.",
        tool_name="cancel_pending_order",
        required_fields=["order_id"],
        prior_tool_results=[{"content": order_json}],
    )
    result = GroundingAgent(mode="deterministic").run(ctx)
    assert result.resolved_args.get("order_id") == "#W5678"


def test_deterministic_resolves_order_id_from_user_message_regex():
    ctx = _make_ctx(
        message="I want to cancel order #W9999.",
        tool_name="cancel_pending_order",
        required_fields=["order_id"],
    )
    result = GroundingAgent(mode="deterministic").run(ctx)
    assert result.resolved_args.get("order_id") == "#W9999"


def test_deterministic_resolves_user_id_from_prior_tool_call():
    ctx = _make_ctx(
        message="Get my details.",
        tool_name="get_user_details",
        required_fields=["user_id"],
        prior_tool_calls=[
            {"tool_name": "find_user_id_by_email", "arguments": {"email": "a@b.com"}},
            {"tool_name": "get_user_details", "arguments": {"user_id": "john_doe_1234"}},
        ],
    )
    result = GroundingAgent(mode="deterministic").run(ctx)
    assert result.resolved_args.get("user_id") == "john_doe_1234"


def test_deterministic_resolves_product_id_from_prior_tool_result():
    product_json = json.dumps(
        {"product_id": "prod_987", "name": "Wireless Headphones", "price": 49.99}
    )
    ctx = _make_ctx(
        message="Tell me about it.",
        tool_name="get_product_details",
        required_fields=["product_id"],
        prior_tool_results=[{"content": product_json}],
    )
    result = GroundingAgent(mode="deterministic").run(ctx)
    assert result.resolved_args.get("product_id") == "prod_987"


def test_deterministic_no_context_leaves_field_unresolved():
    ctx = _make_ctx(
        message="Do the thing.",
        tool_name="cancel_pending_order",
        required_fields=["order_id"],
    )
    result = GroundingAgent(mode="deterministic").run(ctx)
    assert "order_id" not in result.resolved_args
    assert "order_id" in result.unresolved_ids
