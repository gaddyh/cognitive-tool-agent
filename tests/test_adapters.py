"""Adapter delegation tests.

These tests are the red gate for the LLM phase.  Each test verifies that when
an agent is constructed with mode="llm" and a ModelAdapter, the agent's run()
method delegates to the adapter via adapter.complete().

All tests in this file FAIL today (NotImplementedError from _run_llm) and
become green once the corresponding _run_llm body is implemented.
"""
from __future__ import annotations

import pytest

from cognitive_tool_agent.adapters.base import AgentMode, LLMResult, ModelAdapter
from cognitive_tool_agent.graph.node_input import NodeInput
from cognitive_tool_agent.schemas.common import ToolSchema, UserInput
from cognitive_tool_agent.schemas.dataset import DatasetRow, ExpectedBehavior
from cognitive_tool_agent.tools.fake_tools import DEFAULT_REGISTRY


# ── Minimal concrete ModelAdapter stub ───────────────────────────────────────


class _AdapterStub:
    """Concrete ModelAdapter that records calls and returns a pre-set LLMResult."""

    def __init__(self, result: LLMResult) -> None:
        self._result = result
        self.calls: list[tuple[str, type]] = []

    def complete(self, prompt: str, output_schema: type) -> LLMResult:
        self.calls.append((prompt, output_schema))
        return self._result

    @property
    def called(self) -> bool:
        return len(self.calls) > 0


assert isinstance(_AdapterStub(LLMResult(parsed=None)), ModelAdapter), (
    "_AdapterStub must satisfy the ModelAdapter Protocol"
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def simple_user_input() -> UserInput:
    return UserInput(
        message="Where is my order #12345?",
        available_tools=[
            ToolSchema(
                name="get_order_status",
                description="Get order status",
                required_fields=["order_id"],
            )
        ],
    )


@pytest.fixture()
def simple_node_input(simple_user_input: UserInput) -> NodeInput:
    row = DatasetRow(
        id="test-001",
        user_message=simple_user_input.message,
        expected=ExpectedBehavior(expected_action="execute_tool"),
    )
    return NodeInput(
        user_input=simple_user_input,
        registry=DEFAULT_REGISTRY,
        row=row,
    )


@pytest.fixture()
def stub_adapter() -> _AdapterStub:
    from cognitive_tool_agent.schemas.perceive import PerceptionResult
    from cognitive_tool_agent.schemas.common import Confidence

    result = LLMResult(
        parsed=PerceptionResult(
            intent_candidates=["use_get_order_status"],
            ambiguity_detected=False,
            candidate_tools=["get_order_status"],
            confidence=Confidence(score=0.9, reason="llm"),
        )
    )
    return _AdapterStub(result)


# ── PerceiveAgent delegation ───────────────────────────────────────────────────


@pytest.mark.xfail(strict=True, reason="llm mode not yet implemented for PerceiveAgent — becomes green when _run_llm is implemented")
def test_perceive_agent_llm_mode_calls_adapter(simple_node_input, stub_adapter):
    """PerceiveAgent(mode='llm') must call adapter.complete() exactly once.

    EXPECTED STATUS: FAILS today (NotImplementedError from _run_llm stub).
    Becomes green once _run_llm is implemented for PerceiveAgent.
    """
    from cognitive_tool_agent.agents.perceive_agent import PerceiveAgent

    agent = PerceiveAgent(mode="llm", model_adapter=stub_adapter)
    agent.run(simple_node_input)
    assert stub_adapter.called, "adapter.complete() was never called"
    assert len(stub_adapter.calls) == 1


def test_perceive_agent_stub_mode_does_not_call_adapter(simple_node_input, stub_adapter):
    """Stub mode must not touch the adapter even if one is provided."""
    from cognitive_tool_agent.agents.perceive_agent import PerceiveAgent

    agent = PerceiveAgent(mode="stub", model_adapter=stub_adapter)
    agent.run(simple_node_input)
    assert not stub_adapter.called


def test_adapter_stub_satisfies_protocol():
    """_AdapterStub must satisfy the ModelAdapter runtime-checkable Protocol."""
    stub = _AdapterStub(LLMResult(parsed=None))
    assert isinstance(stub, ModelAdapter)
