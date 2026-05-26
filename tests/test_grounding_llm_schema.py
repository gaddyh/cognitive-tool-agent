"""Tests for LLMGroundingOutput schema and to_grounding_result() projection.

No LLM calls are made. All tests are purely schema/logic level.
"""
from __future__ import annotations

import pytest

from cognitive_tool_agent.schemas.grounding_llm import GroundedField, LLMGroundingOutput


def _make_output(**fields) -> LLMGroundingOutput:
    return LLMGroundingOutput(fields=fields, confidence=0.8)


def test_grounded_field_valid():
    f = GroundedField(value="ORDER-123", status="grounded", evidence=["Found in prior call"], confidence=0.9)
    assert f.value == "ORDER-123"
    assert f.status == "grounded"


def test_grounded_field_missing():
    f = GroundedField(value=None, status="missing", confidence=0.0)
    assert f.value is None
    assert f.status == "missing"


def test_grounded_field_ambiguous():
    f = GroundedField(value=None, status="ambiguous", evidence=["Two matching orders"], confidence=0.4)
    assert f.status == "ambiguous"


def test_grounded_field_not_applicable():
    f = GroundedField(value=None, status="not_applicable", confidence=1.0)
    assert f.status == "not_applicable"


def test_grounded_field_confidence_bounds():
    with pytest.raises(Exception):
        GroundedField(value=None, status="missing", confidence=1.5)
    with pytest.raises(Exception):
        GroundedField(value=None, status="missing", confidence=-0.1)


def test_to_grounding_result_grounded_fields_become_resolved_args():
    output = _make_output(
        order_id=GroundedField(value="#W1234", status="grounded", confidence=0.9),
        user_id=GroundedField(value="john_doe_1", status="grounded", confidence=0.95),
    )
    result = output.to_grounding_result()
    assert result.resolved_args == {"order_id": "#W1234", "user_id": "john_doe_1"}
    assert result.unresolved_ids == []
    assert result.grounding_mode == "llm"


def test_to_grounding_result_missing_fields_become_unresolved():
    output = _make_output(
        order_id=GroundedField(value=None, status="missing", confidence=0.0),
        item_ids=GroundedField(value=None, status="ambiguous", confidence=0.3),
    )
    result = output.to_grounding_result()
    assert result.resolved_args == {}
    assert sorted(result.unresolved_ids) == ["item_ids", "order_id"]


def test_to_grounding_result_not_applicable_excluded():
    output = _make_output(
        order_id=GroundedField(value="#W9", status="grounded", confidence=0.8),
        new_item_ids=GroundedField(value=None, status="not_applicable", confidence=1.0),
    )
    result = output.to_grounding_result()
    assert result.resolved_args == {"order_id": "#W9"}
    assert "new_item_ids" not in result.unresolved_ids


def test_to_grounding_result_grounded_with_none_value_excluded():
    output = _make_output(
        order_id=GroundedField(value=None, status="grounded", confidence=0.5),
    )
    result = output.to_grounding_result()
    assert "order_id" not in result.resolved_args


def test_to_grounding_result_confidence_propagated():
    output = LLMGroundingOutput(
        fields={"order_id": GroundedField(value="#W1", status="grounded", confidence=0.9)},
        confidence=0.77,
    )
    result = output.to_grounding_result()
    assert result.grounding_confidence == pytest.approx(0.77)


def test_to_grounding_result_unresolved_deduped():
    output = LLMGroundingOutput(
        fields={"order_id": GroundedField(value=None, status="missing", confidence=0.0)},
        unresolved_ids=["order_id", "item_ids"],
        confidence=0.1,
    )
    result = output.to_grounding_result()
    assert result.unresolved_ids.count("order_id") == 1


def test_llm_grounding_output_clarification_fields():
    output = LLMGroundingOutput(
        fields={},
        clarification_needed=True,
        clarification_question="Which order do you mean?",
        confidence=0.2,
    )
    assert output.clarification_needed is True
    assert "Which order" in output.clarification_question


def test_llm_grounding_output_round_trip_json():
    output = _make_output(
        order_id=GroundedField(value="#W42", status="grounded", evidence=["From prior call"], confidence=0.88),
    )
    serialized = output.model_dump_json()
    restored = LLMGroundingOutput.model_validate_json(serialized)
    assert restored.fields["order_id"].value == "#W42"
    assert restored.confidence == pytest.approx(0.8)
