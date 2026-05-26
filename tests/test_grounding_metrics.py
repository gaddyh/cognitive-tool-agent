"""Tests for grounding_metrics module.

No LLM calls. Covers:
  - collect_values_from_available_state (recursive extraction)
  - compute_grounding_metrics (deterministic and llm_v1 variants)
  - PredictionRow schema
"""
from __future__ import annotations

import pytest

from cognitive_tool_agent.evaluation.grounding_metrics import (
    GroundingMetrics,
    PredictionRow,
    collect_values_from_available_state,
    compute_grounding_metrics,
)


# ─── collect_values_from_available_state ───────────────────────────────────────

def test_collect_flat_strings():
    state = {"prior_tool_calls": [{"tool_name": "get_order", "arguments": {"order_id": "#W123"}}]}
    values = collect_values_from_available_state(state)
    assert "#W123" in values
    assert "get_order" in values


def test_collect_nested_tool_results():
    import json
    result_content = json.dumps({"order_id": "#W999", "status": "pending"})
    state = {"prior_tool_results": [{"content": result_content}]}
    values = collect_values_from_available_state(state)
    assert result_content in values


def test_collect_conversation_context():
    state = {"conversation_context": ["Hello", "I want to cancel order #W777"]}
    values = collect_values_from_available_state(state)
    assert "Hello" in values
    assert "I want to cancel order #W777" in values


def test_collect_empty_state():
    assert collect_values_from_available_state({}) == set()


def test_collect_deeply_nested():
    state = {"a": {"b": {"c": "deep_value_123"}}}
    values = collect_values_from_available_state(state)
    assert "deep_value_123" in values


def test_collect_list_of_strings():
    state = {"items": ["id_A", "id_B", "id_C"]}
    values = collect_values_from_available_state(state)
    assert {"id_A", "id_B", "id_C"}.issubset(values)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_pred(
    id: str = "row-1",
    target_args: dict | None = None,
    target_fields: list | None = None,
    det_resolved: dict | None = None,
    llm_resolved: dict | None = None,
    llm_schema_valid: bool | None = None,
    llm_confidence: float | None = None,
    llm_latency_ms: float | None = None,
    available_state: dict | None = None,
) -> PredictionRow:
    return PredictionRow(
        id=id,
        split="dev",
        target_args=target_args or {},
        target_fields=target_fields or [],
        available_state=available_state or {},
        deterministic_resolved=det_resolved or {},
        llm_resolved=llm_resolved,
        llm_schema_valid=llm_schema_valid,
        llm_confidence=llm_confidence,
        llm_latency_ms=llm_latency_ms,
    )


# ─── compute_grounding_metrics — deterministic ─────────────────────────────────

def test_metrics_empty_predictions():
    m = compute_grounding_metrics([], variant="deterministic", split="dev")
    assert m.total_rows == 0
    assert m.required_arg_match == 0.0


def test_metrics_det_perfect_match():
    preds = [
        _make_pred(
            target_args={"order_id": "#W1"},
            target_fields=["order_id"],
            det_resolved={"order_id": "#W1"},
        )
    ]
    m = compute_grounding_metrics(preds, variant="deterministic", split="dev")
    assert m.required_arg_match == pytest.approx(1.0)
    assert m.field_recall == pytest.approx(1.0)
    assert m.field_precision == pytest.approx(1.0)
    assert m.field_f1 == pytest.approx(1.0)


def test_metrics_det_no_match():
    preds = [
        _make_pred(
            target_args={"order_id": "#W1"},
            target_fields=["order_id"],
            det_resolved={},
        )
    ]
    m = compute_grounding_metrics(preds, variant="deterministic", split="dev")
    assert m.required_arg_match == pytest.approx(0.0)
    assert m.missing_field_rate == pytest.approx(1.0)


def test_metrics_det_partial_match():
    preds = [
        _make_pred(
            target_args={"order_id": "#W1", "item_ids": ["ITEM-9"]},
            target_fields=["order_id", "item_ids"],
            det_resolved={"order_id": "#W1"},
        )
    ]
    m = compute_grounding_metrics(preds, variant="deterministic", split="dev")
    assert m.required_arg_match == pytest.approx(0.0)
    assert m.field_recall == pytest.approx(0.5)


def test_metrics_det_schema_valid_always_true():
    preds = [_make_pred(det_resolved={"order_id": "#W1"})]
    m = compute_grounding_metrics(preds, variant="deterministic", split="dev")
    assert m.schema_valid_rate == pytest.approx(1.0)


def test_metrics_det_multiple_rows():
    preds = [
        _make_pred(
            id="r1",
            target_args={"order_id": "#W1"},
            target_fields=["order_id"],
            det_resolved={"order_id": "#W1"},
        ),
        _make_pred(
            id="r2",
            target_args={"order_id": "#W2"},
            target_fields=["order_id"],
            det_resolved={"order_id": "#W99"},
        ),
    ]
    m = compute_grounding_metrics(preds, variant="deterministic", split="dev")
    assert m.required_arg_match == pytest.approx(0.5)
    assert m.total_rows == 2


# ─── compute_grounding_metrics — grounding_llm_v1 ─────────────────────────────

def test_metrics_llm_schema_invalid_row():
    preds = [
        _make_pred(
            target_args={"order_id": "#W1"},
            target_fields=["order_id"],
            llm_resolved={},
            llm_schema_valid=False,
        )
    ]
    m = compute_grounding_metrics(preds, variant="grounding_llm_v1", split="dev")
    assert m.schema_valid_rate == pytest.approx(0.0)
    assert m.required_arg_match == pytest.approx(0.0)


def test_metrics_llm_perfect_match():
    preds = [
        _make_pred(
            target_args={"order_id": "#W5"},
            target_fields=["order_id"],
            llm_resolved={"order_id": "#W5"},
            llm_schema_valid=True,
            llm_confidence=0.9,
            llm_latency_ms=450.0,
        )
    ]
    m = compute_grounding_metrics(preds, variant="grounding_llm_v1", split="dev")
    assert m.required_arg_match == pytest.approx(1.0)
    assert m.schema_valid_rate == pytest.approx(1.0)
    assert m.avg_confidence == pytest.approx(0.9)
    assert m.avg_latency_ms == pytest.approx(450.0)


def test_metrics_hallucinated_id_detected():
    preds = [
        _make_pred(
            target_args={"order_id": "#W5"},
            target_fields=["order_id"],
            llm_resolved={"order_id": "#WNOT"},
            llm_schema_valid=True,
            available_state={
                "prior_tool_calls": [{"arguments": {"order_id": "#W5"}}],
            },
        )
    ]
    m = compute_grounding_metrics(preds, variant="grounding_llm_v1", split="dev")
    assert m.hallucinated_id_rate > 0.0


def test_metrics_no_hallucination_when_value_in_state():
    preds = [
        _make_pred(
            target_args={"order_id": "#W5"},
            target_fields=["order_id"],
            llm_resolved={"order_id": "#W99"},
            llm_schema_valid=True,
            available_state={
                "prior_tool_calls": [{"arguments": {"order_id": "#W99"}}],
            },
        )
    ]
    m = compute_grounding_metrics(preds, variant="grounding_llm_v1", split="dev")
    assert m.hallucinated_id_rate == pytest.approx(0.0)


def test_metrics_model_output_is_grounding_metrics():
    preds = [_make_pred(det_resolved={})]
    m = compute_grounding_metrics(preds, variant="deterministic", split="train")
    assert isinstance(m, GroundingMetrics)
    assert m.split == "train"
    assert m.variant == "deterministic"
