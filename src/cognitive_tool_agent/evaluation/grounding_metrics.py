"""Grounding evaluation metrics for Pass 2 (offline grounding evaluation).

Primary metric: required_arg_match
    Exact match of all required-field resolved_args vs target_args.

Field-level metrics (over individual arg fields):
    field_precision, field_recall, field_f1

Diagnostic metrics:
    missing_field_rate, hallucinated_id_rate, schema_valid_rate,
    avg_confidence, avg_latency_ms, estimated_cost_usd
"""
from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field


_ID_LIKE = re.compile(r"(?:^#?[A-Z0-9\-_]{4,}$|^\d{5,}$)", re.IGNORECASE)


def collect_values_from_available_state(available_state: dict[str, Any]) -> set[str]:
    """Recursively collect all string leaf values from available_state.

    Searches prior_tool_calls, prior_tool_results, conversation_context.
    Used by hallucinated_id_rate: a predicted value that does not appear here
    and does not match target is considered a hallucination candidate.
    """
    found: set[str] = set()
    _collect(available_state, found)
    return found


def _collect(obj: Any, found: set[str]) -> None:
    if isinstance(obj, str):
        v = obj.strip()
        if v:
            found.add(v)
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect(v, found)
    elif isinstance(obj, list):
        for item in obj:
            _collect(item, found)


def _is_id_like(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return bool(_ID_LIKE.match(value.strip()))


class GroundingMetrics(BaseModel):
    variant: str
    split: str
    total_rows: int

    required_arg_match: float = Field(ge=0.0, le=1.0)

    field_precision: float = Field(ge=0.0, le=1.0)
    field_recall: float = Field(ge=0.0, le=1.0)
    field_f1: float = Field(ge=0.0, le=1.0)

    missing_field_rate: float = Field(ge=0.0, le=1.0)
    hallucinated_id_rate: float = Field(ge=0.0, le=1.0)
    schema_valid_rate: float = Field(ge=0.0, le=1.0)

    avg_confidence: float | None = None
    avg_latency_ms: float | None = None
    estimated_cost_usd: float | None = None


class PredictionRow(BaseModel):
    """One row in predictions_{split}.jsonl — both variant results + raw LLM output."""
    id: str
    split: str
    target_args: dict[str, Any]
    target_fields: list[str]
    available_state: dict[str, Any]

    deterministic_resolved: dict[str, Any]
    deterministic_schema_valid: bool = True
    deterministic_latency_ms: float | None = None

    llm_resolved: dict[str, Any] | None = None
    llm_raw: dict[str, Any] | None = None
    llm_schema_valid: bool | None = None
    llm_error: str | None = None
    llm_confidence: float | None = None
    llm_latency_ms: float | None = None
    llm_cost_usd: float | None = None


def compute_grounding_metrics(
    predictions: list[PredictionRow],
    variant: str,
    split: str,
) -> GroundingMetrics:
    """Compute grounding metrics for a single variant over a list of prediction rows."""
    total = len(predictions)
    if total == 0:
        return GroundingMetrics(
            variant=variant,
            split=split,
            total_rows=0,
            required_arg_match=0.0,
            field_precision=0.0,
            field_recall=0.0,
            field_f1=0.0,
            missing_field_rate=0.0,
            hallucinated_id_rate=0.0,
            schema_valid_rate=0.0,
        )

    exact_matches = 0
    schema_valid_count = 0
    total_precision_num = 0.0
    total_precision_den = 0
    total_recall_num = 0.0
    total_recall_den = 0
    missing_field_count = 0
    missing_field_den = 0
    hallucinated_count = 0
    hallucinated_den = 0
    confidence_sum = 0.0
    confidence_count = 0
    latency_sum = 0.0
    latency_count = 0
    cost_sum = 0.0
    cost_count = 0

    for row in predictions:
        resolved = _get_resolved(row, variant)
        schema_valid = _get_schema_valid(row, variant)
        target = row.target_args
        target_fields = set(row.target_fields)
        state_values = collect_values_from_available_state(row.available_state)

        if schema_valid:
            schema_valid_count += 1

        if resolved is None:
            resolved = {}

        exact_matches += int(resolved == target)

        true_positives = sum(
            1 for f in target_fields
            if f in resolved and resolved[f] == target.get(f)
        )
        predicted_set = set(resolved.keys())
        precision_den = len(predicted_set & target_fields) if target_fields else 0
        recall_den = len(target_fields)

        if precision_den > 0:
            total_precision_num += true_positives / precision_den
            total_precision_den += 1
        if recall_den > 0:
            total_recall_num += true_positives / recall_den
            total_recall_den += 1

        for f in target_fields:
            missing_field_den += 1
            if f not in resolved:
                missing_field_count += 1

        for f, val in resolved.items():
            if f in target_fields:
                target_val = target.get(f)
                if _is_id_like(val) and val != target_val:
                    if not _in_state(val, state_values):
                        hallucinated_count += 1
                hallucinated_den += 1

        if variant == "deterministic":
            lat = row.deterministic_latency_ms
            if lat is not None:
                latency_sum += lat
                latency_count += 1
        elif variant == "grounding_llm_v1":
            conf = row.llm_confidence
            lat = row.llm_latency_ms
            cost = row.llm_cost_usd
            if conf is not None:
                confidence_sum += conf
                confidence_count += 1
            if lat is not None:
                latency_sum += lat
                latency_count += 1
            if cost is not None:
                cost_sum += cost
                cost_count += 1

    precision = total_precision_num / total_precision_den if total_precision_den else 0.0
    recall = total_recall_num / total_recall_den if total_recall_den else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return GroundingMetrics(
        variant=variant,
        split=split,
        total_rows=total,
        required_arg_match=round(exact_matches / total, 4),
        field_precision=round(precision, 4),
        field_recall=round(recall, 4),
        field_f1=round(f1, 4),
        missing_field_rate=round(missing_field_count / missing_field_den, 4) if missing_field_den else 0.0,
        hallucinated_id_rate=round(hallucinated_count / hallucinated_den, 4) if hallucinated_den else 0.0,
        schema_valid_rate=round(schema_valid_count / total, 4),
        avg_confidence=round(confidence_sum / confidence_count, 4) if confidence_count else None,
        avg_latency_ms=round(latency_sum / latency_count, 2) if latency_count else None,
        estimated_cost_usd=round(cost_sum, 6) if cost_count else None,
    )


def _get_resolved(row: PredictionRow, variant: str) -> dict[str, Any] | None:
    if variant == "deterministic":
        return row.deterministic_resolved
    if variant == "grounding_llm_v1":
        return row.llm_resolved
    return {}


def _get_schema_valid(row: PredictionRow, variant: str) -> bool:
    if variant == "deterministic":
        return row.deterministic_schema_valid
    if variant == "grounding_llm_v1":
        return row.llm_schema_valid if row.llm_schema_valid is not None else False
    return True


def _in_state(val: str, state_values: set[str]) -> bool:
    return val in state_values
