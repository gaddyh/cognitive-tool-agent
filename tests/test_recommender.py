from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from cognitive_tool_agent.recommender.signal_extractor import extract_signals
from cognitive_tool_agent.recommender.capability_inference import CapabilityInferenceEngine
from cognitive_tool_agent.recommender.graph_recommender import GraphRecommender
from cognitive_tool_agent.schemas.recommender import CapabilityInferenceResult, CapabilityRequirement


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _empty_report() -> dict:
    return {
        "dataset_summary": {},
        "cognitive_action_topology": [],
        "argument_emergence": [],
        "failure_heatmap": [],
    }


def _high_signal_report() -> dict:
    """Report shaped to trigger all four capabilities."""
    return {
        "dataset_summary": {
            "read_tool_calls": 10,
            "write_tool_calls": 40,
        },
        "cognitive_action_topology": [
            {"tool_name": "update_order", "avg_chain_depth": 4.0},
            {"tool_name": "cancel_order", "avg_chain_depth": 5.0},
        ],
        "argument_emergence": [
            {
                "arg_name": "order_id",
                "total_instances": 20,
                "requires_tool_chaining_pct": 90.0,
                "requires_grounding_pct": 5.0,
            },
            {
                "arg_name": "item_ids",
                "total_instances": 15,
                "requires_tool_chaining_pct": 10.0,
                "requires_grounding_pct": 92.0,
            },
        ],
        "failure_heatmap": [
            {"dimension": "read_write", "value": "write", "count": 30},
            {"dimension": "read_write", "value": "read", "count": 5},
        ],
    }


def _low_signal_report() -> dict:
    """Report shaped so no capabilities are required."""
    return {
        "dataset_summary": {
            "read_tool_calls": 90,
            "write_tool_calls": 5,
        },
        "cognitive_action_topology": [
            {"tool_name": "get_order", "avg_chain_depth": 0.5},
        ],
        "argument_emergence": [
            {
                "arg_name": "order_id",
                "total_instances": 10,
                "requires_tool_chaining_pct": 10.0,
                "requires_grounding_pct": 5.0,
            },
        ],
        "failure_heatmap": [
            {"dimension": "read_write", "value": "write", "count": 1},
            {"dimension": "read_write", "value": "read", "count": 20},
        ],
    }


# ---------------------------------------------------------------------------
# signal_extractor tests
# ---------------------------------------------------------------------------

def test_signal_extractor_empty():
    signals, sources = extract_signals(_empty_report())
    assert signals["chaining_strength"] == 0.0
    assert signals["grounding_strength"] == 0.0
    assert signals["peak_grounding_strength"] == 0.0
    assert signals["peak_grounding_instances"] == 0.0
    assert signals["write_fraction"] == 0.0
    assert signals["avg_chain_depth"] == 0.0
    assert signals["write_failure_fraction"] == 0.0
    assert sources["peak_grounding_arg"] == ""


def test_signal_extractor_known():
    signals, sources = extract_signals(_high_signal_report())
    # write_fraction = 40 / (10+40) = 0.8
    assert signals["write_fraction"] == pytest.approx(0.8, abs=1e-3)
    # write_failure_fraction = 30 / 35 ≈ 0.857
    assert signals["write_failure_fraction"] == pytest.approx(30 / 35, abs=1e-3)
    # avg_chain_depth = (4.0 + 5.0) / 2 = 4.5
    assert signals["avg_chain_depth"] == pytest.approx(4.5, abs=1e-3)
    # chaining: (90*20 + 10*15) / (100 * 35) = (1800+150)/3500 ≈ 0.557
    assert signals["chaining_strength"] == pytest.approx((90 * 20 + 10 * 15) / (100 * 35), abs=1e-3)
    # grounding: (5*20 + 92*15) / (100 * 35) = (100+1380)/3500 ≈ 0.423
    assert signals["grounding_strength"] == pytest.approx((5 * 20 + 92 * 15) / (100 * 35), abs=1e-3)
    # peak: item_ids has 92% grounding and 15 instances ≥ MIN_GROUNDING_INSTANCES(10)
    assert signals["peak_grounding_strength"] == pytest.approx(0.92, abs=1e-3)
    assert signals["peak_grounding_instances"] == 15.0
    assert sources["peak_grounding_arg"] == "item_ids"


def test_peak_grounding_triggers_when_avg_is_low():
    """High peak arg + many zero-grounding volume args: avg is diluted but grounding must fire."""
    report = {
        "dataset_summary": {"read_tool_calls": 50, "write_tool_calls": 10},
        "cognitive_action_topology": [],
        "argument_emergence": [
            # High-volume zero-grounding args that would dominate a naive avg
            {"arg_name": "order_id",  "total_instances": 300, "requires_tool_chaining_pct": 90.0, "requires_grounding_pct": 0.0},
            {"arg_name": "user_id",   "total_instances": 200, "requires_tool_chaining_pct": 100.0, "requires_grounding_pct": 0.0},
            # Peak grounding arg with sufficient instances
            {"arg_name": "item_ids",  "total_instances": 80,  "requires_tool_chaining_pct": 0.0, "requires_grounding_pct": 88.0},
        ],
        "failure_heatmap": [],
    }
    engine = CapabilityInferenceEngine()
    result = engine.run(report)
    caps = result.required_capabilities

    assert caps["grounding"].required is True
    assert result.signal_sources["peak_grounding_arg"] == "item_ids"
    assert result.raw_signals["peak_grounding_instances"] == 80.0
    # Global avg would be ~12% — well below threshold
    assert result.raw_signals["grounding_strength"] < 0.20
    # But peak is 0.88 — drives the decision
    assert result.raw_signals["peak_grounding_strength"] == pytest.approx(0.88, abs=1e-3)


# ---------------------------------------------------------------------------
# capability_inference tests
# ---------------------------------------------------------------------------

def test_capability_inference_all_required():
    engine = CapabilityInferenceEngine()
    result = engine.run(_high_signal_report())
    caps = result.required_capabilities

    assert caps["memory"].required is True
    assert caps["grounding"].required is True
    assert caps["readiness"].required is True
    assert caps["deep_planning"].required is True

    for cap in caps.values():
        assert 0.0 <= cap.strength <= 1.0
        assert isinstance(cap.evidence, list)
        assert len(cap.evidence) >= 1


def test_capability_inference_minimal():
    engine = CapabilityInferenceEngine()
    result = engine.run(_low_signal_report())
    caps = result.required_capabilities

    assert caps["memory"].required is False
    assert caps["grounding"].required is False
    assert caps["deep_planning"].required is False


def test_load_report_from_path():
    report = _high_signal_report()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as fh:
        json.dump(report, fh)
        tmp_path = Path(fh.name)
    try:
        engine = CapabilityInferenceEngine()
        result = engine.run(tmp_path)
        assert isinstance(result, CapabilityInferenceResult)
        assert "memory" in result.required_capabilities
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# graph_recommender tests
# ---------------------------------------------------------------------------

def _make_inference(memory: bool, grounding: bool, readiness: bool, deep_planning: bool) -> CapabilityInferenceResult:
    def cap(required: bool, strength: float) -> CapabilityRequirement:
        return CapabilityRequirement(required=required, strength=strength, evidence=["test"])

    return CapabilityInferenceResult(
        required_capabilities={
            "memory": cap(memory, 0.9 if memory else 0.1),
            "grounding": cap(grounding, 0.8 if grounding else 0.1),
            "readiness": cap(readiness, 0.7 if readiness else 0.1),
            "deep_planning": cap(deep_planning, 0.6 if deep_planning else 0.1),
        },
        raw_signals={
            "chaining_strength": 0.9 if memory else 0.1,
            "grounding_strength": 0.8 if grounding else 0.1,
            "write_fraction": 0.7 if readiness else 0.05,
            "avg_chain_depth": 4.0 if deep_planning else 0.5,
            "write_failure_fraction": 0.7 if readiness else 0.05,
        },
    )


def test_graph_recommender_full_graph():
    inference = _make_inference(memory=True, grounding=True, readiness=True, deep_planning=True)
    rec = GraphRecommender().run(inference)
    node_ids = [n.id for n in rec.graph_spec.topological_order()]
    assert node_ids == ["perceive", "reason", "grounding", "readiness", "plan", "act", "learn"]
    assert rec.memory_required is True
    assert rec.readiness_required is True
    assert rec.parallel_lookup_nodes is False


def test_graph_recommender_minimal_graph():
    inference = _make_inference(memory=False, grounding=False, readiness=False, deep_planning=False)
    rec = GraphRecommender().run(inference)
    node_ids = [n.id for n in rec.graph_spec.topological_order()]
    assert node_ids == ["perceive", "plan", "act"]
    assert rec.memory_required is False
    assert rec.readiness_required is False


def test_recommended_graph_is_valid_graphspec():
    engine = CapabilityInferenceEngine()
    inference = engine.run(_high_signal_report())
    rec = GraphRecommender().run(inference)
    # topological_order() raises ValueError on cycles — must not raise
    order = rec.graph_spec.topological_order()
    assert len(order) == len(rec.graph_spec.nodes)
