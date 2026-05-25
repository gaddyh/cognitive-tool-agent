"""Full-pipeline behavioral equivalence test.

This fixture was generated from the pre-refactor codebase (before any Phase 1
changes) by scripts/generate_equivalence_fixture.py and committed to
tests/fixtures/full_pipeline_expected.json.

It is the primary safety net for full-pipeline wiring correctness:
  - If Phase 1 (agent interface) introduced a behavior change, this test
    will catch it because the fixture predates Phase 1.
  - After Phase 4 (flip dispatch to edge-driven), this test verifies that
    the new execution path produces identical outputs.

Checked fields per row:
  action_type, tool_name, tool_arguments, plan.next_action, reasoning.selected_tool

Phase 3 gate: must be green when committed (documents current behavior).
Phase 4 gate: re-run after dispatch flip; must still be green.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from cognitive_tool_agent.datasets.loader import load_jsonl
from cognitive_tool_agent.tools.fake_tools import DEFAULT_REGISTRY

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "full_pipeline_expected.json"
DATASET_PATH = Path(__file__).parent.parent / "data" / "dev" / "tool_calling_micro.jsonl"


@pytest.fixture(scope="module")
def expected_records() -> list[dict]:
    return json.loads(FIXTURE_PATH.read_text())


@pytest.fixture(scope="module")
def rows():
    return load_jsonl(DATASET_PATH)


@pytest.mark.parametrize("row_index", range(5))
def test_full_pipeline_matches_pre_refactor_fixture(rows, expected_records, row_index):
    """Full pipeline (perceive→reason→readiness→plan→act→learn) must produce
    the same behavioral outputs as the pre-refactor baseline captured in the
    committed fixture."""
    from cognitive_tool_agent.graph.cognitive_graph import GraphExecutor
    from cognitive_tool_agent.graph_builder.graph_candidate_generator import make_full_pipeline
    from cognitive_tool_agent.schemas.experiment import ExperimentSpec

    row = rows[row_index]
    expected = expected_records[row_index]
    assert expected["row_id"] == row.id, (
        f"Fixture row {row_index} has id={expected['row_id']!r}, "
        f"but dataset row {row_index} has id={row.id!r}. "
        f"Re-run scripts/generate_equivalence_fixture.py to regenerate."
    )

    candidate = make_full_pipeline()
    executor = GraphExecutor()
    trace = executor.run(ExperimentSpec(graph=candidate.graph_spec), row, DEFAULT_REGISTRY)

    assert (trace.action.action_type if trace.action else None) == expected["action_type"], (
        f"row {row.id}: action_type mismatch"
    )
    assert (trace.action.tool_name if trace.action else None) == expected["tool_name"], (
        f"row {row.id}: tool_name mismatch"
    )
    assert (trace.action.tool_arguments if trace.action else None) == expected["tool_arguments"], (
        f"row {row.id}: tool_arguments mismatch"
    )
    assert (trace.plan.next_action if trace.plan else None) == expected["plan_next_action"], (
        f"row {row.id}: plan.next_action mismatch"
    )
    assert (trace.reasoning.selected_tool if trace.reasoning else None) == expected["reasoning_selected_tool"], (
        f"row {row.id}: reasoning.selected_tool mismatch"
    )
