"""Wiring validator tests.

Each test documents a specific class of misconfiguration the validator is
expected to catch.  The cross-check gate (last test) verifies that the
validator's output on make_full_pipeline() matches the hardcoded arg
inventory from the plan, ensuring the two sources of truth stay aligned.

Cross-check gate (Phase 3d):
  Validator warnings on make_full_pipeline() (pre-backfill) must include
  exactly zero errors.  The only missing input is 'reasoning' for the plan
  node, which is OPTIONAL, so no error is raised.  The warning count for
  ordering-only edges should be exactly 1 (act→learn).  This matches the
  hardcoded arg inventory which identified reason→plan as the lone missing
  optional edge.
"""
from __future__ import annotations

import pytest

from cognitive_tool_agent.graph.wiring_validator import WiringError, WiringReport, validate_wiring
from cognitive_tool_agent.schemas.graph_spec import EdgeSpec, GraphSpec, NodeSpec


# ── Helpers ───────────────────────────────────────────────────────────────────


def _simple_graph(*nodes_edges) -> GraphSpec:
    """Quick factory: args are alternating NodeSpec/EdgeSpec objects."""
    nodes = [x for x in nodes_edges if isinstance(x, NodeSpec)]
    edges = [x for x in nodes_edges if isinstance(x, EdgeSpec)]
    return GraphSpec(id="test", nodes=nodes, edges=edges)


def _node(id: str, role: str) -> NodeSpec:
    return NodeSpec(id=id, role=role)


def _edge(from_node: str, to_node: str, provides: str | None = None) -> EdgeSpec:
    return EdgeSpec(from_node=from_node, to_node=to_node, provides=provides)


# ── Error cases ───────────────────────────────────────────────────────────────


def test_act_with_no_plan_edge_is_an_error():
    """act node with no incoming plan edge → WiringError (plan is required)."""
    graph = _simple_graph(
        _node("plan", "plan"),
        _node("act", "act"),
        # Deliberately omit the plan→act edge.
    )
    report = validate_wiring(graph)
    assert not report.ok
    assert any("plan" in e and "act" in e and "required" in e for e in report.errors), (
        f"Expected required-input error for act/plan, got: {report.errors}"
    )


def test_explicit_provides_contradicting_role_is_an_error():
    """Explicit provides that contradicts ROLE_OUTPUT → WiringError."""
    graph = _simple_graph(
        _node("perceive", "perceive"),
        _node("reason", "reason"),
        _edge("perceive", "reason", provides="reasoning"),  # wrong: perceive → perception
    )
    report = validate_wiring(graph)
    assert not report.ok
    assert any("contradicts" in e for e in report.errors), (
        f"Expected contradiction error, got: {report.errors}"
    )


def test_duplicate_slot_providers_is_an_error():
    """Two edges supplying the same slot to the same node → WiringError."""
    graph = _simple_graph(
        _node("perceive1", "perceive"),
        _node("perceive2", "perceive"),
        _node("reason", "reason"),
        _edge("perceive1", "reason"),
        _edge("perceive2", "reason"),
    )
    report = validate_wiring(graph)
    assert not report.ok
    assert any("perception" in e and "reason" in e for e in report.errors), (
        f"Expected duplicate-provider error, got: {report.errors}"
    )


def test_unknown_from_node_is_an_error():
    """Edge referencing a non-existent from_node → WiringError."""
    graph = _simple_graph(
        _node("plan", "plan"),
        _node("act", "act"),
        _edge("nonexistent", "act"),
        _edge("plan", "act"),
    )
    report = validate_wiring(graph)
    assert not report.ok
    assert any("nonexistent" in e for e in report.errors), (
        f"Expected unknown-node error, got: {report.errors}"
    )


def test_unknown_to_node_is_an_error():
    """Edge referencing a non-existent to_node → WiringError."""
    graph = _simple_graph(
        _node("plan", "plan"),
        _edge("plan", "nonexistent"),
    )
    report = validate_wiring(graph)
    assert not report.ok
    assert any("nonexistent" in e for e in report.errors), (
        f"Expected unknown-node error, got: {report.errors}"
    )


# ── Warning cases ──────────────────────────────────────────────────────────────


def test_act_to_learn_is_ordering_only_warning():
    """act→learn edge: 'action' is not in ROLE_INPUTS['learn'] → ordering-only warning."""
    graph = _simple_graph(
        _node("plan", "plan"),
        _node("act", "act"),
        _node("learn", "learn"),
        _edge("plan", "act"),
        _edge("act", "learn"),
    )
    report = validate_wiring(graph)
    assert report.ok, f"Expected no errors, got: {report.errors}"
    assert any("ordering-only" in w for w in report.warnings), (
        f"Expected ordering-only warning for act→learn, got: {report.warnings}"
    )


def test_node_after_learn_triggers_learn_invariant_warning():
    """A node appearing after learn in topo order violates the learn invariant."""
    graph = _simple_graph(
        _node("plan", "plan"),
        _node("act", "act"),
        _node("learn", "learn"),
        _node("reason", "reason"),
        _edge("plan", "act"),
        _edge("act", "learn"),
        _edge("learn", "reason"),  # wrong: reason after learn
    )
    report = validate_wiring(graph)
    assert any("learn invariant" in w for w in report.warnings), (
        f"Expected learn-invariant warning, got: {report.warnings}"
    )


# ── Cross-check gate (Phase 3d) ────────────────────────────────────────────────


def test_make_full_pipeline_validator_matches_arg_inventory():
    """Cross-check: validate_wiring() on make_full_pipeline() (post-Phase-4
    backfill) must return exactly the expected validation profile.

    Expected state (after reason→plan edge was added in Phase 4a):
    - Errors: ZERO.  All required inputs are satisfied.
    - Ordering-only warnings: EXACTLY ONE (act→learn supplies 'action' which
      is not in ROLE_INPUTS['learn']).
    - No learn-invariant warnings (learn is topologically last).

    If this test breaks, either the backfill list or ROLE_INPUTS is wrong.
    Update both together.
    """
    from cognitive_tool_agent.graph_builder.graph_candidate_generator import make_full_pipeline

    candidate = make_full_pipeline()
    report = validate_wiring(candidate.graph_spec)

    assert report.ok, (
        f"make_full_pipeline() has unexpected wiring errors: {report.errors}\n"
        f"The hardcoded arg inventory said there should be zero errors. "
        f"If an error was introduced, update the plan and this test together."
    )

    ordering_only_warnings = [w for w in report.warnings if "ordering-only" in w]
    assert len(ordering_only_warnings) == 1, (
        f"Expected exactly 1 ordering-only warning (act→learn), got {len(ordering_only_warnings)}:\n"
        + "\n".join(ordering_only_warnings)
    )

    learn_invariant_warnings = [w for w in report.warnings if "learn invariant" in w]
    assert len(learn_invariant_warnings) == 0, (
        f"Expected no learn-invariant warnings, got: {learn_invariant_warnings}"
    )
