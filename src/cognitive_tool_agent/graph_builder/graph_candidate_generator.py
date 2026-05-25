from __future__ import annotations

from ..schemas.graph_builder import BehaviorDecomposition, GraphCandidate
from ..schemas.graph_spec import EdgeSpec, GraphSpec, NodeSpec


class GraphCandidateGenerator:
    def generate(self, decomposition: BehaviorDecomposition) -> list[GraphCandidate]:
        candidates = [
            _make_monolithic(),
            _make_perceive_plan_act(),
            _make_full_pipeline(),
        ]
        return candidates


def _make_monolithic() -> GraphCandidate:
    graph = GraphSpec(
        id="graph_A_monolithic",
        nodes=[NodeSpec(id="monolithic", role="monolithic")],
        edges=[],
        latency_estimate=1.0,
        cost_estimate=1.0,
    )
    return GraphCandidate(
        id="candidate_A",
        graph_spec=graph,
        rationale=(
            "Monolithic baseline: a single node handles perception, reasoning, and planning "
            "without decomposition. Minimal latency and cost; high coupling."
        ),
        latency_estimate=1.0,
        cost_estimate=1.0,
    )


def _make_perceive_plan_act() -> GraphCandidate:
    nodes = [
        NodeSpec(id="perceive", role="perceive"),
        NodeSpec(id="plan", role="plan"),
        NodeSpec(id="act", role="act"),
    ]
    edges = [
        EdgeSpec(from_node="perceive", to_node="plan"),
        EdgeSpec(from_node="plan", to_node="act"),
    ]
    graph = GraphSpec(
        id="graph_B_perceive_plan_act",
        nodes=nodes,
        edges=edges,
        latency_estimate=2.0,
        cost_estimate=2.0,
    )
    return GraphCandidate(
        id="candidate_B",
        graph_spec=graph,
        rationale=(
            "Three-stage graph: perceive → plan → act. "
            "Separates signal extraction from decision-making and execution. "
            "No readiness gate — lower latency but misses policy checks."
        ),
        latency_estimate=2.0,
        cost_estimate=2.0,
    )


def _make_full_pipeline() -> GraphCandidate:
    nodes = [
        NodeSpec(id="perceive", role="perceive"),
        NodeSpec(id="reason", role="reason"),
        NodeSpec(id="readiness", role="readiness"),
        NodeSpec(id="plan", role="plan"),
        NodeSpec(id="act", role="act"),
        NodeSpec(id="learn", role="learn"),
    ]
    edges = [
        EdgeSpec(from_node="perceive", to_node="reason"),
        EdgeSpec(from_node="reason", to_node="readiness"),
        EdgeSpec(from_node="readiness", to_node="plan"),
        EdgeSpec(from_node="plan", to_node="act"),
        EdgeSpec(from_node="act", to_node="learn"),
    ]
    graph = GraphSpec(
        id="graph_C_full_pipeline",
        nodes=nodes,
        edges=edges,
        latency_estimate=4.0,
        cost_estimate=4.0,
    )
    return GraphCandidate(
        id="candidate_C",
        graph_spec=graph,
        rationale=(
            "Full 6-stage cognitive pipeline: perceive → reason → readiness → plan → act → learn. "
            "Each cognitive responsibility is isolated. "
            "Maximum debuggability and policy enforcement at the cost of higher latency."
        ),
        latency_estimate=4.0,
        cost_estimate=4.0,
    )
