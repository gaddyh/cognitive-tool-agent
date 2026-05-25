from __future__ import annotations

from ..schemas.graph_builder import BehaviorDecomposition, GraphCandidate
from ..schemas.graph_spec import EdgeSpec, GraphSpec, NodeSpec


class GraphCandidateGenerator:
    def generate(self, decomposition: BehaviorDecomposition) -> list[GraphCandidate]:
        candidates = [
            make_monolithic(),
            make_perceive_plan_act(),
            make_full_pipeline(),
        ]
        return candidates


def make_monolithic_baseline() -> GraphCandidate:
    nodes = [
        NodeSpec(id="plan", role="plan"),
        NodeSpec(id="act", role="act"),
    ]
    edges = [
        EdgeSpec(from_node="plan", to_node="act"),
    ]
    graph = GraphSpec(
        id="graph_A_monolithic_baseline",
        nodes=nodes,
        edges=edges,
        latency_estimate=1.0,
        cost_estimate=1.0,
    )
    return GraphCandidate(
        id="candidate_A",
        graph_spec=graph,
        rationale=(
            "Monolithic baseline: plan then act, with no perceive/reason/readiness upstream. "
            "Equivalent to keyword-only planning — the degenerate graph topology. "
            "Minimal latency and cost; no cognitive decomposition."
        ),
        latency_estimate=1.0,
        cost_estimate=1.0,
    )


def make_monolithic() -> GraphCandidate:
    return make_monolithic_baseline()


def make_perceive_plan_act() -> GraphCandidate:
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


def make_full_pipeline() -> GraphCandidate:
    nodes = [
        NodeSpec(id="perceive", role="perceive"),
        NodeSpec(id="reason", role="reason"),
        NodeSpec(id="readiness", role="readiness"),
        NodeSpec(id="plan", role="plan"),
        NodeSpec(id="act", role="act"),
        NodeSpec(id="learn", role="learn"),
    ]
    edges = [
        EdgeSpec(from_node="perceive",  to_node="reason",    provides="perception"),
        EdgeSpec(from_node="reason",    to_node="readiness", provides="reasoning"),
        EdgeSpec(from_node="reason",    to_node="plan",      provides="reasoning"),
        EdgeSpec(from_node="readiness", to_node="plan",      provides="readiness"),
        EdgeSpec(from_node="plan",      to_node="act",       provides="plan"),
        EdgeSpec(from_node="act",       to_node="learn"),
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
