from __future__ import annotations

from ..schemas.graph_spec import EdgeSpec, GraphSpec, NodeSpec
from ..schemas.recommender import CapabilityInferenceResult, RecommendedGraph


class GraphRecommender:
    """
    Maps a CapabilityInferenceResult to a RecommendedGraph.

    Node inclusion rules:
      perceive   — always
      reason     — if deep_planning.required
      grounding  — if grounding.required
      readiness  — if readiness.required
      plan       — always
      act        — always
      learn      — if memory.required
    """

    def run(self, inference: CapabilityInferenceResult) -> RecommendedGraph:
        caps = inference.required_capabilities
        memory_req = caps["memory"].required
        grounding_req = caps["grounding"].required
        readiness_req = caps["readiness"].required
        deep_planning_req = caps["deep_planning"].required

        # ── ordered node sequence ─────────────────────────────────────────────
        node_ids: list[str] = ["perceive"]
        if deep_planning_req:
            node_ids.append("reason")
        if grounding_req:
            node_ids.append("grounding")
        if readiness_req:
            node_ids.append("readiness")
        node_ids.append("plan")
        node_ids.append("act")
        if memory_req:
            node_ids.append("learn")

        nodes = [NodeSpec(id=nid, role=nid) for nid in node_ids]  # type: ignore[arg-type]
        edges = [
            EdgeSpec(from_node=node_ids[i], to_node=node_ids[i + 1])
            for i in range(len(node_ids) - 1)
        ]

        graph_id = "recommended_" + "_".join(
            cap for cap, req in [
                ("memory", memory_req),
                ("grounding", grounding_req),
                ("readiness", readiness_req),
                ("deep_planning", deep_planning_req),
            ]
            if req
        ) or "recommended_minimal"

        graph_spec = GraphSpec(id=graph_id, nodes=nodes, edges=edges)

        # ── rationale ─────────────────────────────────────────────────────────
        rationale: list[str] = [
            f"perceive always included (signal extraction baseline)",
            f"plan + act always included (core action execution)",
        ]
        if deep_planning_req:
            rationale.append(
                f"reason included: avg_chain_depth={inference.raw_signals['avg_chain_depth']:.2f} "
                f"exceeds reasoning threshold"
            )
        if grounding_req:
            rationale.append(
                f"grounding included: grounding_strength={caps['grounding'].strength:.2f} "
                f"— NL→structured mapping required"
            )
        if readiness_req:
            rationale.append(
                f"readiness included: readiness_strength={caps['readiness'].strength:.2f} "
                f"— write-risk gate required"
            )
        if memory_req:
            rationale.append(
                f"learn included: chaining_strength={caps['memory'].strength:.2f} "
                f"— persistent state/memory required"
            )

        # ── confidence: mean strength of included capabilities ─────────────────
        active_strengths = [
            caps["deep_planning"].strength if deep_planning_req else None,
            caps["grounding"].strength if grounding_req else None,
            caps["readiness"].strength if readiness_req else None,
            caps["memory"].strength if memory_req else None,
        ]
        active = [s for s in active_strengths if s is not None]
        confidence = round(sum(active) / len(active), 3) if active else 0.5

        return RecommendedGraph(
            graph_spec=graph_spec,
            memory_required=memory_req,
            readiness_required=readiness_req,
            parallel_lookup_nodes=False,
            rationale=rationale,
            required_capabilities=caps,
            confidence=confidence,
        )
