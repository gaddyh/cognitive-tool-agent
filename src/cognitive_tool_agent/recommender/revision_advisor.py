from __future__ import annotations

from ..schemas.graph_runner import GraphEvaluationReport, GraphEvaluationRow
from ..schemas.recommender import CapabilityInferenceResult
from ..schemas.revision_advisor import GraphRevisionAdvisorReport, GraphRevisionSuggestion


def _find_row(report: GraphEvaluationReport, graph_id: str) -> GraphEvaluationRow | None:
    return next((r for r in report.results if r.graph_id == graph_id), None)


class GraphRevisionAdvisor:
    def advise(
        self,
        eval_report: GraphEvaluationReport,
        capability_inference: CapabilityInferenceResult,
    ) -> GraphRevisionAdvisorReport:
        stub = _find_row(eval_report, "recommended_stub")
        oracle = _find_row(eval_report, "recommended_oracle")
        minimal = _find_row(eval_report, "minimal")
        monolithic = _find_row(eval_report, "monolithic")

        graph_id = stub.graph_id if stub else "recommended"
        suggestions: list[GraphRevisionSuggestion] = []

        grounding_cap = capability_inference.required_capabilities.get("grounding")
        signals = capability_inference.raw_signals

        if stub and oracle:
            oracle_gap = oracle.end_to_end_success - stub.end_to_end_success
            if oracle_gap > 0.10:
                suggestions.append(
                    GraphRevisionSuggestion(
                        target_capability="grounding",
                        failure_pattern="oracle_gap",
                        suggestion=(
                            "Grounding quality is the primary bottleneck. "
                            "Oracle grounding improved E2E success by "
                            f"{oracle_gap:.0%}. Invest in a real grounding agent "
                            "with entity lookup and fuzzy ID resolution."
                        ),
                        rationale=(
                            f"stub E2E={stub.end_to_end_success:.2f} vs "
                            f"oracle E2E={oracle.end_to_end_success:.2f} "
                            f"(Δ={oracle_gap:+.2f})"
                        ),
                        priority="high",
                        evidence=[
                            f"oracle_gap={oracle_gap:.3f}",
                            f"grounding_strength={signals.get('grounding_strength', 0):.3f}",
                            f"peak_grounding_strength={signals.get('peak_grounding_strength', 0):.3f}",
                        ],
                    )
                )

        if stub and stub.argument_exact_match < 0.5:
            peak_arg = capability_inference.signal_sources.get("peak_grounding_arg", "unknown")
            suggestions.append(
                GraphRevisionSuggestion(
                    target_capability="grounding",
                    failure_pattern="argument_resolution_failure",
                    suggestion=(
                        f"Split grounding node: separate '{peak_arg}' resolution "
                        "from general entity grounding. High-frequency ID fields "
                        "need dedicated lookup chains."
                    ),
                    rationale=(
                        f"argument_exact_match={stub.argument_exact_match:.2f} "
                        f"(below 0.50 threshold). Peak grounding arg: '{peak_arg}' "
                        f"with strength={signals.get('peak_grounding_strength', 0):.2f}."
                    ),
                    priority="high",
                    evidence=[
                        f"argument_exact_match={stub.argument_exact_match:.3f}",
                        f"peak_grounding_arg={peak_arg}",
                        f"peak_grounding_instances={signals.get('peak_grounding_instances', 0):.0f}",
                    ],
                )
            )

        if stub and stub.policy_violation_rate > 0.15:
            suggestions.append(
                GraphRevisionSuggestion(
                    target_capability="readiness",
                    failure_pattern="policy_violations_high",
                    suggestion=(
                        "Add an explicit confirmation verifier before act. "
                        "High policy violation rate suggests readiness gate is "
                        "not blocking risky write actions reliably."
                    ),
                    rationale=(
                        f"policy_violation_rate={stub.policy_violation_rate:.2f} "
                        "(above 0.15 threshold). Write fraction="
                        f"{signals.get('write_fraction', 0):.2f}."
                    ),
                    priority="medium",
                    evidence=[
                        f"policy_violation_rate={stub.policy_violation_rate:.3f}",
                        f"write_fraction={signals.get('write_fraction', 0):.3f}",
                        f"write_failure_fraction={signals.get('write_failure_fraction', 0):.3f}",
                    ],
                )
            )

        if stub and minimal and stub.end_to_end_success <= minimal.end_to_end_success:
            suggestions.append(
                GraphRevisionSuggestion(
                    target_capability="graph_topology",
                    failure_pattern="recommended_not_beating_minimal",
                    suggestion=(
                        "Recommended graph is not outperforming the minimal graph. "
                        "This is likely the stub-grounding ceiling. Resolve grounding "
                        "before concluding that the topology is wrong."
                    ),
                    rationale=(
                        f"recommended_stub E2E={stub.end_to_end_success:.2f} ≤ "
                        f"minimal E2E={minimal.end_to_end_success:.2f}. "
                        "Stub grounding cannot resolve IDs, making extra nodes a drag."
                    ),
                    priority="medium",
                    evidence=[
                        f"recommended_stub_e2e={stub.end_to_end_success:.3f}",
                        f"minimal_e2e={minimal.end_to_end_success:.3f}",
                        "grounding_mode=stub",
                    ],
                )
            )

        if signals.get("avg_chain_depth", 0) > 3.5 and stub:
            suggestions.append(
                GraphRevisionSuggestion(
                    target_capability="memory",
                    failure_pattern="high_chain_depth",
                    suggestion=(
                        "Add a state tracker (memory node) before readiness. "
                        "High avg_chain_depth indicates multi-hop tool sequences "
                        "that require retained context across calls."
                    ),
                    rationale=(
                        f"avg_chain_depth={signals.get('avg_chain_depth', 0):.2f} "
                        "(above 3.5 threshold). Chain context likely lost between calls."
                    ),
                    priority="low",
                    evidence=[
                        f"avg_chain_depth={signals.get('avg_chain_depth', 0):.3f}",
                        f"chaining_strength={signals.get('chaining_strength', 0):.3f}",
                    ],
                )
            )

        return GraphRevisionAdvisorReport(graph_id=graph_id, suggestions=suggestions)
