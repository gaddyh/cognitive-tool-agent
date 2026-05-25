from __future__ import annotations

from ..schemas.graph_builder import BehaviorDecomposition, DatasetProfile


_DECOMPOSITIONS: dict[str, tuple[list[str], str]] = {
    "tool_calling": (
        ["perceive", "reason", "readiness", "plan", "act", "learn"],
        "Tool-calling tasks require intent perception, entity grounding, policy checking, "
        "action planning, execution, and runtime memory updates.",
    ),
    "rag": (
        ["query_understanding", "retrieval_planning", "evidence_selection", "answer_synthesis", "faithfulness_check"],
        "RAG tasks require query decomposition, retrieval strategy, evidence ranking, "
        "grounded synthesis, and output faithfulness verification.",
    ),
    "classification": (
        ["input_normalization", "label_discrimination", "confidence_gating"],
        "Classification tasks require input canonicalization, label scoring, "
        "and confidence-based decision gating.",
    ),
    "planning": (
        ["goal_parsing", "constraint_extraction", "plan_generation", "plan_validation"],
        "Planning tasks require goal understanding, constraint identification, "
        "step generation, and feasibility validation.",
    ),
    "generation": (
        ["intent_parsing", "content_planning", "generation", "quality_check"],
        "Generation tasks require intent parsing, content structuring, text generation, "
        "and output quality assessment.",
    ),
    "unknown": (
        ["perceive", "plan", "act"],
        "Unknown task type defaults to minimal perceive→plan→act decomposition.",
    ),
}


class BehaviorDecomposer:
    def decompose(self, profile: DatasetProfile) -> BehaviorDecomposition:
        stages, rationale = _DECOMPOSITIONS.get(
            profile.task_type, _DECOMPOSITIONS["unknown"]
        )
        return BehaviorDecomposition(
            task_type=profile.task_type,
            stages=list(stages),
            rationale=rationale,
        )
