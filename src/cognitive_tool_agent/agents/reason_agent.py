from __future__ import annotations

from ..adapters.base import AgentMode, ModelAdapter
from ..graph.node_input import NodeInput
from ..schemas.common import Confidence, Evidence, UserInput
from ..schemas.perceive import PerceptionResult
from ..schemas.reason import MissingRequirement, ReasoningResult, ResolvedEntity
from ._stub_heuristics import CONFIRMATION_KEYWORDS


class ReasonAgent:
    def __init__(
        self, mode: AgentMode = "stub", model_adapter: ModelAdapter | None = None
    ) -> None:
        self.mode = mode
        self._adapter = model_adapter

    def run(self, ctx: NodeInput) -> ReasoningResult:
        if self.mode == "llm":
            return self._run_llm(ctx)
        if self.mode == "oracle":
            return self._run_oracle(ctx)
        return self._run_stub(ctx)

    def _run_stub(self, ctx: NodeInput) -> ReasoningResult:
        user_input = ctx.user_input
        perception = ctx.perception
        if perception is None:
            return ReasoningResult(
                selected_intent=None,
                selected_tool=None,
                reasoning_status="needs_clarification",
                confidence=Confidence(score=0.1, reason="no perception input"),
            )

        candidate_tools = perception.candidate_tools
        selected_tool = candidate_tools[0] if candidate_tools else None
        selected_intent = perception.intent_candidates[0] if perception.intent_candidates else None

        resolved_entities: list[ResolvedEntity] = []
        for entity in perception.mentioned_entities:
            resolved_entities.append(
                ResolvedEntity(
                    surface_text=entity.text,
                    entity_type=entity.entity_type or "unknown",
                    resolved_id=entity.text.lstrip("#"),
                    resolved_value=entity.text.lstrip("#"),
                    status="resolved",
                )
            )

        missing_requirements: list[MissingRequirement] = []
        if selected_tool:
            tool_schema = next(
                (t for t in user_input.available_tools if t.name == selected_tool), None
            )
            if tool_schema:
                resolved_field_names = {
                    fc.name for fc in perception.raw_field_candidates
                }
                for required_field in tool_schema.required_fields:
                    if required_field not in resolved_field_names:
                        can_infer = required_field in user_input.world_state
                        missing_requirements.append(
                            MissingRequirement(
                                field_name=required_field,
                                reason=f"not found in message or world state",
                                can_infer=can_infer,
                            )
                        )

        if selected_tool is None:
            reasoning_status = "unsupported"
        elif perception.ambiguity_detected and not resolved_entities:
            reasoning_status = "needs_clarification"
        elif missing_requirements and not all(m.can_infer for m in missing_requirements):
            reasoning_status = "needs_clarification"
        else:
            reasoning_status = "ready"

        needs_confirm = (
            selected_tool == "cancel_order"
            and "pending_confirmation" not in user_input.world_state
            and not any(
                kw in user_input.message.lower()
                for kw in CONFIRMATION_KEYWORDS
            )
        )
        if needs_confirm:
            reasoning_status = "needs_clarification"

        confidence_score = 0.8 if reasoning_status == "ready" else 0.4
        evidence = [
            Evidence(
                source="user_message",
                text=user_input.message,
                reason="stub reasoning from perception candidates",
            )
        ]

        return ReasoningResult(
            selected_intent=selected_intent,
            selected_tool=selected_tool,
            resolved_entities=resolved_entities,
            missing_requirements=missing_requirements,
            reasoning_status=reasoning_status,
            confidence=Confidence(score=confidence_score, reason="stub reasoning"),
            evidence=evidence,
        )

    def _run_llm(self, ctx: NodeInput) -> ReasoningResult:
        raise NotImplementedError("llm mode not yet implemented for ReasonAgent")

    def _run_oracle(self, ctx: NodeInput) -> ReasoningResult:
        raise NotImplementedError("oracle mode not yet implemented for ReasonAgent")
