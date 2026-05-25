from __future__ import annotations

from ..adapters.base import AgentMode, ModelAdapter
from ..schemas.common import Confidence, Evidence, UserInput
from ..schemas.perceive import MentionedEntity, PerceptionResult, RawFieldCandidate
from ._stub_heuristics import TOOL_KEYWORDS, extract_order_id, is_address_like


class PerceiveAgent:
    def __init__(
        self, mode: AgentMode = "stub", model_adapter: ModelAdapter | None = None
    ) -> None:
        self.mode = mode
        self._adapter = model_adapter

    def run(self, user_input: UserInput) -> PerceptionResult:
        if self.mode == "llm":
            return self._run_llm(user_input)
        if self.mode == "oracle":
            return self._run_oracle(user_input)
        return self._run_stub(user_input)

    def _run_stub(self, user_input: UserInput) -> PerceptionResult:
        msg = user_input.message.lower()
        tool_names = [t.name for t in user_input.available_tools]

        intent_candidates: list[str] = []
        candidate_tools: list[str] = []

        for tool_name in tool_names:
            keywords = TOOL_KEYWORDS.get(tool_name, [])
            if any(kw in msg for kw in keywords):
                candidate_tools.append(tool_name)
                intent_candidates.append(f"use_{tool_name}")

        if not intent_candidates:
            intent_candidates = ["unknown_intent"]

        ambiguity_detected = len(candidate_tools) > 1 or (
            len(candidate_tools) == 0 and bool(tool_names)
        )
        ambiguity_type = "none"
        if len(candidate_tools) > 1:
            ambiguity_type = "multiple_intents"
        elif len(candidate_tools) == 0 and bool(tool_names):
            ambiguity_type = "underspecified_action"

        mentioned_entities: list[MentionedEntity] = []
        raw_field_candidates: list[RawFieldCandidate] = []

        for token in user_input.message.split():
            order_id = extract_order_id(token)
            if order_id is not None:
                mentioned_entities.append(MentionedEntity(text=token, entity_type="order_id"))
                raw_field_candidates.append(
                    RawFieldCandidate(name="order_id", value=order_id, evidence_text=token)
                )

        if is_address_like(msg):
            raw_field_candidates.append(
                RawFieldCandidate(
                    name="new_address",
                    value=user_input.message,
                    evidence_text=user_input.message,
                )
            )

        confidence_score = 0.85 if candidate_tools else 0.3
        evidence = [
            Evidence(
                source="user_message",
                text=user_input.message,
                reason="keyword matching",
            )
        ]

        return PerceptionResult(
            intent_candidates=intent_candidates,
            mentioned_entities=mentioned_entities,
            raw_field_candidates=raw_field_candidates,
            ambiguity_detected=ambiguity_detected,
            ambiguity_type=ambiguity_type,
            candidate_tools=candidate_tools,
            confidence=Confidence(score=confidence_score, reason="stub keyword match"),
            evidence=evidence,
        )

    def _run_llm(self, user_input: UserInput) -> PerceptionResult:
        raise NotImplementedError("llm mode not yet implemented for PerceiveAgent")

    def _run_oracle(self, user_input: UserInput) -> PerceptionResult:
        raise NotImplementedError("oracle mode not yet implemented for PerceiveAgent")
