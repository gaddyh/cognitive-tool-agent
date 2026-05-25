from __future__ import annotations

from ..adapters.base import AgentMode, ModelAdapter
from ..schemas.common import Confidence, UserInput
from ..schemas.reason import ReasoningResult
from ..schemas.readiness import ReadinessResult
from ..tools.registry import ToolRegistry


_CONFIRMATION_REQUIRED_TOOLS = {"cancel_order"}


class ReadinessAgent:
    def __init__(
        self, mode: AgentMode = "stub", model_adapter: ModelAdapter | None = None
    ) -> None:
        self.mode = mode
        self._adapter = model_adapter

    def run(
        self,
        user_input: UserInput,
        reasoning: ReasoningResult | None,
        registry: ToolRegistry,
    ) -> ReadinessResult:
        if self.mode == "llm":
            return self._run_llm(user_input, reasoning, registry)
        if self.mode == "oracle":
            return self._run_oracle(user_input, reasoning, registry)
        return self._run_stub(user_input, reasoning, registry)

    def _run_stub(
        self,
        user_input: UserInput,
        reasoning: ReasoningResult | None,
        registry: ToolRegistry,
    ) -> ReadinessResult:
        if reasoning is None or reasoning.selected_tool is None:
            return ReadinessResult(
                ready=False,
                blocking_reasons=["no tool selected by reasoning stage"],
                confidence=Confidence(score=0.1, reason="no reasoning input"),
            )

        selected_tool = reasoning.selected_tool
        blocking_reasons: list[str] = []
        policy_violations: list[str] = []
        missing_required_fields: list[str] = []

        tool_schema = registry.lookup(selected_tool)
        if tool_schema is None:
            blocking_reasons.append(f"tool '{selected_tool}' not found in registry")
            return ReadinessResult(
                ready=False,
                blocking_reasons=blocking_reasons,
                confidence=Confidence(score=0.0, reason="tool not registered"),
            )

        for req in reasoning.missing_requirements:
            if not req.can_infer:
                missing_required_fields.append(req.field_name)
                blocking_reasons.append(f"required field '{req.field_name}' is missing")

        if selected_tool in _CONFIRMATION_REQUIRED_TOOLS:
            has_confirmation = (
                "pending_confirmation" in user_input.world_state
                or any(
                    kw in user_input.message.lower()
                    for kw in ("yes", "confirm", "go ahead", "please cancel")
                )
            )
            if not has_confirmation:
                policy_violations.append(
                    f"'{selected_tool}' requires explicit user confirmation before execution"
                )
                blocking_reasons.append("confirmation policy not satisfied")

        if reasoning.reasoning_status == "unsupported":
            blocking_reasons.append("intent is not supported by any available tool")

        ready = len(blocking_reasons) == 0
        confidence_score = 0.9 if ready else 0.5

        return ReadinessResult(
            ready=ready,
            blocking_reasons=blocking_reasons,
            policy_violations=policy_violations,
            missing_required_fields=missing_required_fields,
            confidence=Confidence(score=confidence_score, reason="stub readiness check"),
        )

    def _run_llm(
        self,
        user_input: UserInput,
        reasoning: ReasoningResult | None,
        registry: ToolRegistry,
    ) -> ReadinessResult:
        raise NotImplementedError("llm mode not yet implemented for ReadinessAgent")

    def _run_oracle(
        self,
        user_input: UserInput,
        reasoning: ReasoningResult | None,
        registry: ToolRegistry,
    ) -> ReadinessResult:
        raise NotImplementedError("oracle mode not yet implemented for ReadinessAgent")
