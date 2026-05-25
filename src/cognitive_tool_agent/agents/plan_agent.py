from __future__ import annotations

from typing import Any

from ..adapters.base import AgentMode, ModelAdapter
from ..schemas.common import Confidence, UserInput
from ..schemas.reason import ReasoningResult
from ..schemas.readiness import ReadinessResult
from ..schemas.plan import PlanResult, ToolCallPlan
from ._stub_heuristics import TOOL_KEYWORDS, extract_order_id


class PlanAgent:
    def __init__(
        self, mode: AgentMode = "stub", model_adapter: ModelAdapter | None = None
    ) -> None:
        self.mode = mode
        self._adapter = model_adapter

    def run(
        self,
        user_input: UserInput,
        reasoning: ReasoningResult | None,
        readiness: ReadinessResult | None,
    ) -> PlanResult:
        if self.mode == "llm":
            return self._run_llm(user_input, reasoning, readiness)
        if self.mode == "oracle":
            return self._run_oracle(user_input, reasoning, readiness)
        return self._run_stub(user_input, reasoning, readiness)

    def _run_stub(
        self,
        user_input: UserInput,
        reasoning: ReasoningResult | None,
        readiness: ReadinessResult | None,
    ) -> PlanResult:
        msg = user_input.message.lower()
        tool_names = [t.name for t in user_input.available_tools]

        if reasoning is None:
            return self._plan_from_keywords(msg, tool_names, user_input)

        selected_tool = reasoning.selected_tool

        if selected_tool is None:
            return PlanResult(
                next_action="reject",
                blocking_reasons=["no matching tool for this request"],
                confidence=Confidence(score=0.9, reason="no tool matched"),
            )

        if readiness is not None and not readiness.ready:
            followup = _build_followup(readiness, selected_tool)
            return PlanResult(
                next_action="ask_followup",
                followup_question=followup,
                blocking_reasons=readiness.blocking_reasons,
                confidence=Confidence(score=0.8, reason="readiness blocked"),
            )

        if reasoning.reasoning_status in ("needs_clarification",):
            return PlanResult(
                next_action="ask_followup",
                followup_question=_build_clarification(reasoning),
                blocking_reasons=[],
                confidence=Confidence(score=0.7, reason="reasoning needs clarification"),
            )

        tool_call = _build_tool_call(selected_tool, reasoning, user_input)
        return PlanResult(
            next_action="execute_tool",
            tool_call=tool_call,
            confidence=Confidence(score=0.85, reason="tool and args resolved"),
        )

    def _plan_from_keywords(
        self, msg: str, tool_names: list[str], user_input: UserInput
    ) -> PlanResult:
        matched: list[str] = []
        for tool_name in tool_names:
            keywords = TOOL_KEYWORDS.get(tool_name, [])
            if any(kw in msg for kw in keywords):
                matched.append(tool_name)

        if not matched:
            return PlanResult(
                next_action="reject",
                blocking_reasons=["no matching tool for this request"],
                confidence=Confidence(score=0.8, reason="monolithic stub: no keyword match"),
            )

        selected = matched[0]
        args: dict[str, Any] = {}
        for token in user_input.message.split():
            order_id = extract_order_id(token)
            if order_id is not None:
                args["order_id"] = order_id

        return PlanResult(
            next_action="execute_tool",
            tool_call=ToolCallPlan(tool_name=selected, arguments=args),
            confidence=Confidence(score=0.6, reason="monolithic stub keyword match"),
        )

    def _run_llm(
        self,
        user_input: UserInput,
        reasoning: ReasoningResult | None,
        readiness: ReadinessResult | None,
    ) -> PlanResult:
        raise NotImplementedError("llm mode not yet implemented for PlanAgent")

    def _run_oracle(
        self,
        user_input: UserInput,
        reasoning: ReasoningResult | None,
        readiness: ReadinessResult | None,
    ) -> PlanResult:
        raise NotImplementedError("oracle mode not yet implemented for PlanAgent")


def _build_followup(readiness: ReadinessResult, tool: str) -> str:
    if readiness.policy_violations:
        return f"Please confirm you want to proceed with '{tool}'."
    if readiness.missing_required_fields:
        fields = ", ".join(readiness.missing_required_fields)
        return f"I need more information: {fields}."
    return "Could you please clarify your request?"


def _build_clarification(reasoning: ReasoningResult) -> str:
    missing = [r.field_name for r in reasoning.missing_requirements if not r.can_infer]
    if missing:
        return f"Which {missing[0]} did you mean?"
    if reasoning.selected_tool is None:
        return "I'm not sure what you'd like me to do. Could you be more specific?"
    return "Could you provide more details?"


def _build_tool_call(
    tool_name: str,
    reasoning: ReasoningResult,
    user_input: UserInput,
) -> ToolCallPlan:
    args: dict[str, Any] = {}

    for entity in reasoning.resolved_entities:
        if entity.entity_type == "order_id":
            args["order_id"] = entity.resolved_value

    tool_schema = next(
        (t for t in user_input.available_tools if t.name == tool_name), None
    )
    if tool_schema:
        for req in tool_schema.required_fields:
            if req not in args and req in user_input.world_state:
                args[req] = user_input.world_state[req]

    if tool_name == "update_address":
        msg = user_input.message
        for raw_fc_name in ("new_address",):
            if raw_fc_name not in args:
                args[raw_fc_name] = msg

    return ToolCallPlan(tool_name=tool_name, arguments=args)
