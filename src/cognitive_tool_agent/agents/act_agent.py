from __future__ import annotations

from ..adapters.base import AgentMode, ModelAdapter
from ..graph.node_input import NodeInput
from ..schemas.act import ActionResult
from ..schemas.plan import PlanResult
from ..tools.registry import ToolRegistry


class ActAgent:
    def __init__(
        self, mode: AgentMode = "stub", model_adapter: ModelAdapter | None = None
    ) -> None:
        self.mode = mode
        self._adapter = model_adapter

    def run(self, ctx: NodeInput) -> ActionResult:
        if self.mode == "llm":
            return self._run_llm(ctx)
        if self.mode == "oracle":
            return self._run_oracle(ctx)
        return self._run_stub(ctx)

    def _run_stub(self, ctx: NodeInput) -> ActionResult:
        plan = ctx.plan
        registry = ctx.registry
        if plan is None:
            return ActionResult(
                action_type="abstained",
                success=False,
                error="no plan provided",
            )

        next_action = plan.next_action

        if next_action == "ask_followup":
            return ActionResult(
                action_type="followup_asked",
                success=True,
                user_facing_message=plan.followup_question,
            )

        if next_action == "answer_directly":
            return ActionResult(
                action_type="answered_directly",
                success=True,
                user_facing_message=plan.direct_answer,
            )

        if next_action == "abstain":
            return ActionResult(
                action_type="abstained",
                success=True,
                user_facing_message="I cannot assist with that.",
            )

        if next_action == "reject":
            return ActionResult(
                action_type="rejected",
                success=True,
                user_facing_message="I cannot assist with that request — it is not supported.",
            )

        if next_action == "execute_tool":
            if plan.tool_call is None:
                return ActionResult(
                    action_type="abstained",
                    success=False,
                    error="plan specifies execute_tool but no tool_call present",
                )
            tool_name = plan.tool_call.tool_name
            arguments = plan.tool_call.arguments
            try:
                result = registry.call(tool_name, arguments)
                return ActionResult(
                    action_type="tool_executed",
                    success=True,
                    tool_name=tool_name,
                    tool_arguments=arguments,
                    tool_result=result,
                    user_facing_message=f"Tool '{tool_name}' executed successfully.",
                )
            except Exception as exc:
                return ActionResult(
                    action_type="tool_executed",
                    success=False,
                    tool_name=tool_name,
                    tool_arguments=arguments,
                    error=str(exc),
                )

        return ActionResult(
            action_type="abstained",
            success=False,
            error=f"unknown next_action: {next_action!r}",
        )

    def _run_llm(self, ctx: NodeInput) -> ActionResult:
        raise NotImplementedError("llm mode not yet implemented for ActAgent")

    def _run_oracle(self, ctx: NodeInput) -> ActionResult:
        raise NotImplementedError("oracle mode not yet implemented for ActAgent")
