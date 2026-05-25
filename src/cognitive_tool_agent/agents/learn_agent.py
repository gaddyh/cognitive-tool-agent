from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ..schemas.common import UserInput
from ..schemas.learn import FailureAnalysis, LearningResult
from ..schemas.trace import CognitiveTrace


@runtime_checkable
class ModelAdapter(Protocol):
    def complete(self, prompt: str, output_schema: type) -> Any: ...


class LearnAgent:
    def __init__(self, model_adapter: ModelAdapter | None = None) -> None:
        self._adapter = model_adapter

    def run(self, user_input: UserInput, trace: CognitiveTrace) -> LearningResult:
        failed_stage, failure_type, explanation = _diagnose(trace)

        overall_success = (
            trace.action is not None
            and trace.action.success
        )

        return LearningResult(
            should_add_to_dataset=not overall_success,
            dataset_split_suggestion="dev" if not overall_success else "none",
            failure_analysis=FailureAnalysis(
                failed_stage=failed_stage,
                failure_type=failure_type,
                explanation=explanation,
            ),
            regression_tags=_regression_tags(trace),
            optimization_target=failed_stage if failed_stage != "none" else None,
        )


def _diagnose(
    trace: CognitiveTrace,
) -> tuple[str, str | None, str]:
    if trace.action is None:
        return "acting", "no_action_produced", "Executor produced no ActionResult"

    if not trace.action.success:
        error = trace.action.error or ""
        if "not registered" in error or "not found" in error:
            return "acting", "tool_not_found", error
        return "acting", "execution_error", error

    if trace.plan is not None and trace.plan.next_action == "reject":
        return "planning", "unsupported_action", "planner rejected the request"

    if trace.readiness is not None and not trace.readiness.ready:
        if trace.readiness.policy_violations:
            return "readiness", "policy_violation", "; ".join(trace.readiness.policy_violations)
        if trace.readiness.missing_required_fields:
            return "readiness", "missing_required_fields", str(trace.readiness.missing_required_fields)

    if trace.reasoning is not None and trace.reasoning.selected_tool is None:
        return "reasoning", "tool_not_selected", "reasoning stage did not select a tool"

    if trace.perception is not None and trace.perception.ambiguity_detected:
        return "perception", "ambiguity_unresolved", f"ambiguity_type={trace.perception.ambiguity_type}"

    return "none", None, "no failure detected"


def _regression_tags(trace: CognitiveTrace) -> list[str]:
    tags: list[str] = []
    if trace.perception and trace.perception.ambiguity_detected:
        tags.append("ambiguity")
    if trace.readiness and trace.readiness.policy_violations:
        tags.append("policy_violation")
    if trace.action and not trace.action.success:
        tags.append("execution_failure")
    if trace.plan and trace.plan.next_action == "reject":
        tags.append("unsupported_action")
    return tags
