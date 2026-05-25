from __future__ import annotations

from ..schemas.graph_builder import DatasetProfile, EvaluationPlan, StageMetric


class EvaluationDesigner:
    def design(self, profile: DatasetProfile) -> EvaluationPlan:
        metrics = _base_metrics()

        if profile.task_type == "tool_calling":
            metrics += _tool_calling_metrics()

        if profile.ambiguity_rate > 0.0:
            metrics.append(
                StageMetric(
                    name="ambiguity_detection_rate",
                    stage="perceive",
                    description=(
                        "Fraction of ambiguous inputs where perception correctly flagged ambiguity."
                    ),
                )
            )

        return EvaluationPlan(metrics=metrics)


def _base_metrics() -> list[StageMetric]:
    return [
        StageMetric(
            name="end_to_end_success",
            stage=None,
            description="Fraction of rows where the final action matches the expected action.",
        ),
        StageMetric(
            name="stage_failure_rate",
            stage=None,
            description="Fraction of rows where at least one cognitive stage produced a failure.",
        ),
    ]


def _tool_calling_metrics() -> list[StageMetric]:
    return [
        StageMetric(
            name="tool_name_accuracy",
            stage="plan",
            description="Fraction of tool-calling rows where the correct tool was selected.",
        ),
        StageMetric(
            name="argument_exact_match",
            stage="act",
            description="Fraction of tool-calling rows where arguments exactly match expected.",
        ),
        StageMetric(
            name="policy_violation_rate",
            stage="readiness",
            description=(
                "Fraction of rows where the readiness stage detected a policy violation. "
                "High values indicate insufficient confirmation enforcement."
            ),
        ),
    ]
