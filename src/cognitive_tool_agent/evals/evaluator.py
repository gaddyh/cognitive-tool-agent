from __future__ import annotations

from ..schemas.dataset import DatasetRow
from ..schemas.trace import CognitiveTrace
from .metrics import (
    argument_exact_match,
    end_to_end_success,
    policy_violation_rate,
    stage_failure_rate,
    tool_name_accuracy,
)


class Evaluator:
    def score(
        self,
        traces: list[CognitiveTrace],
        rows: list[DatasetRow],
    ) -> dict[str, float]:
        return {
            "end_to_end_success": end_to_end_success(traces, rows),
            "tool_name_accuracy": tool_name_accuracy(traces, rows),
            "argument_exact_match": argument_exact_match(traces, rows),
            "policy_violation_rate": policy_violation_rate(traces, rows),
            "stage_failure_rate": stage_failure_rate(traces, rows),
        }
