from __future__ import annotations

from collections import Counter

from ..schemas.dataset import DatasetRow
from ..schemas.graph_builder import DatasetProfile


class DatasetProfiler:
    def profile(self, rows: list[DatasetRow]) -> DatasetProfile:
        if not rows:
            return DatasetProfile(
                task_type="unknown",
                input_space="empty",
                output_space="empty",
                label_set=[],
                ambiguity_rate=0.0,
                contradiction_count=0,
                row_count=0,
                tool_count=0,
            )

        task_type = _infer_task_type(rows)
        all_tools: set[str] = set()
        for row in rows:
            all_tools.update(row.tools)

        action_counter: Counter[str] = Counter(r.expected.expected_action for r in rows)
        label_set = list(action_counter.keys())

        ambiguous_count = sum(
            1 for r in rows
            if r.expected.expected_action in ("ask_followup",)
            or "ambiguity" in r.tags
            or "missing_referent" in r.tags
        )
        ambiguity_rate = ambiguous_count / len(rows)

        notes: list[str] = []
        if any("policy" in r.tags for r in rows):
            notes.append("dataset includes policy/confirmation scenarios")
        if any("multi_turn" in r.tags for r in rows):
            notes.append("dataset includes multi-turn conversations")
        if any("unsupported_action" in r.tags for r in rows):
            notes.append("dataset includes rejection/unsupported scenarios")

        input_space = "natural_language_user_messages"
        output_space = f"tool_calls_or_followups ({', '.join(sorted(label_set))})"

        return DatasetProfile(
            task_type=task_type,
            input_space=input_space,
            output_space=output_space,
            label_set=label_set,
            ambiguity_rate=round(ambiguity_rate, 3),
            contradiction_count=0,
            row_count=len(rows),
            tool_count=len(all_tools),
            notes=notes,
        )


def _infer_task_type(rows: list[DatasetRow]) -> str:
    tool_rows = sum(1 for r in rows if r.tools)
    if tool_rows / len(rows) >= 0.5:
        return "tool_calling"
    return "unknown"
