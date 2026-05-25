from __future__ import annotations

import re
from typing import Any, Literal

from ..schemas.common import UserInput
from ..schemas.dataset import DatasetRow
from ..schemas.grounding import GroundingResult
from ..schemas.reason import ReasoningResult

_ID_PATTERN = re.compile(r"(^|_)(id|ids)$", re.IGNORECASE)
_WRITE_VERBS = {"modify", "cancel", "exchange", "update", "return", "create", "delete"}


def _looks_like_id_field(name: str) -> bool:
    return bool(_ID_PATTERN.search(name))


class GroundingAgent:
    def __init__(
        self, grounding_mode: Literal["stub", "oracle", "disabled"] = "stub"
    ) -> None:
        self.grounding_mode = grounding_mode

    def run(
        self,
        user_input: UserInput,
        reasoning: ReasoningResult | None,
        row: DatasetRow,
    ) -> GroundingResult:
        if self.grounding_mode == "disabled":
            return GroundingResult(grounding_mode="disabled")

        if self.grounding_mode == "oracle":
            return self._run_oracle(row)

        return self._run_stub(user_input, reasoning)

    def _run_stub(
        self,
        user_input: UserInput,
        reasoning: ReasoningResult | None,
    ) -> GroundingResult:
        resolved_args: dict[str, Any] = {}
        unresolved_ids: list[str] = []
        candidates_examined = 0

        if reasoning is not None:
            candidates_examined = len(reasoning.resolved_entities)
            for entity in reasoning.resolved_entities:
                key = entity.entity_type or entity.surface_text
                if entity.status == "resolved" and (
                    entity.resolved_id is not None or entity.resolved_value is not None
                ):
                    resolved_args[key] = entity.resolved_id or entity.resolved_value
                elif entity.status in ("ambiguous", "missing"):
                    unresolved_ids.append(key)

        selected_tool = reasoning.selected_tool if reasoning else None
        tool_schema = next(
            (t for t in user_input.available_tools if t.name == selected_tool),
            None,
        )
        if tool_schema:
            for field in tool_schema.required_fields:
                if field not in resolved_args and field not in unresolved_ids:
                    if _looks_like_id_field(field):
                        unresolved_ids.append(field)

        total = len(resolved_args) + len(unresolved_ids)
        confidence = len(resolved_args) / total if total > 0 else 0.5

        return GroundingResult(
            grounding_mode="stub",
            resolved_args=resolved_args,
            unresolved_ids=unresolved_ids,
            grounding_confidence=round(confidence, 3),
            candidates_examined=candidates_examined,
        )

    def _run_oracle(self, row: DatasetRow) -> GroundingResult:
        expected_args: dict[str, Any] = row.expected.expected_arguments or {}
        return GroundingResult(
            grounding_mode="oracle",
            resolved_args=dict(expected_args),
            unresolved_ids=[],
            grounding_confidence=1.0 if expected_args else 0.5,
            candidates_examined=len(expected_args),
        )
