from __future__ import annotations

import re
from typing import Any

from ..adapters.base import AgentMode, ModelAdapter
from ..graph.node_input import NodeInput
from ..schemas.common import UserInput
from ..schemas.dataset import DatasetRow
from ..schemas.grounding import GroundingResult
from ..schemas.reason import ReasoningResult

_ID_PATTERN = re.compile(r"(^|_)(id|ids)$", re.IGNORECASE)


def _looks_like_id_field(name: str) -> bool:
    return bool(_ID_PATTERN.search(name))


class GroundingAgent:
    def __init__(
        self,
        mode: AgentMode = "stub",
        model_adapter: ModelAdapter | None = None,
    ) -> None:
        self.mode = mode
        self._adapter = model_adapter

    def run(self, ctx: NodeInput) -> GroundingResult:
        if self.mode == "disabled":
            return GroundingResult(grounding_mode="disabled")

        if self.mode == "oracle":
            return self._run_oracle(ctx)

        if self.mode == "llm":
            return self._run_llm(ctx)

        return self._run_stub(ctx)

    def _run_llm(self, ctx: NodeInput) -> GroundingResult:
        raise NotImplementedError("llm mode not yet implemented for GroundingAgent")

    def _run_stub(self, ctx: NodeInput) -> GroundingResult:
        user_input = ctx.user_input
        reasoning = ctx.reasoning
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

    def _run_oracle(self, ctx: NodeInput) -> GroundingResult:
        row = ctx.row
        expected_args: dict[str, Any] = row.expected.expected_arguments or {}
        return GroundingResult(
            grounding_mode="oracle",
            resolved_args=dict(expected_args),
            unresolved_ids=[],
            grounding_confidence=1.0 if expected_args else 0.5,
            candidates_examined=len(expected_args),
        )
