from __future__ import annotations

import json
import re
from typing import Any

from ..adapters.base import AgentMode, ModelAdapter
from ..graph.node_input import NodeInput
from ..schemas.common import UserInput
from ..schemas.dataset import DatasetRow
from ..schemas.grounding import GroundingResult
from ..schemas.reason import ReasoningResult

_ID_PATTERN = re.compile(r"(^|_)(id|ids)$", re.IGNORECASE)
_ORDER_ID_PATTERN = re.compile(r"#W\d+")
_USER_ID_PATTERN = re.compile(r"^\w+_\w+_\d+$")


def _looks_like_id_field(name: str) -> bool:
    return bool(_ID_PATTERN.search(name))


def _parse_result_content(content: str) -> Any:
    try:
        return json.loads(content)
    except (ValueError, TypeError):
        return content


def _deep_get(obj: Any, key: str) -> Any | None:
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            found = _deep_get(v, key)
            if found is not None:
                return found
    if isinstance(obj, list):
        for item in obj:
            found = _deep_get(item, key)
            if found is not None:
                return found
    return None


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

        if self.mode == "deterministic":
            return self._run_deterministic(ctx)

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

    def _run_deterministic(self, ctx: NodeInput) -> GroundingResult:
        """Resolve scalar ID fields from prior tool context and user message.

        Allowed evidence:
            - ctx.user_input.message
            - ctx.row.world_state["prior_tool_calls"]
            - ctx.row.world_state["prior_tool_results"]
            - selected tool schema required_fields

        Never touches ctx.row.expected.expected_arguments or expected_tool.
        """
        user_input = ctx.user_input
        reasoning = ctx.reasoning
        ws = ctx.row.world_state

        selected_tool = (
            (reasoning.selected_tool if reasoning else None)
            or ws.get("primary_tool")
        )
        tool_schema = next(
            (t for t in user_input.available_tools if t.name == selected_tool),
            None,
        )
        if not tool_schema:
            return GroundingResult(grounding_mode="deterministic")

        prior_tool_calls: list[dict] = ws.get("prior_tool_calls", [])
        prior_tool_results: list[dict] = ws.get("prior_tool_results", [])
        user_message: str = user_input.message or ""

        parsed_results: list[Any] = [
            _parse_result_content(r["content"])
            for r in prior_tool_results
            if r.get("content")
        ]

        product_catalog: dict[str, str] = {}
        for parsed in parsed_results:
            if isinstance(parsed, dict) and any(
                isinstance(v, str) and v.isdigit() and len(v) >= 9
                for v in parsed.values()
            ):
                product_catalog.update(
                    {k.lower(): v for k, v in parsed.items() if isinstance(v, str)}
                )

        resolved_args: dict[str, Any] = {}

        for field in tool_schema.required_fields:
            if field in ("item_ids", "new_item_ids"):
                continue

            value = self._resolve_field_deterministic(
                field, prior_tool_calls, parsed_results, user_message, product_catalog
            )
            if value is not None:
                resolved_args[field] = value

        unresolved = [
            f
            for f in tool_schema.required_fields
            if f not in resolved_args and f not in ("item_ids", "new_item_ids")
        ]
        total = len(tool_schema.required_fields)
        confidence = len(resolved_args) / total if total > 0 else 0.5

        return GroundingResult(
            grounding_mode="deterministic",
            resolved_args=resolved_args,
            unresolved_ids=unresolved,
            grounding_confidence=round(confidence, 3),
            candidates_examined=total,
        )

    def _resolve_field_deterministic(
        self,
        field: str,
        prior_tool_calls: list[dict],
        parsed_results: list[Any],
        user_message: str,
        product_catalog: dict[str, str],
    ) -> Any | None:
        if field == "order_id":
            for tc in reversed(prior_tool_calls):
                if "order_id" in tc.get("arguments", {}):
                    return tc["arguments"]["order_id"]
            for parsed in reversed(parsed_results):
                val = _deep_get(parsed, "order_id")
                if val is not None:
                    return val
            m = _ORDER_ID_PATTERN.search(user_message)
            if m:
                return m.group(0)
            return None

        if field == "user_id":
            for tc in reversed(prior_tool_calls):
                if "user_id" in tc.get("arguments", {}):
                    return tc["arguments"]["user_id"]
            for parsed in reversed(parsed_results):
                val = _deep_get(parsed, "user_id")
                if val is not None:
                    return val
            for parsed in reversed(parsed_results):
                if isinstance(parsed, str) and _USER_ID_PATTERN.match(parsed.strip()):
                    return parsed.strip()
            return None

        if field == "product_id":
            for tc in reversed(prior_tool_calls):
                if "product_id" in tc.get("arguments", {}):
                    return tc["arguments"]["product_id"]
            for parsed in reversed(parsed_results):
                val = _deep_get(parsed, "product_id")
                if val is not None:
                    return val
            if product_catalog:
                message_lower = user_message.lower()
                for product_name, product_id in product_catalog.items():
                    if product_name and product_name in message_lower:
                        return product_id
            return None

        if field == "payment_method_id":
            for tc in reversed(prior_tool_calls):
                if "payment_method_id" in tc.get("arguments", {}):
                    return tc["arguments"]["payment_method_id"]
            for parsed in reversed(parsed_results):
                val = _deep_get(parsed, "payment_method_id")
                if val is not None:
                    return val
            return None

        if _looks_like_id_field(field):
            for tc in reversed(prior_tool_calls):
                if field in tc.get("arguments", {}):
                    return tc["arguments"][field]
            for parsed in reversed(parsed_results):
                val = _deep_get(parsed, field)
                if val is not None:
                    return val
            return None

        return None

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
