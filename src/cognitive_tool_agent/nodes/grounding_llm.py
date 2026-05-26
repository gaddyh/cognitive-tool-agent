"""LLM-powered grounding node.

Resolves tool argument fields from conversation context and available state.
The prompt builder takes explicit named arguments — never the whole row —
to prevent accidental target_args leakage.

The LLM output (LLMGroundingOutput) is saved as the rich diagnostic artifact.
The downstream graph receives a normalized GroundingResult via to_grounding_result().
"""
from __future__ import annotations

import json
from typing import Any

from ..adapters.base import ModelAdapter
from ..schemas.grounding import GroundingResult
from ..schemas.grounding_llm import LLMGroundingOutput


_EVIDENCE_INSTRUCTION = (
    "For each field you ground, provide specific evidence strings referencing "
    "the exact source (e.g., 'order #W1234 found in prior get_order_details result')."
)


def build_grounding_prompt(
    *,
    user_message: str,
    conversation_context: list[str],
    selected_tool: str,
    tool_schema: dict[str, Any],
    available_state: dict[str, Any],
    current_deterministic_args: dict[str, Any],
) -> str:
    """Build the user-turn prompt for the grounding LLM.

    Args are explicit — target_args is intentionally absent.
    """
    payload = {
        "user_message": user_message,
        "conversation_context": conversation_context[-10:],
        "selected_tool": selected_tool,
        "tool_schema": tool_schema,
        "available_state": available_state,
        "current_deterministic_args": current_deterministic_args,
    }
    return (
        "Resolve the required argument fields for the selected tool.\n\n"
        f"{_EVIDENCE_INSTRUCTION}\n\n"
        "Input:\n"
        f"```json\n{json.dumps(payload, indent=2, ensure_ascii=False)}\n```\n\n"
        "Respond with a JSON object matching the LLMGroundingOutput schema. "
        "Fields not relevant to this tool should have status 'not_applicable'."
    )


class GroundingLLMNodeResult:
    """Carries both the rich LLM artifact and the runtime-compatible projection."""

    def __init__(
        self,
        llm_output: LLMGroundingOutput,
        grounding_result: GroundingResult,
        llm_raw: dict[str, Any] | None,
        confidence: float | None,
        latency_ms: float | None,
        cost_usd: float | None,
    ) -> None:
        self.llm_output = llm_output
        self.resolved_args = grounding_result.resolved_args
        self.grounding_result = grounding_result
        self.llm_raw = llm_raw
        self.confidence = confidence
        self.latency_ms = latency_ms
        self.cost_usd = cost_usd


class GroundingLLMNode:
    """LLM-powered grounding node.

    Can be used standalone (Pass 2 offline evaluation) or inside the graph
    (future Pass 3 integration).
    """

    def __init__(self, adapter: ModelAdapter) -> None:
        self._adapter = adapter

    def run_from_grounding_row(self, row: dict[str, Any]) -> GroundingLLMNodeResult:
        """Run grounding from a grounding_eval JSONL row.

        Explicit field extraction — target_args is intentionally not passed
        to build_grounding_prompt.
        """
        prompt = build_grounding_prompt(
            user_message=row.get("user_message", ""),
            conversation_context=row.get("conversation_context") or [],
            selected_tool=row.get("selected_tool", ""),
            tool_schema=row.get("tool_schema") or {},
            available_state=row.get("available_state") or {},
            current_deterministic_args=row.get("current_deterministic_args") or {},
        )

        llm_result = self._adapter.complete(prompt, LLMGroundingOutput)
        llm_output: LLMGroundingOutput = llm_result.parsed
        grounding_result = llm_output.to_grounding_result()

        raw_dict: dict[str, Any] | None = None
        if llm_result.raw:
            try:
                raw_dict = json.loads(llm_result.raw) if isinstance(llm_result.raw, str) else llm_result.raw
            except (ValueError, TypeError):
                raw_dict = {"raw_text": llm_result.raw}

        cost_usd: float | None = None
        usage = llm_result.usage or {}
        if "prompt_tokens" in usage and "completion_tokens" in usage:
            pass

        return GroundingLLMNodeResult(
            llm_output=llm_output,
            grounding_result=grounding_result,
            llm_raw=raw_dict,
            confidence=llm_output.confidence,
            latency_ms=llm_result.latency_ms,
            cost_usd=_extract_cost(llm_result),
        )


def _extract_cost(llm_result) -> float | None:
    usage = llm_result.usage or {}
    if not usage:
        return None
    from ..adapters.openai_grounding_adapter import _COST_PER_INPUT_TOKEN, _COST_PER_OUTPUT_TOKEN
    model = llm_result.model or ""
    in_cost = _COST_PER_INPUT_TOKEN.get(model, 0.0)
    out_cost = _COST_PER_OUTPUT_TOKEN.get(model, 0.0)
    return (
        usage.get("prompt_tokens", 0) * in_cost
        + usage.get("completion_tokens", 0) * out_cost
    ) or None
