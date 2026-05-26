"""OpenAI structured-output adapter for grounding.

Reads OPENAI_API_KEY from the environment.
Uses response_format with json_schema to get structured JSON back.
"""
from __future__ import annotations

import os
import time
from typing import Any

from .base import LLMResult, ModelAdapter


_COST_PER_INPUT_TOKEN = {
    "gpt-4o-mini": 0.00000015,
    "gpt-4o": 0.0000025,
}
_COST_PER_OUTPUT_TOKEN = {
    "gpt-4o-mini": 0.00000060,
    "gpt-4o": 0.0000100,
}


class OpenAIGroundingAdapter:
    """ModelAdapter backed by OpenAI chat completions with JSON schema output."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> None:
        self.model = model
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._temperature = temperature
        self._max_tokens = max_tokens

        try:
            import openai
            self._client = openai.OpenAI(api_key=self._api_key)
        except ImportError as exc:
            raise ImportError(
                "openai package is required: pip install openai>=1.0"
            ) from exc

    def complete(self, prompt: str, output_schema: type) -> LLMResult:
        import openai

        json_schema = output_schema.model_json_schema()
        schema_name = output_schema.__name__

        start = time.monotonic()
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a grounding function in a tool-calling evaluation harness. "
                            "Your job: resolve tool argument fields from the provided conversation "
                            "and state snapshot. "
                            "You MUST: use only provided state, never invent IDs, mark missing values "
                            "as missing, mark multiple valid candidates as ambiguous, return valid JSON "
                            "matching the schema, include evidence for each grounded value."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        # strict=True is intentionally disabled because LLMGroundingOutput.fields
                        # is a dict[str, GroundedField]. OpenAI strict structured outputs require
                        # closed object schemas with additionalProperties=false, while this behavior
                        # space uses dynamic field names that vary by selected tool.
                        # We still validate with Pydantic after generation and record schema_valid_rate.
                        "strict": False,
                        "schema": json_schema,
                    },
                },
            )
        except openai.APIError as exc:
            raise RuntimeError(f"OpenAI API error: {exc}") from exc

        latency_ms = (time.monotonic() - start) * 1000.0

        choice = response.choices[0]
        raw_text = choice.message.content or ""

        usage = response.usage
        usage_dict: dict[str, Any] = {}
        cost_usd: float | None = None
        if usage:
            usage_dict = {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            }
            in_cost = _COST_PER_INPUT_TOKEN.get(self.model, 0.0)
            out_cost = _COST_PER_OUTPUT_TOKEN.get(self.model, 0.0)
            cost_usd = (
                usage.prompt_tokens * in_cost
                + usage.completion_tokens * out_cost
            )

        try:
            import json
            parsed = output_schema.model_validate(json.loads(raw_text))
        except Exception as exc:
            e = RuntimeError(f"Failed to parse LLM response as {schema_name}: {exc}")
            e.raw_text = raw_text  # type: ignore[attr-defined]
            raise e

        return LLMResult(
            parsed=parsed,
            raw=raw_text,
            usage=usage_dict,
            latency_ms=round(latency_ms, 2),
            model=self.model,
            provider="openai",
        )
