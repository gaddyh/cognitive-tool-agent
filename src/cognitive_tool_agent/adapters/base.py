from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel

AgentMode = Literal["stub", "llm", "oracle", "disabled", "deterministic"]


class LLMResult(BaseModel):
    parsed: Any | None
    raw: Any | None = None
    usage: dict[str, Any] = {}
    latency_ms: float | None = None
    model: str | None = None
    provider: str | None = None


@runtime_checkable
class ModelAdapter(Protocol):
    def complete(self, prompt: str, output_schema: type) -> LLMResult: ...
