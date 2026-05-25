from typing import Any, Callable
from ..schemas.common import ToolSchema


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSchema] = {}
        self._handlers: dict[str, Callable[..., Any]] = {}

    def register(self, schema: ToolSchema, handler: Callable[..., Any]) -> None:
        self._tools[schema.name] = schema
        self._handlers[schema.name] = handler

    def lookup(self, name: str) -> ToolSchema | None:
        return self._tools.get(name)

    def call(self, name: str, arguments: dict[str, Any]) -> Any:
        if name not in self._handlers:
            raise KeyError(f"Tool not registered: {name!r}")
        return self._handlers[name](**arguments)

    def list_tools(self) -> list[ToolSchema]:
        return list(self._tools.values())

    def names(self) -> list[str]:
        return list(self._tools.keys())
