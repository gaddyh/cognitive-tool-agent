from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..schemas.common import ToolSchema
from ..schemas.dataset import DatasetRow, ExpectedBehavior
from ..schemas.trace_converter import TurnSupervisionRow
from ..tools.registry import ToolRegistry


class TurnToolCallAdapter:
    def load(
        self,
        turn_sup_path: Path,
        tool_registry_path: Path,
    ) -> tuple[list[DatasetRow], ToolRegistry]:
        turns = self._load_turns(turn_sup_path)
        registry = self._build_registry(tool_registry_path)
        rows = self._convert(turns, registry)
        return rows, registry

    def _load_turns(self, path: Path) -> list[TurnSupervisionRow]:
        rows: list[TurnSupervisionRow] = []
        with open(path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(TurnSupervisionRow.model_validate_json(line))
                except Exception as exc:
                    raise ValueError(
                        f"Failed to parse turn_supervision line {line_num}: {exc}"
                    ) from exc
        return rows

    def _build_registry(self, path: Path) -> ToolRegistry:
        with open(path, encoding="utf-8") as f:
            raw: dict[str, Any] = json.load(f)

        registry = ToolRegistry()
        for name, entry in raw.items():
            required_args: list[str] = entry.get("required_args", [])
            seen_args: list[str] = entry.get("seen_args", [])
            optional_args = [a for a in seen_args if a not in required_args]
            schema = ToolSchema(
                name=name,
                description=f"Tool: {name}",
                required_fields=required_args,
                optional_fields=optional_args,
                properties={arg: {"type": "string"} for arg in seen_args},
            )
            registry.register(schema, lambda **kwargs: {"status": "ok"})
        return registry

    def _convert(
        self,
        turns: list[TurnSupervisionRow],
        registry: ToolRegistry,
    ) -> list[DatasetRow]:
        rows: list[DatasetRow] = []

        last_user_by_sim: dict[str, str] = {}
        all_tool_names = registry.names()

        for turn in sorted(turns, key=lambda t: (t.simulation_id, t.turn_idx)):
            if turn.role == "user" and turn.content:
                last_user_by_sim[turn.simulation_id] = turn.content
                continue

            if turn.role != "assistant":
                continue

            label = turn.cognitive_label
            if label.plan_next_action != "call_tool":
                continue
            if not label.plan_tool_name:
                continue

            user_message = last_user_by_sim.get(
                turn.simulation_id,
                turn.content or f"Execute {label.plan_tool_name}",
            )

            expected = ExpectedBehavior(
                expected_action="tool_executed",
                expected_tool=label.plan_tool_name,
                expected_arguments=label.plan_arguments or None,
            )

            row = DatasetRow(
                id=f"{turn.task_id}:{turn.simulation_id[:8]}:turn:{turn.turn_idx}",
                user_message=user_message,
                tools=all_tool_names,
                world_state={
                    "simulation_id": turn.simulation_id,
                    "task_id": turn.task_id,
                    "turn_idx": turn.turn_idx,
                    "source": "turn_tool_call",
                    "primary_tool": label.plan_tool_name,
                },
                expected=expected,
            )
            rows.append(row)

        return rows
