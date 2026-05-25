from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..schemas.common import ToolSchema
from ..schemas.dataset import DatasetRow, ExpectedBehavior
from ..schemas.trace_converter import ActionSequenceRow, AlignedAction, TurnSupervisionRow
from ..tools.registry import ToolRegistry

_WRITE_VERBS = frozenset(
    {"modify", "cancel", "exchange", "update", "return", "create", "delete", "transfer"}
)


def _is_write_action(tool_name: str) -> bool:
    name_lower = tool_name.lower()
    return any(verb in name_lower for verb in _WRITE_VERBS)


def _select_primary_action(
    aligned_actions: list[AlignedAction],
) -> tuple[AlignedAction, str]:
    write_actions = [a for a in aligned_actions if _is_write_action(a.expected_tool)]
    if write_actions:
        return write_actions[-1], "last_write_action"
    return aligned_actions[-1], "last_action_fallback"


class ActionSequenceAdapter:
    def load(
        self,
        action_seq_path: Path,
        turn_sup_path: Path,
        tool_registry_path: Path,
    ) -> tuple[list[DatasetRow], ToolRegistry]:
        action_rows = self._load_action_sequence(action_seq_path)
        user_messages = self._build_user_message_map(turn_sup_path)
        registry = self._build_registry(tool_registry_path)
        dataset_rows = self._convert(action_rows, user_messages, registry)
        return dataset_rows, registry

    def _load_action_sequence(self, path: Path) -> list[ActionSequenceRow]:
        rows: list[ActionSequenceRow] = []
        with open(path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                    rows.append(ActionSequenceRow.model_validate(raw))
                except Exception as exc:
                    raise ValueError(
                        f"Failed to parse action_sequence line {line_num}: {exc}"
                    ) from exc
        return rows

    def _build_user_message_map(self, path: Path) -> dict[str, str]:
        sim_to_first_user: dict[str, str] = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                    row = TurnSupervisionRow.model_validate(raw)
                    if row.role == "user" and row.simulation_id not in sim_to_first_user:
                        if row.content:
                            sim_to_first_user[row.simulation_id] = row.content
                except Exception:
                    continue
        return sim_to_first_user

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
        action_rows: list[ActionSequenceRow],
        user_messages: dict[str, str],
        registry: ToolRegistry,
    ) -> list[DatasetRow]:
        dataset_rows: list[DatasetRow] = []

        for seq in action_rows:
            if not seq.aligned_actions:
                continue

            primary_action, selection_reason = _select_primary_action(seq.aligned_actions)
            user_message = user_messages.get(
                seq.simulation_id,
                f"Task {seq.task_id}: execute {primary_action.expected_tool}",
            )

            all_tools = list(
                dict.fromkeys(
                    a.expected_tool for a in seq.aligned_actions if a.expected_tool
                )
            )

            expected = ExpectedBehavior(
                expected_action="tool_executed",
                expected_tool=primary_action.expected_tool,
                expected_arguments=primary_action.expected_arguments or None,
            )

            world_state: dict[str, Any] = {
                "simulation_id": seq.simulation_id,
                "task_id": seq.task_id,
                "action_count": len(seq.aligned_actions),
                "primary_action_selection_reason": selection_reason,
                "primary_tool": primary_action.expected_tool,
            }

            row = DatasetRow(
                id=f"{seq.task_id}:{seq.simulation_id[:8]}",
                user_message=user_message,
                tools=all_tools,
                world_state=world_state,
                expected=expected,
            )
            dataset_rows.append(row)

        return dataset_rows
