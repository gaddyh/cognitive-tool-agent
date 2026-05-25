from __future__ import annotations

from ..schemas.simulation import SimulationFile
from ..schemas.trace_converter import ToolRegistryEntry


def scan_tool_registry(sim_file: SimulationFile) -> dict[str, ToolRegistryEntry]:
    """Build a tool registry from expected actions and actual tool calls."""
    registry: dict[str, ToolRegistryEntry] = {}

    def _get_or_create(name: str) -> ToolRegistryEntry:
        if name not in registry:
            registry[name] = ToolRegistryEntry(name=name)
        return registry[name]

    for task in sim_file.tasks:
        for action in task.expected_actions():
            entry = _get_or_create(action.name)
            for arg in action.arguments:
                if arg not in entry.required_args:
                    entry.required_args.append(arg)
            if action.tool_type and not entry.tool_type:
                entry.tool_type = action.tool_type

    for sim in sim_file.simulations:
        for ac in sim.reward_info.action_checks:
            entry = _get_or_create(ac.action.name)
            if ac.tool_type and not entry.tool_type:
                entry.tool_type = ac.tool_type

        for msg in sim.messages:
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    entry = _get_or_create(tc.name)
                    entry.usage_count += 1
                    for arg in tc.arguments:
                        if arg not in entry.seen_args:
                            entry.seen_args.append(arg)

    return registry
