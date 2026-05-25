from __future__ import annotations

from ..schemas.simulation import RawSimulation, RawTask
from ..schemas.simulation_profile import SimulationProfile

_LOOKUP_TOOLS: frozenset[str] = frozenset({
    "find_user_id_by_email",
    "find_user_id_by_name_zip",
    "get_user_details",
    "get_order_details",
    "get_product_details",
    "list_all_product_types",
})

_TERMINAL_FAMILY: dict[str, str] = {
    "cancel_pending_order": "cancel",
    "modify_pending_order_items": "modify_order_items",
    "modify_pending_order_address": "modify_address",
    "return_delivered_order_items": "return",
    "exchange_delivered_order_items": "exchange",
    "modify_user_address": "update_address",
    "transfer_to_human_agents": "transfer",
}


def profile_simulation(
    sim: RawSimulation,
    task: RawTask,
    num_tool_calls: int,
) -> SimulationProfile:
    expected_actions = task.expected_actions()
    num_expected_actions = len(expected_actions)

    terminal_tools: list[str] = []
    terminal_families_seen: list[str] = []

    for action in expected_actions:
        tool = action.name
        if tool in _LOOKUP_TOOLS:
            continue
        terminal_tools.append(tool)
        family = _TERMINAL_FAMILY.get(tool, tool)
        if family not in terminal_families_seen:
            terminal_families_seen.append(family)

    if terminal_families_seen:
        primary_scenario = terminal_families_seen[-1]
        is_multi_action = len(terminal_families_seen) > 1
        unique_terminal_tools = sorted(set(terminal_tools))
        terminal_tool_fingerprint = "+".join(unique_terminal_tools)
    else:
        primary_scenario = "lookup_only"
        is_multi_action = False
        terminal_tool_fingerprint = "none"

    scenario_type = (
        f"{primary_scenario}|{'multi_action' if is_multi_action else 'single_action'}"
    )

    all_args: set[str] = set()
    for action in expected_actions:
        all_args.update(action.arguments.keys())

    has_item_ids = bool(all_args & {"item_ids", "new_item_ids"})
    has_order_id = "order_id" in all_args
    has_product_id = "product_id" in all_args
    requires_grounding = has_item_ids or has_order_id or has_product_id
    requires_tool_chaining = num_expected_actions >= 3

    if has_item_ids:
        difficulty_bucket = "hard"
    elif has_order_id or has_product_id:
        difficulty_bucket = "medium"
    else:
        difficulty_bucket = "easy"

    return SimulationProfile(
        simulation_id=sim.id,
        task_id=sim.task_id,
        split=None,
        primary_scenario=primary_scenario,
        scenario_type=scenario_type,
        is_multi_action=is_multi_action,
        terminal_tool_fingerprint=terminal_tool_fingerprint,
        requires_grounding=requires_grounding,
        requires_tool_chaining=requires_tool_chaining,
        has_item_ids=has_item_ids,
        has_order_id=has_order_id,
        has_product_id=has_product_id,
        difficulty_bucket=difficulty_bucket,
        num_expected_actions=num_expected_actions,
        num_tool_calls=num_tool_calls,
    )
