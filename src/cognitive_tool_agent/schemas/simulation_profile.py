from __future__ import annotations

from typing import Literal
from pydantic import BaseModel


class SimulationProfile(BaseModel):
    simulation_id: str
    task_id: str
    split: Literal["train", "dev", "test"] | None = None

    primary_scenario: str
    scenario_type: str
    is_multi_action: bool
    terminal_tool_fingerprint: str

    requires_grounding: bool
    requires_tool_chaining: bool
    has_item_ids: bool
    has_order_id: bool
    has_product_id: bool
    difficulty_bucket: str

    num_expected_actions: int
    num_tool_calls: int
