from __future__ import annotations

from pydantic import BaseModel


class SimulationTiming(BaseModel):
    """Per-simulation timing metadata extracted from raw tau-bench traces.

    These fields are orthogonal to cognitive behavior extraction and should
    not appear in the grounding eval dataset. Use as a join key via simulation_id.

    Naming conventions:
        tau_duration_seconds        — simulation.duration reported by tau-bench framework
        tau_agent_generation_*      — sum/avg of message.generation_time_seconds (assistant turns only)
        message_span_seconds        — ISO timestamp span from first to last message
                                      (includes user/tool/framework overhead; NOT equal to duration)
    """

    simulation_id: str
    split: str | None = None

    tau_duration_seconds: float | None = None
    tau_agent_generation_time_seconds_total: float | None = None
    tau_agent_generation_time_seconds_avg_per_assistant_turn: float | None = None
    tau_agent_generation_turns: int = 0
    message_span_seconds: float | None = None
