"""Extract tau-bench timing metadata from raw simulation dicts.

Operates on the raw JSON dicts before normalization to avoid polluting
RawSimulation / SimulationMessage schemas with timing fields.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ..schemas.simulation_timing import SimulationTiming
from ..schemas.simulation_profile import SimulationProfile


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def extract_timings(
    raw_simulations: list[dict[str, Any]],
    profile_map: dict[str, SimulationProfile],
) -> list[SimulationTiming]:
    """Build one SimulationTiming per simulation from raw JSON dicts.

    Args:
        raw_simulations: The list of raw simulation dicts from the source JSON.
        profile_map:     simulation_id → SimulationProfile (used to attach split).
    """
    results: list[SimulationTiming] = []

    for sim in raw_simulations:
        sim_id: str = sim.get("id", "")
        profile = profile_map.get(sim_id)
        split = profile.split if profile else None

        tau_duration: float | None = sim.get("duration")

        messages: list[dict[str, Any]] = sim.get("messages") or []

        gen_times: list[float] = [
            m["generation_time_seconds"]
            for m in messages
            if m.get("generation_time_seconds") is not None
        ]
        gen_total = sum(gen_times) if gen_times else None
        gen_avg = (gen_total / len(gen_times)) if gen_times else None
        gen_turns = len(gen_times)

        timestamps = [_parse_iso(m.get("timestamp")) for m in messages]
        timestamps = [t for t in timestamps if t is not None]
        if len(timestamps) >= 2:
            span_seconds = (max(timestamps) - min(timestamps)).total_seconds()
        else:
            span_seconds = None

        results.append(SimulationTiming(
            simulation_id=sim_id,
            split=split,
            tau_duration_seconds=round(tau_duration, 6) if tau_duration is not None else None,
            tau_agent_generation_time_seconds_total=round(gen_total, 6) if gen_total is not None else None,
            tau_agent_generation_time_seconds_avg_per_assistant_turn=round(gen_avg, 6) if gen_avg is not None else None,
            tau_agent_generation_turns=gen_turns,
            message_span_seconds=round(span_seconds, 6) if span_seconds is not None else None,
        ))

    return results
