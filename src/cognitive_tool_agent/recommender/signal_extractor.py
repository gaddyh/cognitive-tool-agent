from __future__ import annotations

from typing import Any

from .thresholds import MIN_GROUNDING_INSTANCES


def extract_signals(
    report: dict[str, Any],
) -> tuple[dict[str, float], dict[str, str]]:
    """
    Derive numeric signals and string sources from a CognitiveDatasetReport dict.

    Returns:
      signals: dict[str, float] with keys:
        chaining_strength       — instances-weighted avg tool-chaining fraction
        grounding_strength      — instances-weighted avg grounding fraction
        peak_grounding_strength — max grounding fraction among args with >= MIN_GROUNDING_INSTANCES
        peak_grounding_instances — instance count of the peak grounding arg
        write_fraction          — write_calls / (read_calls + write_calls)
        avg_chain_depth         — mean avg_chain_depth across topology rows
        write_failure_fraction  — write-type failures / total failures

      sources: dict[str, str] with keys:
        peak_grounding_arg — name of the arg driving peak_grounding_strength (or "" if none)
    """
    emergence: list[dict] = report.get("argument_emergence", [])
    topology: list[dict] = report.get("cognitive_action_topology", [])
    heatmap: list[dict] = report.get("failure_heatmap", [])
    summary: dict = report.get("dataset_summary", {})

    # ── chaining_strength ────────────────────────────────────────────────────
    total_instances = sum(r.get("total_instances", 0) for r in emergence)
    if total_instances > 0:
        chaining_strength = sum(
            r.get("requires_tool_chaining_pct", 0.0) * r.get("total_instances", 0)
            for r in emergence
        ) / (100.0 * total_instances)
    else:
        chaining_strength = 0.0

    # ── grounding_strength (global weighted avg) ─────────────────────────────
    if total_instances > 0:
        grounding_strength = sum(
            r.get("requires_grounding_pct", 0.0) * r.get("total_instances", 0)
            for r in emergence
        ) / (100.0 * total_instances)
    else:
        grounding_strength = 0.0

    # ── peak_grounding_strength (local max over qualified args) ──────────────
    qualified = [r for r in emergence if r.get("total_instances", 0) >= MIN_GROUNDING_INSTANCES]
    if qualified:
        peak_row = max(qualified, key=lambda r: r.get("requires_grounding_pct", 0.0))
        peak_grounding_strength = peak_row.get("requires_grounding_pct", 0.0) / 100.0
        peak_grounding_instances = float(peak_row.get("total_instances", 0))
        peak_grounding_arg = peak_row.get("arg_name", "")
    else:
        peak_grounding_strength = 0.0
        peak_grounding_instances = 0.0
        peak_grounding_arg = ""

    # ── write_fraction ───────────────────────────────────────────────────────
    read_calls: int = summary.get("read_tool_calls", 0)
    write_calls: int = summary.get("write_tool_calls", 0)
    total_calls = read_calls + write_calls
    write_fraction = write_calls / total_calls if total_calls > 0 else 0.0

    # ── avg_chain_depth ──────────────────────────────────────────────────────
    depth_values = [r.get("avg_chain_depth", 0.0) for r in topology if r.get("avg_chain_depth") is not None]
    avg_chain_depth = sum(depth_values) / len(depth_values) if depth_values else 0.0

    # ── write_failure_fraction ───────────────────────────────────────────────
    rw_rows = [r for r in heatmap if r.get("dimension") == "read_write"]
    total_failures = sum(r.get("count", 0) for r in rw_rows)
    write_failures = sum(
        r.get("count", 0) for r in rw_rows if str(r.get("value", "")).lower() == "write"
    )
    write_failure_fraction = write_failures / total_failures if total_failures > 0 else 0.0

    signals: dict[str, float] = {
        "chaining_strength": round(chaining_strength, 4),
        "grounding_strength": round(grounding_strength, 4),
        "peak_grounding_strength": round(peak_grounding_strength, 4),
        "peak_grounding_instances": peak_grounding_instances,
        "write_fraction": round(write_fraction, 4),
        "avg_chain_depth": round(avg_chain_depth, 4),
        "write_failure_fraction": round(write_failure_fraction, 4),
    }
    sources: dict[str, str] = {
        "peak_grounding_arg": peak_grounding_arg,
    }
    return signals, sources
