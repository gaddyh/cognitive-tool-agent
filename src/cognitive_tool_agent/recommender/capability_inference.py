from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..schemas.recommender import CapabilityInferenceResult, CapabilityRequirement
from .signal_extractor import extract_signals
from .thresholds import (
    GROUNDING_THRESHOLD,
    MEMORY_CHAINING_THRESHOLD,
    READINESS_WRITE_FRACTION_THRESHOLD,
    REASONING_DEPTH_THRESHOLD,
)


def _load_report(report: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(report, dict):
        return report
    path = Path(report)
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _top_chaining_arg(emergence: list[dict]) -> str | None:
    ranked = sorted(
        emergence,
        key=lambda r: r.get("requires_tool_chaining_pct", 0.0) * r.get("total_instances", 0),
        reverse=True,
    )
    return ranked[0]["arg_name"] if ranked else None


class CapabilityInferenceEngine:
    def run(self, report: dict[str, Any] | str | Path) -> CapabilityInferenceResult:
        report_dict = _load_report(report)
        signals, sources = extract_signals(report_dict)
        emergence: list[dict] = report_dict.get("argument_emergence", [])
        heatmap: list[dict] = report_dict.get("failure_heatmap", [])

        # ── memory ───────────────────────────────────────────────────────────
        memory_strength = min(signals["chaining_strength"], 1.0)
        memory_required = memory_strength > MEMORY_CHAINING_THRESHOLD
        memory_evidence: list[str] = []
        top_chain_arg = _top_chaining_arg(emergence)
        if top_chain_arg:
            pct = next(
                (r.get("requires_tool_chaining_pct", 0.0) for r in emergence if r["arg_name"] == top_chain_arg),
                0.0,
            )
            memory_evidence.append(f"{pct:.1f}% of {top_chain_arg!r} values are tool-chained")
        if not memory_evidence:
            memory_evidence.append(f"chaining_strength={memory_strength:.2f}")

        # ── grounding ────────────────────────────────────────────────────────
        effective_grounding = min(
            max(signals["grounding_strength"], signals["peak_grounding_strength"]), 1.0
        )
        grounding_required = effective_grounding > GROUNDING_THRESHOLD
        peak_arg = sources.get("peak_grounding_arg", "")
        peak_pct = round(signals["peak_grounding_strength"] * 100, 1)
        peak_n = int(signals["peak_grounding_instances"])
        grounding_evidence: list[str] = []
        if peak_arg:
            grounding_evidence.append(
                f"{peak_pct:.1f}% of {peak_arg!r} values require grounding (peak arg, {peak_n} instances)"
            )
            grounding_evidence.append(
                f"global grounding_strength={signals['grounding_strength']:.3f} "
                f"(diluted by high-volume zero-grounding args)"
            )
        else:
            grounding_evidence.append(f"grounding_strength={effective_grounding:.2f}")

        # ── readiness ────────────────────────────────────────────────────────
        readiness_strength = min(
            max(signals["write_fraction"], signals["write_failure_fraction"]), 1.0
        )
        readiness_required = readiness_strength > READINESS_WRITE_FRACTION_THRESHOLD
        rw_rows = [r for r in heatmap if r.get("dimension") == "read_write"]
        total_failures = sum(r.get("count", 0) for r in rw_rows)
        write_failures = sum(
            r.get("count", 0) for r in rw_rows if str(r.get("value", "")).lower() == "write"
        )
        readiness_evidence: list[str] = [
            f"write actions comprise {write_failures} of {total_failures} failures",
            f"write_fraction={signals['write_fraction']:.2f}",
        ]

        # ── deep_planning ────────────────────────────────────────────────────
        max_normalised_depth = REASONING_DEPTH_THRESHOLD * 2
        deep_planning_strength = min(signals["avg_chain_depth"] / max_normalised_depth, 1.0)
        deep_planning_required = signals["avg_chain_depth"] > REASONING_DEPTH_THRESHOLD
        deep_planning_evidence: list[str] = [
            f"avg chain depth is {signals['avg_chain_depth']:.2f} across tools"
        ]

        capabilities: dict[str, CapabilityRequirement] = {
            "memory": CapabilityRequirement(
                required=memory_required,
                strength=round(memory_strength, 3),
                evidence=memory_evidence,
            ),
            "grounding": CapabilityRequirement(
                required=grounding_required,
                strength=round(effective_grounding, 3),
                evidence=grounding_evidence,
            ),
            "readiness": CapabilityRequirement(
                required=readiness_required,
                strength=round(readiness_strength, 3),
                evidence=readiness_evidence,
            ),
            "deep_planning": CapabilityRequirement(
                required=deep_planning_required,
                strength=round(deep_planning_strength, 3),
                evidence=deep_planning_evidence,
            ),
        }

        return CapabilityInferenceResult(
            required_capabilities=capabilities,
            raw_signals=signals,
            signal_sources=sources,
        )
