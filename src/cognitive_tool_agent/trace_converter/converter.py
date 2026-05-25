from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..schemas.simulation import SimulationFile
from ..schemas.simulation_profile import SimulationProfile
from ..schemas.trace_converter import (
    ActionSequenceRow,
    ConversionSummary,
    FailureRow,
    ToolRegistryEntry,
    TurnSupervisionRow,
)
from .action_aligner import align_actions
from .failure_extractor import extract_failures
from .scenario_profiler import profile_simulation
from .simulation_loader import load_simulation_file
from .stratified_splitter import assign_splits, stratified_split
from .tool_registry_scanner import scan_tool_registry
from .turn_supervisor import supervise_turns

_SPLIT_VERSION = "scenario_stratified_grounding_v1"
_SPLIT_SEED = 42
_SPLIT_RATIOS = (0.6, 0.2, 0.2)


class TraceConverter:
    """
    Orchestrates the full Trace-to-Cognitive-Dataset conversion.

    Produces five core output files:
      tool_registry.json
      action_sequence.jsonl
      turn_supervision.jsonl        (enriched with split + scenario fields)
      failure_rows.jsonl
      conversion_summary.json

    Plus ten new split artifacts:
      simulation_profiles.jsonl
      scenario_distribution.json
      split_manifest.json
      splits/train_supervision.jsonl
      splits/dev_supervision.jsonl
      splits/test_supervision.jsonl
      splits/train_simulation_ids.json
      splits/dev_simulation_ids.json
      splits/test_simulation_ids.json
      (split_report.md written by build_split_report.py)
    """

    def run(self, input_path: Path | str, out_dir: Path | str) -> ConversionSummary:
        input_path = Path(input_path)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        splits_dir = out_dir / "splits"
        splits_dir.mkdir(parents=True, exist_ok=True)

        sim_file = load_simulation_file(input_path)
        task_map = {t.id: t for t in sim_file.tasks}

        tool_registry = scan_tool_registry(sim_file)

        all_sequences: list[ActionSequenceRow] = []
        all_supervision: list[TurnSupervisionRow] = []
        all_failures: list[FailureRow] = []
        profiles_raw: list[SimulationProfile] = []

        messages_count = 0
        actual_tool_calls_count = 0

        for sim in sim_file.simulations:
            task = task_map.get(sim.task_id)
            if task is None:
                continue

            messages_count += len(sim.messages)
            sim_tool_calls = 0
            for msg in sim.messages:
                if msg.tool_calls:
                    sim_tool_calls += len(msg.tool_calls)
                    actual_tool_calls_count += len(msg.tool_calls)

            sequence = align_actions(sim, task)
            all_sequences.append(sequence)

            supervision = supervise_turns(sim, task)
            all_supervision.extend(supervision)

            failures = extract_failures(sim, sequence)
            all_failures.extend(failures)

            profile = profile_simulation(sim, task, num_tool_calls=sim_tool_calls)
            profiles_raw.append(profile)

        assignments = stratified_split(profiles_raw, ratios=_SPLIT_RATIOS, seed=_SPLIT_SEED)
        profiles = assign_splits(profiles_raw, assignments)
        profile_map: dict[str, SimulationProfile] = {p.simulation_id: p for p in profiles}

        for row in all_supervision:
            p = profile_map.get(row.simulation_id)
            if p is not None:
                row.split = p.split
                row.scenario_type = p.scenario_type
                row.primary_scenario = p.primary_scenario
                row.is_multi_action = p.is_multi_action
                row.requires_grounding = p.requires_grounding
                row.difficulty_bucket = p.difficulty_bucket

        expected_actions_count = sum(
            len(t.expected_actions()) for t in sim_file.tasks
        )
        matched_count = sum(
            1 for seq in all_sequences
            for aa in seq.aligned_actions
            if aa.action_match
        )
        failed_count = len(all_failures)

        summary = ConversionSummary(
            tasks_count=len(sim_file.tasks),
            simulations_count=len(sim_file.simulations),
            messages_count=messages_count,
            expected_actions_count=expected_actions_count,
            actual_tool_calls_count=actual_tool_calls_count,
            matched_actions_count=matched_count,
            failed_actions_count=failed_count,
        )

        _write_json(out_dir / "tool_registry.json", _registry_to_dict(tool_registry))
        _write_jsonl(out_dir / "action_sequence.jsonl", [s.model_dump() for s in all_sequences])
        _write_jsonl(out_dir / "turn_supervision.jsonl", [r.model_dump() for r in all_supervision])
        _write_jsonl(out_dir / "failure_rows.jsonl", [f.model_dump() for f in all_failures])
        _write_json(out_dir / "conversion_summary.json", summary.model_dump())

        _write_jsonl(
            out_dir / "simulation_profiles.jsonl",
            [p.model_dump() for p in profiles],
        )

        scenario_distribution = _build_scenario_distribution(profiles)
        _write_json(out_dir / "scenario_distribution.json", scenario_distribution)

        split_manifest = {
            "split_version": _SPLIT_VERSION,
            "seed": _SPLIT_SEED,
            "unit": "simulation_id",
            "ratio": {
                "train": _SPLIT_RATIOS[0],
                "dev": _SPLIT_RATIOS[1],
                "test": _SPLIT_RATIOS[2],
            },
            "strategy": {
                "primary": ["scenario_type", "requires_grounding"],
                "balance_checks": [
                    "primary_scenario",
                    "terminal_tool_fingerprint",
                    "has_item_ids",
                    "has_order_id",
                    "has_product_id",
                    "requires_tool_chaining",
                    "difficulty_bucket",
                ],
            },
            "assignments": assignments,
        }
        _write_json(out_dir / "split_manifest.json", split_manifest)

        for split_name in ("train", "dev", "test"):
            split_rows = [r for r in all_supervision if r.split == split_name]
            _write_jsonl(
                splits_dir / f"{split_name}_supervision.jsonl",
                [r.model_dump() for r in split_rows],
            )
            split_sim_ids = [
                p.simulation_id for p in profiles if p.split == split_name
            ]
            _write_json(splits_dir / f"{split_name}_simulation_ids.json", split_sim_ids)

        return summary


_DISTRIBUTION_DIMS = [
    "scenario_type",
    "primary_scenario",
    "terminal_tool_fingerprint",
    "difficulty_bucket",
    "requires_grounding",
    "requires_tool_chaining",
    "has_item_ids",
    "has_order_id",
    "has_product_id",
    "is_multi_action",
]


def _count_dim(profiles: list[SimulationProfile], dim: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for p in profiles:
        val = getattr(p, dim)
        counts[str(val)] += 1
    return dict(sorted(counts.items()))


def _build_scenario_distribution(
    profiles: list[SimulationProfile],
) -> dict[str, Any]:
    overall = {dim: _count_dim(profiles, dim) for dim in _DISTRIBUTION_DIMS}
    by_split: dict[str, Any] = {}
    for split_name in ("train", "dev", "test"):
        split_profiles = [p for p in profiles if p.split == split_name]
        by_split[split_name] = {
            dim: _count_dim(split_profiles, dim) for dim in _DISTRIBUTION_DIMS
        }
    return {"overall": overall, "by_split": by_split}


def _registry_to_dict(registry: dict[str, ToolRegistryEntry]) -> dict[str, Any]:
    return {name: entry.model_dump() for name, entry in registry.items()}


def _write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def _write_jsonl(path: Path, rows: list[Any]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
