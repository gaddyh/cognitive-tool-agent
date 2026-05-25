from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..schemas.simulation import SimulationFile
from ..schemas.trace_converter import (
    ActionSequenceRow,
    ConversionSummary,
    FailureRow,
    ToolRegistryEntry,
    TurnSupervisionRow,
)
from .action_aligner import align_actions
from .failure_extractor import extract_failures
from .simulation_loader import load_simulation_file
from .tool_registry_scanner import scan_tool_registry
from .turn_supervisor import supervise_turns


class TraceConverter:
    """
    Orchestrates the full Trace-to-Cognitive-Dataset conversion.

    Produces five output files:
      tool_registry.json
      action_sequence.jsonl
      turn_supervision.jsonl
      failure_rows.jsonl
      conversion_summary.json
    """

    def run(self, input_path: Path | str, out_dir: Path | str) -> ConversionSummary:
        input_path = Path(input_path)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        sim_file = load_simulation_file(input_path)
        task_map = {t.id: t for t in sim_file.tasks}

        tool_registry = scan_tool_registry(sim_file)

        all_sequences: list[ActionSequenceRow] = []
        all_supervision: list[TurnSupervisionRow] = []
        all_failures: list[FailureRow] = []

        messages_count = 0
        actual_tool_calls_count = 0

        for sim in sim_file.simulations:
            task = task_map.get(sim.task_id)
            if task is None:
                continue

            messages_count += len(sim.messages)
            for msg in sim.messages:
                if msg.tool_calls:
                    actual_tool_calls_count += len(msg.tool_calls)

            sequence = align_actions(sim, task)
            all_sequences.append(sequence)

            supervision = supervise_turns(sim, task)
            all_supervision.extend(supervision)

            failures = extract_failures(sim, sequence)
            all_failures.extend(failures)

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

        return summary


def _registry_to_dict(registry: dict[str, ToolRegistryEntry]) -> dict[str, Any]:
    return {name: entry.model_dump() for name, entry in registry.items()}


def _write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def _write_jsonl(path: Path, rows: list[Any]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
