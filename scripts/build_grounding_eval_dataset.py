#!/usr/bin/env python3
"""Pass 1 — Build offline grounding evaluation dataset.

Reads turn_supervision.jsonl + tool_registry.json and produces per-split
grounding_eval JSONL files. Each row is a grounding example:

  - selected_tool = expected tool (clean experiment)
  - available_state = prior_tool_calls + prior_tool_results + conversation_context
  - current_deterministic_args = pre-computed deterministic grounding result
  - target_args = expected_arguments filtered to target_fields
  - target_args is NEVER passed to the LLM; it is only used by Pass 2 evaluator

Usage:
    python scripts/build_grounding_eval_dataset.py
    python scripts/build_grounding_eval_dataset.py --target-fields all-expected
    python scripts/build_grounding_eval_dataset.py --turn-sup data/out/turn_supervision.jsonl
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.table import Table
from rich import box

from cognitive_tool_agent.agents.grounding_agent import GroundingAgent
from cognitive_tool_agent.graph.node_input import NodeInput
from cognitive_tool_agent.graph_runner.turn_tool_call_adapter import TurnToolCallAdapter
from cognitive_tool_agent.schemas.common import UserInput
from cognitive_tool_agent.schemas.dataset import DatasetRow, ExpectedBehavior
from cognitive_tool_agent.schemas.trace_converter import TurnSupervisionRow

console = Console()

_SPLITS = ("train", "dev", "test")


def _load_turns(path: Path) -> list[TurnSupervisionRow]:
    rows: list[TurnSupervisionRow] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(TurnSupervisionRow.model_validate_json(line))
    return rows


def _run_deterministic_grounding(
    user_message: str,
    tool_name: str,
    prior_tool_calls: list[dict],
    prior_tool_results: list[dict],
    registry,
) -> dict[str, Any]:
    schema = registry.lookup(tool_name)
    if schema is None:
        return {}

    user_input = UserInput(message=user_message, available_tools=[schema])
    row = DatasetRow(
        id="grounding-build-tmp",
        user_message=user_message,
        world_state={
            "primary_tool": tool_name,
            "prior_tool_calls": prior_tool_calls,
            "prior_tool_results": prior_tool_results,
        },
        expected=ExpectedBehavior(expected_action="tool_executed"),
    )
    ctx = NodeInput(user_input=user_input, registry=registry, row=row)
    result = GroundingAgent(mode="deterministic").run(ctx)
    return result.resolved_args


def _build_rows(
    turns: list[TurnSupervisionRow],
    registry,
    target_fields_mode: str,
) -> dict[str, list[dict]]:
    """Build grounding eval rows grouped by split."""
    sorted_turns = sorted(turns, key=lambda t: (t.simulation_id, t.turn_idx))

    sim_turns: dict[str, list[TurnSupervisionRow]] = {}
    for turn in sorted_turns:
        sim_turns.setdefault(turn.simulation_id, []).append(turn)

    split_rows: dict[str, list[dict]] = {"train": [], "dev": [], "test": [], "unknown": []}

    for sim_id, sim_turn_list in sim_turns.items():
        last_user_message: str = ""
        prior_tool_calls: list[dict] = []
        prior_tool_results: list[dict] = []
        conversation_context: list[str] = []

        for turn in sim_turn_list:
            label = turn.cognitive_label

            if turn.role == "user":
                if turn.content:
                    last_user_message = turn.content
                    conversation_context.append(turn.content)
                continue

            if turn.role == "tool":
                if turn.content:
                    prior_tool_results.append({"content": turn.content})
                    conversation_context.append(turn.content)
                continue

            if turn.role != "assistant":
                continue

            if not (label.plan_next_action == "call_tool" and label.plan_tool_name):
                if turn.content:
                    conversation_context.append(turn.content)
                continue

            tool_name = label.plan_tool_name
            expected_args: dict[str, Any] = dict(label.plan_arguments or {})

            schema = registry.lookup(tool_name)
            if schema is None:
                if turn.content:
                    conversation_context.append(turn.content)
                prior_tool_calls.append(
                    {"tool_name": tool_name, "arguments": expected_args}
                )
                continue

            required_fields = schema.required_fields or []

            if target_fields_mode == "required":
                target_fields = [f for f in required_fields if f in expected_args]
                target_args = {f: expected_args[f] for f in target_fields}
            else:
                target_fields = list(expected_args.keys())
                target_args = dict(expected_args)

            tool_schema_dict = {
                "name": schema.name,
                "description": schema.description,
                "required_fields": required_fields,
                "optional_fields": list(schema.optional_fields or []),
            }

            available_state = {
                "prior_tool_calls": list(prior_tool_calls),
                "prior_tool_results": list(prior_tool_results),
                "conversation_context": list(conversation_context),
            }

            det_args = _run_deterministic_grounding(
                user_message=last_user_message,
                tool_name=tool_name,
                prior_tool_calls=list(prior_tool_calls),
                prior_tool_results=list(prior_tool_results),
                registry=registry,
            )

            row_id = f"{turn.task_id}:{sim_id[:8]}:turn:{turn.turn_idx}"
            split = turn.split or "unknown"

            grounding_row = {
                "id": row_id,
                "split": split,
                "simulation_id": sim_id,
                "turn_idx": turn.turn_idx,
                "task_id": turn.task_id,
                "user_message": last_user_message,
                "conversation_context": list(conversation_context),
                "selected_tool": tool_name,
                "tool_schema": tool_schema_dict,
                "available_state": available_state,
                "current_deterministic_args": det_args,
                "target_args": target_args,
                "target_fields": target_fields,
                "difficulty_bucket": turn.difficulty_bucket or "unknown",
                "scenario_type": turn.scenario_type or "unknown",
                "requires_grounding": turn.requires_grounding,
            }

            split_rows.setdefault(split, []).append(grounding_row)

            prior_tool_calls.append(
                {"tool_name": tool_name, "arguments": expected_args}
            )
            if turn.content:
                conversation_context.append(turn.content)

    return split_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build offline grounding evaluation dataset")
    parser.add_argument("--turn-sup", default="data/out/turn_supervision.jsonl")
    parser.add_argument("--tool-registry", default="data/out/tool_registry.json")
    parser.add_argument("--out-dir", default="data/out/grounding")
    parser.add_argument(
        "--target-fields",
        choices=["required", "all-expected"],
        default="required",
        help="Which expected fields to include in target_args (default: required)",
    )
    args = parser.parse_args()

    turn_sup_path = Path(args.turn_sup)
    tool_registry_path = Path(args.tool_registry)
    out_dir = Path(args.out_dir)

    for p in (turn_sup_path, tool_registry_path):
        if not p.exists():
            console.print(f"[red]File not found: {p}[/red]")
            sys.exit(1)

    console.print(f"\n[bold]Building grounding eval dataset[/bold]")
    console.print(f"  turn_sup:      {turn_sup_path}")
    console.print(f"  tool_registry: {tool_registry_path}")
    console.print(f"  out_dir:       {out_dir}")
    console.print(f"  target_fields: {args.target_fields}")

    adapter = TurnToolCallAdapter()
    _, registry = adapter.load(turn_sup_path, tool_registry_path)

    console.print("\nLoading turns...")
    turns = _load_turns(turn_sup_path)
    console.print(f"  {len(turns)} turns loaded")

    console.print("Building grounding rows...")
    split_rows = _build_rows(turns, registry, args.target_fields)

    out_dir.mkdir(parents=True, exist_ok=True)

    table = Table(box=box.SIMPLE, show_header=True)
    table.add_column("split", style="cyan")
    table.add_column("rows", justify="right")
    table.add_column("path", style="dim")

    for split in _SPLITS:
        rows = split_rows.get(split, [])
        if not rows:
            continue
        out_path = out_dir / f"grounding_eval_{split}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        table.add_row(split, str(len(rows)), str(out_path))

    unknown = split_rows.get("unknown", [])
    if unknown:
        out_path = out_dir / "grounding_eval_unknown.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for row in unknown:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        console.print(f"  [yellow]Warning: {len(unknown)} rows had unknown split → {out_path}[/yellow]")

    console.print()
    console.print(table)
    console.print("\n[green]Done.[/green]")


if __name__ == "__main__":
    main()
