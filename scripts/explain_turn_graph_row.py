#!/usr/bin/env python3
"""Explain a single turn-level row through a named graph config.

Prints the full trace: input evidence, each node's output, and the score.

Usage:
    python scripts/explain_turn_graph_row.py --row-index 0 --graph recommended_oracle
    python scripts/explain_turn_graph_row.py --row-index 0 --graph recommended_stub
    python scripts/explain_turn_graph_row.py --row-index 0 --graph recommended_deterministic
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich import box
from rich.table import Table

from cognitive_tool_agent.graph.cognitive_graph import GraphExecutor
from cognitive_tool_agent.graph_builder.graph_candidate_generator import (
    make_monolithic_baseline,
    make_perceive_plan_act,
)
from cognitive_tool_agent.graph_runner.turn_tool_call_adapter import TurnToolCallAdapter
from cognitive_tool_agent.schemas.experiment import ExperimentSpec, NodeRuntimeConfig
from cognitive_tool_agent.schemas.recommender import RecommendedGraph
from cognitive_tool_agent.schemas.trace import CognitiveTrace
from cognitive_tool_agent.schemas.dataset import DatasetRow

console = Console()

_GRAPH_GROUNDING_MODE: dict[str, str | None] = {
    "monolithic": None,
    "minimal": None,
    "recommended_stub": "stub",
    "recommended_deterministic": "deterministic",
    "recommended_oracle": "oracle",
}


def _load_recommended(path: Path) -> RecommendedGraph:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return RecommendedGraph.model_validate(raw)


def _build_experiment(graph_id: str, recommended: RecommendedGraph) -> ExperimentSpec:
    if graph_id == "monolithic":
        return ExperimentSpec(graph=make_monolithic_baseline().graph_spec)
    if graph_id == "minimal":
        return ExperimentSpec(graph=make_perceive_plan_act().graph_spec)

    grounding_mode = _GRAPH_GROUNDING_MODE.get(graph_id)
    if grounding_mode is None:
        raise ValueError(f"Unknown graph id: {graph_id!r}")

    return ExperimentSpec(
        graph=recommended.graph_spec,
        runtime=[NodeRuntimeConfig(node_id="grounding", mode=grounding_mode)],
    )


def _score(trace: CognitiveTrace, row: DatasetRow) -> tuple[bool, bool, bool, str]:
    expected_action = row.expected.expected_action
    expected_tool = row.expected.expected_tool
    expected_args = row.expected.expected_arguments

    if trace.action is None:
        return False, False, False, "no action produced"

    e2e = trace.action.action_type in (expected_action, f"tool_{expected_action}")
    if expected_action == "tool_executed":
        e2e = trace.action.action_type == "tool_executed"

    tool_ok = (expected_tool is None) or (trace.action.tool_name == expected_tool)
    args_ok = (expected_args is None) or (trace.action.tool_arguments == expected_args)

    if not e2e:
        reason = f"action={trace.action.action_type!r}, expected={expected_action!r}"
    elif not tool_ok:
        reason = f"tool={trace.action.tool_name!r}, expected={expected_tool!r}"
    elif not args_ok:
        got = trace.action.tool_arguments or {}
        exp = expected_args or {}
        missing = {k: v for k, v in exp.items() if k not in got}
        wrong = {k: got[k] for k in got if k in exp and got[k] != exp[k]}
        extra = {k: v for k, v in got.items() if k not in exp}
        parts = []
        if missing:
            parts.append(f"missing={list(missing.keys())}")
        if wrong:
            parts.append(f"wrong={wrong}")
        if extra:
            parts.append(f"extra={list(extra.keys())}")
        reason = "args mismatch: " + ", ".join(parts)
    else:
        reason = "all correct"

    return e2e, tool_ok, args_ok, reason


def _fmt(value, max_len: int = 80) -> str:
    s = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    if len(s) > max_len:
        return s[:max_len - 3] + "..."
    return s


def _print_row_trace(row: DatasetRow, trace: CognitiveTrace, graph_id: str) -> None:
    e2e, tool_ok, args_ok, reason = _score(trace, row)
    score_color = "green" if (e2e and tool_ok and args_ok) else "red"
    score_label = "PASS" if (e2e and tool_ok and args_ok) else "FAIL"

    ws = row.world_state
    prior_calls = ws.get("prior_tool_calls", [])
    prior_results = ws.get("prior_tool_results", [])

    console.print(Rule(f"[bold cyan]{graph_id}[/bold cyan]  row=[yellow]{row.id}[/yellow]"))

    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    t.add_column("key", style="dim", min_width=22)
    t.add_column("value")

    t.add_row("user_message", _fmt(row.user_message))
    t.add_row("expected_tool", str(row.expected.expected_tool))
    t.add_row("expected_args", _fmt(row.expected.expected_arguments or {}))
    t.add_row("", "")
    t.add_row(
        "prior_tool_calls",
        _fmt(prior_calls) if prior_calls else "[dim](none)[/dim]",
    )
    t.add_row(
        "prior_tool_results",
        _fmt([r["content"] for r in prior_results[:3]]) if prior_results else "[dim](none)[/dim]",
    )
    t.add_row("", "")

    p = trace.perception
    t.add_row(
        "perception.candidates",
        _fmt(p.candidate_tools if p else None),
    )
    r = trace.reasoning
    t.add_row(
        "reasoning.selected",
        str(r.selected_tool if r else None),
    )
    g = trace.grounding
    t.add_row(
        "grounding.mode",
        str(g.grounding_mode if g else None),
    )
    t.add_row(
        "grounding.resolved",
        _fmt(g.resolved_args if g else {}),
    )
    t.add_row(
        "grounding.unresolved",
        _fmt(g.unresolved_ids if g else []),
    )
    rd = trace.readiness
    t.add_row(
        "readiness.ready",
        str(rd.ready if rd else None),
    )
    if rd and rd.missing_required_fields:
        t.add_row("readiness.missing", _fmt(rd.missing_required_fields))
    pl = trace.plan
    t.add_row(
        "plan.next_action",
        str(pl.next_action if pl else None),
    )
    if pl and pl.tool_call:
        t.add_row("plan.tool_call", _fmt(pl.tool_call.model_dump()))
    a = trace.action
    t.add_row("action.tool_name", str(a.tool_name if a else None))
    t.add_row("action.arguments", _fmt(a.tool_arguments if a else None))
    t.add_row("", "")
    t.add_row(
        "score",
        f"[{score_color}]{score_label}[/{score_color}]  {reason}",
    )

    console.print(t)


def main() -> None:
    parser = argparse.ArgumentParser(description="Explain a single turn-level row")
    parser.add_argument("--row-index", type=int, default=0, help="0-based row index")
    parser.add_argument(
        "--graph",
        default="recommended_oracle",
        choices=list(_GRAPH_GROUNDING_MODE.keys()),
        help="Graph config to run",
    )
    parser.add_argument("--recommended", default="reports/recommended_graph.json")
    parser.add_argument("--turn-sup", default="data/out/turn_supervision.jsonl")
    parser.add_argument("--tool-registry", default="data/out/tool_registry.json")
    args = parser.parse_args()

    recommended_path = Path(args.recommended)
    turn_sup_path = Path(args.turn_sup)
    tool_registry_path = Path(args.tool_registry)

    for p in (recommended_path, turn_sup_path, tool_registry_path):
        if not p.exists():
            console.print(f"[red]File not found: {p}[/red]")
            sys.exit(1)

    adapter = TurnToolCallAdapter()
    rows, registry = adapter.load(turn_sup_path, tool_registry_path)

    if args.row_index >= len(rows):
        console.print(
            f"[red]Row index {args.row_index} out of range (0–{len(rows) - 1})[/red]"
        )
        sys.exit(1)

    row = rows[args.row_index]
    recommended = _load_recommended(recommended_path)

    experiment = _build_experiment(args.graph, recommended)
    executor = GraphExecutor()
    trace = executor.run(experiment, row, registry)

    _print_row_trace(row, trace, args.graph)


if __name__ == "__main__":
    main()
