#!/usr/bin/env python3
"""Run only the monolithic baseline candidate and print per-row results.

Usage:
    python scripts/run_baseline.py --dataset data/dev/tool_calling_micro.jsonl
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.table import Table
from rich import box

from cognitive_tool_agent.datasets.loader import load_jsonl
from cognitive_tool_agent.graph_builder.baseline_runner import BaselineRunner
from cognitive_tool_agent.graph_builder.graph_candidate_generator import _make_monolithic
from cognitive_tool_agent.tools.fake_tools import DEFAULT_REGISTRY


console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run monolithic baseline candidate")
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    path = Path(args.dataset)
    rows = load_jsonl(path)
    console.print(f"Loaded [bold]{len(rows)}[/bold] rows from [green]{path}[/green]\n")

    candidate = _make_monolithic()
    runner = BaselineRunner()
    traces, scores = runner.run(candidate, rows, DEFAULT_REGISTRY)

    table = Table(title=f"Baseline: {candidate.id}", box=box.SIMPLE)
    table.add_column("Row ID", style="cyan")
    table.add_column("Expected")
    table.add_column("Actual")
    table.add_column("Tool")
    table.add_column("Pass", justify="center")

    action_map = {
        "tool_executed": "execute_tool",
        "followup_asked": "ask_followup",
        "answered_directly": "answer_directly",
        "abstained": "abstain",
        "rejected": "reject",
    }

    for trace, row in zip(traces, rows):
        expected = row.expected.expected_action
        actual = action_map.get(trace.action.action_type, "?") if trace.action else "none"
        tool = trace.action.tool_name or "-" if trace.action else "-"
        passed = "[green]✓[/green]" if expected == actual else "[red]✗[/red]"
        table.add_row(row.id, expected, actual, tool, passed)

    console.print(table)
    console.print("\n[bold]Scores:[/bold]")
    for metric, value in scores.items():
        console.print(f"  {metric}: {value:.1%}")


if __name__ == "__main__":
    main()
