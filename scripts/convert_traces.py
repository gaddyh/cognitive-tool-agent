#!/usr/bin/env python3
"""Convert raw tau-bench simulation JSON to cognitive dataset artifacts.

Usage:
    python scripts/convert_traces.py --input data/raw/simulation.json --out-dir data/out/
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from cognitive_tool_agent.trace_converter.converter import TraceConverter


console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace-to-Cognitive-Dataset Converter")
    parser.add_argument("--input", required=True, help="Path to simulation JSON file")
    parser.add_argument("--out-dir", default="data/out", help="Output directory (default: data/out)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        console.print(f"[red]Input file not found: {input_path}[/red]")
        sys.exit(1)

    out_dir = Path(args.out_dir)

    console.print(Panel.fit(
        f"[bold cyan]Trace-to-Cognitive-Dataset Converter[/bold cyan]\n"
        f"Input:   [green]{input_path}[/green]\n"
        f"Out dir: [green]{out_dir}[/green]",
        border_style="cyan",
    ))

    converter = TraceConverter()
    summary = converter.run(input_path, out_dir)

    table = Table(title="Conversion Summary", box=box.SIMPLE)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right", style="cyan")

    table.add_row("Tasks", str(summary.tasks_count))
    table.add_row("Simulations", str(summary.simulations_count))
    table.add_row("Messages", str(summary.messages_count))
    table.add_row("Expected actions", str(summary.expected_actions_count))
    table.add_row("Actual tool calls", str(summary.actual_tool_calls_count))
    table.add_row("Matched actions", str(summary.matched_actions_count))
    table.add_row("Failed actions", str(summary.failed_actions_count))

    console.print(table)

    console.print("\n[bold]Output files:[/bold]")
    core_files = [
        "tool_registry.json",
        "action_sequence.jsonl",
        "turn_supervision.jsonl",
        "failure_rows.jsonl",
        "conversion_summary.json",
        "simulation_timings.jsonl",
    ]
    split_files = [
        "simulation_profiles.jsonl",
        "scenario_distribution.json",
        "split_manifest.json",
        "splits/train_supervision.jsonl",
        "splits/dev_supervision.jsonl",
        "splits/test_supervision.jsonl",
        "splits/train_simulation_ids.json",
        "splits/dev_simulation_ids.json",
        "splits/test_simulation_ids.json",
    ]
    for fname in core_files:
        fpath = out_dir / fname
        size = fpath.stat().st_size if fpath.exists() else 0
        console.print(f"  [green]{fpath}[/green]  ({size:,} bytes)")
    console.print("\n[bold]Split artifacts:[/bold]")
    for fname in split_files:
        fpath = out_dir / fname
        size = fpath.stat().st_size if fpath.exists() else 0
        console.print(f"  [cyan]{fpath}[/cyan]  ({size:,} bytes)")


if __name__ == "__main__":
    main()
