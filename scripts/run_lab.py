#!/usr/bin/env python3
"""Run the full Graph Builder Lab loop on a JSONL dataset.

Usage:
    python scripts/run_lab.py --dataset data/dev/tool_calling_micro.jsonl
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from cognitive_tool_agent.datasets.loader import load_jsonl
from cognitive_tool_agent.graph_builder.lab import Lab
from cognitive_tool_agent.schemas.graph_builder import LabReport


console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Cognitive Graph Lab")
    parser.add_argument("--dataset", required=True, help="Path to JSONL dataset file")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        console.print(f"[red]Dataset not found: {dataset_path}[/red]")
        sys.exit(1)

    console.print(Panel.fit(
        f"[bold cyan]Cognitive Graph Lab[/bold cyan]\n"
        f"Dataset: [green]{dataset_path}[/green]",
        border_style="cyan",
    ))

    rows = load_jsonl(dataset_path)
    console.print(f"Loaded [bold]{len(rows)}[/bold] rows.\n")

    lab = Lab()
    report = lab.run(rows)
    _print_report(report)


def _print_report(report: LabReport) -> None:
    profile = report.dataset_profile
    notes_text = ("\n" + "\n".join(f"  * {n}" for n in profile.notes)) if profile.notes else ""
    console.print(Panel.fit(
        f"[bold]Task type:[/bold] {profile.task_type}\n"
        f"[bold]Rows:[/bold] {profile.row_count}   "
        f"[bold]Tools:[/bold] {profile.tool_count}   "
        f"[bold]Ambiguity rate:[/bold] {profile.ambiguity_rate:.0%}\n"
        f"[bold]Labels:[/bold] {', '.join(profile.label_set)}"
        + notes_text,
        title="Dataset Profile",
        border_style="blue",
    ))

    decomp = report.behavior_decomposition
    console.print(Panel.fit(
        f"[bold]Stages:[/bold] {' -> '.join(decomp.stages)}\n"
        f"[dim]{decomp.rationale}[/dim]",
        title="Behavior Decomposition",
        border_style="blue",
    ))

    console.print("\n[bold]Graph Candidates[/bold]")
    for c in report.candidates:
        roles = " -> ".join(n.role for n in c.graph_spec.topological_order())
        console.print(
            f"  [cyan]{c.id}[/cyan]  {roles}  "
            f"(latency={c.latency_estimate:.0f}x, cost={c.cost_estimate:.0f}x)"
        )

    console.print()
    _print_scores_table("Baseline Scores", report.baseline_scores, "yellow")

    failure = report.failure_map
    failures_text = _format_failures(failure.failures)
    console.print(Panel.fit(
        f"[bold]Candidate:[/bold] {failure.candidate_id}\n"
        f"[bold]Failures:[/bold] {failure.failure_count} / {failure.total_rows}\n"
        f"[bold]Dominant stage:[/bold] [red]{failure.dominant_failure_stage}[/red]\n"
        f"[bold]Dominant type:[/bold]  [red]{failure.dominant_failure_type}[/red]"
        + failures_text,
        title="Failure Map",
        border_style="red",
    ))

    if report.revision:
        rev = report.revision
        console.print(Panel.fit(
            f"[bold]From:[/bold] {rev.from_candidate_id}  ->  [bold]To:[/bold] {rev.to_candidate_id}\n"
            f"[bold]Change:[/bold] {rev.change_type}\n"
            f"[dim]{rev.rationale}[/dim]",
            title="Graph Revision",
            border_style="green",
        ))

    _print_scores_table("Optimized Scores", report.optimized_scores, "green")

    console.print("\n[bold]Tradeoff Summary[/bold]")
    table = Table(box=box.SIMPLE)
    table.add_column("Candidate", style="cyan")
    table.add_column("E2E Success", justify="right")
    table.add_column("Latency", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("Failures", justify="right")
    table.add_column("Recommendation")
    for entry in report.tradeoff_summary:
        table.add_row(
            entry.candidate_id,
            f"{entry.end_to_end_success:.0%}",
            f"{entry.latency_estimate:.0f}x",
            f"{entry.cost_estimate:.0f}x",
            str(entry.failure_count),
            entry.recommendation,
        )
    console.print(table)


def _print_scores_table(
    title: str, scores: dict, border_style: str
) -> None:
    if not scores:
        return
    lines = []
    for candidate_id, metrics in scores.items():
        lines.append(f"[bold]{candidate_id}[/bold]")
        for metric, value in metrics.items():
            lines.append(f"  {metric}: [bold]{value:.1%}[/bold]")
    console.print(Panel.fit("\n".join(lines), title=title, border_style=border_style))
    console.print()


def _format_failures(failures) -> str:
    if not failures:
        return "\n[green]No failures detected.[/green]"
    lines = ["\n[bold]Row failures:[/bold]"]
    for f in failures:
        lines.append(
            f"  [yellow]{f.row_id}[/yellow]  "
            f"expected={f.expected_action} actual={f.actual_action}  "
            f"stage=[red]{f.failure_stage}[/red]  type=[red]{f.failure_type}[/red]"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
