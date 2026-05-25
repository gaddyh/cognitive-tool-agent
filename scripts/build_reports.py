#!/usr/bin/env python3
"""Build cognitive dataset reports from converter output.

Usage:
    python scripts/build_reports.py --out-dir data/out/ --reports-dir reports/
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from cognitive_tool_agent.reports.report_builder import ReportBuilder


console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Cognitive Dataset Report Builder")
    parser.add_argument("--out-dir", default="data/out", help="Converter output directory")
    parser.add_argument("--reports-dir", default="reports", help="Destination for report files")
    parser.add_argument("--source", default="", help="Optional label for the source dataset")
    parser.add_argument(
        "--train-only", default=True, action=argparse.BooleanOptionalAction,
        help="Restrict behavioral data to train split (default: True). Use --no-train-only for full-dataset inspection.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    reports_dir = Path(args.reports_dir)

    if not out_dir.exists():
        console.print(f"[red]Output directory not found: {out_dir}[/red]")
        sys.exit(1)

    if not args.train_only:
        console.print(Panel.fit(
            "[bold red]WARNING: full-dataset inspection mode.[/bold red]\n"
            "[red]This report must not be used for graph recommendation or optimization claims.[/red]",
            border_style="red",
        ))

    scope_label = "[yellow]TRAIN ONLY[/yellow]" if args.train_only else "[red]ALL SPLITS (inspection)[/red]"
    console.print(Panel.fit(
        f"[bold cyan]Cognitive Dataset Report Builder[/bold cyan]\n"
        f"Input:   [green]{out_dir}[/green]\n"
        f"Reports: [green]{reports_dir}[/green]\n"
        f"Scope:   {scope_label}",
        border_style="cyan",
    ))

    builder = ReportBuilder()
    report = builder.run(out_dir, reports_dir, source_label=args.source, train_only=args.train_only)

    summary = report["dataset_summary"]
    console.print("\n[bold]Dataset Summary (extended)[/bold]")
    t = Table(box=box.SIMPLE)
    t.add_column("Metric", style="bold")
    t.add_column("Value", justify="right", style="cyan")
    t.add_row("Tool entropy", f"{summary['tool_entropy_bits']} bits")
    t.add_row("Avg tools / sim", str(summary["avg_tools_per_simulation"]))
    t.add_row("Avg turns before write", str(summary["avg_turns_before_write_action"]))
    t.add_row("Read / write ratio", str(summary["read_write_ratio"]))
    console.print(t)

    topology = sorted(
        report["cognitive_action_topology"], key=lambda r: -r["complexity_score"]
    )
    console.print("\n[bold]Top tools by complexity score[/bold]")
    t2 = Table(box=box.SIMPLE)
    t2.add_column("Tool", style="cyan")
    t2.add_column("Type")
    t2.add_column("Score", justify="right", style="bold")
    t2.add_column("Extraction")
    t2.add_column("Memory")
    t2.add_column("Readiness")
    t2.add_column("Grounding")
    for row in topology[:8]:
        t2.add_row(
            row["tool_name"],
            row["tool_type"],
            str(row["complexity_score"]),
            row["extraction_burden"],
            row["memory_burden"],
            row["readiness_burden"],
            row["grounding_burden"],
        )
    console.print(t2)

    console.print("\n[bold]Failure heatmap — top failures[/bold]")
    t3 = Table(box=box.SIMPLE)
    t3.add_column("Dimension")
    t3.add_column("Value", style="cyan")
    t3.add_column("Count", justify="right")
    for row in report["failure_heatmap"]:
        t3.add_row(row["dimension"], row["value"], str(row["count"]))
    console.print(t3)

    console.print("\n[bold]Report files:[/bold]")
    for fname in [
        "cognitive_dataset_report.md",
        "cognitive_dataset_report.json",
        "cognitive_action_topology.csv",
        "failure_heatmap.csv",
        "argument_emergence.csv",
    ]:
        fpath = reports_dir / fname
        size = fpath.stat().st_size if fpath.exists() else 0
        console.print(f"  [green]{fpath}[/green]  ({size:,} bytes)")


if __name__ == "__main__":
    main()
