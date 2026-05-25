#!/usr/bin/env python3
"""Run multi-graph evaluation: monolithic · minimal · recommended_stub · recommended_oracle.

Usage:
    python scripts/run_graph_evaluation.py \
        --recommended  reports/recommended_graph.json \
        --report       reports/cognitive_dataset_report.json \
        --out-dir      data/out \
        --reports-dir  reports
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from cognitive_tool_agent.graph_runner.run_recommended_graph import GraphEvaluationRunner
from cognitive_tool_agent.recommender.capability_inference import CapabilityInferenceEngine
from cognitive_tool_agent.recommender.revision_advisor import GraphRevisionAdvisor
from cognitive_tool_agent.schemas.graph_runner import GraphEvaluationReport


console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Graph Evaluation Harness")
    parser.add_argument(
        "--recommended",
        default="reports/recommended_graph.json",
        help="Path to recommended_graph.json",
    )
    parser.add_argument(
        "--report",
        default="reports/cognitive_dataset_report.json",
        help="Path to cognitive_dataset_report.json",
    )
    parser.add_argument(
        "--action-seq",
        default="data/out/action_sequence.jsonl",
        help="Path to action_sequence.jsonl",
    )
    parser.add_argument(
        "--turn-sup",
        default="data/out/turn_supervision.jsonl",
        help="Path to turn_supervision.jsonl",
    )
    parser.add_argument(
        "--tool-registry",
        default="data/out/tool_registry.json",
        help="Path to tool_registry.json",
    )
    parser.add_argument("--out-dir", default="data/out", help="Output directory for traces")
    parser.add_argument("--reports-dir", default="reports", help="Output directory for reports")
    args = parser.parse_args()

    recommended_path = Path(args.recommended)
    report_path = Path(args.report)
    action_seq_path = Path(args.action_seq)
    turn_sup_path = Path(args.turn_sup)
    tool_registry_path = Path(args.tool_registry)
    out_dir = Path(args.out_dir)
    reports_dir = Path(args.reports_dir)

    for p in (recommended_path, report_path, action_seq_path, turn_sup_path, tool_registry_path):
        if not p.exists():
            console.print(f"[red]File not found: {p}[/red]")
            sys.exit(1)

    console.print(Panel.fit(
        "[bold cyan]Graph Evaluation Harness[/bold cyan]\n"
        f"Recommended: [green]{recommended_path}[/green]\n"
        f"Dataset:     [green]{action_seq_path}[/green]\n"
        f"Out:         [green]{out_dir}[/green]",
        border_style="cyan",
    ))

    runner = GraphEvaluationRunner()
    console.print("\nRunning 4 graph configurations...")
    eval_report = runner.run(
        recommended_graph_path=recommended_path,
        action_seq_path=action_seq_path,
        turn_sup_path=turn_sup_path,
        tool_registry_path=tool_registry_path,
        out_dir=out_dir,
    )

    _print_comparison_table(eval_report)

    engine = CapabilityInferenceEngine()
    inference = engine.run(report_path)
    advisor = GraphRevisionAdvisor()
    revision_report = advisor.advise(eval_report, inference)

    eval_report.revision_suggestions = [s.suggestion for s in revision_report.suggestions]

    _print_revision_suggestions(revision_report)

    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    eval_json_path = out_dir / "graph_evaluation_report.json"
    with eval_json_path.open("w", encoding="utf-8") as f:
        json.dump(eval_report.model_dump(), f, indent=2, ensure_ascii=False)

    eval_md_path = reports_dir / "graph_evaluation_report.md"
    _write_markdown(eval_md_path, eval_report, revision_report)

    console.print()
    for p in (eval_json_path, eval_md_path):
        console.print(f"[green]{p}[/green]  ({p.stat().st_size:,} bytes)")


def _print_comparison_table(report: GraphEvaluationReport) -> None:
    console.print(f"\n[bold]Graph Comparison[/bold]  ({report.rows_evaluated} rows)\n")
    table = Table(box=box.SIMPLE)
    table.add_column("Graph", style="cyan")
    table.add_column("Nodes", justify="right")
    table.add_column("E2E Success", justify="right")
    table.add_column("Tool Acc", justify="right")
    table.add_column("Arg Match", justify="right")
    table.add_column("Policy Viol", justify="right")
    table.add_column("Failures", justify="right")
    table.add_column("Grounding", style="dim")

    for row in report.results:
        e2e_color = (
            "green" if row.end_to_end_success >= 0.80
            else "yellow" if row.end_to_end_success >= 0.50
            else "red"
        )
        table.add_row(
            row.graph_id,
            str(row.node_count),
            f"[{e2e_color}]{row.end_to_end_success:.0%}[/{e2e_color}]",
            f"{row.tool_name_accuracy:.0%}",
            f"{row.argument_exact_match:.0%}",
            f"{row.policy_violation_rate:.0%}",
            str(row.failure_count),
            row.grounding_mode,
        )
    console.print(table)


def _print_revision_suggestions(revision_report) -> None:
    if not revision_report.suggestions:
        console.print("\n[green]No revision suggestions.[/green]")
        return

    console.print("\n[bold]Revision Suggestions[/bold]")
    priority_color = {"high": "red", "medium": "yellow", "low": "dim"}
    for s in revision_report.suggestions:
        color = priority_color.get(s.priority, "white")
        console.print(
            f"  [{color}][{s.priority.upper()}][/{color}] "
            f"[bold]{s.target_capability}[/bold] — {s.failure_pattern}"
        )
        console.print(f"    {s.suggestion}")
        console.print(f"    [dim]{s.rationale}[/dim]")


def _write_markdown(path: Path, eval_report: GraphEvaluationReport, revision_report) -> None:
    from datetime import datetime, timezone

    generated_at = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = []
    w = lines.append

    w("# Graph Evaluation Report")
    w("")
    w(f"_Generated: {generated_at}_")
    w("")
    w(f"**Source dataset**: `{eval_report.source_dataset}`  ")
    w(f"**Rows evaluated**: {eval_report.rows_evaluated}")
    w("")

    w("## Comparison Table")
    w("")
    w("| Graph | Nodes | E2E Success | Tool Acc | Arg Match | Policy Viol | Failures | Grounding |")
    w("|---|---:|---:|---:|---:|---:|---:|---|")
    for row in eval_report.results:
        w(
            f"| `{row.graph_id}` "
            f"| {row.node_count} "
            f"| {row.end_to_end_success:.0%} "
            f"| {row.tool_name_accuracy:.0%} "
            f"| {row.argument_exact_match:.0%} "
            f"| {row.policy_violation_rate:.0%} "
            f"| {row.failure_count} "
            f"| {row.grounding_mode} |"
        )
    w("")

    if revision_report.suggestions:
        w("## Revision Suggestions")
        w("")
        for s in revision_report.suggestions:
            w(f"### [{s.priority.upper()}] `{s.target_capability}` — {s.failure_pattern}")
            w("")
            w(f"**Suggestion**: {s.suggestion}")
            w("")
            w(f"**Rationale**: {s.rationale}")
            w("")
            w("**Evidence**:")
            for ev in s.evidence:
                w(f"- `{ev}`")
            w("")

    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
