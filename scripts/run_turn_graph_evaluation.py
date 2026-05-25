#!/usr/bin/env python3
"""Turn-level graph evaluation: monolithic · minimal · recommended_stub · recommended_oracle.

Each row is one assistant tool-call turn from turn_supervision.jsonl.
Input message = last user message before that turn in the same simulation.

Usage:
    python scripts/run_turn_graph_evaluation.py \
        --recommended  reports/recommended_graph.json \
        --turn-sup     data/out/turn_supervision.jsonl \
        --tool-registry data/out/tool_registry.json \
        --out-dir      data/out \
        --reports-dir  reports \
        --limit        20
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

from cognitive_tool_agent.graph_runner.run_turn_recommended_graph import TurnGraphEvaluationRunner
from cognitive_tool_agent.schemas.graph_runner import GraphEvaluationReport

console = Console()


def _print_row_preview(rows) -> None:
    console.print(f"\n[bold]Rows built:[/bold] {len(rows)}\n")
    console.print("[bold]First 5 rows:[/bold]")
    table = Table(box=box.SIMPLE, show_header=True)
    table.add_column("id", style="dim", max_width=30)
    table.add_column("user_message", max_width=40)
    table.add_column("expected_tool", style="cyan")
    table.add_column("expected_args", max_width=30)
    table.add_column("tools_count", justify="right")

    for row in rows[:5]:
        args_preview = json.dumps(row.expected.expected_arguments or {})
        if len(args_preview) > 28:
            args_preview = args_preview[:25] + "..."
        msg_preview = (row.user_message or "")[:38]
        if len(row.user_message or "") > 38:
            msg_preview += "..."
        table.add_row(
            row.id,
            msg_preview,
            row.expected.expected_tool or "",
            args_preview,
            str(len(row.tools)),
        )
    console.print(table)


def _print_comparison_table(report: GraphEvaluationReport) -> None:
    console.print(f"\n[bold]Graph Comparison[/bold]  ({report.rows_evaluated} rows)\n")
    table = Table(box=box.SIMPLE)
    table.add_column("Graph", style="cyan", min_width=26)
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


_TRACKED_FIELDS = ["order_id", "product_id", "user_id", "payment_method_id", "item_ids", "new_item_ids"]


def _compute_field_grounding_stats(traces, rows) -> list[tuple]:
    stats = []
    for field in _TRACKED_FIELDS:
        rows_with = 0
        resolved = 0
        exact = 0
        for trace, row in zip(traces, rows):
            exp_args = row.expected.expected_arguments or {}
            if field not in exp_args:
                continue
            rows_with += 1
            g = trace.grounding
            if g and field in g.resolved_args:
                resolved += 1
                if g.resolved_args[field] == exp_args[field]:
                    exact += 1
        stats.append((field, rows_with, resolved, exact))
    return stats


def _print_field_grounding_summary(traces, rows) -> None:
    console.print("\n[bold]Field-Level Grounding Summary[/bold]  (recommended_deterministic)\n")
    table = Table(box=box.SIMPLE)
    table.add_column("arg_field", style="cyan")
    table.add_column("rows_with_field", justify="right")
    table.add_column("det_resolved", justify="right")
    table.add_column("exact_match", justify="right")
    table.add_column("resolve_rate", justify="right")
    table.add_column("match_rate", justify="right")

    for field, rows_with, resolved, exact in _compute_field_grounding_stats(traces, rows):
        resolve_rate = f"{resolved / rows_with:.0%}" if rows_with else "—"
        match_rate = f"{exact / rows_with:.0%}" if rows_with else "—"
        table.add_row(
            field,
            str(rows_with),
            str(resolved),
            str(exact),
            resolve_rate,
            match_rate,
        )
    console.print(table)


def _write_markdown(path: Path, eval_report: GraphEvaluationReport, det_traces_path: Path | None = None, rows=None) -> None:
    from datetime import datetime, timezone

    generated_at = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = []
    w = lines.append

    w("# Turn-Level Graph Evaluation Report")
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

    if det_traces_path and det_traces_path.exists() and rows is not None:
        from cognitive_tool_agent.graph_runner.trace_writer import TraceWriter
        det_traces = TraceWriter().load(det_traces_path)
        w("")
        w("## Field-Level Grounding Summary (recommended_deterministic)")
        w("")
        w("| arg_field | rows_with_field | det_resolved | exact_match | resolve_rate | match_rate |")
        w("|---|---:|---:|---:|---:|---:|")
        for field, rows_with, resolved, exact in _compute_field_grounding_stats(det_traces, rows):
            resolve_rate = f"{resolved / rows_with:.0%}" if rows_with else "—"
            match_rate = f"{exact / rows_with:.0%}" if rows_with else "—"
            w(
                f"| `{field}` "
                f"| {rows_with} "
                f"| {resolved} "
                f"| {exact} "
                f"| {resolve_rate} "
                f"| {match_rate} |"
            )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Turn-Level Graph Evaluation Harness")
    parser.add_argument("--recommended", default="reports/recommended_graph.json")
    parser.add_argument("--turn-sup", default="data/out/turn_supervision.jsonl")
    parser.add_argument("--tool-registry", default="data/out/tool_registry.json")
    parser.add_argument("--out-dir", default="data/out")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of rows")
    args = parser.parse_args()

    recommended_path = Path(args.recommended)
    turn_sup_path = Path(args.turn_sup)
    tool_registry_path = Path(args.tool_registry)
    out_dir = Path(args.out_dir)
    reports_dir = Path(args.reports_dir)

    for p in (recommended_path, turn_sup_path, tool_registry_path):
        if not p.exists():
            console.print(f"[red]File not found: {p}[/red]")
            sys.exit(1)

    limit_label = f"limit={args.limit}" if args.limit else "all rows"
    console.print(Panel.fit(
        "[bold cyan]Turn-Level Graph Evaluation Harness[/bold cyan]\n"
        f"Recommended: [green]{recommended_path}[/green]\n"
        f"Dataset:     [green]{turn_sup_path}[/green]\n"
        f"Out:         [green]{out_dir}[/green]\n"
        f"Mode:        [yellow]{limit_label}[/yellow]",
        border_style="cyan",
    ))

    runner = TurnGraphEvaluationRunner()
    eval_report, rows = runner.run(
        recommended_graph_path=recommended_path,
        turn_sup_path=turn_sup_path,
        tool_registry_path=tool_registry_path,
        out_dir=out_dir,
        limit=args.limit,
    )

    _print_row_preview(rows)

    console.print("\nRunning 5 graph configurations...")
    _print_comparison_table(eval_report)

    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    eval_json_path = out_dir / "turn_graph_evaluation_report.json"
    with eval_json_path.open("w", encoding="utf-8") as f:
        json.dump(eval_report.model_dump(), f, indent=2, ensure_ascii=False)

    det_traces_path = out_dir / "turn_traces_recommended_deterministic.jsonl"

    if det_traces_path.exists():
        from cognitive_tool_agent.graph_runner.trace_writer import TraceWriter
        det_traces = TraceWriter().load(det_traces_path)
        _print_field_grounding_summary(det_traces, rows)

    eval_md_path = reports_dir / "turn_graph_evaluation_report.md"
    _write_markdown(eval_md_path, eval_report, det_traces_path if det_traces_path.exists() else None, rows)

    console.print()
    for p in (eval_json_path, eval_md_path):
        console.print(f"[green]{p}[/green]  ({p.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
