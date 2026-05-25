#!/usr/bin/env python3
"""Full pipeline with explicit EDD boundary.

DESIGN PHASE (train only):
  1/5  Convert traces + stratified split
  2/5  Build cognitive report  [TRAIN ONLY]
  3/5  Recommend graph          [TRAIN ONLY -> frozen]
  4/5  Build split report       [descriptive]

EVALUATION PHASE:
  5/5  Evaluate frozen graph on: all / train / dev / test

Usage:
    python scripts/run_pipeline.py \\
        --input data/raw/simulations/baseline_retail_100/results.json

    # Incremental (skip already-done steps):
    python scripts/run_pipeline.py \\
        --input data/raw/simulations/baseline_retail_100/results.json \\
        --skip-convert --skip-reports --skip-recommend
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich import box

from cognitive_tool_agent.trace_converter.converter import TraceConverter
from cognitive_tool_agent.reports.report_builder import ReportBuilder
from cognitive_tool_agent.recommender.capability_inference import CapabilityInferenceEngine
from cognitive_tool_agent.recommender.graph_recommender import GraphRecommender
from cognitive_tool_agent.graph_runner.run_turn_recommended_graph import TurnGraphEvaluationRunner
from build_split_report import build_report as _build_split_report

console = Console()

_VARIANTS_TO_SHOW = [
    "recommended_stub",
    "recommended_deterministic",
    "recommended_oracle",
]
_METRICS = [
    ("end_to_end_success", "E2E Success"),
    ("tool_name_accuracy", "Tool Acc"),
    ("argument_exact_match", "Arg Match"),
]


def _ok(msg: str) -> None:
    console.print(f"  [green]✓[/green] {msg}")


def _skip(msg: str) -> None:
    console.print(f"  [dim]↷ skipped — {msg}[/dim]")


def _run_evaluate(
    turn_sup_path: Path,
    recommended_path: Path,
    tool_registry_path: Path,
    out_dir: Path,
    label: str,
    limit: int | None,
) -> dict[str, dict[str, float]]:
    runner = TurnGraphEvaluationRunner()
    eval_report, _ = runner.run(
        recommended_graph_path=recommended_path,
        turn_sup_path=turn_sup_path,
        tool_registry_path=tool_registry_path,
        out_dir=out_dir,
        limit=limit,
    )
    _ok(f"{label}: {eval_report.rows_evaluated} rows")
    return {
        r.graph_id: {
            "end_to_end_success": r.end_to_end_success,
            "tool_name_accuracy": r.tool_name_accuracy,
            "argument_exact_match": r.argument_exact_match,
        }
        for r in eval_report.results
    }


def _print_proof_table(
    results: dict[str, dict[str, dict[str, float]]],
    metric: str,
    metric_label: str,
) -> None:
    splits = list(results.keys())
    table = Table(box=box.SIMPLE, title=f"[bold]{metric_label}[/bold]")
    table.add_column("variant", style="cyan", min_width=30)
    for split in splits:
        table.add_column(split, justify="right", min_width=7)

    for variant in _VARIANTS_TO_SHOW:
        row_vals = []
        for split in splits:
            val = results[split].get(variant, {}).get(metric)
            if val is None:
                row_vals.append("[dim]—[/dim]")
            else:
                color = (
                    "green" if val >= 0.80
                    else "yellow" if val >= 0.40
                    else "red" if val > 0.02
                    else "dim"
                )
                row_vals.append(f"[{color}]{val:.0%}[/{color}]")
        table.add_row(variant, *row_vals)

    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Full cognitive pipeline with EDD boundary")
    parser.add_argument("--input", required=True, help="Raw simulation JSON file")
    parser.add_argument("--out-dir", default="data/out")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--limit", type=int, default=None, help="Limit rows per eval run (smoke test)")
    parser.add_argument("--skip-convert", action="store_true")
    parser.add_argument("--skip-reports", action="store_true")
    parser.add_argument("--skip-recommend", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    reports_dir = Path(args.reports_dir)
    splits_dir = out_dir / "splits"

    console.print(Panel.fit(
        "[bold cyan]Cognitive Tool Agent — Full Pipeline[/bold cyan]\n"
        f"Input:    [green]{input_path}[/green]\n"
        f"Out dir:  [green]{out_dir}[/green]\n"
        f"Reports:  [green]{reports_dir}[/green]",
        border_style="cyan",
    ))

    # ══════════════════════════════════════════════════════════════════════════
    # DESIGN PHASE — train only
    # ══════════════════════════════════════════════════════════════════════════
    console.print()
    console.print(Rule("[bold yellow]DESIGN PHASE  (train only)[/bold yellow]", style="yellow"))

    # ── 1. Convert ────────────────────────────────────────────────────────────
    console.print("\n[bold]1/5  Convert traces + stratified split[/bold]")
    if args.skip_convert:
        _skip("--skip-convert")
    else:
        if not input_path.exists():
            console.print(f"[red]Input not found: {input_path}[/red]")
            sys.exit(1)
        summary = TraceConverter().run(input_path, out_dir)
        _ok(f"{summary.simulations_count} simulations converted → {out_dir}")
        for split_name in ("train", "dev", "test"):
            p = splits_dir / f"{split_name}_supervision.jsonl"
            if p.exists():
                _ok(f"splits/{split_name}_supervision.jsonl  ({p.stat().st_size:,} bytes)")

    # ── 2. Cognitive report (train only) ──────────────────────────────────────
    console.print("\n[bold]2/5  Build cognitive dataset report  [yellow][TRAIN ONLY][/yellow][/bold]")
    cognitive_report_path = reports_dir / "cognitive_dataset_report.json"
    if args.skip_reports:
        _skip("--skip-reports")
    else:
        ReportBuilder().run(out_dir, reports_dir, train_only=True)
        report_data = json.loads(cognitive_report_path.read_text())
        _ok(
            f"source = \"{report_data.get('source')}\"  "
            f"| boundary = {report_data.get('experimental_boundary', {}).get('data_scope')}"
        )
        _ok(f"cognitive_dataset_report.json  ({cognitive_report_path.stat().st_size:,} bytes)")

    # ── 3. Recommend graph (from train-only report) ────────────────────────────
    console.print("\n[bold]3/5  Recommend graph  [yellow][TRAIN ONLY → frozen][/yellow][/bold]")
    recommended_path = reports_dir / "recommended_graph.json"
    if args.skip_recommend:
        _skip("--skip-recommend")
    else:
        if not cognitive_report_path.exists():
            console.print(f"[red]Cognitive report not found: {cognitive_report_path}[/red]")
            sys.exit(1)
        inference = CapabilityInferenceEngine().run(cognitive_report_path)
        recommendation = GraphRecommender().run(inference)
        reports_dir.mkdir(parents=True, exist_ok=True)
        output = recommendation.model_dump()
        output["experimental_boundary"] = {
            "artifact_type": "design",
            "data_scope": "train_only",
            "allowed_to_influence_graph": True,
        }
        output["derived_from_report"] = str(cognitive_report_path.resolve())
        with recommended_path.open("w", encoding="utf-8") as fh:
            json.dump(output, fh, indent=2, ensure_ascii=False)
        node_ids = [n.id for n in recommendation.graph_spec.topological_order()]
        _ok(f"Graph: {' → '.join(node_ids)}  (confidence={recommendation.confidence:.2f})")
        _ok(f"recommended_graph.json  ({recommended_path.stat().st_size:,} bytes)")

    # ── 4. Split report (descriptive) ─────────────────────────────────────────
    console.print("\n[bold]4/5  Build split report  [dim][descriptive — not an input to graph design][/dim][/bold]")
    split_report_path = reports_dir / "split_report.md"
    _build_split_report(out_dir, split_report_path)
    _ok(f"split_report.md  ({split_report_path.stat().st_size:,} bytes)")

    # ══════════════════════════════════════════════════════════════════════════
    # EVALUATION PHASE — measures performance of the frozen graph
    # ══════════════════════════════════════════════════════════════════════════
    console.print()
    console.print(Rule("[bold green]EVALUATION PHASE  (frozen recommended_graph.json)[/bold green]", style="green"))
    console.print("\n[bold]5/5  Evaluate on all / train / dev / test[/bold]")

    tool_registry_path = out_dir / "tool_registry.json"
    for p in (recommended_path, tool_registry_path):
        if not p.exists():
            console.print(f"[red]Required file not found: {p}[/red]")
            sys.exit(1)

    eval_datasets: list[tuple[str, Path]] = [
        ("all",   out_dir / "turn_supervision.jsonl"),
        ("train", splits_dir / "train_supervision.jsonl"),
        ("dev",   splits_dir / "dev_supervision.jsonl"),
        ("test",  splits_dir / "test_supervision.jsonl"),
    ]

    split_results: dict[str, dict[str, dict[str, float]]] = {}
    for label, sup_path in eval_datasets:
        if not sup_path.exists():
            console.print(f"  [yellow]⚠ skipping {label}: {sup_path} not found[/yellow]")
            continue
        split_results[label] = _run_evaluate(
            sup_path, recommended_path, tool_registry_path, out_dir, label, args.limit
        )

    # ── Proof table ───────────────────────────────────────────────────────────
    console.print()
    console.print(Rule("[bold]Proof Table — same frozen graph evaluated across splits[/bold]", style="bold"))
    for metric, label in _METRICS:
        _print_proof_table(split_results, metric=metric, metric_label=label)

    console.print()
    console.print("[bold green]Pipeline complete.[/bold green]")
    console.print(
        "[dim]Claim: The graph was derived from train-only behavioral analysis, "
        "frozen, then evaluated independently across train/dev/test.[/dim]"
    )


if __name__ == "__main__":
    main()
