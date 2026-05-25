#!/usr/bin/env python3
"""Infer required capabilities from a CognitiveDatasetReport and recommend a graph topology.

Usage:
    python scripts/recommend_graph.py \
        --report reports/cognitive_dataset_report.json \
        --out    reports/recommended_graph.json
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

from cognitive_tool_agent.recommender.capability_inference import CapabilityInferenceEngine
from cognitive_tool_agent.recommender.graph_recommender import GraphRecommender


console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Cognitive Graph Recommender")
    parser.add_argument(
        "--report",
        default="reports/cognitive_dataset_report.json",
        help="Path to cognitive_dataset_report.json",
    )
    parser.add_argument(
        "--out",
        default="reports/recommended_graph.json",
        help="Destination for recommended_graph.json",
    )
    args = parser.parse_args()

    report_path = Path(args.report)
    out_path = Path(args.out)

    if not report_path.exists():
        console.print(f"[red]Report not found: {report_path}[/red]")
        sys.exit(1)

    console.print(Panel.fit(
        f"[bold cyan]Cognitive Graph Recommender[/bold cyan]\n"
        f"Report: [green]{report_path}[/green]\n"
        f"Output: [green]{out_path}[/green]",
        border_style="cyan",
    ))

    engine = CapabilityInferenceEngine()
    inference = engine.run(report_path)

    recommender = GraphRecommender()
    recommendation = recommender.run(inference)

    # ── Capability Inference table ────────────────────────────────────────────
    console.print("\n[bold]Capability Inference[/bold]")
    cap_table = Table(box=box.SIMPLE)
    cap_table.add_column("Capability", style="bold")
    cap_table.add_column("Required")
    cap_table.add_column("Strength", justify="right")
    cap_table.add_column("Evidence", style="dim")

    cap_order = ["memory", "grounding", "readiness", "deep_planning"]
    for cap_name in cap_order:
        cap = inference.required_capabilities[cap_name]
        required_label = "[green]YES[/green]" if cap.required else "[dim]low[/dim]"
        strength_label = f"({cap.strength:.2f})"
        evidence_text = cap.evidence[0] if cap.evidence else ""
        cap_table.add_row(cap_name, required_label, strength_label, evidence_text)

    console.print(cap_table)

    # ── Raw signals table ─────────────────────────────────────────────────────
    console.print("[bold]Raw Signals[/bold]")
    sig_table = Table(box=box.SIMPLE)
    sig_table.add_column("Signal", style="cyan")
    sig_table.add_column("Value", justify="right")
    for signal_name, value in inference.raw_signals.items():
        sig_table.add_row(signal_name, f"{value:.4f}")
    console.print(sig_table)

    # ── Recommended graph ─────────────────────────────────────────────────────
    console.print("[bold]Recommended Graph[/bold]")
    node_ids = [n.id for n in recommendation.graph_spec.topological_order()]
    graph_str = " → ".join(node_ids)
    console.print(f"  [cyan]{graph_str}[/cyan]")
    console.print(f"  confidence: [bold]{recommendation.confidence:.2f}[/bold]")
    console.print(f"  memory_required: {recommendation.memory_required}")
    console.print(f"  readiness_required: {recommendation.readiness_required}")

    console.print("\n[bold]Rationale[/bold]")
    for line in recommendation.rationale:
        console.print(f"  • {line}")

    # ── Write output ─────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(recommendation.model_dump(), fh, indent=2, ensure_ascii=False)

    md_path = out_path.with_suffix(".md")
    _write_markdown(md_path, inference, recommendation, report_path)

    console.print()
    for p in (out_path, md_path):
        console.print(f"[green]{p}[/green]  ({p.stat().st_size:,} bytes)")


def _write_markdown(path: Path, inference, recommendation, source: Path) -> None:
    from datetime import datetime, timezone
    from cognitive_tool_agent.schemas.recommender import CapabilityInferenceResult, RecommendedGraph

    generated_at = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    node_ids = [n.id for n in recommendation.graph_spec.topological_order()]
    cap_order = ["memory", "grounding", "readiness", "deep_planning"]

    lines: list[str] = []
    w = lines.append

    w("# Cognitive Graph Recommendation")
    w("")
    w(f"_Generated: {generated_at} — Source: `{source}`_")
    w("")

    # ── Capability Inference ──────────────────────────────────────────────────
    w("## Capability Inference")
    w("")
    w("| Capability | Required | Strength | Evidence |")
    w("|---|:---:|---:|---|")
    for cap_name in cap_order:
        cap = inference.required_capabilities[cap_name]
        req = "**YES**" if cap.required else "low"
        evidence = cap.evidence[0] if cap.evidence else ""
        w(f"| `{cap_name}` | {req} | {cap.strength:.2f} | {evidence} |")
    w("")

    # ── Raw Signals ───────────────────────────────────────────────────────────
    w("## Raw Signals")
    w("")
    w("| Signal | Value |")
    w("|---|---:|")
    for signal_name, value in inference.raw_signals.items():
        w(f"| `{signal_name}` | {value:.4f} |")
    w("")

    # ── Recommended Graph ─────────────────────────────────────────────────────
    w("## Recommended Graph")
    w("")
    w(f"> `{' → '.join(node_ids)}`")
    w("")
    w("| Property | Value |")
    w("|---|---|")
    w(f"| Graph ID | `{recommendation.graph_spec.id}` |")
    w(f"| Nodes | {len(recommendation.graph_spec.nodes)} |")
    w(f"| Confidence | **{recommendation.confidence:.2f}** |")
    w(f"| memory_required | {recommendation.memory_required} |")
    w(f"| readiness_required | {recommendation.readiness_required} |")
    w(f"| parallel_lookup_nodes | {recommendation.parallel_lookup_nodes} |")
    w("")

    w("### Node sequence")
    w("")
    w("| # | Node | Role |")
    w("|---:|---|---|")
    for i, node in enumerate(recommendation.graph_spec.topological_order(), 1):
        w(f"| {i} | `{node.id}` | {node.role} |")
    w("")

    # ── Rationale ─────────────────────────────────────────────────────────────
    w("## Rationale")
    w("")
    for line in recommendation.rationale:
        w(f"- {line}")
    w("")

    # ── Capability Detail ─────────────────────────────────────────────────────
    w("## Capability Detail")
    w("")
    for cap_name in cap_order:
        cap = inference.required_capabilities[cap_name]
        status = "Required" if cap.required else "Not required"
        w(f"### `{cap_name}` — {status} (strength {cap.strength:.2f})")
        w("")
        for ev in cap.evidence:
            w(f"- {ev}")
        w("")

    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
