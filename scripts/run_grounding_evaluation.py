#!/usr/bin/env python3
"""Pass 2 — Standalone grounding evaluator.

Runs both deterministic and LLM grounding variants over the offline
grounding_eval dataset and produces a proof table + per-split reports.

Usage:
    # Deterministic baseline only (no LLM calls, no API key needed):
    python scripts/run_grounding_evaluation.py --split dev --no-llm

    # Full run (deterministic + LLM):
    python scripts/run_grounding_evaluation.py --split dev --model gpt-4o-mini

    # All splits:
    python scripts/run_grounding_evaluation.py --split all --no-llm

Outputs:
    data/out/grounding/predictions_{split}.jsonl
    reports/grounding_eval_{split}.json
    reports/grounding_eval_{split}.md
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from cognitive_tool_agent.evaluation.grounding_metrics import (
    GroundingMetrics,
    PredictionRow,
    compute_grounding_metrics,
)

console = Console()

_VARIANTS = ["deterministic", "grounding_llm_v1"]
_SPLITS = ["train", "dev", "test"]


def _load_grounding_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _run_llm_grounding(
    row: dict[str, Any],
    grounding_node,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, bool, str | None, float | None, float | None, float | None]:
    """Call LLM grounding node. Returns (resolved, raw_output, schema_valid, error, confidence, latency_ms, cost_usd)."""
    try:
        result = grounding_node.run_from_grounding_row(row)
        return (
            result.resolved_args,
            result.llm_raw,
            True,
            None,
            result.confidence,
            result.latency_ms,
            result.cost_usd,
        )
    except Exception as exc:
        raw_text = getattr(exc, "raw_text", None)
        return (
            None,
            {"raw_text": raw_text or str(exc)},
            False,
            str(exc),
            None,
            None,
            None,
        )


def _evaluate_split(
    split: str,
    grounding_dir: Path,
    out_dir: Path,
    reports_dir: Path,
    no_llm: bool,
    limit: int | None,
    grounding_node,
) -> list[GroundingMetrics]:
    eval_path = grounding_dir / f"grounding_eval_{split}.jsonl"
    if not eval_path.exists():
        console.print(f"  [yellow]⚠ {eval_path} not found — skipping {split}[/yellow]")
        return []

    raw_rows = _load_grounding_rows(eval_path)
    if limit is not None:
        raw_rows = raw_rows[:limit]

    console.print(f"\n[bold]{split}[/bold]: {len(raw_rows)} rows")

    prediction_rows: list[PredictionRow] = []

    for raw in raw_rows:
        # Offline deterministic projection latency (dict-read only, not full node runtime)
        _t0 = time.monotonic()
        det_resolved = raw.get("current_deterministic_args") or {}
        det_latency_ms = (time.monotonic() - _t0) * 1000.0

        llm_resolved = None
        llm_raw = None
        llm_schema_valid = None
        llm_error = None
        llm_confidence = None
        llm_latency_ms = None
        llm_cost_usd = None

        if not no_llm and grounding_node is not None:
            (
                llm_resolved,
                llm_raw,
                llm_schema_valid,
                llm_error,
                llm_confidence,
                llm_latency_ms,
                llm_cost_usd,
            ) = _run_llm_grounding(raw, grounding_node)

            if not llm_schema_valid:
                llm_resolved = {}

        prediction_rows.append(
            PredictionRow(
                id=raw["id"],
                split=split,
                target_args=raw.get("target_args") or {},
                target_fields=raw.get("target_fields") or [],
                available_state=raw.get("available_state") or {},
                deterministic_resolved=det_resolved,
                deterministic_schema_valid=True,
                deterministic_latency_ms=det_latency_ms,
                llm_resolved=llm_resolved,
                llm_raw=llm_raw,
                llm_schema_valid=llm_schema_valid,
                llm_error=llm_error,
                llm_confidence=llm_confidence,
                llm_latency_ms=llm_latency_ms,
                llm_cost_usd=llm_cost_usd,
            )
        )

    preds_path = out_dir / f"predictions_{split}.jsonl"
    with open(preds_path, "w", encoding="utf-8") as f:
        for pred in prediction_rows:
            f.write(pred.model_dump_json() + "\n")
    console.print(f"  [dim]→ {preds_path}[/dim]")

    variants_to_run = ["deterministic"]
    if not no_llm and grounding_node is not None:
        variants_to_run.append("grounding_llm_v1")

    metrics_list: list[GroundingMetrics] = []
    for variant in variants_to_run:
        m = compute_grounding_metrics(prediction_rows, variant=variant, split=split)
        metrics_list.append(m)

    tau_timing = _load_tau_timing(grounding_dir, split)
    _write_split_report(split, metrics_list, prediction_rows, reports_dir, tau_timing)
    return metrics_list


def _load_tau_timing(grounding_dir: Path, split: str) -> dict[str, Any] | None:
    """Load simulation_timings.jsonl and compute aggregate tau-bench timing for the split.

    Returns None if the file is not found (optional artifact — skip quietly).
    """
    timings_path = grounding_dir.parent / "simulation_timings.jsonl"
    if not timings_path.exists():
        return None
    rows = []
    with open(timings_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                if row.get("split") == split:
                    rows.append(row)
    if not rows:
        return None

    duration_vals = [r["tau_duration_seconds"] for r in rows if r.get("tau_duration_seconds") is not None]
    gen_total_vals = [r["tau_agent_generation_time_seconds_total"] for r in rows if r.get("tau_agent_generation_time_seconds_total") is not None]
    gen_turns_vals = [r["tau_agent_generation_turns"] for r in rows if r.get("tau_agent_generation_turns") is not None]
    span_vals = [r["message_span_seconds"] for r in rows if r.get("message_span_seconds") is not None]

    total_gen_turns = sum(gen_turns_vals)
    total_gen_seconds = sum(gen_total_vals) if gen_total_vals else None
    gen_avg = (total_gen_seconds / total_gen_turns) if (total_gen_seconds is not None and total_gen_turns > 0) else None

    dur_total = sum(duration_vals) if duration_vals else None
    dur_avg = (dur_total / len(duration_vals)) if duration_vals else None
    span_total = sum(span_vals) if span_vals else None
    span_avg = (span_total / len(span_vals)) if span_vals else None

    return {
        "simulation_count": len(rows),
        "tau_duration_seconds_total": round(dur_total, 3) if dur_total is not None else None,
        "tau_duration_seconds_avg": round(dur_avg, 3) if dur_avg is not None else None,
        "tau_agent_generation_time_seconds_total": round(total_gen_seconds, 3) if total_gen_seconds is not None else None,
        "tau_agent_generation_time_seconds_avg_per_assistant_turn": round(gen_avg, 3) if gen_avg is not None else None,
        "tau_agent_generation_turns_total": total_gen_turns,
        "message_span_seconds_total": round(span_total, 3) if span_total is not None else None,
        "message_span_seconds_avg": round(span_avg, 3) if span_avg is not None else None,
    }


def _sim_latency_table(
    predictions: list[PredictionRow],
) -> list[dict[str, Any]]:
    """Group rows by simulation key and compute per-simulation latency."""
    sims: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "turns": 0,
        "det_total_ms": 0.0,
        "llm_total_ms": 0.0,
    })
    for row in predictions:
        sim_key = row.id.rsplit(":turn:", 1)[0]
        sims[sim_key]["turns"] += 1
        if row.deterministic_latency_ms is not None:
            sims[sim_key]["det_total_ms"] += row.deterministic_latency_ms
        if row.llm_latency_ms is not None:
            sims[sim_key]["llm_total_ms"] += row.llm_latency_ms
    rows = []
    for sim_key, d in sorted(sims.items()):
        turns = d["turns"]
        llm_total = d["llm_total_ms"]
        rows.append({
            "sim_key": sim_key,
            "turns": turns,
            "det_total_ms": round(d["det_total_ms"], 3),
            "llm_total_ms": round(llm_total, 1),
            "llm_avg_ms_per_turn": round(llm_total / turns, 1) if turns else None,
        })
    return rows


def _write_split_report(
    split: str,
    metrics_list: list[GroundingMetrics],
    predictions: list[PredictionRow],
    reports_dir: Path,
    tau_timing: dict[str, Any] | None = None,
) -> None:
    generated_at = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sim_rows = _sim_latency_table(predictions)

    json_doc: dict[str, Any] = {
        "generated_at": generated_at,
        "split": split,
        "total_rows": len(predictions),
        "metrics": [m.model_dump() for m in metrics_list],
        "sim_latency": sim_rows,
    }
    if tau_timing is not None:
        json_doc["tau_baseline_timing"] = tau_timing

    json_path = reports_dir / f"grounding_eval_{split}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_doc, f, indent=2, ensure_ascii=False)

    md_path = reports_dir / f"grounding_eval_{split}.md"
    lines: list[str] = []
    w = lines.append
    w(f"# Grounding Evaluation — {split}")
    w("")
    w(f"_Generated: {generated_at}_")
    w("")
    w(f"**Total rows**: {len(predictions)}")
    w("")
    w("## Grounding Quality")
    w("")
    w("| variant | req_arg_match | field_prec | field_rec | field_f1 | missing | halluc | schema_ok | confidence | cost_usd |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for m in metrics_list:
        w(
            f"| `{m.variant}` "
            f"| {m.required_arg_match:.0%} "
            f"| {m.field_precision:.0%} "
            f"| {m.field_recall:.0%} "
            f"| {m.field_f1:.0%} "
            f"| {m.missing_field_rate:.0%} "
            f"| {m.hallucinated_id_rate:.0%} "
            f"| {m.schema_valid_rate:.0%} "
            f"| {f'{m.avg_confidence:.2f}' if m.avg_confidence is not None else '—'} "
            f"| {f'{m.estimated_cost_usd:.4f}' if m.estimated_cost_usd is not None else '—'} |"
        )
    w("")
    w("## Grounding Node Latency")
    w("")
    w("| variant | avg_latency_s |")
    w("|---|---:|")
    for m in metrics_list:
        lat = f"{m.avg_latency_ms / 1000:.2f}" if m.avg_latency_ms is not None else "—"
        w(f"| `{m.variant}` | {lat} |")
    w("")
    w("## Per-Simulation Grounding Latency")
    w("")
    w("| sim_key | turns | det_total_s | llm_total_s | llm_avg_s_per_turn |")
    w("|---|---:|---:|---:|---:|")
    for s in sim_rows:
        llm_avg = f"{s['llm_avg_ms_per_turn'] / 1000:.3f}" if s["llm_avg_ms_per_turn"] is not None else "—"
        w(
            f"| `{s['sim_key']}` "
            f"| {s['turns']} "
            f"| {s['det_total_ms'] / 1000:.3f} "
            f"| {s['llm_total_ms'] / 1000:.3f} "
            f"| {llm_avg} |"
        )
    if tau_timing is not None:
        w("")
        w("## Baseline Tau-Bench Timing")
        w("")
        w(f"_simulation_count_: {tau_timing['simulation_count']}")
        w("")
        w("| metric | value |")
        w("|---|---:|")
        w(f"| tau_duration_seconds_total | {tau_timing.get('tau_duration_seconds_total', '—')} |")
        w(f"| tau_duration_seconds_avg | {tau_timing.get('tau_duration_seconds_avg', '—')} |")
        w(f"| tau_agent_generation_time_seconds_total | {tau_timing.get('tau_agent_generation_time_seconds_total', '—')} |")
        w(f"| tau_agent_generation_time_seconds_avg_per_assistant_turn | {tau_timing.get('tau_agent_generation_time_seconds_avg_per_assistant_turn', '—')} |")
        w(f"| tau_agent_generation_turns_total | {tau_timing.get('tau_agent_generation_turns_total', '—')} |")
        w(f"| message_span_seconds_total | {tau_timing.get('message_span_seconds_total', '—')} |")
        w(f"| message_span_seconds_avg | {tau_timing.get('message_span_seconds_avg', '—')} |")
        w("")
        w("_Note: `message_span_seconds` = timestamp span from first to last message (includes user/tool/framework overhead)._")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    console.print(f"  [dim]→ {md_path}[/dim]")


def _print_proof_table(all_metrics: dict[str, list[GroundingMetrics]]) -> None:
    console.print("\n[bold]Grounding-Only Evaluation[/bold]\n")
    table = Table(box=box.SIMPLE)
    table.add_column("variant", style="cyan", min_width=22)
    active_splits = [s for s in _SPLITS if s in all_metrics]
    for split in active_splits:
        table.add_column(f"{split}_acc", justify="right", min_width=10)
        table.add_column(f"{split}_s", justify="right", min_width=9)

    variants_seen: list[str] = []
    for split_metrics in all_metrics.values():
        for m in split_metrics:
            if m.variant not in variants_seen:
                variants_seen.append(m.variant)

    for variant in variants_seen:
        row_vals = []
        for split in active_splits:
            match = next((m for m in all_metrics[split] if m.variant == variant), None)
            if match is None:
                row_vals.extend(["[dim]—[/dim]", "[dim]—[/dim]"])
            else:
                v = match.required_arg_match
                color = "green" if v >= 0.60 else "yellow" if v >= 0.35 else "red"
                acc_str = f"[{color}]{v:.0%}[/{color}]"
                lat = match.avg_latency_ms
                lat_str = f"{lat / 1000:.2f}" if lat is not None else "[dim]—[/dim]"
                row_vals.extend([acc_str, lat_str])
        table.add_row(variant, *row_vals)

    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Grounding evaluation — Pass 2")
    parser.add_argument(
        "--split",
        choices=["train", "dev", "test", "all"],
        default="dev",
        help="Which split to evaluate (default: dev)",
    )
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model (used when --no-llm is not set)")
    parser.add_argument("--limit", type=int, default=None, help="Limit rows per split")
    parser.add_argument("--grounding-dir", default="data/out/grounding")
    parser.add_argument("--out-dir", default="data/out/grounding")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM calls; run deterministic baseline only")
    args = parser.parse_args()

    grounding_dir = Path(args.grounding_dir)
    out_dir = Path(args.out_dir)
    reports_dir = Path(args.reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    splits = _SPLITS if args.split == "all" else [args.split]
    limit_label = f"limit={args.limit}" if args.limit else "all rows"
    llm_label = "deterministic only (--no-llm)" if args.no_llm else f"deterministic + LLM ({args.model})"

    console.print(Panel.fit(
        "[bold cyan]Grounding Evaluation — Pass 2[/bold cyan]\n"
        f"Splits:  [green]{', '.join(splits)}[/green]\n"
        f"Mode:    [yellow]{llm_label}[/yellow]\n"
        f"Rows:    [yellow]{limit_label}[/yellow]",
        border_style="cyan",
    ))

    grounding_node = None
    if not args.no_llm:
        try:
            from cognitive_tool_agent.nodes.grounding_llm import GroundingLLMNode
            from cognitive_tool_agent.adapters.openai_grounding_adapter import OpenAIGroundingAdapter
            grounding_node = GroundingLLMNode(adapter=OpenAIGroundingAdapter(model=args.model))
            console.print(f"  [green]✓ LLM grounding node ready ({args.model})[/green]")
        except ImportError as exc:
            console.print(f"  [yellow]⚠ LLM node unavailable ({exc}) — running deterministic only[/yellow]")
            args.no_llm = True

    all_metrics: dict[str, list[GroundingMetrics]] = {}
    for split in splits:
        metrics = _evaluate_split(
            split=split,
            grounding_dir=grounding_dir,
            out_dir=out_dir,
            reports_dir=reports_dir,
            no_llm=args.no_llm,
            limit=args.limit,
            grounding_node=grounding_node,
        )
        if metrics:
            all_metrics[split] = metrics

    if all_metrics:
        _print_proof_table(all_metrics)

    console.print("\n[green]Done.[/green]")


if __name__ == "__main__":
    main()
