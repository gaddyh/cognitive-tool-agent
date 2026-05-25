from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .argument_emergence import compute_argument_emergence
from .cognitive_burden import compute_cognitive_burden
from .data_loader import load_reports_data
from .dataset_summary import compute_extended_summary
from .failure_heatmap import compute_failure_heatmap


class ReportBuilder:
    """
    Reads converter output from out_dir, computes all report sections,
    and writes five files to reports_dir:

      cognitive_dataset_report.md
      cognitive_dataset_report.json
      cognitive_action_topology.csv
      failure_heatmap.csv
      argument_emergence.csv
    """

    def run(
        self,
        out_dir: Path | str,
        reports_dir: Path | str,
        source_label: str = "",
        train_only: bool = True,
    ) -> dict[str, Any]:
        out_dir = Path(out_dir)
        reports_dir = Path(reports_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)

        data = load_reports_data(out_dir, train_only=train_only)

        summary = compute_extended_summary(data)
        burden = compute_cognitive_burden(data)
        emergence = compute_argument_emergence(data)
        heatmap = compute_failure_heatmap(data)

        generated_at = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        if train_only and data.get("_train_sim_count") is not None:
            effective_source = source_label or f"train split ({data['_train_sim_count']} sims)"
            data_scope = "train_only"
        else:
            effective_source = source_label or str(out_dir)
            data_scope = "all_splits"

        full_report: dict[str, Any] = {
            "generated_at": generated_at,
            "source": effective_source,
            "experimental_boundary": {
                "artifact_type": "design",
                "data_scope": data_scope,
                "allowed_to_influence_graph": True,
            },
            "dataset_summary": summary,
            "cognitive_action_topology": list(burden.values()),
            "argument_emergence": emergence,
            "failure_heatmap": heatmap,
        }

        _write_json(reports_dir / "cognitive_dataset_report.json", full_report)
        _write_topology_csv(reports_dir / "cognitive_action_topology.csv", burden)
        _write_heatmap_csv(reports_dir / "failure_heatmap.csv", heatmap)
        _write_emergence_csv(reports_dir / "argument_emergence.csv", emergence)
        _write_markdown(
            reports_dir / "cognitive_dataset_report.md",
            summary, burden, emergence, heatmap,
            generated_at, source_label or str(out_dir),
        )

        return full_report


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------

def _write_topology_csv(path: Path, burden: dict[str, dict]) -> None:
    if not burden:
        return
    fields = [
        "tool_name", "tool_type", "required_args_count", "usage_count",
        "avg_turn_distance", "avg_chain_depth", "grounding_fraction",
        "extraction_burden", "memory_burden", "readiness_burden",
        "reasoning_burden", "grounding_burden", "complexity_score",
    ]
    rows = sorted(burden.values(), key=lambda r: -r["complexity_score"])
    _write_csv(path, fields, rows)


def _write_heatmap_csv(path: Path, rows: list[dict]) -> None:
    _write_csv(path, ["dimension", "value", "count"], rows)


def _write_emergence_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = [
        "arg_name", "total_instances",
        "appears_explicitly", "appears_explicitly_pct",
        "requires_tool_chaining", "requires_tool_chaining_pct",
        "requires_grounding", "requires_grounding_pct",
        "requires_inference", "requires_inference_pct",
    ]
    _write_csv(path, fields, rows)


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _write_markdown(
    path: Path,
    summary: dict,
    burden: dict[str, dict],
    emergence: list[dict],
    heatmap: list[dict],
    generated_at: str,
    source: str,
) -> None:
    lines: list[str] = []
    w = lines.append

    w(f"# Cognitive Dataset Report")
    w(f"")
    w(f"_Generated: {generated_at} — Source: `{source}`_")
    w(f"")

    # ── Dataset Summary ──────────────────────────────────────────────────────
    w("## Dataset Summary")
    w("")
    w("| Metric | Value |")
    w("|---|---:|")
    w(f"| Tasks | {summary['tasks_count']} |")
    w(f"| Simulations | {summary['simulations_count']} |")
    w(f"| Messages | {summary['messages_count']} |")
    w(f"| Expected actions | {summary['expected_actions_count']} |")
    w(f"| Actual tool calls | {summary['actual_tool_calls_count']} |")
    w(f"| Matched actions | {summary['matched_actions_count']} |")
    w(f"| Failed actions | {summary['failed_actions_count']} |")
    w(f"| **Tool entropy** | **{summary['tool_entropy_bits']} bits** |")
    w(f"| **Avg tools / simulation** | **{summary['avg_tools_per_simulation']}** |")
    w(f"| **Avg turns before write** | **{summary['avg_turns_before_write_action']}** |")
    w(f"| Read tool calls | {summary['read_tool_calls']} |")
    w(f"| Write tool calls | {summary['write_tool_calls']} |")
    w(f"| **Read / write ratio** | **{summary['read_write_ratio']}** |")
    w("")

    # ── Cognitive Burden Breakdown ────────────────────────────────────────────
    w("## Cognitive Burden Breakdown")
    w("")
    w("Tools ranked by complexity score (descending).")
    w("")
    w("| Tool | Type | Args | Extract | Memory | Readiness | Reasoning | Grounding | Score |")
    w("|---|---|---:|---|---|---|---|---|---:|")
    sorted_tools = sorted(burden.values(), key=lambda r: -r["complexity_score"])
    for t in sorted_tools:
        w(
            f"| `{t['tool_name']}` | {t['tool_type']} | {t['required_args_count']} "
            f"| {t['extraction_burden']} | {t['memory_burden']} | {t['readiness_burden']} "
            f"| {t['reasoning_burden']} | {t['grounding_burden']} | **{t['complexity_score']}** |"
        )
    w("")

    w("### Burden signal legend")
    w("")
    w("| Signal | Meaning |")
    w("|---|---|")
    w("| extraction burden | number of required arguments |")
    w("| memory burden | avg turns between user hint and tool call |")
    w("| readiness burden | write tools require explicit confirmation |")
    w("| reasoning burden | depth of preceding tool calls (chain length) |")
    w("| grounding burden | fraction of args not explicitly provided by user |")
    w("")

    # ── Argument Emergence Matrix ─────────────────────────────────────────────
    w("## Argument Emergence Matrix")
    w("")
    w("How each required argument reaches the agent — as a percentage of all instances.")
    w("")
    w("| Argument | Instances | Explicit % | Tool-Chained % | Grounding % | Inference % |")
    w("|---|---:|---:|---:|---:|---:|")
    for row in emergence:
        w(
            f"| `{row['arg_name']}` | {row['total_instances']} "
            f"| {row['appears_explicitly_pct']}% "
            f"| {row['requires_tool_chaining_pct']}% "
            f"| {row['requires_grounding_pct']}% "
            f"| {row['requires_inference_pct']}% |"
        )
    w("")
    w("> **Explicit** — value appeared verbatim in user message  ")
    w("> **Tool-Chained** — value came from a preceding tool result  ")
    w("> **Grounding** — action matched but not directly extractable (NL→structured)  ")
    w("> **Inference** — not resolved (failed actions or ambiguous)")
    w("")

    # ── Failure Heatmap ───────────────────────────────────────────────────────
    w("## Failure Heatmap")
    w("")

    def _section(label: str, dim: str) -> None:
        w(f"### By {label}")
        w("")
        w("| Value | Failures |")
        w("|---|---:|")
        for row in heatmap:
            if row["dimension"] == dim:
                w(f"| `{row['value']}` | {row['count']} |")
        w("")

    _section("Tool", "tool")
    _section("Cognitive Stage", "stage")
    _section("Read / Write", "read_write")
    _section("Argument", "argument")

    # ── Complexity Score Ranking ──────────────────────────────────────────────
    w("## Cognitive Complexity Score Ranking")
    w("")
    w("```")
    w("score = required_args_count")
    w("      + memory_burden_score  (low=0 medium=1 high=2 very_high=3)")
    w("      + write_penalty        (write=2, else 0)")
    w("      + grounding_penalty    (grounding_fraction × n_args, rounded)")
    w("      + confirmation_penalty (write=1, else 0)")
    w("      + chain_depth_score    (min(avg_chain_depth, 3))")
    w("```")
    w("")
    w("| Rank | Tool | Score |")
    w("|---:|---|---:|")
    for rank, t in enumerate(sorted_tools, 1):
        w(f"| {rank} | `{t['tool_name']}` | {t['complexity_score']} |")
    w("")

    path.write_text("\n".join(lines), encoding="utf-8")
