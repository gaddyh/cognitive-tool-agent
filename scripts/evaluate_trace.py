#!/usr/bin/env python3
"""Load saved CognitiveTrace JSONL and print failure analysis.

Usage:
    python scripts/evaluate_trace.py --traces path/to/traces.jsonl \
                                     --dataset data/dev/tool_calling_micro.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.table import Table
from rich import box

from cognitive_tool_agent.datasets.loader import load_jsonl
from cognitive_tool_agent.evals.evaluator import Evaluator
from cognitive_tool_agent.graph_builder.failure_analyzer import FailureAnalyzer
from cognitive_tool_agent.schemas.trace import CognitiveTrace


console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved traces against a dataset")
    parser.add_argument("--traces", required=True, help="JSONL file of CognitiveTrace objects")
    parser.add_argument("--dataset", required=True, help="JSONL dataset file (DatasetRow objects)")
    parser.add_argument("--candidate-id", default="external", help="Candidate ID label")
    args = parser.parse_args()

    traces = _load_traces(Path(args.traces))
    rows = load_jsonl(Path(args.dataset))

    if len(traces) != len(rows):
        console.print(
            f"[yellow]Warning: {len(traces)} traces vs {len(rows)} dataset rows[/yellow]"
        )

    console.print(f"\nEvaluating [bold]{len(traces)}[/bold] traces against dataset...\n")

    evaluator = Evaluator()
    scores = evaluator.score(traces, rows)

    console.print("[bold]Metric Scores[/bold]")
    for metric, value in scores.items():
        color = "green" if value >= 0.7 else "yellow" if value >= 0.4 else "red"
        console.print(f"  {metric}: [{color}]{value:.1%}[/{color}]")

    failure_map = FailureAnalyzer().analyze(args.candidate_id, traces, rows)

    console.print(
        f"\n[bold]Failures:[/bold] {failure_map.failure_count} / {failure_map.total_rows}  "
        f"dominant stage=[red]{failure_map.dominant_failure_stage}[/red]  "
        f"type=[red]{failure_map.dominant_failure_type}[/red]"
    )

    if failure_map.failures:
        table = Table(title="Row Failures", box=box.SIMPLE)
        table.add_column("Row ID", style="cyan")
        table.add_column("Expected")
        table.add_column("Actual")
        table.add_column("Stage", style="red")
        table.add_column("Type", style="red")
        table.add_column("Explanation")
        for f in failure_map.failures:
            table.add_row(
                f.row_id,
                f.expected_action,
                f.actual_action or "-",
                f.failure_stage,
                f.failure_type,
                f.explanation[:60] + ("..." if len(f.explanation) > 60 else ""),
            )
        console.print(table)


def _load_traces(path: Path) -> list[CognitiveTrace]:
    traces: list[CognitiveTrace] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                traces.append(CognitiveTrace.model_validate(raw))
            except Exception as exc:
                raise ValueError(f"Line {line_num} in {path}: {exc}") from exc
    return traces


if __name__ == "__main__":
    main()
