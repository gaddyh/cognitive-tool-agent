"""Pre-refactor equivalence fixture generator.

Run this ONCE against the unmodified codebase (before any refactor changes)
to capture the ground-truth behavioral outputs of the full pipeline.

Output: tests/fixtures/full_pipeline_expected.json

The full-pipeline equivalence test loads this fixture and asserts the
refactored executor produces identical outputs.  Never regenerate this
file from post-refactor code.

Usage (from repo root):
    python scripts/generate_equivalence_fixture.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "dev" / "tool_calling_micro.jsonl"
OUT_PATH = ROOT / "tests" / "fixtures" / "full_pipeline_expected.json"


def main() -> None:
    from cognitive_tool_agent.datasets.loader import load_jsonl
    from cognitive_tool_agent.graph.cognitive_graph import GraphExecutor
    from cognitive_tool_agent.graph_builder.graph_candidate_generator import make_full_pipeline
    from cognitive_tool_agent.schemas.experiment import ExperimentSpec
    from cognitive_tool_agent.tools.fake_tools import DEFAULT_REGISTRY

    rows = load_jsonl(DATA_PATH)
    candidate = make_full_pipeline()
    executor = GraphExecutor()
    experiment = ExperimentSpec(graph=candidate.graph_spec)

    records: list[dict] = []
    for row in rows:
        trace = executor.run(experiment, row, DEFAULT_REGISTRY)
        records.append({
            "row_id": row.id,
            "action_type": trace.action.action_type if trace.action else None,
            "tool_name": trace.action.tool_name if trace.action else None,
            "tool_arguments": trace.action.tool_arguments if trace.action else None,
            "plan_next_action": trace.plan.next_action if trace.plan else None,
            "reasoning_selected_tool": trace.reasoning.selected_tool if trace.reasoning else None,
        })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(records, indent=2))
    print(f"Wrote {len(records)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
