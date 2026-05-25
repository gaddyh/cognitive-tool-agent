from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_reports_data(out_dir: Path | str, train_only: bool = False) -> dict[str, Any]:
    """Load converter output files from out_dir into plain dicts/lists.

    When train_only=True, all behavioral data is restricted to the train split,
    enforcing the EDD boundary: no dev/test signals may influence design artifacts.
    tool_registry, conversion_summary, and split_manifest remain global.
    """
    out_dir = Path(out_dir)

    def read_json(fname: str) -> Any:
        p = out_dir / fname
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    def read_jsonl(fname: str) -> list[dict]:
        p = out_dir / fname
        if not p.exists():
            return []
        rows = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows

    train_ids: set[str] | None = None
    if train_only:
        ids_path = out_dir / "splits" / "train_simulation_ids.json"
        if ids_path.exists():
            train_ids = set(json.loads(ids_path.read_text(encoding="utf-8")))

    def filter_by_train(rows: list[dict]) -> list[dict]:
        if train_ids is None:
            return rows
        return [r for r in rows if r.get("simulation_id") in train_ids]

    if train_only and train_ids is not None:
        turn_supervision = read_jsonl("splits/train_supervision.jsonl")
    else:
        turn_supervision = read_jsonl("turn_supervision.jsonl")

    return {
        "tool_registry": read_json("tool_registry.json"),
        "action_sequences": filter_by_train(read_jsonl("action_sequence.jsonl")),
        "turn_supervision": turn_supervision,
        "failure_rows": filter_by_train(read_jsonl("failure_rows.jsonl")),
        "conversion_summary": read_json("conversion_summary.json"),
        "simulation_profiles": filter_by_train(read_jsonl("simulation_profiles.jsonl")),
        "scenario_distribution": read_json("scenario_distribution.json"),
        "split_manifest": read_json("split_manifest.json"),
        "_train_only": train_only,
        "_train_sim_count": len(train_ids) if train_ids is not None else None,
    }
