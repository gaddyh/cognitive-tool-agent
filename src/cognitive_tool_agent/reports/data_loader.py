from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_reports_data(out_dir: Path | str) -> dict[str, Any]:
    """Load all five converter output files from out_dir into plain dicts/lists."""
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

    return {
        "tool_registry": read_json("tool_registry.json"),
        "action_sequences": read_jsonl("action_sequence.jsonl"),
        "turn_supervision": read_jsonl("turn_supervision.jsonl"),
        "failure_rows": read_jsonl("failure_rows.jsonl"),
        "conversion_summary": read_json("conversion_summary.json"),
    }
