from __future__ import annotations

import json
from pathlib import Path

from ..schemas.trace import CognitiveTrace


class TraceWriter:
    def write(self, traces: list[CognitiveTrace], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for trace in traces:
                f.write(trace.model_dump_json() + "\n")

    def load(self, path: Path) -> list[CognitiveTrace]:
        traces: list[CognitiveTrace] = []
        with open(path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                    traces.append(CognitiveTrace.model_validate(raw))
                except Exception as exc:
                    raise ValueError(
                        f"Failed to parse trace line {line_num} in {path}: {exc}"
                    ) from exc
        return traces
