import json
from pathlib import Path
from ..schemas.dataset import DatasetRow


def load_jsonl(path: str | Path) -> list[DatasetRow]:
    rows: list[DatasetRow] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                rows.append(DatasetRow.model_validate(raw))
            except Exception as exc:
                raise ValueError(f"Failed to parse line {line_num} in {path}: {exc}") from exc
    return rows
