from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List


def to_jsonable(value: Any) -> Any:
    """Convert common Python objects into JSON-serializable types."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, Path):
        return str(value)
    return str(value)


def save_json(path: str, data: Any) -> None:
    """Save *data* as UTF-8 JSON with pretty indentation."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, ensure_ascii=False, indent=2)


def append_csv(path: str, fieldnames: List[str], row: Dict[str, Any]) -> None:
    """Append a row to a CSV file; create the header on first write."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
