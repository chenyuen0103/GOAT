from __future__ import annotations

from pathlib import Path
from typing import Iterable

from goat.core.io import read_jsonl
from goat.core.io import read_json
from goat.core.schema import RunRecord


def infer_dataset_from_path(path: str | Path) -> str:
    parts = Path(path).parts
    if "mnist" in parts:
        return "mnist"
    if "portraits" in parts:
        return "portraits"
    if "covtype" in parts:
        return "covtype"
    if "color_mnist" in parts:
        return "color_mnist"
    return "unknown"


def load_legacy_curve_records(path: str | Path) -> list[dict]:
    return list(read_jsonl(path))


def load_run_records(path: str | Path, *, dataset: str | None = None) -> list[RunRecord]:
    path = Path(path)
    if path.name == "run.json":
        return [RunRecord.from_dict(read_json(path))]
    if path.is_dir():
        return [
            RunRecord.from_dict(read_json(run_path))
            for run_path in sorted(path.rglob("run.json"))
        ]
    dataset = dataset or infer_dataset_from_path(path)
    return [
        RunRecord.from_legacy_curve_record(
            row,
            dataset=dataset,
            artifacts={"source_jsonl": str(path)},
        )
        for row in read_jsonl(path)
    ]


def method_names(records: Iterable[RunRecord]) -> list[str]:
    names = set()
    for record in records:
        names.update(record.methods)
    return sorted(names)
