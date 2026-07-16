from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def write_json(path: str | Path, payload: Any, *, indent: int = 2) -> None:
    Path(path).write_text(json.dumps(payload, indent=indent, sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    for line in Path(path).read_text().splitlines():
        if line.strip():
            yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    text = "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows)
    Path(path).write_text(text)


def append_jsonl(path: str | Path, row: Mapping[str, Any]) -> None:
    with Path(path).open("a") as handle:
        handle.write(json.dumps(dict(row), sort_keys=True) + "\n")

