from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def atomic_write_text(path: str | Path, text: str) -> None:
    """Write text with an atomic same-directory replace."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_json(path: str | Path, payload: Any, *, indent: int = 2) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, indent=indent, sort_keys=True, allow_nan=False) + "\n",
    )


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    for line in Path(path).read_text().splitlines():
        if line.strip():
            yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    text = "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows)
    atomic_write_text(path, text)


def append_jsonl(path: str | Path, row: Mapping[str, Any]) -> None:
    with Path(path).open("a") as handle:
        handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
