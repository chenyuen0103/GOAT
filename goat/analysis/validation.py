from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from goat.core.io import read_json


@dataclass(frozen=True)
class ValidationSummary:
    """Typed view of a validation.json artifact."""

    passed: bool
    expected_result_rows: int | None = None
    actual_result_rows: int | None = None
    expected_methods: tuple[str, ...] = ()
    missing_result_rows: int | None = None
    duplicate_result_rows: int | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ValidationSummary":
        return cls(
            passed=bool(payload.get("passed", False)),
            expected_result_rows=_maybe_int(payload.get("expected_result_rows")),
            actual_result_rows=_maybe_int(payload.get("actual_result_rows")),
            expected_methods=tuple(str(x) for x in payload.get("expected_methods", ())),
            missing_result_rows=_maybe_int(payload.get("missing_result_rows")),
            duplicate_result_rows=_maybe_int(payload.get("duplicate_result_rows")),
            raw=dict(payload),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "ValidationSummary":
        return cls.from_dict(read_json(path))

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "expected_result_rows": self.expected_result_rows,
            "actual_result_rows": self.actual_result_rows,
            "expected_methods": list(self.expected_methods),
            "missing_result_rows": self.missing_result_rows,
            "duplicate_result_rows": self.duplicate_result_rows,
            "raw": dict(self.raw),
        }


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)

