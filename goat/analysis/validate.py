from __future__ import annotations

import argparse
import json
from typing import Sequence

from goat.analysis.validation import ValidationSummary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read a GOAT validation.json summary.")
    parser.add_argument("path", help="Path to validation.json")
    parser.add_argument(
        "--allow-failed",
        action="store_true",
        help="Return success even when the validation payload has passed=false.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = ValidationSummary.from_file(args.path)
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    if not summary.passed and not args.allow_failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

