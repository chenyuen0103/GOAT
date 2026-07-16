from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def build_command(args: argparse.Namespace) -> list[str]:
    cmd = [sys.executable, "run_missing_seeds.py"]
    forward_args = list(args.forward_args)
    if forward_args and forward_args[0] == "--":
        forward_args = forward_args[1:]
    for item in forward_args:
        cmd.append(item)
    return cmd


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compatibility wrapper for missing-seed reruns."
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "forward_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to run_missing_seeds.py. Prefix with -- if needed.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cmd = build_command(args)
    print(" ".join(cmd))
    if args.dry_run:
        return 0
    if not Path("run_missing_seeds.py").exists():
        raise FileNotFoundError("run_missing_seeds.py not found in current directory")
    subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
