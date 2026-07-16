from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from goat.core.artifacts import ArtifactPaths
from goat.core.schema import ExperimentConfig


DATASET_CHOICES = ("mnist", "portraits", "covtype", "color_mnist")


def build_legacy_experiment_command(
    config: ExperimentConfig,
    *,
    paths: ArtifactPaths | None = None,
    legacy_module: str = "experiment_refrac.py",
) -> list[str]:
    paths = paths or ArtifactPaths.from_env()
    cmd = [
        sys.executable,
        legacy_module,
        "--dataset",
        config.dataset,
        "--gt-domains",
        str(config.gt_domains),
        "--generated-domains",
        str(config.generated_domains),
        "--seed",
        str(config.seed),
        "--label-source",
        config.label_source,
        "--em-match",
        config.em_match,
        "--em-select",
        config.em_select,
        "--log-root",
        str(paths.log_root),
        "--plot-root",
        str(paths.plot_root),
    ]
    if config.small_dim is not None:
        cmd.extend(["--small-dim", str(config.small_dim)])
    extra = dict(config.extra)
    if "rotation_angle" in extra:
        cmd.extend(["--rotation-angle", str(extra["rotation_angle"])])
    if "goat_gen_methods" in extra:
        cmd.extend(["--goat-gen-methods", str(extra["goat_gen_methods"])])
    return cmd


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Canonical GOAT experiment runner with legacy delegation."
    )
    parser.add_argument("--dataset", choices=DATASET_CHOICES, default="mnist")
    parser.add_argument("--gt-domains", type=int, default=0)
    parser.add_argument("--generated-domains", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--small-dim", type=int, default=2048)
    parser.add_argument("--label-source", default="pseudo")
    parser.add_argument("--em-match", default="prototypes")
    parser.add_argument("--em-select", default="bic")
    parser.add_argument("--rotation-angle", type=int, default=None)
    parser.add_argument("--goat-gen-methods", default=None)
    parser.add_argument("--legacy-module", default="experiment_refrac.py")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    extra = {}
    if args.rotation_angle is not None:
        extra["rotation_angle"] = args.rotation_angle
    if args.goat_gen_methods is not None:
        extra["goat_gen_methods"] = args.goat_gen_methods
    config = ExperimentConfig(
        dataset=args.dataset,
        seed=args.seed,
        gt_domains=args.gt_domains,
        generated_domains=args.generated_domains,
        small_dim=args.small_dim,
        label_source=args.label_source,
        em_match=args.em_match,
        em_select=args.em_select,
        extra=extra,
    )
    cmd = build_legacy_experiment_command(
        config,
        paths=ArtifactPaths.from_env(Path.cwd()),
        legacy_module=args.legacy_module,
    )
    print(" ".join(cmd))
    if args.dry_run:
        return 0
    subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

