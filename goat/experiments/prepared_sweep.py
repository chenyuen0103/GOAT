from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


DATASET_SMALL_DIM = {
    "mnist": 2048,
    "portraits": 2048,
    "color_mnist": 2048,
    "covtype": 54,
}


@dataclass(frozen=True)
class PreparedSweepJob:
    dataset: str
    seed: int
    gt_domains: int
    generated_domains: int
    em_match: str
    rotation_angle: int = 45


def _common_command(
    args: argparse.Namespace,
    *,
    seed: int,
    gt_domains: int,
    generated_domains: int,
    em_match: str,
) -> list[str]:
    small_dim = int(args.small_dim or DATASET_SMALL_DIM[args.dataset])
    command = [
        sys.executable,
        str(args.legacy_module),
        "--dataset",
        str(args.dataset),
        "--seed",
        str(seed),
        "--gt-domains",
        str(gt_domains),
        "--generated-domains",
        str(generated_domains),
        "--small-dim",
        str(small_dim),
        "--label-source",
        str(args.label_source),
        "--em-match",
        str(em_match),
        "--em-select",
        str(args.em_select),
        "--em-seed-mode",
        str(args.em_seed_mode),
        "--em-seeds",
        *[str(value) for value in args.em_seeds],
        "--em-cov-types",
        *[str(value) for value in args.em_cov_types],
        "--em-pca-dims",
        *[str(value) for value in args.em_pca_dims],
        "--pseudo-confidence-q",
        str(args.pseudo_confidence_q),
        "--goat-gen-methods",
        str(args.goat_gen_methods),
        "--prepared-artifact-root",
        str(Path(args.prepared_artifact_root).expanduser()),
        "--log-root",
        str(Path(args.log_root).expanduser()),
        "--plot-root",
        str(Path(args.plot_root).expanduser()),
    ]
    if args.dataset == "mnist":
        command.extend(["--rotation-angle", str(args.rotation_angle)])
    if not args.with_plots:
        command.append("--no-plots")
    if args.em_ensemble:
        command.append("--em-ensemble")
    return command


def build_prepare_commands(args: argparse.Namespace) -> list[list[str]]:
    # Raw EM fitting is mapping-independent. Use agreement mapping during the
    # preparation process so high-dimensional source covariance statistics are
    # not computed merely to populate the raw artifact.
    mapping = "pseudo"
    commands: list[list[str]] = []
    for seed in args.seeds:
        for gt_domains in args.gt_domains:
            command = _common_command(
                args,
                seed=int(seed),
                gt_domains=int(gt_domains),
                generated_domains=0,
                em_match=mapping,
            )
            command.append("--prepare-only")
            commands.append(command)
    return commands


def build_worker_jobs(args: argparse.Namespace) -> list[PreparedSweepJob]:
    jobs: list[PreparedSweepJob] = []
    for seed in args.seeds:
        for gt_domains in args.gt_domains:
            for generated_domains in args.generated_domains:
                mappings: Iterable[str]
                if int(generated_domains) == 0:
                    mappings = args.em_matches[:1]
                else:
                    mappings = args.em_matches
                for em_match in mappings:
                    jobs.append(
                        PreparedSweepJob(
                            dataset=args.dataset,
                            seed=int(seed),
                            gt_domains=int(gt_domains),
                            generated_domains=int(generated_domains),
                            em_match=str(em_match),
                            rotation_angle=int(args.rotation_angle),
                        )
                    )
    return jobs


def build_worker_commands(args: argparse.Namespace) -> list[list[str]]:
    commands: list[list[str]] = []
    for job in build_worker_jobs(args):
        command = _common_command(
            args,
            seed=job.seed,
            gt_domains=job.gt_domains,
            generated_domains=job.generated_domains,
            em_match=job.em_match,
        )
        if (
            job.generated_domains > 0
            and job.em_match != str(args.em_matches[0])
        ):
            command.append("--skip-pooled-goat")
        command.append("--require-prepared-artifacts")
        commands.append(command)
    return commands


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare immutable encoded/EM artifacts once, then run each Gsyn and "
            "EM-mapping branch in a fresh subprocess."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_SMALL_DIM),
        default="mnist",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--gt-domains", nargs="+", type=int, default=[0])
    parser.add_argument("--generated-domains", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument(
        "--em-matches",
        nargs="+",
        choices=["prototypes", "pseudo"],
        default=["prototypes", "pseudo"],
    )
    parser.add_argument("--rotation-angle", type=int, default=45)
    parser.add_argument("--small-dim", type=int, default=None)
    parser.add_argument("--label-source", choices=["pseudo", "em"], default="pseudo")
    parser.add_argument("--em-select", choices=["bic", "cost", "ll"], default="bic")
    parser.add_argument("--em-ensemble", action="store_true")
    parser.add_argument("--em-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--em-seed-mode",
        choices=["offset", "absolute"],
        default="offset",
    )
    parser.add_argument(
        "--em-cov-types",
        nargs="+",
        choices=["diag", "full"],
        default=["diag"],
    )
    parser.add_argument("--em-pca-dims", nargs="+", default=["none"])
    parser.add_argument("--pseudo-confidence-q", type=float, default=0.9)
    parser.add_argument("--goat-gen-methods", default="w2")
    parser.add_argument(
        "--prepared-artifact-root",
        default=os.environ.get("GOAT_PREPARED_ARTIFACT_ROOT", "prepared_artifacts"),
    )
    parser.add_argument("--log-root", default=os.environ.get("LOG_ROOT", "logs_rerun"))
    parser.add_argument("--plot-root", default=os.environ.get("PLOT_ROOT", "plots_rerun"))
    parser.add_argument("--legacy-module", default="experiment_refrac.py")
    parser.add_argument("--with-plots", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _run_commands(commands: Iterable[list[str]], *, dry_run: bool) -> None:
    for command in commands:
        print(shlex.join(command), flush=True)
        if not dry_run:
            subprocess.run(command, check=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not Path(args.legacy_module).exists():
        raise FileNotFoundError(f"experiment entrypoint not found: {args.legacy_module}")

    if not args.skip_prepare:
        print("[prepared-sweep] phase=prepare", flush=True)
        _run_commands(build_prepare_commands(args), dry_run=args.dry_run)
    if args.prepare_only:
        return 0

    print("[prepared-sweep] phase=isolated-workers", flush=True)
    _run_commands(build_worker_commands(args), dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
