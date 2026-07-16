#!/usr/bin/env python3
import argparse
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Set

import pandas as pd


METHODS = ["goat", "goatcw", "eta", "ours_fr"]
DEFAULT_SEEDS = [0, 1, 2, 3, 4]

DATASET_TO_CLI = {
    "rotated_mnist": "mnist",
    "colored_mnist": "color_mnist",
    "portraits": "portraits",
    "covtype": "covtype",
}

DATASET_TO_LOG_DIR = {
    "rotated_mnist": "mnist",
    "colored_mnist": "color_mnist",
    "portraits": "portraits",
    "covtype": "covtype",
}

DATASET_TO_SMALL_DIM = {
    "rotated_mnist": 2048,
    "colored_mnist": 2048,
    "portraits": 2048,
    "covtype": 54,
}


@dataclass(frozen=True)
class Job:
    dataset: str
    degree: float
    gt: int
    gen: int
    seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rerun missing canonical pseudo/prototypes/bic non-ensemble jobs "
            "for seeds 0..4 based on results_all_with_settings.csv."
        )
    )
    parser.add_argument(
        "--csv",
        default="results_all_with_settings.csv",
        help="Path to the consolidated results CSV.",
    )
    parser.add_argument(
        "--log-root",
        default=os.environ.get("LOG_ROOT", "logs_rerun"),
        help="Root directory for logs.",
    )
    parser.add_argument(
        "--plot-root",
        default=os.environ.get("PLOT_ROOT", "plots_rerun"),
        help="Root directory for plots.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Expected seed set for each run family.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=METHODS,
        choices=METHODS,
        help="Method columns to require coverage for.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        choices=sorted(DATASET_TO_CLI),
        help="Optional dataset filter. Can be passed multiple times.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without executing them.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of missing jobs to run.",
    )
    parser.add_argument(
        "--goat-gen-methods",
        default="w2",
        help=(
            "Value passed through to experiment_refrac.py --goat-gen-methods. "
            "Default is 'w2' to skip GOAT-FR and GOAT-NAT during reruns."
        ),
    )
    parser.add_argument(
        "--include-mnist-90",
        action="store_true",
        help="Include rotated_mnist runs with degree 90. By default they are skipped.",
    )
    return parser.parse_args()


def canonical_filter(
    df: pd.DataFrame,
    seeds: Set[int],
    datasets: Optional[Set[str]],
    include_mnist_90: bool,
) -> pd.DataFrame:
    out = df[
        (~df["em_ensemble"].fillna(False))
        & (df["em_match"].astype(str) == "prototypes")
        & (df["em_select"].astype(str) == "bic")
        & (df["seed"].isin(seeds))
    ].copy()
    if datasets:
        out = out[out["dataset"].isin(datasets)].copy()
    if not include_mnist_90:
        out = out[
            ~(
                (out["dataset"] == "rotated_mnist")
                & (pd.to_numeric(out["degree"], errors="coerce") == 90)
            )
        ].copy()
    return out


def normalize_degree(value: object) -> float:
    if pd.isna(value):
        return math.nan
    return float(value)


def find_missing_jobs(plot_src: pd.DataFrame, methods: Iterable[str], expected_seeds: Set[int]) -> list[Job]:
    # One experiment run emits all methods for a given
    # (dataset, degree, gt, gen, seed) setting, so collect jobs in a set
    # and rerun each missing setting at most once even if multiple methods
    # report the same seed gap.
    jobs: Set[Job] = set()

    for method in methods:
        valid = plot_src[plot_src[method].notna()].copy()
        coverage = (
            valid.groupby(["dataset", "degree", "gt", "gen"], dropna=False)["seed"]
            .agg(lambda s: set(int(v) for v in s))
            .reset_index(name="seeds_present")
        )
        coverage["missing_seeds"] = coverage["seeds_present"].apply(
            lambda s: sorted(expected_seeds - s)
        )
        bad = coverage[coverage["missing_seeds"].map(bool)]
        for row in bad.itertuples(index=False):
            for seed in row.missing_seeds:
                jobs.add(
                    Job(
                        dataset=row.dataset,
                        degree=normalize_degree(row.degree),
                        gt=int(row.gt),
                        gen=int(row.gen),
                        seed=int(seed),
                    )
                )

    return sorted(
        jobs,
        key=lambda j: (
            j.dataset,
            1 if math.isnan(j.degree) else 0,
            -1 if math.isnan(j.degree) else int(j.degree),
            j.gt,
            j.gen,
            j.seed,
        ),
    )


def job_log_dir(job: Job, log_root: Path) -> Path:
    base = log_root / DATASET_TO_LOG_DIR[job.dataset] / f"s{job.seed}"
    if job.dataset == "rotated_mnist":
        base = base / f"target{int(job.degree)}"
    return base


def job_plot_dir(job: Job, plot_root: Path) -> Path:
    base = plot_root / DATASET_TO_LOG_DIR[job.dataset] / f"s{job.seed}"
    if job.dataset == "rotated_mnist":
        base = base / f"target{int(job.degree)}"
    return base


def clear_job_cache(job: Job, log_root: Path, plot_root: Path) -> None:
    small_dim = DATASET_TO_SMALL_DIM[job.dataset]
    stem = f"test_acc_dim{small_dim}_int{job.gt}_gen{job.gen}_pseudo_prototypes_bic"
    log_dir = job_log_dir(job, log_root)
    plot_dir = job_plot_dir(job, plot_root)
    paths = [
        log_dir / f"{stem}.txt",
        log_dir / f"{stem}_curves.jsonl",
        plot_dir / f"{stem}.png",
    ]
    for path in paths:
        if path.exists():
            print(f"  removing cache: {path}")
            path.unlink()


def build_command(
    job: Job,
    log_root: Path,
    plot_root: Path,
    goat_gen_methods: str,
) -> list[str]:
    cmd = [
        sys.executable,
        "experiment_refrac.py",
        "--plot-root",
        str(plot_root),
        "--log-root",
        str(log_root),
        "--dataset",
        DATASET_TO_CLI[job.dataset],
        "--label-source",
        "pseudo",
        "--em-match",
        "prototypes",
        "--em-select",
        "bic",
        "--goat-gen-methods",
        str(goat_gen_methods),
        "--seed",
        str(job.seed),
        "--gt-domains",
        str(job.gt),
        "--generated-domains",
        str(job.gen),
        "--small-dim",
        str(DATASET_TO_SMALL_DIM[job.dataset]),
    ]
    if job.dataset == "rotated_mnist":
        cmd.extend(["--rotation-angle", str(int(job.degree))])
    return cmd


def format_job(job: Job) -> str:
    bits = [f"dataset={job.dataset}"]
    if job.dataset == "rotated_mnist":
        bits.append(f"degree={int(job.degree)}")
    bits.extend([f"seed={job.seed}", f"gt={job.gt}", f"gen={job.gen}"])
    return ", ".join(bits)


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    expected_seeds = set(args.seeds)
    datasets = set(args.datasets) if args.datasets else None
    plot_src = canonical_filter(df, expected_seeds, datasets, args.include_mnist_90)

    if plot_src.empty:
        print("No canonical rows matched the requested filters.")
        return 0

    jobs = find_missing_jobs(plot_src, args.methods, expected_seeds)
    if args.limit is not None:
        jobs = jobs[: args.limit]

    print(f"Canonical rows considered: {len(plot_src)}")
    print(f"Missing jobs found: {len(jobs)}")
    if not jobs:
        return 0

    log_root = Path(args.log_root)
    plot_root = Path(args.plot_root)

    for idx, job in enumerate(jobs, start=1):
        print(f"[{idx}/{len(jobs)}] {format_job(job)}")
        clear_job_cache(job, log_root, plot_root)
        cmd = build_command(job, log_root, plot_root, args.goat_gen_methods)
        print("  " + " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
