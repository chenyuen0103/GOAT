from __future__ import annotations

import argparse
import hashlib
import os
import shlex
import subprocess
import sys
import threading
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from goat.core.io import write_json
from goat.core.prepared_artifacts import canonical_json
from goat.core.run_outputs import (
    build_run_config,
    canonical_run_dir,
    completed_run_exists,
    equivalent_completed_run,
    run_id_for_config,
    write_status,
)


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
    run_id: str = "",
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
        "--ssl-weight",
        str(args.ssl_weight),
        "--label-source",
        str(args.label_source),
        "--em-match",
        str(em_match),
        "--em-select",
        str(args.em_select),
        "--em-bic-delta",
        str(args.em_bic_delta),
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
    if run_id:
        command.extend(["--run-id", run_id])
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
        config = config_for_job(args, job)
        run_id = run_id_for_config(config)
        command = _common_command(
            args,
            seed=job.seed,
            gt_domains=job.gt_domains,
            generated_domains=job.generated_domains,
            em_match=job.em_match,
            run_id=run_id,
        )
        if (
            job.generated_domains > 0
            and job.em_match != str(args.em_matches[0])
        ):
            command.append("--skip-pooled-goat")
        command.append("--require-prepared-artifacts")
        commands.append(command)
    return commands


def config_for_job(args: argparse.Namespace, job: PreparedSweepJob) -> dict:
    worker_args = argparse.Namespace(**vars(args))
    worker_args.seed = int(job.seed)
    worker_args.gt_domains = int(job.gt_domains)
    worker_args.generated_domains = int(job.generated_domains)
    worker_args.em_match = str(job.em_match)
    worker_args.small_dim = int(args.small_dim or DATASET_SMALL_DIM[args.dataset])
    worker_args.skip_pooled_goat = bool(
        job.generated_domains > 0 and job.em_match != str(args.em_matches[0])
    )
    target = int(job.rotation_angle) if job.dataset == "mnist" else None
    return build_run_config(worker_args, target=target)


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
    parser.add_argument("--ssl-weight", type=float, default=0.1)
    parser.add_argument("--label-source", choices=["pseudo", "em"], default="pseudo")
    parser.add_argument("--em-select", choices=["bic", "cost", "ll"], default="bic")
    parser.add_argument("--em-ensemble", action="store_true")
    parser.add_argument("--em-bic-delta", type=float, default=10.0)
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
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume",
        action="store_true",
        help="Skip workers that already have a validated completed canonical record.",
    )
    resume_group.add_argument(
        "--force",
        action="store_true",
        help="Rerun and replace a completed record for the same canonical run id.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _tee_stream(source, destination, console) -> None:
    try:
        for line in iter(source.readline, ""):
            destination.write(line)
            destination.flush()
            console.write(line)
            console.flush()
    finally:
        source.close()


def _run_logged_command(
    command: Sequence[str],
    *,
    stdout_path: Path,
    stderr_path: Path,
) -> None:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w") as stdout_handle, stderr_path.open("w") as stderr_handle:
        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        assert process.stderr is not None
        stdout_thread = threading.Thread(
            target=_tee_stream,
            args=(process.stdout, stdout_handle, sys.stdout),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=_tee_stream,
            args=(process.stderr, stderr_handle, sys.stderr),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()
        try:
            returncode = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            stdout_thread.join()
            stderr_thread.join()
            raise
        else:
            stdout_thread.join()
            stderr_thread.join()
    if returncode:
        raise subprocess.CalledProcessError(returncode, list(command))


def _prepare_log_paths(args: argparse.Namespace, command: Sequence[str]) -> tuple[Path, Path]:
    def value(flag: str, default: str) -> str:
        return str(command[command.index(flag) + 1]) if flag in command else default

    directory = (
        Path(args.log_root).expanduser()
        / "driver"
        / "prepare"
        / str(args.dataset)
        / f"s{value('--seed', '0')}"
        / f"gobs{value('--gt-domains', '0')}"
    )
    return directory / "stdout.log", directory / "stderr.log"


def _run_prepare_commands(args: argparse.Namespace) -> None:
    for command in build_prepare_commands(args):
        print(shlex.join(command), flush=True)
        if args.dry_run:
            continue
        stdout_path, stderr_path = _prepare_log_paths(args, command)
        _run_logged_command(command, stdout_path=stdout_path, stderr_path=stderr_path)


def _run_worker_commands(args: argparse.Namespace) -> None:
    jobs = build_worker_jobs(args)
    commands = build_worker_commands(args)
    for job, command in zip(jobs, commands):
        config = config_for_job(args, job)
        run_id = run_id_for_config(config)
        run_dir = canonical_run_dir(args.log_root, config, run_id=run_id)
        if completed_run_exists(run_dir):
            if args.resume:
                print(f"[prepared-sweep] skip completed run {run_id}: {run_dir}", flush=True)
                continue
            if not args.force:
                raise FileExistsError(
                    f"completed canonical run already exists: {run_dir}; "
                    "pass --resume to skip it or --force to replace it"
                )
        elif args.resume:
            equivalent = equivalent_completed_run(args.log_root, config)
            if equivalent is not None:
                print(
                    f"[prepared-sweep] skip logically equivalent completed run: {equivalent}",
                    flush=True,
                )
                continue

        print(shlex.join(command), flush=True)
        if args.dry_run:
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        write_status(run_dir, state="running", run_id=run_id, command=command)
        try:
            _run_logged_command(
                command,
                stdout_path=run_dir / "stdout.log",
                stderr_path=run_dir / "stderr.log",
            )
            if not (run_dir / "run.json").exists():
                raise RuntimeError(f"worker exited successfully without run.json: {run_dir}")
        except BaseException as exc:
            returncode = exc.returncode if isinstance(exc, subprocess.CalledProcessError) else None
            write_status(
                run_dir,
                state="failed",
                run_id=run_id,
                command=command,
                returncode=returncode,
                message=str(exc),
            )
            raise
        write_status(
            run_dir,
            state="completed",
            run_id=run_id,
            command=command,
            returncode=0,
        )


def _write_sweep_manifest(args: argparse.Namespace) -> Path:
    jobs = build_worker_jobs(args)
    expected = []
    for job in jobs:
        config = config_for_job(args, job)
        run_id = run_id_for_config(config)
        expected.append(
            {
                "run_id": run_id,
                "config": config,
                "run_dir": str(canonical_run_dir(args.log_root, config, run_id=run_id)),
            }
        )
    stable = {
        "dataset": args.dataset,
        "seeds": list(args.seeds),
        "gt_domains": list(args.gt_domains),
        "generated_domains": list(args.generated_domains),
        "em_matches": list(args.em_matches),
        "expected_run_ids": [item["run_id"] for item in expected],
    }
    sweep_id = hashlib.sha256(canonical_json(stable).encode("utf-8")).hexdigest()[:16]
    payload = {
        "schema_version": 1,
        "sweep_id": sweep_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "argv": list(sys.argv),
        "options": {key: value for key, value in vars(args).items() if key != "dry_run"},
        "expected_runs": expected,
        "expected_run_count": len(expected),
    }
    log_root = Path(args.log_root).expanduser()
    manifest_path = log_root / "manifests" / f"{sweep_id}.json"
    write_json(manifest_path, payload)
    write_json(log_root / "sweep_manifest.json", payload)
    return manifest_path


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not Path(args.legacy_module).exists():
        raise FileNotFoundError(f"experiment entrypoint not found: {args.legacy_module}")

    if not args.dry_run:
        manifest_path = _write_sweep_manifest(args)
        print(f"[prepared-sweep] manifest={manifest_path}", flush=True)

    if not args.skip_prepare:
        print("[prepared-sweep] phase=prepare", flush=True)
        _run_prepare_commands(args)
    if args.prepare_only:
        return 0

    print("[prepared-sweep] phase=isolated-workers", flush=True)
    _run_worker_commands(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
