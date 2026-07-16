from __future__ import annotations

import csv
import hashlib
import io
import math
import os
import platform
import socket
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from goat.core.io import atomic_write_text, write_json
from goat.core.prepared_artifacts import canonical_json
from goat.core.schema import MethodResult, RunRecord


RUN_OUTPUT_SCHEMA = 2
_CODE_IDENTITY_KEYS = {"code_revision", "code_dirty", "code_diff_sha256"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def jsonable(value: Any) -> Any:
    """Convert scientific Python values into strict-JSON-compatible data."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    if hasattr(value, "item"):
        try:
            return jsonable(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


@lru_cache(maxsize=8)
def git_state(root: str) -> dict[str, Any]:
    root_path = Path(root)

    def _git(*arguments: str) -> str:
        try:
            result = subprocess.run(
                ["git", *arguments],
                cwd=root_path,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return ""
        return result.stdout.strip()

    revision = _git("rev-parse", "HEAD") or "unknown"
    branch = _git("rev-parse", "--abbrev-ref", "HEAD") or "unknown"
    dirty_listing = _git("status", "--porcelain")
    diff = _git("diff", "HEAD", "--binary")
    untracked = _git("ls-files", "--others", "--exclude-standard").splitlines()
    dirty_hasher = hashlib.sha256()
    if diff:
        dirty_hasher.update(diff.encode("utf-8"))
    for relative in sorted(untracked):
        path = root_path / relative
        if not path.is_file():
            continue
        dirty_hasher.update(relative.encode("utf-8"))
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                dirty_hasher.update(block)
    diff_sha256 = dirty_hasher.hexdigest() if diff or untracked else None
    return {
        "revision": revision,
        "branch": branch,
        "dirty": bool(dirty_listing),
        "diff_sha256": diff_sha256,
    }


def _namespace_value(args: Any, name: str, default: Any = None) -> Any:
    return getattr(args, name, default)


def build_run_config(args: Any, *, target: int | None = None) -> dict[str, Any]:
    """Return every option that can materially change a worker result."""

    repository_root = Path(__file__).resolve().parents[2]
    code = git_state(str(repository_root))
    pca_dims = _namespace_value(args, "em_pca_dims", [None])
    config = {
        "dataset": str(_namespace_value(args, "dataset", "unknown")),
        "target": None if target is None else int(target),
        "seed": int(_namespace_value(args, "seed", 0)),
        "gt_domains": int(_namespace_value(args, "gt_domains", 0)),
        "generated_domains": int(_namespace_value(args, "generated_domains", 0)),
        "small_dim": int(_namespace_value(args, "small_dim", 0)),
        "ssl_weight": float(_namespace_value(args, "ssl_weight", 0.1)),
        "label_source": str(_namespace_value(args, "label_source", "pseudo")),
        "em_match": str(_namespace_value(args, "em_match", "prototypes")),
        "em_select": str(_namespace_value(args, "em_select", "bic")),
        "em_ensemble": bool(_namespace_value(args, "em_ensemble", False)),
        "em_bic_delta": float(_namespace_value(args, "em_bic_delta", 10.0)),
        "em_seeds": [int(value) for value in _namespace_value(args, "em_seeds", [0, 1, 2])],
        "em_seed_mode": str(_namespace_value(args, "em_seed_mode", "offset")),
        "em_cov_types": [str(value) for value in _namespace_value(args, "em_cov_types", ["diag"])],
        "em_pca_dims": [None if value in (None, "none", "null", 0, "0", -1, "-1") else int(value) for value in pca_dims],
        "pseudo_confidence_q": float(_namespace_value(args, "pseudo_confidence_q", 0.9)),
        "goat_gen_methods": str(_namespace_value(args, "goat_gen_methods", "w2")),
        "interp_class_agnostic": bool(_namespace_value(args, "interp_class_agnostic", False)),
        "use_labels": bool(_namespace_value(args, "use_labels", False)),
        "diet": bool(_namespace_value(args, "diet", False)),
        "skip_pooled_goat": bool(_namespace_value(args, "skip_pooled_goat", False)),
        "code_revision": code["revision"],
        "code_dirty": code["dirty"],
        "code_diff_sha256": code["diff_sha256"],
    }
    return jsonable(config)


def run_id_for_config(config: Mapping[str, Any]) -> str:
    envelope = {"schema_version": RUN_OUTPUT_SCHEMA, "config": jsonable(config)}
    return hashlib.sha256(canonical_json(envelope).encode("utf-8")).hexdigest()[:16]


def canonical_run_dir(
    log_root: str | os.PathLike[str],
    config: Mapping[str, Any],
    *,
    run_id: str | None = None,
) -> Path:
    run_id = run_id or run_id_for_config(config)
    target = config.get("target")
    target_token = "target-none" if target is None else f"target{int(target)}"
    return (
        Path(log_root).expanduser()
        / "runs"
        / str(config["dataset"])
        / target_token
        / f"s{int(config['seed'])}"
        / f"gobs{int(config['gt_domains'])}"
        / f"gsyn{int(config['generated_domains'])}"
        / f"match-{config['em_match']}"
        / run_id
    )


def status_payload(
    *,
    state: str,
    run_id: str,
    command: Sequence[str],
    returncode: int | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    return jsonable(
        {
            "schema_version": RUN_OUTPUT_SCHEMA,
            "run_id": run_id,
            "state": state,
            "updated_at": utc_now(),
            "command": list(command),
            "returncode": returncode,
            "message": message,
        }
    )


def write_status(
    run_dir: str | os.PathLike[str],
    *,
    state: str,
    run_id: str,
    command: Sequence[str],
    returncode: int | None = None,
    message: str | None = None,
) -> Path:
    path = Path(run_dir) / "status.json"
    write_json(
        path,
        status_payload(
            state=state,
            run_id=run_id,
            command=command,
            returncode=returncode,
            message=message,
        ),
    )
    return path


def completed_run_exists(run_dir: str | os.PathLike[str]) -> bool:
    run_dir = Path(run_dir)
    status_path = run_dir / "status.json"
    run_path = run_dir / "run.json"
    if not status_path.exists() or not run_path.exists():
        return False
    try:
        import json

        return json.loads(status_path.read_text()).get("state") == "completed"
    except (OSError, ValueError):
        return False


def equivalent_completed_run(
    log_root: str | os.PathLike[str], config: Mapping[str, Any]
) -> Path | None:
    """Find a completed logical run produced by a different code revision.

    This is primarily for canonical records backfilled from pre-schema outputs.
    Exact run-id matches remain the preferred resume path.
    """

    expected = {
        key: jsonable(value)
        for key, value in config.items()
        if key not in _CODE_IDENTITY_KEYS
    }
    runs_root = Path(log_root).expanduser() / "runs"
    if not runs_root.exists():
        return None
    import json

    for run_path in runs_root.rglob("run.json"):
        try:
            payload = json.loads(run_path.read_text())
            actual = {
                key: jsonable(value)
                for key, value in (payload.get("config") or {}).items()
                if key not in _CODE_IDENTITY_KEYS
            }
        except (OSError, ValueError):
            continue
        if actual == expected and completed_run_exists(run_path.parent):
            return run_path.parent
    return None


def _method_result(name: str, result: Any) -> MethodResult:
    metrics = {}
    em_acc = getattr(result, "em_acc", None)
    if em_acc is not None:
        metrics["em_acc"] = jsonable(em_acc)
    return MethodResult(
        name=name,
        train_curve=jsonable(getattr(result, "train_curve", [])),
        test_curve=jsonable(getattr(result, "test_curve", [])),
        st_curve=jsonable(getattr(result, "st_curve", [])),
        st_all_curve=jsonable(getattr(result, "st_all_curve", [])),
        generated_curve=jsonable(getattr(result, "generated_curve", [])),
        metrics=metrics,
        duration_sec=jsonable(getattr(result, "duration_sec", None)),
    )


def _csv_text(fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: jsonable(row.get(key)) for key in fieldnames})
    return buffer.getvalue()


def write_run_record_outputs(record: RunRecord, run_dir: str | os.PathLike[str]) -> Path:
    """Persist one canonical run record and its analysis-friendly sidecars."""

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = jsonable(record.to_dict())
    write_json(run_dir / "run.json", payload)
    write_json(
        run_dir / "curves.json",
        {name: result.to_dict() for name, result in record.methods.items()},
    )
    write_json(run_dir / "em_diagnostics.json", payload.get("em_diagnostics", {}))

    method_rows = []
    config = record.config
    for name, result in record.methods.items():
        final_accuracy = result.test_curve[-1] if result.test_curve else None
        method_rows.append(
            {
                "run_id": record.run_id,
                "dataset": record.dataset,
                "target": config.get("target"),
                "seed": record.seed,
                "gt_domains": config.get("gt_domains"),
                "generated_domains": config.get("generated_domains"),
                "em_match": config.get("em_match"),
                "method": name,
                "final_accuracy": final_accuracy,
                "em_accuracy": result.metrics.get("em_acc"),
                "duration_sec": result.duration_sec,
            }
        )
    fields = [
        "run_id",
        "dataset",
        "target",
        "seed",
        "gt_domains",
        "generated_domains",
        "em_match",
        "method",
        "final_accuracy",
        "em_accuracy",
        "duration_sec",
    ]
    atomic_write_text(run_dir / "methods.csv", _csv_text(fields, method_rows))
    return run_dir


def write_canonical_run_outputs(
    *,
    args: Any,
    target: int | None,
    results: Mapping[str, Any],
    elapsed: float,
    legacy_log_path: str | os.PathLike[str],
) -> Path:
    config = build_run_config(args, target=target)
    computed_run_id = run_id_for_config(config)
    requested_run_id = str(_namespace_value(args, "run_id", "") or "")
    if requested_run_id and requested_run_id != computed_run_id:
        raise RuntimeError(
            f"worker run id mismatch: requested={requested_run_id} computed={computed_run_id}"
        )
    run_id = requested_run_id or computed_run_id
    log_root = _namespace_value(args, "log_root", os.environ.get("LOG_ROOT", "logs_rerun"))
    run_dir = canonical_run_dir(log_root, config, run_id=run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    methods = {name: _method_result(name, result) for name, result in results.items()}
    legacy_log_path = Path(legacy_log_path)
    legacy_curves_path = Path(str(legacy_log_path).replace(".txt", "_curves.jsonl"))
    artifacts = {
        "legacy_summary": str(legacy_log_path),
        "legacy_curves": str(legacy_curves_path),
        "stdout": str(run_dir / "stdout.log"),
        "stderr": str(run_dir / "stderr.log"),
        "status": str(run_dir / "status.json"),
        "prepared_artifact_root": str(_namespace_value(args, "prepared_artifact_root", "")),
    }
    artifacts.update(jsonable(_namespace_value(args, "_prepared_artifact_refs", {})))
    code = git_state(str(Path(__file__).resolve().parents[2]))
    now = utc_now()
    record = RunRecord(
        run_id=run_id,
        dataset=str(config["dataset"]),
        seed=int(config["seed"]),
        config=config,
        methods=methods,
        metrics={"method_count": len(methods)},
        artifacts=artifacts,
        em_diagnostics=jsonable(_namespace_value(args, "_em_diagnostics", {})),
        provenance={
            "kind": "native",
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "code": code,
        },
        command=list(sys.argv),
        created_at=jsonable(_namespace_value(args, "run_started_at", None)) or now,
        completed_at=now,
        elapsed_sec=float(elapsed),
    )
    output_dir = write_run_record_outputs(record, run_dir)
    if not requested_run_id:
        write_status(
            output_dir,
            state="completed",
            run_id=run_id,
            command=list(sys.argv),
            returncode=0,
            message="direct experiment invocation",
        )
    return output_dir
