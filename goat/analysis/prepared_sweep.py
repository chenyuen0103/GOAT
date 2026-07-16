from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from goat.analysis.logs import infer_dataset_from_path
from goat.core.io import atomic_write_text, read_json, read_jsonl, write_json
from goat.core.prepared_artifacts import PreparedArtifactStore
from goat.core.run_outputs import (
    canonical_run_dir,
    jsonable,
    run_id_for_config,
    write_run_record_outputs,
    write_status,
)
from goat.core.schema import MethodResult, RunRecord


LEGACY_NAME = re.compile(
    r"^test_acc_dim(?P<small_dim>\d+)_int(?P<gt>\d+)_gen(?P<gen>\d+)_"
    r"(?P<label_source>[^_]+)_(?P<em_match>prototypes|pseudo|none)_"
    r"(?P<em_select>bic|cost|ll)(?P<ensemble>_em-ensemble)?(?P<variant>.*)$"
)

CODE_CONFIG_KEYS = {"code_revision", "code_dirty", "code_diff_sha256"}


def discover_run_records(log_root: str | Path) -> list[RunRecord]:
    root = Path(log_root).expanduser()
    return [
        RunRecord.from_dict(read_json(path))
        for path in sorted((root / "runs").rglob("run.json"))
    ]


def logical_key(config: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        config.get("dataset"),
        config.get("target"),
        int(config.get("seed", 0)),
        int(config.get("gt_domains", 0)),
        int(config.get("generated_domains", 0)),
        config.get("em_match"),
        bool(config.get("em_ensemble", False)),
        float(config.get("em_bic_delta", 10.0)),
        tuple(config.get("em_seeds") or ()),
        config.get("em_seed_mode"),
        tuple(config.get("em_cov_types") or ()),
        tuple(config.get("em_pca_dims") or ()),
    )


def _target_from_path(path: Path) -> int | None:
    for part in path.parts:
        match = re.fullmatch(r"target(-?\d+)", part)
        if match:
            return int(match.group(1))
    return None


def _seed_from_path(path: Path, fallback: int) -> int:
    for part in path.parts:
        match = re.fullmatch(r"s(\d+)", part)
        if match:
            return int(match.group(1))
    return fallback


def _legacy_config(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    stem = path.name.removesuffix("_curves.jsonl")
    match = LEGACY_NAME.fullmatch(stem)
    filename = match.groupdict() if match else {}
    dataset = infer_dataset_from_path(path)
    target = _target_from_path(path) if dataset == "mnist" else None
    pca_dims = payload.get("em_pca_dims", [None])
    return jsonable(
        {
            "dataset": dataset,
            "target": target,
            "seed": _seed_from_path(path, int(payload.get("seed", 0))),
            "gt_domains": int(payload.get("gt_domains", filename.get("gt") or 0)),
            "generated_domains": int(
                payload.get("generated_domains", filename.get("gen") or 0)
            ),
            "small_dim": int(filename.get("small_dim") or payload.get("small_dim", 0)),
            "ssl_weight": float(payload.get("ssl_weight", 0.1)),
            "label_source": str(
                payload.get("label_source", filename.get("label_source") or "pseudo")
            ),
            "em_match": str(
                payload.get("em_match", filename.get("em_match") or "prototypes")
            ),
            "em_select": str(
                payload.get("em_select", filename.get("em_select") or "bic")
            ),
            "em_ensemble": bool(
                payload.get("em_ensemble", bool(filename.get("ensemble")))
            ),
            "em_bic_delta": float(payload.get("em_bic_delta", 10.0)),
            "em_seeds": [int(value) for value in payload.get("em_seeds", [0, 1, 2])],
            "em_seed_mode": str(payload.get("em_seed_mode", "offset")),
            "em_cov_types": [str(value) for value in payload.get("em_cov_types", ["diag"])],
            "em_pca_dims": [
                None if value in (None, "none", "null", 0, "0", -1, "-1") else int(value)
                for value in pca_dims
            ],
            "pseudo_confidence_q": float(payload.get("pseudo_confidence_q", 0.9)),
            "goat_gen_methods": str(payload.get("goat_gen_methods", "w2")),
            "interp_class_agnostic": bool(payload.get("interp_class_agnostic", False)),
            "use_labels": bool(payload.get("use_labels", False)),
            "diet": bool(payload.get("diet", False)),
            "skip_pooled_goat": bool(
                int(payload.get("generated_domains", filename.get("gen") or 0)) > 0
                and str(payload.get("em_match", filename.get("em_match") or "prototypes"))
                == "pseudo"
            ),
            "code_revision": "legacy-unknown",
            "code_dirty": None,
            "code_diff_sha256": None,
        }
    )


def _manifest_matches_config(metadata: Mapping[str, Any], config: Mapping[str, Any]) -> bool:
    if metadata.get("artifact") != "raw-em-grid":
        return False
    if str(metadata.get("dataset")) != str(config.get("dataset")):
        return False
    if int(metadata.get("seed", -1)) != int(config.get("seed", 0)):
        return False
    if int(metadata.get("gt_domains", -1)) != int(config.get("gt_domains", 0)):
        return False
    comparisons = (
        (metadata.get("seeds"), config.get("em_seeds")),
        (metadata.get("cov_types"), config.get("em_cov_types")),
        (metadata.get("pca_dims"), config.get("em_pca_dims")),
    )
    for actual, expected in comparisons:
        if expected is not None and list(actual or []) != list(expected or []):
            return False
    expected_seed_mode = config.get("em_seed_mode")
    if expected_seed_mode and metadata.get("seed_mode") != expected_seed_mode:
        return False
    return True


def reconstruct_em_diagnostics(
    prepared_artifact_root: str | Path | None,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if not prepared_artifact_root:
        return {}
    raw_root = Path(prepared_artifact_root).expanduser() / "raw_em"
    if not raw_root.exists():
        return {}

    diagnostics: dict[str, Any] = {}
    for manifest_path in sorted(raw_root.glob("*.json")):
        try:
            manifest = read_json(manifest_path)
            metadata = manifest.get("metadata") or {}
        except (OSError, ValueError):
            continue
        if not _manifest_matches_config(metadata, config):
            continue
        payload_path = manifest_path.with_suffix(".pt")
        if not payload_path.exists():
            continue
        raw_models = PreparedArtifactStore(prepared_artifact_root).load(
            "raw_em", metadata, required=True
        )
        bics = np.asarray([float(model["bic"]) for model in raw_models], dtype=float)
        best = float(np.min(bics))
        deltas = bics - best
        keep = deltas <= float(config.get("em_bic_delta", 10.0))
        kept_indices = np.flatnonzero(keep)
        kept_deltas = deltas[keep]
        if len(kept_deltas):
            weights = np.exp(-0.5 * kept_deltas)
            weights /= weights.sum()
        else:
            weights = np.asarray([], dtype=float)
        domain = str(metadata.get("domain", "unknown"))
        key = domain if domain not in diagnostics else f"{domain}:{manifest.get('key', '')[:8]}"
        diagnostics[key] = jsonable(
            {
                "domain": domain,
                "mapping_scheme": config.get("em_match"),
                "selection_criterion": config.get("em_select"),
                "ensemble_enabled": config.get("em_ensemble"),
                "bundle": {
                    "criterion": "bic_trimmed_weighted_ensemble",
                    "max_delta_bic": config.get("em_bic_delta", 10.0),
                    "bic_best": best,
                    "weights": weights,
                    "bics": bics[keep],
                    "all_bics": bics,
                    "kept_indices": kept_indices,
                    "reconstructed": True,
                },
                "models": [
                    {
                        "index": index,
                        "config": model.get("cfg", {}),
                        "bic": model.get("bic"),
                        "final_ll": model.get("final_ll"),
                        "cost": model.get("cost"),
                    }
                    for index, model in enumerate(raw_models)
                ],
                "prepared_artifact": {
                    "key": manifest.get("key"),
                    "payload": str(payload_path),
                    "manifest": str(manifest_path),
                    "metadata": metadata,
                },
                "target_label_accuracy_is_evaluation_only": True,
            }
        )
    return diagnostics


def backfill_legacy_records(
    log_root: str | Path,
    *,
    prepared_artifact_root: str | Path | None = None,
    force: bool = False,
) -> list[Path]:
    log_root = Path(log_root).expanduser()
    written: list[Path] = []
    claimed_legacy_paths = set()
    for run_path in (log_root / "runs").rglob("run.json") if (log_root / "runs").exists() else ():
        try:
            legacy_path = (read_json(run_path).get("artifacts") or {}).get("legacy_curves")
        except (OSError, ValueError):
            continue
        if legacy_path:
            claimed_legacy_paths.add(str(Path(legacy_path).expanduser().resolve()))
    for curves_path in sorted(log_root.rglob("*_curves.jsonl")):
        if "runs" in curves_path.relative_to(log_root).parts:
            continue
        if str(curves_path.resolve()) in claimed_legacy_paths:
            continue
        rows = list(read_jsonl(curves_path))
        if len(rows) != 1:
            continue
        payload = rows[0]
        config = _legacy_config(curves_path, payload)
        run_id = run_id_for_config(config)
        run_dir = canonical_run_dir(log_root, config, run_id=run_id)
        if (run_dir / "run.json").exists() and not force:
            continue
        methods = {
            name: MethodResult.from_legacy(name, method_payload)
            for name, method_payload in (payload.get("methods") or {}).items()
        }
        modified_at = datetime.fromtimestamp(
            curves_path.stat().st_mtime, tz=timezone.utc
        ).isoformat()
        diagnostics = reconstruct_em_diagnostics(prepared_artifact_root, config)
        record = RunRecord(
            run_id=run_id,
            dataset=str(config["dataset"]),
            seed=int(config["seed"]),
            config=config,
            methods=methods,
            metrics={"method_count": len(methods)},
            artifacts={
                "legacy_curves": str(curves_path),
                "legacy_summary": str(curves_path).replace("_curves.jsonl", ".txt"),
                "prepared_artifact_root": str(prepared_artifact_root or ""),
                "stdout": str(run_dir / "stdout.log"),
                "stderr": str(run_dir / "stderr.log"),
                "status": str(run_dir / "status.json"),
            },
            em_diagnostics=diagnostics,
            provenance={
                "kind": "legacy_backfill",
                "source": str(curves_path),
                "stdout_available": False,
                "em_diagnostics_reconstructed": bool(diagnostics),
            },
            command=[],
            created_at=modified_at,
            completed_at=modified_at,
            elapsed_sec=payload.get("elapsed"),
        )
        write_run_record_outputs(record, run_dir)
        atomic_write_text(
            run_dir / "stdout.log",
            "Unavailable: this canonical record was backfilled from a legacy curve file.\n",
        )
        atomic_write_text(
            run_dir / "stderr.log",
            "Unavailable: this canonical record was backfilled from a legacy curve file.\n",
        )
        write_status(
            run_dir,
            state="completed",
            run_id=run_id,
            command=[],
            returncode=0,
            message="backfilled from legacy output",
        )
        written.append(run_dir)
    return written


def _csv_text(fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: jsonable(row.get(key)) for key in fieldnames})
    return buffer.getvalue()


def method_rows(records: Iterable[RunRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        config = record.config
        for name, result in record.methods.items():
            rows.append(
                {
                    "run_id": record.run_id,
                    "schema_version": record.schema_version,
                    "provenance": record.provenance.get("kind"),
                    "dataset": record.dataset,
                    "target": config.get("target"),
                    "seed": record.seed,
                    "gt_domains": config.get("gt_domains"),
                    "generated_domains": config.get("generated_domains"),
                    "em_match": config.get("em_match"),
                    "em_ensemble": config.get("em_ensemble"),
                    "em_bic_delta": config.get("em_bic_delta"),
                    "em_seeds": json.dumps(config.get("em_seeds") or []),
                    "em_seed_mode": config.get("em_seed_mode"),
                    "em_cov_types": json.dumps(config.get("em_cov_types") or []),
                    "em_pca_dims": json.dumps(config.get("em_pca_dims") or []),
                    "code_revision": config.get("code_revision"),
                    "method": name,
                    "final_accuracy": result.test_curve[-1] if result.test_curve else None,
                    "em_accuracy": result.metrics.get("em_acc"),
                    "duration_sec": result.duration_sec,
                    "run_elapsed_sec": record.elapsed_sec,
                }
            )
    return rows


_T_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    15: 2.131,
    20: 2.086,
    30: 2.042,
}


def _critical_t95(degrees_of_freedom: int) -> float:
    if degrees_of_freedom in _T_975:
        return _T_975[degrees_of_freedom]
    larger = [df for df in _T_975 if df >= degrees_of_freedom]
    return _T_975[min(larger)] if larger else 1.96


def summary_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    group_fields = (
        "dataset",
        "target",
        "gt_domains",
        "generated_domains",
        "em_match",
        "em_ensemble",
        "em_bic_delta",
        "method",
    )
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        value = row.get("final_accuracy")
        if value is not None:
            grouped[tuple(row.get(field) for field in group_fields)].append(float(value))
    output = []
    for key, values in sorted(grouped.items(), key=lambda item: tuple(str(x) for x in item[0])):
        n = len(values)
        mean = statistics.fmean(values)
        std = statistics.stdev(values) if n > 1 else None
        ci = None
        if std is not None:
            ci = _critical_t95(n - 1) * std / math.sqrt(n)
        row = dict(zip(group_fields, key))
        row.update(
            {
                "n": n,
                "mean_final_accuracy": mean,
                "std_final_accuracy": std,
                "ci95_half_width": ci,
                "min_final_accuracy": min(values),
                "max_final_accuracy": max(values),
            }
        )
        output.append(row)
    return output


def expected_keys(
    *,
    dataset: str,
    target: int | None,
    seeds: Sequence[int],
    gt_domains: Sequence[int],
    generated_domains: Sequence[int],
    em_matches: Sequence[str],
    em_ensemble: bool,
    em_bic_delta: float,
    em_seeds: Sequence[int],
    em_seed_mode: str,
    em_cov_types: Sequence[str],
    em_pca_dims: Sequence[int | None],
) -> set[tuple[Any, ...]]:
    expected = set()
    for seed in seeds:
        for gobs in gt_domains:
            for gsyn in generated_domains:
                mappings = em_matches[:1] if int(gsyn) == 0 else em_matches
                for em_match in mappings:
                    expected.add(
                        (
                            dataset,
                            target,
                            int(seed),
                            int(gobs),
                            int(gsyn),
                            str(em_match),
                            bool(em_ensemble),
                            float(em_bic_delta),
                            tuple(em_seeds),
                            em_seed_mode,
                            tuple(em_cov_types),
                            tuple(em_pca_dims),
                        )
                    )
    return expected


def aggregate(
    records: Sequence[RunRecord],
    *,
    output_dir: str | Path,
    expected: set[tuple[Any, ...]] | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = method_rows(records)
    run_fields = [
        "run_id",
        "schema_version",
        "provenance",
        "dataset",
        "target",
        "seed",
        "gt_domains",
        "generated_domains",
        "em_match",
        "em_ensemble",
        "em_bic_delta",
        "em_seeds",
        "em_seed_mode",
        "em_cov_types",
        "em_pca_dims",
        "code_revision",
        "method",
        "final_accuracy",
        "em_accuracy",
        "duration_sec",
        "run_elapsed_sec",
    ]
    atomic_write_text(output_dir / "runs.csv", _csv_text(run_fields, rows))
    summaries = summary_rows(rows)
    summary_fields = [
        "dataset",
        "target",
        "gt_domains",
        "generated_domains",
        "em_match",
        "em_ensemble",
        "em_bic_delta",
        "method",
        "n",
        "mean_final_accuracy",
        "std_final_accuracy",
        "ci95_half_width",
        "min_final_accuracy",
        "max_final_accuracy",
    ]
    atomic_write_text(output_dir / "summary.csv", _csv_text(summary_fields, summaries))

    keys = [logical_key(record.config) for record in records]
    counts = Counter(keys)
    actual = set(keys)
    expected = expected or actual
    missing = expected - actual
    unexpected = actual - expected
    duplicates = {str(key): count for key, count in counts.items() if count > 1}
    validation = {
        "schema_version": 1,
        "passed": not missing and not unexpected and not duplicates,
        "expected_result_rows": len(expected),
        "actual_result_rows": len(actual),
        "actual_record_files": len(records),
        "method_rows": len(rows),
        "expected_methods": sorted({row["method"] for row in rows}),
        "missing_result_rows": len(missing),
        "duplicate_result_rows": sum(count - 1 for count in counts.values() if count > 1),
        "unexpected_result_rows": len(unexpected),
        "missing_keys": [list(key) for key in sorted(missing, key=str)],
        "unexpected_keys": [list(key) for key in sorted(unexpected, key=str)],
        "duplicates": duplicates,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output_dir / "validation.json", validation)
    return validation


def _parse_pca_dims(values: Sequence[str]) -> list[int | None]:
    return [
        None if value.strip().lower() in {"none", "null", "0", "-1"} else int(value)
        for value in values
    ]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill, aggregate, and validate canonical prepared-sweep outputs."
    )
    parser.add_argument("--log-root", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--prepared-artifact-root", default="")
    parser.add_argument("--backfill-legacy", action="store_true")
    parser.add_argument("--force-backfill", action="store_true")
    parser.add_argument("--dataset", default="mnist")
    parser.add_argument("--target", type=int, default=45)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--gt-domains", nargs="+", type=int, default=[0])
    parser.add_argument("--generated-domains", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--em-matches", nargs="+", default=["prototypes", "pseudo"])
    parser.add_argument("--em-ensemble", action="store_true")
    parser.add_argument("--em-bic-delta", type=float, default=10.0)
    parser.add_argument("--em-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--em-seed-mode", default="offset")
    parser.add_argument("--em-cov-types", nargs="+", default=["diag"])
    parser.add_argument("--em-pca-dims", nargs="+", default=["none"])
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    log_root = Path(args.log_root).expanduser()
    if args.backfill_legacy:
        written = backfill_legacy_records(
            log_root,
            prepared_artifact_root=args.prepared_artifact_root or None,
            force=args.force_backfill,
        )
        print(f"[aggregate] backfilled={len(written)}")
    records = discover_run_records(log_root)
    pca_dims = _parse_pca_dims(args.em_pca_dims)
    target = args.target if args.dataset == "mnist" else None
    expected = expected_keys(
        dataset=args.dataset,
        target=target,
        seeds=args.seeds,
        gt_domains=args.gt_domains,
        generated_domains=args.generated_domains,
        em_matches=args.em_matches,
        em_ensemble=args.em_ensemble,
        em_bic_delta=args.em_bic_delta,
        em_seeds=args.em_seeds,
        em_seed_mode=args.em_seed_mode,
        em_cov_types=args.em_cov_types,
        em_pca_dims=pca_dims,
    )
    relevant = [record for record in records if logical_key(record.config) in expected]
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else log_root / "aggregate"
    validation = aggregate(relevant, output_dir=output_dir, expected=expected)
    print(json.dumps(validation, indent=2, sort_keys=True))
    if not validation["passed"] and not args.allow_incomplete:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
