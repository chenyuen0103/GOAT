import json
from pathlib import Path

import numpy as np

from goat.analysis.prepared_sweep import (
    aggregate,
    backfill_legacy_records,
    discover_run_records,
    expected_keys,
    logical_key,
)
from goat.analysis.plot_prepared_sweep import (
    plot_aggregate_final_accuracy,
    plot_run_curves,
)
from goat.core.io import write_jsonl
from goat.core.prepared_artifacts import PreparedArtifactStore


def _legacy_payload(*, generated_domains: int, em_match: str):
    return {
        "seed": 0,
        "gt_domains": 0,
        "generated_domains": generated_domains,
        "em_match": em_match,
        "em_select": "bic",
        "em_ensemble": True,
        "em_bic_delta": 10.0,
        "em_seeds": [0, 1, 2],
        "em_seed_mode": "offset",
        "elapsed": 12.0,
        "methods": {
            "goat_classwise": {
                "train_curve": [50.0],
                "test_curve": [60.0, 65.0],
                "st_curve": [55.0],
                "st_all_curve": [56.0],
                "generated_curve": [58.0],
                "em_acc": 0.8,
                "duration_sec": 3.0,
            }
        },
    }


def _legacy_path(root: Path, *, generated_domains: int, em_match: str) -> Path:
    name = (
        f"test_acc_dim2048_int0_gen{generated_domains}_pseudo_{em_match}_"
        "bic_em-ensemble_curves.jsonl"
    )
    path = root / "mnist" / "s0" / "target45" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def test_legacy_backfill_reconstructs_bic_weights(tmp_path: Path):
    log_root = tmp_path / "logs"
    prepared_root = tmp_path / "prepared"
    path = _legacy_path(log_root, generated_domains=1, em_match="prototypes")
    write_jsonl(path, [_legacy_payload(generated_domains=1, em_match="prototypes")])

    metadata = {
        "artifact": "raw-em-grid",
        "dataset": "mnist",
        "seed": 0,
        "gt_domains": 0,
        "domain": "mnist-angle-45",
        "feature_sha256": "abc",
        "n_classes": 10,
        "K_list": [10],
        "cov_types": ["diag"],
        "seeds": [0, 1, 2],
        "pca_dims": [None],
        "rng_base": 0,
        "seed_mode": "offset",
        "pool": "gap",
        "reg": 1e-4,
        "max_iter": 300,
    }
    raw_models = [
        {"cfg": {"seed": 0}, "bic": 100.0, "final_ll": -40.0},
        {"cfg": {"seed": 1}, "bic": 102.0, "final_ll": -41.0},
        {"cfg": {"seed": 2}, "bic": 111.0, "final_ll": -42.0},
    ]
    PreparedArtifactStore(prepared_root).save("raw_em", metadata, raw_models)

    written = backfill_legacy_records(
        log_root, prepared_artifact_root=prepared_root
    )
    assert len(written) == 1
    payload = json.loads((written[0] / "run.json").read_text())
    bundle = payload["em_diagnostics"]["mnist-angle-45"]["bundle"]
    weights = np.asarray(bundle["weights"])
    expected = np.asarray([1.0, np.exp(-1.0)])
    expected /= expected.sum()
    assert np.allclose(weights, expected)
    assert bundle["kept_indices"] == [0, 1]
    assert payload["provenance"]["kind"] == "legacy_backfill"
    assert backfill_legacy_records(log_root, prepared_artifact_root=prepared_root) == []


def test_canary_aggregation_validates_three_expected_records(tmp_path: Path):
    log_root = tmp_path / "logs"
    configurations = [(0, "prototypes"), (1, "prototypes"), (1, "pseudo")]
    for generated_domains, em_match in configurations:
        path = _legacy_path(
            log_root,
            generated_domains=generated_domains,
            em_match=em_match,
        )
        write_jsonl(
            path,
            [_legacy_payload(generated_domains=generated_domains, em_match=em_match)],
        )
    backfill_legacy_records(log_root)
    records = discover_run_records(log_root)
    expected = expected_keys(
        dataset="mnist",
        target=45,
        seeds=[0],
        gt_domains=[0],
        generated_domains=[0, 1],
        em_matches=["prototypes", "pseudo"],
        em_ensemble=True,
        em_bic_delta=10.0,
        em_seeds=[0, 1, 2],
        em_seed_mode="offset",
        em_cov_types=["diag"],
        em_pca_dims=[None],
    )
    relevant = [record for record in records if logical_key(record.config) in expected]
    validation = aggregate(
        relevant,
        output_dir=log_root / "aggregate",
        expected=expected,
    )
    assert validation["passed"]
    assert validation["expected_result_rows"] == 3
    assert validation["actual_result_rows"] == 3
    assert (log_root / "aggregate" / "runs.csv").exists()
    assert (log_root / "aggregate" / "summary.csv").exists()

    plot_root = tmp_path / "plots"
    run_plot = plot_run_curves(
        records[0], plot_root=plot_root, methods=None, dpi=72
    )
    aggregate_plot = plot_aggregate_final_accuracy(
        records, plot_root=plot_root, methods=None, dpi=72
    )
    assert run_plot is not None and run_plot.exists()
    assert aggregate_plot is not None and aggregate_plot.exists()
