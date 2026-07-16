import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from goat.core.run_outputs import (
    build_run_config,
    canonical_run_dir,
    completed_run_exists,
    equivalent_completed_run,
    run_id_for_config,
    write_canonical_run_outputs,
    write_status,
)
from goat.experiments.prepared_sweep import _run_logged_command


def _args(tmp_path: Path, **overrides):
    values = {
        "dataset": "mnist",
        "seed": 0,
        "gt_domains": 0,
        "generated_domains": 1,
        "small_dim": 2048,
        "ssl_weight": 0.1,
        "label_source": "pseudo",
        "em_match": "prototypes",
        "em_select": "bic",
        "em_ensemble": True,
        "em_bic_delta": 10.0,
        "em_seeds": [0, 1, 2],
        "em_seed_mode": "offset",
        "em_cov_types": ["diag"],
        "em_pca_dims": [None],
        "pseudo_confidence_q": 0.9,
        "goat_gen_methods": "w2",
        "interp_class_agnostic": False,
        "use_labels": False,
        "diet": False,
        "skip_pooled_goat": False,
        "log_root": str(tmp_path / "logs"),
        "prepared_artifact_root": str(tmp_path / "prepared"),
        "run_id": "",
        "_em_diagnostics": {},
        "_prepared_artifact_refs": {},
        "run_started_at": "2026-01-01T00:00:00+00:00",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_run_id_is_stable_and_changes_with_bic_threshold(tmp_path: Path):
    first = build_run_config(_args(tmp_path), target=45)
    equal = build_run_config(_args(tmp_path), target=45)
    changed = build_run_config(_args(tmp_path, em_bic_delta=0.0), target=45)

    assert run_id_for_config(first) == run_id_for_config(equal)
    assert run_id_for_config(first) != run_id_for_config(changed)


def test_canonical_writer_produces_structured_sidecars(tmp_path: Path):
    args = _args(
        tmp_path,
        _em_diagnostics={
            "mnist-angle-45": {
                "bundle": {"all_bics": [100.0, 102.0], "weights": [0.731, 0.269]}
            }
        },
    )
    result = SimpleNamespace(
        train_curve=[50.0],
        test_curve=[60.0, 65.0],
        st_curve=[55.0],
        st_all_curve=[56.0],
        generated_curve=[58.0],
        em_acc=0.8,
        duration_sec=2.5,
    )
    legacy_path = tmp_path / "logs" / "mnist" / "s0" / "target45" / "legacy.txt"
    legacy_path.parent.mkdir(parents=True)
    run_dir = write_canonical_run_outputs(
        args=args,
        target=45,
        results={"goat_classwise": result},
        elapsed=3.0,
        legacy_log_path=legacy_path,
    )

    assert (run_dir / "run.json").exists()
    assert (run_dir / "methods.csv").exists()
    assert (run_dir / "curves.json").exists()
    assert (run_dir / "em_diagnostics.json").exists()
    payload = json.loads((run_dir / "run.json").read_text())
    assert payload["schema_version"] == 2
    assert payload["config"]["em_bic_delta"] == 10.0
    assert payload["em_diagnostics"]["mnist-angle-45"]["bundle"]["all_bics"] == [100.0, 102.0]
    with (run_dir / "methods.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["method"] == "goat_classwise"
    assert float(rows[0]["final_accuracy"]) == 65.0


def test_equivalent_completed_run_ignores_only_code_identity(tmp_path: Path):
    args = _args(tmp_path)
    config = build_run_config(args, target=45)
    legacy_config = dict(config)
    legacy_config.update(
        {"code_revision": "legacy-unknown", "code_dirty": None, "code_diff_sha256": None}
    )
    legacy_id = run_id_for_config(legacy_config)
    run_dir = canonical_run_dir(args.log_root, legacy_config, run_id=legacy_id)
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps({"run_id": legacy_id, "config": legacy_config, "methods": {}})
    )
    write_status(run_dir, state="completed", run_id=legacy_id, command=[], returncode=0)

    assert completed_run_exists(run_dir)
    assert equivalent_completed_run(args.log_root, config) == run_dir
    different = dict(config, generated_domains=3)
    assert equivalent_completed_run(args.log_root, different) is None


def test_logged_command_captures_stdout_and_stderr(tmp_path: Path):
    stdout_path = tmp_path / "stdout.log"
    stderr_path = tmp_path / "stderr.log"
    _run_logged_command(
        [
            sys.executable,
            "-c",
            "import sys; print('hello'); print('warning', file=sys.stderr)",
        ],
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
    assert stdout_path.read_text().strip() == "hello"
    assert stderr_path.read_text().strip() == "warning"

