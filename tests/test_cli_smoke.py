from goat.core.schema import ExperimentConfig
from goat.analysis.validate import main as validation_main
from goat.experiments.runner import build_legacy_experiment_command, main
from goat.experiments.rmnist_label_shift import main as rmnist_label_shift_main


def _validation_path() -> str:
    path = "analysis_outputs/rmnist_label_shift_spectrum/validation.json"
    try:
        from pathlib import Path

        if Path(path).exists():
            return path
    except Exception:
        pass
    return "tests/fixtures/validation.json"


def test_build_legacy_experiment_command_contains_current_entrypoint():
    cmd = build_legacy_experiment_command(
        ExperimentConfig(
            dataset="mnist",
            seed=1,
            gt_domains=0,
            generated_domains=2,
            small_dim=2048,
            extra={"rotation_angle": 45, "goat_gen_methods": "w2"},
        )
    )

    assert "experiment_refrac.py" in cmd
    assert "--dataset" in cmd
    assert "mnist" in cmd
    assert "--rotation-angle" in cmd
    assert "45" in cmd


def test_runner_dry_run(capsys):
    code = main(
        [
            "--dataset",
            "mnist",
            "--generated-domains",
            "1",
            "--seed",
            "0",
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "experiment_refrac.py" in captured.out


def test_rmnist_label_shift_dry_run(capsys):
    code = rmnist_label_shift_main(["--dry-run", "--", "--output-dir", "analysis_outputs/smoke"])

    captured = capsys.readouterr()
    assert code == 0
    assert "run_rmnist_label_shift.py" in captured.out
    assert "--output-dir analysis_outputs/smoke" in captured.out


def test_validation_cli(capsys):
    code = validation_main([_validation_path()])

    captured = capsys.readouterr()
    assert code == 0
    assert '"passed": true' in captured.out
