from pathlib import Path

import pytest

from goat.analysis.logs import load_run_records, method_names
from goat.analysis.validation import ValidationSummary
from goat.core.metrics import final_risks_from_accuracy_curves


def _path_or_fixture(primary: str, fixture: str) -> Path:
    primary_path = Path(primary)
    if primary_path.exists():
        return primary_path
    return Path("tests/fixtures") / fixture


def test_load_existing_curve_jsonl():
    path = _path_or_fixture(
        "logs/mnist/s0/target45/test_acc_dim2048_int0_gen2_pseudo_prototypes_bic_curves.jsonl",
        "legacy_curves.jsonl",
    )
    records = load_run_records(path)

    assert len(records) == 1
    record = records[0]
    assert record.seed == 0
    assert "goat_classwise" in record.methods
    assert method_names(records) == ["goat", "goat_classwise", "ours_eta", "ours_fr"]
    assert final_risks_from_accuracy_curves(
        [record.methods["goat_classwise"].test_curve]
    ) == [pytest.approx(0.3739)]


def test_load_existing_validation_json():
    path = _path_or_fixture(
        "analysis_outputs/rmnist_label_shift_spectrum/validation.json",
        "validation.json",
    )
    summary = ValidationSummary.from_file(path)

    assert summary.passed is True
    assert summary.expected_result_rows == 630
    assert summary.actual_result_rows == 630
    assert summary.expected_methods == ("goat", "cgda_fr", "cgda_fr_oracle")
