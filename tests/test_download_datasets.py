import gzip
import io
import tarfile
import zipfile
from pathlib import Path

from goat.core.dataset_downloads import (
    extract_covtype_archive,
    extract_portrait_classes,
    main,
    normalize_datasets,
    validate_covtype,
)


def test_normalize_datasets_expands_all_and_deduplicates():
    assert normalize_datasets(["all"]) == ("mnist", "portraits", "covtype")
    assert normalize_datasets(["covtype", "mnist", "covtype"]) == (
        "covtype",
        "mnist",
    )


def test_extract_portrait_classes_finds_nested_m_and_f(tmp_path: Path):
    archive = tmp_path / "portraits.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        for class_name in ("M", "F"):
            payload = f"{class_name}-image".encode()
            info = tarfile.TarInfo(
                f"faces_aligned_small/nested/{class_name}/{class_name.lower()}001.jpg"
            )
            info.size = len(payload)
            handle.addfile(info, io.BytesIO(payload))

    destination = tmp_path / "dataset_32x32"
    extract_portrait_classes(archive, destination)

    assert (destination / "M" / "m001.jpg").read_bytes() == b"M-image"
    assert (destination / "F" / "f001.jpg").read_bytes() == b"F-image"


def test_extract_covtype_archive_decompresses_nested_gzip(tmp_path: Path):
    row = ",".join(str(value) for value in range(55)) + "\n"
    archive = tmp_path / "covertype.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("covtype/covtype.data.gz", gzip.compress(row.encode()))

    destination = tmp_path / "covtype.data"
    extract_covtype_archive(archive, destination)
    validate_covtype(destination)

    assert destination.read_text() == row


def test_all_dataset_dry_run_does_not_create_data(tmp_path: Path, capsys):
    data_root = tmp_path / "data"
    code = main(
        [
            "--datasets",
            "all",
            "--data-root",
            str(data_root),
            "--keras-home",
            str(tmp_path / "keras"),
            "--dry-run",
        ]
    )

    output = capsys.readouterr().out
    assert code == 0
    assert "torchvision MNIST" in output
    assert "extract M/ and F/" in output
    assert "extract covtype.data" in output
    assert not data_root.exists()
