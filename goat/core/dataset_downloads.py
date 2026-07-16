"""Idempotent download and preparation helpers for GOAT datasets."""

from __future__ import annotations

import argparse
import gzip
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable, Sequence

from goat.core.artifacts import ArtifactPaths


PORTRAITS_URL = os.environ.get(
    "GOAT_PORTRAITS_URL",
    "https://www.dropbox.com/s/ubjjoo0b2wz4vgz/"
    "faces_aligned_small_mirrored_co_aligned_cropped_cleaned.tar.gz?dl=1",
)
COVTYPE_URL = os.environ.get(
    "GOAT_COVTYPE_URL",
    "https://archive.ics.uci.edu/static/public/31/covertype.zip",
)
DATASET_CHOICES = ("mnist", "portraits", "covtype")


def _print_command(parts: Iterable[str]) -> None:
    print("+ " + " ".join(str(part) for part in parts), flush=True)


def download_file(
    url: str,
    destination: Path,
    *,
    force: bool = False,
    dry_run: bool = False,
    retries: int = 3,
) -> Path:
    destination = Path(destination)
    if destination.is_file() and destination.stat().st_size > 0 and not force:
        print(f"[download] Reusing {destination}")
        return destination
    if dry_run:
        _print_command(("download", url, "->", str(destination)))
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "GOAT-dataset-setup/1.0"})
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                expected = int(response.headers.get("Content-Length", "0") or 0)
                downloaded = 0
                next_report = 10
                with partial.open("wb") as output:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)
                        downloaded += len(chunk)
                        if expected:
                            percent = int(downloaded * 100 / expected)
                            if percent >= next_report:
                                print(f"[download] {destination.name}: {percent}%", flush=True)
                                next_report = min(100, percent + 10)
            if expected and downloaded != expected:
                raise OSError(
                    f"incomplete download for {destination}: {downloaded}/{expected} bytes"
                )
            partial.replace(destination)
            return destination
        except Exception:
            partial.unlink(missing_ok=True)
            if attempt == retries:
                raise
            print(f"[download] Attempt {attempt} failed; retrying...", flush=True)
            time.sleep(attempt * 2)
    raise RuntimeError("unreachable")


def _safe_destination(root: Path, member_name: str) -> Path:
    root = root.resolve()
    destination = (root / member_name).resolve()
    if os.path.commonpath((str(root), str(destination))) != str(root):
        raise RuntimeError(f"archive member escapes extraction root: {member_name}")
    return destination


def _extract_tar_safely(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(archive, "r:*") as handle:
            members = handle.getmembers()
            for member in members:
                _safe_destination(destination, member.name)
                if member.issym() or member.islnk() or member.isdev():
                    raise RuntimeError(f"unsupported link/device in archive: {member.name}")
            handle.extractall(destination, members=members)
    except tarfile.TarError as exc:
        raise RuntimeError(
            f"Portraits download is not a readable tar archive: {archive}. "
            "Delete it or rerun with --force."
        ) from exc


def _find_portrait_class(root: Path, class_name: str) -> Path:
    candidates = [path for path in root.rglob(class_name) if path.is_dir()]
    if not candidates:
        raise RuntimeError(f"Portraits archive does not contain a {class_name}/ directory")
    return max(candidates, key=lambda path: sum(item.is_file() for item in path.rglob("*")))


def extract_portrait_classes(archive: Path, destination: Path) -> None:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="goat-portraits-", dir=destination.parent) as tmp:
        extraction_root = Path(tmp)
        _extract_tar_safely(Path(archive), extraction_root)
        for class_name in ("M", "F"):
            source = _find_portrait_class(extraction_root, class_name)
            target = destination / class_name
            shutil.copytree(source, target, dirs_exist_ok=True)


def _directory_has_files(path: Path) -> bool:
    return path.is_dir() and any(item.is_file() for item in path.rglob("*"))


def prepare_mnist(data_root: Path, keras_home: Path, *, dry_run: bool = False) -> None:
    mnist_root = data_root / "mnist"
    if dry_run:
        _print_command(("torchvision MNIST", "->", str(mnist_root)))
        _print_command(("Keras MNIST", "->", str(keras_home)))
        return

    mnist_root.mkdir(parents=True, exist_ok=True)
    keras_home.mkdir(parents=True, exist_ok=True)
    os.environ["KERAS_HOME"] = str(keras_home)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    from torchvision.datasets import MNIST

    MNIST(root=str(mnist_root), train=True, download=True)
    MNIST(root=str(mnist_root), train=False, download=True)

    from tensorflow.keras.datasets import mnist as keras_mnist

    keras_mnist.load_data()
    print("[mnist] Torchvision and Keras caches are ready.")


def prepare_portraits(
    data_root: Path,
    *,
    preprocess: bool,
    force: bool,
    dry_run: bool,
    repo_root: Path,
) -> None:
    portraits_root = data_root / "portraits"
    raw_dir = portraits_root / "dataset_32x32"
    output_file = portraits_root / "dataset_32x32.mat"
    archive = portraits_root / "portraits.tar.gz"

    raw_ready = all(_directory_has_files(raw_dir / name) for name in ("M", "F"))
    if raw_ready and not force:
        print(f"[portraits] Reusing extracted classes in {raw_dir}")
    else:
        download_file(PORTRAITS_URL, archive, force=force, dry_run=dry_run)
        if dry_run:
            _print_command(("extract M/ and F/", str(archive), "->", str(raw_dir)))
        else:
            extract_portrait_classes(archive, raw_dir)
            print(f"[portraits] Extracted M/ and F/ into {raw_dir}")

    if not preprocess:
        return
    if output_file.is_file() and output_file.stat().st_size > 0 and not force:
        print(f"[portraits] Reusing processed file {output_file}")
        return
    command = [
        sys.executable,
        str(repo_root / "create_dataset.py"),
        "--input-dir",
        str(raw_dir),
        "--output-file",
        str(output_file),
    ]
    _print_command(command)
    if not dry_run:
        subprocess.run(command, cwd=repo_root, check=True)


def extract_covtype_archive(archive: Path, destination: Path) -> None:
    archive = Path(archive)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    with zipfile.ZipFile(archive) as handle:
        candidates = [
            name
            for name in handle.namelist()
            if name.endswith("covtype.data") or name.endswith("covtype.data.gz")
        ]
        if not candidates:
            raise RuntimeError(f"Covertype archive has no covtype.data(.gz): {archive}")
        member = min(candidates, key=len)
        _safe_destination(destination.parent, Path(member).name)
        with handle.open(member) as source, partial.open("wb") as output:
            if member.endswith(".gz"):
                with gzip.GzipFile(fileobj=source) as decompressed:
                    shutil.copyfileobj(decompressed, output)
            else:
                shutil.copyfileobj(source, output)
    partial.replace(destination)


def validate_covtype(path: Path) -> None:
    with Path(path).open("rt", encoding="utf-8") as handle:
        first_row = handle.readline().strip().split(",")
    if len(first_row) != 55:
        raise RuntimeError(
            f"Expected 55 comma-separated Covertype columns, found {len(first_row)}"
        )


def prepare_covtype(
    data_root: Path,
    *,
    force: bool = False,
    dry_run: bool = False,
) -> None:
    covtype_root = data_root / "covtype"
    archive = covtype_root / "covertype.zip"
    destination = covtype_root / "covtype.data"
    if destination.is_file() and destination.stat().st_size > 0 and not force:
        validate_covtype(destination)
        print(f"[covtype] Reusing {destination}")
        return
    download_file(COVTYPE_URL, archive, force=force, dry_run=dry_run)
    if dry_run:
        _print_command(("extract covtype.data", str(archive), "->", str(destination)))
        return
    extract_covtype_archive(archive, destination)
    validate_covtype(destination)
    print(f"[covtype] Ready: {destination}")


def normalize_datasets(values: Sequence[str]) -> tuple[str, ...]:
    if "all" in values:
        return DATASET_CHOICES
    return tuple(dict.fromkeys(values))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    paths = ArtifactPaths.from_env()
    project_root = os.environ.get("GOAT_PROJECT_ROOT")
    default_keras = (
        Path(os.environ["KERAS_HOME"]).expanduser()
        if os.environ.get("KERAS_HOME")
        else Path(project_root).expanduser() / "cache" / "keras"
        if project_root
        else paths.cache_dir / "keras"
    )
    parser = argparse.ArgumentParser(
        description="Download and prepare every dataset used by current GOAT experiments."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("all", *DATASET_CHOICES),
        default=["all"],
    )
    parser.add_argument("--data-root", type=Path, default=paths.data_dir)
    parser.add_argument("--keras-home", type=Path, default=default_keras)
    parser.add_argument(
        "--no-preprocess-portraits",
        action="store_true",
        help="Download/extract Portraits but do not create dataset_32x32.mat.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    datasets = normalize_datasets(args.datasets)
    data_root = args.data_root.expanduser().resolve()
    keras_home = args.keras_home.expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[2]
    print(f"[datasets] data_root={data_root}")
    print(f"[datasets] selected={','.join(datasets)}")

    if "mnist" in datasets:
        prepare_mnist(data_root, keras_home, dry_run=args.dry_run)
    if "portraits" in datasets:
        prepare_portraits(
            data_root,
            preprocess=not args.no_preprocess_portraits,
            force=args.force,
            dry_run=args.dry_run,
            repo_root=repo_root,
        )
    if "covtype" in datasets:
        prepare_covtype(data_root, force=args.force, dry_run=args.dry_run)
    print("[datasets] Complete.")
    return 0
