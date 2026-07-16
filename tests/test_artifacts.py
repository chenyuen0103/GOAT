from pathlib import Path

from goat.core.artifacts import (
    LEGACY_MNIST_DATA_ROOT,
    LEGACY_MNIST_MODEL_DIR,
    ArtifactPaths,
    covtype_data_file,
    dataset_model_dir,
    experiment_cache_dir,
    mnist_data_root,
    mnist_model_dir,
    portraits_data_file,
    portraits_raw_dir,
)


def test_artifact_defaults_preserve_legacy_roots(monkeypatch):
    for name in (
        "GOAT_DATA_DIR",
        "GOAT_MODEL_DIR",
        "GOAT_CACHE_DIR",
        "GOAT_OUTPUT_DIR",
        "LOG_ROOT",
        "PLOT_ROOT",
    ):
        monkeypatch.delenv(name, raising=False)

    paths = ArtifactPaths.from_env(Path("/tmp/goat-root"))

    assert paths.data_dir == Path("/tmp/goat-root/data")
    assert paths.model_dir == Path("/tmp/goat-root/models")
    assert paths.cache_dir == Path("/tmp/goat-root/cache0.1")
    assert paths.output_dir == Path("/tmp/goat-root")
    assert paths.log_root == Path("/tmp/goat-root/logs_rerun")
    assert paths.plot_root == Path("/tmp/goat-root/plots_rerun")
    assert paths.log_dir("mnist", 2, target=45) == Path("/tmp/goat-root/logs_rerun/mnist/s2/target45")


def test_artifact_env_overrides(monkeypatch):
    monkeypatch.setenv("GOAT_OUTPUT_DIR", "/tmp/goat-out")
    monkeypatch.setenv("GOAT_CACHE_DIR", "/tmp/goat-cache")
    monkeypatch.setenv("GOAT_DATA_DIR", "/tmp/goat-data")
    monkeypatch.setenv("GOAT_MODEL_DIR", "/tmp/goat-models")

    paths = ArtifactPaths.from_env(Path("/tmp/goat-root"))

    assert paths.output_dir == Path("/tmp/goat-out")
    assert paths.cache_dir == Path("/tmp/goat-cache")
    assert paths.data_dir == Path("/tmp/goat-data")
    assert paths.model_dir == Path("/tmp/goat-models")


def test_current_experiment_paths_follow_portable_roots(monkeypatch):
    for name in (
        "GOAT_MNIST_ROOT",
        "GOAT_MNIST_MODEL_DIR",
        "GOAT_PORTRAITS_RAW_DIR",
        "GOAT_PORTRAITS_FILE",
        "GOAT_COVTYPE_FILE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GOAT_DATA_DIR", "/project/yuen_chen/data")
    monkeypatch.setenv("GOAT_MODEL_DIR", "/project/yuen_chen/models")
    monkeypatch.setenv("GOAT_CACHE_DIR", "/project/yuen_chen/cache/goat")

    assert mnist_data_root() == Path("/project/yuen_chen/data/mnist")
    assert mnist_model_dir() == Path("/project/yuen_chen/models/mnist")
    assert portraits_raw_dir() == Path(
        "/project/yuen_chen/data/portraits/dataset_32x32"
    )
    assert portraits_data_file() == Path(
        "/project/yuen_chen/data/portraits/dataset_32x32.mat"
    )
    assert covtype_data_file() == Path(
        "/project/yuen_chen/data/covtype/covtype.data"
    )
    assert dataset_model_dir("portraits", "legacy") == Path(
        "/project/yuen_chen/models/portraits"
    )
    assert experiment_cache_dir(
        dataset="mnist",
        ssl_weight=0.1,
        seed=2,
        model_token="abc",
        small_dim=2048,
        target=45,
    ) == Path(
        "/project/yuen_chen/cache/goat/mnist/ssl0.1/target45/"
        "prepared_v1/abc/small_dim2048"
    )


def test_dataset_specific_paths_override_roots_and_legacy_defaults(monkeypatch):
    monkeypatch.delenv("GOAT_DATA_DIR", raising=False)
    monkeypatch.delenv("GOAT_MODEL_DIR", raising=False)
    monkeypatch.delenv("GOAT_MNIST_ROOT", raising=False)
    monkeypatch.delenv("GOAT_MNIST_MODEL_DIR", raising=False)
    assert mnist_data_root() == LEGACY_MNIST_DATA_ROOT
    assert mnist_model_dir() == LEGACY_MNIST_MODEL_DIR

    monkeypatch.setenv("GOAT_MNIST_ROOT", "/tmp/special-mnist")
    monkeypatch.setenv("GOAT_MNIST_MODEL_DIR", "/tmp/special-models")
    assert mnist_data_root() == Path("/tmp/special-mnist")
    assert mnist_model_dir() == Path("/tmp/special-models")
