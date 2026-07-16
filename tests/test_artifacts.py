from pathlib import Path

from goat.core.artifacts import ArtifactPaths


def test_artifact_defaults_preserve_legacy_roots(monkeypatch):
    for name in ("GOAT_DATA_DIR", "GOAT_CACHE_DIR", "GOAT_OUTPUT_DIR", "LOG_ROOT", "PLOT_ROOT"):
        monkeypatch.delenv(name, raising=False)

    paths = ArtifactPaths.from_env(Path("/tmp/goat-root"))

    assert paths.data_dir == Path("/tmp/goat-root/data")
    assert paths.cache_dir == Path("/tmp/goat-root/cache0.1")
    assert paths.output_dir == Path("/tmp/goat-root")
    assert paths.log_root == Path("/tmp/goat-root/logs_rerun")
    assert paths.plot_root == Path("/tmp/goat-root/plots_rerun")
    assert paths.log_dir("mnist", 2, target=45) == Path("/tmp/goat-root/logs_rerun/mnist/s2/target45")


def test_artifact_env_overrides(monkeypatch):
    monkeypatch.setenv("GOAT_OUTPUT_DIR", "/tmp/goat-out")
    monkeypatch.setenv("GOAT_CACHE_DIR", "/tmp/goat-cache")
    monkeypatch.setenv("GOAT_DATA_DIR", "/tmp/goat-data")

    paths = ArtifactPaths.from_env(Path("/tmp/goat-root"))

    assert paths.output_dir == Path("/tmp/goat-out")
    assert paths.cache_dir == Path("/tmp/goat-cache")
    assert paths.data_dir == Path("/tmp/goat-data")

