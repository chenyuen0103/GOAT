from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


LEGACY_MNIST_DATA_ROOT = Path("/data/common/yuenchen")
LEGACY_MNIST_MODEL_DIR = Path("/data/common/yuenchen/GDA/mnist_models")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


def _explicit_path(name: str) -> Optional[Path]:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else None


def mnist_data_root() -> Path:
    """Resolve the torchvision MNIST root without breaking the Euler default."""

    explicit = _explicit_path("GOAT_MNIST_ROOT")
    if explicit is not None:
        return explicit
    data_root = _explicit_path("GOAT_DATA_DIR")
    if data_root is not None:
        return data_root / "mnist"
    return LEGACY_MNIST_DATA_ROOT


def mnist_model_dir() -> Path:
    """Resolve RMNIST checkpoints without breaking the Euler default."""

    explicit = _explicit_path("GOAT_MNIST_MODEL_DIR")
    if explicit is not None:
        return explicit
    model_root = _explicit_path("GOAT_MODEL_DIR")
    if model_root is not None:
        return model_root / "mnist"
    return LEGACY_MNIST_MODEL_DIR


def dataset_model_dir(dataset: str, legacy_default: str | Path) -> Path:
    """Use GOAT_MODEL_DIR when configured, otherwise preserve a legacy path."""

    model_root = _explicit_path("GOAT_MODEL_DIR")
    if model_root is not None:
        return model_root / str(dataset)
    return Path(legacy_default).expanduser()


def portraits_raw_dir() -> Path:
    explicit = _explicit_path("GOAT_PORTRAITS_RAW_DIR")
    if explicit is not None:
        return explicit
    data_root = _explicit_path("GOAT_DATA_DIR")
    if data_root is not None:
        return data_root / "portraits" / "dataset_32x32"
    return Path("dataset_32x32")


def portraits_data_file() -> Path:
    explicit = _explicit_path("GOAT_PORTRAITS_FILE")
    if explicit is not None:
        return explicit
    data_root = _explicit_path("GOAT_DATA_DIR")
    if data_root is not None:
        return data_root / "portraits" / "dataset_32x32.mat"
    return Path("dataset_32x32.mat")


def covtype_data_file() -> Path:
    explicit = _explicit_path("GOAT_COVTYPE_FILE")
    if explicit is not None:
        return explicit
    data_root = _explicit_path("GOAT_DATA_DIR")
    if data_root is not None:
        return data_root / "covtype" / "covtype.data"
    return Path("covtype.data")


def experiment_cache_dir(
    *,
    dataset: str,
    ssl_weight: float,
    seed: int,
    model_token: str,
    small_dim: int,
    gt_domains: int = 0,
    target: Optional[int] = None,
) -> Path:
    """Resolve an encoded-feature cache while preserving legacy defaults."""

    configured_root = _explicit_path("GOAT_CACHE_DIR")
    if configured_root is not None:
        base = configured_root / str(dataset) / f"ssl{ssl_weight}"
    elif dataset == "mnist":
        base = Path(f"cache{ssl_weight}")
    else:
        base = Path(str(dataset)) / f"cache{ssl_weight}"

    if dataset == "mnist":
        if target is None:
            raise ValueError("target must be provided for an MNIST cache path")
        return (
            base
            / f"target{int(target)}"
            / "prepared_v1"
            / str(model_token)
            / f"small_dim{int(small_dim)}"
        )
    return (
        base
        / f"s{int(seed)}"
        / "prepared_v1"
        / str(model_token)
        / f"gt{int(gt_domains)}"
        / f"small_dim{int(small_dim)}"
    )


@dataclass(frozen=True)
class ArtifactPaths:
    """Resolved artifact roots with backward-compatible defaults."""

    repo_root: Path
    data_dir: Path
    model_dir: Path
    cache_dir: Path
    output_dir: Path
    log_root: Path
    plot_root: Path

    @classmethod
    def from_env(cls, root: Optional[Path] = None) -> "ArtifactPaths":
        root = Path(root) if root is not None else repo_root()
        data_dir = _env_path("GOAT_DATA_DIR", root / "data")
        model_dir = _env_path("GOAT_MODEL_DIR", root / "models")
        cache_dir = _env_path("GOAT_CACHE_DIR", root / "cache0.1")
        output_dir = _env_path("GOAT_OUTPUT_DIR", root)
        log_root = _env_path("LOG_ROOT", output_dir / "logs_rerun")
        plot_root = _env_path("PLOT_ROOT", output_dir / "plots_rerun")
        return cls(
            repo_root=root,
            data_dir=data_dir,
            model_dir=model_dir,
            cache_dir=cache_dir,
            output_dir=output_dir,
            log_root=log_root,
            plot_root=plot_root,
        )

    def log_dir(self, dataset: str, seed: int, target: Optional[int] = None) -> Path:
        path = self.log_root / dataset / f"s{int(seed)}"
        if target is not None:
            path = path / f"target{int(target)}"
        return path

    def plot_dir(self, dataset: str, seed: int, target: Optional[int] = None) -> Path:
        path = self.plot_root / dataset / f"s{int(seed)}"
        if target is not None:
            path = path / f"target{int(target)}"
        return path

    def to_dict(self) -> dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}
