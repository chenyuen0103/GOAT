from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


@dataclass(frozen=True)
class ArtifactPaths:
    """Resolved artifact roots with backward-compatible defaults."""

    repo_root: Path
    data_dir: Path
    cache_dir: Path
    output_dir: Path
    log_root: Path
    plot_root: Path

    @classmethod
    def from_env(cls, root: Optional[Path] = None) -> "ArtifactPaths":
        root = Path(root) if root is not None else repo_root()
        data_dir = _env_path("GOAT_DATA_DIR", root / "data")
        cache_dir = _env_path("GOAT_CACHE_DIR", root / "cache0.1")
        output_dir = _env_path("GOAT_OUTPUT_DIR", root)
        log_root = _env_path("LOG_ROOT", output_dir / "logs_rerun")
        plot_root = _env_path("PLOT_ROOT", output_dir / "plots_rerun")
        return cls(
            repo_root=root,
            data_dir=data_dir,
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

