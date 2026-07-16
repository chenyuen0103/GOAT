"""Lightweight public package for GOAT experiment orchestration.

The package is intentionally import-light. Heavy research dependencies such as
Torch, TensorFlow, and POT are imported only inside the adapters that need them.
"""

from goat.core.artifacts import ArtifactPaths, repo_root
from goat.core.schema import ExperimentConfig, MethodResult, RunRecord, RunSpec

__all__ = [
    "ArtifactPaths",
    "ExperimentConfig",
    "MethodResult",
    "RunRecord",
    "RunSpec",
    "repo_root",
]

