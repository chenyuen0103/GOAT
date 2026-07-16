"""Core typed interfaces and low-level utilities for GOAT."""

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

