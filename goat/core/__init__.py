"""Core typed interfaces and low-level utilities for GOAT."""

from goat.core.artifacts import (
    ArtifactPaths,
    covtype_data_file,
    dataset_model_dir,
    experiment_cache_dir,
    mnist_data_root,
    mnist_model_dir,
    portraits_data_file,
    portraits_raw_dir,
    repo_root,
)
from goat.core.schema import ExperimentConfig, MethodResult, RunRecord, RunSpec

__all__ = [
    "ArtifactPaths",
    "ExperimentConfig",
    "MethodResult",
    "RunRecord",
    "RunSpec",
    "covtype_data_file",
    "dataset_model_dir",
    "experiment_cache_dir",
    "mnist_data_root",
    "mnist_model_dir",
    "portraits_data_file",
    "portraits_raw_dir",
    "repo_root",
]
