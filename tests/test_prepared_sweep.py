from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from em_utils import map_em_models_to_classes
from goat.core.prepared_artifacts import (
    PreparedArtifactStore,
    fingerprint_encoded_dataset,
)
from goat.experiments.prepared_sweep import (
    build_prepare_commands,
    build_worker_commands,
    parse_args,
)


def test_encoded_fingerprint_uses_content_not_object_identity():
    first = SimpleNamespace(data=torch.arange(12, dtype=torch.float32).reshape(3, 4))
    equal_copy = SimpleNamespace(data=first.data.clone())
    changed = SimpleNamespace(data=first.data.clone())
    changed.data[0, 0] = -1

    assert fingerprint_encoded_dataset(first) == fingerprint_encoded_dataset(equal_copy)
    assert fingerprint_encoded_dataset(first) != fingerprint_encoded_dataset(changed)


def test_prepared_store_validates_metadata_and_payload(tmp_path: Path):
    store = PreparedArtifactStore(tmp_path)
    metadata = {"feature_sha256": "abc", "seeds": [0, 1, 2]}
    payload = {"labels": np.array([0, 1, 0]), "value": 7}

    payload_path = store.save("raw_em", metadata, payload)
    loaded = store.load("raw_em", metadata, required=True)
    assert loaded["value"] == 7
    assert loaded["labels"].tolist() == [0, 1, 0]
    assert store.load("raw_em", {**metadata, "seeds": [3]}, required=False) is None

    payload_path.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="failed validation"):
        store.load("raw_em", metadata, required=True)


def _raw_em_model():
    gamma = np.array(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.1, 0.9],
            [0.2, 0.8],
        ]
    )
    return {
        "cfg": {"K": 2, "cov_type": "diag", "pca_dim": None, "seed": 0},
        "em_res": {
            "labels": gamma.argmax(axis=1),
            "gamma": gamma,
            "mu": {0: np.array([0.0]), 1: np.array([10.0])},
            "Sigma": {0: np.array([1.0]), 1: np.array([1.0])},
            "pi": {0: 0.5, 1: 0.5},
            "ll_curve": [-10.0, -5.0],
        },
        "final_ll": -5.0,
        "bic": 20.0,
    }


def test_raw_em_fit_can_be_mapped_twice_without_mutation():
    raw = [_raw_em_model()]
    prototype_args = Namespace(
        em_match="prototypes",
        _cached_source_stats=(
            {0: np.array([10.0]), 1: np.array([0.0])},
            {0: np.array([1.0]), 1: np.array([1.0])},
            {0: 0.5, 1: 0.5},
        ),
    )
    pseudo_args = Namespace(
        em_match="pseudo",
        _cached_pseudolabels=np.array([0, 0, 1, 1]),
    )

    by_prototype = map_em_models_to_classes(raw, args=prototype_args, n_classes=2)
    by_agreement = map_em_models_to_classes(raw, args=pseudo_args, n_classes=2)

    assert by_prototype[0]["labels_mapped"].tolist() == [1, 1, 0, 0]
    assert by_agreement[0]["labels_mapped"].tolist() == [0, 0, 1, 1]
    assert "mapping" not in raw[0]
    assert "mapped_soft" not in raw[0]


def test_prepared_sweep_plans_one_prep_and_isolated_mapping_workers():
    args = parse_args(
        [
            "--dataset",
            "mnist",
            "--seeds",
            "0",
            "--gt-domains",
            "0",
            "--generated-domains",
            "0",
            "1",
            "2",
            "3",
            "--em-matches",
            "prototypes",
            "pseudo",
            "--dry-run",
        ]
    )
    prepare = build_prepare_commands(args)
    workers = build_worker_commands(args)

    assert len(prepare) == 1
    assert "--prepare-only" in prepare[0]
    assert "--require-prepared-artifacts" not in prepare[0]
    assert len(workers) == 7  # gen=0 once; gen=1,2,3 under both mappings
    assert all("--require-prepared-artifacts" in command for command in workers)

    gen0 = [command for command in workers if command[command.index("--generated-domains") + 1] == "0"]
    assert len(gen0) == 1
    mapped = {
        command[command.index("--em-match") + 1]
        for command in workers
        if command[command.index("--generated-domains") + 1] != "0"
    }
    assert mapped == {"prototypes", "pseudo"}
    prototype_workers = [
        command
        for command in workers
        if command[command.index("--em-match") + 1] == "prototypes"
    ]
    pseudo_workers = [
        command
        for command in workers
        if command[command.index("--em-match") + 1] == "pseudo"
    ]
    assert all("--skip-pooled-goat" not in command for command in prototype_workers)
    assert all("--skip-pooled-goat" in command for command in pseudo_workers)
