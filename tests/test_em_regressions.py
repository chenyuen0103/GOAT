import argparse
import gc
import weakref

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

import em_utils


class _Encoded:
    def __init__(self, data, targets=None):
        self.data = np.asarray(data, dtype=np.float32)
        self.targets = np.asarray(targets if targets is not None else [0] * len(data))

    def __len__(self):
        return len(self.data)


def test_implicit_em_cache_expires_with_dataset_owner():
    em_utils.clear_em_representation_cache()
    first = _Encoded([[0.0], [1.0], [10.0]])
    x_first, _, _, key = em_utils.prepare_em_representation(
        first, do_pca=False, pca_dim=None
    )
    x_again, _, _, same_key = em_utils.prepare_em_representation(
        first, do_pca=False, pca_dim=None
    )

    assert same_key == key
    assert x_again is x_first
    assert key in em_utils._EM_CACHE

    owner = weakref.ref(first)
    del first
    gc.collect()

    assert owner() is None
    assert key not in em_utils._EM_CACHE

    second = _Encoded([[0.0], [5.0], [10.0]])
    x_second, _, _, _ = em_utils.prepare_em_representation(
        second, do_pca=False, pca_dim=None
    )
    assert not np.allclose(x_first, x_second)


def test_em_seed_offsets_are_combined_with_run_seed(monkeypatch):
    seen_rng = []

    def fake_run_em_on_encoded(_dataset, **kwargs):
        seen_rng.append(kwargs["rng"])
        return {
            "X": np.zeros((4, 2), dtype=np.float32),
            "gamma": np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=float),
            "labels": np.array([0, 1, 0, 1]),
            "ll_curve": [-1.0],
        }

    def fake_map(_em_res, **_kwargs):
        return {0: 0, 1: 1}, np.array([0, 1, 0, 1]), np.zeros((2, 2))

    monkeypatch.setattr(em_utils, "run_em_on_encoded", fake_run_em_on_encoded)
    monkeypatch.setattr(em_utils, "map_em_clusters", fake_map)

    args = argparse.Namespace(
        em_match="prototypes",
        _cached_source_stats=(None, None, None),
    )
    models = em_utils.fit_many_em_on_target(
        _Encoded(np.zeros((4, 2)), targets=[0, 1, 0, 1]),
        K_list=[2],
        cov_types=["diag"],
        seeds=[0, 2],
        pca_dims=[None],
        rng_base=11,
        args=args,
    )

    assert seen_rng == [11, 13]
    assert [model["cfg"]["seed"] for model in models] == [0, 2]
    assert [model["cfg"]["rng_seed"] for model in models] == [11, 13]

    legacy_models = em_utils.fit_many_em_on_target(
        _Encoded(np.zeros((4, 2)), targets=[0, 1, 0, 1]),
        K_list=[2],
        cov_types=["diag"],
        seeds=[2],
        pca_dims=[None],
        rng_base=11,
        seed_mode="absolute",
        args=args,
    )
    assert seen_rng[-1] == 2
    assert legacy_models[0]["cfg"]["seed_mode"] == "absolute"


def test_self_train_can_leave_target_eval_only():
    pytest.importorskip("tensorflow")
    import da_algo

    args = argparse.Namespace(batch_size=2, num_workers=0, lr=1e-3)
    target = TensorDataset(
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([0, 1]),
    )
    head = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        head.weight.copy_(torch.eye(2))

    direct, final, train_curve, test_curve, predictions = da_algo.self_train(
        args,
        head,
        [target],
        epochs=1,
        label_source="em",
        train_on_target=False,
    )

    assert direct == 100.0
    assert final == direct
    assert train_curve == []
    assert test_curve == [direct]
    assert predictions.tolist() == [0, 1]
