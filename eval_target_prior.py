#!/usr/bin/env python3
"""
Evaluate how well the GMM-based target prior estimate

    p̂_T(k) = (1/m) ∑_{j=1}^m 1{ŷ_j = k}

matches the true target label marginal p_T(Y) computed from ground-truth labels.

This script is designed to run on the same cached encoded domains used by the
experiments (e.g., cache*/target*/small_dim*/encoded_*.pt).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


def _require_torch():
    try:
        import torch  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "This script requires PyTorch to load cached `encoded_*.pt` files."
        ) from e


def _load_encoded_pt(path: str):
    _require_torch()
    import torch

    # PyTorch >= 2.6 defaults `weights_only=True`, which refuses to unpickle custom
    # dataset objects (our cached `encoded_*.pt` are full objects like EncodeDataset).
    # These caches are produced locally by this repo’s pipeline, so we load them as
    # full pickles.
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # Older PyTorch: no `weights_only` kwarg.
        obj = torch.load(path, map_location="cpu")
    return obj


def _get_attr(obj, name: str):
    if isinstance(obj, dict):
        return obj.get(name, None)
    return getattr(obj, name, None)


def _to_numpy(x) -> np.ndarray:
    if x is None:
        return None
    try:
        import torch

        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def _bincount_prob(y: np.ndarray, K: int) -> np.ndarray:
    y = np.asarray(y, dtype=int).reshape(-1)
    if y.size == 0:
        return np.zeros((K,), dtype=float)
    counts = np.bincount(y, minlength=K).astype(float)
    return counts / max(1.0, float(counts.sum()))


def _l1(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.abs(a - b).sum())


def _kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * (np.log(p) - np.log(q))))


@dataclass
class PriorEval:
    K: int
    m: int
    p_true: np.ndarray
    p_hat_hard: np.ndarray
    p_hat_soft: Optional[np.ndarray]
    l1: float
    kl_true_hat: float
    kl_hat_true: float
    delta_l1_vs_source: Optional[float] = None

    def to_jsonable(self) -> Dict[str, Any]:
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
        return d


def _fit_em_and_get_posts(
    X: np.ndarray,
    *,
    K: int,
    cov_type: str,
    seed: int,
    do_pca: bool,
    pca_dim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      gamma: (N, K_clusters) responsibilities
      z:     (N,) hard cluster ids = argmax_k gamma
    """
    from em_utils import prepare_em_representation, run_em_on_encoded_fast

    class _Wrapper:
        def __init__(self, data):
            self.data = data

    # Reuse the exact preprocessing path used by EM in the experiments.
    enc = _Wrapper(X)
    X_std, _scaler, _pca, _ = prepare_em_representation(
        enc,
        pool="gap",   # ignored for 2D features
        do_pca=bool(do_pca),
        pca_dim=int(pca_dim),
        dtype="float32",
        rng=int(seed),
        verbose=False,
    )
    em_res = run_em_on_encoded_fast(
        X_std,
        K=int(K),
        cov_type=str(cov_type),
        reg=1e-6,
        max_iter=300,
        tol=1e-5,
        n_init=5,
        subsample_init=len(X_std),
        warm_start=None,
        rng=int(seed),
        verbose=False,
    )
    gamma = np.asarray(em_res["gamma"], dtype=float)
    z = np.asarray(em_res["labels"], dtype=int)
    return gamma, z


def _oracle_mapping(cluster_ids: np.ndarray, y_true: np.ndarray) -> Dict[int, int]:
    from em_utils import best_mapping_accuracy

    _acc, mapping, _C = best_mapping_accuracy(cluster_ids, y_true)
    return mapping


def _map_clusters_to_classes(
    *,
    cluster_ids: np.ndarray,
    gamma: np.ndarray,
    mapping: Dict[int, int],
    K_classes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map:
      - hard cluster ids -> hard class ids
      - soft cluster responsibilities -> soft class posteriors
    """
    cluster_ids = np.asarray(cluster_ids, dtype=int).reshape(-1)
    gamma = np.asarray(gamma, dtype=float)
    y_hat = np.empty_like(cluster_ids)
    for i, c in enumerate(cluster_ids):
        y_hat[i] = int(mapping.get(int(c), int(c)))

    # soft: sum responsibilities of clusters mapped to the same class
    p_soft = np.zeros((gamma.shape[0], K_classes), dtype=float)
    for c_from, c_to in mapping.items():
        if 0 <= int(c_from) < gamma.shape[1] and 0 <= int(c_to) < K_classes:
            p_soft[:, int(c_to)] += gamma[:, int(c_from)]
    row = p_soft.sum(axis=1, keepdims=True)
    row[row == 0.0] = 1.0
    p_soft = p_soft / row
    return y_hat, p_soft


def evaluate_prior(
    *,
    encoded_source_path: Optional[str],
    encoded_target_path: str,
    K: Optional[int],
    cov_type: str,
    seed: int,
    do_pca: bool,
    pca_dim: int,
    mapping_mode: str,
    pseudo_labels_path: Optional[str],
) -> PriorEval:
    tgt = _load_encoded_pt(encoded_target_path)
    X = _to_numpy(_get_attr(tgt, "data"))
    y_true = _to_numpy(_get_attr(tgt, "targets"))
    if X is None or y_true is None:
        raise RuntimeError(
            f"Target file {encoded_target_path} must expose `.data` and `.targets`."
        )

    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    if K is None:
        K = int(y_true.max()) + 1 if y_true.size else 0
    K = int(K)

    gamma, z = _fit_em_and_get_posts(
        X,
        K=K,
        cov_type=cov_type,
        seed=seed,
        do_pca=do_pca,
        pca_dim=pca_dim,
    )

    mapping: Dict[int, int]
    mapping_mode = str(mapping_mode).lower()
    if mapping_mode == "oracle":
        mapping = _oracle_mapping(z, y_true)
    elif mapping_mode == "pseudo":
        if not pseudo_labels_path:
            raise ValueError("--pseudo-labels-path is required for --mapping pseudo.")
        pseudo = np.load(pseudo_labels_path)
        from em_utils import map_em_clusters

        # map_em_clusters expects res["labels"] and uses pseudo labels to Hungarian-align clusters
        mapping, _labels_mapped, _ = map_em_clusters(
            {"labels": z}, method="pseudo", n_classes=K, pseudo_labels=pseudo
        )
    elif mapping_mode == "none":
        mapping = {int(k): int(k) for k in range(int(z.max()) + 1 if z.size else 0)}
    else:
        raise ValueError("--mapping must be one of {oracle,pseudo,none}.")

    y_hat, p_soft = _map_clusters_to_classes(
        cluster_ids=z,
        gamma=gamma,
        mapping=mapping,
        K_classes=K,
    )

    p_true = _bincount_prob(y_true, K)
    p_hat_hard = _bincount_prob(y_hat, K)
    p_hat_soft = np.asarray(p_soft, dtype=float).mean(axis=0) if p_soft is not None else None

    out = PriorEval(
        K=K,
        m=int(y_true.size),
        p_true=p_true,
        p_hat_hard=p_hat_hard,
        p_hat_soft=p_hat_soft,
        l1=_l1(p_true, p_hat_hard),
        kl_true_hat=_kl(p_true, p_hat_hard),
        kl_hat_true=_kl(p_hat_hard, p_true),
        delta_l1_vs_source=None,
    )

    if encoded_source_path:
        src = _load_encoded_pt(encoded_source_path)
        y0 = _to_numpy(_get_attr(src, "targets"))
        if y0 is not None:
            y0 = np.asarray(y0, dtype=int).reshape(-1)
            p0 = _bincount_prob(y0, K)
            out.delta_l1_vs_source = _l1((p_true - p0), (p_hat_hard - p0))

    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--encoded-target", required=True, help="Path to encoded target .pt (must contain .data and .targets).")
    p.add_argument("--encoded-source", default=None, help="Optional path to encoded source .pt (for Δp evaluation).")
    p.add_argument("--K", type=int, default=None, help="Number of classes; default: infer from target ground truth.")
    p.add_argument("--cov-type", default="diag", choices=["diag", "full"], help="GMM covariance type.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--do-pca", action="store_true", help="Apply PCA before EM (matches some EM configs).")
    p.add_argument("--pca-dim", type=int, default=64)
    p.add_argument("--mapping", default="oracle", choices=["oracle", "pseudo", "none"])
    p.add_argument("--pseudo-labels-path", default=None, help="Numpy file with teacher pseudo labels (required for --mapping pseudo).")
    p.add_argument("--save-json", default=None, help="Write metrics as JSON to this path.")
    args = p.parse_args()

    res = evaluate_prior(
        encoded_source_path=args.encoded_source,
        encoded_target_path=args.encoded_target,
        K=args.K,
        cov_type=args.cov_type,
        seed=args.seed,
        do_pca=args.do_pca,
        pca_dim=args.pca_dim,
        mapping_mode=args.mapping,
        pseudo_labels_path=args.pseudo_labels_path,
    )

    print(f"K={res.K}  m={res.m}")
    print(f"L1(p_true, p_hat_hard) = {res.l1:.6f}")
    print(f"KL(p_true || p_hat_hard) = {res.kl_true_hat:.6f}")
    print(f"KL(p_hat_hard || p_true) = {res.kl_hat_true:.6f}")
    if res.delta_l1_vs_source is not None:
        print(f"L1(Δp_true, Δp_hat) vs source = {res.delta_l1_vs_source:.6f}")

    # Side-by-side marginals
    print("\nclass\tp_true\tp_hat_hard" + ("\tp_hat_soft" if res.p_hat_soft is not None else ""))
    for k in range(res.K):
        if res.p_hat_soft is not None:
            print(f"{k}\t{res.p_true[k]:.6f}\t{res.p_hat_hard[k]:.6f}\t{res.p_hat_soft[k]:.6f}")
        else:
            print(f"{k}\t{res.p_true[k]:.6f}\t{res.p_hat_hard[k]:.6f}")

    if args.save_json:
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save_json).write_text(json.dumps(res.to_jsonable(), indent=2), encoding="utf-8")
        print(f"Wrote {args.save_json}")


if __name__ == "__main__":
    main()
