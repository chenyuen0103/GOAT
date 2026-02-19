#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiment_chain_diagnostics.py

Runs gradual adaptation experiments and saves:
  (1) curves JSONL (accuracy curves etc.)
  (2) drift JSONL needed for Figure A:
      - global per-step drift
      - mean class-conditional per-step drift
      - worst-class per-step drift
      - optional per-class drift matrix (for a heatmap)

Designed to reuse your existing codebase:
  - imports domain encoding, OT / FR / Nat generators, and self-training from experiment_new.py when possible
  - uses minimal local dataset wrappers for compatibility

Example:
  python experiment_chain_diagnostics.py \
    --dataset mnist --target 90 --gt_domains 0 --generated_domains 3 \
    --seeds 0 1 2 \
    --methods goat cc_wass cc_nat cc_fr \
    --log_root /home/yuen_chen/GOAT/logs
"""

import os
import json
import time
import math
import copy
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset
except Exception as e:
    raise RuntimeError("This script requires torch.") from e


# -------------------------
# Robust imports from your repo
# -------------------------
def _try_imports():
    """
    Try to import from your existing experiment files.
    Adjust these names if your local modules differ.
    """
    imports = {}

    # Prefer experiment_new for generators and utilities (it has the classwise helpers).
    try:
        import experiment_new as expn  # type: ignore
        imports["expn"] = expn
    except Exception:
        imports["expn"] = None

    # Optional: refactor file sometimes has nicer CLI helpers; not required.
    try:
        import experiment_refrac as expr  # type: ignore
        imports["expr"] = expr
    except Exception:
        imports["expr"] = None

    return imports


IMPORTS = _try_imports()
EXPN = IMPORTS.get("expn", None)
EXPR = IMPORTS.get("expr", None)


# -------------------------
# Minimal dataset wrapper (compatible with your self_train)
# -------------------------
class EncodedTensorDataset(Dataset):
    """
    A minimal encoded dataset:
      - data: (N, d) float tensor
      - targets: (N,) long tensor (optional; can be None for unlabeled)
    Additional attributes used by your plotting / debugging are included as optional fields.
    """

    def __init__(
        self,
        data: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        *,
        is_synthetic: bool = False,
        name: str = "",
        targets_em: Optional[torch.Tensor] = None,
        targets_pseudo: Optional[torch.Tensor] = None,
    ):
        assert torch.is_tensor(data), "data must be a torch.Tensor"
        self.data = data
        self.targets = targets
        self.targets_em = targets_em
        self.targets_pseudo = targets_pseudo
        self.is_synthetic = is_synthetic
        self.name = name

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def __getitem__(self, idx: int):
        x = self.data[idx]
        if self.targets is None:
            return x
        return x, self.targets[idx]


# -------------------------
# Drift metrics (Figure A)
# -------------------------
def _cov_full(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Full covariance with Tikhonov regularization eps*I.
    X: (N, d)
    """
    if X.shape[0] <= 1:
        # Degenerate: return eps*I
        d = X.shape[1]
        return eps * np.eye(d, dtype=np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    # (d,d) unbiased covariance
    C = (Xc.T @ Xc) / max(X.shape[0] - 1, 1)
    C = 0.5 * (C + C.T)
    C = C + eps * np.eye(C.shape[0], dtype=np.float64)
    return C


def _sqrtm_psd(A: np.ndarray) -> np.ndarray:
    """
    Symmetric PSD matrix square root via eigen-decomposition.
    """
    w, V = np.linalg.eigh(A)
    w = np.clip(w, 0.0, None)
    return (V * np.sqrt(w)[None, :]) @ V.T


def w2_gaussian(mu1: np.ndarray, C1: np.ndarray, mu2: np.ndarray, C2: np.ndarray) -> float:
    """
    Squared 2-Wasserstein distance between Gaussians:
      W2^2 = ||m1-m2||^2 + Tr(C1 + C2 - 2*(C2^{1/2} C1 C2^{1/2})^{1/2})
    """
    dm = mu1 - mu2
    term_mean = float(dm @ dm)

    S2 = _sqrtm_psd(C2)
    middle = S2 @ C1 @ S2
    Smiddle = _sqrtm_psd(0.5 * (middle + middle.T))
    term_cov = float(np.trace(C1 + C2 - 2.0 * Smiddle))

    # Numerical floor
    return max(term_mean + term_cov, 0.0)


def _get_labels(ds: Any, label_source: str) -> Optional[torch.Tensor]:
    """
    label_source in {"real","em","pseudo","targets"}.
    """
    # For diagnostics, "real" can safely fall back to EM/pseudo when true labels are absent.
    # This avoids all-NaN classwise drift for methods whose synthetic steps only carry pseudo/EM labels.
    if label_source == "real":
        y = getattr(ds, "targets", None)
        if y is not None:
            return y
        y = getattr(ds, "targets_em", None)
        if y is not None:
            return y
        return getattr(ds, "targets_pseudo", None)
    if label_source == "targets":
        return getattr(ds, "targets", None)
    if label_source == "em":
        return getattr(ds, "targets_em", None)
    if label_source == "pseudo":
        return getattr(ds, "targets_pseudo", None)
    return getattr(ds, "targets", None)


@dataclass
class StepStats:
    mu: np.ndarray
    cov: np.ndarray
    class_mu: List[Optional[np.ndarray]]
    class_cov: List[Optional[np.ndarray]]
    class_n: List[int]


def compute_step_stats(
    ds: Any,
    K: int,
    label_for_conditional: str = "real",
    eps: float = 1e-6,
) -> StepStats:
    """
    Compute:
      - global mean/cov over features
      - per-class mean/cov (using chosen label source for conditional splits)
    """
    X = ds.data.detach().cpu().numpy().astype(np.float64)
    mu = X.mean(axis=0)
    cov = _cov_full(X, eps=eps)

    y = _get_labels(ds, label_for_conditional)
    class_mu: List[Optional[np.ndarray]] = [None] * K
    class_cov: List[Optional[np.ndarray]] = [None] * K
    class_n: List[int] = [0] * K

    if y is None:
        return StepStats(mu=mu, cov=cov, class_mu=class_mu, class_cov=class_cov, class_n=class_n)

    y_np = y.detach().cpu().numpy().astype(np.int64)
    for k in range(K):
        idx = np.where(y_np == k)[0]
        class_n[k] = int(idx.shape[0])
        if idx.shape[0] <= 1:
            continue
        Xk = X[idx]
        class_mu[k] = Xk.mean(axis=0)
        class_cov[k] = _cov_full(Xk, eps=eps)

    return StepStats(mu=mu, cov=cov, class_mu=class_mu, class_cov=class_cov, class_n=class_n)


def drift_between(
    A: StepStats,
    B: StepStats,
    K: int,
) -> Tuple[float, float, float, List[Optional[float]]]:
    """
    Returns:
      global_drift, mean_cond_drift, worst_cond_drift, per_class_drift (len K, None if undefined)
    """
    global_d = w2_gaussian(A.mu, A.cov, B.mu, B.cov)

    per_class: List[Optional[float]] = [None] * K
    vals: List[float] = []
    for k in range(K):
        if A.class_mu[k] is None or A.class_cov[k] is None:
            continue
        if B.class_mu[k] is None or B.class_cov[k] is None:
            continue
        d = w2_gaussian(A.class_mu[k], A.class_cov[k], B.class_mu[k], B.class_cov[k])
        per_class[k] = d
        vals.append(d)

    if len(vals) == 0:
        return global_d, float("nan"), float("nan"), per_class

    mean_cond = float(np.mean(vals))
    worst_cond = float(np.max(vals))
    return global_d, mean_cond, worst_cond, per_class


# -------------------------
# Chain construction helpers
# -------------------------
def _as_encoded_dataset(obj: Any, *, is_synthetic: bool, name: str) -> EncodedTensorDataset:
    """
    Convert your repo's encoded dataset objects to our minimal wrapper if needed.
    If obj already looks compatible, return as-is.
    """
    if hasattr(obj, "data") and torch.is_tensor(obj.data):
        # If it already has __len__/__getitem__, we can pass it through.
        # But to ensure we can attach targets_pseudo/em cleanly, we wrap.
        return EncodedTensorDataset(
            data=obj.data,
            targets=getattr(obj, "targets", None),
            is_synthetic=is_synthetic,
            name=name,
            targets_em=getattr(obj, "targets_em", None),
            targets_pseudo=getattr(obj, "targets_pseudo", None),
        )
    raise ValueError("Object does not have torch tensor .data; cannot convert.")


def subset_by_class(ds: Any, y: torch.Tensor, k: int, *, is_synthetic: bool, name: str) -> EncodedTensorDataset:
    """
    Subset ds to class k using label vector y (same length as ds).
    Produces a labeled dataset with targets all equal to k (by construction).
    """
    idx = (y == k).nonzero(as_tuple=False).flatten()
    if idx.numel() == 0:
        # Empty dataset (keep shape consistent: (0,d))
        d = ds.data.shape[1]
        return EncodedTensorDataset(
            data=ds.data.new_zeros((0, d)),
            targets=ds.data.new_zeros((0,), dtype=torch.long),
            is_synthetic=is_synthetic,
            name=name,
        )
    Xk = ds.data[idx]
    yk = torch.full((Xk.shape[0],), int(k), dtype=torch.long, device=Xk.device)
    # Keep em/pseudo labels aligned for generators that expect targets_em on target-side datasets.
    return EncodedTensorDataset(
        data=Xk,
        targets=yk,
        targets_em=yk.clone(),
        targets_pseudo=yk.clone(),
        is_synthetic=is_synthetic,
        name=name,
    )


def merge_class_steps(step_buckets: List[List[EncodedTensorDataset]], *, name_prefix: str) -> List[EncodedTensorDataset]:
    """
    step_buckets[ell] = list of class-datasets at step ell (each has (z, k)).
    Returns merged datasets per step.
    """
    merged: List[EncodedTensorDataset] = []
    for ell, class_list in enumerate(step_buckets):
        Xs, Ys = [], []
        for ds_k in class_list:
            if len(ds_k) == 0:
                continue
            Xs.append(ds_k.data)
            if ds_k.targets is None:
                raise ValueError("Classwise synthetic datasets must be labeled.")
            Ys.append(ds_k.targets)
        if len(Xs) == 0:
            # Completely empty step
            merged.append(
                EncodedTensorDataset(
                    data=torch.zeros((0, 1), dtype=torch.float32),
                    targets=torch.zeros((0,), dtype=torch.long),
                    is_synthetic=True,
                    name=f"{name_prefix}_step{ell+1}",
                )
            )
            continue
        X = torch.cat(Xs, dim=0)
        Y = torch.cat(Ys, dim=0)
        merged.append(
            EncodedTensorDataset(
                data=X,
                targets=Y,
                targets_em=Y.clone(),
                targets_pseudo=Y.clone(),
                is_synthetic=True,
                name=f"{name_prefix}_step{ell+1}",
            )
        )
    return merged


# -------------------------
# Generation backends from your repo (preferred)
# -------------------------
def _require_expn_symbol(name: str):
    if EXPN is None or not hasattr(EXPN, name):
        raise RuntimeError(f"Expected symbol '{name}' in experiment_new.py but it is not available.")
    return getattr(EXPN, name)


def build_chain_goat(
    generated_domains: int,
    encoded_reals: List[Any],
    *,
    name: str,
) -> List[EncodedTensorDataset]:
    """
    Class-agnostic OT interpolation between consecutive real domains.
    Returns chain excluding the initial source (self_train uses source teacher).
    """
    generate_domains = _require_expn_symbol("generate_domains")  # from da_algo (imported by experiment_new)
    chain: List[EncodedTensorDataset] = []

    for i in range(len(encoded_reals) - 1):
        left = encoded_reals[i]
        right = encoded_reals[i + 1]
        # out[0] is list length (generated_domains+1): includes right endpoint at the end
        out = generate_domains(generated_domains, left, right)
        pair = out[0]
        for j, ds in enumerate(pair):
            chain.append(_as_encoded_dataset(ds, is_synthetic=(j < len(pair) - 1), name=f"{name}_gap{i}_t{j+1}"))
    return chain


def build_chain_classwise_wass(
    generated_domains: int,
    encoded_reals: List[Any],
    y_for_split: List[torch.Tensor],
    K: int,
    *,
    name: str,
) -> List[EncodedTensorDataset]:
    """
    Class-conditional OT interpolation: split endpoints by class, generate within each class, merge per step.
    y_for_split[i] is label vector for encoded_reals[i] used for splitting.
      - for source: true labels
      - for unlabeled reals: EM-mapped labels (or any initial assignment)
    """
    generate_domains = _require_expn_symbol("generate_domains")

    chain: List[EncodedTensorDataset] = []
    for gap in range(len(encoded_reals) - 1):
        L = encoded_reals[gap]
        R = encoded_reals[gap + 1]
        yL = y_for_split[gap]
        yR = y_for_split[gap + 1]

        # Prepare per-step buckets (generated_domains synthetic steps) per class
        step_buckets: List[List[EncodedTensorDataset]] = [[] for _ in range(generated_domains)]

        for k in range(K):
            Lk = subset_by_class(L, yL, k, is_synthetic=False, name=f"{name}_gap{gap}_L_k{k}")
            Rk = subset_by_class(R, yR, k, is_synthetic=False, name=f"{name}_gap{gap}_R_k{k}")
            if len(Lk) == 0 or len(Rk) == 0:
                continue

            # generate within class k
            out = generate_domains(generated_domains, Lk, Rk)
            pair = out[0]  # length generated_domains+1; final is endpoint (ignored here)
            # For ell=0..generated_domains-1, pair[ell] is synthetic at step ell+1
            for ell in range(generated_domains):
                ds_ell = _as_encoded_dataset(pair[ell], is_synthetic=True, name=f"{name}_gap{gap}_k{k}_t{ell+1}")
                # Ensure labeled by construction:
                if ds_ell.targets is None:
                    ds_ell.targets = torch.full((len(ds_ell),), int(k), dtype=torch.long, device=ds_ell.data.device)
                if ds_ell.targets_em is None:
                    ds_ell.targets_em = ds_ell.targets.detach().clone()
                if ds_ell.targets_pseudo is None:
                    ds_ell.targets_pseudo = ds_ell.targets.detach().clone()
                step_buckets[ell].append(ds_ell)

        # Merge classes per step
        merged_steps = merge_class_steps(step_buckets, name_prefix=f"{name}_gap{gap}")

        # Append merged synthetic steps, then append the real right endpoint (as a real domain step)
        for ds in merged_steps:
            chain.append(ds)
        chain.append(_as_encoded_dataset(R, is_synthetic=False, name=f"{name}_gap{gap}_REAL_R"))
    return chain


def build_chain_classwise_gaussian(
    generated_domains: int,
    encoded_reals: List[Any],
    y_for_split: List[torch.Tensor],
    K: int,
    gen_method: str,
    *,
    name: str,
    cov_type: str = "full",
    args_for_repo: Optional[Any] = None,
) -> List[EncodedTensorDataset]:
    """
    Class-conditional Gaussian geodesic interpolation (FR or Nat):
      - split endpoints by class
      - run the corresponding generator between classwise datasets
      - merge per step
      - append real right endpoint
    """
    gen_method = gen_method.lower()
    if gen_method in {"fr", "fisher-rao", "fisher_rao"}:
        gen_fn = _require_expn_symbol("generate_fr_domains_between_optimized")
    elif gen_method in {"nat", "natural", "np", "eta"}:
        gen_fn = _require_expn_symbol("generate_natural_domains_between")
    else:
        raise ValueError(f"Unknown gen_method: {gen_method}")

    chain: List[EncodedTensorDataset] = []
    for gap in range(len(encoded_reals) - 1):
        L = encoded_reals[gap]
        R = encoded_reals[gap + 1]
        yL = y_for_split[gap]
        yR = y_for_split[gap + 1]

        step_buckets: List[List[EncodedTensorDataset]] = [[] for _ in range(generated_domains)]

        for k in range(K):
            Lk = subset_by_class(L, yL, k, is_synthetic=False, name=f"{name}_gap{gap}_L_k{k}")
            Rk = subset_by_class(R, yR, k, is_synthetic=False, name=f"{name}_gap{gap}_R_k{k}")
            if len(Lk) == 0 or len(Rk) == 0:
                continue

            out = gen_fn(
                generated_domains,
                Lk,
                Rk,
                cov_type=cov_type,
                save_path=None,
                args=args_for_repo,
            )
            pair = out[0]  # list length generated_domains+1 (should mirror OT generator convention)
            for ell in range(generated_domains):
                ds_ell = _as_encoded_dataset(pair[ell], is_synthetic=True, name=f"{name}_gap{gap}_k{k}_t{ell+1}")
                if ds_ell.targets is None:
                    ds_ell.targets = torch.full((len(ds_ell),), int(k), dtype=torch.long, device=ds_ell.data.device)
                step_buckets[ell].append(ds_ell)

        merged_steps = merge_class_steps(step_buckets, name_prefix=f"{name}_gap{gap}")
        for ds in merged_steps:
            chain.append(ds)
        chain.append(_as_encoded_dataset(R, is_synthetic=False, name=f"{name}_gap{gap}_REAL_R"))
    return chain


# -------------------------
# Self-training step (uses your repo's implementation if available)
# -------------------------
def run_self_training_over_chain(
    args: Any,
    source_mlp: Any,
    chain: List[EncodedTensorDataset],
    *,
    epochs: int,
    label_source: str,
):
    """
    Uses your repo's self_train (preferred) so results are consistent with your paper.
    """
    self_train = _require_expn_symbol("self_train")
    set_all_seeds = _require_expn_symbol("set_all_seeds")

    set_all_seeds(int(args.seed))
    # self_train returns: direct_acc_syn, generated_acc, train_acc_by_domain, test_acc_by_domain, last_predictions
    out = self_train(args, source_mlp, chain, epochs=epochs, label_source=label_source)
    return out


# -------------------------
# MNIST domain loading + encoding (reuse your existing pipeline)
# -------------------------
@dataclass
class PreparedMNIST:
    # encoded real domains (in feature space)
    encoded_reals: List[Any]  # [src, (optional real inters...), tgt]
    # for splitting class-conditionally (initial assignments per real domain)
    y_split: List[torch.Tensor]
    # oracle split (true labels) per real domain, for diagnostics/ablations
    y_split_oracle: List[torch.Tensor]
    # number of classes
    K: int
    # EM-label accuracy on target (diagnostic)
    em_acc_target: Optional[float]
    # source model (has .mlp)
    source_model: Any
    # args object that your self_train expects (namespace)
    args: Any


def prepare_mnist(args) -> PreparedMNIST:
    """
    Reuse experiment_new.run_mnist_experiment-like preparation, but stop before running methods.

    Requirements from experiment_new.py:
      - get_single_rotate / get_source_model / encode_all_domains
      - fit_many_em_on_target / build_em_bundle / apply_em_bundle_to_target / get_pseudo_labels
    """
    if EXPN is None:
        raise RuntimeError("experiment_new.py must be importable for MNIST preparation in this script.")

    # Pull needed symbols
    get_single_rotate = _require_expn_symbol("get_single_rotate")
    get_source_model = _require_expn_symbol("get_source_model")
    encode_all_domains = _require_expn_symbol("encode_all_domains")
    fit_many_em_on_target = _require_expn_symbol("fit_many_em_on_target")
    build_em_bundle = _require_expn_symbol("build_em_bundle")
    apply_em_bundle_to_target = _require_expn_symbol("apply_em_bundle_to_target")
    get_pseudo_labels = _require_expn_symbol("get_pseudo_labels")
    set_all_seeds = _require_expn_symbol("set_all_seeds")
    ENCODER = _require_expn_symbol("ENCODER")
    device = _require_expn_symbol("device")

    set_all_seeds(int(args.seed))

    # 1) Build MNIST rotated domains in the same convention as experiment_new.py
    src_trainset = get_single_rotate(False, 0)
    tgt_trainset = get_single_rotate(False, int(args.target))
    all_sets, deg_idx = [], []
    for i in range(1, int(args.gt_domains) + 1):
        angle = i * int(args.target) // (int(args.gt_domains) + 1)
        all_sets.append(get_single_rotate(False, angle))
        deg_idx.append(angle)
    all_sets.append(tgt_trainset)
    deg_idx.append(int(args.target))

    # 2) Build source model with the current experiment_new.py signature
    cache_dir = f"cache{args.ssl_weight}/target{args.target}/small_dim{args.small_dim}/"
    if getattr(args, "source_model_path", None):
        model_path = str(args.source_model_path)
    else:
        model_name = f"src0_tgt{args.target}_ssl{args.ssl_weight}_dim{args.small_dim}.pth"
        model_dir = str(getattr(args, "source_model_dir", "/data/common/yuenchen/GDA/mnist_models/"))
        model_path = os.path.join(model_dir, model_name)
    encoder = ENCODER().to(device)
    source_model = get_source_model(
        args,
        src_trainset,
        tgt_trainset,
        n_class=10,
        mode="mnist",
        encoder=encoder,
        epochs=int(getattr(args, "source_epochs", 10)),
        model_path=model_path,
        target_dataset=tgt_trainset,
        force_recompute=bool(getattr(args, "force_recompute_source", False)),
        compress=True,
        in_dim=25088,
        out_dim=int(args.small_dim),
    )

    # 3) Encode all real domains into feature space
    # encoded_intersets is [src, inter_1, ..., tgt]
    ref_encoder = nn.Sequential(
        source_model.encoder,
        nn.Flatten(start_dim=1),
        getattr(source_model, "compressor", nn.Identity()),
    ).eval()
    e_src, e_tgt, encoded_intersets = encode_all_domains(
        src_trainset,
        tgt_trainset,
        all_sets,
        deg_idx,
        ref_encoder,
        cache_dir=cache_dir,
        target=int(args.target),
        force_recompute=bool(getattr(args, "force_recompute_encode", False)),
        args=args,
    )
    encoded_reals = list(encoded_intersets)

    # 4) Infer K from source labels
    y_src = e_src.targets
    if y_src is None:
        raise RuntimeError("Source encoded dataset must have true labels.")
    K = int(y_src.max().item()) + 1

    # 5) Build per-domain initial class assignment on non-source real domains via EM.
    # Keep this consistent with the current experiment_new.py per-angle flow.
    em_teacher = copy.deepcopy(source_model).to(device).eval()
    raw_domains = [src_trainset] + all_sets
    y_split: List[torch.Tensor] = []
    y_split_oracle: List[torch.Tensor] = []
    for i, ds in enumerate(encoded_reals):
        # Oracle split is always true labels for MNIST diagnostics.
        if getattr(ds, "targets", None) is None:
            raise RuntimeError("Encoded real domain is missing true labels; oracle split unavailable.")
        y_split_oracle.append(ds.targets.detach().clone())

        if i == 0:
            # source: true labels
            y_split.append(ds.targets.detach().clone())
            continue

        raw_ds = raw_domains[i]
        with torch.no_grad():
            pseudo_labels, _ = get_pseudo_labels(
                raw_ds,
                em_teacher,
                confidence_q=float(getattr(args, "pseudo_confidence_q", 0.9)),
                device_override=device,
            )
        args._cached_pseudolabels = pseudo_labels.detach().cpu().numpy()

        em_models = fit_many_em_on_target(
            ds,
            K_list=[K],
            cov_types=list(getattr(args, "em_cov_types", ["diag"])),
            seeds=list(getattr(args, "em_seeds", [0, 1, 2])),
            pool="gap",
            pca_dims=[None],
            reg=1e-4,
            max_iter=300,
            rng_base=int(args.seed),
            args=args,
        )
        em_bundle = build_em_bundle(em_models, args)
        apply_em_bundle_to_target(em_bundle, ds, raw_ds)
        if getattr(ds, "targets_em", None) is None:
            raise RuntimeError("EM labels not attached; expected ds.targets_em after apply_em_bundle_to_target.")
        y_split.append(ds.targets_em.detach().clone())

    em_acc_target = None
    try:
        y_true_t = encoded_reals[-1].targets.detach().cpu().numpy().astype(np.int64)
        y_em_t = y_split[-1].detach().cpu().numpy().astype(np.int64)
        if y_true_t.shape[0] == y_em_t.shape[0] and y_true_t.size > 0:
            em_acc_target = float((y_true_t == y_em_t).mean())
    except Exception:
        em_acc_target = None

    return PreparedMNIST(
        encoded_reals=encoded_reals,
        y_split=y_split,
        y_split_oracle=y_split_oracle,
        K=K,
        em_acc_target=em_acc_target,
        source_model=source_model,
        args=args,
    )


# -------------------------
# Logging (curves + drift)
# -------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_jsonl(path: str, record: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def compute_drift_log(
    steps_with_source: List[EncodedTensorDataset],
    K: int,
    label_for_conditional: str = "real",
) -> Dict[str, Any]:
    """
    steps_with_source includes source at index 0.
    Returns arrays aligned to *edges* between consecutive steps:
      edge t: (step t) -> (step t+1)
    """
    stats = [compute_step_stats(ds, K=K, label_for_conditional=label_for_conditional) for ds in steps_with_source]

    global_list = []
    mean_cond_list = []
    worst_cond_list = []
    per_class_matrix = []  # list of length (#edges), each is length K with float or None
    is_real_next = []
    step_names = []

    for t in range(len(stats) - 1):
        g, m, w, per_class = drift_between(stats[t], stats[t + 1], K=K)
        global_list.append(g)
        mean_cond_list.append(m)
        worst_cond_list.append(w)
        per_class_matrix.append(per_class)
        is_real_next.append(bool(getattr(steps_with_source[t + 1], "is_synthetic", False)) is False)
        step_names.append(getattr(steps_with_source[t + 1], "name", f"step{t+1}"))

    return dict(
        global_drift=global_list,
        mean_cond_drift=mean_cond_list,
        worst_cond_drift=worst_cond_list,
        per_class_drift=per_class_matrix,
        next_is_real=is_real_next,
        next_step_name=step_names,
    )


# -------------------------
# Main runner
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()

    # Core experiment params
    p.add_argument("--dataset", type=str, default="mnist")
    p.add_argument("--target", type=int, default=90)
    p.add_argument("--gt_domains", "--gt-domains", dest="gt_domains", type=int, default=0)
    p.add_argument("--generated_domains", "--generated-domains", dest="generated_domains", type=int, default=3)

    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--methods", type=str, nargs="+", default=["goat", "cc_wass", "cc_nat", "cc_fr"])

    # Training params (keep consistent with your existing scripts)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--rho", type=float, default=0.9)
    p.add_argument("--label_source", type=str, default="pseudo")  # how self_train labels real domains during training
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--ssl_weight", "--ssl-weight", dest="ssl_weight", type=float, default=0.1)
    p.add_argument("--diet", action="store_true")
    p.add_argument("--use_labels", "--use-labels", dest="use_labels", action="store_true")
    p.add_argument("--pseudo_confidence_q", type=float, default=0.9)
    p.add_argument("--source_epochs", type=int, default=10)
    p.add_argument("--force_recompute_source", action="store_true")
    p.add_argument("--force_recompute_encode", action="store_true")
    p.add_argument("--source_model_dir", type=str, default="/data/common/yuenchen/GDA/mnist_models/")
    p.add_argument("--source_model_path", type=str, default=None)

    # Logging
    p.add_argument("--log_root", type=str, default="./logs")
    p.add_argument("--tag", type=str, default="chain_diag")

    # A few knobs your repo expects (safe defaults)
    p.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int, default=256)
    p.add_argument("--num_workers", "--num-workers", dest="num_workers", type=int, default=4)
    p.add_argument("--small_dim", "--small-dim", dest="small_dim", type=int, default=2048)

    # EM / matching knobs (only used if your pipeline reads them)
    p.add_argument("--em_select", type=str, default="bic")
    p.add_argument("--em_match", type=str, default="pseudo")  # cluster-to-class matching heuristic
    p.add_argument("--em_ensemble", action="store_true")
    p.add_argument("--em_cov_types", nargs="+", default=["diag"])
    p.add_argument("--em_seeds", type=int, nargs="+", default=[0, 1, 2])

    return p.parse_args()


def main():
    args = parse_args()

    # Sanity: need experiment_new import for generation + self_train + MNIST prep
    if EXPN is None:
        raise RuntimeError("Could not import experiment_new.py; run this from the same environment/path as your repo.")

    # Run each seed independently (one JSONL line per seed per file)
    for seed in args.seeds:
        args.seed = int(seed)

        # Prepare dataset + encoded real domains + initial labels
        if "mnist" not in args.dataset.lower():
            raise NotImplementedError("This script currently implements the full pipeline for MNIST variants only.")
        prep = prepare_mnist(args)
        encoded_reals = prep.encoded_reals
        y_split = prep.y_split
        y_split_oracle = prep.y_split_oracle
        K = prep.K
        em_acc_target = prep.em_acc_target
        source_model = prep.source_model

        # Log paths
        run_dir = os.path.join(args.log_root, args.dataset, f"s{seed}", f"target{args.target}")
        ensure_dir(run_dir)
        curves_path = os.path.join(
            run_dir,
            f"{args.tag}_gt{args.gt_domains}_gen{args.generated_domains}_curves.jsonl",
        )
        drift_path = os.path.join(
            run_dir,
            f"{args.tag}_gt{args.gt_domains}_gen{args.generated_domains}_drift.jsonl",
        )

        # One record for curves, one record per method for drift (keeps plotting simple)
        setting_name = f"mnist-t{int(args.target)}" if "mnist" in args.dataset.lower() else str(args.dataset)
        curves_record: Dict[str, Any] = dict(
            seed=int(seed),
            dataset=args.dataset,
            target=int(args.target),
            setting=setting_name,
            gt_domains=int(args.gt_domains),
            generated_domains=int(args.generated_domains),
            tag=args.tag,
            em_acc=(float(em_acc_target) if em_acc_target is not None else None),
            elapsed=None,
            methods={},
            methods_legacy=[],
        )

        t0 = time.time()

        for method in args.methods:
            method = method.lower().strip()

            # -----------------------
            # Build training chain
            # -----------------------
            if method == "goat":
                chain = build_chain_goat(
                    generated_domains=int(args.generated_domains),
                    encoded_reals=encoded_reals,
                    name="goat",
                )
                train_label_source = args.label_source  # for real domains
                drift_label_source = "real"            # use true labels for diagnostics (MNIST has them)

            elif method == "cc_wass":
                chain = build_chain_classwise_wass(
                    generated_domains=int(args.generated_domains),
                    encoded_reals=encoded_reals,
                    y_for_split=y_split,
                    K=K,
                    name="cc_wass",
                )
                train_label_source = args.label_source
                drift_label_source = "real"

            elif method in {"cc_wass_oracle", "goat_classwise_oracle"}:
                chain = build_chain_classwise_wass(
                    generated_domains=int(args.generated_domains),
                    encoded_reals=encoded_reals,
                    y_for_split=y_split_oracle,
                    K=K,
                    name="goat_classwise_oracle",
                )
                train_label_source = args.label_source
                drift_label_source = "real"

            elif method == "cc_nat":
                chain = build_chain_classwise_gaussian(
                    generated_domains=int(args.generated_domains),
                    encoded_reals=encoded_reals,
                    y_for_split=y_split,
                    K=K,
                    gen_method="nat",
                    name="cc_nat",
                    cov_type="full",
                    args_for_repo=args,
                )
                train_label_source = args.label_source
                drift_label_source = "real"

            elif method == "cc_fr":
                chain = build_chain_classwise_gaussian(
                    generated_domains=int(args.generated_domains),
                    encoded_reals=encoded_reals,
                    y_for_split=y_split,
                    K=K,
                    gen_method="fr",
                    name="cc_fr",
                    cov_type="full",
                    args_for_repo=args,
                )
                train_label_source = args.label_source
                drift_label_source = "real"

            else:
                raise ValueError(f"Unknown method: {method}")

            # -----------------------
            # Drift log (Figure A)
            # steps = [source] + chain
            # -----------------------
            source_step = _as_encoded_dataset(encoded_reals[0], is_synthetic=False, name="SOURCE")
            steps = [source_step] + chain

            drift_obj = compute_drift_log(
                steps_with_source=steps,
                K=K,
                label_for_conditional=drift_label_source,
            )

            # -----------------------
            # Training (curves)
            # -----------------------
            out = run_self_training_over_chain(
                args=args,
                source_mlp=source_model.mlp,
                chain=chain,
                epochs=int(args.epochs),
                label_source=train_label_source,
            )
            # self_train returns: direct_acc_syn, generated_acc, train_acc_by_domain, test_acc_by_domain, last_predictions
            direct_acc_syn, generated_acc, train_acc_by_domain, test_acc_by_domain, last_predictions = out

            # Canonical method key for downstream scripts/tables.
            method_key = method
            if method == "cc_wass":
                method_key = "goat_classwise"
            elif method in {"cc_wass_oracle", "goat_classwise_oracle"}:
                method_key = "goat_classwise_oracle"

            method_payload = dict(
                test_curve=test_acc_by_domain,
                train_curve=train_acc_by_domain,
                st_curve=[direct_acc_syn] if direct_acc_syn is not None else [],
                st_all_curve=[],
                generated_curve=[generated_acc] if generated_acc is not None else [],
                em_acc=(float(em_acc_target) if em_acc_target is not None else None),
                label_source=train_label_source,
                raw_name=method,
            )
            curves_record["methods"][method_key] = method_payload
            # Backward-compatible payload for existing parsers expecting a list.
            curves_record["methods_legacy"].append(
                dict(
                    name=method,
                    acc_by_domain=test_acc_by_domain,
                    train_acc_by_domain=train_acc_by_domain,
                    generated_acc=generated_acc,
                    em_acc=(float(em_acc_target) if em_acc_target is not None else None),
                )
            )

            # Write drift record per method per seed
            write_jsonl(
                drift_path,
                dict(
                    seed=int(seed),
                    dataset=args.dataset,
                    target=int(args.target),
                    setting=setting_name,
                    gt_domains=int(args.gt_domains),
                    generated_domains=int(args.generated_domains),
                    tag=args.tag,
                    method=method,
                    method_key=method_key,
                    K=int(K),
                    drift=drift_obj,
                ),
            )

        curves_record["elapsed"] = float(time.time() - t0)
        write_jsonl(curves_path, curves_record)

        print(f"[OK] seed={seed} wrote:")
        print(f"  curves: {curves_path}")
        print(f"  drift : {drift_path}")


if __name__ == "__main__":
    main()
