"""
GOAT Experiments — Cleaned Script

This module wires up several domain-adaptation experiments (e.g., Rotated MNIST,
portraits, covtype, color-shifted MNIST). The typical flow is:

1) Train a source model on labeled source data (optionally with SSL on target).
2) Encode source/real intermediate/target domains with the trained encoder.
3) (Optional) Generate synthetic intermediate domains (e.g., via OT interpolation).
4) Self-train the classifier along real+synthetic domains toward the target.
5) Evaluate direct vs pooled self-training, and baselines (e.g., KMeans++).

Dependencies expected from your local package:
- model: ENCODER, MLP, Classifier, MLP_Encoder, VAE (used in color_mnist)
- train_model: self_train, self_train_one_domain, test, get_pseudo_labels
- util / expansion_util / ot_util: generate_domains, generate_domains_find_next, etc.
- dataset: dataset factories (get_single_rotate, EncodeDataset, ColorShiftMNIST, ...),
           get_encoded_dataset, ToTensor, train_vae (used in color_mnist)

Notes:
- This file tries to keep references to your existing helpers intact while removing
  dead code, debug breakpoints, and fixing obvious signature bugs.
- The script avoids duplicate imports, adds docstrings, unifies device handling,
  and makes the contrastive/DIET helpers consistent.
"""
from __future__ import annotations

import os
import csv
import time
import copy
import argparse
import random
from typing import Optional, Tuple, List, Sequence, Iterable, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
# --- Visualization: PCA(2) of standardized target features + EM Gaussians ---
from matplotlib.patches import Ellipse
# Project-local deps (must exist in your repo)
from model import *
from train_model import *
from util import *  # noqa: F401,F403 (kept to preserve your helpers)
from ot_util import ot_ablation, generate_domains  # generation helpers
from a_star_util import *
from a_star_util import _compute_source_gaussians  # explicit import for underscore-prefixed function
from dataset import *

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from da_algo import *
import matplotlib.pyplot as plt
import json
try:
    import kornia.augmentation as K
except Exception:
    K = None  # Kornia is optional; see build_augment()
import math

import numpy as np
# Robust tqdm import with a no-op fallback
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable=None, total=None, **kwargs):
        return iterable if iterable is not None else range(total or 0)

from itertools import product
# -------------------------------------------------------------
# Global config / utilities
# -------------------------------------------------------------

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- em_registry.py (or near your EM helpers) ---
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import torch

# Global in-memory cache; key -> EM bundle
_EM_REGISTRY: Dict[str, "EMBundle"] = {}

from PIL import Image, ImageOps

import torchvision.transforms as T

class ToRGBThenTensor:
    def __call__(self, x):
        # x can be np.ndarray (H,W) / (H,W,1) or torch.Tensor with same shapes
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x = np.asarray(x)

        # squeeze last dim if grayscale stored as (H,W,1)
        if x.ndim == 3 and x.shape[2] == 1:
            x = x[..., 0]

        # normalize to [0,255] uint8 for PIL
        if x.dtype != np.uint8:
            x = x.astype(np.float32)
            if x.max() <= 1.0:
                x = (x * 255.0).round().astype(np.uint8)
            else:
                x = x.round().astype(np.uint8)

        pil = Image.fromarray(x, mode="L").convert("RGB")  # replicate channels
        return T.functional.to_tensor(pil)  # (3,H,W) in [0,1]


# simple palette (R,G,B hex or names)
PALETTE = [
    ("#e6194B", "#000000"),  # (high_color, low_color)
    ("#3cb44b", "#000000"),
    ("#0082c8", "#000000"),
    ("#f58231", "#000000"),
    ("#911eb4", "#000000"),
    ("#46f0f0", "#000000"),
    ("#f032e6", "#000000"),
    ("#d2f53c", "#000000"),
    ("#fabebe", "#000000"),
    ("#008080", "#000000"),
]

def colorize_by_label(img_np, y):
    """img_np: (H,W) or (H,W,1) in [0,255]/uint8 or [0,1]/float; y: int label"""
    if img_np.ndim == 3 and img_np.shape[2] == 1:
        img_np = img_np[..., 0]
    if img_np.dtype != np.uint8:
        x = img_np.astype(np.float32)
        if x.max() <= 1.0:
            x = (x * 255.0).round().astype(np.uint8)
        else:
            x = x.round().astype(np.uint8)
    else:
        x = img_np
    pil_gray = Image.fromarray(x, mode="L")
    hi, lo = PALETTE[y % len(PALETTE)]
    pil_col = ImageOps.colorize(pil_gray, black=lo, white=hi)  # returns RGB PIL Image
    return pil_col

def save_colored_grid(images, labels, n=16, cols=8, out="grid_colored.png"):
    import matplotlib.pyplot as plt
    n = min(n, len(images))
    rows = (n + cols - 1) // cols
    plt.figure(figsize=(cols*1.6, rows*1.6))
    for i in range(n):
        pil_rgb = colorize_by_label(images[i], int(labels[i]))
        ax = plt.subplot(rows, cols, i+1)
        ax.imshow(pil_rgb)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close()

@dataclass
class EMBundle:
    key: str
    em_res: Dict[str, Any]                 # raw run_em_on_encoded(...) result
    mapping: Dict[int, int]                # cluster_id -> class_id mapping
    labels_em: np.ndarray                  # (N,) mapped hard labels on target
    P_soft: Optional[np.ndarray] = None    # (N,K_classes) mapped soft class posteriors (optional)
    info: Dict[str, Any] = None            # any diagnostics (BIC, ll, etc.)

def _em_cache_key(
    dataset: str,
    target_angle: int,
    small_dim: int,
    pool: str,
    do_pca: bool,
    pca_dim: int,
    cov_type: str,
    K: int,
    seed: int,
    dtype: str = "float32",
) -> str:
    # include knobs that affect the fitted model / feature space
    return (f"{dataset}|t={target_angle}|d={small_dim}|pool={pool}"
            f"|pca={int(do_pca)}:{pca_dim}|cov={cov_type}|K={K}"
            f"|seed={seed}|dtype={dtype}")

def ensure_shared_em_for_target(
    e_tgt,                       # encoded target domain (same object used by all methods)
    *,
    args,
    K: Optional[int] = None,     # classes; None -> infer
    cov_type: str = "diag",
    pool: str = "gap",
    do_pca: bool = False,
    pca_dim: int = 64,
    reg: float = 1e-6,
    max_iter: int = 150,
    tol: float = 1e-5,
    n_init: int = 5,
    dtype: str = "float32",
) -> EMBundle:
    """
    Fit EM exactly once on e_tgt (or reuse from cache), map clusters to classes per args.em_match,
    and return a reusable EMBundle.
    """
    # 1) infer K from labels (fallback=10)
    if K is None:
        try:
            y_true = e_tgt.targets.cpu().numpy() if torch.is_tensor(e_tgt.targets) else np.asarray(e_tgt.targets)
            K = int(y_true.max()) + 1
        except Exception:
            K = 10

    # 2) cache key
    dataset = getattr(args, "dataset", "mnist")
    key = _em_cache_key(
        dataset=dataset,
        target_angle=getattr(args, "target_angle", getattr(args, "target", 0)),
        small_dim=getattr(args, "small_dim", 2048),
        pool=pool, do_pca=do_pca, pca_dim=pca_dim,
        cov_type=cov_type, K=K,
        seed=getattr(args, "seed", 0),
        dtype=dtype,
    )
    if key in _EM_REGISTRY:
        bundle = _EM_REGISTRY[key]
        # attach for convenience
        args._shared_em = bundle
        return bundle

    # 3) prepare standardized features ONCE and fit EM ONCE
    X_std, scaler, pca, _ = prepare_em_representation(
        e_tgt, pool=pool, do_pca=do_pca, pca_dim=pca_dim,
        dtype=dtype, rng=getattr(args, "seed", 0), verbose=False
    )
    em_res = run_em_on_encoded_fast(
        X_std,
        K=K, cov_type=cov_type, reg=reg, max_iter=max_iter, tol=tol,
        n_init=n_init, subsample_init=min(len(e_tgt), 20000),
        warm_start=None, rng=getattr(args, "seed", 0), verbose=False
    )
    em_res["scaler"] = scaler
    em_res["pca"] = pca

    # 4) map clusters -> classes (shared for all methods)
    #    Note: we reuse your existing mapping choices.
    method = getattr(args, "em_match", "pseudo")
    if method == "prototypes":
        mu_s, Sigma_s, priors_s = fit_source_gaussian_params(X=e_tgt.data, y=e_tgt.targets)
        mapping, labels_mapped, _ = map_em_clusters(
            em_res, method="prototypes", n_classes=K,
            mus_s=mu_s, Sigma_s=Sigma_s, priors_s=priors_s
        )
    else:
        # Expect args._cached_pseudolabels to be set by caller once (teacher PL)
        if not hasattr(args, "_cached_pseudolabels"):
            raise RuntimeError("args._cached_pseudolabels missing; compute teacher pseudo-labels once, then call ensure_shared_em_for_target.")
        mapping, labels_mapped, _ = map_em_clusters(
            em_res, method="pseudo", n_classes=K,
            pseudo_labels=args._cached_pseudolabels, metric='FR'
        )

    # 5) optional mapped soft posteriors (for downstream soft-label training)
    def _map_cluster_posts_to_classes(gamma: np.ndarray, mapping: Dict[int, int], Kc: int) -> np.ndarray:
        N = gamma.shape[0]
        P = np.zeros((N, Kc), dtype=float)
        for c_from, c_to in mapping.items():
            if 0 <= c_from < gamma.shape[1] and 0 <= c_to < Kc:
                P[:, c_to] += gamma[:, c_from]
        s = P.sum(axis=1, keepdims=True); s[s == 0] = 1.0
        return P / s
    P_soft = _map_cluster_posts_to_classes(np.asarray(em_res["gamma"]), mapping, K)

    bundle = EMBundle(
        key=key, em_res=em_res, mapping=mapping,
        labels_em=np.asarray(labels_mapped, dtype=int),
        P_soft=P_soft, info={"K": K, "cov_type": cov_type, "pool": pool, "pca_dim": pca_dim, "dtype": dtype},
    )
    _EM_REGISTRY[key] = bundle
    # attach to args so children don’t have to compute the key
    args._shared_em = bundle
    return bundle

def _has_full(domain_stats: dict) -> bool:
    return ("Sigma" in domain_stats) and (domain_stats["Sigma"] is not None)

def _steps(domain_stats: dict) -> np.ndarray:
    return np.asarray(domain_stats["steps"], dtype=float)

def _present_mask(domain_stats: dict, step: int) -> np.ndarray:
    # classes considered present at this step (count>0)
    return np.asarray(domain_stats["counts"][step]) > 0

def _get_mu(domain_stats: dict, step: int) -> np.ndarray:
    return np.asarray(domain_stats["mu"][step], dtype=float)  # (K,d)

def _get_var(domain_stats: dict, step: int) -> np.ndarray:
    # (K,d) diag variances; safe even if Sigma exists (for plots that want diag view)
    return np.asarray(domain_stats["var"][step], dtype=float)

def _get_sigma(domain_stats: dict, step: int) -> np.ndarray:
    """Return (K,d,d) covariance for this step.
       If only 'var' exists, expand to diag."""
    if _has_full(domain_stats):
        return np.asarray(domain_stats["Sigma"][step], dtype=float)  # (K,d,d)
    # expand diag:
    var = _get_var(domain_stats, step)
    K, d = var.shape
    Sig = np.zeros((K, d, d), dtype=float)
    for k in range(K):
        np.fill_diagonal(Sig[k], np.clip(var[k], 0.0, None))
    return Sig

def _trace_cov(Sig: np.ndarray) -> np.ndarray:
    # Sig: (K,d,d)
    return np.trace(Sig, axis1=-2, axis2=-1)  # (K,)

def _logdet_cov(Sig: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Returns logdet per item.
    Accepts:
      - (K,d,d): returns (K,)
      - (d,d)  : returns scalar array shape (1,)
      - (d,)   : treats as diag variances -> returns scalar array shape (1,)
    """
    Sig = np.asarray(Sig)
    if Sig.ndim == 1:                 # (d,)
        w = np.clip(Sig, eps, None)   # diagonal variances
        return np.array([np.sum(np.log(w))], dtype=float)
    if Sig.ndim == 2:                 # (d,d)
        w = np.linalg.eigvalsh(Sig)
        w = np.clip(w, eps, None)
        return np.array([np.sum(np.log(w))], dtype=float)
    if Sig.ndim == 3:                 # (K,d,d)
        K = Sig.shape[0]
        out = np.zeros(K, dtype=float)
        for i in range(K):
            A = Sig[i]
            if A.ndim == 1:           # safety: diag vector slipped in
                w = np.clip(A, eps, None)
                out[i] = np.sum(np.log(w))
            else:
                w = np.linalg.eigvalsh(A)
                w = np.clip(w, eps, None)
                out[i] = np.sum(np.log(w))
        return out
    raise ValueError(f"_logdet_cov: unsupported shape {Sig.shape}")


def _to_2d_numpy(x: torch.Tensor, pool: str = "flatten") -> np.ndarray:
    t = torch.as_tensor(x)
    if t.ndim > 2:
        if pool == "gap":
            reduce_dims = tuple(range(2, t.ndim))
            t = t.mean(dim=reduce_dims)
        else:
            t = t.view(t.size(0), -1)
    return t.detach().cpu().numpy()


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


# -------------------------------------------------------------
# Logging & visualization
# -------------------------------------------------------------

from torch.utils.data import DataLoader
# ---- helper: init Gaussian head from source features ----
import torch
import torch.nn.functional as F

def _save_list(path, obj):
    """
    Save arbitrary Python objects (lists/tuples/dicts/numpy/tensors/scalars) to JSON.

    This now handles "everything returned by each method" by recursively converting
    non-JSON-native types (e.g., numpy arrays, torch tensors, numpy scalars) into
    JSON-serializable structures. Tuples are stored as lists; dict keys are cast to
    strings. Unknown objects fall back to their string representation.
    """
    import numpy as _np
    import torch as _torch
    import os as _os

    def _to_jsonable(x):
        # numpy types
        if isinstance(x, _np.ndarray):
            return x.tolist()
        if isinstance(x, (_np.integer, _np.floating)):
            return x.item()

        # torch tensors
        if isinstance(x, _torch.Tensor):
            return x.detach().cpu().tolist()

        # basic containers
        if isinstance(x, dict):
            return {str(k): _to_jsonable(v) for k, v in x.items()}
        if isinstance(x, (list, tuple, set)):
            return [_to_jsonable(v) for v in x]

        # builtins (ints, floats, bools, strings, None)
        if isinstance(x, (int, float, bool, str)) or x is None:
            return x

        # attempt to use ._asdict() for namedtuples and similar
        if hasattr(x, "_asdict") and callable(getattr(x, "_asdict")):
            try:
                return _to_jsonable(x._asdict())
            except Exception:
                pass

        # final fallback: use string repr
        try:
            return str(x)
        except Exception:
            return "<unserializable>"

    try:
        # ensure directory exists
        _os.makedirs(_os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(_to_jsonable(obj), f)
        # print(f"[MainAlgo] Saved {path}")
    except Exception as e:
        print(f"[MainAlgo] Failed to save {path}: {e}")


def _save_dict(path, obj: dict):
    """
    Save a dict to JSON via _save_list.
    """
    _save_list(path, obj)


def check_logdet_identity(Sig_s, Sig_t, Sig_t_list, ts, eps=1e-12):

    def slogdet(S):
        S = 0.5*(S+S.T)
        sgn, ld = np.linalg.slogdet(S + eps*np.eye(S.shape[0]))
        return float(ld)

    ld_s = slogdet(Sig_s)
    ld_T = slogdet(Sig_t)
    for t, Sig in zip(ts, Sig_t_list):
        lhs = slogdet(Sig)
        rhs = (1.0 - t)*ld_s + t*ld_T
        print(f"t={t:.2f}: logdet(Σ(t))={lhs:.4f}  vs  linear blend={rhs:.4f}  (Δ={lhs-rhs:.4e})")

def check_fr_logvar_linearity(domain_stats: dict, target_class: int = None):
    """
    Diag case: checks linearity of the *mean log-variance* per class across t.
    Full case: checks linearity of logdet(Σ) per class across t.

    Prints the mean absolute deviation from the best linear fit (MAE)
    and the R^2 of that fit. If target_class is given, reports only that class.
    """
    ts = np.asarray(_steps(domain_stats), dtype=float)
    S, K = len(ts), int(domain_stats["K"])
    is_full = _has_full(domain_stats)

    def _scalar_logdet(sig, eps=1e-12):
        sig = np.asarray(sig)
        if sig.ndim == 2:            # full (d,d)
            # robust slogdet
            sgn, ld = np.linalg.slogdet(0.5*(sig+sig.T) + eps*np.eye(sig.shape[0]))
            return float(ld)
        elif sig.ndim == 1:          # diag (d,)
            v = np.clip(sig, eps, None)
            return float(np.sum(np.log(v)))
        else:
            raise ValueError(f"_scalar_logdet: unexpected shape {sig.shape}")

    def _fit_err(x, y):
        # returns MAE and R^2 of linear fit y ~ a*x + b
        coef = np.polyfit(x, y, 1)
        yhat = np.polyval(coef, x)
        mae  = float(np.mean(np.abs(y - yhat)))
        ss_res = float(np.sum((y - yhat)**2))
        ss_tot = float(np.sum((y - np.mean(y))**2)) if len(y) > 1 else 0.0
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        return mae, r2, coef

    ks = [target_class] if target_class is not None else list(range(K))
    errs, r2s = [], []

    if not is_full:
        # ---- DIAG: use mean log-variance per class ----
        for k in ks:
            y = []
            x = []
            for s in range(S):
                if not _present_mask(domain_stats, s)[k]:
                    continue
                v = _get_var(domain_stats, s)[k]                 # (d,)
                y.append(float(np.mean(np.log(np.clip(v, 1e-12, None)))))
                x.append(ts[s])
            if len(x) >= 2:
                x = np.asarray(x); y = np.asarray(y)
                mae, r2, coef = _fit_err(x, y)
                errs.append(mae); r2s.append(r2)
                if target_class is not None:
                    print(f"[FR][diag] class {k}: MAE={mae:.3e}, R^2={r2:.4f}, slope={coef[0]:.3e}")
        if target_class is None:
            print(f"[FR][check-diag] mean |Δ logvar| = {np.nanmean(errs):.3e}, mean R^2={np.nanmean(r2s):.4f}")
    else:
        # ---- FULL: use logdet(Σ) per class ----
        for k in ks:
            y = []
            x = []
            for s in range(S):
                if not _present_mask(domain_stats, s)[k]:
                    continue
                Sig_k = _get_sigma(domain_stats, s)[k]           # (d,d) or (d,)
                y.append(_scalar_logdet(Sig_k))
                x.append(ts[s])
            if len(x) >= 2:
                x = np.asarray(x); y = np.asarray(y)
                mae, r2, coef = _fit_err(x, y)
                errs.append(mae); r2s.append(r2)
                if target_class is not None:
                    print(f"[FR][full] class {k}: MAE={mae:.3e}, R^2={r2:.4f}, slope={coef[0]:.3e}")
        if target_class is None:
            print(f"[FR][check-full] mean |Δ logdet(Σ)| = {np.nanmean(errs):.3e}, mean R^2={np.nanmean(r2s):.4f}")

def _unpack_domain_stats(stats):
    """
    Returns: steps (S,), mu (S,K,d), var (S,K,d), counts (S,K)
    Works for both the new dict schema and the old list-of-tuples.
    """


    if isinstance(stats, dict):
        steps  = np.asarray(stats["steps"], dtype=float)              # (S,)
        mu     = np.asarray(stats["mu"],    dtype=float)              # (S,K,d)
        var    = np.asarray(stats["var"],   dtype=float)              # (S,K,d)
        counts = np.asarray(stats["counts"], dtype=np.int64)          # (S,K)
        return steps, mu, var, counts

    # legacy: [(mu_step, var_step, counts_step), ...]
    S = len(stats)
    mu     = np.stack([np.asarray(x[0], dtype=float) for x in stats], axis=0)
    var    = np.stack([np.asarray(x[1], dtype=float) for x in stats], axis=0)
    counts = np.stack([np.asarray(x[2], dtype=np.int64) for x in stats], axis=0)
    steps  = np.linspace(0.0, 1.0, num=S, dtype=float)
    return steps, mu, var, counts



def plot_pca_em_pair_side_by_side(
    X_std,
    y_left,
    em_mu,
    em_Sigma,
    mapping_ps,
    right_labels,
    right_title: str,
    save_path: str,
):
    """
    Two-panel PCA(2) visualization with EM Gaussian ellipses overlaid:
    - Left: points colored by y_left (ground-truth) + EM cluster ellipses colored by mapped class.
    - Right: points colored by right_labels (e.g., EM-mapped labels or pseudo labels) + same ellipses.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    from sklearn.decomposition import PCA

    import os

    if X_std is None or not hasattr(X_std, "ndim") or X_std.ndim != 2 or X_std.shape[0] == 0:
        return

    y_left = np.asarray(y_left)
    right_labels = np.asarray(right_labels)

    pca = PCA(n_components=2)
    Z = pca.fit_transform(X_std)
    C2 = pca.components_[:2, :]
    m2 = pca.mean_
    cmap = plt.get_cmap('tab10')

    def _get_mu(cid):
        if isinstance(em_mu, dict):
            return np.asarray(em_mu[cid]).reshape(-1)
        arr = np.asarray(em_mu)
        return np.asarray(arr[cid]).reshape(-1)

    def _get_sigma_full(cid):
        if em_Sigma is None:
            return np.eye(C2.shape[1], dtype=float)
        if isinstance(em_Sigma, dict):
            S = np.asarray(em_Sigma[cid])
        else:
            S = np.asarray(em_Sigma)[cid]
        return np.diag(S) if S.ndim == 1 else S

    def _draw_gaussian(ax, mean_d, cov_d, color, n_std=2.0, lw=2.0):
        cov_d = 0.5 * (cov_d + cov_d.T)
        w, v = np.linalg.eigh(cov_d)
        w = np.clip(w, 1e-12, None)
        width, height = 2.0 * n_std * np.sqrt(w)
        angle = np.degrees(np.arctan2(v[1, 0], v[0, 0]))
        e = Ellipse(xy=mean_d, width=width, height=height, angle=angle,
                    edgecolor=color, facecolor='none', lw=lw)
        ax.add_patch(e)

    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    ax0, ax1 = axs[0], axs[1]

    # Left panel: GT coloring
    if y_left.size:
        n_left = int(y_left.max()) + 1
        for c in range(n_left):
            m = (y_left == c)
            if m.any():
                ax0.scatter(Z[m, 0], Z[m, 1], s=6, alpha=0.6, color=cmap(c % 10), label=f"class {c}")

    # EM Gaussians colored by mapped class on both panels
    cluster_ids = range(len(em_mu)) if not isinstance(em_mu, dict) else sorted(em_mu.keys())
    for cid in cluster_ids:
        mu_k = _get_mu(cid)
        Sig_k = _get_sigma_full(cid)
        mu2d = (C2 @ (mu_k - m2)).reshape(-1)
        Sig2d = C2 @ Sig_k @ C2.T
        cls = mapping_ps.get(int(cid), 0) if isinstance(mapping_ps, dict) else 0
        color = cmap(int(cls) % 10)
        _draw_gaussian(ax0, mu2d, Sig2d, color)
        _draw_gaussian(ax1, mu2d, Sig2d, color)

    # Right panel: provided labels coloring
    if right_labels.size:
        n_pred = int(right_labels.max()) + 1
        for c in range(n_pred):
            m = (right_labels == c)
            if m.any():
                ax1.scatter(Z[m, 0], Z[m, 1], s=6, alpha=0.6, color=cmap(c % 10), label=f"label {c}")

    ax0.set_title("GT labels")
    ax1.set_title(right_title)
    for a in axs:
        a.set_xlabel("PC 1")
    ax0.set_ylabel("PC 2")
    for a in axs:
        h, l = a.get_legend_handles_labels()
        by = dict(zip(l, h))
        if by:
            a.legend(by.values(), by.keys(), loc='best', fontsize=8)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[MNIST-EXP] Saved {save_path}")
    print(f"[MainAlgo] Saved plot to {save_path}")



def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def stats_to_numpy(mus, sigmas):
    # mus, sigmas can be tensors or dicts {k: (d,)} / {k: (d,)}
    if isinstance(mus, dict):
        # sort by class id to produce (K,d)
        keys = sorted(mus.keys())
        mu_np = np.stack([to_numpy(mus[k]) for k in keys], axis=0)       # (K,d)
        sg_np = np.stack([to_numpy(sigmas[k]) for k in keys], axis=0)    # (K,d) diag
        return mu_np, sg_np
    else:
        return to_numpy(mus), to_numpy(sigmas)



@torch.no_grad()
def init_head_from_source(model, dataset, batch_size=256, num_workers=2, device=None):
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    # Locate the Gaussian head robustly
    head = getattr(model, "classifier", getattr(model, "head", None))
    assert head is not None, "Gaussian head not found (expected .classifier or .head)."

    Z_list, Y_list = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        # IMPORTANT: use the SAME embedding pipeline the head sees
        h = model.encoder(x)           # fmap or vector (already includes compressor if you wrapped it)
        z = model.mlp(h)               # → N × emb_dim
        if getattr(model, "normalize", False):
            z = F.normalize(z, dim=1, eps=1e-6)
        Z_list.append(z.cpu())
        Y_list.append(y.cpu())

    Z = torch.cat(Z_list, 0)           # M × d
    Y = torch.cat(Y_list, 0)           # M
    K = head.mu.shape[0]
    d = Z.shape[1]
    assert d == head.mu.shape[1], f"Embedding dim {d} != head.mu dim {head.mu.shape[1]}"

    means = torch.zeros(K, d)
    eps = 1e-6

    if getattr(head, "cov", "diagonal") == "diagonal":
        logvar = torch.zeros(K, d)
    else:  # isotropic
        logvar = torch.zeros(K)

    for k in range(K):
        Zk = Z[Y == k]                 # (#k) × d
        if Zk.numel() == 0:
            # fallback: mean=0, var=1 (logvar=0)
            continue
        means[k] = Zk.mean(0)
        if getattr(head, "cov", "diagonal") == "diagonal":
            vk = Zk.var(0, unbiased=False).clamp_min(eps)   # d
            logvar[k] = vk.log()
        else:  # isotropic: use average variance across dims
            vk = Zk.var(0, unbiased=False).mean().clamp_min(eps)
            logvar[k] = vk.log()

    # Laplace-smoothed class priors
    counts = torch.tensor([(Y == k).sum().item() for k in range(K)], dtype=torch.float)
    probs  = (counts + 1.0) / (counts.sum() + K)
    log_pi = probs.log()

    # Copy into the model (don’t rebind .data tensors)
    head.mu.data.copy_(means.to(head.mu.device, dtype=head.mu.dtype))
    head.log_pi.data.copy_(log_pi.to(head.log_pi.device, dtype=head.log_pi.dtype))
    head.log_var.data.copy_(logvar.to(head.log_var.device, dtype=head.log_var.dtype))


## Note: Previously this file forwarded CLI to experiments.py via an early __main__ block.
## That made it impossible to use expanded mnist modes here. The forwarder has been removed
## so the unified main() at the bottom handles CLI.


def init_tensorboard(log_dir: str = "logs/tensorboard") -> SummaryWriter:
    os.makedirs(log_dir, exist_ok=True)
    return SummaryWriter(log_dir=log_dir)


def log_progress(
    log_file: str,
    step: int,
    step_type: str,
    domain_idx: int,
    dataset_name: str,
    acc1: Optional[float] = None,
    acc2: Optional[float] = None,
    acc3: Optional[float] = None,
    target_acc: Optional[float] = None,
) -> None:
    """Append a CSV row with metrics and a timestamp."""
    header = [
        "Step",
        "Type",
        "Domain_Index",
        "Dataset",
        "Direct_Acc",
        "ST_Acc",
        "Generated_Acc",
        "Target_Acc",
        "Timestamp",
    ]
    is_new = not os.path.exists(log_file)
    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(header)
        writer.writerow(
            [
                step,
                step_type,
                domain_idx,
                dataset_name,
                round(acc1, 4) if acc1 is not None else "",
                round(acc2, 4) if acc2 is not None else "",
                round(acc3, 4) if acc3 is not None else "",
                round(target_acc, 4) if target_acc is not None else "",
                time.time(),
            ]
        )


def plot_encoded_domains(
    encoded_source,
    encoded_inter,
    encoded_target,
    title_src: str = "Encoded Source",
    title_inter: str = "Encoded Inter",
    title_tgt: str = "Encoded Target",
    method: str = "goat",
    save_dir: str = "plots",
    pca: Optional[PCA] = None,
) -> PCA:
    """Project three encoded datasets into PCA(2) and save a 1x3 scatter figure.

    PCA is fit on source+target only (to avoid leaking info from the synthetic
    in-between set), unless a PCA instance is provided.
    """
    os.makedirs(save_dir, exist_ok=True)

    def _to_tensor_2d(d):
        if hasattr(d, "data"):
            x = d.data
        elif hasattr(d, "tensors") and len(d.tensors) > 0:
            x = d.tensors[0]
        elif isinstance(d, (tuple, list)) and len(d) > 0:
            x = d[0]
        else:
            x = d
        x = torch.as_tensor(x)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        x = x.view(x.size(0), -1)  # flatten
        return x

    def _get_targets(d, n_samples):
        if hasattr(d, "targets_em") and d.targets_em is not None:
            t = d.targets_em
        elif hasattr(d, "targets") and d.targets is not None:
            t = d.targets
        elif hasattr(d, "tensors") and len(d.tensors) > 1:
            t = d.tensors[1]
        else:
            t = None
        if t is None:
            return np.zeros(n_samples, dtype=int)
        if torch.is_tensor(t):
            return t.cpu().numpy()
        return np.asarray(t)

    src = _to_tensor_2d(encoded_source)
    inter = _to_tensor_2d(encoded_inter)
    tgt = _to_tensor_2d(encoded_target)

    fit_data = torch.cat([src, tgt], dim=0)
    all_data = torch.cat([src, inter, tgt], dim=0)

    if pca is None:
        pca = PCA(n_components=2)
        pca.fit(fit_data.cpu().numpy())

    z_all = pca.transform(all_data.cpu().numpy())
    n_src, n_inter = len(src), len(inter)
    z_src = z_all[:n_src]
    z_inter = z_all[n_src : n_src + n_inter]
    z_tgt = z_all[n_src + n_inter :]

    y_src = _get_targets(encoded_source, len(src))
    y_inter = _get_targets(encoded_inter, len(inter))
    y_tgt = _get_targets(encoded_target, len(tgt))

    fig, axs = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)

    for c in np.unique(y_src):
        axs[0].scatter(z_src[y_src == c, 0], z_src[y_src == c, 1], label=f"Class {c}", alpha=0.6, s=10)
    axs[0].set_title(title_src)
    axs[0].set_xlabel("PC 1")
    axs[0].set_ylabel("PC 2")
    axs[0].legend()
    axs[0].grid(True)

    for c in np.unique(y_inter):
        axs[1].scatter(z_inter[y_inter == c, 0], z_inter[y_inter == c, 1], label=f"Class {c}", alpha=0.6, s=10)
    axs[1].set_title(title_inter)
    axs[1].set_xlabel("PC 1")
    axs[1].legend()
    axs[1].grid(True)

    for c in np.unique(y_tgt):
        axs[2].scatter(z_tgt[y_tgt == c, 0], z_tgt[y_tgt == c, 1], label=f"Class {c}", alpha=0.6, s=10)
    axs[2].set_title(title_tgt)
    axs[2].set_xlabel("PC 1")
    axs[2].legend()
    axs[2].grid(True)

    plt.suptitle("Encoded Source vs Target Projections")
    plt.tight_layout()
    out_path = os.path.join(save_dir, f"encoded_domains_{method}.png")
    plt.savefig(out_path)
    plt.close()
    return pca


# -------------------------------------------------------------
# SSL / DIET utilities
# -------------------------------------------------------------


def nt_xent_loss(z_i: torch.Tensor, z_j: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Normalized temperature-scaled cross-entropy loss (SimCLR-style)."""
    z = torch.cat([z_i, z_j], dim=0)  # (2N, d)
    z = F.normalize(z + 1e-6, dim=1)
    sim = z @ z.T
    N = z_i.size(0)

    labels = torch.arange(N, device=z.device)
    labels = torch.cat([labels, labels], dim=0)

    mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, -9e15)
    sim = sim / temperature
    sim = torch.clamp(sim, min=-100, max=100)
    return F.cross_entropy(sim, labels)


def build_augment(image_size: Tuple[int, int] = (28, 28)) -> nn.Module:
    """Return a default augmentation pipeline.

    If Kornia isn't available, falls back to Identity.
    """
    if K is None:
        return nn.Identity()
    return nn.Sequential(
        K.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        K.RandomHorizontalFlip(),
        K.RandomAffine(degrees=10, translate=(0.1, 0.1)),
    ).to(device)


class DatasetWithIndices(Dataset):
    """Wrap a dataset to return (x, idx) for DIET-style self-labeling."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __getitem__(self, index: int):
        x, _ = self.dataset[index]
        return x, index

    def __len__(self) -> int:
        return len(self.dataset)


def extract_features(encoder: nn.Module, dataset: Dataset, batch_size: int = 128) -> Tuple[np.ndarray, np.ndarray]:
    encoder.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    feats, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            z = encoder(x).view(x.size(0), -1)
            feats.append(z.cpu())
            labels.append(y)
    X = torch.cat(feats).numpy()
    y = torch.cat(labels).numpy()
    return X, y


def evaluate_linear_probe(encoder: nn.Module, trainset: Dataset, testset: Dataset, batch_size: int = 128) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    X_tr, y_tr = extract_features(encoder, trainset, batch_size)
    X_te, y_te = extract_features(encoder, testset, batch_size)

    clf = LogisticRegression(max_iter=1000, solver="lbfgs", multi_class="multinomial")
    clf.fit(X_tr, y_tr)
    y_hat = clf.predict(X_te)
    return accuracy_score(y_te, y_hat)


def train_encoder_diet(
    model: Classifier,
    trainset: Dataset,
    testset: Optional[Dataset],
    epochs: int = 1000,
    batch_size: int = 128,
    lr: float = 1e-3,
    label_smoothing: float = 0.8,
    weight_decay: float = 1e-5,
    eval_interval: int = 20,
) -> nn.Module:
    """DIET: learn encoder by classifying instance indices with label smoothing.

    Returns the trained encoder (the model's projection head is not touched here).
    """
    print("[DIET] Self-supervised training on (wrapped) target domain")
    encoder = model.encoder.to(device)
    encoder.train()

    # Determine encoder output dimensionality once
    with torch.no_grad():
        dummy_in = torch.randn(1, *trainset[0][0].shape).to(device)
        flat_dim = encoder(dummy_in).view(1, -1).shape[1]

    num_classes = len(trainset)
    W = nn.Linear(flat_dim, num_classes, bias=False).to(device)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    opt = optim.Adam(list(encoder.parameters()) + list(W.parameters()), lr=lr, weight_decay=weight_decay)

    loader = DataLoader(DatasetWithIndices(trainset), batch_size=batch_size, shuffle=True)

    for epoch in range(1, epochs + 1):
        encoder.train()
        W.train()
        total = 0.0
        for x, idx in loader:
            x, idx = x.to(device), idx.to(device)
            z = encoder(x).view(x.size(0), -1)
            logits = W(z)
            loss = loss_fn(logits, idx)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"[DIET] Epoch {epoch}/{epochs}: Loss = {total / len(loader):.4f}")

        if testset is not None and (epoch % eval_interval == 0):
            acc = evaluate_linear_probe(encoder, trainset, testset, batch_size)
            print(f"[DIET] Eval @epoch {epoch}: Linear Probe Acc = {acc * 100:.2f}%")

    return encoder


def train_joint(
    model: Classifier,
    trainloader: DataLoader,
    tgt_loader: Optional[DataLoader],
    optimizer: optim.Optimizer,
    augment_fn: Optional[nn.Module] = None,
    ssl_weight: float = 0.1,
) -> Tuple[float, Optional[float]]:
    """One epoch of supervised CE on source (+ optional contrastive SSL on target)."""
    model.train()
    total_sup, total_ssl = 0.0, 0.0

    tgt_iter = iter(tgt_loader) if tgt_loader is not None else None

    for batch in trainloader:
        if len(batch) == 2:
            x, y = batch
            w = None
        else:
            x, y, w = batch
            w = w.to(device)

        x, y = x.to(device), y.to(device)
        out = model(x)

        if w is None:
            sup_loss = F.cross_entropy(out, y)
        else:
            ce = nn.CrossEntropyLoss(reduction="none")
            sup_loss = (ce(out, y) * w).mean()

        ssl_loss = 0.0
        if tgt_iter is not None and augment_fn is not None:
            try:
                x_tgt, _ = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader)
                x_tgt, _ = next(tgt_iter)
            x_tgt = x_tgt.to(device)
            x1, x2 = augment_fn(x_tgt), augment_fn(x_tgt)
            z1 = F.normalize(model.encoder(x1).view(x1.size(0), -1) + 1e-6, dim=1)
            z2 = F.normalize(model.encoder(x2).view(x2.size(0), -1) + 1e-6, dim=1)
            ssl_loss = nt_xent_loss(z1, z2)

        loss = sup_loss + ssl_weight * ssl_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_sup += float(sup_loss)
        total_ssl += float(ssl_loss)

    avg_sup = total_sup / len(trainloader)
    avg_ssl = (total_ssl / len(trainloader)) if tgt_iter is not None else None
    return avg_sup, avg_ssl




# ---------- small helpers ----------

def _gap_flat(x: torch.Tensor) -> torch.Tensor:
    """Global average pool any N×C×H×W to N×C; pass-through if already N×C."""
    if x.dim() == 4:
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
    return x


def _ensure_centers_buffers(model: nn.Module, z: torch.Tensor, y: torch.Tensor, n_classes: int = None):
    device = z.device
    d = z.view(z.size(0), -1).size(1)
    if n_classes is None:
        n_classes = int(y.max().item()) + 1 if y.numel() > 0 else 0
    if not hasattr(model, 'centers') or not hasattr(model, 'counts'):
        model.register_buffer('centers', torch.zeros(n_classes, d, device=device))
        model.register_buffer('counts',  torch.zeros(n_classes, device=device))
    else:
        K, d_old = model.centers.shape
        if K != n_classes or d_old != d or model.centers.device != device:
            model.centers = torch.zeros(n_classes, d, device=device)
            model.counts  = torch.zeros(n_classes, device=device)

@torch.no_grad()
def _update_class_means_ema(model, z, y, n_classes=None, momentum: float = 0.9):
    _ensure_centers_buffers(model, z, y, n_classes)
    K = model.centers.size(0)
    z2d = z.view(z.size(0), -1)
    for k in range(K):
        mask = (y == k)
        if mask.any():
            zk = z2d[mask].mean(0)
            model.centers[k] = momentum * model.centers[k] + (1 - momentum) * zk
            model.counts[k]  = model.counts[k] + mask.sum()

def _center_loss(model, z, y, weight=1e-3):
    _ensure_centers_buffers(model, z, y)
    z2d = z.view(z.size(0), -1)
    centers = model.centers.detach()  # (K,d)
    loss = z2d.new_tensor(0.0)
    for k in range(centers.size(0)):
        mask = (y == k)
        if mask.any():
            diff = z2d[mask] - centers[k]
            loss = loss + (diff.pow(2).sum(dim=1).mean())
    return weight * loss

def _coral_loss(zs: torch.Tensor, zt: torch.Tensor, weight: float = 1e-3):
    """CORAL: align covariances of source and target embeddings (after GAP)."""
    if zs.numel() == 0 or zt.numel() == 0:
        return torch.tensor(0.0, device=zs.device)
    zs = zs - zs.mean(dim=0, keepdim=True)
    zt = zt - zt.mean(dim=0, keepdim=True)
    cs = (zs.t() @ zs) / (zs.size(0) - 1 + 1e-6)
    ct = (zt.t() @ zt) / (zt.size(0) - 1 + 1e-6)
    return weight * (cs - ct).pow(2).mean()

def _nt_xent(q1: torch.Tensor, q2: torch.Tensor, temp: float = 0.2) -> torch.Tensor:
    """SimCLR InfoNCE on normalized projections (batch-wise)."""
    q1 = F.normalize(q1, dim=1, eps=1e-6)
    q2 = F.normalize(q2, dim=1, eps=1e-6)
    N = q1.size(0)
    reps = torch.cat([q1, q2], dim=0)             # 2N × d
    sim = reps @ reps.t() / temp                  # 2N × 2N
    mask = torch.eye(2 * N, device=reps.device, dtype=torch.bool)
    sim.masked_fill_(mask, -1e9)

    # positives are (i, i+N) and (i+N, i)
    pos = torch.arange(N, device=reps.device)
    logits = sim
    labels = torch.cat([pos + N, pos], dim=0)
    return F.cross_entropy(logits, labels)

def _get_features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Robustly get a pooled feature z (N×C) from different model types."""
    if hasattr(model, "backbone"):       # your new Classifier
        z = model.backbone(x)            # already GAP+Flatten (N×C)
    elif hasattr(model, "encoder"):      # older models
        f = model.encoder(x)             # could be fmap or vector
        z = _gap_flat(f)
    else:
        # fallback: try the full model then strip classifier
        f = model(x)
        z = _gap_flat(f)
    return z

# ---------- upgraded training loop ----------

def train_joint_plus(
    model: nn.Module,
    trainloader,
    tgt_loader,
    optimizer,
    augment_fn=None,
    ssl_weight: float = 0.1,
    coral_weight: float = 1e-3,
    center_weight: float = 1e-3,
    temp: float = 0.2,
    warmup_epochs: int = 2,
    epoch_idx: int = 1,
    device: torch.device = torch.device("cuda"),
    args: argparse.Namespace = None,
) -> Tuple[float, Optional[float]]:

    model.train()
    total_sup, total_ssl = 0.0, 0.0
    tgt_iter = iter(tgt_loader) if tgt_loader is not None else None

    for batch in trainloader:
        if len(batch) == 2:  x, y = batch; w = None
        else:                x, y, w = batch; w = w.to(device)
        x, y = x.to(device), y.to(device)

        # CE on logits from full model

        logits = model(x)
        if w is None:
            sup_loss = F.cross_entropy(logits, y)
        else:
            ce = F.cross_entropy(logits, y, reduction="none")
            sup_loss = (ce * w).mean()

        # Center loss on features (z)
        z_src = model.encoder(x)
        if 'compressor' in dir(model) and model.compressor is not None:
            z_src = z_src.view(z_src.size(0), -1)
            z_src = model.compressor(z_src)
        else:
            z_src = _gap_flat(z_src)
        _update_class_means_ema(model, z_src.detach(), y)     # uses model.centers/counts
        cen_loss = _center_loss(model, z_src, y, weight=center_weight)
 
        # SSL & CORAL (if target present and past warmup)
        ssl_loss = torch.tensor(0.0, device=device)
        coral    = torch.tensor(0.0, device=device)
        if tgt_iter is not None and augment_fn is not None and (epoch_idx > warmup_epochs):
            try:
                x_tgt, _ = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader)
                x_tgt, _ = next(tgt_iter)
            x_tgt = x_tgt.to(device)
            x1, x2 = augment_fn(x_tgt), augment_fn(x_tgt)

            z1, z2 = model.encoder(x1), model.encoder(x2)
            if 'compressor' in dir(model) and model.compressor is not None:
                z1 = z1.view(z1.size(0), -1)
                z2 = z2.view(z2.size(0), -1)
                z1 = model.compressor(z1)
                z2 = model.compressor(z2)
            else:
                z1 = _gap_flat(z1)
                z2 = _gap_flat(z2)
            ssl_loss = _nt_xent(z1, z2, temp=temp)
            coral    = _coral_loss(z_src.detach(), z1.detach(), weight=coral_weight)

        loss = sup_loss + cen_loss + ssl_weight * ssl_loss + coral
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_sup += float(sup_loss + cen_loss)
        total_ssl += float(ssl_loss + coral)

    avg_sup = total_sup / len(trainloader)
    avg_ssl = (total_ssl / len(trainloader)) if tgt_iter is not None else None
    return avg_sup, avg_ssl

# -------------------------------------------------------------
# Source model training (with caching)
# -------------------------------------------------------------

def get_source_model(
    args: argparse.Namespace,
    trainset: Dataset,
    testset: Dataset,
    n_class: int,
    mode: str,
    encoder: Optional[nn.Module] = None,
    epochs: int = 50,
    verbose: bool = True,
    model_path: Optional[str] = None,
    target_dataset: Optional[Dataset] = None,
    force_recompute: bool = False,
    compress: bool = False,
    in_dim: int = 25088,
    out_dim: int = 1024,
) -> Classifier:
    """Train (or load) a source classifier head on top of an encoder.

    If args.diet is True, runs DIET to refine the encoder using the target set,
    then attaches the classifier head and CE-trains on the labeled source.
    Otherwise, supports joint CE + contrastive SSL on target if available.
    """
    if model_path is None:
        model_path = f"cache{args.ssl_weight}/source_model.pth"

    # model = Classifier(encoder, MLP(mode=mode, n_class=n_class, hidden=1024)).to(device)
    encoder = encoder.to(device)  # <- important if compat class probes encoder
    model = Classifier(encoder, MLP(mode=mode, n_class=n_class, hidden=1024))
    if compress:
        # Infer encoder flattened dimension from real data to avoid mismatches across datasets
        flat_dim = in_dim
        try:
            probe_loader = DataLoader(trainset, batch_size=1, shuffle=False, num_workers=0)
            xb = next(iter(probe_loader))[0]
            if isinstance(xb, (list, tuple)):
                xb = xb[0]
            xb = xb.to(device)
            with torch.no_grad():
                h = model.encoder(xb)
            flat_dim = int(h.view(1, -1).shape[1])
            print(f"[get_source_model] detected encoder flat_dim={flat_dim}")
        except Exception:
            pass
        model = CompressClassifier(model, in_dim=flat_dim, out_dim=out_dim).to(device)

    model = model.to(device)      # <- apply to BOTH branches
    if os.path.exists(model_path) and not force_recompute:
        print(f"[Cache] Loading trained model from {model_path}")
        ckpt = torch.load(model_path, map_location=device)
        missing, unexpected = model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
        return model



    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    tgt_loader = (
        DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        if target_dataset is not None
        else None
    )

    # Optionally refine encoder with DIET prior to CE training
    if getattr(args, "diet", False) and target_dataset is not None:
        _ = train_encoder_diet(model, target_dataset, testset, epochs=200, batch_size=args.batch_size)

    # CE (supervised) + optional SSL on target
    # Choose augmentation resolution per dataset
    # inside get_source_model(...)
    # ---------------- choose augmentation policy by dataset ----------------
    is_image_mode = (mode in ("portraits", "mnist", "color_mnist"))

    if is_image_mode:
        if mode == "portraits":
            aug_size = (32, 32)
        else:  # "mnist" or "color_mnist"
            aug_size = (28, 28)
        augment_fn = build_augment(image_size=aug_size)
    else:
        aug_size = None               # ensure the name always exists
        augment_fn = None             # tabular: no image augmentations
        # turn off SSL explicitly for tabular to avoid any augment-using path
        try:
            args.ssl_weight = 0.0
        except Exception:
            pass
    for epoch in range(1, epochs + 1):
        sup_loss, ssl_loss = train_joint_plus(
            model, trainloader, tgt_loader, optimizer,
            augment_fn,
            ssl_weight=args.ssl_weight,
            epoch_idx=epoch,                                  # <-- important
            device=next(model.parameters()).device            # <-- makes it robust
        )
        msg = f"[Epoch {epoch}] Supervised CE: {sup_loss:.4f}"
        if ssl_loss is not None:
            msg += f" | SSL: {ssl_loss:.4f}"
        print(msg)
        if epoch % 5 == 0:
            test(testloader, model, verbose=verbose)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    # Save with robust metadata (optional fields)
    def _infer_n_classes_from_head(m: nn.Module):
        try:
            if hasattr(m, 'n_classes'):
                return int(m.n_classes)
            if hasattr(m, 'mlp') and hasattr(m.mlp, 'mlp') and isinstance(m.mlp.mlp, nn.Sequential):
                last = m.mlp.mlp[-1]
                if isinstance(last, nn.Linear):
                    return int(last.out_features)
        except Exception:
            pass
        return None

    meta = {"arch": type(model).__name__}
    if hasattr(model, 'emb_dim'):
        try:
            meta["emb_dim"] = int(getattr(model, 'emb_dim'))
        except Exception:
            pass
    n_classes_meta = _infer_n_classes_from_head(model)
    if n_classes_meta is not None:
        meta["n_classes"] = n_classes_meta

    torch.save({
        "state_dict": model.state_dict(),
        "meta": meta,
    }, model_path)

    return model


# -------------------------------------------------------------
# KMeans++ baseline (target-only clustering + majority map)
# -------------------------------------------------------------


def run_kmeanspp_baseline(args: argparse.Namespace, target_angle: int, n_classes: int = 10, pool: str = "gap", n_init: int = 10):
    cache_dir = f"cache{args.ssl_weight}/target{target_angle}/"
    e_src = torch.load(os.path.join(cache_dir, "encoded_0.pt"))
    e_tgt = torch.load(os.path.join(cache_dir, f"encoded_{target_angle}.pt"))

    X_tr = _to_2d_numpy(e_src.data, pool=pool)
    X_tgt = _to_2d_numpy(e_tgt.data, pool=pool)

    y_tgt = (
        e_tgt.targets.cpu().numpy() if torch.is_tensor(e_tgt.targets) else np.asarray(e_tgt.targets)
    )

    km = KMeans(n_clusters=n_classes, init="k-means++", n_init=n_init, max_iter=300, random_state=args.seed)
    cluster_ids = km.fit_predict(X_tgt)

    if y_tgt is not None:
        mapping = {}
        for c in range(n_classes):
            mask = cluster_ids == c
            if mask.any():
                vals, cnts = np.unique(y_tgt[mask], return_counts=True)
                mapping[c] = vals[cnts.argmax()]
            else:
                mapping[c] = 0
        y_hat = np.vectorize(mapping.get)(cluster_ids)
        from sklearn.metrics import accuracy_score

        acc = accuracy_score(y_tgt, y_hat)
        print(f"[KMeans++] pool='{pool}'  acc={acc:.4f}")

    return km, cluster_ids


# -------------------------------------------------------------
# Experiment runners
# -------------------------------------------------------------


def encode_all_domains(
    src_trainset: Dataset,
    tgt_trainset: Dataset,
    all_sets: List[Dataset],
    deg_idx: List[int],
    encoder: nn.Module,
    cache_dir: str,
    target: int,
    *,
    force_recompute: bool = False,
    args = None
) -> Tuple[Dataset, Dataset, List[Dataset]]:
    """Encode source/intermediate/target datasets once and cache results."""
    os.makedirs(cache_dir, exist_ok=True)

    if args.dataset == 'mnist':
        e_src = get_encoded_dataset(
            src_trainset,
            cache_path=os.path.join(cache_dir, "encoded_0.pt"),
            encoder=encoder,
            force_recompute=force_recompute,
        )
        e_tgt = get_encoded_dataset(
            tgt_trainset,
            cache_path=os.path.join(cache_dir, f"encoded_{target}.pt"),
            encoder=encoder,
            force_recompute=force_recompute,
        )
    else:
        e_src = get_encoded_dataset(
            src_trainset,
            cache_path=os.path.join(cache_dir, "encoded_source.pt"),
            encoder=encoder,
            force_recompute=force_recompute,
        )
        e_tgt = get_encoded_dataset(
            tgt_trainset,
            cache_path=os.path.join(cache_dir, f"encoded_target.pt"),
            encoder=encoder,
            force_recompute=force_recompute,
        )

    encoded_intersets: List[Dataset] = [e_src]
    intersets = all_sets[:-1]
    for idx, inter in enumerate(intersets):
        if args.dataset == 'mnist':
            cache_name = f"encoded_{deg_idx[idx]}.pt"
        else:
            cache_name = f"encoded_{idx}.pt"
        encoded_intersets.append(
            get_encoded_dataset(
                inter,
                cache_path=os.path.join(cache_dir, cache_name),
                encoder=encoder,
                force_recompute=force_recompute,
            )
        )
    encoded_intersets.append(e_tgt)
    return e_src, e_tgt, encoded_intersets


def run_goat(
    model_copy: Classifier,
    source_model: Classifier,
    src_trainset: Dataset,
    tgt_trainset: Dataset,
    all_sets: List[Dataset],
    deg_idx: List[int],
    generated_domains: int,
    epochs: int = 10,
    target: int = 60,
    args=None,
    tag: str = None,
):
    """GOAT-style baseline: direct ST on target vs pooled ST across real domains,
    optionally ST on synthetics generated between consecutive encoded domains.
    Also: plots train/test accuracies over domains for the pooled and synthetic runs.
    """
    # ----- Direct adapt (self-train only on target) -----
    # Keep original behavior; self_train requires >=2 datasets, so we use self_train_og here.
    set_all_seeds(args.seed)
    direct_acc, st_acc, train_acc_by_domain0, test_acc_by_domain0, _ = self_train(args, model_copy, [tgt_trainset], epochs=epochs, label_source="pseudo")

    # ----- Pooled ST on real intermediates + target -----
    # Use updated self_train to get per-domain accuracy lists.
    set_all_seeds(args.seed)
    direct_acc_all, st_acc_all, train_acc_list_all, test_acc_list_all, _ = self_train(
        args, source_model, all_sets, epochs=epochs, label_source="pseudo",
        use_labels=getattr(args, "use_labels", False)
    )

    # Dirs
    if args.dataset != 'mnist':
        cache_dir = f"{args.dataset}/cache{args.ssl_weight}/"
        plot_dir  = f"plots/{args.dataset}/"
    else:
        cache_dir = f"cache{args.ssl_weight}/target{target}/small_dim{args.small_dim}/"
        plot_dir  = f"plots/target{target}/"
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(plot_dir,  exist_ok=True)


    # ----- Encode all domains (unchanged functional flow) -----
    e_src, e_tgt, encoded_intersets = encode_all_domains(
        src_trainset,
        tgt_trainset,
        all_sets,
        deg_idx,
        nn.Sequential(
            source_model.encoder,
            nn.Flatten(start_dim=1),
            getattr(source_model, 'compressor', nn.Identity())
        ),
        cache_dir,
        target,
        force_recompute=False,
        args=args
    )

    # ----- Optionally generate synthetic domains and ST on them -----
    generated_acc = 0.0
    if generated_domains > 0:
        all_domains: List[Dataset] = []
        for i in range(len(encoded_intersets) - 1):
            # breakpoint()
            out = generate_domains(generated_domains, encoded_intersets[i], encoded_intersets[i + 1])
            new_domains = out[0]
            first_src_stats = out[2] 
            all_domains += new_domains
        # Use updated self_train to get per-domain accuracy lists on synthetic chain
        set_all_seeds(args.seed)
        _, generated_acc, train_acc_by_domain, test_acc_by_domain, domain_stats, last_predictions = self_train(
            args,
            source_model.mlp,
            all_domains,
            epochs=epochs,
            label_source='pseudo',
            use_labels=getattr(args, "use_labels", False),
            return_stats=True
        )

        # _save_list(os.path.join(plot_dir, "goat_train_acc_by_domain.json"), train_acc_by_domain)
        # _save_list(os.path.join(plot_dir, "goat_test_acc_by_domain.json"),  test_acc_by_domain)
        # _save_dict(os.path.join(plot_dir, f"domain_stats_gen{args.generated_domains}_dim{args.small_dim}_{args.label_source}_{args.em_match}_goat.json"), domain_stats)
        # PCA grids: real domains and synthetic chain (source → generated → target)mains
        plot_pca_classes_grid(
            encoded_intersets,
            classes=(3,6, 8, 9) if 'mnist' in args.dataset else (0,1),
            save_path=os.path.join(plot_dir, f"pca_dim{args.small_dim}_gen{args.generated_domains}_real_domains_goat.png"),
            ground_truths=True,
            pca=getattr(args, "shared_pca", None)  # <<— SAME basis
        )

        # Synthetic chain: group per pair (generated_domains + 1) and drop the appended right endpoint
        step_len = int(generated_domains) + 1
        chain_only = []
        for k in range(0, len(all_domains), max(step_len, 1)):
            chunk = all_domains[k:k + step_len]
            if not chunk:
                continue
            chain_only.extend(chunk[:-1] if step_len > 0 else chunk)
        chain_for_plot = [encoded_intersets[0]] + chain_only + [encoded_intersets[-1]]
        # Ensure the final target encoding carries pseudo labels in .targets_em so we can color by them
        tgt_pl, _ = get_pseudo_labels(
            encoded_intersets[-1],
            getattr(source_model, 'mlp', source_model),
            confidence_q=getattr(args, 'pseudo_confidence_q', 0.9),
        )
        encoded_intersets[-1].targets_em = tgt_pl.clone() if hasattr(tgt_pl, 'clone') else torch.as_tensor(tgt_pl, dtype=torch.long)

        plot_pca_classes_grid(
            chain_for_plot,
            classes=(3,6, 8, 9) if 'mnist' in args.dataset else (0,1),
            save_path=os.path.join(plot_dir, f"pca_dim{args.small_dim}_gen{args.generated_domains}_{args.label_source}_{args.em_match}_goat.png"),
            label_source='pseudo',
            pseudolabels=last_predictions,  # from self_train
            pca=getattr(args, "shared_pca", None)  # <<— SAME basis
        )

    return train_acc_by_domain, test_acc_by_domain, st_acc, st_acc_all, generated_acc


# def run_goat_classwise(
#     model_copy: Classifier,
#     source_model: Classifier,
#     src_trainset: Dataset,
#     tgt_trainset: Dataset,
#     all_sets: List[Dataset],
#     deg_idx: List[int],
#     generated_domains: int,
#     epochs: int = 10,
#     target: int = 60,
#     args=None
# ):
#     """GOAT baseline with class-wise synthetic generation (still calls generate_domains).
#        This function will compute .targets_em for any encoded dataset that lacks it.
#     """

#     # Freeze a clean teacher for EM/pseudo labels to avoid any leakage
#     # from subsequent adaptation. Use the head that consumes encoded features.
#     em_teacher = copy.deepcopy(source_model).to(device).eval()
#     em_head = getattr(em_teacher, 'mlp', em_teacher)

#     # ---------- Direct adapt (target only) ----------
#     direct_acc, st_acc, train_acc_by_domain0, test_acc_by_domain0 = self_train(
#         args, model_copy, [tgt_trainset], epochs=epochs, label_source="pseudo"
#     )

#     # ---------- Pooled ST on real intermediates + target ----------
#     direct_acc_all, st_acc_all, train_acc_list_all, test_acc_list_all = self_train(
#         args, source_model, all_sets, epochs=epochs, label_source="pseudo",
#         use_labels=getattr(args, "use_labels", False)
#     )

#     # ---------- Dirs ----------
#     cache_dir = f"cache{args.ssl_weight}/target{target}/small_dim{args.small_dim}/"
#     plot_dir  = f"plots/target{target}/"
#     os.makedirs(cache_dir, exist_ok=True)
#     os.makedirs(plot_dir,  exist_ok=True)


#     # ---------- Encode all domains ----------
#     e_src, e_tgt, encoded_intersets = encode_all_domains(
#         src_trainset,
#         tgt_trainset,
#         all_sets,
#         deg_idx,
#         nn.Sequential(
#             source_model.encoder,
#             nn.Flatten(start_dim=1),
#             getattr(source_model, 'compressor', nn.Identity())
#         ),
#         cache_dir,
#         target,
#         force_recompute=False,
#     )

#     # Use frozen teacher for pseudo labels on target to keep it target-GT agnostic
#     pseudo_labels, _pseudo_keep = get_pseudo_labels(
#         tgt_trainset,
#         em_teacher,
#         confidence_q=getattr(args, "pseudo_confidence_q", 0.1),
#         device_override=next(em_teacher.parameters()).device,
#     )
#     pseudolabels = pseudo_labels.cpu().numpy()
#     K = int(pseudo_labels.max().item()) + 1
#     # def _ensure_targets_em(ds: Dataset, infer_model: nn.Module):
#     #     """Compute ds.targets_em via EM clustered labels mapped to pseudo-labels from infer_model."""
#     #     # skip if already exists and not all -1
#     #     if getattr(ds, "targets_em", None) is not None and not (ds.targets_em == -1).all():
#     #         return
#     #     loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

#     #     # Pseudo labels from the classifier head that consumes encoded features
#     #     pseudo_lab, _ = get_pseudo_labels(
#     #         loader, infer_model, device_override=next(infer_model.parameters()).device
#     #     )
#     #     if pseudo_lab.numel() == 0:
#     #         raise ValueError("Failed to compute pseudo labels for EM mapping.")
#     #     pseudo_np = pseudo_lab.cpu().numpy()
#     #     K = int(pseudo_lab.max().item()) + 1

#     #     # EM on the encoded dataset (same feature space)
        
#     #     em_res = run_em_on_encoded(
#     #         ds, K=K, cov_type="diag", pool="flatten", do_pca=False,
#     #         return_transforms=True, verbose=False
#     #     )

#     #     if args.em_match == "prototypes":
#     #         mu_s, Sigma_s, priors_s = fit_source_gaussian_params(
#     #             X = e_src.data, y = e_src.targets)
#     #         mapping_pseudo_main, labels_mapped_pseudo_main, _ = map_em_clusters(
#     #             em_res,
#     #             method=args.em_match,
#     #             n_classes=K,
#     #             mus_s=mu_s,
#     #             Sigma_s=Sigma_s,
#     #             priors_s=priors_s,
#     #         )
#     #         ds.targets_em = torch.as_tensor(labels_mapped_pseudo_main, dtype=torch.long)
#     #         return
#     #     # Map clusters to classes by Hungarian vs pseudo labels
#     #     _, labels_mapped, _ = map_em_clusters(
#     #         em_res, method=args.em_match, n_classes=K, pseudo_labels=pseudo_np
#     #     )
#     #     ds.targets_em = torch.as_tensor(labels_mapped, dtype=torch.long)


#     def _labels_for_split(ds: Dataset, is_source: bool, ) -> torch.Tensor:
#         """
#         Class labels used for per-class splitting:
#         - If is_source==True and ds.targets exists → use ds.targets (allowed).
#         - Otherwise (non-source domains, including target) → use inferred ds.targets_em.
#         """
#         if is_source and hasattr(ds, "targets") and ds.targets is not None:
#             y = ds.targets
#             return torch.as_tensor(y).long().cpu()

#         # Always use the frozen EM head for non-source domains
#         # _ensure_targets_em(ds, em_head)
#         return torch.as_tensor(ds.targets_em).long().cpu()

#     em_res_tgt = run_em_on_encoded(
#         e_tgt,
#         K=K,
#         cov_type="diag",
#         pool="gap",            # GAP pooling
#         do_pca=False,            # Enable PCA for stability of full covariances
#         pca_dim=None,             # Reduce to 64 dimensions
#         reg=1e-4,               # Stronger ridge for full covariances
#         max_iter=500,  # Reduce max iterations from default 100 to 50
#         return_transforms=True,
#         verbose=True,  # Enable to see progress and timing
#         subsample_init=10000,  # Reduce subsampling for faster initialization
#     )
#     if args.em_match == "prototypes":
#         mu_s, Sigma_s, priors_s = fit_source_gaussian_params(
#             X = e_src.data, y = e_src.targets)

#         mapping_pseudo_main, labels_mapped_pseudo_main, _ = map_em_clusters(
#             em_res_tgt,
#             method=args.em_match,
#             n_classes=K,
#             metric='FR',
#             mus_s=mu_s,
#             Sigma_s=Sigma_s,
#             priors_s=priors_s,
#         )
#     else:
#         mapping_pseudo_main, labels_mapped_pseudo_main, _ = map_em_clusters(
#             em_res_tgt,
#             method=args.em_match,
#             n_classes=K,
#             pseudo_labels=pseudolabels,
#         )

#     def _subset_by_class(ds: Dataset, cls: int, is_source: bool) -> Optional[Dataset]:
#         """Return a per-class DomainDataset compatible with generate_domains."""
#         labels = _labels_for_split(ds, is_source=is_source)
#         X = ds.data if torch.is_tensor(getattr(ds, "data", None)) else torch.as_tensor(ds.data)
#         X = X.cpu()
#         m = (labels == int(cls))
#         if m.sum().item() == 0:
#             return None
#         Xc = X[m]
#         yc = labels[m]
#         w  = torch.ones(len(yc))
#         # DomainDataset(data, weights, targets)
#         return DomainDataset(Xc, w, yc)


#     def _merge_domains_per_step(list_of_lists: List[List[Dataset]]) -> List[Dataset]:
#         """Merge step j across classes into a single DomainDataset."""
#         if not list_of_lists:
#             return []
#         n_steps = min(len(L) for L in list_of_lists)   # should be n_inter + 1 (includes appended target)
#         merged: List[Dataset] = []
#         for j in range(n_steps):
#             parts = [L[j] for L in list_of_lists if L[j] is not None]
#             if not parts:
#                 continue
#             Xs, Ws, Ys = [], [], []
#             for D in parts:
#                 Xs.append(D.data if torch.is_tensor(D.data) else torch.as_tensor(D.data))
#                 ws = getattr(D, "weights", None)
#                 if ws is None:
#                     ws = torch.ones(len(D.targets))
#                 Ws.append(ws if torch.is_tensor(ws) else torch.as_tensor(ws))
#                 Ys.append(D.targets if torch.is_tensor(D.targets) else torch.as_tensor(D.targets))
#             X = torch.cat([x.cpu().float() for x in Xs], dim=0)
#             W = torch.cat([w.cpu().float() for w in Ws], dim=0)
#             Y = torch.cat([y.cpu().long()  for y in Ys], dim=0)
#             merged.append(DomainDataset(X, W, Y, Y))  # set targets_em := Y for training
#         return merged
#     e_tgt.targets_em = torch.as_tensor(labels_mapped_pseudo_main, dtype=torch.long)
#     # ---------- Class-wise synthetic generation loop (inside run_goat_classwise) ----------
#     generated_acc = 0.0
#     if generated_domains > 0:
#         all_domains: List[Dataset] = []
#         # Number of classes from the *inferred* labels on the final target encoding
#         # _ensure_targets_em(e_tgt, em_head)
#         K = int(torch.as_tensor(e_tgt.targets_em).max().item()) + 1

#         for i in range(len(encoded_intersets) - 1):
#             s_ds = encoded_intersets[i]
#             t_ds = encoded_intersets[i + 1]
#             # breakpoint()

#             # Define whether the left side of the pair is the original source encoding
#             is_source_left = (i == 0)

#             # Ensure EM labels for the right side (non-source)
#             # _ensure_targets_em(t_ds, getattr(source_model, 'mlp', source_model))

#             # Build per-class chains by calling the original generator per class
#             per_class_chains: List[List[Dataset]] = []
#             for c in range(K):
#                 s_c = _subset_by_class(s_ds, c, is_source=is_source_left)
#                 t_c = _subset_by_class(t_ds, c, is_source=False)  # never use GT on right
#                 if s_c is None or t_c is None:
#                     continue
#                 chain_c, _, domain_stats_c = generate_domains(generated_domains, s_c, t_c)

#                 for D in chain_c:
#                     # force GLOBAL class ID for both targets and targets_em
#                     y_global = torch.full((len(D.targets),), c, dtype=torch.long)
#                     D.targets = y_global
#                     if getattr(D, "targets_em", None) is not None:
#                         D.targets_em = y_global.clone()
#                     else:
#                         D.targets_em = y_global.clone()  # ensure it exists for self_train
#                 if chain_c:
#                     per_class_chains.append(chain_c)
#                     # check that the labels are all c
#                     for step_ds in chain_c:
#                         labs = step_ds.targets if torch.is_tensor(step_ds.targets) else torch.as_tensor(step_ds.targets)
#                         labs = labs.cpu().numpy()
#                         assert (labs == c).all(), f"Class mismatch in generated chain for class {c} at step {step_ds}"

#             # Merge per step across classes, then append
#             merged_chain = _merge_domains_per_step(per_class_chains)
#             # breakpoint()
#             all_domains += merged_chain

#         # Ensure evaluation target matches other methods: use the full encoded target
#         # as the final held-out dataset for consistency of the 0th plot point.
#         if len(all_domains) > 0:
#             all_domains[-1] = e_tgt

#         # Self-train on the synthetic class-wise chain
#         _, generated_acc, train_acc_by_domain, test_acc_by_domain = self_train(
#             args, source_model.mlp, all_domains,
#             epochs=epochs, label_source=args.label_source,
#             use_labels=getattr(args, "use_labels", False),
#             return_stats=True
#         )


#         _save_list(os.path.join(plot_dir, "goat_train_acc_by_domain.json"), train_acc_by_domain)
#         _save_list(os.path.join(plot_dir, "goat_test_acc_by_domain.json"),  test_acc_by_domain)

#         # PCA grids: real domains and synthetic chain (source → generated → target)
#         try:
#             # Real domains
#             plot_pca_classes_grid(
#                 encoded_intersets,
#                 classes=(3, 6, 8, 9) if "mnist" in args.dataset else (0, 1),
#                 save_path=os.path.join(plot_dir, f"pca_classes_real_domains_goatcw.png"),
#             )
#             # Synthetic chain: group per pair and drop appended right endpoint
#             step_len = int(generated_domains) + 1
#             chain_only = []
#             for k in range(0, len(all_domains), max(step_len, 1)):
#                 chunk = all_domains[k:k + step_len]
#                 if not chunk:
#                     continue
#                 chain_only.extend(chunk[:-1] if step_len > 0 else chunk)
#             chain_for_plot = [encoded_intersets[0]] + chain_only + [encoded_intersets[-1]]
#             plot_pca_classes_grid(
#                 chain_for_plot,
#                 classes=(3, 6, 8, 9) if "mnist" in args.dataset else (0, 1),
#                 save_path=os.path.join(plot_dir, f"pca_classes_synth_goatcw_{args.label_source}_{args.em_match}.png"),
#                 label_source=args.label_source,
#             )
#         except Exception as e:
#             print(f"[GOAT-CW][PCA] Skipped PCA plotting: {e}")

#         return train_acc_by_domain, test_acc_by_domain, st_acc, st_acc_all, generated_acc

#     # If no synthetics requested, still return meaningful values/lists
#     # return train_acc_by_domain0, test_acc_by_domain0, st_acc, st_acc_all, generated_acc




def run_goat_classwise(
    model_copy: Classifier,
    source_model: Classifier,
    src_trainset: Dataset,
    tgt_trainset: Dataset,
    all_sets: List[Dataset],
    deg_idx: List[int],
    generated_domains: int,
    epochs: int = 10,
    target: int = 60,
    args=None,
):
    """
    GOAT baseline with class-wise synthetic generation.
    Builds a single, global domain_params over steps t in [0,1] that matches the schema of FR/Natural generators:
      {
        K, d, cov_type, steps(S,), mu(S,K,d), var(S,K,d), [Sigma(S,K,d,d)],
        counts(S,K), pi(S,K), eta1(S,K,d), eta2_diag(S,K,d),
        present_source(K,), present_target(K,)
      }
    """
    device = next(source_model.parameters()).device

    # -------------------- helpers --------------------
    def _to_np(x):
        return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


    def _labels_for_split(ds: Dataset, is_source: bool) -> torch.Tensor:
        if is_source and hasattr(ds, "targets") and ds.targets is not None:
            return torch.as_tensor(ds.targets).long().cpu()
        return torch.as_tensor(ds.targets_em).long().cpu()

    def _subset_by_class(ds: Dataset, cls: int, is_source: bool) -> Optional[Dataset]:
        labels = _labels_for_split(ds, is_source=is_source)
        X = ds.data if torch.is_tensor(getattr(ds, "data", None)) else torch.as_tensor(ds.data)
        X = X.cpu()
        m = (labels == int(cls))
        if m.sum().item() == 0:
            return None
        Xc = X[m]
        yc = labels[m]
        w  = torch.ones(len(yc))
        return DomainDataset(Xc, w, yc)

    def _merge_domains_per_step(list_of_lists: List[List[Dataset]]) -> List[Dataset]:
        """Merge step j across classes into a single DomainDataset."""
        if not list_of_lists:
            return []
        n_steps = min(len(L) for L in list_of_lists)   # expected: n_inter + 1 including appended right endpoint
        merged: List[Dataset] = []
        for j in range(n_steps):
            parts = [L[j] for L in list_of_lists if L[j] is not None]
            if not parts:
                continue
            Xs, Ws, Ys = [], [], []
            for D in parts:
                Xs.append(D.data if torch.is_tensor(D.data) else torch.as_tensor(D.data))
                ws = getattr(D, "weights", None)
                if ws is None:
                    ws = torch.ones(len(D.targets))
                Ws.append(ws if torch.is_tensor(ws) else torch.as_tensor(ws))
                Ys.append(D.targets if torch.is_tensor(D.targets) else torch.as_tensor(D.targets))
            X = torch.cat([x.cpu().float() for x in Xs], dim=0)
            W = torch.cat([w.cpu().float() for w in Ws], dim=0)
            Y = torch.cat([y.cpu().long()  for y in Ys], dim=0)
            merged.append(DomainDataset(X, W, Y, Y))
        return merged


    set_all_seeds(args.seed)
    direct_acc, st_acc, train_acc_by_domain0, test_acc_by_domain0,_ = self_train(
        args, model_copy, [tgt_trainset], epochs=epochs, label_source="pseudo"
    )
    set_all_seeds(args.seed)
    direct_acc_all, st_acc_all, train_acc_list_all, test_acc_list_all, _ = self_train(
        args, source_model, all_sets, epochs=epochs, label_source="pseudo"
    )

    if args.dataset != 'mnist':
        cache_dir = f"{args.dataset}/cache{args.ssl_weight}/small_dim{args.small_dim}/"
        plot_dir  = f"plots/{args.dataset}/"
    else:
        cache_dir = f"cache{args.ssl_weight}/target{target}/small_dim{args.small_dim}/"
        plot_dir  = f"plots/target{target}/"
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(plot_dir,  exist_ok=True)

    # -------------------- 1) encode domains --------------------
    e_src, e_tgt, encoded_intersets = encode_all_domains(
        src_trainset,
        tgt_trainset,
        all_sets,
        deg_idx,
        nn.Sequential(
            source_model.encoder,
            nn.Flatten(start_dim=1),
            getattr(source_model, 'compressor', nn.Identity())
        ),
        cache_dir,
        target,
        force_recompute=False,
        args=args
    )
    # breakpoint()
    with torch.no_grad():
        teacher = copy.deepcopy(source_model).to(device).eval()
        pseudo_labels, _ = get_pseudo_labels(
            tgt_trainset, teacher,
            confidence_q=getattr(args, "pseudo_confidence_q", 0.9),
            device_override=device,
        )
    pseudolabels = pseudo_labels.cpu().numpy()
    setattr(args, "_cached_pseudolabels", pseudolabels)

    # pseudo labels on target (teacher frozen)
    K = int(e_src.targets.max().item()) + 1
    y_true = e_tgt.targets.cpu().numpy() if torch.is_tensor(e_tgt.targets) else np.asarray(e_tgt.targets)
    if args.em_match != "none":
        # 3) Reuse or fit EM (single fit) and map clusters → classes
        if hasattr(args, "_shared_em") and args._shared_em is not None:
            bundle = args._shared_em
            em_res = bundle.em_res
            labels_em = bundle.labels_em
            P_soft = bundle.P_soft
            mapping = bundle.mapping
            print("[GOAT-CW] Using shared EM bundle from cache.")
        else:
            print("[GOAT-CW] No shared EM found → fitting single EM once on target embedding.")
            # Optional: source stats for prototype mapping
            if getattr(args, "em_match", "pseudo") == "prototypes":
                mu_s, Sigma_s, priors_s = fit_source_gaussian_params(X=e_src.data, y=e_src.targets)

            # Prepare standardized target features once and run fast EM with multi-init
            X_std, scaler, pca, _ = prepare_em_representation(
                e_tgt, pool="gap",
                do_pca=False, pca_dim=None,
                dtype="float32", rng=getattr(args, "seed", 0), verbose=False
            )
            em_res = run_em_on_encoded_fast(
                X_std, K=K, cov_type="diag", reg=1e-4, max_iter=500, tol=1e-5,
                n_init=5, subsample_init=min(10000, len(e_tgt)), warm_start=None,
                rng=getattr(args, "seed", 0), verbose=False
            )
            em_res["scaler"] = scaler
            em_res["pca"] = pca

            if getattr(args, "em_match", "pseudo") == "prototypes":
                mapping, labels_em, _ = map_em_clusters(
                    em_res, method="prototypes", n_classes=K,
                    mus_s=mu_s, Sigma_s=Sigma_s, priors_s=priors_s
                )
            else:
                mapping, labels_em, _ = map_em_clusters(
                    em_res, method="pseudo", n_classes=K,
                    pseudo_labels=pseudolabels, metric='FR'
                )

            P_soft = _map_cluster_posts_to_classes(np.asarray(em_res["gamma"]), mapping, K)

            # Stash for others to reuse later
            args._shared_em = EMBundle(
                key="__inline__", em_res=em_res, mapping=mapping,
                labels_em=np.asarray(labels_em, dtype=int), P_soft=P_soft,
                info={"K": K, "cov_type": "diag", "pca_dim": None}
            )

        e_tgt.targets_em = torch.as_tensor(labels_em, dtype=torch.long)
        tgt_trainset.targets_em = e_tgt.targets_em.cpu().clone()
    e_tgt.targets_pseudo = torch.as_tensor(pseudolabels, dtype=torch.long)
    tgt_trainset.targets_pseudo = e_tgt.targets_pseudo.cpu().clone()
    y_true_np = np.asarray(y_true)
    acc_em_pseudo = float((np.asarray(labels_em) == y_true_np).mean())
    print(f"[MainAlgo] EM→class (pseudo mapping) accuracy: {acc_em_pseudo:.4f}")

    # -------------------- 2) class-wise generation --------------------
    generated_acc = 0.0
    if generated_domains <= 0:
        return train_acc_by_domain0, test_acc_by_domain0, st_acc, st_acc_all, generated_acc

    all_domains: List[Dataset] = []




    # — Source snapshot (t=0) computed on the *left* dataset of the first pair
    s0 = encoded_intersets[0]
    src_labels = _labels_for_split(s0, is_source=True)

    # generate per pair, per class; merge per step; append to all_domains
    for i in range(len(encoded_intersets) - 1):
        s_ds = encoded_intersets[i]
        t_ds = encoded_intersets[i + 1]
        is_source_left = (i == 0)

        per_class_chains: List[List[Dataset]] = []
        for c in range(K):
            s_c = _subset_by_class(s_ds, c, is_source=is_source_left)
            t_c = _subset_by_class(t_ds, c, is_source=False)
            if s_c is None or t_c is None:
                continue
            # breakpoint()
            chain_c, _, _ = generate_domains(generated_domains, s_c, t_c,
                                            #  cov_type=cov_type, reg=reg, ddof=ddof
                                             )

            # force global class id c
            for D in chain_c:
                y_global = torch.full((len(D.targets),), c, dtype=torch.long)
                D.targets = y_global
                D.targets_em = y_global.clone()
            if chain_c:
                # sanity
                for step_ds in chain_c:
                    labs = step_ds.targets if torch.is_tensor(step_ds.targets) else torch.as_tensor(step_ds.targets)
                    assert (labs.cpu().numpy() == c).all()
                per_class_chains.append(chain_c)

        merged_chain = _merge_domains_per_step(per_class_chains)
        all_domains += merged_chain



    # ensure last training domain is the full encoded target
    if len(all_domains) > 0:
        all_domains[-1] = DomainDataset(
            e_tgt.data if torch.is_tensor(e_tgt.data) else torch.as_tensor(e_tgt.data),
            torch.ones(len(e_tgt.targets)),
            e_tgt.targets, e_tgt.targets_em
        )

    # -------------------- 3) train on merged synthetic chain --------------------
    set_all_seeds(args.seed)
    _, generated_acc, train_acc_by_domain, test_acc_by_domain, domain_stats, last_prediction = self_train(
        args, source_model.mlp, all_domains,
        epochs=epochs, label_source=getattr(args, "label_source", "pseudo"),
        use_labels=getattr(args, "use_labels", False),
        return_stats=True  
    )
    # _save_list(os.path.join(plot_dir, f"goatcw_train_acc_by_domain_gen{args.generated_domains}.json"), train_acc_by_domain)
    # _save_list(os.path.join(plot_dir, f"goatcw_test_acc_by_domain_gen{args.generated_domains}.json"),  test_acc_by_domain)
    # _save_dict(os.path.join(plot_dir, f"domain_stats_gen{args.generated_domains}_dim{args.small_dim}_{args.label_source}_{args.em_match}_goatcw.json"), domain_stats)
    # PCA grids: real domains and synthetic chain (source → generated → target)mains
    plot_pca_classes_grid(
        encoded_intersets,
        classes=(3, 6, 8, 9) if "mnist" in args.dataset else (0, 1),
        save_path=os.path.join(plot_dir, f"pca_classes_real_domains_dim{args.small_dim}_gen{args.generated_domains}_goatcw.png"),
        ground_truths=True,
        pca=getattr(args, "shared_pca", None)  # <<— SAME basis
    )
    # Synthetic chain: group per pair (generated_domains + 1) and drop the appended right endpoint
    step_len = int(generated_domains) + 1
    chain_only = []
    for k in range(0, len(all_domains), max(step_len, 1)):
        chunk = all_domains[k:k + step_len]
        if not chunk:
            continue
        chain_only.extend(chunk[:-1] if step_len > 0 else chunk)
    chain_for_plot = [encoded_intersets[0]] + chain_only + [encoded_intersets[-1]]
    # Ensure the final target encoding carries pseudo labels in .targets_em so we can color by them
    tgt_pl, _ = get_pseudo_labels(
        encoded_intersets[-1],
        getattr(source_model, 'mlp', source_model),
        confidence_q=getattr(args, 'pseudo_confidence_q', 0.9),
    )
    # encoded_intersets[-1].targets_em = tgt_pl.clone() if hasattr(tgt_pl, 'clone') else torch.as_tensor(tgt_pl, dtype=torch.long)

    plot_pca_classes_grid(
        chain_for_plot,
        classes=(3, 6, 8, 9) if "mnist" in args.dataset else (0, 1),
        save_path=os.path.join(plot_dir, f"pca_dim{args.small_dim}_gen{args.generated_domains}_{args.label_source}_{args.em_match}_goatcw.png"),
        label_source='pseudo',
        pseudolabels=last_prediction,
        pca=getattr(args, "shared_pca", None)  # <<— SAME basis
    )
    plot_pca_classes_grid(
        chain_for_plot,
        classes=(3, 6, 8, 9) if "mnist" in args.dataset else (0, 1),
        save_path=os.path.join(plot_dir, f"pca_dim{args.small_dim}_gen{args.generated_domains}_emlabels_goatcw.png"),
        label_source='em',
        # pseudolabels=last_predictions,
        pca=getattr(args, "shared_pca", None)  # <<— SAME basis
    )
    plot_pca_classes_grid(
        chain_for_plot,
        classes=(3, 6, 8, 9) if "mnist" in args.dataset else (0, 1),
        save_path=os.path.join(plot_dir, f"pca_dim{args.small_dim}_gen{args.generated_domains}_source_pseudo_goatcw.png"),
        label_source='pseudo',
        pseudolabels=pseudolabels,
        pca=getattr(args, "shared_pca", None)  # <<— SAME basis
    )

    return train_acc_by_domain, test_acc_by_domain, st_acc, st_acc_all, generated_acc, acc_em_pseudo







def _safe_get_eta(domain_params):
    if "eta1" not in domain_params or "eta2_diag" not in domain_params:
        print("[plot_natural_params] eta1 / eta2_diag not found in domain_params; skipping.")
        return None, None, None
    ts   = np.asarray(domain_params["steps"], dtype=float)            # (S,)
    eta1 = np.asarray(domain_params["eta1"], dtype=float)             # (S,K,d)
    eta2 = np.asarray(domain_params["eta2_diag"], dtype=float)        # (S,K,d)
    return ts, eta1, eta2

def plot_natparam_norms(domain_params,
                        save_path: Optional[str] = None,
                        target_class: Optional[int] = None,
                        title_prefix: str = "natural"):
    """
    Plot L2 norms over steps:
      - ||η1||_2
      - ||η2_diag||_2
    If target_class is None, plots all classes (thin lines). If set, only that class.
    """
    ts, eta1, eta2 = _safe_get_eta(domain_params)
    if ts is None: 
        return

    S, K, d = eta1.shape
    if target_class is not None:
        ks = [int(target_class)]
    else:
        ks = list(range(K))

    plt.figure(figsize=(8, 5))
    for k in ks:
        n1 = np.linalg.norm(eta1[:, k, :], axis=1)
        n2 = np.linalg.norm(eta2[:, k, :], axis=1)
        lbl1 = f"class {k} ||η1||"
        lbl2 = f"class {k} ||η2||"
        alpha = 1.0 if target_class is not None else 0.5
        plt.plot(ts, n1, label=lbl1, linewidth=2 if target_class is not None else 1, alpha=alpha)
        plt.plot(ts, n2, label=lbl2, linestyle="--", linewidth=2 if target_class is not None else 1, alpha=alpha)

    plt.xlabel("t (step)")
    plt.ylabel("L2 norm")
    ttl = f"{title_prefix}: L2 norms of η (eta1, eta2_diag)"
    if target_class is not None:
        ttl += f" — class {target_class}"
    plt.title(ttl)
    plt.grid(True, alpha=0.25)
    if target_class is not None or K <= 12:
        plt.legend(ncol=2, fontsize=9)
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=160)
        plt.close()
        print(f"[plot_natparam_norms] Saved {save_path}")
    else:
        plt.show()

def plot_natparam_coords(domain_params,
                         k: int,
                         dims: Optional[Sequence[int]] = None,
                         max_dims: int = 8,
                         save_path: Optional[str] = None,
                         title_prefix: str = "natural"):
    """
    For a class k, plot selected coordinates of η1 over steps, together with the
    linear interpolation of the endpoints for each coordinate:
       pred(t) = (1 - t) * η1(0) + t * η1(1)

    This lets you visually compare “actual saved” vs “ideal linear” per entry.
    """
    ts, eta1, eta2 = _safe_get_eta(domain_params)
    if ts is None:
        return

    S, K, d = eta1.shape
    k = int(k)
    if k < 0 or k >= K:
        print(f"[plot_natparam_coords] invalid class {k} (K={K})")
        return

    if dims is None:
        # pick a small, semi-random but deterministic subset
        idx_all = np.arange(d)
        rng = np.random.default_rng(12345)
        rng.shuffle(idx_all)
        dims = np.sort(idx_all[:max_dims])
    else:
        dims = [int(x) for x in dims if 0 <= int(x) < d]
        dims = dims[:max_dims] if len(dims) > max_dims else dims
        if len(dims) == 0:
            print("[plot_natparam_coords] empty dims; nothing to plot.")
            return

    e1   = eta1[:, k, :]           # (S,d)
    e1_0 = e1[0]                   # (d,)
    e1_T = e1[-1]                  # (d,)
    # ideal linear prediction for every coord: shape (S,d)
    rhs  = (1.0 - ts)[:, None] * e1_0[None, :] + ts[:, None] * e1_T[None, :]

    plt.figure(figsize=(9, 6))
    for j in dims:
        plt.plot(ts, e1[:, j],       label=f"η1[{j}] (saved)")
        plt.plot(ts, rhs[:, j], '--', label=f"η1[{j}] lin-pred", alpha=0.8)

    plt.xlabel("t (step)")
    plt.ylabel("value")
    plt.title(f"{title_prefix}: η1 coordinates — class {k}")
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2, fontsize=8)
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=160)
        plt.close()
        print(f"[plot_natparam_coords] Saved {save_path}")
    else:
        plt.show()



def covariance_sizes(domain_stats, reduce="weighted", eps=1e-12, cls=None):
    """
    Return covariance size metrics per domain.

    If `cls is None` (default):
        - sizes_by_class: dict of arrays with shape (S, K)
        - sizes_by_domain: dict of vectors with shape (S,) after reducing across classes
          using priors ('weighted') or uniform mean ('mean').

    If `cls` (int) is provided:
        - sizes_by_class: same as above (useful for debugging/plotting)
        - sizes_by_domain: dict of vectors with shape (S,) **for that single class** only.
          No across-class reduction is done.
    """
    cov_type = domain_stats["cov_type"]
    mu = domain_stats["mu"]                       # (S,K,d)
    counts = domain_stats["counts"]               # (S,K)
    S, K, d = mu.shape

    # class weights for optional reduction
    priors = counts / np.clip(counts.sum(axis=1, keepdims=True), 1, None)  # (S,K)

    sizes_by_class = {
        "trace":  np.full((S, K), np.nan),
        "fro":    np.full((S, K), np.nan),
        "logdet": np.full((S, K), np.nan),
    }

    if cov_type == "full":
        Sigma = domain_stats["Sigma"]            # (S,K,d,d)
        for s in range(S):
            for k in range(K):
                Sk = Sigma[s, k]
                if not np.isfinite(Sk).all():
                    continue
                # symmetrize & guard SPD numerically
                Sk = 0.5 * (Sk + Sk.T)
                w = np.linalg.eigvalsh(Sk)
                w = np.clip(w, eps, None)
                sizes_by_class["trace"][s, k]  = np.sum(w)               # == np.trace(Sk)
                sizes_by_class["fro"][s, k]    = np.linalg.norm(Sk, "fro")
                sizes_by_class["logdet"][s, k] = np.sum(np.log(w))       # log(det Σ)
    else:
        var = domain_stats["var"]               # (S,K,d)
        v = np.clip(var, eps, None)
        sizes_by_class["trace"]  = v.sum(axis=2)
        sizes_by_class["fro"]    = np.sqrt((v**2).sum(axis=2))           # Frobenius of diag(var)
        sizes_by_class["logdet"] = np.sum(np.log(v), axis=2)

    # If a specific class is requested, return that class' vectors
    if cls is not None:
        if not (0 <= cls < K):
            raise IndexError(f"class index {cls} is out of range [0, {K-1}]")
        sizes_by_domain = {
            "trace":  sizes_by_class["trace"][:, cls],
            "fro":    sizes_by_class["fro"][:, cls],
            "logdet": sizes_by_class["logdet"][:, cls],
        }
        return sizes_by_class, sizes_by_domain

    # Otherwise reduce across classes
    if reduce is None:
        sizes_by_domain = None
    else:
        if reduce == "weighted":
            W = priors
        elif reduce == "mean":
            W = np.ones_like(priors) / np.maximum(1, K)
        else:
            raise ValueError("reduce must be 'weighted', 'mean', or None")

        sizes_by_domain = {
            "trace":  (sizes_by_class["trace"]  * W).sum(axis=1),
            "fro":    (sizes_by_class["fro"]    * W).sum(axis=1),
            "logdet": (sizes_by_class["logdet"] * W).sum(axis=1),
        }

    return sizes_by_class, sizes_by_domain

# def run_main_algo(
#     model_copy: nn.Module,
#     source_model: nn.Module,
#     src_trainset,
#     tgt_trainset,
#     all_sets,
#     deg_idx,
#     generated_domains: int,
#     epochs: int = 3,
#     target: int = 60,
#     args= None,
#     gen_method: str = "fr",
# ):
#     """Teacher self-training along generated domains **in embedding space**.
#     Encoder stays fixed; only the head adapts."""
#     # breakpoint()
#     # check if the source_model and model_copy have the same initial weights
#     for p1, p2 in zip(model_copy.parameters(), source_model.parameters()):
#         if not torch.equal(p1, p2):
#             print("[run_main_algo] Warning: model_copy and source_model have different initial weights.")
#             breakpoint()
#             break
#     set_all_seeds(args.seed)
#     direct_acc, st_acc, train_acc_by_domain0, test_acc_by_domain0,_ = self_train(
#         args, model_copy, [tgt_trainset], epochs=epochs, label_source="pseudo")
#     set_all_seeds(args.seed)
#     # Pooled ST on real intermediates + target
#     direct_acc_all, st_acc_all, train_acc_list_all, test_acc_list_all,_ = self_train(
#         args, source_model, all_sets, epochs=epochs, label_source="pseudo")

#     # check st_acc == st_acc_all
#     if abs(st_acc - st_acc_all) > 1e-4:
#         print(f"[run_main_algo] Warning: st_acc ({st_acc}) != st_acc_all ({st_acc_all})")
#         breakpoint()


#     em_teacher = copy.deepcopy(source_model).to(device).eval()
#     with torch.no_grad():  # avoids accidental stat updates anywhere
#         pseudo_labels, _ = get_pseudo_labels(
#             tgt_trainset, em_teacher,
#             confidence_q=getattr(args, "pseudo_confidence_q", 0.9),
#             device_override=next(em_teacher.parameters()).device,
#         )


#     pseudolabels = pseudo_labels.cpu().numpy()
#     if args.dataset != "mnist":
#         cache_dir = f"{args.dataset}_features/cache{args.ssl_weight}/"
#         plot_dir  = f"plots/{args.dataset}/"
#     else:
#         cache_dir = f"cache{args.ssl_weight}/target{target}/small_dim{args.small_dim}/"
#         plot_dir  = f"plots/target{target}/"
#     os.makedirs(cache_dir, exist_ok=True)
#     os.makedirs(plot_dir, exist_ok=True)

#     e_src, e_tgt, encoded_intersets = encode_all_domains(
#         src_trainset,
#         tgt_trainset,
#         all_sets,
#         deg_idx,
#         nn.Sequential(
#             source_model.encoder,
#             nn.Flatten(start_dim=1),
#             source_model.compressor if hasattr(source_model, 'compressor') else nn.Identity()
#         ),  # <- encode with full model (encoder -> flatten -> compressor)
#         cache_dir,
#         target,
#         force_recompute=False,
#         args=args
#     )


#     y_true = e_tgt.targets
#     if torch.is_tensor(y_true):
#         y_true = y_true.cpu()

#     preds = torch.as_tensor(pseudolabels, device=y_true.device)
#     acc = (preds == y_true).float().mean().item()
#     print(f"Pseudo-label accuracy: {acc:.4f}")

#     # -------- Step 1: EM soft-targets on target embeddings, train head with soft labels --------
#     try:
#         K = int(max(int(preds.max().item()), int(y_true.max().item()))) + 1
#     except Exception:
#         K = 10

#     print(f"[MainAlgo] Running EM clustering with K={K} components...")
#     start_time = time.time()
    

#     # fit 
#     # EM on encoded target (use same pooling/shape as encoded data)
#     # Optimizations: PCA for dimensionality reduction, GAP pooling instead of flatten, fewer iterations
#     em_res_tgt = run_em_on_encoded(
#         e_tgt,
#         K=K,
#         cov_type="diag",
#         pool="gap",
#         do_pca=False,
#         pca_dim=None,
#         reg=1e-4,
#         max_iter=500,  # Reduce max iterations from default 100 to 50
#         return_transforms=True,
#         verbose=True,  # Enable to see progress and timing
#         subsample_init=10000,  # Reduce subsampling for faster initialization
#         rng=args.seed,
#     )
#     em_time = time.time() - start_time
#     print(f"[MainAlgo] EM clustering completed in {em_time:.2f} seconds")
    

#     if args.em_match == "prototypes":
#         mu_s, Sigma_s, priors_s = fit_source_gaussian_params(
#             X = e_src.data, y = e_src.targets)

#         mapping_pseudo_main, labels_mapped_pseudo_main, _ = map_em_clusters(
#             em_res_tgt,
#             method=args.em_match,
#             n_classes=K,
#             mus_s=mu_s,
#             Sigma_s=Sigma_s,
#             priors_s=priors_s,
#         )
#     else:
#         mapping_pseudo_main, labels_mapped_pseudo_main, _ = map_em_clusters(
#             em_res_tgt,
#             method=args.em_match,
#             n_classes=K,
#             pseudo_labels=pseudolabels,
#             metric='FR'
#         )

#     y_true_np = np.asarray(y_true)
#     acc_em_pseudo = float((np.asarray(labels_mapped_pseudo_main) == y_true_np).mean())
#     print(f"[MainAlgo] EM→class (pseudo mapping) accuracy: {acc_em_pseudo:.4f}")
#     # breakpoint()
#     try:
#         best_acc, _best_map, _C = best_mapping_accuracy(em_res_tgt["labels"], y_true_np)
#         print(f"[MainAlgo] Best one-to-one mapping accuracy: {best_acc:.4f}")
#     except Exception as e:
#         print(f"[MainAlgo] Best-mapping computation failed: {e}")

#     # Build EM soft class targets using the prototype mapping if available; otherwise pseudo mapping
#     # use_mapping = mapping_proto_main if mapping_proto_main is not None else mapping_pseudo_main
#     # use_mapping = mapping_pseudo_main
#     # em_soft = em_soft_targets_from_mapping(em_res_tgt["gamma"], use_mapping, n_classes=K)

#     # Ensure target datasets carry EM labels for downstream steps
#     e_tgt.targets_em = torch.as_tensor(labels_mapped_pseudo_main, dtype=torch.long)
#     tgt_trainset.targets_em = e_tgt.targets_em.cpu().clone()




#     X_std = np.asarray(em_res_tgt.get("X"))
#     if X_std is not None and X_std.ndim == 2 and X_std.shape[0] == len(e_tgt):
#         pca = PCA(n_components=2)
#         Z = pca.fit_transform(X_std)

#         # Prepare shared items
#         y_np      = np.asarray(y_true)                    # ground-truth for left panel
#         right_em  = np.asarray(labels_mapped_pseudo_main) # EM-mapped labels (int array)
#         right_pl  = np.asarray(pseudolabels)              # raw pseudo labels (int array)
#         cmap      = plt.get_cmap('tab10')

#         # Project EM means/covs into PCA(2)
#         C2 = pca.components_[:2, :]
#         m2 = pca.mean_
#         mu_src     = em_res_tgt.get("mu")
#         Sigma_src  = em_res_tgt.get("Sigma")
#         mapping_ps = mapping_pseudo_main  # cluster id -> class id

#         # 1) Second panel colored by EM labels
#         plot_pca_em_pair_side_by_side(
#             X_std,
#             y_np,
#             mu_src,
#             Sigma_src,
#             mapping_ps,
#             right_labels=right_em,
#             right_title="EM-mapped labels",
#             save_path=os.path.join(plot_dir, f"target_pca_emlabels_dim{args.small_dim}_side_by_side_{gen_method}.png"),
#         )

#         # 2) Second panel colored by pseudo labels
#         plot_pca_em_pair_side_by_side(
#             X_std,
#             y_np,
#             mu_src,
#             Sigma_src,
#             mapping_ps,
#             right_labels=right_pl,
#             right_title="Pseudo labels",
#             save_path=os.path.join(plot_dir, f"target_pca_pseudolabels_dim{args.small_dim}_side_by_side_{gen_method}.png"),
#         )


#     # Evaluate updated model on target images
#     tgt_loader_eval = DataLoader(tgt_trainset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
#     _, tgt_acc_after = test(tgt_loader_eval, source_model)
#     # print(f"[Head-Soft] Target accuracy after soft-target tuning: {tgt_acc_after:.4f}")
#     # breakpoint()
#     if generated_domains <= 0:
#         return direct_acc, st_acc, direct_acc_all, st_acc_all, 0.0

#     synthetic_domains: List[Dataset] = []
#     # choose generator based on flag (default 'fr'); accept common aliases
#     _method = (gen_method or "fr").lower()
#     if _method in {"fr", "fisher-rao", "fisher_rao"}:
#         _gen_fn = generate_fr_domains_between_optimized
#     elif _method in {"natural", "natureal", "eta", "nat", "np"}:
#         _gen_fn = generate_natural_domains_between
#     else:
#         raise ValueError(f"Unknown gen_method '{gen_method}'. Use 'fr' or 'natural'.")

#     for i in range(len(encoded_intersets) - 1):
#         # breakpoint()
#         out = _gen_fn(
#             generated_domains,
#             encoded_intersets[i],
#             encoded_intersets[i + 1],
#             # source_model=source_model,
#             # pseudolabels=pseudolabels,
#             # visualize=False,
#             cov_type="full",
#             save_path=plot_dir,
#             args=args,
#         )
#         pair_domains = out[0]
#         domain_stats = out[2]

#         sizes_by_class, sizes_c1 = covariance_sizes(domain_stats, cls=1)
#         print("Class 1 trace per domain:", sizes_c1["trace"])
#         print("Class 1 logdet per domain:", sizes_c1["logdet"])



#         # include the appended target from the generator; self_train will hold it out
#         synthetic_domains += pair_domains



#     if not synthetic_domains:
#         return direct_acc, st_acc, direct_acc_all, st_acc_all, 0.0
#     set_all_seeds(args.seed)
#     breakpoint()
#     direct_acc_syn, generated_acc, train_acc_by_domain, test_acc_by_domain, last_predictions = self_train(
#         args,
#         source_model.mlp,
#         synthetic_domains,
#         epochs=epochs,
#         label_source=args.label_source,
#     )


#     plot_pca_classes_grid(
#         encoded_intersets,
#         classes=(3, 6, 8, 9) if "mnist" in args.dataset else (0, 1),
#         save_path=os.path.join(plot_dir, f"pca_dim{args.small_dim}_gen{args.generated_domains}_real_domains.png"),
#         label_source='real',
#         ground_truths=True,
#         pca=getattr(args, "shared_pca", None)  # <<— SAME basis
#     )
#     # Synthetic chain if available: source (domain 0), then generated steps, then final target
#     if synthetic_domains:
#         chain = []
#         step_len = int(generated_domains) + 1  # per pair: n_inter steps + appended right-side domain
#         for k in range(0, len(synthetic_domains), max(step_len, 1)):
#             chunk = synthetic_domains[k:k + step_len]
#             if not chunk:
#                 continue
#             # exclude the last element (the right-hand endpoint) from each chunk
#             chain.extend(chunk[:-1] if step_len > 0 else chunk)
#         # prepend source and append final target to show full path
#         chain_for_plot = [encoded_intersets[0]] + chain + [encoded_intersets[-1]]
#         # Color the target by pseudo labels rather than GT or EM labels
#         # try:
#         #     encoded_intersets[-1].targets_em = torch.as_tensor(pseudolabels, dtype=torch.long)
#         # except Exception as _e:
#         #     print(f"[MainAlgo][PCA] Warning: failed to attach pseudo labels to target for coloring: {_e}")
#         plot_pca_classes_grid(
#             chain_for_plot,
#             classes=(3, 6, 8, 9) if "mnist" in args.dataset else (0, 1),
#             save_path=os.path.join(plot_dir, f"pca_classes_synth_dim{args.small_dim}_gen{args.generated_domains}_{args.label_source}_{args.em_match}_{_method}.png"),
#             label_source=args.label_source,
#             pseudolabels=last_predictions,
#             pca=getattr(args, "shared_pca", None)  # <<— SAME basis
#         )

#         plot_pca_classes_grid(
#             chain_for_plot,
#             classes=(3, 6, 8, 9) if "mnist" in args.dataset else (0, 1),
#             save_path=os.path.join(plot_dir, f"pca_dim{args.small_dim}_gen{args.generated_domains}_emlabels_{_method}.png"),
#             label_source='em',
#             # pseudolabels=last_predictions,
#             pca=getattr(args, "shared_pca", None)  # <<— SAME basis
#         )
#         plot_pca_classes_grid(
#             chain_for_plot,
#             classes=(3, 6, 8, 9) if "mnist" in args.dataset else (0, 1),
#             save_path=os.path.join(plot_dir, f"pca_dim{args.small_dim}_gen{args.generated_domains}_source_pseudo_{_method}.png"),
#             label_source='pseudo',
#             pseudolabels=pseudolabels,
#             pca=getattr(args, "shared_pca", None)  # <<— SAME basis
#         )
        


#     return train_acc_by_domain, test_acc_by_domain, st_acc, st_acc_all, generated_acc, acc_em_pseudo


# Assumes these are available in your environment
# from your_module import (
#   set_all_seeds, self_train, get_pseudo_labels, encode_all_domains,
#   run_em_on_encoded, fit_source_gaussian_params, map_em_clusters,
#   best_mapping_accuracy, plot_pca_em_pair_side_by_side, covariance_sizes,
#   generate_fr_domains_between_optimized, generate_natural_domains_between,
#   plot_pca_classes_grid, test
# )

# ------------------------- Utilities for multi-EM --------------------------

def _count_gmm_params(K: int, D: int, cov_type: str) -> int:
    """Number of free parameters in a K-component, D-dim GMM."""
    cov_type = cov_type.lower()
    n_means = K * D
    if cov_type == "diag":
        n_cov = K * D
    elif cov_type == "full":
        n_cov = K * (D * (D + 1) // 2)
    else:
        raise ValueError(f"Unsupported cov_type={cov_type}")
    n_weights = K - 1  # mixture weights sum to 1
    return n_means + n_cov + n_weights

def _bic(loglik: float, n_params: int, n_samples: int) -> float:
    """Bayesian Information Criterion (lower is better)."""
    return -2.0 * loglik + n_params * math.log(max(n_samples, 1))

def _safe_last_ll(em_res: Dict) -> Optional[float]:
    """Extract final log-likelihood from EM result dict."""
    # Expect either 'll_trace' or a scalar 'll'
    if "ll_trace" in em_res and len(em_res["ll_trace"]) > 0:
        return float(em_res["ll_trace"][-1])
    if "ll" in em_res:
        return float(em_res["ll"])
    return None

def _map_cluster_posts_to_classes(gamma: np.ndarray, mapping: Dict[int, int], K: int) -> np.ndarray:
    """
    Map cluster responsibilities (N x K_clusters) to class posteriors (N x K_classes)
    using a cluster->class mapping. If multiple clusters map to same class, sum them.
    """
    n = gamma.shape[0]
    out = np.zeros((n, K), dtype=float)
    for c_from, c_to in mapping.items():
        if 0 <= c_from < gamma.shape[1] and 0 <= c_to < K:
            out[:, c_to] += gamma[:, c_from]
    # Normalize row-wise to guard against numerical drift
    row_sums = out.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return out / row_sums

def _soft_average(p_list: List[np.ndarray], weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Weighted average of a list of (N x K) probability matrices."""
    assert len(p_list) > 0
    N, K = p_list[0].shape
    W = np.ones(len(p_list)) if weights is None else np.asarray(weights, dtype=float)
    W = W / W.sum()
    acc = np.zeros((N, K), dtype=float)
    for w, P in zip(W, p_list):
        acc += w * P
    # normalize for safety (should already be normalized)
    acc /= acc.sum(axis=1, keepdims=True)
    return acc

def _entropy(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise entropy for (N x K) probs."""
    q = np.clip(p, eps, 1.0)
    return -(q * np.log(q)).sum(axis=1)


# ---------- Ensemble trimming, entropy, and soft-target finetune utilities ----------

def _trim_em_models_by_bic(em_models, max_delta_bic: float = 10.0):
    """
    Keep only EM runs whose BIC is within max_delta_bic of the best (lowest) BIC.
    Returns (trimmed_models, weights) where weights are BIC-based (exp(-0.5 * ΔBIC)).
    """
    bics = np.array([m["bic"] for m in em_models], dtype=float)
    if not np.all(np.isfinite(bics)):
        # Guard: drop NaN/inf BIC runs first
        keep = np.isfinite(bics)
        em_models = [m for i, m in enumerate(em_models) if keep[i]]
        bics = bics[keep]
        if len(em_models) == 0:
            raise RuntimeError("All EM runs have non-finite BIC; fix EM numerics first.")

    best = float(np.min(bics))
    deltas = bics - best
    keep = deltas <= max_delta_bic
    trimmed = [m for i, m in enumerate(em_models) if keep[i]]
    kept_deltas = deltas[keep]

    # BIC weights ~ exp(-0.5 * ΔBIC)
    w = np.exp(-0.5 * kept_deltas)
    w = w / w.sum()

    return trimmed, w


def _ensemble_posteriors_trimmed(em_models_trimmed, weights=None):
    """
    Weighted average of mapped class posteriors from trimmed EM models.
    em_models_trimmed: list of dicts with 'mapped_soft' (N x K).
    weights: numpy array summing to 1 (len == len(em_models_trimmed)); if None, uniform.
    Returns P_ens (N x K) and per-sample entropy (N,).
    """
    P_list = [np.asarray(m["mapped_soft"], dtype=float) for m in em_models_trimmed]
    N, K = P_list[0].shape
    if weights is None:
        weights = np.ones(len(P_list), dtype=float) / len(P_list)
    else:
        weights = np.asarray(weights, dtype=float)
        weights = weights / weights.sum()

    P_ens = np.zeros((N, K), dtype=float)
    for w, P in zip(weights, P_list):
        P_ens += w * P

    # normalize defensively
    P_ens /= np.clip(P_ens.sum(axis=1, keepdims=True), 1e-12, None)

    # entropy (natural log)
    Q = np.clip(P_ens, 1e-12, 1.0)
    H = -(Q * np.log(Q)).sum(axis=1)

    return P_ens, H


def _lowest_entropy_mask(H: np.ndarray, p_keep: float):
    """
    Return boolean mask selecting the lowest-entropy p% samples.
    p_keep in [0, 100]. Example: p_keep=60 selects the lowest 60% entropy samples.
    """
    assert 0.0 <= p_keep <= 100.0
    if p_keep >= 100.0:
        return np.ones_like(H, dtype=bool)
    if p_keep <= 0.0:
        return np.zeros_like(H, dtype=bool)
    q = np.percentile(H, p_keep)
    return H <= q


class _SoftTargetSubset(Dataset):
    """
    Wrap a base dataset (images, labels) to provide only a subset of indices,
    and to attach a soft target distribution per sample.
    """
    def __init__(self, base_dataset, indices, soft_targets: np.ndarray):
        self.base = base_dataset
        self.indices = np.asarray(indices, dtype=int)
        # soft_targets assumed to be N x K over the FULL target set; we slice by indices
        self.soft = soft_targets[self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        base_idx = int(self.indices[i])
        x, _y = self.base[base_idx]  # ignore original hard label for this fine-tune
        # Return x and soft target (as float32 tensor)
        return x, torch.from_numpy(self.soft[i]).float()


def _soft_ce_loss_from_probs(logits: torch.Tensor, soft_targets: torch.Tensor):
    """
    Soft cross-entropy (a.k.a. KL with one-hot replaced by soft probs, temperature=1).
    logits: (B, K); soft_targets: (B, K), sums to 1.
    Returns scalar loss.
    """
    log_probs = torch.log_softmax(logits, dim=1)
    # mean over batch of - sum_k p_k * log q_k
    return -(soft_targets * log_probs).sum(dim=1).mean()


@torch.no_grad()
def _freeze_except_head(model):
    """
    Freeze all parameters except those in the classifier head 'mlp' (change if your head name differs).
    """
    for p in model.parameters():
        p.requires_grad_(False)
    # Unfreeze head
    head = getattr(model, "mlp", None)
    if head is None:
        raise AttributeError("Expected source_model to have attribute 'mlp' for the head.")
    for p in head.parameters():
        p.requires_grad_(True)


def finetune_head_with_soft_targets_on_low_entropy(
    args,
    source_model: nn.Module,
    tgt_trainset,
    P_ens: np.ndarray,
    H: np.ndarray,
    p_keep: float = 60.0,
    epochs: int = 1,
    batch_size: int = None,
    lr: float = None,
    weight_decay: float = 0.0,
    num_workers: int = None,
    device: torch.device = None,
):
    """
    Freeze encoder, fine-tune classifier head on the lowest-entropy p% samples using soft targets.

    - P_ens: (N x K) ensemble class probs for each sample in tgt_trainset (in the SAME order as indexing).
    - H:     (N,) entropy for each sample.
    """
    if device is None:
        device = next(source_model.parameters()).device
    if batch_size is None:
        batch_size = getattr(args, "batch_size", 256)
    if lr is None:
        lr = getattr(args, "soft_ft_lr", 1e-3)
    if num_workers is None:
        num_workers = getattr(args, "num_workers", 2)

    # Build mask and subset
    mask = _lowest_entropy_mask(H, p_keep=float(p_keep))
    idx = np.nonzero(mask)[0]
    if idx.size == 0:
        print("[SoftFT] No samples selected by entropy mask; skipping soft fine-tune.")
        return 0.0

    ds = _SoftTargetSubset(tgt_trainset, idx, P_ens)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    source_model.train()
    _freeze_except_head(source_model)
    head = source_model.mlp

    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)

    running_loss = 0.0
    n_batches = 0
    for ep in range(int(epochs)):
        for x, p in loader:
            x = x.to(device, non_blocking=True)
            p = p.to(device, non_blocking=True)
            logits = source_model(x)  # forward through full model; only head has grads
            loss = _soft_ce_loss_from_probs(logits, p)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running_loss += float(loss.item())
            n_batches += 1

    mean_loss = (running_loss / max(n_batches, 1)) if n_batches > 0 else 0.0
    print(f"[SoftFT] Done: p_keep={p_keep:.1f}%, epochs={epochs}, batches={n_batches}, mean_loss={mean_loss:.4f}, kept={idx.size}/{len(tgt_trainset)}")
    return mean_loss

import math


# --- log-likelihood extraction
def _safe_last_ll(em_res) -> float:
    llc = em_res.get("ll_curve", None)
    if isinstance(llc, (list, tuple)) and len(llc) > 0:
        return float(llc[-1])
    if "X" in em_res and all(k in em_res for k in ("mu", "Sigma", "pi")):
        # Fallback: compute LL with your earlier helper
        return float(estimate_log_likelihood(
            em_res["X"], em_res["mu"], em_res["Sigma"], em_res["pi"],
            cov_type=em_res.get("cov_type", "diag"),
            reg=1e-6
        ))
    return float("nan")

# --- parameter count for GMM
def _count_gmm_params(K: int, D: int, cov_type: str) -> int:
    cov_type = str(cov_type).lower()
    means = K * D
    if cov_type == "diag":
        covs = K * D
    elif cov_type == "spherical":
        covs = K
    elif cov_type == "full":
        covs = K * (D * (D + 1) // 2)
    else:
        raise ValueError(f"Unknown cov_type {cov_type}")
    weights = (K - 1)  # simplex
    return means + covs + weights

# --- information criteria
def _bic(ll: float, n_params: int, n_samples: int) -> float:
    return -2.0 * float(ll) + float(n_params) * math.log(max(1, n_samples))

def _icl(bic: float, gamma: np.ndarray) -> float:
    # ICL = BIC - 2 * sum_i H(gamma_i)
    eps = 1e-12
    Pi = np.clip(gamma, eps, 1.0)
    ent = -(Pi * np.log(Pi)).sum(axis=1).sum()
    return float(bic - 2.0 * ent)

# --- selection among fitted models
def _model_simplicity_key(cfg: dict) -> tuple:
    # simpler first: diag < spherical < full ; smaller K ; smaller pca_dim (None treated as 0)
    cov_rank = {"diag": 0, "spherical": 1, "full": 2}.get(str(cfg["cov_type"]).lower(), 3)
    k = int(cfg["K"])
    pd = cfg.get("pca_dim", None)
    pca_rank = 0 if (pd in [None, 0]) else int(pd)
    return (cov_rank, k, pca_rank)


# --- trimmed-BIC ensemble
def _trim_em_models_by_bic(models: list, max_delta_bic: float = 10.0):
    """
    Keep models within ΔBIC <= max_delta_bic of the best-BIC model; compute normalized weights.
    Weights ~ exp(-0.5 * ΔBIC) (a standard approximation to posterior model weights).
    """
    best = _select_best_em(models, criterion="bic")
    b0 = float(best["bic"])
    kept = []
    ws = []
    for m in models:
        db = float(m["bic"]) - b0
        if db <= max_delta_bic:
            kept.append(m)
            ws.append(math.exp(-0.5 * max(0.0, db)))
    W = np.asarray(ws, dtype=float)
    if W.sum() <= 0:
        W = np.ones_like(W)
    W = W / W.sum()
    return kept, W

def _ensemble_posteriors_trimmed(models_trimmed: list, weights: np.ndarray):
    """
    Weighted average of mapped class-posteriors (already K_classes aligned) across models.
    Returns:
      P_ens (N x K_classes), H_ens (N,)
    """
    assert len(models_trimmed) == len(weights)
    Ps = [np.asarray(m["mapped_soft"], dtype=float) for m in models_trimmed]
    N, Kc = Ps[0].shape
    P = np.zeros((N, Kc), dtype=float)
    for w, Pi in zip(weights, Ps):
        P += w * Pi
    # normalize row-wise just in case of numerical drift
    s = P.sum(axis=1, keepdims=True); s[s == 0] = 1.0
    P = P / s
    eps = 1e-12
    H = -(np.clip(P, eps, 1.0) * np.log(np.clip(P, eps, 1.0))).sum(axis=1)
    return P, H

def fit_many_em_on_target(
    e_tgt,
    K_list: List[int],
    cov_types: List[str],
    seeds: List[int],
    pool: str = "gap",
    pca_dims: Optional[List[Optional[int]]] = None,
    reg: float = 1e-4,
    max_iter: int = 300,
    rng_base: int = 0,
    args=None,
) -> List[Dict]:
    """
    Run a grid of EM configurations on encoded target embeddings.
    Returns a list of dicts (one per config):
      {
        'cfg': {'K', 'cov_type', 'pca_dim', 'seed},
        'em_res': dict,
        'final_ll': float,
        'bic': float,
        'mapped_soft': np.ndarray,   # (N x K_classes)
        'labels_mapped': np.ndarray, # (N,)
        'mapping': Dict[int,int],    # cluster -> class
      }
    tqdm is used for grid progress; falls back to a no-op if unavailable.
    """
    results: List[Dict] = []
    N = len(e_tgt)
    if pca_dims is None:
        pca_dims = [None]

    # Inferred class count for mapped outputs
    try:
        y_true = e_tgt.targets.cpu().numpy() if torch.is_tensor(e_tgt.targets) else np.asarray(e_tgt.targets)
        K_default = int(max(int(y_true.max()), 0)) + 1
    except Exception:
        K_default = 10
    K_for_classes = max([K_default] + [int(k) for k in (K_list or [])]) if K_list else K_default

    # Mapping prerequisites
    use_proto = (getattr(args, "em_match", "pseudo") == "prototypes")
    if use_proto and not hasattr(args, "_cached_source_stats"):
        raise RuntimeError("Prototype mapping requested but args._cached_source_stats is missing.")
    if (not use_proto) and not hasattr(args, "_cached_pseudolabels"):
        raise RuntimeError("Pseudo mapping requested but args._cached_pseudolabels is missing.")
    # no interactive breakpoints in library code
    mu_s = Sigma_s = priors_s = None
    if use_proto:
        mu_s, Sigma_s, priors_s = args._cached_source_stats

    Ks        = [int(k) for k in (K_list if K_list else [K_default])]
    Covs      = [str(c) for c in cov_types]
    Pcas      = [None if (p in [None, 0]) else int(p) for p in pca_dims]
    Seeds     = [int(s) for s in seeds]
    total_cfg = len(Ks) * len(Covs) * len(Pcas) * len(Seeds)

    # >>> NEW: compute & print total GMM count at the start
    total_cfg = len(Ks) * len(Covs) * len(Pcas) * len(Seeds)
    print(f"[EM-grid] Total GMMs to fit: {total_cfg}  "
          f"(K={Ks}, cov={Covs}, pca={Pcas}, seeds={Seeds})")

    # Grid progress bar (degrades to no-op if tqdm is the shim)
    bar = tqdm(total=total_cfg, desc="Fitting EM grid", leave=True)

    for K in Ks:
        for cov_type in Covs:
            for pca_dim in Pcas:
                for seed in Seeds:
                    cfg = dict(K=K, cov_type=cov_type, pca_dim=pca_dim, seed=seed)
                    # Update caption (safe for no-op tqdm)
                    try:
                        bar.set_postfix_str(f"K={K}, cov={cov_type}, pca={pca_dim}, seed={seed}")
                    except Exception:
                        pass
                    # breakpoint()
                    # ---- Fit EM for this configuration ----
                    em_res = run_em_on_encoded(
                        e_tgt,
                        K=K,
                        cov_type=cov_type,
                        pool=pool,
                        do_pca=(pca_dim is not None),
                        pca_dim=pca_dim,
                        reg=reg,
                        max_iter=max_iter,
                        return_transforms=True,
                        verbose=False,                 # avoid clashing prints with tqdm
                        subsample_init=min(20000, N),
                        rng=seed if seed is not None else rng_base,
                        # Optional: if your EM supports per-iteration tqdm via a hook, wire it:
                        # show_iter_progress=getattr(args, "em_show_iter_progress", False),
                    )

                    # ---- Score: LL + BIC ----
                    final_ll = _safe_last_ll(em_res)
                    if "X" in em_res and em_res["X"] is not None:
                        D_eff = int(em_res["X"].shape[1]); n_samples = int(em_res["X"].shape[0])
                    else:
                        # fallback dimensionality
                        D_eff = getattr(e_tgt, "dim", None)
                        if D_eff is None and hasattr(e_tgt, "data"):
                            arr = np.asarray(e_tgt.data)
                            D_eff = int(arr.reshape(arr.shape[0], -1).shape[1])
                        D_eff = int(D_eff) if D_eff is not None else 64
                        n_samples = N
                    n_params = _count_gmm_params(K, D_eff, cov_type)
                    bic_val  = _bic(final_ll if final_ll is not None and not math.isnan(final_ll) else -1e12,
                                    n_params, n_samples)

                    # ---- Map clusters -> classes ----
                    if use_proto:
                        mapping, labels_mapped, cost = map_em_clusters(
                            em_res, method="prototypes", n_classes=K_for_classes,
                            mus_s=mu_s, Sigma_s=Sigma_s, priors_s=priors_s
                        )
                    else:
                        mapping, labels_mapped, cost = map_em_clusters(
                            em_res, method="pseudo", n_classes=K_for_classes,
                            pseudo_labels=args._cached_pseudolabels, metric='FR'
                        )

                    gamma = np.asarray(em_res["gamma"])      # (N x K_clusters)
                    mapped_soft = _map_cluster_posts_to_classes(gamma, mapping, K=K_for_classes)

                
                    if isinstance(cost_raw, dict):
                        cost_scalar = float(np.nansum(list(cost_raw.values())))
                    else:
                        cost_scalar = float(np.nansum(np.asarray(cost_raw, dtype=float)))


                    results.append(dict(
                        cfg=cfg,
                        em_res=em_res,
                        final_ll=(float(final_ll) if final_ll is not None else float("nan")),
                        bic=float(bic_val),
                        cost=cost_scalar,
                        mapped_soft=mapped_soft,
                        labels_mapped=np.asarray(labels_mapped, dtype=int),
                        mapping=mapping,
                        # pm_acc=(float(pm_acc) if pm_acc is not None else float("nan")),
                    ))

                    # advance the grid bar
                    try:
                        bar.update(1)
                    except Exception:
                        pass

    try:
        bar.close()
    except Exception:
        pass

    return results

from scipy.optimize import linear_sum_assignment

import math

def score_em_by_cost(cost_matrix, cluster_sizes=None, class_priors=None, tau=1.0):
    C = np.asarray(cost_matrix, dtype=float)  # shape (Kc, K)
    # Hungarian assignment on cost
    r, c = linear_sum_assignment(C)
    J_assign = float(C[r, c].sum())

    # Optional weighting of assignment cost
    if cluster_sizes is not None:
        w = np.asarray(cluster_sizes, dtype=float)[r]
        w = w / (w.sum() + 1e-12)
        J_assign = float((w * C[r, c]).sum())
    elif class_priors is not None:
        w = np.asarray(class_priors, dtype=float)[c]
        w = w / (w.sum() + 1e-12)
        J_assign = float((w * C[r, c]).sum())

    # Confidence (margin) & entropy (for tie-break)
    best = C.min(axis=1)
    second = np.partition(C, 1, axis=1)[:,1]
    margin = float(np.nansum(second - best))  # larger is better

    A = np.exp(-C / max(tau, 1e-6))
    A = A / (A.sum(axis=1, keepdims=True) + 1e-12)
    ent = float(-(A * (np.log(A + 1e-12))).sum())  # smaller is better

    return dict(J_assign=J_assign, margin=margin, entropy=ent, assignment=(r, c))

def _select_best_em(results, criterion="cost"):
    # results[i]["cost_matrix"] should hold the full C; we'll derive scalar keys
    scored = []
    for r in results:
        sc = score_em_by_cost(
            r["cost_matrix"],
            cluster_sizes=r.get("cluster_sizes"),
            class_priors=r.get("class_priors"),
            tau=1.0
        )
        r = {**r,
             "cost": sc["J_assign"],
             "margin": sc["margin"],
             "entropy": sc["entropy"],
             "assignment": sc["assignment"]}
        scored.append(r)
    # primary: minimize assignment cost; secondary: maximize margin; tertiary: minimize entropy
    return min(scored, key=lambda x: (x["cost"], -x["margin"], x["entropy"]))

import math
from typing import List, Dict

def _safe(v, bad_for_min=math.inf, bad_for_max=-math.inf):
    # Replace None/NaN with sentinels depending on whether we will min or max it.
    if v is None:
        return bad_for_min  # default assumption; pass bad_for_max explicitly when needed
    try:
        return v if not math.isnan(v) else bad_for_min
    except (TypeError, ValueError):
        return bad_for_min

def _select_best_em(results: List[Dict], criterion: str = "bic") -> Dict:
    """
    Pick the single best EM model.

    criterion:
      - 'bic'  : minimize BIC; tie-break by larger final_ll
      - 'll'   : maximize final log-likelihood
      - 'cost' : minimize assignment cost from score_em_by_cost(C); tie-break by larger margin, then smaller entropy, then BIC
      - 'pm'/'match'/'perfmatch': maximize perfect-matching accuracy
    """
    assert len(results) > 0, "No EM results provided."

    if criterion == "bic":
        # Lower BIC is better; tie-break with higher LL
        return min(
            results,
            key=lambda r: (_safe(r.get("bic"), bad_for_min=math.inf),
                           -_safe(r.get("final_ll"), bad_for_min=-math.inf, bad_for_max=-math.inf))
        )

    if criterion == "ll":
        # Higher LL is better
        return max(
            results,
            key=lambda r: _safe(r.get("final_ll"), bad_for_min=-math.inf, bad_for_max=-math.inf)
        )

    if criterion == "cost":
        # Requires a cost matrix per result
        best = None
        best_key = None
        for r in results:
            C = r.get("cost_matrix", None)
            if C is None:
                # If absent, treat as very bad (infinite cost)
                key = (math.inf, 0.0, math.inf, _safe(r.get("bic"), bad_for_min=math.inf))
            else:
                sc = score_em_by_cost(
                    C,
                    cluster_sizes=r.get("cluster_sizes"),
                    class_priors=r.get("class_priors"),
                    tau=1.0
                )
                # Persist the diagnostics on the original dict
                r["cost"]       = sc["J_assign"]
                r["margin"]     = sc["margin"]
                r["entropy"]    = sc["entropy"]
                r["assignment"] = sc["assignment"]

                # Primary: minimize cost; Secondary: maximize margin; Tertiary: minimize entropy; Quaternary: minimize BIC
                key = ( _safe(r["cost"],    bad_for_min=math.inf),
                       -_safe(r["margin"],  bad_for_min=-math.inf, bad_for_max=-math.inf),
                        _safe(r["entropy"], bad_for_min=math.inf),
                        _safe(r.get("bic"), bad_for_min=math.inf) )

            if best is None or key < best_key:
                best, best_key = r, key
        return best

    if criterion in ("pm", "match", "perfmatch"):
        # Higher perfect-matching accuracy is better; if missing/NaN, treat as -inf
        return max(
            results,
            key=lambda r: _safe(r.get("pm_acc"), bad_for_min=-math.inf, bad_for_max=-math.inf)
        )

    raise ValueError(f"Unknown selection criterion: {criterion}")

# ------------------------- Main algorithm (revised) --------------------------

def run_main_algo(
    model_copy: nn.Module,
    source_model: nn.Module,
    src_trainset,
    tgt_trainset,
    all_sets,
    deg_idx,
    generated_domains: int,
    epochs: int = 3,
    target: int = 60,
    args=None,
    gen_method: str = "fr",
    # multi-EM controls (unchanged defaults)
    use_multi_em: bool = False,
    em_cov_types: Tuple[str, ...] = ("diag"),
    em_K_list: Optional[List[int]] = None,
    em_seeds: Tuple[int, ...] = (0, 1, 2, 3, 4),
    em_pca_dims: Tuple[Optional[int], ...] = (None,),
    em_select: str = "bic",
    em_ensemble_weights: str = "bic",
    # NEW: sharing controls
    use_shared_em: bool = True,                 # ← enable reuse by default
    shared_em_cfg: dict = None,                 # ← knobs for one-time EM fit
):
    """Teacher self-training along generated domains in embedding space, with shared-EM reuse."""
    device = next(source_model.parameters()).device

    # 0) Sanity: identical initialization?
    for p1, p2 in zip(model_copy.parameters(), source_model.parameters()):
        if not torch.equal(p1, p2):
            print("[run_main_algo] Warning: model_copy and source_model have different initial weights.")
            break

    # 1) Baselines
    set_all_seeds(args.seed)
    direct_acc, st_acc, train_acc_by_domain0, test_acc_by_domain0, _ = self_train(
        args, model_copy, [tgt_trainset], epochs=epochs, label_source="pseudo")
    set_all_seeds(args.seed)
    direct_acc_all, st_acc_all, train_acc_list_all, test_acc_list_all, _ = self_train(
        args, source_model, all_sets, epochs=epochs, label_source="pseudo")
    if abs(st_acc - st_acc_all) > 1e-4:
        print(f"[run_main_algo] Warning: st_acc ({st_acc}) != st_acc_all ({st_acc_all})")

    # 2) Teacher pseudo-labels on target (cached for mapping)
    em_teacher = copy.deepcopy(source_model).to(device).eval()
    with torch.no_grad():
        pseudo_labels, _ = get_pseudo_labels(
            tgt_trainset, em_teacher,
            confidence_q=getattr(args, "pseudo_confidence_q", 0.9),
            device_override=device,
        )
    pseudolabels = pseudo_labels.cpu().numpy()
    setattr(args, "_cached_pseudolabels", pseudolabels)

    # 3) Cache dirs
    if args.dataset != "mnist":
        cache_dir = f"{args.dataset}/cache{args.ssl_weight}/small_dim{args.small_dim}"
        plot_dir  = f"plots/{args.dataset}/"
    else:
        cache_dir = f"cache{args.ssl_weight}/target{target}/small_dim{args.small_dim}/"
        plot_dir  = f"plots/target{target}/"
    os.makedirs(cache_dir, exist_ok=True); os.makedirs(plot_dir, exist_ok=True)

    # 4) Encode domains once (encoder → flatten → compressor)
    e_src, e_tgt, encoded_intersets = encode_all_domains(
        src_trainset, tgt_trainset, all_sets, deg_idx,
        nn.Sequential(
            source_model.encoder,
            nn.Flatten(start_dim=1),
            source_model.compressor if hasattr(source_model, 'compressor') else nn.Identity()
        ),
        cache_dir, target, force_recompute=False, args=args
    )

    # 5) Pseudo-label accuracy diagnostic
    y_true = e_tgt.targets
    if torch.is_tensor(y_true): y_true = y_true.cpu()
    preds = torch.as_tensor(pseudolabels, device=y_true.device)
    acc_pl = (preds == y_true).float().mean().item()
    print(f"Pseudo-label accuracy: {acc_pl:.4f}")

    # 6) Determine K
    try:
        K_infer = int(max(int(preds.max().item()), int(y_true.max().item()))) + 1
    except Exception:
        K_infer = 10
    if not em_K_list:
        em_K_list = [K_infer]

    # 7) (optional) source stats for prototype mapping
    if getattr(args, "em_match", "pseudo") == "prototypes":
        mu_s, Sigma_s, priors_s = fit_source_gaussian_params(X=e_src.data, y=e_src.targets)
        setattr(args, "_cached_source_stats", (mu_s, Sigma_s, priors_s))

    # =========================
    # 8) SHARED EM: reuse or fit once
    # =========================
    # If a shared EM bundle is already present, reuse it.
    if args.em_match != "none":
        if use_shared_em and hasattr(args, "_shared_em") and args._shared_em is not None:
            bundle = args._shared_em
            em_res_ref = bundle.em_res
            mapping_ref = bundle.mapping
            labels_mapped = bundle.labels_em
            P_soft = bundle.P_soft
            print("[MainAlgo] Using shared EM bundle from cache.")
        else:
            # Otherwise fit once (single-EM or multi-EM), then store on args for others to reuse.
            start_time = time.time()
            if not use_multi_em:
                # Allow one-time EM config via shared_em_cfg
                cfg = dict(
                    K=K_infer, cov_type="diag", pool="gap",
                    do_pca=False, pca_dim=None, reg=1e-4, max_iter=500,
                    return_transforms=True, verbose=True,
                    subsample_init=min(10000, len(e_tgt)), rng=args.seed
                )
                if shared_em_cfg: cfg.update(shared_em_cfg)
                print(f"[MainAlgo] Running single EM: K={cfg['K']}, cov='{cfg['cov_type']}', pca_dim={cfg['pca_dim']}")
                em_res_ref = run_em_on_encoded(e_tgt, **cfg)
                # Map clusters → classes
                if getattr(args, "em_match", "pseudo") == "prototypes":
                    mapping_ref, labels_mapped, _ = map_em_clusters(
                        em_res_ref, method="prototypes", n_classes=cfg['K'],
                        mus_s=mu_s, Sigma_s=Sigma_s, priors_s=priors_s
                    )
                else:
                    mapping_ref, labels_mapped, _ = map_em_clusters(
                        em_res_ref, method="pseudo", n_classes=cfg['K'],
                        pseudo_labels=pseudolabels, metric='FR'
                    )
                P_soft = _map_cluster_posts_to_classes(np.asarray(em_res_ref["gamma"]), mapping_ref, cfg['K'])

                # Stash shared bundle
                args._shared_em = EMBundle(
                    key="__inline__", em_res=em_res_ref, mapping=mapping_ref,
                    labels_em=np.asarray(labels_mapped, dtype=int), P_soft=P_soft,
                    info={"K": cfg['K'], "cov_type": cfg['cov_type'], "pca_dim": cfg['pca_dim']}
                )
            else:
            #     print(f"[MainAlgo] Running multi-EM grid: K={em_K_list}, cov={em_cov_types}, seeds={em_seeds}, pca={em_pca_dims}")
            #     em_models = fit_many_em_on_target(
            #         e_tgt, K_list=list(em_K_list), cov_types=list(em_cov_types),
            #         seeds=list(em_seeds), pool="gap", pca_dims=list(em_pca_dims),
            #         reg=1e-4, max_iter=300, rng_base=args.seed, args=args,
            #     )
            #     best_em = _select_best_em(em_models, criterion=em_select)
            #     print(f"[MainAlgo] Best EM by {em_select}: cfg={best_em['cfg']}, final_ll={best_em.get('final_ll')}, bic={best_em.get('bic')}")
            #     em_res_ref = best_em["em_res"]
            #     labels_mapped = best_em["labels_mapped"]
            #     # For completeness, keep mapping if provided; otherwise reconstruct from soft posts
            #     # Here we reconstruct soft class posteriors from mapping/labels
            #     # (fit_many_em_on_target already produced mapped_soft)
            #     P_soft = best_em.get("mapped_soft", None)
            #     mapping_ref = best_em.get("mapping", None)
            #     # Stash shared bundle
            #     args._shared_em = EMBundle(
            #         key="__inline__", em_res=em_res_ref, mapping=mapping_ref,
            #         labels_em=np.asarray(labels_mapped, dtype=int), P_soft=P_soft,
            #         info={"K": best_em["cfg"]["K"], "cov_type": best_em["cfg"]["cov_type"], "pca_dim": best_em["cfg"]["pca_dim"]}
            #     )

                print(f"[MainAlgo] Running multi-EM grid: K={em_K_list}, cov={em_cov_types}, seeds={em_seeds}, pca={em_pca_dims}")
                em_models = fit_many_em_on_target(
                    e_tgt,
                    K_list=list(em_K_list), cov_types=list(em_cov_types),
                    seeds=list(em_seeds), pool="gap", pca_dims=list(em_pca_dims),
                    reg=1e-4, max_iter=300, rng_base=args.seed, args=args,
                )

                # Utility: soft weights from BIC and optional trimming
                def _bic_soft_weights(models, temperature=1.0, max_delta_bic=None):
                    bics = np.array([m.get("bic", np.inf) for m in models], dtype=float)
                    b0 = np.min(bics)
                    if max_delta_bic is not None:
                        keep = (bics - b0) <= float(max_delta_bic)
                        models = [m for m, k in zip(models, keep) if k]
                        bics = bics[keep]
                    logits = -(bics - np.min(bics)) / max(1e-12, 2.0 * float(temperature))
                    w = np.exp(logits - np.max(logits)); w /= w.sum()
                    return models, w

                def _recompute_soft_if_needed(m, X_embed):
                    """
                    Return soft class posteriors aligned to ground-truth class space.
                    Prefer m['mapped_soft']; else rebuild from ('weights','means','covs') + 'mapping'.
                    """
                    if m.get("mapped_soft", None) is not None:
                        return np.asarray(m["mapped_soft"], dtype=float)
                    # Fall back: raw component posteriors → mapped to classes via 'mapping'
                    if not ({"weights","means","covs"} <= set(m.keys())):
                        raise ValueError("EM model lacks both 'mapped_soft' and raw params needed to rebuild posteriors.")
                    pi = np.asarray(m["weights"], dtype=float)           # (K_m,)
                    mu = np.asarray(m["means"], dtype=float)             # (K_m,d)
                    S  = np.asarray(m["covs"], dtype=float)              # (K_m,d,d)
                    # responsibilities over components
                    R_comp = _predict_gmm_responsibilities(pi, mu, S, X_embed)   # (N,K_m)
                    # map components → classes using provided 'mapping' (component -> class id)
                    mapping = np.asarray(m.get("mapping", None))
                    if mapping is None:
                        raise ValueError("Need 'mapping' to align components to classes for posterior stacking.")
                    n_classes = int(mapping.max()) + 1
                    R_cls = np.zeros((R_comp.shape[0], n_classes), dtype=float)
                    for k_m, c in enumerate(mapping):
                        R_cls[:, c] += R_comp[:, k_m]
                    # renormalize rows (numerical safety)
                    R_cls /= np.clip(R_cls.sum(axis=1, keepdims=True), 1e-12, None)
                    return R_cls

                def _predict_gmm_responsibilities(pi, mu, covs, X):
                    # numerically robust Gaussian mixture responsibilities (NumPy)
                    def logpdf_gauss(X, m, S):
                        # symmetrize + safe chol
                        S = 0.5 * (S + S.T)
                        try:
                            L = np.linalg.cholesky(S)
                        except np.linalg.LinAlgError:
                            w, V = np.linalg.eigh(S)
                            w = np.maximum(w, 1e-12)
                            S = (V * w) @ V.T
                            L = np.linalg.cholesky(S)
                        logdet = 2.0 * np.log(np.diag(L)).sum()
                        Z = np.linalg.solve(L, (X - m).T)      # (d,N)
                        quad = np.sum(Z * Z, axis=0)
                        d = X.shape[1]
                        return -0.5 * (d * np.log(2*np.pi) + logdet + quad)

                    N, d = X.shape
                    K = len(pi)
                    log_comp = np.empty((N, K), dtype=float)
                    for k in range(K):
                        log_comp[:, k] = np.log(max(pi[k], 1e-300)) + logpdf_gauss(X, mu[k], covs[k])
                    m = np.max(log_comp, axis=1, keepdims=True)
                    Z = m + np.log(np.exp(log_comp - m).sum(axis=1, keepdims=True))
                    R = np.exp(log_comp - Z)
                    return R  # (N,K)
                em_time = time.time() - start_time
                print(f"[MainAlgo] EM fitting completed in {em_time:.2f} seconds")
                # ----- EM grid over target embedding -----
                # Decide selection vs ensemble
                em_mode = getattr(args, "em_mode", "ensemble")        # 'best' | 'ensemble'
                ensemble_temperature = float(getattr(args, "em_ens_temperature", 1.0))
                trim_delta = float(getattr(args, "em_trim_delta_bic", 10.0)) if getattr(args, "em_do_ensemble", True) else None

                if em_mode == "best":
                    best_em = _select_best_em(em_models, criterion=em_select)
                    print(f"[MainAlgo] Best EM by {em_select}: cfg={best_em['cfg']}, final_ll={best_em.get('final_ll')}, bic={best_em.get('bic')}")
                    em_res_ref = best_em["em_res"]
                    labels_mapped = np.asarray(best_em["labels_mapped"], dtype=int)
                    P_soft = best_em.get("mapped_soft", None)
                    mapping_ref = best_em.get("mapping", None)
                    args._shared_em = EMBundle(
                        key="__inline__", em_res=em_res_ref, mapping=mapping_ref,
                        labels_em=labels_mapped, P_soft=P_soft,
                        info={"mode":"best", "K": best_em["cfg"]["K"], "cov_type": best_em["cfg"]["cov_type"], "pca_dim": best_em["cfg"]["pca_dim"]}
                    )

                else:
                    # ----- Ensemble: posterior stacking with BIC-soft weights -----
                    # Optional trimming (keeps near-best by BIC), then softmax weights
                    models_kept, w = _bic_soft_weights(
                        em_models, temperature=ensemble_temperature,
                        max_delta_bic=trim_delta if getattr(args, "em_do_ensemble", True) else None
                    )
                    print(f"[MainAlgo] EM ensemble: kept={len(models_kept)} of {len(em_models)}; weights={np.round(w,4).tolist()}")

                    # Build stacked posteriors in class space
                    X_embed = np.asarray(e_tgt.data.cpu(), dtype=float)
                    Ps = []
                    for m in models_kept:
                        Ps.append(_recompute_soft_if_needed(m, X_embed))  # each (N, C)
                    Ps = np.stack(Ps, axis=0)                              # (M, N, C)
                    # weighted average over models
                    P_ens = np.tensordot(w, Ps, axes=(0,0))               # (N, C)
                    P_ens /= np.clip(P_ens.sum(axis=1, keepdims=True), 1e-12, None)
                    labels_ens = P_ens.argmax(axis=1).astype(int)

                    # Use the BIC-best EM as a reference carrier of params/mapping in the bundle
                    best_em = min(models_kept, key=lambda m: m.get("bic", np.inf))
                    args._shared_em = EMBundle(
                        key="multi_em_ensemble",
                        em_res=best_em["em_res"],                         # reference (e.g., for metadata)
                        mapping=best_em.get("mapping", None),
                        labels_em=labels_ens, P_soft=P_ens,
                        info={
                            "mode": "ensemble",
                            "K_ref": best_em["cfg"]["K"],
                            "cov_type_ref": best_em["cfg"]["cov_type"],
                            "pca_dim_ref": best_em["cfg"]["pca_dim"],
                            "kept": len(models_kept),
                            "weights": w.tolist(),
                            "trim_delta_bic": float(trim_delta) if trim_delta is not None else None
                        }
                    )
                    print(f"[MainAlgo] Ensemble EM ready: kept={len(models_kept)}; reference cfg={best_em['cfg']}")

        # 9) Diagnostics against GT (best one-to-one)

        y_true_np = np.asarray(y_true)
        best_acc, _best_map, _C = best_mapping_accuracy(em_res_ref["labels"], y_true_np)
        print(f"[MainAlgo] Best one-to-one mapping accuracy: {best_acc:.4f}, current em accuracy: {(np.asarray(labels_mapped) == y_true_np).mean():.4f}")


        # 10) Attach EM labels/soft targets to the target dataset (shared for all methods)
        e_tgt.targets_em = torch.as_tensor(labels_mapped, dtype=torch.long)
        

    e_tgt.targets_pseudo = torch.as_tensor(pseudolabels, dtype=torch.long)
    tgt_trainset.targets_em = e_tgt.targets_em.cpu().clone()
    tgt_trainset.targets_pseudo = e_tgt.targets_pseudo.cpu().clone()
    if P_soft is not None:
        tgt_trainset.soft_targets = torch.from_numpy(P_soft).float()

    acc_em_pseudo = float((np.asarray(labels_mapped) == np.asarray(y_true)).mean())
    # print(f"[MainAlgo] EM→class (mapped) accuracy: {acc_em_pseudo:.4f}")

    # 11) Evaluate updated model on target images
    tgt_loader_eval = DataLoader(tgt_trainset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    _, tgt_acc_after = test(tgt_loader_eval, source_model)

    # 12) Generate synthetic domains (unchanged)
    if generated_domains <= 0:
        return direct_acc, st_acc, direct_acc_all, st_acc_all, 0.0, acc_em_pseudo

    synthetic_domains: List = []
    _method = (gen_method or "fr").lower()
    if _method in {"fr", "fisher-rao", "fisher_rao"}:
        _gen_fn = generate_fr_domains_between_optimized
    elif _method in {"natural", "natureal", "eta", "nat", "np"}:
        _gen_fn = generate_natural_domains_between
    else:
        raise ValueError(f"Unknown gen_method '{gen_method}'. Use 'fr' or 'natural'.")

    for i in range(len(encoded_intersets) - 1):
        out = _gen_fn(
            generated_domains, encoded_intersets[i], encoded_intersets[i + 1],
            cov_type="full", save_path=plot_dir, args=args,
        )
        pair_domains = out[0]; domain_stats = out[2]
        sizes_by_class, sizes_c1 = covariance_sizes(domain_stats, cls=1)
        print("Class 1 trace per domain:", sizes_c1["trace"])
        print("Class 1 logdet per domain:", sizes_c1["logdet"])
        synthetic_domains += pair_domains

    if not synthetic_domains:
        return direct_acc, st_acc, direct_acc_all, st_acc_all, 0.0, acc_em_pseudo

    # 13) Self-training on synthetic domains
    set_all_seeds(args.seed)
    direct_acc_syn, generated_acc, train_acc_by_domain, test_acc_by_domain, last_predictions = self_train(
        args, source_model.mlp, synthetic_domains, epochs=epochs,
        label_source=getattr(args, "label_source", "em"),
    )

    # 14) Plots (unchanged)
    plot_pca_classes_grid(
        encoded_intersets,
        classes=(3, 6, 8, 9) if "mnist" in args.dataset else (0, 1),
        save_path=os.path.join(plot_dir, f"pca_dim{args.small_dim}_gen{args.generated_domains}_real_domains.png"),
        label_source='real', ground_truths=True, pca=getattr(args, "shared_pca", None)
    )
    if synthetic_domains:
        chain = []
        step_len = int(generated_domains) + 1
        for k in range(0, len(synthetic_domains), max(step_len, 1)):
            chunk = synthetic_domains[k:k + step_len]
            if chunk:
                chain.extend(chunk[:-1] if step_len > 0 else chunk)
        chain_for_plot = [encoded_intersets[0]] + chain + [encoded_intersets[-1]]

        plot_pca_classes_grid(
            chain_for_plot,
            classes=(3, 6, 8, 9) if "mnist" in args.dataset else (0, 1),
            save_path=os.path.join(plot_dir, f"pca_dim{args.small_dim}_gen{args.generated_domains}_{args.label_source}_{getattr(args,'em_match','pseudo')}_{_method}.png"),
            label_source=getattr(args, "label_source", "em"),
            pseudolabels=last_predictions,
            pca=getattr(args, "shared_pca", None)
        )
        plot_pca_classes_grid(
            chain_for_plot,
            classes=(3, 6, 8, 9) if "mnist" in args.dataset else (0, 1),
            save_path=os.path.join(plot_dir, f"pca_dim{args.small_dim}_gen{args.generated_domains}_emlabels_{_method}.png"),
            label_source='em',
            pca=getattr(args, "shared_pca", None)
        )
        plot_pca_classes_grid(
            chain_for_plot,
            classes=(3, 6, 8, 9) if "mnist" in args.dataset else (0, 1),
            save_path=os.path.join(plot_dir, f"pca_dim{args.small_dim}_gen{args.generated_domains}_source_pseudo_{_method}.png"),
            label_source='pseudo',
            pseudolabels=pseudolabels,
            pca=getattr(args, "shared_pca", None)
        )

    return train_acc_by_domain, test_acc_by_domain, st_acc, st_acc_all, generated_acc, acc_em_pseudo


# ---------------- Generic plotting helper: N series + per-method baselines ----------------
def _plot_series_with_baselines(
    series,
    labels,
    baselines=None,  # list of (st, st_all) per series (optional)
    ref_line_value=None,
    ref_line_label=None,
    ref_line_style="--",
    title="",
    ylabel="Accuracy",
    xlabel="Training Domain Index",
    save_path=None,
):

    import matplotlib.pyplot as plt

    def _to_array(v):
        return np.array([np.nan if x is None else float(x) for x in (v or [])], dtype=float)

    S = [ _to_array(s) for s in (series or []) ]
    if not S:
        print(f"[plot] Skip {title}: no data.")
        return

    L = max(len(s) for s in S)
    if L == 0:
        print(f"[plot] Skip {title}: empty series.")
        return
    S = [ (np.pad(s, (0, L - len(s)), constant_values=np.nan) if len(s) < L else s) for s in S ]
    x = np.arange(0, L, dtype=int)

    plt.figure()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    markers = ['o', 's', '^', 'D', 'v', '>', '<', 'P', 'X']

    n = len(S)
    for i, s in enumerate(S):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        label  = labels[i] if labels and i < len(labels) else f"Series {i+1}"
        plt.plot(x, s, marker=marker, linewidth=1.8, label=label, color=color)

        if baselines and i < len(baselines) and baselines[i] is not None:
            st, st_all = baselines[i]
            if st is not None:
                plt.axhline(float(st), linestyle=':', linewidth=1.5, color=color, alpha=0.9, label=f"st")
            if st_all is not None:
                plt.axhline(float(st_all), linestyle='--', linewidth=1.5, color=color, alpha=0.9, label=f"st_all")

    if ref_line_value is not None:
        plt.axhline(float(ref_line_value), linestyle=ref_line_style, linewidth=1.6,
                    color='k', alpha=0.7, label=(ref_line_label or "reference"))

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[MNIST-EXP] Saved {save_path}")
    plt.close()


def plot_class_counts(domain_stats: dict, save_path: str):
    ts = _steps(domain_stats)
    K = int(domain_stats["K"])
    counts = np.asarray(domain_stats["counts"])  # (S,K)
    plt.figure(figsize=(6,3.5))
    for k in range(K):
        plt.plot(ts, counts[:,k], lw=1, alpha=0.8)
    plt.title("Per-class counts across steps")
    plt.xlabel("t"); plt.ylabel("count")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150); plt.close()
    print(f"[plot_class_counts] Saved {save_path}")
def adjacent_distances(domain_stats: dict, report_bures: bool = True, target_class: int = 9):
    ts = _steps(domain_stats)

    for s in range(len(ts) - 1):
        # masks: class must exist at both steps AND equal to target_class
        present = _present_mask(domain_stats, s) & _present_mask(domain_stats, s + 1)
        cls_mask = np.zeros_like(present, dtype=bool)
        if target_class < len(cls_mask):
            cls_mask[target_class] = True
        present = present & cls_mask
        if not present.any():
            print(f"[adjacent] t={ts[s]:.2f}→{ts[s+1]:.2f} : class {target_class} not present at both steps")
            continue

        mu_a = _get_mu(domain_stats, s)
        mu_b = _get_mu(domain_stats, s + 1)
        d_mu = np.linalg.norm((mu_b - mu_a)[present], ord='fro')  # single class ⇒ scalar

        if not _has_full(domain_stats):
            v_a = _get_var(domain_stats, s)[present]
            v_b = _get_var(domain_stats, s + 1)[present]
            d_logv = float(np.mean(np.abs(
                np.log(np.clip(v_b, 1e-12, None)) - np.log(np.clip(v_a, 1e-12, None))
            )))
            print(f"[adjacent:c{target_class}] t={ts[s]:.2f}→{ts[s+1]:.2f} : ||Δμ||={d_mu:.4g}, |Δ logvar|={d_logv:.4g}")
        else:
            Sig_a = _get_sigma(domain_stats, s)[present][0]
            Sig_b = _get_sigma(domain_stats, s + 1)[present][0]

            ld_a = _logdet_cov(Sig_a[None, ...])[0]
            ld_b = _logdet_cov(Sig_b[None, ...])[0]
            d_ld = float(abs(ld_b - ld_a))

            line = (f"[adjacent:c{target_class}] t={ts[s]:.2f}→{ts[s+1]:.2f} : "
                    f"||Δμ||={d_mu:.4g}, |Δ logdet|={d_ld:.4g}")

            if report_bures:
                def bures2(A, B):
                    wA, VA = np.linalg.eigh(A); wA = np.clip(wA, 0, None)
                    As = (VA * np.sqrt(wA)) @ VA.T
                    M = As @ B @ As
                    wM, VM = np.linalg.eigh(M); wM = np.clip(wM, 0, None)
                    Ms = (VM * np.sqrt(wM)) @ VM.T
                    return float(np.trace(A) + np.trace(B) - 2.0 * np.trace(Ms))

                b2 = bures2(Sig_a, Sig_b)
                line += f", Bures^2={b2:.4g}"

            print(line)

def summarize_domain_stats(domain_stats: dict, target_class: int = None):
    """
    Print a short audit of the domain parameters across steps.

    If `target_class` is None (default), report averages across classes present
    at each step. If an integer is provided (e.g., 9), report ONLY that class.

    Columns:
      - trace(Σ): sum of variances (or matrix trace for full Σ)
      - logdet(Σ): sum of log eigenvalues (diag → sum log variances)
      - geo-var : exp(logdet(Σ)/d)  (geometric mean variance per dimension)
    """


    def _logdet_cov_per_class(Sig, eps=1e-12):
        """Per-class logdet vector (shape (K,)). Works for full or diag Σ."""
        Sig = np.asarray(Sig)
        if Sig.ndim == 3:  # full: (K,d,d)
            K = Sig.shape[0]
            out = np.empty(K, dtype=np.float64)
            for k in range(K):
                w = np.linalg.eigvalsh(Sig[k])
                w = np.clip(w, eps, None)
                out[k] = float(np.sum(np.log(w)))
            return out
        elif Sig.ndim == 2:  # diag: (K,d)
            w = np.clip(Sig, eps, None)
            return np.sum(np.log(w), axis=1).astype(np.float64)
        else:
            raise ValueError(f"_logdet_cov_per_class: unexpected Sig.ndim={Sig.ndim}")

    S = len(domain_stats["steps"])
    K = int(domain_stats["K"])
    d = int(domain_stats["d"])
    cov_type = "full" if _has_full(domain_stats) else "diag"
    hdr = f"[audit:{cov_type.upper()}] S={S}, K={K}, d={d}"
    if target_class is not None:
        hdr += f" (class {target_class})"
    print(hdr)

    for s, t in enumerate(_steps(domain_stats)):
        present_mask = _present_mask(domain_stats, s)

        Sig = _get_sigma(domain_stats, s)      # (K,d,d) or (K,d)
        tr_all = _trace_cov(Sig)               # (K,)
        try:
            logdet_all = _logdet_cov_per_class(Sig)  # (K,)
        except Exception as e:
            print(f"[audit] Step {s} logdet computation failed: {e}")
            logdet_all = np.full(K, np.nan, dtype=float)

        if target_class is None:
            # aggregate over classes that are present at this step
            cnt_sum = int(np.asarray(domain_stats["counts"][s])[present_mask].sum())
            tr_avg = float(np.nanmean(tr_all[present_mask])) if present_mask.any() else float("nan")
            ld_avg = float(np.nanmean(logdet_all[present_mask])) if present_mask.any() else float("nan")
            geo_var = float(np.exp(ld_avg / max(d, 1))) if np.isfinite(ld_avg) else float("nan")

            print(
                f"  step {s} (t={t:.2f}): present={present_mask.sum()}/{K}, "
                f"sum_count={cnt_sum}, avg trace(Σ)={tr_avg:.6g}, "
                f"avg logdet(Σ)={ld_avg:.6g}, avg geo-var={geo_var:.6g}"
            )
        else:
            # single-class report
            if not (0 <= target_class < K) or not present_mask[target_class]:
                print(f"  step {s} (t={t:.2f}): class {target_class} not present")
                continue
            tr_k = float(tr_all[target_class])
            ld_k = float(logdet_all[target_class])
            geo_var_k = float(np.exp(ld_k / max(d, 1))) if np.isfinite(ld_k) else float("nan")
            cnt_k = int(np.asarray(domain_stats["counts"][s])[target_class])

            print(
                f"  step {s} (t={t:.2f}): class {target_class}, count={cnt_k}, "
                f"trace(Σ)={tr_k:.6g}, logdet(Σ)={ld_k:.6g}, geo-var={geo_var_k:.6g}"
            )

def check_means_between(domain_stats: dict, target_class: int = None):
    """
    Verify intermediate means lie within the endpoint segment per class (component-wise).

    If target_class is None → check all classes and print a summary.
    If target_class is an int → only check that class and print per-step status.
    """
    ts  = _steps(domain_stats)
    mu0 = _get_mu(domain_stats, 0)     # (K,d)
    muT = _get_mu(domain_stats, -1)    # (K,d)

    lb = np.minimum(mu0, muT)          # (K,d)
    ub = np.maximum(mu0, muT)

    if target_class is not None:
        k = int(target_class)
        ok = True
        for s in range(1, len(ts)-1):
            mus = _get_mu(domain_stats, s)       # (K,d)
            present = _present_mask(domain_stats, s)
            if present[k]:
                inside = np.all((mus[k] >= lb[k]) & (mus[k] <= ub[k]))
                print(f"[check_means_between] step {s} (t={ts[s]:.2f}) "
                      f"class {k}: {'inside' if inside else 'OUTSIDE'}")
                ok = ok and inside
        if ok:
            print(f"[check_means_between] class {k}: all intermediate means inside segment.")
        else:
            print(f"[check_means_between] class {k}: some steps outside endpoint box.")
        return

    # all classes
    ok_all = True
    offenders = []
    for s in range(1, len(ts)-1):
        mus = _get_mu(domain_stats, s)           # (K,d)
        present = _present_mask(domain_stats, s)
        inside = np.all((mus >= lb) & (mus <= ub), axis=1)   # (K,)
        bad = present & (~inside)
        if bad.any():
            ok_all = False
            offenders.append((s, ts[s], np.where(bad)[0]))
    if ok_all:
        print("[check_means_between] All intermediate means lie within endpoint segments.")
    else:
        for s, t, ks in offenders[:5]:
            print(f"[check_means_between] step {s} (t={t:.2f}) outside for classes {ks.tolist()}")
        if len(offenders) > 5:
            print(f"[check_means_between] (+{len(offenders)-5} more offending steps)")


def plot_size_proxy(
    domain_stats: dict,
    mode: str = "trace",           # {'trace', 'logdet'}
    save_path: str = None,
    target_class: int = None,      # None = average over present classes; int = plot that class only
):
    """
    Plot a size proxy across interpolation steps.

    - 'trace'  → tr(Σ) (sum of variances)
    - 'logdet' → log det(Σ) (sum of log eigenvalues for full; sum log variances for diag)
    Works for both diag and full covariances stored in `domain_stats`.
    """
    import os

    import matplotlib.pyplot as plt

    def _logdet_cov_per_class(Sig, eps=1e-12):
        Sig = np.asarray(Sig)
        if Sig.ndim == 3:  # full (K,d,d)
            out = np.empty(Sig.shape[0], dtype=np.float64)
            for k in range(Sig.shape[0]):
                w = np.linalg.eigvalsh(Sig[k])
                w = np.clip(w, eps, None)
                out[k] = np.sum(np.log(w))
            return out
        elif Sig.ndim == 2:  # diag (K,d)
            w = np.clip(Sig, eps, None)
            return np.sum(np.log(w), axis=1).astype(np.float64)
        else:
            raise ValueError(f"_logdet_cov_per_class: unexpected Sig.ndim={Sig.ndim}")

    ts = _steps(domain_stats)
    K  = int(domain_stats["K"])

    yvals = []
    for s, t in enumerate(ts):
        present = _present_mask(domain_stats, s)          # (K,)
        Sig = _get_sigma(domain_stats, s)                 # (K,d,d) or (K,d)

        if mode == "trace":
            per_class = _trace_cov(Sig)                   # (K,)
        elif mode == "logdet":
            per_class = _logdet_cov_per_class(Sig)        # (K,)
        else:
            raise ValueError("mode must be 'trace' or 'logdet'")

        if target_class is None:
            m = present
            val = float(np.nanmean(per_class[m])) if m.any() else np.nan
        else:
            k = int(target_class)
            if 0 <= k < K and present[k]:
                val = float(per_class[k])
            else:
                val = np.nan
        yvals.append(val)

    plt.figure(figsize=(5.6, 3.2))
    label = f"{mode} (avg over classes)" if target_class is None else f"{mode} (class {target_class})"
    plt.plot(ts, yvals, marker='o', label=label)
    plt.title(f"Evolution of {mode}")
    plt.xlabel("t"); plt.ylabel(mode)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"[plot_size_proxy] Saved {save_path}")
    plt.close()

@torch.no_grad()
def plot_source_real_vs_parametric(
    source_ds, domain_stats: dict, classes=None, n_per_class="match",
    pool: str = "auto", seed: int = 0, save_path: str = None, step: int = 0
):
    # If full Σ is present, call the 3-panel function and show only REAL vs FULL;
    # else fall back to REAL vs DIAG.
    if _has_full(domain_stats):
        plot_real_vs_fullcov_vs_diag_samples(
            source_ds=source_ds,
            domain_stats=domain_stats,
            classes=classes,
            n_per_class=n_per_class,
            pool=pool,
            pca_dim=2,
            seed=seed,
            save_path=save_path,
            step=step,
        )
    else:
        # your existing 2-panel (REAL vs DIAG) implementation works as-is
        # (keep your previous code here, or reuse the 3-panel and hide the middle)
        plot_real_vs_fullcov_vs_diag_samples(
            source_ds=source_ds,
            domain_stats=domain_stats,
            classes=classes,
            n_per_class=n_per_class,
            pool=pool,
            pca_dim=2,
            seed=seed,
            save_path=save_path,
            step=step,
        )



import torch
from sklearn.decomposition import PCA

@torch.no_grad()
def print_class_trace_ratios(
    source_ds,
    domain_stats: dict,      # dict with keys: "var", "mu", "counts"
    pool: str = "auto",      # "gap", "flatten", "auto"
    ddof: int = 0,           # 0 → population covariance, matches class_stats_diag
    pca_dim: int = 2,        # None to skip PCA diagnostics
    seed: int = 0,
):
    """
    For each class c:
      - Compare trace of *full empirical covariance* (real) vs trace of diag model var0[c]
      - If pca_dim is given, compute in a shared PCA subspace:
          * eigenvalues (real vs diag), anisotropy λmax/λmin,
          * KL(N(0, Σ_real) || N(0, Σ_diag)),
          * Bures^2(Σ_real, Σ_diag).
    """

    from sklearn.decomposition import PCA


    # ---------- helpers ----------
    def _to_np(a):
        return a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)

    def full_cov(Xc: np.ndarray) -> np.ndarray:
        """Full empirical covariance with ddof."""
        Xc = np.asarray(Xc, dtype=np.float64)
        if Xc.shape[0] <= ddof:
            # degenerate; return tiny diagonal to avoid NaNs
            return 1e-12 * np.eye(Xc.shape[1], dtype=np.float64)
        mu = Xc.mean(axis=0, keepdims=True)
        Z  = Xc - mu
        return (Z.T @ Z) / float(max(1, (Xc.shape[0] - ddof)))

    def proj_full_cov(Xc, P):
        """Σ_proj = P Σ_full P^T using the full empirical covariance of Xc."""
        return P @ full_cov(Xc) @ P.T

    def eigh_sorted(A):
        w, V = np.linalg.eigh(A)
        return np.maximum(w, 0.0), V  # clip tiny negatives

    def mat_sqrt(A):
        w, V = np.linalg.eigh(A)
        w = np.clip(w, 0.0, None)
        return (V * np.sqrt(w)) @ V.T

    def bures_distance(A, B):
        # d_B^2(A,B) = tr(A) + tr(B) - 2 tr( (A^{1/2} B A^{1/2})^{1/2} )
        As = mat_sqrt(A)
        M = As @ B @ As
        Ms = mat_sqrt(M)
        return float(np.trace(A) + np.trace(B) - 2.0 * np.trace(Ms))

    def kl_gaussians_zero_mean(Sig_real, Sig_diag):
        # KL(N(0,Σ_r) || N(0,Σ_d)) = 0.5( tr(Σ_d^{-1} Σ_r) - log det(Σ_d^{-1} Σ_r) - k )
        k = Sig_real.shape[0]
        w_d, V_d = np.linalg.eigh(Sig_diag)
        w_d = np.clip(w_d, 1e-12, None)
        Sig_d_inv = (V_d * (1.0 / w_d)) @ V_d.T
        A = Sig_d_inv @ Sig_real
        w_a = np.linalg.eigvalsh(A)
        w_a = np.clip(w_a, 1e-12, None)
        return float(0.5 * (np.trace(A) - np.sum(np.log(w_a)) - k))

    # ---------- pull source features/labels ----------
    X = source_ds.data
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()

    # pool/flatten if needed
    if X.ndim == 4:
        if pool == "gap" or pool == "auto":
            X = X.mean(axis=(2, 3))
        else:
            X = X.reshape(X.shape[0], -1)
    elif X.ndim == 3:
        X = X.reshape(X.shape[0], -1)

    y = getattr(source_ds, "targets", None)
    if y is None:
        y = getattr(source_ds, "targets_em", None)
    y = _to_np(y)

    # ---------- stats from domain_stats (step 0 = source) ----------
    var0   = np.asarray(domain_stats["var"][0], dtype=np.float64)     # (K, d) diagonal model
    counts = np.asarray(domain_stats["counts"][0], dtype=np.int64)    # (K,)
    K, d   = var0.shape

    # ---------- optional shared PCA ----------
    P = None
    if pca_dim is not None and pca_dim > 0 and d > pca_dim:
        pca = PCA(n_components=pca_dim, random_state=seed)
        pca.fit(X)
        P = pca.components_  # (p,d)
        p = P.shape[0]
    else:
        p = None

    # ---------- header ----------
    base_cols = "class | n_real | trace(real) | trace(diag) | ratio"
    if P is None:
        print("[trace-ratio]", base_cols)
    else:
        ext_cols = "eig(real) | eig(diag) | aniso(real) | aniso(diag) | KL(real‖diag) | Bures^2"
        print("[trace-ratio]", base_cols, "|", ext_cols)

    # ---------- per-class diagnostics ----------
    for c in range(K):
        m = (y == c)
        n_c = int(m.sum())
        if n_c < 2 or counts[c] <= 0:
            continue

        Xc = X[m]  # (n_c, d)

        # --- Real: FULL covariance (not just diagonal) ---
        Sigma_real = full_cov(Xc)              # (d, d)
        tr_real    = float(np.trace(Sigma_real))

        # --- Diagonal model from domain_stats ---
        v_diag = np.nan_to_num(var0[c], nan=0.0)  # (d,)
        tr_diag = float(np.sum(v_diag))
        ratio   = tr_real / (tr_diag + 1e-12)

        if P is None:
            print(f"[trace-ratio] {c:5d} | {n_c:6d} | {tr_real:11.4g} | {tr_diag:11.4g} | {ratio:6.3f}")
            continue

        # --- Projected quantities in PCA subspace ---
        Sig_real_p = P @ Sigma_real @ P.T                     # (p, p)
        Sig_diag_p = (P * v_diag[None, :]) @ P.T              # (p, p) == P diag(v) P^T

        # eigen diagnostics
        lam_r, _ = eigh_sorted(Sig_real_p)
        lam_d, _ = eigh_sorted(Sig_diag_p)
        aniso_r = float((lam_r[-1] / max(lam_r[0], 1e-12)) if p >= 2 else 1.0)
        aniso_d = float((lam_d[-1] / max(lam_d[0], 1e-12)) if p >= 2 else 1.0)

        # divergences (zero-mean assumption in PCA space)
        kl_rd   = kl_gaussians_zero_mean(Sig_real_p, Sig_diag_p)
        bures2  = bures_distance(Sig_real_p, Sig_diag_p)

        eig_r_str = "[" + ", ".join(f"{v:.3e}" for v in lam_r) + "]"
        eig_d_str = "[" + ", ".join(f"{v:.3e}" for v in lam_d) + "]"

        print(
            f"[trace-ratio] {c:5d} | {n_c:6d} | {tr_real:11.4g} | {tr_diag:11.4g} | {ratio:6.3f} | "
            f"{eig_r_str} | {eig_d_str} | {aniso_r:11.3f} | {aniso_d:11.3f} | {kl_rd:12.4g} | {bures2:9.4g}"
        )


@torch.no_grad()
def plot_real_vs_fullcov_vs_diag_samples(
    source_ds,
    domain_stats: dict,
    classes=None,                 # e.g. (3,6,8,9); None → all present at this step & in real data
    n_per_class="match",          # "match" → match real class counts; or an int (e.g. 500)
    pool: str = "auto",           # "gap", "flatten", or "auto"
    pca_dim: int = 2,             # PCA for visualization (shared across panels)
    seed: int = 0,
    save_path: str = None,
    step: int = 0,                # which step from domain_stats to visualize (0 = source)
    jitter: float = 1e-6,         # SPD safety when sampling from full Σ
):
    """
    Three panels, same PCA:
      (1) Real source embeddings of selected classes
      (2) Samples from full-covariance N(mu, Σ) if 'Sigma' is provided for this step
      (3) Samples from diagonal model N(mu, diag(var))

    domain_stats must have:
      - "mu"      : (S, K, d)
      - "var"     : (S, K, d)      (diagonal variances; NaN where class absent)
      - "counts"  : (S, K)
      - optional "Sigma": (S, K, d, d) for cov_type = 'full' (NaN where class absent)
    """

    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(seed)
    cmap = plt.get_cmap('tab10')

    # ---------- helpers ----------
    def _np(a):
        return a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)

    def _pool_features(X):
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        if X.ndim == 4:
            if pool == "gap" or pool == "auto":
                return X.mean(axis=(2, 3))
            return X.reshape(X.shape[0], -1)
        if X.ndim == 3:
            return X.reshape(X.shape[0], -1)
        return X

    def _sample_diag(mu, var, n):
        std = np.sqrt(np.clip(var, 1e-8, None))
        return rng.normal(loc=mu, scale=std, size=(n, mu.shape[0]))

    def _sample_full(mu, Sig, n):
        # Eig-based sampling with SPD clamp
        w, V = np.linalg.eigh(Sig)
        w = np.clip(w, 0.0, None) + jitter
        A = (V * np.sqrt(w)) @ V.T
        Z = rng.standard_normal(size=(n, mu.shape[0]))
        return Z @ A.T + mu

    # ---------- pull step parameters ----------
    mu_s     = _np(domain_stats["mu"][step])         # (K, d)
    var_s    = _np(domain_stats["var"][step])        # (K, d)
    counts_s = _np(domain_stats["counts"][step])     # (K,)
    Sigma_s  = None
    if "Sigma" in domain_stats and domain_stats["Sigma"] is not None:
        Sigma_s = _np(domain_stats["Sigma"][step])   # (K, d, d)

    K, d = mu_s.shape

    # ---------- real source features/labels ----------
    Xr = _pool_features(source_ds.data)
    yr = getattr(source_ds, "targets", None)
    if yr is None:
        yr = getattr(source_ds, "targets_em", None)
    yr = _np(yr)

    # ---------- pick classes ----------
    if classes is None:
        present = (counts_s > 0)
        present[np.isnan(counts_s)] = False
        uniq_real = np.unique(yr)
        mask_real = np.zeros(K, dtype=bool)
        mask_real[np.intersect1d(uniq_real, np.arange(K), assume_unique=False)] = True
        cls = np.where(present & mask_real)[0]
    else:
        cls = np.asarray(list(classes), dtype=int)

    if cls.size == 0:
        print("[plot] No classes to visualize at this step.")
        return

    # ---------- decide per-class sample counts ----------
    Ns = {}
    for c in cls:
        n_real_c = int((yr == c).sum())
        if n_real_c == 0: 
            continue
        if n_per_class == "match":
            Ns[c] = n_real_c
        else:
            Ns[c] = int(n_per_class)
        Ns[c] = max(0, min(Ns[c], n_real_c))  # cap by real availability

    if not Ns:
        print("[plot] No real points after selection.")
        return

    # ---------- gather real + synthetic ----------
    X_real_list, y_real_list = [], []
    X_full_list, y_full_list = [], []
    X_diag_list, y_diag_list = [], []

    for j, c in enumerate(cls):
        n_take = Ns.get(c, 0)
        if n_take <= 0:
            continue

        # real subset
        m = (yr == c)
        Xc = Xr[m]
        idx = rng.choice(Xc.shape[0], size=n_take, replace=False)
        X_real_list.append(Xc[idx])
        y_real_list.append(np.full(n_take, c, dtype=int))

        # parameters for class c
        mu_c  = mu_s[c]
        var_c = var_s[c]
        finite_diag = np.isfinite(mu_c).all() and np.isfinite(var_c).all() and counts_s[c] > 0

        # full-cov samples (if available & finite)
        if Sigma_s is not None:
            Sig_c = Sigma_s[c]
            finite_full = finite_diag and np.isfinite(Sig_c).all()
            if finite_full:
                try:
                    X_full_list.append(_sample_full(mu_c, Sig_c, n_take))
                    y_full_list.append(np.full(n_take, c, dtype=int))
                except np.linalg.LinAlgError:
                    # fallback to diag if covariance is too ill-conditioned
                    X_diag_list.append(_sample_diag(mu_c, var_c, n_take))
                    y_diag_list.append(np.full(n_take, c, dtype=int))

        # diag samples
        if finite_diag:
            X_diag_list.append(_sample_diag(mu_c, var_c, n_take))
            y_diag_list.append(np.full(n_take, c, dtype=int))

    if not X_real_list:
        print("[plot] No real points after selection.")
        return
    if not X_diag_list and not X_full_list:
        print("[plot] No synthetic samples could be drawn (diag/full unavailable).")
        return

    X_real = np.vstack(X_real_list)
    y_real = np.concatenate(y_real_list)

    X_full = np.vstack(X_full_list) if X_full_list else None
    y_full = np.concatenate(y_full_list) if y_full_list else None

    X_diag = np.vstack(X_diag_list) if X_diag_list else None
    y_diag = np.concatenate(y_diag_list) if y_diag_list else None

    # ---------- one shared PCA across everything available ----------
    fit_blocks = [X_real]
    if X_full is not None: fit_blocks.append(X_full)
    if X_diag is not None: fit_blocks.append(X_diag)
    X_all = np.vstack(fit_blocks)

    if (pca_dim is None) or (pca_dim <= 0):
        # fallback: first two dims
        def _proj(X):
            if X is None: return None
            if X.shape[1] >= 2: return X[:, :2]
            return np.pad(X, ((0,0),(0, 2-X.shape[1])), mode="constant")
        Z_real = _proj(X_real)
        Z_full = _proj(X_full) if X_full is not None else None
        Z_diag = _proj(X_diag) if X_diag is not None else None
    else:
        pca = PCA(n_components=pca_dim, random_state=seed)
        Z_all = pca.fit_transform(X_all)
        n_r = X_real.shape[0]
        off = n_r
        Z_real = Z_all[:n_r]
        Z_full = None
        Z_diag = None
        if X_full is not None:
            n_f = X_full.shape[0]
            Z_full = Z_all[off:off+n_f]
            off += n_f
        if X_diag is not None:
            n_d = X_diag.shape[0]
            Z_diag = Z_all[off:off+n_d]

    # ---------- plotting ----------
    n_cols = 3 if Z_full is not None else 2
    fig, axs = plt.subplots(1, n_cols, figsize=(4*n_cols + 2, 4), sharex=True, sharey=True)

    def _scatter(ax, Z, y, title):
        for jj, c in enumerate(cls):
            m = (y == c) if (y is not None) else None
            if Z is not None and m is not None and m.any():
                ax.scatter(Z[m, 0], Z[m, 1], s=6, alpha=0.7, color=cmap(jj % 10), label=f"{c}")
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])

    if n_cols == 3:
        _scatter(axs[0], Z_real, y_real, "Real")
        _scatter(axs[1], Z_full, y_full, "Full-Σ samples")
        _scatter(axs[2], Z_diag, y_diag, "Diag samples")
        axs[0].legend(loc="best", fontsize=8)
    else:
        _scatter(axs[0], Z_real, y_real, "Real")
        _scatter(axs[1], Z_diag, y_diag, "Diag samples")
        axs[0].legend(loc="best", fontsize=8)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"[plot] Saved {save_path}")
    plt.close()



def fit_global_pca(
    domains: Iterable,                        # iterable of datasets with .data and labels (targets or targets_em)
    classes: Optional[Sequence[int]] = None, # restrict to these classes if provided
    pool: str = "gap",                        # 'gap'|'flatten'|'auto'
    n_components: int = 2,
    per_domain_cap: Optional[int] = 10000,    # cap total samples per domain (None = no cap)
    random_state: int = 0,
) -> PCA:
    rng = np.random.default_rng(random_state)

    def _pool(X):
        if isinstance(X, torch.Tensor): X = X.detach().cpu().numpy()
        if X.ndim == 4:
            if pool in ("gap", "auto"): return X.mean(axis=(2, 3))
            elif pool == "flatten":     return X.reshape(X.shape[0], -1)
            else:                       return X.mean(axis=(2, 3))
        if X.ndim == 3: return X.reshape(X.shape[0], -1)
        return X

    X_all = []
    for j, D in enumerate(domains):
        Xp = _pool(D.data)
        # prefer EM, then real; adjust if you wish:
        y = getattr(D, "targets_em", None)
        if y is None: y = getattr(D, "targets", None)
        if y is not None and isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()

        if classes is not None and y is not None:
            m = np.isin(y, np.asarray(classes, dtype=int))
            Xp = Xp[m]

        if per_domain_cap is not None and len(Xp) > per_domain_cap:
            idx = rng.choice(len(Xp), size=per_domain_cap, replace=False)
            Xp = Xp[idx]

        if len(Xp):
            X_all.append(Xp)

    if not X_all:
        raise ValueError("fit_global_pca: no data collected to fit PCA.")

    X_fit = np.concatenate(X_all, axis=0)
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=random_state)
    pca.fit(X_fit)
    return pca

def plot_pca_classes_grid(
    domains,
    classes=(3, 8),
    save_path=None,
    pool: str = 'gap',
    label_source: str = 'pseudo',     # 'pseudo' | 'em' | 'real'
    pseudolabels=None,
    ground_truths=False,    # if True, use .targets for all domains
    pca: "PCA|None" = None,           # NEW: pass a pre-fit PCA to project onto the SAME axes
    return_pca: bool = False,         # NEW: optionally return the PCA used
    pca_fit_policy: str = "endpoints" # 'endpoints'|'all' (only used if pca is None)
):
    """
    If `pca` is provided, it is used to transform *every* panel.
    If `pca` is None, we fit one locally according to `pca_fit_policy` and use it.
    """
    import os

    import matplotlib.pyplot as plt
    import torch
    from sklearn.decomposition import PCA

    # ---------- helpers ----------
    def _to_np(y):
        if y is None: return None
        if isinstance(y, torch.Tensor): return y.detach().cpu().numpy()
        return np.asarray(y)

    def pool_feats(X):
        if isinstance(X, torch.Tensor): X = X.detach().cpu().numpy()
        if X.ndim == 4:
            if pool in ('gap', 'auto'): return X.mean(axis=(2, 3))
            elif pool == 'flatten':      return X.reshape(X.shape[0], -1)
            else:                        return X.mean(axis=(2, 3))
        if X.ndim == 3: return X.reshape(X.shape[0], -1)
        return X

    def get_labels_for_domain(D, j):
        if ground_truths:
            return _to_np(D.targets)
        if label_source == 'pseudo':
            y = pseudolabels
            return _to_np(y)
        
        if label_source == 'em':
            y = D.targets_em
            # breakpoint()
            return _to_np(y)
        
        if label_source == 'real':
            return _to_np(D.targets)
        return None

    # ---------- prepare ----------
    cols = len(domains)
    if cols == 0:
        return (None if not return_pca else (None, None))
    classes = np.array(list(classes), dtype=int)

    # ---------- collect pooled features & masks (once) ----------
    pooled_per_domain, masks_per_domain, labels_per_domain = [], [], []
    for j, D in enumerate(domains):
        Xp = pool_feats(D.data)
        pooled_per_domain.append(Xp)
        # if len(domains) > 2:
        #     breakpoint()
        if j == 0:
            y = D.targets
        elif j == len(domains) - 1:
            # breakpoint()
            y = get_labels_for_domain(D, j)
        else:
            if label_source == 'pseudo':
                # prefer pseudo (targets), fall back to EM for intermediates if you want the opposite, swap next line:
                # y = pseudolabels
                y = D.targets
            elif label_source == 'em':
                y = _to_np(D.targets_em)
        labels_per_domain.append(y)

        if y is None:
            masks_per_domain.append(None)
        else:
            masks_per_domain.append(np.isin(y, classes))

    # ---------- fit or reuse PCA ----------
    used_pca = pca
    if used_pca is None:
        X_lists = []
        if pca_fit_policy == "all":
            for j, Xp in enumerate(pooled_per_domain):
                m = masks_per_domain[j]
                if m is not None and m.any():
                    X_lists.append(Xp[m])
        else:  # 'endpoints': first and last with masks
            m0 = masks_per_domain[0]
            mL = masks_per_domain[-1]
            X0 = pooled_per_domain[0][m0] if (m0 is not None and m0.any()) else pooled_per_domain[0]
            XL = pooled_per_domain[-1][mL] if (mL is not None and mL.any()) else pooled_per_domain[-1]
            X_lists = [X0, XL]
        X_fit = np.concatenate(X_lists, axis=0) if len(X_lists) > 1 else X_lists[0]
        used_pca = PCA(n_components=2)
        used_pca.fit(X_fit)

    # ---------- plot ----------
    fig_w = max(4, 3 * cols)
    fig, axs = plt.subplots(1, cols, figsize=(fig_w, 3.6), squeeze=False)
    axs = axs[0]
    cmap = plt.get_cmap('tab10')

    for j in range(cols):
        ax = axs[j]
        Xp = pooled_per_domain[j]
        y  = labels_per_domain[j]
        m  = masks_per_domain[j]

        if (y is None) or (m is None) or (not np.any(m)):
            ax.set_title(f"Domain {j}: no classes {tuple(classes)}")
            ax.axis('off')
            continue

        Xsel = Xp[m]
        ysel = y[m]

        try:
            if Xsel.shape[1] > 2:
                Z = used_pca.transform(Xsel)
            else:
                Z = Xsel
        except Exception:
            Z = Xsel[:, :2] if Xsel.shape[1] >= 2 else np.pad(
                Xsel, ((0, 0), (0, max(0, 2 - Xsel.shape[1]))), mode='constant'
            )

        for idx, c in enumerate(classes):
            cmask = (ysel == c)
            if cmask.any():
                ax.scatter(Z[cmask, 0], Z[cmask, 1], s=6, alpha=0.7,
                           color=cmap(idx % 10), label=str(c))

        ax.set_title(f"Domain {j}")
        ax.set_xticks([]); ax.set_yticks([])
        if j == 0:
            ax.legend(loc='best', fontsize=8)

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        import matplotlib.pyplot as plt
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"[MNIST-EXP] Saved {save_path}")
    import matplotlib.pyplot as plt
    plt.close()

    if return_pca:
        return used_pca




# def plot_pca_classes_grid(
#     domains,
#     classes=(3, 8),
#     save_path=None,
#     pool: str = 'gap',
#     label_source: str = 'pseudo',     # 'pseudo' | 'em' | 'real'
#     pseudolabels=None,
#     ground_truths=False,    # if True, use .targets for all domains
# ):
#     """
#     Fit ONE PCA on all selected samples across domains, then apply it per-domain.

#     - domains: iterable of datasets with .data and labels (.targets or .targets_em)
#     - classes: tuple/list of class ids to display
#     - save_path: file path to save the figure (directory is created)
#     - pool: 'gap' to global-average-pool 4D (N,C,H,W) -> (N,C); 'flatten' to flatten; 'auto' chooses GAP for 4D
#     - label_source: 'pseudo' (use provided pseudo labels), 'em' (use .targets_em fallback to .targets), 'real' (use .targets)
#     - pseudolabels: dict[id(D)]->labels OR list/tuple aligned with `domains` OR array/tensor of labels per dataset
#     """
#     import os
# 
#     import matplotlib.pyplot as plt
#     import torch
#     from sklearn.decomposition import PCA

#     # ---------- helpers ----------
#     def _to_np(y):
#         if y is None:
#             return None
#         if isinstance(y, torch.Tensor):
#             return y.detach().cpu().numpy()
#         return np.asarray(y)

#     def pool_feats(X):
#         if isinstance(X, torch.Tensor):
#             X = X.detach().cpu().numpy()
#         if X.ndim == 4:
#             if pool == 'gap' or pool == 'auto':
#                 return X.mean(axis=(2, 3))
#             elif pool == 'flatten':
#                 return X.reshape(X.shape[0], -1)
#             else:
#                 # default to GAP for 4D if an unexpected value is passed
#                 return X.mean(axis=(2, 3))
#         if X.ndim == 3:
#             return X.reshape(X.shape[0], -1)
#         return X

#     def get_labels_for_domain(D, j):
#         """Return labels per `label_source`, robust to different pseudolabels types."""
#         y = None
#         if ground_truths:
#             # breakpoint()
#             return _to_np(getattr(D, 'targets', None))
#         if label_source == 'pseudo':
#             if isinstance(pseudolabels, dict):
#                 y = pseudolabels.get(id(D), None)
#             elif isinstance(pseudolabels, (list, tuple)):
#                 if 0 <= j < len(pseudolabels):
#                     y = pseudolabels[j]
#             elif isinstance(pseudolabels, (np.ndarray, torch.Tensor)):
#                 # assume it's aligned with D
#                 y = pseudolabels
#             # fallbacks if pseudo not available
#             if y is None:
#                 y = getattr(D, 'targets_em', None)
#                 if y is None:
#                     y = getattr(D, 'targets', None)

#         elif label_source == 'em':
#             y = getattr(D, 'targets_em', None)
#             if y is None or (isinstance(y, torch.Tensor) and y.numel() > 0 and (y < 0).all()):
#                 y = getattr(D, 'targets', None)

#         elif label_source == 'real':
#             y = getattr(D, 'targets', None)

#         return _to_np(y)

#     # ---------- prepare ----------
#     cols = len(domains)
#     if cols == 0:
#         return
#     classes = np.array(list(classes), dtype=int)

#     # ---------- 1) Collect data for a single global PCA ----------
#     X_all = []
#     y_all = []
#     pooled_per_domain = []
#     masks_per_domain = []

#     for j, D in enumerate(domains):
        
#         Xp = pool_feats(D.data)
#         pooled_per_domain.append(Xp)
#         if j == 0:
#             y = D.targets
#         elif j == len(domains) - 1:
#             y = get_labels_for_domain(D, j)
#         else:
#             # breakpoint()
#             y = D.targets if label_source =='pseudo' else D.targets_em
#         if y is None:
#             masks_per_domain.append(None)
#             continue

#         # keep only requested classes
#         m = np.isin(y, classes)
#         masks_per_domain.append(m)

#         if m.any():
#             try:
#                 X_all.append(Xp[m])
#                 y_all.append(y[m])
#             except Exception as e:
#                 breakpoint()
#                 print(f"[plot] Failed to append data for domain {j} ({e})")

#     if len(X_all) == 0:
#         print("[plot] No samples from requested classes; nothing to plot.")
#         return

#     X_pca = np.concatenate([X_all[0], X_all[-1]], axis=0)
#     X_all = np.concatenate(X_all, axis=0)
#     y_all = np.concatenate(y_all, axis=0)

#     # ---------- PCA fit (shared) ----------
#     try:
#         pca = PCA(n_components=2)
#         pca.fit(X_pca)
#         use_pca = True
#     except Exception as e:
#         print(f"[plot] PCA fit failed ({e}); falling back to first two dims.")
#         use_pca = False

#     # ---------- 2) Plot per domain using the SAME PCA ----------
#     fig_w = max(4, 3 * cols)
#     fig, axs = plt.subplots(1, cols, figsize=(fig_w, 3.6), squeeze=False)
#     axs = axs[0]
#     cmap = plt.get_cmap('tab10')

#     for j, D in enumerate(domains):
#         ax = axs[j]
#         Xp = pooled_per_domain[j]
#         if j == 0:
#             y = D.targets
#         elif j == len(domains) - 1:
#             y = get_labels_for_domain(D, j)
#         else:
#             y = D.targets if label_source =='pseudo' else D.targets_em

#         if y is None:
#             ax.set_title(f"Domain {j}: no labels")
#             ax.axis('off')
#             continue

#         m = masks_per_domain[j]
#         if m is None or not np.any(m):
#             ax.set_title(f"Domain {j}: no classes {tuple(classes)}")
#             ax.axis('off')
#             continue

#         Xsel = Xp[m]
#         ysel = y[m]

#         if use_pca:
#             try:
#                 Z = pca.transform(Xsel)
#             except Exception:
#                 Z = Xsel[:, :2] if Xsel.shape[1] >= 2 else np.pad(
#                     Xsel, ((0, 0), (0, max(0, 2 - Xsel.shape[1]))), mode='constant'
#                 )
#         else:
#             Z = Xsel[:, :2] if Xsel.shape[1] >= 2 else np.pad(
#                 Xsel, ((0, 0), (0, max(0, 2 - Xsel.shape[1]))), mode='constant'
#             )

#         # plot
#         for idx, c in enumerate(classes):
#             cmask = (ysel == c)
#             if cmask.any():
#                 ax.scatter(Z[cmask, 0], Z[cmask, 1], s=6, alpha=0.7,
#                            color=cmap(idx % 10), label=str(c))

#         ax.set_title(f"Domain {j}")
#         ax.set_xticks([]); ax.set_yticks([])
#         if j == 0:
#             ax.legend(loc='best', fontsize=8)

#     if save_path:
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         plt.tight_layout()
#         plt.savefig(save_path, dpi=150)
#         print(f"[MNIST-EXP] Saved {save_path}")
#     plt.close()

def _apply_em_bundle_to_target(bundle, e_tgt, tgt_trainset):
    """
    Put EM labels and mapped soft posteriors on the target dataset so *all* methods
    can read them without refitting EM.
    """
    # hard labels
    labels_em = np.asarray(bundle.labels_em, dtype=np.int64)
    e_tgt.targets_em = torch.as_tensor(labels_em, dtype=torch.long)
    tgt_trainset.targets_em = e_tgt.targets_em.cpu().clone()

    # optional soft posteriors (class posteriors after cluster->class mapping)
    if getattr(bundle, "P_soft", None) is not None:
        tgt_trainset.soft_targets = torch.from_numpy(bundle.P_soft).float()

# ---------------- Run all three methods and compare on shared plots ----------------
def run_mnist_experiment(target: int, gt_domains: int, generated_domains: int, args=None):
    src_trainset = get_single_rotate(False, 0)
    tgt_trainset = get_single_rotate(False, target)
    # breakpoint()
    model_dir = f"/data/common/yuenchen/GDA/mnist_models/"

    encoder = ENCODER().to(device)
    model_name_smalldim = f"src0_tgt{target}_ssl{args.ssl_weight}_dim{args.small_dim}.pth"
    source_model_smalldim = get_source_model(
        args, src_trainset, tgt_trainset, n_class=10, mode="mnist",
        encoder=encoder, epochs=10, model_path=f"{model_dir}/{model_name_smalldim}",
        target_dataset=tgt_trainset, force_recompute=False, compress=True,
        in_dim=25088, out_dim=args.small_dim
    )

    # SAME reference for all runs
    ref_model = source_model_smalldim
    ref_encoder = nn.Sequential(
        ref_model.encoder,
        nn.Flatten(start_dim=1),
        getattr(ref_model, 'compressor', nn.Identity())
    ).eval()

    # Build real intermediate domains
    all_sets, deg_idx = [], []
    for i in range(1, gt_domains + 1):
        angle = i * target // (gt_domains + 1)
        all_sets.append(get_single_rotate(False, angle))
        deg_idx.append(angle)
    all_sets.append(tgt_trainset)
    deg_idx.append(target)

    # Evaluate initial (for sanity)
    tgt_loader_eval = DataLoader(tgt_trainset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


    e_src, e_tgt, encoded_intersets = encode_all_domains(
        src_trainset, tgt_trainset, all_sets, deg_idx, ref_encoder,
        cache_dir=f"cache{args.ssl_weight}/target{target}/small_dim{args.small_dim}/",
        target=target, force_recompute=False, args=args
    )

    

    # (C) Fit a shared PCA on these *real* domains (robust and method-agnostic)
    shared_pca = fit_global_pca(
        domains=encoded_intersets,     # same space every method will use
        classes=None,               # all classes
        pool="auto",                   # already embedded → 'auto' no-ops
        n_components=2,
        per_domain_cap=10000,
        random_state=args.seed if hasattr(args, 'seed') else 0,
    )

    # (D) Store it so children helpers can see it (simple: put into args)
    args.shared_pca = shared_pca
    # After you have e_src (encoded source domain)
    if getattr(args, "em_match", "pseudo") == "prototypes":
        # Compute source Gaussian params from SOURCE labels only (no target GT).
        mu_s, Sigma_s, priors_s = fit_source_gaussian_params(X=e_src.data, y=e_src.targets)
        args._cached_source_stats = (mu_s, Sigma_s, priors_s)


    with torch.no_grad():
        em_teacher = copy.deepcopy(ref_model).to(device).eval()
        pseudo_labels, _ = get_pseudo_labels(
            tgt_trainset, em_teacher,
            confidence_q=getattr(args, "pseudo_confidence_q", 0.9),
            device_override=device,
        )
    args._cached_pseudolabels = pseudo_labels.cpu().numpy()

    # Fit EM ONCE on the shared target embedding and store on args
    # em_bundle = ensure_shared_em_for_target(
    #     e_tgt, args=args,
    #     K=10, cov_type="diag", pool="gap",
    #     do_pca=False, pca_dim=64,  # or True & 128 if you prefer
    #     reg=1e-6, max_iter=150, tol=1e-5, n_init=5, dtype="float32"
    # )

    if args.em_match != "none":
        em_models = fit_many_em_on_target(
            e_tgt,
            K_list=getattr(args, "em_K_list", [10]),
            cov_types=getattr(args, "em_cov_types", ["diag"]),
            seeds=getattr(args, "em_seeds", [0,1,2]),
            pool="gap",
            pca_dims=getattr(args, "em_pca_dims", [None]),
            reg=1e-4, max_iter=300, rng_base=args.seed, args=args
        )
        best = _select_best_em(em_models, criterion=getattr(args, "em_select", "bic"))
        if getattr(args, "em_do_ensemble", True) and len(em_models) > 1:
            trimmed, w = _trim_em_models_by_bic(em_models, max_delta_bic=getattr(args, "em_trim_delta_bic", 10.0))
            P_ens, H_ens = _ensemble_posteriors_trimmed(trimmed, w)
            labels_ens = P_ens.argmax(axis=1).astype(int)
            em_bundle = EMBundle(
                key="multi_em_trimmed", em_res=best["em_res"],
                mapping=best.get("mapping", None),
                labels_em=labels_ens, P_soft=P_ens,
                info={"criterion":"bic+trim", "bic_best":float(best["bic"])}
            )
        else:
            em_bundle = EMBundle(
                key="multi_em_best", em_res=best["em_res"],
                mapping=best.get("mapping", None),
                labels_em=np.asarray(best["labels_mapped"], dtype=int),
                P_soft=np.asarray(best["mapped_soft"], dtype=float),
                info={"criterion":getattr(args,"em_select","bic"), "bic_best":float(best["bic"])}
            )

        args._shared_em = em_bundle
        _apply_em_bundle_to_target(em_bundle, e_tgt, tgt_trainset)

    # For ETA/natural path
    our_source_eta = copy.deepcopy(ref_model)
    ours_copy_eta  = copy.deepcopy(ref_model)

    # # # ---- (1b) Ours/ETA (natural-parameter) path ----
    set_all_seeds(args.seed)
    ours_eta_train, ours_eta_test, ours_eta_st, ours_eta_st_all, ours_eta_gen, EM_acc_eta = run_main_algo(
        ours_copy_eta, our_source_eta, src_trainset, tgt_trainset, all_sets, deg_idx,
        generated_domains, epochs=5, target=target, args=args, gen_method="natural"
    )
    set_all_seeds(args.seed)
    goat_cw_src = copy.deepcopy(ref_model)
    goat_cw_cp  = copy.deepcopy(goat_cw_src)
    # ---- (3) GOAT (class-wise synthetics) ----
    goatcw_train, goatcw_test, goatcw_st, goatcw_st_all, goatcw_gen, EM_acc_goatcw = run_goat_classwise(
        goat_cw_cp, goat_cw_src, src_trainset, tgt_trainset, all_sets, deg_idx,
        generated_domains, epochs=5, target=target, args=args
    )


    # # # breakpoint()


    # # # ---- (1) Ours/FR path (returns EM mapping accuracy too) ----
    # # breakpoint()
    set_all_seeds(args.seed)
    our_source = copy.deepcopy(ref_model)
    ours_copy   = copy.deepcopy(ref_model)
    ours_train, ours_test, ours_st, ours_st_all, ours_gen, EM_acc = run_main_algo(
        ours_copy, our_source, src_trainset, tgt_trainset, all_sets, deg_idx,
        generated_domains, epochs=5, target=target, args=args
    )
    # breakpoint()
    # assert abs(EM_acc - EM_acc_eta) < 1e-4, "EM accuracies differ between methods!"


    goat_source = copy.deepcopy(ref_model)
    goat_copy   = copy.deepcopy(goat_source)

    set_all_seeds(args.seed)
    # ---- (2) GOAT (pair-wise synthetics) ----
    goat_train, goat_test, goat_st, goat_st_all, goat_gen = run_goat(
        goat_copy, goat_source, src_trainset, tgt_trainset, all_sets, deg_idx,
        generated_domains, epochs=5, target=target, args=args
    )

    # ---- Persist series ----
    plot_dir = f"plots/target{target}/"
    os.makedirs(plot_dir, exist_ok=True)

    # ---- Plot 2: Test/target accuracy (4-way + EM% reference) ----
    _plot_series_with_baselines(
        series=[ours_test, goat_test, goatcw_test, ours_eta_test],
        labels=[
            f"Ours-FR",
            "GOAT",
            f"GOAT-Classwise",
            f"Ours-ETA",
        ],
        baselines=[
            (ours_st, ours_st_all),
            # (goat_st, goat_st_all),
            # (goatcw_st, goatcw_st_all),
            # (ours_eta_st, ours_eta_st_all),
        ],
        ref_line_value=(EM_acc * 100.0 if EM_acc is not None else None),
        ref_line_label=f"EM ({args.em_match})",
        ref_line_style="--",
        title=f"Target Accuracy (ST: {args.label_source} labels; Cluster Map: {args.em_match})",
        ylabel="Accuracy", xlabel="Domain Index",
        save_path=os.path.join(plot_dir, f"test_acc_dim{args.small_dim}_gen{args.generated_domains}_{args.label_source}_{args.em_match}.png"),
    )


def run_mnist_ablation(target: int, gt_domains: int, generated_domains: int):
    encoder = ENCODER().to(device)
    src_trainset = get_single_rotate(False, 0)
    tgt_trainset = get_single_rotate(False, target)

    source_model = get_source_model(args, src_trainset, src_trainset, 10, "mnist", encoder=encoder, epochs=20)

    all_sets = [get_single_rotate(False, i * target // (gt_domains + 1)) for i in range(1, gt_domains + 1)]
    all_sets.append(tgt_trainset)

    # Baselines
    model_copy = copy.deepcopy(source_model)
    direct_acc, st_acc = self_train(args, model_copy, [tgt_trainset], epochs=10)
    direct_acc_all, st_acc_all = self_train(args, source_model, all_sets, epochs=10)

    # Encode domains
    e_src = get_encoded_dataset(src_trainset, encoder=source_model.encoder)
    e_tgt = get_encoded_dataset(tgt_trainset, encoder=source_model.encoder)
    encoded_inter = [e_src] + [get_encoded_dataset(ds, encoder=source_model.encoder) for ds in all_sets[:-1]] + [e_tgt]

    # Random plan
    all_domains1 = []
    for i in range(len(encoded_inter) - 1):
        plan = ot_ablation(len(src_trainset), "random")
        all_domains1 += generate_domains(generated_domains, encoded_inter[i], encoded_inter[i + 1], plan=plan)[0]
    _, generated_acc1 = self_train(args, copy.deepcopy(source_model).mlp, all_domains1, epochs=10)

    # Uniform plan
    all_domains4 = []
    for i in range(len(encoded_inter) - 1):
        plan = ot_ablation(len(src_trainset), "uniform")
        all_domains4 += generate_domains(generated_domains, encoded_inter[i], encoded_inter[i + 1], plan=plan)[0]
    _, generated_acc4 = self_train(args, copy.deepcopy(source_model).mlp, all_domains4, epochs=10)

    # OT plan
    all_domains2 = []
    for i in range(len(encoded_inter) - 1):
        all_domains2 += generate_domains(generated_domains, encoded_inter[i], encoded_inter[i + 1])[0]
    _, generated_acc2 = self_train(args, copy.deepcopy(source_model).mlp, all_domains2, epochs=10)

    # GT plan (identity)
    all_domains3 = []
    for i in range(len(encoded_inter) - 1):
        _ = np.identity(len(src_trainset))  # placeholder if your generator uses it
        all_domains3 += generate_domains(generated_domains, encoded_inter[i], encoded_inter[i + 1])[0]
    _, generated_acc3 = self_train(args, copy.deepcopy(source_model).mlp, all_domains3, epochs=10)

    os.makedirs("logs", exist_ok=True)
    with open(f"logs/mnist_{target}_{generated_domains}_ablation.txt", "a") as f:
        f.write(
            f"seed{args.seed}generated{generated_domains},{round(direct_acc, 2)},{round(st_acc, 2)},{round(st_acc_all, 2)},{round(generated_acc1, 2)},{round(float(generated_acc4), 2)},{round(generated_acc2, 2)},{round(generated_acc3, 2)}\n"
        )





def run_portraits_experiment(gt_domains: int, generated_domains: int, args=None):
    """
    Mirror the MNIST experiment suite on the portraits dataset:
    - Train/load source model (also a compressed small-dim variant)
    - Encode domains, fit a shared PCA for consistent plots
    - Run Ours (FR), Ours (ETA/natural), GOAT, GOAT-Classwise
    - Plot accuracy series with ST baselines
    """
    t0 = time.time()

    # ----- Build source/target and intermediates from portraits data -----
    (
        src_tr_x,
        src_tr_y,
        src_val_x,
        src_val_y,
        inter_x,
        inter_y,
        dir_inter_x,
        dir_inter_y,
        trg_val_x,
        trg_val_y,
        trg_test_x,
        trg_test_y,
    ) = make_portraits_data(1000, 1000, 14000, 2000, 1000, 1000)

    tr_x = np.concatenate([src_tr_x, src_val_x])
    tr_y = np.concatenate([src_tr_y, src_val_y])
    ts_x = np.concatenate([trg_val_x, trg_test_x])
    ts_y = np.concatenate([trg_val_y, trg_test_y])

    encoder = ENCODER().to(device)
    transforms = ToTensor()

    src_trainset = EncodeDataset(tr_x, tr_y.astype(int), transforms)
    tgt_trainset = EncodeDataset(ts_x, ts_y.astype(int), transforms)

    # Train/load source model; also build a compressed variant for stable embedding space
    model_dir = f"portraits/cache{args.ssl_weight}/"
    os.makedirs(model_dir, exist_ok=True)
    model_name_smalldim= f"ssl{args.ssl_weight}_dim{args.small_dim}.pth"
    source_model = get_source_model(
        args,
        src_trainset,
        tgt_trainset,
        n_class=2,
        mode="portraits",
        encoder=encoder,
        epochs=20,
        model_path=os.path.join(model_dir, model_name_smalldim),
        target_dataset=tgt_trainset,
        force_recompute=False,
        compress=True,
        in_dim=32768,
        out_dim=args.small_dim
    )

    # Common reference used across all methods
    ref_model = source_model
    ref_encoder = nn.Sequential(
        ref_model.encoder,
        nn.Flatten(start_dim=1),
        getattr(ref_model, 'compressor', nn.Identity())
    ).eval()
    source_model_goat = copy.deepcopy(ref_model)
    source_model_main = copy.deepcopy(ref_model)
    model_copy_goat = copy.deepcopy(source_model_goat)
    model_copy_main = copy.deepcopy(source_model_main)

    def get_domains(n_domains):
        domain_set = []
        n2idx = {0:[], 1:[3], 2:[2,4], 3:[1,3,5], 4:[0,2,4,6], 7:[0,1,2,3,4,5,6]}
        domain_idx = n2idx[n_domains]
        for i in domain_idx:
            start, end = i*2000, (i+1)*2000
            domain_set.append(EncodeDataset(inter_x[start:end], inter_y[start:end].astype(int), transforms))
        return domain_set

    all_sets = get_domains(gt_domains)
    all_sets.append(tgt_trainset)
    _e_src, _e_tgt, encoded_intersets = encode_all_domains(
        src_trainset, tgt_trainset, all_sets, 0,
        encoder=ref_encoder,
        cache_dir=f"portraits/cache{args.ssl_weight}/small_dim{args.small_dim}/",
        target=1,
        force_recompute=False,
        args=args
    )
    shared_pca = fit_global_pca(
        domains=encoded_intersets,
        classes=None,
        pool="auto",
        n_components=2,
        per_domain_cap=10000,
        random_state=args.seed if hasattr(args, 'seed') else 0,
    )
    args.shared_pca = shared_pca

    if getattr(args, "em_match", "pseudo") == "prototypes":
        # Compute source Gaussian params from SOURCE labels only (no target GT).
        mu_s, Sigma_s, priors_s = fit_source_gaussian_params(X=_e_src.data, y=_e_src.targets)
        args._cached_source_stats = (mu_s, Sigma_s, priors_s)


    with torch.no_grad():
        em_teacher = copy.deepcopy(ref_model).to(device).eval()
        pseudo_labels, _ = get_pseudo_labels(
            tgt_trainset, em_teacher,
            confidence_q=getattr(args, "pseudo_confidence_q", 0.9),
            device_override=device,
        )
    args._cached_pseudolabels = pseudo_labels.cpu().numpy()

    # Fit EM ONCE on the shared target embedding and store on args
    # em_bundle = ensure_shared_em_for_target(
    #     e_tgt, args=args,
    #     K=10, cov_type="diag", pool="gap",
    #     do_pca=False, pca_dim=64,  # or True & 128 if you prefer
    #     reg=1e-6, max_iter=150, tol=1e-5, n_init=5, dtype="float32"
    # )

    # Instead of ensure_shared_em_for_target(...):
    em_models = fit_many_em_on_target(
        _e_tgt,
        K_list=getattr(args, "em_K_list",  [src_trainset.targets.max().item() + 1]),
        cov_types=getattr(args, "em_cov_types", ["diag"]),
        seeds=getattr(args, "em_seeds", [0,1,2]),
        pool="gap",
        pca_dims=getattr(args, "em_pca_dims", [None]),
        reg=1e-4, max_iter=300, rng_base=args.seed, args=args
    )
    # Select EM by perfect-matching accuracy by default for Color-MNIST
    best = _select_best_em(em_models, criterion=getattr(args, "em_select", "pm"))
    if getattr(args, "em_do_ensemble", True) and len(em_models) > 1:
        trimmed, w = _trim_em_models_by_bic(em_models, max_delta_bic=getattr(args, "em_trim_delta_bic", 10.0))
        P_ens, H_ens = _ensemble_posteriors_trimmed(trimmed, w)
        labels_ens = P_ens.argmax(axis=1).astype(int)
        em_bundle = EMBundle(
            key="multi_em_trimmed", em_res=best["em_res"],
            mapping=best.get("mapping", None),
            labels_em=labels_ens, P_soft=P_ens,
            info={"criterion":"bic+trim", "bic_best":float(best["bic"])}
        )
    else:
        em_bundle = EMBundle(
            key="multi_em_best", em_res=best["em_res"],
            mapping=best.get("mapping", None),
            labels_em=np.asarray(best["labels_mapped"], dtype=int),
            P_soft=np.asarray(best["mapped_soft"], dtype=float),
            info={"criterion":getattr(args,"em_select","bic"), "bic_best":float(best["bic"])}
        )

    args._shared_em = em_bundle
    _apply_em_bundle_to_target(em_bundle, _e_tgt, tgt_trainset)


    y_true = np.asarray(tgt_trainset.targets, dtype=int)
    y_em = np.asarray(em_bundle.labels_em, dtype=int)
    em_acc_now = float((y_em == y_true).mean())
    print(f"[ColorMNIST] EM→class accuracy: {em_acc_now:.4f}")

    # ---------------- Ours (ETA) ----------------
    set_all_seeds(args.seed)
    ours_eta_src = copy.deepcopy(ref_model)
    ours_eta_cp  = copy.deepcopy(ref_model)
    ours_eta_train, ours_eta_test, ours_eta_st, ours_eta_st_all, ours_eta_gen, EM_acc_eta = run_main_algo(
        ours_eta_cp, ours_eta_src, src_trainset, tgt_trainset, all_sets, 0,
        generated_domains, epochs=5, target=1, args=args, gen_method="natural"
    )

    # ---------------- Ours (FR) ----------------
    set_all_seeds(args.seed)
    ours_src = copy.deepcopy(ref_model)
    ours_cp  = copy.deepcopy(ref_model)
    ours_train, ours_test, ours_st, ours_st_all, ours_gen, EM_acc = run_main_algo(
        ours_cp, ours_src, src_trainset, tgt_trainset, all_sets, 0,
        generated_domains, epochs=5, target=1, args=args, gen_method="fr"
    )

    # ---------------- GOAT (pair-wise synthetics) ----------------
    set_all_seeds(args.seed)
    goat_src = copy.deepcopy(ref_model)
    goat_cp  = copy.deepcopy(goat_src)
    goat_train, goat_test, goat_st, goat_st_all, goat_gen = run_goat(
        goat_cp, goat_src, src_trainset, tgt_trainset, all_sets, 0,
        generated_domains, epochs=5, target=1, args=args
    )

    # ---------------- GOAT (class-wise synthetics) ----------------
    set_all_seeds(args.seed)
    goatcw_src = copy.deepcopy(ref_model)
    goatcw_cp  = copy.deepcopy(goatcw_src)
    goatcw_train, goatcw_test, goatcw_st, goatcw_st_all, goatcw_gen, EM_acc_goatcw = run_goat_classwise(
        goatcw_cp, goatcw_src, src_trainset, tgt_trainset, all_sets, 0,
        generated_domains, epochs=5, target=1, args=args
    )

    if EM_acc is not None and EM_acc_goatcw is not None and abs(EM_acc - EM_acc_goatcw) > 1e-4:
        print(f"[WARNING] EM acc differs between Ours ({EM_acc:.2f}%) and GOAT-Classwise ({EM_acc_goatcw:.2f}%) on portraits!")
    # breakpoint()
    # ---------------- Plot combined target accuracy series ----------------
    plot_dir = f"plots/portraits/"
    os.makedirs(plot_dir, exist_ok=True)
    _plot_series_with_baselines(
        series=[ours_test, goat_test, goatcw_test, ours_eta_test],
        labels=["Ours-FR", "GOAT", "GOAT-Classwise", "Ours-ETA"],
        baselines=[(ours_st, ours_st_all)],
        ref_line_value=(EM_acc * 100.0 if EM_acc is not None else None),
        ref_line_label=f"EM ({args.em_match})",
        ref_line_style="--",
        title=f"Portraits: Target Accuracy (ST: {args.label_source}; Cluster Map: {args.em_match})",
        ylabel="Accuracy",
        xlabel="Domain Index",
        save_path=os.path.join(plot_dir, f"test_acc_dim{args.small_dim}_gen{args.generated_domains}_{args.label_source}_{args.em_match}.png"),
    )

    # ---------------- Log summary ----------------
    os.makedirs("logs", exist_ok=True)
    elapsed = round(time.time() - t0, 2)
    with open("logs/portraits_exp_time.txt", "a") as f:
        f.write(
            f"seed{args.seed}with{gt_domains}gt{generated_domains}generated,{elapsed},"
            f"OursFR:{round((ours_test[-1] if ours_test else 0.0), 2)},GOAT:{round((goat_test[-1] if goat_test else 0.0), 2)},"
            f"GOATCW:{round((goatcw_test[-1] if goatcw_test else 0.0), 2)},ETA:{round((ours_eta_test[-1] if ours_eta_test else 0.0), 2)}\n"
        )




def run_covtype_experiment(gt_domains: int, generated_domains: int, args=None):
    """
    CovType experiment mirroring the portraits pipeline:
    - Train/load compressed source model
    - Encode real domains; fit shared PCA
    - Fit many EMs on target embedding; select/ensemble; cache bundle
    - Run Ours (FR), Ours (ETA), GOAT, GOAT-Classwise
    - Plot accuracy series with ST baselines; log summary
    """
    t0 = time.time()

    # ----- Load / synthesize CovType-style data -----
    (
        src_tr_x, src_tr_y,
        src_val_x, src_val_y,
        inter_x, inter_y,
        dir_inter_x, dir_inter_y,
        trg_val_x, trg_val_y,
        trg_test_x, trg_test_y,
    ) = make_cov_data(40000, 10000, 400000, 50000, 25000, 20000)

    # Use train+val as "source", val+test as "target" (eval GT present, not used for training)
    src_x = np.concatenate([src_tr_x, src_val_x])
    src_y = np.concatenate([src_tr_y, src_val_y])
    tgt_x = np.concatenate([trg_val_x, trg_test_x])
    tgt_y = np.concatenate([trg_val_y, trg_test_y])

    # Datasets
    src_trainset = EncodeDataset(torch.from_numpy(src_x).float(), src_y.astype(int))
    tgt_trainset = EncodeDataset(torch.from_numpy(tgt_x).float(), tgt_y.astype(int))

    # ----- Model (compressed) -----
    encoder = MLP_Encoder().to(device)
    model_dir = f"covtype/cache{args.ssl_weight}/"
    os.makedirs(model_dir, exist_ok=True)
    def infer_encoder_out_dim(encoder, example, device):
        encoder = encoder.to(device).eval()
        with torch.no_grad():
            z = encoder(example.to(device))
            z = z.view(z.size(0), -1)
            return int(z.size(1))
    _ex = torch.as_tensor(src_trainset.data[:2]).float()
    enc_out_dim = infer_encoder_out_dim(encoder, _ex, device)
    if enc_out_dim < args.small_dim:
        args.small_dim = enc_out_dim
    model_name = f"ssl{args.ssl_weight}_dim{min(enc_out_dim, args.small_dim)}.pth"
    source_model = get_source_model(
        args,
        src_trainset, tgt_trainset,
        n_class=2,                 # CovType binary in this setup
        mode="covtype",
        encoder=encoder,
        epochs=10,
        model_path=os.path.join(model_dir, model_name),
        target_dataset=tgt_trainset,
        force_recompute=False,
        compress=True,
        in_dim=enc_out_dim,
        out_dim=min(enc_out_dim, args.small_dim),
    )

    # Common reference encoder
    ref_model = source_model
    ref_encoder = nn.Sequential(
        ref_model.encoder,
        nn.Flatten(start_dim=1),
        getattr(ref_model, 'compressor', nn.Identity())
    ).eval()

    # ----- Build ground-truth intermediate domains selector -----
    def get_domains(n_domains: int) -> List[Dataset]:
        # Same spacing scheme as older code, but now returns full domains
        idx_map = {0: [], 1: [6], 2: [3, 7], 3: [2, 5, 8], 4: [2, 4, 6, 8], 5: [1, 3, 5, 7, 9],
                   10: list(range(10)), 200: list(range(200))}
        domain_idx = idx_map.get(n_domains, [])
        domains = []
        # Each slot spans 40k, we take a 2k slice per slot (as in your prior code)
        for i in domain_idx:
            start, end = i * 40000, i * 40000 + 2000
            Xi = torch.from_numpy(inter_x[start:end]).float()
            yi = inter_y[start:end].astype(int)
            domains.append(EncodeDataset(Xi, yi))
        return domains

    all_sets = get_domains(gt_domains)
    all_sets.append(tgt_trainset)

    # ----- Encode domains and fit a shared PCA for consistent plots -----
    _e_src, _e_tgt, encoded_intersets = encode_all_domains(
        src_trainset, tgt_trainset, all_sets, deg_idx=0,
        encoder=ref_encoder,
        cache_dir=f"covtype/cache{args.ssl_weight}/small_dim{args.small_dim}/",
        target=1,
        force_recompute=False,
        args=args
    )
    shared_pca = fit_global_pca(
        domains=encoded_intersets, classes=None, pool="auto", n_components=2,
        per_domain_cap=10000, random_state=args.seed if hasattr(args, "seed") else 0,
    )
    args.shared_pca = shared_pca

    # (Optional) prototype mapping support
    if getattr(args, "em_match", "pseudo") == "prototypes":
        mu_s, Sigma_s, priors_s = fit_source_gaussian_params(X=_e_src.data, y=_e_src.targets)
        args._cached_source_stats = (mu_s, Sigma_s, priors_s)

    # ----- Pseudo labels from frozen teacher (only for mapping/eval plumbing) -----
    with torch.no_grad():
        teacher = copy.deepcopy(ref_model).to(device).eval()
        pseudo_labels, _ = get_pseudo_labels(
            tgt_trainset, teacher,
            confidence_q=getattr(args, "pseudo_confidence_q", 0.9),
            device_override=device,
        )
    args._cached_pseudolabels = pseudo_labels.cpu().numpy()

    # ----- Fit many EMs on the target embedding; pick/ensemble; cache bundle -----
    em_models = fit_many_em_on_target(
        _e_tgt,
        K_list=getattr(args, "em_K_list", [int(src_trainset.targets.max()) + 1]),
        cov_types=getattr(args, "em_cov_types", ["diag"]),
        seeds=getattr(args, "em_seeds", [0, 1, 2]),
        pool="gap",
        pca_dims=getattr(args, "em_pca_dims", [None]),
        reg=1e-4, max_iter=300, rng_base=args.seed, args=args,
    )
    best = _select_best_em(em_models, criterion=getattr(args, "em_select", "bic"))
    if getattr(args, "em_do_ensemble", True) and len(em_models) > 1:
        trimmed, w = _trim_em_models_by_bic(em_models, max_delta_bic=getattr(args, "em_trim_delta_bic", 10.0))
        P_ens, _ = _ensemble_posteriors_trimmed(trimmed, w)
        labels_ens = P_ens.argmax(axis=1).astype(int)
        em_bundle = EMBundle(
            key="multi_em_trimmed", em_res=best["em_res"], mapping=best.get("mapping", None),
            labels_em=labels_ens, P_soft=P_ens,
            info={"criterion": "bic+trim", "bic_best": float(best["bic"])}
        )
    else:
        em_bundle = EMBundle(
            key="multi_em_best", em_res=best["em_res"], mapping=best.get("mapping", None),
            labels_em=np.asarray(best["labels_mapped"], dtype=int),
            P_soft=np.asarray(best["mapped_soft"], dtype=float),
            info={"criterion": getattr(args, "em_select", "bic"), "bic_best": float(best["bic"])}
        )
    args._shared_em = em_bundle
    _apply_em_bundle_to_target(em_bundle, _e_tgt, tgt_trainset)

    # ----- Run methods -----
    # Ours-ETA
    # set_all_seeds(args.seed)
    # ours_eta_src = copy.deepcopy(ref_model);  ours_eta_cp = copy.deepcopy(ref_model)
    # ours_eta_train, ours_eta_test, ours_eta_st, ours_eta_st_all, ours_eta_gen, EM_acc_eta = run_main_algo(
    #     ours_eta_cp, ours_eta_src, src_trainset, tgt_trainset, all_sets, 0,
    #     generated_domains, epochs=5, target=1, args=args, gen_method="natural"
    # )

        # GOAT-Classwise
    set_all_seeds(args.seed)
    goatcw_src = copy.deepcopy(ref_model);  goatcw_cp = copy.deepcopy(goatcw_src)
    goatcw_train, goatcw_test, goatcw_st, goatcw_st_all, goatcw_gen, EM_acc_goatcw = run_goat_classwise(
        goatcw_cp, goatcw_src, src_trainset, tgt_trainset, all_sets, 0,
        generated_domains, epochs=5, target=1, args=args
    )

    # Ours-FR
    set_all_seeds(args.seed)
    ours_src = copy.deepcopy(ref_model);  ours_cp = copy.deepcopy(ref_model)
    ours_train, ours_test, ours_st, ours_st_all, ours_gen, EM_acc = run_main_algo(
        ours_cp, ours_src, src_trainset, tgt_trainset, all_sets, 0,
        generated_domains, epochs=5, target=1, args=args, gen_method="fr"
    )

    # GOAT
    set_all_seeds(args.seed)
    goat_src = copy.deepcopy(ref_model);  goat_cp = copy.deepcopy(goat_src)
    goat_train, goat_test, goat_st, goat_st_all, goat_gen = run_goat(
        goat_cp, goat_src, src_trainset, tgt_trainset, all_sets, 0,
        generated_domains, epochs=5, target=1, args=args
    )



    # ----- Plot -----
    plot_dir = f"plots/covtype/"
    os.makedirs(plot_dir, exist_ok=True)
    # _plot_series_with_baselines(
    #     series=[ours_test, goat_test, goatcw_test, ours_eta_test],
    #     labels=["Ours-FR", "GOAT", "GOAT-Classwise", "Ours-ETA"],
    #     baselines=[(ours_st, ours_st_all)],
    #     ref_line_value=(EM_acc * 100.0 if EM_acc is not None else None),
    #     ref_line_label=f"EM ({args.em_match})",
    #     ref_line_style="--",
    #     title=f"CovType: Target Accuracy (ST: {args.label_source}; Cluster Map: {args.em_match})",
    #     ylabel="Accuracy", xlabel="Domain Index",
    #     save_path=os.path.join(plot_dir, f"test_acc_dim{args.small_dim}_gen{args.generated_domains}_{args.label_source}_{args.em_match}.png"),
    # )

    # # ----- Log -----
    # os.makedirs("logs", exist_ok=True)
    # elapsed = round(time.time() - t0, 2)
    # with open(f"logs/covtype_exp_{args.log_file}.txt", "a") as f:
    #     f.write(
    #         f"seed{args.seed}with{gt_domains}gt{generated_domains}generated,{elapsed},"
    #         f"OursFR:{round((ours_test[-1] if ours_test else 0.0), 2)},GOAT:{round((goat_test[-1] if goat_test else 0.0), 2)},"
    #         f"GOATCW:{round((goatcw_test[-1] if goatcw_test else 0.0), 2)},ETA:{round((ours_eta_test[-1] if ours_eta_test else 0.0), 2)}\n"
    #     )
    _plot_series_with_baselines(
        series=[ours_test, goat_test, goatcw_test],
        labels=["Ours-FR", "GOAT", "GOAT-Classwise"],
        baselines=[(ours_st, ours_st_all)],
        ref_line_value=(EM_acc * 100.0 if EM_acc is not None else None),
        ref_line_label=f"EM ({args.em_match})",
        ref_line_style="--",
        title=f"CovType: Target Accuracy (ST: {args.label_source}; Cluster Map: {args.em_match})",
        ylabel="Accuracy", xlabel="Domain Index",
        save_path=os.path.join(plot_dir, f"test_acc_dim{args.small_dim}_gen{args.generated_domains}_{args.label_source}_{args.em_match}.png"),
    )

    # ----- Log -----
    os.makedirs("logs", exist_ok=True)
    elapsed = round(time.time() - t0, 2)
    with open(f"logs/covtype_exp_{args.log_file}.txt", "a") as f:
        f.write(
            f"seed{args.seed}with{gt_domains}gt{generated_domains}generated,{elapsed},"
            f"OursFR:{round((ours_test[-1] if ours_test else 0.0), 2)},GOAT:{round((goat_test[-1] if goat_test else 0.0), 2)},"
            f"GOATCW:{round((goatcw_test[-1] if goatcw_test else 0.0), 2)}\n"
        )


def run_color_mnist_experiment(gt_domains: int, generated_domains: int, args=None):
    """
    Colored-MNIST experiment mirroring the portraits pipeline.
    - VAE encoder → compressed classifier
    - Encode real domains; shared PCA
    - Fit+select/ensemble EM on target embedding; cache bundle
    - Run Ours (FR/ETA), GOAT, GOAT-Classwise
    - Plot + log
    """
    t0 = time.time()
    shift = 1
    total_domains = 20

    (
        src_tr_x, src_tr_y,
        src_val_x, src_val_y,
        dir_inter_x, dir_inter_y,
        dir_inter_x2, dir_inter_y2,
        trg_val_x, trg_val_y,
        trg_test_x, trg_test_y,
    ) = ColorShiftMNIST(shift=shift)

    # Derive evenly spaced “real” intermediates across the provided directional inter data
    inter_x, inter_y = transform_inter_data(dir_inter_x, dir_inter_y, 0, shift, interval=len(dir_inter_x) // total_domains, n_domains=total_domains)
    # Pack source/target
    src_x = np.concatenate([src_tr_x, src_val_x])
    src_y = np.concatenate([src_tr_y, src_val_y])
    tgt_x = np.concatenate([trg_val_x, trg_test_x])
    tgt_y = np.concatenate([trg_val_y, trg_test_y])

    # Datasets (images → tensors)
    src_trainset = EncodeDataset(src_x, src_y.astype(int), ToTensor())
    tgt_trainset = EncodeDataset(tgt_x, tgt_y.astype(int), ToTensor())


    # ----- Encoder & compressed classifier -----
    encoder = ENCODER().to(device)

    def infer_encoder_out_dim(encoder, dataset, device, n=1):
        encoder = encoder.to(device).eval()
        with torch.no_grad():
            # pull n items through __getitem__ so transforms run
            xb = torch.stack([dataset[i][0] for i in range(n)], dim=0)  # (N,C,H,W)
            xb = xb.to(device).float()
            z = encoder(xb)
            z = z.reshape(z.size(0), -1)
            return int(z.size(1))

    # one sample is enough
    enc_out_dim = infer_encoder_out_dim(encoder, src_trainset, device, n=1)

    # _ex = torch.as_tensor(src_trainset.data[:2]).float()
    # enc_out_dim = infer_encoder_out_dim(encoder, _ex, device)
    if enc_out_dim < args.small_dim:
        args.small_dim = enc_out_dim
    model_dir = f"color_mnist/cache{args.ssl_weight}"
    os.makedirs(model_dir, exist_ok=True)

    

    model_name = f"ssl{args.ssl_weight}_dim{args.small_dim}.pth"

    source_model = get_source_model(
        args,
        src_trainset, tgt_trainset,
        n_class=np.unique(src_trainset.targets).shape[0],
        mode="mnist",             # reuse MNIST head/training recipe
        encoder=encoder,
        epochs=20,
        model_path=os.path.join(model_dir, model_name),
        target_dataset=tgt_trainset,
        force_recompute=False,
        compress=True,
        in_dim=enc_out_dim,
        out_dim=args.small_dim,
    )
    # breakpoint()
    # Reference encoder (frozen)
    ref_model = source_model
    ref_encoder = nn.Sequential(
        ref_model.encoder,
        nn.Flatten(start_dim=1),
        getattr(ref_model, 'compressor', nn.Identity())
    ).eval()

    # Intermediates selector
    def get_domains(n_domains: int) -> List[Dataset]:
        if n_domains == total_domains:
            domain_idx = list(range(n_domains))
        else:
            # evenly spaced interior points
            domain_idx = [total_domains // (n_domains + 1) * i for i in range(1, n_domains + 1)]
        interval = len(inter_x) // total_domains
        domains = []
        for i in domain_idx:
            start, end = i * interval, (i + 1) * interval
            domains.append(EncodeDataset(inter_x[start:end], inter_y[start:end].astype(int), ToTensor()))
        return domains

    all_sets = get_domains(gt_domains)
    all_sets.append(tgt_trainset)

    # Encode & shared PCA
    _e_src, _e_tgt, encoded_intersets = encode_all_domains(
        src_trainset, tgt_trainset, all_sets, deg_idx=0,
        encoder=ref_encoder,
        cache_dir=f"color_mnist/cache{args.ssl_weight}/small_dim{args.small_dim}/",
        target=1,
        force_recompute=False,
        args=args
    )
    shared_pca = fit_global_pca(
        domains=encoded_intersets, classes=None, pool="auto", n_components=2,
        per_domain_cap=10000, random_state=args.seed if hasattr(args, "seed") else 0,
    )
    args.shared_pca = shared_pca

    # Optional prototype mapping
    if getattr(args, "em_match", "pseudo") == "prototypes":
        mu_s, Sigma_s, priors_s = fit_source_gaussian_params(X=_e_src.data, y=_e_src.targets)
        args._cached_source_stats = (mu_s, Sigma_s, priors_s)

    # Teacher pseudo-labels for mapping (not used for supervised training on target)
    with torch.no_grad():
        teacher = copy.deepcopy(ref_model).to(device).eval()
        pseudo_labels, _ = get_pseudo_labels(
            tgt_trainset, teacher,
            confidence_q=getattr(args, "pseudo_confidence_q", 0.9),
            device_override=device,
        )
    args._cached_pseudolabels = pseudo_labels.cpu().numpy()
    # EM model grid on target embedding
    em_models = fit_many_em_on_target(
        _e_tgt,
        K_list=[int(src_trainset.targets.max()) + 1],
        cov_types=getattr(args, "em_cov_types", ["diag"]),
        seeds=getattr(args, "em_seeds", [0, 1, 2]),
        pool="gap",
        pca_dims=getattr(args, "em_pca_dims", [None]),
        reg=1e-4, max_iter=300, rng_base=args.seed, args=args
    )
    best = _select_best_em(em_models, criterion="cost")
    if getattr(args, "em_do_ensemble", True) and len(em_models) > 1:
        trimmed, w = _trim_em_models_by_bic(em_models, max_delta_bic=getattr(args, "em_trim_delta_bic", 10.0))
        P_ens, _ = _ensemble_posteriors_trimmed(trimmed, w)
        labels_ens = P_ens.argmax(axis=1).astype(int)
        em_bundle = EMBundle(
            key="multi_em_trimmed", em_res=best["em_res"], mapping=best.get("mapping", None),
            labels_em=labels_ens, P_soft=P_ens,
            info={"criterion": "bic+trim", "bic_best": float(best["bic"])}
        )
    else:
        em_bundle = EMBundle(
            key="multi_em_best", em_res=best["em_res"], mapping=best.get("mapping", None),
            labels_em=np.asarray(best["labels_mapped"], dtype=int),
            P_soft=np.asarray(best["mapped_soft"], dtype=float),
            info={"criterion": getattr(args, "em_select", "bic"), "bic_best": float(best["bic"])}
        )
    args._shared_em = em_bundle
    _apply_em_bundle_to_target(em_bundle, _e_tgt, tgt_trainset)

    # Print EM accuracy (mapped labels vs true target labels)
    y_true = np.asarray(tgt_trainset.targets, dtype=int)
    y_em = np.asarray(em_bundle.labels_em, dtype=int)
    em_acc_now = float((y_em == y_true).mean())
    print(f"[ColorMNIST] EM→class accuracy: {em_acc_now:.4f}")
    # Also report the best achievable one-to-one mapping accuracy from raw EM clusters
    try:
        em_res_ref = best.get("em_res", {})
        if "labels" in em_res_ref:
            y_true_np = np.asarray(y_true, dtype=int)
            best_acc, _best_map, _C = best_mapping_accuracy(em_res_ref["labels"], y_true_np)
            print(f"[MainAlgo] Best one-to-one mapping accuracy: {best_acc:.4f}, current em accuracy: {em_acc_now:.4f}")
    except Exception as e:
        print(f"[ColorMNIST] best-mapping computation failed: {e}")

    # ---------- Run methods ----------
    # Ours-FR
    set_all_seeds(args.seed)
    ours_src = copy.deepcopy(ref_model);  ours_cp = copy.deepcopy(ref_model)
    ours_train, ours_test, ours_st, ours_st_all, ours_gen, EM_acc = run_main_algo(
        ours_cp, ours_src, src_trainset, tgt_trainset, all_sets, 0,
        generated_domains, epochs=5, target=1, args=args, gen_method="fr"
    )
    # GOAT-Classwise
    set_all_seeds(args.seed)
    goatcw_src = copy.deepcopy(ref_model);  goatcw_cp = copy.deepcopy(goatcw_src)
    goatcw_train, goatcw_test, goatcw_st, goatcw_st_all, goatcw_gen, EM_acc_goatcw = run_goat_classwise(
        goatcw_cp, goatcw_src, src_trainset, tgt_trainset, all_sets, 0,
        generated_domains, epochs=5, target=1, args=args
    )

    # Ours-ETA
    set_all_seeds(args.seed)
    ours_eta_src = copy.deepcopy(ref_model);  ours_eta_cp = copy.deepcopy(ref_model)
    ours_eta_train, ours_eta_test, ours_eta_st, ours_eta_st_all, ours_eta_gen, EM_acc_eta = run_main_algo(
        ours_eta_cp, ours_eta_src, src_trainset, tgt_trainset, all_sets, 0,
        generated_domains, epochs=5, target=1, args=args, gen_method="natural"
    )



    # GOAT
    set_all_seeds(args.seed)
    goat_src = copy.deepcopy(ref_model);  goat_cp = copy.deepcopy(goat_src)
    goat_train, goat_test, goat_st, goat_st_all, goat_gen = run_goat(
        goat_cp, goat_src, src_trainset, tgt_trainset, all_sets, 0,
        generated_domains, epochs=5, target=1, args=args
    )


    # ---------- Plot ----------
    plot_dir = f"plots/color_mnist/"
    os.makedirs(plot_dir, exist_ok=True)
    _plot_series_with_baselines(
        series=[ours_test,  goat_test, goatcw_test,ours_eta_test,],
        labels=["Ours-FR", "GOAT", "GOAT-Classwise", "Ours-ETA"],
        baselines=[(ours_st, ours_st_all)],
        ref_line_value=(EM_acc * 100.0 if EM_acc is not None else None),
        ref_line_label=f"EM ({args.em_match})",
        ref_line_style="--",
        title=f"Colored-MNIST: Target Accuracy (ST: {args.label_source}; Cluster Map: {args.em_match})",
        ylabel="Accuracy", xlabel="Domain Index",
        save_path=os.path.join(plot_dir, f"test_acc_dim{args.small_dim}_gen{args.generated_domains}_{args.label_source}_{args.em_match}.png"),
    )

    # ---------- Log ----------
    os.makedirs("logs", exist_ok=True)
    elapsed = round(time.time() - t0, 2)
    with open(f"logs/color_{args.log_file}.txt", "a") as f:
        f.write(
            f"seed{args.seed}with{gt_domains}gt{generated_domains}generated,{elapsed},"
            f"OursFR:{round((ours_test[-1] if ours_test else 0.0), 2)},GOAT:{round((goat_test[-1] if goat_test else 0.0), 2)},"
            f"GOATCW:{round((goatcw_test[-1] if goatcw_test else 0.0), 2)}\n"
        )



# -------------------------------------------------------------
# Main / CLI
# -------------------------------------------------------------

def main(cli_args: argparse.Namespace) -> None:
    set_all_seeds(cli_args.seed)
    # breakpoint()
    print(cli_args)

    if cli_args.dataset == "mnist":
        if cli_args.mnist_mode == "normal":
            run_mnist_experiment(cli_args.rotation_angle, cli_args.gt_domains, cli_args.generated_domains, args = cli_args)
        elif cli_args.mnist_mode == "ablation":
            run_mnist_ablation(cli_args.rotation_angle, cli_args.gt_domains, cli_args.generated_domains)
        else:
            raise ValueError(f"Unknown mnist-mode: {cli_args.mnist_mode}")
    elif cli_args.dataset == "portraits":
        run_portraits_experiment(cli_args.gt_domains, cli_args.generated_domains, args=cli_args)
    elif cli_args.dataset == "covtype":
        run_covtype_experiment(cli_args.gt_domains, cli_args.generated_domains, args=cli_args)
    elif cli_args.dataset == "color_mnist":
        run_color_mnist_experiment(cli_args.gt_domains, cli_args.generated_domains, args=cli_args)
    else:
        raise ValueError(f"Unknown dataset: {cli_args.dataset}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GOAT experiments (cleaned)")
    parser.add_argument("--dataset", choices=["mnist", "portraits", "covtype", "color_mnist"], default="mnist")
    parser.add_argument("--gt-domains", type=int, default=0)
    parser.add_argument("--generated-domains", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mnist-mode", choices=["normal", "ablation", "sweep", "compare"], default="normal")
    parser.add_argument("--rotation-angle", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-file", type=str, default="")
    parser.add_argument("--ssl-weight", type=float, default=0.1)
    parser.add_argument("--use-labels", action="store_true", help="Use true labels when available in generators")
    parser.add_argument("--diet", action="store_true", help="Run DIET to refine encoder before CE training")
    parser.add_argument("--small-dim", type=int, default=2048, help="Add a small-dim compressor before the head (0 to disable)")
    parser.add_argument("--label-source", choices=["pseudo", "em"], default="pseudo", help="For self-training, which labels to use for pseudo-labeling")
    parser.add_argument("--em-match",  choices=["pseudo", "prototypes","none"], default="pseudo", help="For self-training, which labels to use for pseudo-labeling")

    args = parser.parse_args()
    main(args)
