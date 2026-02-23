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
from typing import Optional, Tuple, List, Sequence, Iterable, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
# --- Visualization: PCA(2) of standardized target features + EM Gaussians ---
# Project-local deps (must exist in your repo)
from model import *
from train_model import *
from util import *  # noqa: F401,F403 (kept to preserve your helpers)
from ot_util import ot_ablation, generate_domains  # generation helpers
from a_star_util import *
from dataset import *

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from da_algo import *
import json
try:
    import kornia.augmentation as K
except Exception:
    K = None  # Kornia is optional; see build_augment()
import numpy as np
# Robust tqdm import with a no-op fallback

from em_utils import *
from check_dist import *
# -------------------------------------------------------------
# Global config / utilities
# -------------------------------------------------------------

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- em_registry.py (or near your EM helpers) ---
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


def _deg_idx_to_tuple(deg_idx):
    """Normalize deg_idx into a tuple so callers can pass ints or sequences."""
    if deg_idx is None:
        return None
    if isinstance(deg_idx, tuple):
        return deg_idx
    if isinstance(deg_idx, list):
        return tuple(deg_idx)
    try:
        return tuple(deg_idx)
    except TypeError:
        return (deg_idx,)


def _main_algo_cache_key(
    args,
    *,
    src_trainset,
    tgt_trainset,
    all_sets,
    deg_idx,
    generated_domains,
    epochs,
    target,
):
    dataset_ids = tuple(id(ds) for ds in all_sets)
    return (
        getattr(args, "dataset", None),
        getattr(args, "seed", None),
        getattr(args, "ssl_weight", None),
        getattr(args, "small_dim", None),
        getattr(args, "pseudo_confidence_q", None),
        getattr(args, "gt_domains", None),
        getattr(args, "label_source", None),
        target,
        _deg_idx_to_tuple(deg_idx),
        epochs,
        generated_domains,
        id(src_trainset),
        id(tgt_trainset),
        dataset_ids,
    )

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

# ---- helper: init Gaussian head from source features ----

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

    def _canon(ds):
        if ds is None:
            return None
        for attr in ("targets", "targets_em", "targets_pseudo"):
            val = getattr(ds, attr, None)
            if val is None:
                continue
            if not torch.is_tensor(val):
                val = torch.as_tensor(val)
            setattr(ds, attr, val.view(-1).long().cpu())
        return ds

    if args.dataset == 'mnist':
        e_src = get_encoded_dataset(
            src_trainset,
            cache_path=os.path.join(cache_dir, "encoded_0.pt"),
            encoder=encoder,
            force_recompute=force_recompute,
        )
        _canon(e_src)
        e_tgt = get_encoded_dataset(
            tgt_trainset,
            cache_path=os.path.join(cache_dir, f"encoded_{target}.pt"),
            encoder=encoder,
            force_recompute=force_recompute,
        )
        _canon(e_tgt)
    else:
        e_src = get_encoded_dataset(
            src_trainset,
            cache_path=os.path.join(cache_dir, "encoded_source.pt"),
            encoder=encoder,
            force_recompute=force_recompute,
        )
        _canon(e_src)
        e_tgt = get_encoded_dataset(
            tgt_trainset,
            cache_path=os.path.join(cache_dir, f"encoded_target.pt"),
            encoder=encoder,
            force_recompute=force_recompute,
        )
        _canon(e_tgt)

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
        _canon(encoded_intersets[-1])
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
        cache_dir = f"{args.dataset}/cache{args.ssl_weight}/small_dim{args.small_dim}/"
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
        # Recompute to avoid using stale caches with mismatched dimensions
        force_recompute=True,
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
            save_path=os.path.join(plot_dir, f"pca_dim{args.small_dim}_int{args.gt_domains}_gen{args.generated_domains}_real_domains_goat.png"),
            ground_truths=True,
            pca=getattr(args, "shared_pca", None)  # <<— SAME basis
        )

        # Synthetic chain: group per pair (generated_domains + 1) and drop the appended right endpoint
        # Synthetic chain: interleave real domains with synthetic ones
        # step_len = (#synthetic steps between a pair) + 1 (the appended right endpoint)
        step_len = int(generated_domains) + 1
        chain_for_plot = []

        n_pairs = len(encoded_intersets) - 1  # (source→inter1), (inter1→inter2), ..., (interN→target)

        for i in range(n_pairs):
            # For the very first pair, start with the left real domain (source)
            if i == 0:
                chain_for_plot.append(encoded_intersets[0])   # Domain 0 [Real]

            # Synthetic domains for this pair live in all_domains in contiguous blocks
            start = i * step_len
            chunk = all_domains[start:start + step_len]
            if not chunk:
                continue

            # Add the synthetic steps between encoded_intersets[i] and encoded_intersets[i+1]
            # (exclude the last element of chunk, which is the right endpoint real domain)
            if step_len > 1:
                chain_for_plot.extend(chunk[:-1])

            # Then append the right real endpoint of this pair
            # (intermediate real domain or final target)
            chain_for_plot.append(encoded_intersets[i + 1])

        # Keep EM labels intact; store pseudo labels in a separate field for plotting.
        tgt_pl, _ = get_pseudo_labels(
            encoded_intersets[-1],
            getattr(source_model, 'mlp', source_model),
            confidence_q=getattr(args, 'pseudo_confidence_q', 0.9),
        )
        encoded_intersets[-1].targets_pseudo = (
            tgt_pl.clone() if hasattr(tgt_pl, 'clone') else torch.as_tensor(tgt_pl, dtype=torch.long)
        )

        plot_pca_classes_grid(
            chain_for_plot,
            classes=(3,6, 8, 9) if 'mnist' in args.dataset else (0,1),
            save_path=os.path.join(plot_dir, f"pca_dim{args.small_dim}_int{args.gt_domains}_gen{args.generated_domains}_{args.label_source}_{args.em_match}_goat.png"),
            label_source='pseudo',
            pseudolabels=last_predictions,  # from self_train
            pca=getattr(args, "shared_pca", None)  # <<— SAME basis
        )

    return train_acc_by_domain, test_acc_by_domain, st_acc, st_acc_all, generated_acc




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

    Returns:
        train_curve, test_curve, st_curve, st_all_curve, generated_curve, em_acc
    so it can be wrapped by _wrap_result(..., has_em=True).
    """
    device = next(source_model.parameters()).device

    # -------------------- helpers --------------------
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
        n_steps = min(len(L) for L in list_of_lists)   # expected: n_inter + 1 incl. right endpoint
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
            Y = torch.cat([y.cpu().long() for y in Ys], dim=0)
            merged.append(DomainDataset(X, W, Y, Y))
        return merged

    # -------------------- Case 1: GST baseline (no synthetic domains, no EM) --------------------
    if generated_domains <= 0:
        set_all_seeds(args.seed)

        # Baseline on target only (GST-style)
        direct_acc, st_acc, train_acc_by_domain0, test_acc_by_domain0, _ = self_train(
            args,
            model_copy,
            [tgt_trainset],
            epochs=epochs,
            label_source="pseudo",
        )

        # Baseline on full chain of real domains
        set_all_seeds(args.seed)
        direct_acc_all, st_acc_all, train_acc_list_all, test_acc_list_all, _ = self_train(
            args,
            source_model,
            all_sets,
            epochs=epochs,
            label_source="pseudo",
        )

        generated_acc = 0.0
        acc_em_pseudo = float("nan")  # no EM mapping in this regime

        # Return length-1 curves to keep plotting/logging consistent
        return (
            [float(train_acc_by_domain0[-1])],   # train_curve
            [float(test_acc_by_domain0[-1])],    # test_curve
            [float(st_acc)],                     # st_curve
            [float(st_acc_all)],                 # st_all_curve
            [float(generated_acc)],              # generated_curve
            acc_em_pseudo,
        )

    # -------------------- Case 2: generated_domains > 0 (full GOAT-CW with EM available) --------------------

    # Reuse cached baselines + encodings when possible
    cache_key = _main_algo_cache_key(
        args,
        src_trainset=src_trainset,
        tgt_trainset=tgt_trainset,
        all_sets=all_sets,
        deg_idx=deg_idx,
        generated_domains=generated_domains,
        epochs=epochs,
        target=target,
    )
    if not hasattr(args, "_refrac_main_cache"):
        args._refrac_main_cache = {}
    cached_setup = args._refrac_main_cache.get(cache_key)

    if cached_setup is not None:
        direct_acc        = cached_setup["direct_acc"]
        st_acc            = cached_setup["st_acc"]
        direct_acc_all    = cached_setup["direct_acc_all"]
        st_acc_all        = cached_setup["st_acc_all"]
        e_src             = cached_setup["e_src"]
        e_tgt             = cached_setup["e_tgt"]
        encoded_intersets = cached_setup["encoded_intersets"]
        pseudolabels      = cached_setup["pseudolabels"]
    else:
        # 1) Baselines
        set_all_seeds(args.seed)
        direct_acc, st_acc, train_acc_by_domain0, test_acc_by_domain0, _ = self_train(
            args,
            model_copy,
            [tgt_trainset],
            epochs=epochs,
            label_source="pseudo",
        )

        set_all_seeds(args.seed)
        direct_acc_all, st_acc_all, train_acc_list_all, test_acc_list_all, _ = self_train(
            args,
            source_model,
            all_sets,
            epochs=epochs,
            label_source="pseudo",
        )

        if abs(st_acc - st_acc_all) > 1e-4:
            print(f"[GOAT-CW] Warning: st_acc ({st_acc}) != st_acc_all ({st_acc_all})")

        # 2) Encode domains once (encoder → flatten → compressor)
        if args.dataset != "mnist":
            cache_dir = f"{args.dataset}/cache{args.ssl_weight}/small_dim{args.small_dim}/"
            plot_dir  = f"plots/{args.dataset}/"
        else:
            cache_dir = f"cache{args.ssl_weight}/target{target}/small_dim{args.small_dim}/"
            plot_dir  = f"plots/target{target}/"
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(plot_dir,  exist_ok=True)

        e_src, e_tgt, encoded_intersets = encode_all_domains(
            src_trainset,
            tgt_trainset,
            all_sets,
            deg_idx,
            nn.Sequential(
                source_model.encoder,
                nn.Flatten(start_dim=1),
                getattr(source_model, "compressor", nn.Identity()),
            ),
            cache_dir,
            target,
            force_recompute=False,
            args=args,
        )

        # 3) Teacher pseudo-labels on TARGET (diagnostics only; not EM)
        with torch.no_grad():
            teacher = copy.deepcopy(source_model).to(device).eval()
            pseudo_labels, _ = get_pseudo_labels(
                tgt_trainset,
                teacher,
                confidence_q=getattr(args, "pseudo_confidence_q", 0.9),
                device_override=device,
            )
        pseudolabels = pseudo_labels.cpu().numpy()

        args._refrac_main_cache[cache_key] = {
            "direct_acc": direct_acc,
            "st_acc": st_acc,
            "direct_acc_all": direct_acc_all,
            "st_acc_all": st_acc_all,
            "e_src": e_src,
            "e_tgt": e_tgt,
            "encoded_intersets": encoded_intersets,
            "pseudolabels": pseudolabels,
        }

    # Plot/cache directories (for synthetic runs only)
    if args.dataset != "mnist":
        cache_dir = f"{args.dataset}/cache{args.ssl_weight}/small_dim{args.small_dim}/"
        plot_dir  = f"plots/{args.dataset}/"
    else:
        cache_dir = f"cache{args.ssl_weight}/target{target}/small_dim{args.small_dim}/"
        plot_dir  = f"plots/target{target}/"
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(plot_dir,  exist_ok=True)

    # Make sure encoded domains carry the right labels for class-wise splits
    raw_domains = [src_trainset] + all_sets
    enc_domains = encoded_intersets
    if len(raw_domains) != len(enc_domains):
        raise RuntimeError(
            f"raw/encoded domains count mismatch ({len(raw_domains)=} vs {len(enc_domains)=}); "
            "encoded_intersets should include source + intermediates + target."
        )

    for raw_ds, enc_ds in zip(raw_domains, enc_domains):
        if hasattr(raw_ds, "targets_em"):
            labels_em = torch.as_tensor(raw_ds.targets_em)
        else:
            labels_em = torch.as_tensor(raw_ds.targets)
        enc_ds.targets_em = labels_em.to(enc_ds.data.device)
        enc_ds.targets    = torch.as_tensor(raw_ds.targets).to(enc_ds.data.device)

    # Attach pseudo-labels on target
    setattr(args, "_cached_pseudolabels", pseudolabels)
    # Here we assume tgt_trainset.targets_em was set upstream by EM mapping
    e_tgt.targets_em = tgt_trainset.targets_em.to(device)

    # K: number of classes from source
    K = int(e_src.targets.max().item()) + 1
    y_true = e_tgt.targets.cpu().numpy() if torch.is_tensor(e_tgt.targets) else np.asarray(e_tgt.targets)
    e_tgt.targets_pseudo = torch.as_tensor(pseudolabels, dtype=torch.long)
    tgt_trainset.targets_pseudo = e_tgt.targets_pseudo.cpu().clone()

    acc_em_pseudo = (
        tgt_trainset.targets_em == torch.as_tensor(
            y_true,
            device=tgt_trainset.targets_em.device,
            dtype=tgt_trainset.targets_em.dtype,
        )
    ).to(torch.float32).mean().item()
    print(f"[GOAT-CW] EM→class (pseudo mapping) accuracy: {acc_em_pseudo:.4f}")

    # -------------------- class-wise generation --------------------
    generated_acc = 0.0
    all_domains: List[Dataset] = []

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

            chain_c, _, _ = generate_domains(
                generated_domains,
                s_c,
                t_c,
            )

            # force global class id c
            for D in chain_c:
                y_global = torch.full((len(D.targets),), c, dtype=torch.long)
                D.targets = y_global
                D.targets_em = y_global.clone()

            if chain_c:
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
            e_tgt.targets,
            e_tgt.targets_em,
        )

    # -------------------- train on merged synthetic chain --------------------
    set_all_seeds(args.seed)
    _, generated_acc, train_acc_by_domain, test_acc_by_domain, domain_stats, last_prediction = self_train(
        args,
        source_model.mlp,
        all_domains,
        epochs=epochs,
        label_source=getattr(args, "label_source", "pseudo"),
        use_labels=getattr(args, "use_labels", False),
        return_stats=True,
    )

    # plotting code unchanged ...

    return (
        train_acc_by_domain,
        test_acc_by_domain,
        st_acc,
        st_acc_all,
        generated_acc,
        acc_em_pseudo,
    )






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
    # multi-EM controls (kept for API compatibility; not used here)
    use_multi_em: bool = False,
    em_cov_types: Tuple[str, ...] = ("diag",),
    em_K_list: Optional[List[int]] = None,
    em_seeds: Tuple[int, ...] = (0, 1, 2, 3, 4),
    em_pca_dims: Tuple[Optional[int], ...] = (None,),
    em_select: str = "bic",
    em_ensemble_weights: str = "bic",
    # sharing controls; EM bundles are built outside (e.g., in run_mnist_experiment)
    use_shared_em: bool = True,
    shared_em_cfg: dict = None,
):
    device = next(source_model.parameters()).device

    # 0) Sanity: identical initialization?
    for p1, p2 in zip(model_copy.parameters(), source_model.parameters()):
        if not torch.equal(p1, p2):
            print("[run_main_algo] Warning: model_copy and source_model have different initial weights.")
            break

    # 1) Baselines
    set_all_seeds(args.seed)
    direct_acc, st_acc, train_acc_by_domain0, test_acc_by_domain0, _ = self_train(
        args, model_copy, [tgt_trainset], epochs=epochs, label_source="pseudo"
    )

    set_all_seeds(args.seed)
    direct_acc_all, st_acc_all, train_acc_list_all, test_acc_list_all, _ = self_train(
        args, source_model, all_sets, epochs=epochs, label_source="pseudo"
    )

    if abs(st_acc - st_acc_all) > 1e-4:
        print(f"[run_main_algo] Warning: st_acc ({st_acc}) != st_acc_all ({st_acc_all})")

    # 2) Teacher pseudo-labels on TARGET (diagnostics only)
    em_teacher = copy.deepcopy(source_model).to(device).eval()
    with torch.no_grad():
        pseudo_labels, _ = get_pseudo_labels(
            tgt_trainset,
            em_teacher,
            confidence_q=getattr(args, "pseudo_confidence_q", 0.9),
            device_override=device,
        )
    pseudolabels = pseudo_labels.cpu().numpy()

    # 3) Cache dirs
    if args.dataset != "mnist":
        cache_dir = f"{args.dataset}/cache{args.ssl_weight}/small_dim{args.small_dim}"
        plot_dir = f"plots/{args.dataset}/"
    else:
        cache_dir = f"cache{args.ssl_weight}/target{target}/small_dim{args.small_dim}/"
        plot_dir = f"plots/target{target}/"
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # 4) Encode domains once (encoder → flatten → compressor)
    e_src, e_tgt, encoded_intersets = encode_all_domains(
        src_trainset,
        tgt_trainset,
        all_sets,
        deg_idx,
        nn.Sequential(
            source_model.encoder,
            nn.Flatten(start_dim=1),
            source_model.compressor if hasattr(source_model, "compressor") else nn.Identity(),
        ),
        cache_dir,
        target,
        force_recompute=False,
        args=args,
    )

    # Ensure EM labels for target
    if hasattr(tgt_trainset, "targets_em") and tgt_trainset.targets_em is not None:
        e_tgt.targets_em = tgt_trainset.targets_em.clone()
    else:
        print("[run_main_algo] Warning: tgt_trainset.targets_em is missing; EM labels not set for target.")

    # Ensure EVERY encoded domain has .targets_em before calling generators
    for ds in encoded_intersets:
        if getattr(ds, "targets_em", None) is not None:
            breakpoint()
            continue

        if getattr(ds, "targets", None) is not None:
            t = ds.targets
            ds.targets_em = t.clone().long() if torch.is_tensor(t) else torch.as_tensor(t, dtype=torch.long)
            breakpoint()
            continue

        if getattr(ds, "targets_pseudo", None) is not None:
            t = ds.targets_pseudo
            ds.targets_em = t.clone().long() if torch.is_tensor(t) else torch.as_tensor(t, dtype=torch.long)
            breakpoint()
            continue

        raise ValueError(
            "[run_main_algo] Encoded domain is missing targets, targets_em, and targets_pseudo; "
            "natural / FR generators cannot proceed."
        )

    # 5) Diagnostics: pseudo-label accuracy on target
    y_true = e_tgt.targets
    if torch.is_tensor(y_true):
        y_true = y_true.cpu()
    preds = torch.as_tensor(pseudolabels, device=y_true.device)
    acc_pl = (preds == y_true).float().mean().item()
    print(f"Pseudo-label accuracy (teacher on target): {acc_pl:.4f}")

    # 6) Determine K from SOURCE (stable)
    try:
        K_infer = int(e_src.targets.max().item()) + 1
    except Exception:
        K_infer = 10
    if not em_K_list:
        em_K_list = [K_infer]

    # 7) Attach pseudo labels to target datasets (for plotting / diagnostics)
    e_tgt.targets_pseudo = torch.as_tensor(pseudolabels, dtype=torch.long)
    tgt_trainset.targets_pseudo = e_tgt.targets_pseudo.cpu().clone()

    # 8) EM→class accuracy on TARGET from precomputed labels
    if hasattr(e_tgt, "targets_em") and e_tgt.targets_em is not None:
        acc_em_pseudo = (
            e_tgt.targets_em.cpu() == torch.as_tensor(y_true, dtype=e_tgt.targets_em.dtype)
        ).to(torch.float32).mean().item()
    else:
        acc_em_pseudo = float("nan")
    print(f"[MainAlgo] EM→class (mapped) accuracy on target: {acc_em_pseudo:.4f}")

    # Optional: check presence of target bundle
    if use_shared_em:
        if hasattr(args, "_shared_em_per_domain") and target in getattr(args, "_shared_em_per_domain", {}):
            em_bundle_target = args._shared_em_per_domain[target]
        else:
            em_bundle_target = getattr(args, "_shared_em", None)
        if em_bundle_target is None:
            print("[run_main_algo] Warning: no shared EM bundle found for target domain.")

    # 9) Evaluate current source_model on target images (sanity)
    tgt_loader_eval = DataLoader(
        tgt_trainset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    _, tgt_acc_after = test(tgt_loader_eval, source_model)
    print(f"[MainAlgo] Baseline target accuracy before synthetic training: {tgt_acc_after:.4f}")

    # 10) Generate synthetic domains
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
        breakpoint()
        out = _gen_fn(
            generated_domains,
            encoded_intersets[i],
            encoded_intersets[i + 1],
            cov_type="full",
            save_path=plot_dir,
            args=args,
        )
        pair_domains = out[0]
        domain_stats = out[2]
        synthetic_domains += pair_domains

    if not synthetic_domains:
        return direct_acc, st_acc, direct_acc_all, st_acc_all, 0.0, acc_em_pseudo

    # 11) Self-training on synthetic domains (EM labels drive training)
    set_all_seeds(args.seed)
    direct_acc_syn, generated_acc, train_acc_by_domain, test_acc_by_domain, last_predictions = self_train(
        args,
        source_model.mlp,
        synthetic_domains,
        epochs=epochs,
        label_source=getattr(args, "label_source", "em"),
    )

    # 12) Plots
    plot_pca_classes_grid(
        encoded_intersets,
        classes=(3, 6, 8, 9) if "mnist" in args.dataset else (0, 1),
        save_path=os.path.join(
            plot_dir,
            f"pca_dim{args.small_dim}_int{args.gt_domains}_gen{args.generated_domains}_real_domains.png",
        ),
        label_source="real",
        ground_truths=True,
        pca=getattr(args, "shared_pca", None),
    )

    # --- build chain with real + synthetic domains interleaved ---
    if synthetic_domains:
        step_len = int(generated_domains) + 1
        chain_for_plot: List[Dataset] = []
        n_pairs = len(encoded_intersets) - 1

        for i in range(n_pairs):
            if i == 0:
                chain_for_plot.append(encoded_intersets[0])  # first real domain

            start = i * step_len
            chunk = synthetic_domains[start:start + step_len]
            if not chunk:
                continue

            if step_len > 1:
                chain_for_plot.extend(chunk[:-1])           # synthetic only (drop appended endpoint)
            chain_for_plot.append(encoded_intersets[i + 1])  # right real endpoint

        # (a) labels used during self_train (typically 'em')
        plot_pca_classes_grid(
            chain_for_plot,
            classes=(3, 6, 8, 9) if "mnist" in args.dataset else (0, 1),
            save_path=os.path.join(
                plot_dir,
                f"pca_dim{args.small_dim}_int{args.gt_domains}_gen{args.generated_domains}_"
                f"{args.label_source}_{getattr(args, 'em_match', 'pseudo')}_"
                f"{args.em_select}{'_em-ensemble' if args.em_ensemble else ''}_{_method}.png",
            ),
            label_source=getattr(args, "label_source", "em"),
            pseudolabels=last_predictions,
            pca=getattr(args, "shared_pca", None),
        )

        # (b) explicit EM labels
        plot_pca_classes_grid(
            chain_for_plot,
            classes=(3, 6, 8, 9) if "mnist" in args.dataset else (0, 1),
            save_path=os.path.join(
                plot_dir,
                f"pca_dim{args.small_dim}_int{args.gt_domains}_gen{args.generated_domains}_"
                f"emlabels_{args.em_select}{'_em-ensemble' if args.em_ensemble else ''}_{_method}.png",
            ),
            label_source="em",
            pca=getattr(args, "shared_pca", None),
        )

        # (c) teacher pseudo labels along chain
        plot_pca_classes_grid(
            chain_for_plot,
            classes=(3, 6, 8, 9) if "mnist" in args.dataset else (0, 1),
            save_path=os.path.join(
                plot_dir,
                f"pca_dim{args.small_dim}_int{args.gt_domains}_gen{args.generated_domains}_source_pseudo_{_method}.png",
            ),
            label_source="pseudo",
            pseudolabels=pseudolabels,
            pca=getattr(args, "shared_pca", None),
        )

    return (
        train_acc_by_domain,
        test_acc_by_domain,
        st_acc,
        st_acc_all,
        generated_acc,
        acc_em_pseudo,
    )


# ---------------- Generic plotting helper: N series + per-method baselines ----------------
# def _plot_series_with_baselines(
#     series,
#     labels,
#     baselines=None,  # list of (st, st_all) per series (optional)
#     ref_line_value=None,
#     ref_line_label=None,
#     ref_line_style="--",
#     title="",
#     ylabel="Accuracy",
#     xlabel="Training Domain Index",
#     save_path=None,
#     # New: distinguish real vs synthetic domains
#     synth_per_segment: int = None,   # number of synthetic domains between two real domains
#     n_real_segments: int = None,     # number of real gaps (src→inter1, inter1→inter2, ..., →tgt)
# ):


#     def _to_array(v):
#         return np.array([np.nan if x is None else float(x) for x in (v or [])], dtype=float)

#     S = [ _to_array(s) for s in (series or []) ]
#     if not S:
#         print(f"[plot] Skip {title}: no data.")
#         return

#     L = max(len(s) for s in S)
#     if L == 0:
#         print(f"[plot] Skip {title}: empty series.")
#         return
#     S = [ (np.pad(s, (0, L - len(s)), constant_values=np.nan) if len(s) < L else s) for s in S ]
#     x = np.arange(0, L, dtype=int)

#     plt.figure()
#     colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
#     markers = ['o', 's', '^', 'D', 'v', '>', '<', 'P', 'X']

#     n = len(S)
#     # Compute boundaries of real domains along the x-axis if info provided
#     boundaries = None
#     if (
#         synth_per_segment is not None
#         and n_real_segments is not None
#         and isinstance(synth_per_segment, int)
#         and isinstance(n_real_segments, int)
#         and synth_per_segment >= 0
#         and n_real_segments >= 0
#     ):
#         # Real domain positions in x (including source at 0 and target at the end)
#         # Each segment contributes `synth_per_segment` synthetics + 1 real endpoint,
#         # plus an extra initial point (direct accuracy) at index 0.
#         step = synth_per_segment + 1
#         boundaries = [k * step for k in range(n_real_segments + 1)]
#         # de-duplicate and sort in case synth_per_segment == 0
#         boundaries = sorted(set(boundaries))
#         # Keep within current plot length
#         boundaries = [b for b in boundaries if b < L]

#     for i, s in enumerate(S):
#         color = colors[i % len(colors)]
#         marker = markers[i % len(markers)]
#         label  = labels[i] if labels and i < len(labels) else f"Series {i+1}"
#         plt.plot(x, s, marker=marker, linewidth=1.8, label=label, color=color)

#         # Highlight real domain points, if boundary info is available
#         if boundaries:
#             # valid points only (avoid NaNs)
#             idxs = [b for b in boundaries if b < len(s) and np.isfinite(s[b])]
#             if idxs:
#                 plt.scatter(
#                     idxs,
#                     s[idxs],
#                     s=36,
#                     facecolors='white',
#                     edgecolors=color,
#                     linewidths=1.2,
#                     zorder=4,
#                 )

#         if baselines and i < len(baselines) and baselines[i] is not None:
#             st, st_all = baselines[i]
#             if st is not None:
#                 plt.axhline(float(st), linestyle=':', linewidth=1.5, color=color, alpha=0.9, label=f"st")
#             if st_all is not None:
#                 plt.axhline(float(st_all), linestyle='--', linewidth=1.5, color=color, alpha=0.9, label=f"st_all")

#     # Draw vertical lines at real-domain boundaries (common for all series)
#     if boundaries:
#         for b in boundaries:
#             plt.axvline(b, linestyle=':', linewidth=1.0, color='gray', alpha=0.7)

#     if ref_line_value is not None:
#         plt.axhline(float(ref_line_value), linestyle=ref_line_style, linewidth=1.6,
#                     color='k', alpha=0.7, label=(ref_line_label or "reference"))

#     plt.title(title)
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
#     # Compose legend, adding entries for real/synthetic cues if provided
#     handles, labels_ = plt.gca().get_legend_handles_labels()
#     if boundaries:
#         from matplotlib.lines import Line2D
#         extra = [
#             Line2D([0], [0], color='gray', linestyle=':', linewidth=1.0, label='Real boundary'),
#             Line2D([0], [0], marker='o', markerfacecolor='white', markeredgecolor='gray', linestyle='None', label='Real domain'),
#         ]
#         handles += extra
#         labels_ += [h.get_label() for h in extra]
#     plt.legend(handles, labels_)
#     plt.tight_layout()
#     if save_path:
#         plt.savefig(save_path, dpi=150)
#         print(f"[MNIST-EXP] Saved {save_path}")
#     plt.close()

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
    # New: distinguish real vs synthetic domains
    synth_per_segment: int = None,   # number of synthetic domains between two real domains
    n_real_segments: int = None,     # number of real gaps (src→inter1, inter1→inter2, ..., →tgt)
):
    import numbers

    def _to_array(v):
        # No data
        if v is None:
            return np.array([], dtype=float)

        # Torch tensor
        try:
            import torch
            if isinstance(v, torch.Tensor):
                if v.ndim == 0:
                    return np.array([float(v.item())], dtype=float)
                return v.detach().cpu().flatten().numpy().astype(float)
        except Exception:
            pass

        # Scalar (float, int, numpy scalar, etc.)
        if isinstance(v, numbers.Number):
            return np.array([float(v)], dtype=float)

        # Try as a generic iterable
        try:
            return np.array(
                [np.nan if x is None else float(x) for x in v],
                dtype=float,
            )
        except TypeError:
            # Fallback: treat as scalar
            return np.array([float(v)], dtype=float)

    S = [_to_array(s) for s in (series or [])]
    if not S:
        print(f"[plot] Skip {title}: no data.")
        return

    L = max(len(s) for s in S)
    if L == 0:
        print(f"[plot] Skip {title}: empty series.")
        return
    S = [
        (np.pad(s, (0, L - len(s)), constant_values=np.nan) if len(s) < L else s)
        for s in S
    ]
    x = np.arange(0, L, dtype=int)

    plt.figure()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    markers = ['o', 's', '^', 'D', 'v', '>', '<', 'P', 'X']

    n = len(S)
    # Compute boundaries of real domains along the x-axis if info provided
    boundaries = None
    if (
        synth_per_segment is not None
        and n_real_segments is not None
        and isinstance(synth_per_segment, int)
        and isinstance(n_real_segments, int)
        and synth_per_segment >= 0
        and n_real_segments >= 0
    ):
        step = synth_per_segment + 1
        boundaries = [k * step for k in range(n_real_segments + 1)]
        boundaries = sorted(set(boundaries))
        boundaries = [b for b in boundaries if b < L]

    for i, s in enumerate(S):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        label = labels[i] if labels and i < len(labels) else f"Series {i+1}"
        plt.plot(x, s, marker=marker, linewidth=1.8, label=label, color=color)

        if boundaries:
            idxs = [b for b in boundaries if b < len(s) and np.isfinite(s[b])]
            if idxs:
                plt.scatter(
                    idxs,
                    s[idxs],
                    s=36,
                    facecolors='white',
                    edgecolors=color,
                    linewidths=1.2,
                    zorder=4,
                )

        if baselines and i < len(baselines) and baselines[i] is not None:
            st, st_all = baselines[i]
            if st is not None:
                plt.axhline(
                    float(st),
                    linestyle=':',
                    linewidth=1.5,
                    color=color,
                    alpha=0.9,
                    label="st",
                )
            if st_all is not None:
                plt.axhline(
                    float(st_all),
                    linestyle='--',
                    linewidth=1.5,
                    color=color,
                    alpha=0.9,
                    label="st_all",
                )

    if boundaries:
        for b in boundaries:
            plt.axvline(b, linestyle=':', linewidth=1.0, color='gray', alpha=0.7)

    if ref_line_value is not None:
        plt.axhline(
            float(ref_line_value),
            linestyle=ref_line_style,
            linewidth=1.6,
            color='k',
            alpha=0.7,
            label=(ref_line_label or "reference"),
        )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)

    handles, labels_ = plt.gca().get_legend_handles_labels()
    if boundaries:
        from matplotlib.lines import Line2D
        extra = [
            Line2D([0], [0], color='gray', linestyle=':', linewidth=1.0, label='Real boundary'),
            Line2D(
                [0],
                [0],
                marker='o',
                markerfacecolor='white',
                markeredgecolor='gray',
                linestyle='None',
                label='Real domain',
            ),
        ]
        handles += extra
        labels_ += [h.get_label() for h in extra]
    plt.legend(handles, labels_)
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
    pca_fit_policy: str = "endpoints", # 'endpoints'|'all' (only used if pca is None)
    # NEW: distinguish real vs synthetic panels
    real_indices: "Sequence[int]|None" = None,   # indices of real domains; others treated as synthetic
    real_facecolor: str = "#f0f6ff",
    syn_facecolor: str = "#fff8f0",
    annotate_kind: bool = True,
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
            if hasattr(D, "targets_pseudo") and D.targets_pseudo is not None:
                return _to_np(D.targets_pseudo)
            if hasattr(D, "targets") and D.targets is not None:
                return _to_np(D.targets)
            if hasattr(D, "targets_em") and D.targets_em is not None:
                return _to_np(D.targets_em)
            if pseudolabels is not None:
                y = np.asarray(pseudolabels)
                n = len(D.data) if hasattr(D, "data") else len(y)
                if len(y) != n:
                    y = y[:n]
                return y
            return None
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
        y = get_labels_for_domain(D, j)
        if y is not None:
            y = np.asarray(y)
            n_samples = len(Xp)
            if len(y) != n_samples:
                if len(y) > n_samples:
                    y = y[:n_samples]
                else:
                    pad_val = -1 if y.ndim == 1 else 0
                    pad_width = (0, n_samples - len(y))
                    y = np.pad(y, pad_width, constant_values=pad_val)
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

    # ---------- infer real vs synthetic indices (if not provided) ----------
    real_set = None
    if isinstance(real_indices, (list, tuple, set)):
        real_set = set(int(i) for i in real_indices if 0 <= int(i) < cols)
    else:
        # Heuristics: if plotting only real domains, mark all as real.
        if ground_truths or label_source == 'real':
            real_set = set(range(cols))
        else:
            # Typical chain: [src] + synthetics + [tgt]
            if cols >= 2:
                real_set = {0, cols - 1}
            else:
                real_set = {0}

    # ---------- plot ----------
    fig_w = max(4, 3 * cols)
    fig, axs = plt.subplots(1, cols, figsize=(fig_w, 3.6), squeeze=False)
    axs = axs[0]
    cmap = plt.get_cmap('tab10')

    for j in range(cols):
        ax = axs[j]
        # Style: background and frame to indicate real vs synthetic
        if real_set is not None and j in real_set:
            ax.set_facecolor(real_facecolor)
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)
                spine.set_edgecolor('#2c3e50')
        else:
            ax.set_facecolor(syn_facecolor)
            for spine in ax.spines.values():
                spine.set_linewidth(1.0)
                spine.set_edgecolor('#999999')

        Xp = pooled_per_domain[j]
        y  = labels_per_domain[j]
        m  = masks_per_domain[j]

        if (y is None) or (m is None) or (not np.any(m)):
            kind = "Real" if (real_set is not None and j in real_set) else "Syn"
            ax.set_title(f"Domain {j} [{kind}]: no classes {tuple(classes)}")
            ax.axis('off')
            continue

        Xsel = Xp[m]
        ysel = y[m]
        # breakpoint()
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

        if annotate_kind:
            kind = "Real" if (real_set is not None and j in real_set) else "Syn"
            ax.set_title(f"Domain {j} [{kind}]")
        else:
            ax.set_title(f"Domain {j}")
        ax.set_xticks([]); ax.set_yticks([])
        if j == 0:
            # Extend legend with Real/Syn markers
            handles, labels_ = ax.get_legend_handles_labels()
            from matplotlib.patches import Patch
            extra = []
            if annotate_kind:
                extra = [
                    Patch(facecolor=real_facecolor, edgecolor='#2c3e50', label='Real domain'),
                    Patch(facecolor=syn_facecolor, edgecolor='#999999', label='Synthetic domain'),
                ]
            ax.legend(handles + extra, labels_ + [e.get_label() for e in extra], loc='best', fontsize=8)

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




def _to_numpy_1d_labels(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().reshape(-1)
    x = np.asarray(x)
    return x.reshape(-1)

def print_em_model_accuracies(em_models, true_labels):
    """
    em_models: list[dict] from fit_many_em_on_target; each has 'labels_mapped'
    true_labels: array-like or torch tensor of ground-truth target labels
    """
    y_true = _to_numpy_1d_labels(true_labels)
    rows = []
    for i, m in enumerate(em_models):
        y_pred = _to_numpy_1d_labels(m["labels_mapped"])
        if y_pred.shape != y_true.shape:
            raise ValueError(f"Shape mismatch for model {i}: y_pred {y_pred.shape} vs y_true {y_true.shape}")
        acc = (y_pred == y_true).mean()
        cfg = m.get("cfg", {})
        rows.append({
            "idx": i,
            "acc": acc,
            "seed": cfg.get("seed"),
            "cov": cfg.get("cov_type"),
            "K": cfg.get("K"),
            "bic": m.get("bic"),
            "final_ll": m.get("final_ll"),
        })

    # Pretty print
    for r in rows:
        print(f"[EM {r['idx']:02d}] acc={r['acc']*100:6.2f}% | seed={r['seed']} | cov={r['cov']} | K={r['K']} | "
              f"BIC={r['bic']} | final_ll={r['final_ll']}")

    # Also return rows if you want to sort/select programmatically
    return rows


def run_mnist_experiment(target: int, gt_domains: int, generated_domains: int, args=None):
    src_trainset = get_single_rotate(False, 0)
    tgt_trainset = get_single_rotate(False, target)
    model_dir = f"/data/common/yuenchen/GDA/mnist_models/"

    # ---- (A) Train / load compressed source model ----
    encoder = ENCODER().to(device)
    model_name_smalldim = f"src0_tgt{target}_ssl{args.ssl_weight}_dim{args.small_dim}.pth"
    source_model_smalldim = get_source_model(
        args,
        src_trainset,
        tgt_trainset,
        n_class=10,
        mode="mnist",
        encoder=encoder,
        epochs=10,
        model_path=f"{model_dir}/{model_name_smalldim}",
        target_dataset=tgt_trainset,
        force_recompute=False,
        compress=True,
        in_dim=25088,
        out_dim=args.small_dim,
    )

    # SAME reference for all runs
    ref_model = source_model_smalldim
    ref_encoder = nn.Sequential(
        ref_model.encoder,
        nn.Flatten(start_dim=1),
        getattr(ref_model, "compressor", nn.Identity()),
    ).eval()

    # ---- (B) Build real intermediate domains ----
    # all_sets: [inter1, inter2, ..., target]
    # deg_idx: [angle_inter1, ..., angle_target]
    all_sets, deg_idx = [], []
    for i in range(1, gt_domains + 1):
        angle = i * target // (gt_domains + 1)
        all_sets.append(get_single_rotate(False, angle))
        deg_idx.append(angle)
    all_sets.append(tgt_trainset)
    deg_idx.append(target)

    # ---- (C) Encode source + intermediates + target once ----
    cache_dir = f"cache{args.ssl_weight}/target{target}/small_dim{args.small_dim}/"
    e_src, e_tgt, encoded_intersets = encode_all_domains(
        src_trainset,
        tgt_trainset,
        all_sets,
        deg_idx,
        ref_encoder,
        cache_dir=cache_dir,
        target=target,
        force_recompute=False,
        args=args,
    )
    # encoded_intersets = [e_src, e_inter(angle1), ..., e_inter(angleK), e_tgt]

    # ---- (D) Shared PCA on real domains (unchanged) ----
    shared_pca = fit_global_pca(
        domains=encoded_intersets,
        classes=None,
        pool="auto",
        n_components=2,
        per_domain_cap=10000,
        random_state=args.seed if hasattr(args, "seed") else 0,
    )
    args.shared_pca = shared_pca

    # Optional: cache source Gaussian params for prototype matching
    if getattr(args, "em_match", "pseudo") == "prototypes":
        mu_s, Sigma_s, priors_s = fit_source_gaussian_params(
            X=e_src.data, y=e_src.targets
        )
        args._cached_source_stats = (mu_s, Sigma_s, priors_s)

    # ---- (E) Frozen teacher for pseudo-labels on every REAL domain ----
    with torch.no_grad():
        em_teacher = copy.deepcopy(ref_model).to(device).eval()

    # Align raw + encoded + angle:
    # raw_domains   = [src, inter1, ..., interK, tgt]
    # enc_domains   = [e_src, e_inter1, ..., e_interK, e_tgt]
    # angle_list    = [0, angle_inter1, ..., angle_interK, target]
    raw_domains = [src_trainset] + all_sets
    # enc_domains = encoded_intersets
    angle_list = [0] + deg_idx
    enc_domains = []
    for angle in angle_list:
        enc_path = os.path.join(cache_dir, f"encoded_{angle}.pt")
        if not os.path.exists(enc_path):
            raise FileNotFoundError(f"Expected encoded features for angle={angle} at {enc_path}, but file is missing.")
        enc_ds = torch.load(enc_path, weights_only=False)
        enc_domains.append(enc_ds)


    for ang, enc_ds in zip(angle_list, enc_domains):
        print(f"[DEBUG] angle={ang} encoded shape={enc_ds.data.shape}, first_norm={enc_ds.data[0].norm().item():.4f}")

    n_classes = int(src_trainset.targets.max().item()) + 1

    # Per-angle EM bundle storage
    em_bundles_by_angle: Dict[int, Any] = {}
    em_acc_by_angle: Dict[int, float] = {}

    # Keep teacher pseudo-labels on the final target as "canonical" if later needed
    with torch.no_grad():
        tgt_teacher_pl, _ = get_pseudo_labels(
            tgt_trainset,
            em_teacher,
            confidence_q=getattr(args, "pseudo_confidence_q", 0.9),
            device_override=device,
        )
    teacher_pl_target_np = tgt_teacher_pl.cpu().numpy()

    # Save original cached pseudolabels if present (for backwards compatibility)
    original_cached_pl = getattr(args, "_cached_pseudolabels", None)

    # ---- (E*) Fit EM + build bundles for each non-source domain ----
    for idx in range(1, len(raw_domains)):
        angle = angle_list[idx]
        raw_ds = raw_domains[idx]   # raw MNIST-rotate dataset at this angle
        enc_ds = enc_domains[idx]   # encoded DomainDataset at the same angle

        # Pseudo-labels on THIS raw domain using the teacher
        with torch.no_grad():
            pseudo_labels, _ = get_pseudo_labels(
                raw_ds,
                em_teacher,
                confidence_q=getattr(args, "pseudo_confidence_q", 0.9),
                device_override=device,
            )
        pseudo_np = pseudo_labels.cpu().numpy()

        # This is what build_em_bundle uses to align clusters → classes.
        # Set it per-domain, right before fitting EM for that domain.
        args._cached_pseudolabels = pseudo_np

        # ---- Fit multiple EMs on THIS encoded domain ----
        em_models = fit_many_em_on_target(
            enc_ds,
            K_list=[n_classes],
            cov_types=["diag"],
            seeds=[0, 1, 2],
            pool="gap",
            pca_dims=[None],
            reg=1e-4,
            max_iter=300,
            rng_base=args.seed,
            args=args,
        )

        # Per-model accuracies BEFORE bundling (diagnostic)
        rows = print_em_model_accuracies(em_models, raw_ds.targets)
        rows = [{**r, "original_idx": i} for i, r in enumerate(rows)]
        for r in rows:
            bic = r.get("bic")
            print(
                f"[EM angle ={angle:3d} idx={r['original_idx']:02d}] "
                f"acc={r['acc']*100:6.2f}% | seed={r.get('seed')} | cov={r.get('cov')} | "
                f"K={r.get('K')} | BIC={bic if bic is not None else 'NA'} | "
                f"final_ll={r.get('final_ll')}"
            )

        # ---- Build and apply EM ensemble for THIS domain ----
        em_bundle = build_em_bundle(em_models, args)
        apply_em_bundle_to_target(em_bundle, enc_ds, raw_ds)

        # Ensemble accuracy on THIS domain (encoded vs GT)
        em_acc = (
            enc_ds.targets_em
            == raw_ds.targets.to(enc_ds.targets_em.device)
        ).float().mean().item()
        em_bundles_by_angle[angle] = em_bundle
        em_acc_by_angle[angle] = em_acc
        print(
            f"[MNIST-EXP] EM ensemble accuracy at angle={angle}: "
            f"{em_acc * 100:.2f}%"
        )

        # Optional: direct check using em_bundle.labels_em if present
        if hasattr(em_bundle, "labels_em"):
            true_labels = raw_ds.targets.cpu().numpy()
            labels_ens = np.asarray(em_bundle.labels_em)
            ensemble_acc = (labels_ens == true_labels).mean()
            print(
                f"[DEBUG] angle={angle} ensemble_acc (labels_em vs GT): "
                f"{ensemble_acc * 100:.2f}%"
            )

    # ---- Restore "canonical" cached pseudo labels for downstream FR/Natural code ----
    # Use teacher pseudo labels on the FINAL target (what old code expected).
    args._cached_pseudolabels = teacher_pl_target_np

    # Expose bundles per domain (keyed by angle) to downstream code
    args._shared_em_per_domain = em_bundles_by_angle

    # Backwards-compatibility: a single shared EM for the target angle
    if target in em_bundles_by_angle:
        args._shared_em = em_bundles_by_angle[target]
    else:
        args._shared_em = None

    # For logging: initial EM accuracy on the final target (ensemble)
    em_acc_init = em_acc_by_angle.get(target, 0.0)
    print(
        f"[MNIST-EXP] Initial target EM accuracy ({args.em_match}): "
        f"{em_acc_init * 100:.2f}%"
    )

    # GOAT classwise
    set_all_seeds(args.seed)
    goat_cw_src = copy.deepcopy(ref_model)
    goat_cw_cp = copy.deepcopy(goat_cw_src)
    goatcw_train, goatcw_test, goatcw_st, goatcw_st_all, goatcw_gen, EM_acc_goatcw = run_goat_classwise(
        goat_cw_cp,
        goat_cw_src,
        src_trainset,
        tgt_trainset,
        all_sets,
        deg_idx,
        generated_domains,
        epochs=5,
        target=target,
        args=args,
    )

    # ---- (F) Prepare model copies for different methods ----
    # ETA / natural path
    our_source_eta = copy.deepcopy(ref_model)
    ours_copy_eta = copy.deepcopy(ref_model)

    set_all_seeds(args.seed)
    ours_eta_train, ours_eta_test, ours_eta_st, ours_eta_st_all, ours_eta_gen, EM_acc_eta = run_main_algo(
        ours_copy_eta,
        our_source_eta,
        src_trainset,
        tgt_trainset,
        all_sets,
        deg_idx,
        generated_domains,
        epochs=5,
        target=target,
        args=args,
        gen_method="natural",
    )



    # Ours-FR
    set_all_seeds(args.seed)
    our_source = copy.deepcopy(ref_model)
    ours_copy = copy.deepcopy(ref_model)
    ours_train, ours_test, ours_st, ours_st_all, ours_gen, EM_acc = run_main_algo(
        ours_copy,
        our_source,
        src_trainset,
        tgt_trainset,
        all_sets,
        deg_idx,
        generated_domains,
        epochs=5,
        target=target,
        args=args,
    )

    # GOAT (pairwise synthetics)
    goat_source = copy.deepcopy(ref_model)
    goat_copy = copy.deepcopy(goat_source)
    set_all_seeds(args.seed)
    goat_train, goat_test, goat_st, goat_st_all, goat_gen = run_goat(
        goat_copy,
        goat_source,
        src_trainset,
        tgt_trainset,
        all_sets,
        deg_idx,
        generated_domains,
        epochs=5,
        target=target,
        args=args,
    )

    # ---- (G) Persist plots ----
    plot_dir = f"plots/target{target}/"
    os.makedirs(plot_dir, exist_ok=True)

    _plot_series_with_baselines(
        series=[ours_test, goat_test, goatcw_test, ours_eta_test],
        labels=[
            "Ours-FR",
            "GOAT",
            "GOAT-Classwise",
            "Ours-ETA",
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
        title=(
            f"Target Accuracy (ST: {args.label_source} labels; "
            f"Cluster Map: {args.em_match})"
        ),
        ylabel="Accuracy",
        xlabel="Domain Index",
        save_path=os.path.join(
            plot_dir,
            f"test_acc_dim{args.small_dim}_int{args.gt_domains}_gen{args.generated_domains}_"
            f"{args.label_source}_{args.em_match}_{args.em_select}"
            f"{'_em-ensemble' if args.em_ensemble else ''}.png",
        ),
        synth_per_segment=int(generated_domains),
        n_real_segments=len(deg_idx),
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
    start_time_em = time.time()
    if args.em_match == "prototypes":
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
    em_bundle = build_em_bundle(em_models, args)
    args._shared_em = em_bundle
    apply_em_bundle_to_target(em_bundle, _e_tgt, tgt_trainset)
    print(f"[Portraits] Fitted and cached EM on target in {round(time.time() - start_time_em, 2)}s")

    y_true = np.asarray(tgt_trainset.targets, dtype=int)
    y_em = np.asarray(em_bundle.labels_em, dtype=int)
    em_acc_now = float((y_em == y_true).mean())
    print(f"[Portraits] EM→class accuracy: {em_acc_now:.4f}")
    # ------    ---------- Ours (ETA) ----------------
    start_time_eta = time.time()
    set_all_seeds(args.seed)
    ours_eta_src = copy.deepcopy(ref_model)
    ours_eta_cp  = copy.deepcopy(ref_model)
    ours_eta_train, ours_eta_test, ours_eta_st, ours_eta_st_all, ours_eta_gen, EM_acc_eta = run_main_algo(
        ours_eta_cp, ours_eta_src, src_trainset, tgt_trainset, all_sets, 0,
        generated_domains, epochs=5, target=1, args=args, gen_method="natural"
    )
    print(f"[Portraits] Ours-ETA completed in {round(time.time() - start_time_eta, 2)}s")


    # ---------------- Ours (FR) ----------------
    start_time_fr = time.time()
    set_all_seeds(args.seed)
    ours_src = copy.deepcopy(ref_model)
    ours_cp  = copy.deepcopy(ref_model)
    ours_train, ours_test, ours_st, ours_st_all, ours_gen, EM_acc = run_main_algo(
        ours_cp, ours_src, src_trainset, tgt_trainset, all_sets, 0,
        generated_domains, epochs=5, target=1, args=args, gen_method="fr"
    )
    print(f"[Portraits] Ours-FR completed in {round(time.time() - start_time_fr, 2)}s")
    # ---------------- GOAT (class-wise synthetics) ----------------
    start_time_goatcw = time.time()
    set_all_seeds(args.seed)
    goatcw_src = copy.deepcopy(ref_model)
    goatcw_cp  = copy.deepcopy(goatcw_src)
    goatcw_train, goatcw_test, goatcw_st, goatcw_st_all, goatcw_gen, EM_acc_goatcw = run_goat_classwise(
        goatcw_cp, goatcw_src, src_trainset, tgt_trainset, all_sets, 0,
        generated_domains, epochs=5, target=1, args=args
    )
    print(f"[Portraits] GOAT-Classwise completed in {round(time.time() - start_time_goatcw, 2)}s")
    if EM_acc is not None and EM_acc_goatcw is not None and abs(EM_acc - EM_acc_goatcw) > 1e-4:
        print(f"[WARNING] EM acc differs between Ours ({EM_acc:.2f}%) and GOAT-Classwise ({EM_acc_goatcw:.2f}%) on portraits!")
    
    # ---------------- GOAT (pair-wise synthetics) ----------------
    start_time_goat = time.time()
    set_all_seeds(args.seed)
    goat_src = copy.deepcopy(ref_model)
    goat_cp  = copy.deepcopy(goat_src)
    goat_train, goat_test, goat_st, goat_st_all, goat_gen = run_goat(
        goat_cp, goat_src, src_trainset, tgt_trainset, all_sets, 0,
        generated_domains, epochs=5, target=1, args=args
    )
    print(f"[Portraits] GOAT completed in {round(time.time() - start_time_goat, 2)}s")

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
        save_path=os.path.join(plot_dir, f"test_acc_dim{args.small_dim}_int{args.gt_domains}_gen{args.generated_domains}_{args.label_source}_{args.em_match}_{args.em_select}{'_em-ensemble' if args.em_ensemble else '' }.png"),
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
        compress=(args.small_dim < enc_out_dim),
        in_dim=enc_out_dim,
        out_dim=args.small_dim,
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
    time_start_em_pca = time.time()
    # (Optional) prototype mapping support
    if getattr(args, "em_match", "pseudo") == "prototypes":
        mu_s, Sigma_s, priors_s = fit_source_gaussian_params(X=_e_src.data, y=_e_src.targets)
        args._cached_source_stats = (mu_s, Sigma_s, priors_s)

    # ----- Pseudo labels from frozen teacher + per-domain EM fitting -----
    # Mirror MNIST logic: assign EM labels to every non-source real domain
    # so downstream FR/Natural generators consume valid targets_em everywhere.
    with torch.no_grad():
        teacher = copy.deepcopy(ref_model).to(device).eval()

    raw_domains = [src_trainset] + all_sets
    enc_domains = encoded_intersets
    n_classes = int(src_trainset.targets.max().item()) + 1
    em_bundles_by_domain: Dict[int, Any] = {}
    em_acc_by_domain: Dict[int, float] = {}

    # Canonical target pseudo labels for compatibility with legacy downstream code.
    with torch.no_grad():
        tgt_pl, _ = get_pseudo_labels(
            tgt_trainset,
            teacher,
            confidence_q=getattr(args, "pseudo_confidence_q", 0.9),
            device_override=device,
        )
    args._cached_pseudolabels = tgt_pl.cpu().numpy()

    for idx in range(1, len(raw_domains)):
        raw_ds = raw_domains[idx]
        enc_ds = enc_domains[idx]

        with torch.no_grad():
            pseudo_labels, _ = get_pseudo_labels(
                raw_ds,
                teacher,
                confidence_q=getattr(args, "pseudo_confidence_q", 0.9),
                device_override=device,
            )
        args._cached_pseudolabels = pseudo_labels.cpu().numpy()

        em_models = fit_many_em_on_target(
            enc_ds,
            K_list=[n_classes],
            cov_types=["diag"],
            seeds=[0, 1, 2],
            pool="gap",
            pca_dims=[None],
            reg=1e-4, max_iter=300, rng_base=args.seed, args=args,
        )
        em_bundle = build_em_bundle(em_models, args)
        apply_em_bundle_to_target(em_bundle, enc_ds, raw_ds)

        em_bundles_by_domain[idx] = em_bundle
        em_acc = (
            enc_ds.targets_em == raw_ds.targets.to(enc_ds.targets_em.device)
        ).float().mean().item()
        em_acc_by_domain[idx] = em_acc
        print(f"[CovType] EM accuracy domain_idx={idx}: {em_acc:.4f}")

    # Keep target-domain EM bundle as shared default for run_main_algo.
    target_idx = len(raw_domains) - 1
    em_bundle = em_bundles_by_domain[target_idx]
    args._shared_em = em_bundle
    args._shared_em_per_domain = em_bundles_by_domain
    args._cached_pseudolabels = tgt_pl.cpu().numpy()
    # Canonical target EM labels: restore these for every method run.
    canonical_target_em = None
    if hasattr(tgt_trainset, "targets_em") and tgt_trainset.targets_em is not None:
        canonical_target_em = torch.as_tensor(tgt_trainset.targets_em).view(-1).long().cpu().clone()
        args._canonical_target_em = canonical_target_em.clone()

    # Keep immutable references and clone them per method to avoid cross-method mutation.
    _base_src_trainset = src_trainset
    _base_tgt_trainset = tgt_trainset
    _base_all_sets = all_sets

    def _fresh_covtype_domains():
        src_local = copy.deepcopy(_base_src_trainset)
        tgt_local = copy.deepcopy(_base_tgt_trainset)
        if canonical_target_em is not None and hasattr(tgt_local, "targets_em"):
            tgt_local.targets_em = canonical_target_em.clone()
        all_sets_local = [copy.deepcopy(ds) for ds in _base_all_sets[:-1]] + [tgt_local]
        if canonical_target_em is not None and len(all_sets_local) > 0 and hasattr(all_sets_local[-1], "targets_em"):
            all_sets_local[-1].targets_em = canonical_target_em.clone()
        return src_local, tgt_local, all_sets_local

    print(f"[CovType] EM→class accuracy: {(em_bundle.labels_em == np.asarray(tgt_trainset.targets, dtype=int)).mean():.4f}")
    print(f"[CovType] EM fitting time (s): {time.time() - time_start_em_pca:.2f}")

    # ----- Run methods -----
    # Ours-ETA
    start_time_eta = time.time()
    set_all_seeds(args.seed)
    src_eta, tgt_eta, sets_eta = _fresh_covtype_domains()
    ours_eta_src = copy.deepcopy(ref_model);  ours_eta_cp = copy.deepcopy(ref_model)
    ours_eta_train, ours_eta_test, ours_eta_st, ours_eta_st_all, ours_eta_gen, EM_acc_eta = run_main_algo(
        ours_eta_cp, ours_eta_src, src_eta, tgt_eta, sets_eta, 0,
        generated_domains, epochs=5, target=1, args=args, gen_method="natural"
    )
    print(f"[CovType] Ours-ETA time (s): {time.time() - start_time_eta:.2f}")
    start_time_goatcw = time.time()
        # GOAT-Classwise
    set_all_seeds(args.seed)
    src_goatcw, tgt_goatcw, sets_goatcw = _fresh_covtype_domains()
    goatcw_src = copy.deepcopy(ref_model);  goatcw_cp = copy.deepcopy(goatcw_src)
    goatcw_train, goatcw_test, goatcw_st, goatcw_st_all, goatcw_gen, EM_acc_goatcw = run_goat_classwise(
        goatcw_cp, goatcw_src, src_goatcw, tgt_goatcw, sets_goatcw, 0,
        generated_domains, epochs=5, target=1, args=args
    )
    print(f"[CovType] GOAT-Classwise time (s): {time.time() - start_time_goatcw:.2f}")
    start_time_fr = time.time()
    # Ours-FR
    set_all_seeds(args.seed)
    src_fr, tgt_fr, sets_fr = _fresh_covtype_domains()
    ours_src = copy.deepcopy(ref_model);  ours_cp = copy.deepcopy(ref_model)
    ours_train, ours_test, ours_st, ours_st_all, ours_gen, EM_acc = run_main_algo(
        ours_cp, ours_src, src_fr, tgt_fr, sets_fr, 0,
        generated_domains, epochs=5, target=1, args=args, gen_method="fr"
    )
    print(f"[CovType] Ours-FR time (s): {time.time() - start_time_fr:.2f}")
    start_time_goat = time.time()
    # GOAT
    set_all_seeds(args.seed)
    src_goat, tgt_goat, sets_goat = _fresh_covtype_domains()
    goat_src = copy.deepcopy(ref_model);  goat_cp = copy.deepcopy(goat_src)
    goat_train, goat_test, goat_st, goat_st_all, goat_gen = run_goat(
        goat_cp, goat_src, src_goat, tgt_goat, sets_goat, 0,
        generated_domains, epochs=5, target=1, args=args
    )
    print(f"[CovType] GOAT time (s): {time.time() - start_time_goat:.2f}")


    # ----- Plot -----
    plot_dir = f"plots/covtype/"
    os.makedirs(plot_dir, exist_ok=True)

    _plot_series_with_baselines(
        series=[ours_test, goat_test, goatcw_test, ours_eta_test],
        labels=["Ours-FR", "GOAT", "GOAT-Classwise", "Ours-ETA"],
        baselines=[(ours_st, ours_st_all)],
        ref_line_value=(EM_acc * 100.0 if EM_acc is not None else None),
        ref_line_label=f"EM ({args.em_match})",
        ref_line_style="--",
        title=f"CovType: Target Accuracy (ST: {args.label_source}; Cluster Map: {args.em_match})",
        ylabel="Accuracy", xlabel="Domain Index",
        save_path=os.path.join(plot_dir, f"test_acc_dim{args.small_dim}_int{args.gt_domains}_gen{args.generated_domains}_{args.label_source}_{args.em_match}_{args.em_select}{'_em-ensemble' if args.em_ensemble else '' }.png"),
    )

    # ----- Log -----
    os.makedirs("logs", exist_ok=True)
    elapsed = round(time.time() - t0, 2)
    with open(f"logs/covtype_exp_{args.log_file}.txt", "a") as f:
        f.write(
            f"seed{args.seed}with{gt_domains}gt{generated_domains}generated,{elapsed},"
            f"OursFR:{round((ours_test[-1] if ours_test else 0.0), 2)},GOAT:{round((goat_test[-1] if goat_test else 0.0), 2)},"
            f"GOATCW:{round((goatcw_test[-1] if goatcw_test else 0.0), 2)},ETA:{round((ours_eta_test[-1] if ours_eta_test else 0.0), 2)}\n"
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
    # === INSERT AFTER shared_pca IS DEFINED (DIAGNOSTICS) ===
    # Pull arrays from encoded datasets
    Zs = np.asarray(_e_src.data, dtype=np.float64)
    ys = np.asarray(_e_src.targets, dtype=int)
    Zt = np.asarray(_e_tgt.data, dtype=np.float64)
    yt = np.asarray(_e_tgt.targets, dtype=int)



    time_start_em = time.time()
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
        pca_dims=[None],
        reg=1e-4, max_iter=300, rng_base=args.seed, args=args
    )

    rows = print_em_model_accuracies(em_models, tgt_trainset.targets)

    rows = [{**r, "original_idx": i} for i, r in enumerate(rows)]

    # 1) Print ALL models in their original order
    for r in rows:
        bic = r.get("bic")
        print(
            f"[EM {r['original_idx']:02d}] "
            f"acc={r['acc']*100:6.2f}% | seed={r.get('seed')} | cov={r.get('cov')} | "
            f"K={r.get('K')} | BIC={bic if bic is not None else 'NA'} | "
            f"final_ll={r.get('final_ll')}"
        )
    em_bundle = build_em_bundle(em_models, args)
    args._shared_em = em_bundle
    apply_em_bundle_to_target(em_bundle, _e_tgt, tgt_trainset)
    # Canonical target EM labels: restore these for every method run.
    canonical_target_em = None
    if hasattr(tgt_trainset, "targets_em") and tgt_trainset.targets_em is not None:
        canonical_target_em = torch.as_tensor(tgt_trainset.targets_em).view(-1).long().cpu().clone()
        args._canonical_target_em = canonical_target_em.clone()

    # Keep immutable references and clone them per method to avoid cross-method mutation.
    _base_src_trainset = src_trainset
    _base_tgt_trainset = tgt_trainset
    _base_all_sets = all_sets

    def _fresh_color_domains():
        src_local = copy.deepcopy(_base_src_trainset)
        tgt_local = copy.deepcopy(_base_tgt_trainset)
        if canonical_target_em is not None and hasattr(tgt_local, "targets_em"):
            tgt_local.targets_em = canonical_target_em.clone()
        all_sets_local = [copy.deepcopy(ds) for ds in _base_all_sets[:-1]] + [tgt_local]
        if canonical_target_em is not None and len(all_sets_local) > 0 and hasattr(all_sets_local[-1], "targets_em"):
            all_sets_local[-1].targets_em = canonical_target_em.clone()
        return src_local, tgt_local, all_sets_local

    print(f"[ColorMNIST] EM fitting time (s): {time.time() - time_start_em:.2f}")
    # Print EM accuracy (mapped labels vs true target labels)
    y_true = np.asarray(tgt_trainset.targets, dtype=int)
    y_em = np.asarray(em_bundle.labels_em, dtype=int)
    em_acc_now = float((y_em == y_true).mean())
    print(f"[ColorMNIST] EM→class accuracy: {em_acc_now:.4f}")
    # Also report the best achievable one-to-one mapping accuracy from raw EM clusters

    # ---------- Run methods ----------
    start_time_fr = time.time()
    # Ours-FR
    set_all_seeds(args.seed)
    src_fr, tgt_fr, sets_fr = _fresh_color_domains()
    ours_src = copy.deepcopy(ref_model);  ours_cp = copy.deepcopy(ref_model)
    ours_train, ours_test, ours_st, ours_st_all, ours_gen, EM_acc = run_main_algo(
        ours_cp, ours_src, src_fr, tgt_fr, sets_fr, 0,
        generated_domains, epochs=5, target=1, args=args, gen_method="fr"
    )
    print(f"[ColorMNIST] Ours-FR time (s): {time.time() - start_time_fr:.2f}")
    start_time_goatcw = time.time()

    # GOAT-Classwise
    set_all_seeds(args.seed)
    src_goatcw, tgt_goatcw, sets_goatcw = _fresh_color_domains()
    goatcw_src = copy.deepcopy(ref_model);  goatcw_cp = copy.deepcopy(goatcw_src)
    goatcw_train, goatcw_test, goatcw_st, goatcw_st_all, goatcw_gen, EM_acc_goatcw = run_goat_classwise(
        goatcw_cp, goatcw_src, src_goatcw, tgt_goatcw, sets_goatcw, 0,
        generated_domains, epochs=5, target=1, args=args
    )
    print(f"[ColorMNIST] GOAT-Classwise time (s): {time.time() - start_time_goatcw:.2f}")
    start_time_eta = time.time()

    # Ours-ETA
    set_all_seeds(args.seed)
    src_eta, tgt_eta, sets_eta = _fresh_color_domains()
    ours_eta_src = copy.deepcopy(ref_model);  ours_eta_cp = copy.deepcopy(ref_model)
    ours_eta_train, ours_eta_test, ours_eta_st, ours_eta_st_all, ours_eta_gen, EM_acc_eta = run_main_algo(
        ours_eta_cp, ours_eta_src, src_eta, tgt_eta, sets_eta, 0,
        generated_domains, epochs=5, target=1, args=args, gen_method="natural"
    )
    print(f"[ColorMNIST] Ours-ETA time (s): {time.time() - start_time_eta:.2f}")
    start_time_goat = time.time()


    # GOAT
    set_all_seeds(args.seed)
    src_goat, tgt_goat, sets_goat = _fresh_color_domains()
    goat_src = copy.deepcopy(ref_model);  goat_cp = copy.deepcopy(goat_src)
    goat_train, goat_test, goat_st, goat_st_all, goat_gen = run_goat(
        goat_cp, goat_src, src_goat, tgt_goat, sets_goat, 0,
        generated_domains, epochs=5, target=1, args=args
    )

    print(f"[ColorMNIST] GOAT time (s): {time.time() - start_time_goat:.2f}")
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
        save_path=os.path.join(plot_dir, f"test_acc_dim{args.small_dim}_int{args.gt_domains}_gen{args.generated_domains}_{args.label_source}_{args.em_match}_{args.em_select}{'_em-ensemble' if args.em_ensemble else '' }.png"),
    )

    # ---------- Log ----------
    log_dir = os.path.join("logs", "color_mnist", f"s{args.seed}")
    os.makedirs(log_dir, exist_ok=True)
    elapsed = round(time.time() - t0, 2)
    base_name = (
        args.log_file
        if args.log_file
        else (
            f"test_acc_dim{args.small_dim}_int{args.gt_domains}_gen{args.generated_domains}_"
            f"{args.label_source}_{args.em_match}_{args.em_select}"
            f"{'_em-ensemble' if args.em_ensemble else ''}"
        )
    )
    with open(os.path.join(log_dir, f"{base_name}.txt"), "a") as f:
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
    parser.add_argument("--em-ensemble", action="store_true", help="Whether to ensemble multiple EM models")
    parser.add_argument("--em-select", choices=["bic", "cost", "ll"], default="bic", help="Criterion to select best EM model")
    args = parser.parse_args()
    main(args)
