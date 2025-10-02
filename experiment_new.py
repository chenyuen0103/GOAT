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
from typing import Optional, Tuple, List
import numpy as np
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

# -------------------------------------------------------------
# Global config / utilities
# -------------------------------------------------------------

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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

def _load_list(path, default=None):
    """
    Load JSON previously saved by _save_list. Returns Python primitives/containers.

    Backward compatible: if the file is missing or invalid, returns `default`.
    """
    try:
        with open(path, "r") as f:
            obj = json.load(f)
        return obj
    except Exception as e:
        print(f"[MainAlgo] Failed to load {path}: {e}")
        return default
def check_fr_logvar_linearity(domain_stats: dict):
    ts = _steps(domain_stats)
    S, K = len(ts), int(domain_stats["K"])

    if not _has_full(domain_stats):
        # (diag case unchanged)
        errs = []
        for k in range(K):
            y = []
            for s in range(S):
                if _present_mask(domain_stats, s)[k]:
                    v = _get_var(domain_stats, s)[k]
                    y.append(np.mean(np.log(np.clip(v, 1e-12, None))))
                else:
                    y.append(np.nan)
            y = np.asarray(y); m = ~np.isnan(y)
            if m.sum() >= 2:
                coeff = np.polyfit(ts[m], y[m], 1)
                yhat  = np.polyval(coeff, ts[m])
                errs.append(np.mean(np.abs(y[m] - yhat)))
        print(f"[FR][check-diag] mean |Δ logvar| = {np.nanmean(errs):.3e}")
        return

    # FULL covariance: check linearity of logdet(Σ) along t
    errs = []
    for k in range(K):
        y = []
        for s in range(S):
            if not _present_mask(domain_stats, s)[k]:
                y.append(np.nan)
                continue
            Sig_k = _get_sigma(domain_stats, s)[k]   # accepts (d,d) or (d,) safely
            ld = _logdet_cov(Sig_k)[0]               # scalar
            y.append(ld)
        y = np.asarray(y); m = ~np.isnan(y)
        if m.sum() >= 2:
            coeff = np.polyfit(ts[m], y[m], 1)
            yhat  = np.polyval(coeff, ts[m])
            errs.append(np.mean(np.abs(y[m] - yhat)))
    print(f"[FR][check-full] mean |Δ logdet(Σ)| = {np.nanmean(errs):.3e}")

def _unpack_domain_stats(stats):
    """
    Returns: steps (S,), mu (S,K,d), var (S,K,d), counts (S,K)
    Works for both the new dict schema and the old list-of-tuples.
    """
    import numpy as np

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
    import numpy as np
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
        model = CompressClassifier(model, in_dim=in_dim, out_dim=out_dim).to(device)

    model = model.to(device)      # <- apply to BOTH branches

    if os.path.exists(model_path) and not force_recompute:
        try:
            print(f"[Cache] Loading trained model from {model_path}")
            ckpt = torch.load(model_path, map_location=device)
            missing, unexpected = model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)

            return model
        except Exception as e:
            print(f"[Cache] Error loading model from {model_path}: {e}")
            print("[Load]", "missing:", missing, "| unexpected:", unexpected)



        


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
    augment_fn = build_augment(image_size=(28, 28))  # tweak if needed per dataset
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
) -> Tuple[Dataset, Dataset, List[Dataset]]:
    """Encode source/intermediate/target datasets once and cache results."""
    os.makedirs(cache_dir, exist_ok=True)

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

    encoded_intersets: List[Dataset] = [e_src]
    intersets = all_sets[:-1]
    for idx, inter in enumerate(intersets):
        cache_name = f"encoded_{deg_idx[idx]}.pt" if idx < len(deg_idx) else f"encoded_inter_{idx}.pt"
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
    args=None
):
    """GOAT-style baseline: direct ST on target vs pooled ST across real domains,
    optionally ST on synthetics generated between consecutive encoded domains.
    Also: plots train/test accuracies over domains for the pooled and synthetic runs.
    """
    # ----- Direct adapt (self-train only on target) -----
    # Keep original behavior; self_train requires >=2 datasets, so we use self_train_og here.
    direct_acc, st_acc, train_acc_by_domain0, test_acc_by_domain0 = self_train(args, model_copy, [tgt_trainset], epochs=epochs, label_source="pseudo")

    # ----- Pooled ST on real intermediates + target -----
    # Use updated self_train to get per-domain accuracy lists.
    direct_acc_all, st_acc_all, train_acc_list_all, test_acc_list_all = self_train(
        args, source_model, all_sets, epochs=epochs, label_source="pseudo",
        use_labels=getattr(args, "use_labels", False)
    )

    # Dirs
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
    )

    # ----- Optionally generate synthetic domains and ST on them -----
    generated_acc = 0.0
    if generated_domains > 0:
        all_domains: List[Dataset] = []
        for i in range(len(encoded_intersets) - 1):
            out = generate_domains(generated_domains, encoded_intersets[i], encoded_intersets[i + 1])
            new_domains = out[0]
            all_domains += new_domains

        # Use updated self_train to get per-domain accuracy lists on synthetic chain
        _, generated_acc, train_acc_by_domain, test_acc_by_domain = self_train(
            args,
            source_model.mlp,
            all_domains,
            epochs=epochs,
            label_source='pseudo',
            use_labels=getattr(args, "use_labels", False),
        )
        # Persist & plot synthetic accuracies
        # train_acc_by_domain = train_acc_by_domain0 + train_acc_by_domain
        # test_acc_by_domain  = test_acc_by_domain0  + test_acc_by_domain
        _save_list(os.path.join(plot_dir, "goat_train_acc_by_domain.json"), train_acc_by_domain)
        _save_list(os.path.join(plot_dir, "goat_test_acc_by_domain.json"),  test_acc_by_domain)
        # PCA grids: real domains and synthetic chain (source → generated → target)
        try:
            # Real domains
            plot_pca_classes_grid(
                encoded_intersets,
                classes=(3, 6, 8, 9),
                save_path=os.path.join(plot_dir, "pca_classes_real_domains_goat.png"),
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
            try:
                tgt_pl, _ = get_pseudo_labels(
                    encoded_intersets[-1],
                    getattr(source_model, 'mlp', source_model),
                    confidence_q=getattr(args, 'pseudo_confidence_q', 0.0),
                )
                encoded_intersets[-1].targets_em = tgt_pl.clone() if hasattr(tgt_pl, 'clone') else torch.as_tensor(tgt_pl, dtype=torch.long)
            except Exception as _e:
                print(f"[GOAT][PCA] Warning: failed to attach pseudo labels to target for coloring: {_e}")
            plot_pca_classes_grid(
                chain_for_plot,
                classes=(3, 6, 8, 9),
                save_path=os.path.join(plot_dir, "pca_classes_synth_goat.png"),
                label_source='pseudo',
                pseudolabels=tgt_pl
            )
        except Exception as e:
            print(f"[GOAT][PCA] Skipped PCA plotting: {e}")
        # _plot_line(
        #     train_acc_by_domain,
        #     title="Self-Training on Synthetic Domains: Training Accuracy by Domain",
        #     ylabel="Accuracy",
        #     xlabel="Training Domain Index",
        #     save_path=os.path.join(plot_dir, "goat_train_acc_by_domain.png"),
        # )
        # _plot_line(
        #     test_acc_by_domain,
        #     title="Self-Training on Synthetic Domains: Target Accuracy After Each Domain",
        #     ylabel="Accuracy",
        #     xlabel="Training Domain Index",
        #     save_path=os.path.join(plot_dir, "goat_test_acc_by_domain.png"),
        # )

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
    args=None
):
    """GOAT baseline with class-wise synthetic generation (still calls generate_domains).
       This function will compute .targets_em for any encoded dataset that lacks it.
    """

    # Freeze a clean teacher for EM/pseudo labels to avoid any leakage
    # from subsequent adaptation. Use the head that consumes encoded features.
    em_teacher = copy.deepcopy(source_model).to(device).eval()
    em_head = getattr(em_teacher, 'mlp', em_teacher)

    # ---------- Direct adapt (target only) ----------
    direct_acc, st_acc, train_acc_by_domain0, test_acc_by_domain0 = self_train(
        args, model_copy, [tgt_trainset], epochs=epochs, label_source="pseudo"
    )

    # ---------- Pooled ST on real intermediates + target ----------
    direct_acc_all, st_acc_all, train_acc_list_all, test_acc_list_all = self_train(
        args, source_model, all_sets, epochs=epochs, label_source="pseudo",
        use_labels=getattr(args, "use_labels", False)
    )

    # ---------- Dirs ----------
    cache_dir = f"cache{args.ssl_weight}/target{target}/small_dim{args.small_dim}/"
    plot_dir  = f"plots/target{target}/"
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(plot_dir,  exist_ok=True)


    # ---------- Encode all domains ----------
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
    )

    # Use frozen teacher for pseudo labels on target to keep it target-GT agnostic
    pseudo_labels, _pseudo_keep = get_pseudo_labels(
        tgt_trainset,
        em_teacher,
        confidence_q=getattr(args, "pseudo_confidence_q", 0.1),
        device_override=next(em_teacher.parameters()).device,
    )
    pseudolabels = pseudo_labels.cpu().numpy()
    K = int(pseudo_labels.max().item()) + 1
    # def _ensure_targets_em(ds: Dataset, infer_model: nn.Module):
    #     """Compute ds.targets_em via EM clustered labels mapped to pseudo-labels from infer_model."""
    #     # skip if already exists and not all -1
    #     if getattr(ds, "targets_em", None) is not None and not (ds.targets_em == -1).all():
    #         return
    #     loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    #     # Pseudo labels from the classifier head that consumes encoded features
    #     pseudo_lab, _ = get_pseudo_labels(
    #         loader, infer_model, device_override=next(infer_model.parameters()).device
    #     )
    #     if pseudo_lab.numel() == 0:
    #         raise ValueError("Failed to compute pseudo labels for EM mapping.")
    #     pseudo_np = pseudo_lab.cpu().numpy()
    #     K = int(pseudo_lab.max().item()) + 1

    #     # EM on the encoded dataset (same feature space)
        
    #     em_res = run_em_on_encoded(
    #         ds, K=K, cov_type="diag", pool="flatten", do_pca=False,
    #         return_transforms=True, verbose=False
    #     )

    #     if args.em_match == "prototypes":
    #         mu_s, Sigma_s, priors_s = fit_source_gaussian_params(
    #             X = e_src.data, y = e_src.targets)
    #         mapping_pseudo_main, labels_mapped_pseudo_main, _ = map_em_clusters(
    #             em_res,
    #             method=args.em_match,
    #             n_classes=K,
    #             mus_s=mu_s,
    #             Sigma_s=Sigma_s,
    #             priors_s=priors_s,
    #         )
    #         ds.targets_em = torch.as_tensor(labels_mapped_pseudo_main, dtype=torch.long)
    #         return
    #     # Map clusters to classes by Hungarian vs pseudo labels
    #     _, labels_mapped, _ = map_em_clusters(
    #         em_res, method=args.em_match, n_classes=K, pseudo_labels=pseudo_np
    #     )
    #     ds.targets_em = torch.as_tensor(labels_mapped, dtype=torch.long)


    def _labels_for_split(ds: Dataset, is_source: bool, ) -> torch.Tensor:
        """
        Class labels used for per-class splitting:
        - If is_source==True and ds.targets exists → use ds.targets (allowed).
        - Otherwise (non-source domains, including target) → use inferred ds.targets_em.
        """
        if is_source and hasattr(ds, "targets") and ds.targets is not None:
            y = ds.targets
            return torch.as_tensor(y).long().cpu()

        # Always use the frozen EM head for non-source domains
        # _ensure_targets_em(ds, em_head)
        return torch.as_tensor(ds.targets_em).long().cpu()

    em_res_tgt = run_em_on_encoded(
        e_tgt,
        K=K,
        cov_type="diag",
        pool="gap",            # GAP pooling
        do_pca=False,            # Enable PCA for stability of full covariances
        pca_dim=None,             # Reduce to 64 dimensions
        reg=1e-4,               # Stronger ridge for full covariances
        max_iter=500,  # Reduce max iterations from default 100 to 50
        return_transforms=True,
        verbose=True,  # Enable to see progress and timing
        subsample_init=10000,  # Reduce subsampling for faster initialization
    )
    if args.em_match == "prototypes":
        mu_s, Sigma_s, priors_s = fit_source_gaussian_params(
            X = e_src.data, y = e_src.targets)

        mapping_pseudo_main, labels_mapped_pseudo_main, _ = map_em_clusters(
            em_res_tgt,
            method=args.em_match,
            n_classes=K,
            metric='FR',
            mus_s=mu_s,
            Sigma_s=Sigma_s,
            priors_s=priors_s,
        )
    else:
        mapping_pseudo_main, labels_mapped_pseudo_main, _ = map_em_clusters(
            em_res_tgt,
            method=args.em_match,
            n_classes=K,
            pseudo_labels=pseudolabels,
        )

    def _subset_by_class(ds: Dataset, cls: int, is_source: bool) -> Optional[Dataset]:
        """Return a per-class DomainDataset compatible with generate_domains."""
        labels = _labels_for_split(ds, is_source=is_source)
        X = ds.data if torch.is_tensor(getattr(ds, "data", None)) else torch.as_tensor(ds.data)
        X = X.cpu()
        m = (labels == int(cls))
        if m.sum().item() == 0:
            return None
        Xc = X[m]
        yc = labels[m]
        w  = torch.ones(len(yc))
        # DomainDataset(data, weights, targets)
        return DomainDataset(Xc, w, yc)


    def _merge_domains_per_step(list_of_lists: List[List[Dataset]]) -> List[Dataset]:
        """Merge step j across classes into a single DomainDataset."""
        if not list_of_lists:
            return []
        n_steps = min(len(L) for L in list_of_lists)   # should be n_inter + 1 (includes appended target)
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
            merged.append(DomainDataset(X, W, Y, Y))  # set targets_em := Y for training
        return merged
    e_tgt.targets_em = torch.as_tensor(labels_mapped_pseudo_main, dtype=torch.long)
    # ---------- Class-wise synthetic generation loop (inside run_goat_classwise) ----------
    generated_acc = 0.0
    if generated_domains > 0:
        all_domains: List[Dataset] = []
        # Number of classes from the *inferred* labels on the final target encoding
        # _ensure_targets_em(e_tgt, em_head)
        K = int(torch.as_tensor(e_tgt.targets_em).max().item()) + 1

        for i in range(len(encoded_intersets) - 1):
            s_ds = encoded_intersets[i]
            t_ds = encoded_intersets[i + 1]
            # breakpoint()

            # Define whether the left side of the pair is the original source encoding
            is_source_left = (i == 0)

            # Ensure EM labels for the right side (non-source)
            # _ensure_targets_em(t_ds, getattr(source_model, 'mlp', source_model))

            # Build per-class chains by calling the original generator per class
            per_class_chains: List[List[Dataset]] = []
            for c in range(K):
                s_c = _subset_by_class(s_ds, c, is_source=is_source_left)
                t_c = _subset_by_class(t_ds, c, is_source=False)  # never use GT on right
                if s_c is None or t_c is None:
                    continue
                chain_c, _ = generate_domains(generated_domains, s_c, t_c)

                for D in chain_c:
                    # force GLOBAL class ID for both targets and targets_em
                    y_global = torch.full((len(D.targets),), c, dtype=torch.long)
                    D.targets = y_global
                    if getattr(D, "targets_em", None) is not None:
                        D.targets_em = y_global.clone()
                    else:
                        D.targets_em = y_global.clone()  # ensure it exists for self_train
                if chain_c:
                    per_class_chains.append(chain_c)
                    # check that the labels are all c
                    for step_ds in chain_c:
                        labs = step_ds.targets if torch.is_tensor(step_ds.targets) else torch.as_tensor(step_ds.targets)
                        labs = labs.cpu().numpy()
                        assert (labs == c).all(), f"Class mismatch in generated chain for class {c} at step {step_ds}"

            # Merge per step across classes, then append
            merged_chain = _merge_domains_per_step(per_class_chains)
            # breakpoint()
            all_domains += merged_chain

        # Ensure evaluation target matches other methods: use the full encoded target
        # as the final held-out dataset for consistency of the 0th plot point.
        if len(all_domains) > 0:
            all_domains[-1] = e_tgt

        # Self-train on the synthetic class-wise chain
        _, generated_acc, train_acc_by_domain, test_acc_by_domain = self_train(
            args, source_model.mlp, all_domains,
            epochs=epochs, label_source=args.label_source,
            use_labels=getattr(args, "use_labels", False),
        )


        _save_list(os.path.join(plot_dir, "goat_train_acc_by_domain.json"), train_acc_by_domain)
        _save_list(os.path.join(plot_dir, "goat_test_acc_by_domain.json"),  test_acc_by_domain)

        # PCA grids: real domains and synthetic chain (source → generated → target)
        try:
            # Real domains
            plot_pca_classes_grid(
                encoded_intersets,
                classes=(3, 6, 8, 9),
                save_path=os.path.join(plot_dir, f"pca_classes_real_domains_goatcw.png"),
            )
            # Synthetic chain: group per pair and drop appended right endpoint
            step_len = int(generated_domains) + 1
            chain_only = []
            for k in range(0, len(all_domains), max(step_len, 1)):
                chunk = all_domains[k:k + step_len]
                if not chunk:
                    continue
                chain_only.extend(chunk[:-1] if step_len > 0 else chunk)
            chain_for_plot = [encoded_intersets[0]] + chain_only + [encoded_intersets[-1]]
            plot_pca_classes_grid(
                chain_for_plot,
                classes=(3, 6, 8, 9),
                save_path=os.path.join(plot_dir, f"pca_classes_synth_goatcw_{args.label_source}_{args.em_match}.png"),
                label_attr='targets_em',  # classwise uses EM/global class labels in .targets_em
            )
        except Exception as e:
            print(f"[GOAT-CW][PCA] Skipped PCA plotting: {e}")

        return train_acc_by_domain, test_acc_by_domain, st_acc, st_acc_all, generated_acc

    # If no synthetics requested, still return meaningful values/lists
    # return train_acc_by_domain0, test_acc_by_domain0, st_acc, st_acc_all, generated_acc


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
    args= None,
    gen_method: str = "fr",
):
    """Teacher self-training along generated domains **in embedding space**.
    Encoder stays fixed; only the head adapts."""
    # breakpoint()
    direct_acc, st_acc, train_acc_by_domain0, test_acc_by_domain0 = self_train(
        args, model_copy, [tgt_trainset], epochs=epochs, label_source="pseudo")

    # Pooled ST on real intermediates + target
    direct_acc_all, st_acc_all, train_acc_list_all, test_acc_list_all = self_train(
        args, source_model, all_sets, epochs=epochs, label_source="pseudo")
    
    pseudo_labels, _pseudo_keep = get_pseudo_labels(
        tgt_trainset,
        source_model,
        confidence_q=getattr(args, "pseudo_confidence_q", 0.1),
        device_override=next(source_model.parameters()).device,
    )
    pseudolabels = pseudo_labels.cpu().numpy()
    cache_dir = f"cache{args.ssl_weight}/target{target}/small_dim{args.small_dim}/"
    plot_dir = f"plots/target{target}/"
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    e_src, e_tgt, encoded_intersets = encode_all_domains(
        src_trainset,
        tgt_trainset,
        all_sets,
        deg_idx,
        nn.Sequential(
            source_model.encoder,
            nn.Flatten(start_dim=1),
            source_model.compressor
        ),  # <- encode with full model (encoder -> flatten -> compressor)
        cache_dir,
        target,
        force_recompute=False,
    )


    y_true = e_tgt.targets
    if torch.is_tensor(y_true):
        y_true = y_true.cpu()

    preds = torch.as_tensor(pseudolabels, device=y_true.device)
    acc = (preds == y_true).float().mean().item()
    print(f"Pseudo-label accuracy: {acc:.4f}")

    # -------- Step 1: EM soft-targets on target embeddings, train head with soft labels --------
    try:
        K = int(max(int(preds.max().item()), int(y_true.max().item()))) + 1
    except Exception:
        K = 10

    print(f"[MainAlgo] Running EM clustering with K={K} components...")
    start_time = time.time()
    

    # fit 
    # EM on encoded target (use same pooling/shape as encoded data)
    # Optimizations: PCA for dimensionality reduction, GAP pooling instead of flatten, fewer iterations
    em_res_tgt = run_em_on_encoded(
        e_tgt,
        K=K,
        cov_type="diag",
        pool="gap",
        do_pca=False,
        pca_dim=None,
        reg=1e-4,
        max_iter=500,  # Reduce max iterations from default 100 to 50
        return_transforms=True,
        verbose=True,  # Enable to see progress and timing
        subsample_init=10000,  # Reduce subsampling for faster initialization
    )
    em_time = time.time() - start_time
    print(f"[MainAlgo] EM clustering completed in {em_time:.2f} seconds")
    

    if args.em_match == "prototypes":
        mu_s, Sigma_s, priors_s = fit_source_gaussian_params(
            X = e_src.data, y = e_src.targets)

        mapping_pseudo_main, labels_mapped_pseudo_main, _ = map_em_clusters(
            em_res_tgt,
            method=args.em_match,
            n_classes=K,
            mus_s=mu_s,
            Sigma_s=Sigma_s,
            priors_s=priors_s,
        )
    else:
        mapping_pseudo_main, labels_mapped_pseudo_main, _ = map_em_clusters(
            em_res_tgt,
            method=args.em_match,
            n_classes=K,
            pseudo_labels=pseudolabels,
            metric='FR'
        )

    y_true_np = np.asarray(y_true)
    acc_em_pseudo = float((np.asarray(labels_mapped_pseudo_main) == y_true_np).mean())
    print(f"[MainAlgo] EM→class (pseudo mapping) accuracy: {acc_em_pseudo:.4f}")

    try:
        best_acc, _best_map, _C = best_mapping_accuracy(em_res_tgt["labels"], y_true_np)
        print(f"[MainAlgo] Best one-to-one mapping accuracy: {best_acc:.4f}")
    except Exception as e:
        print(f"[MainAlgo] Best-mapping computation failed: {e}")

    # Build EM soft class targets using the prototype mapping if available; otherwise pseudo mapping
    # use_mapping = mapping_proto_main if mapping_proto_main is not None else mapping_pseudo_main
    use_mapping = mapping_pseudo_main
    em_soft = em_soft_targets_from_mapping(em_res_tgt["gamma"], use_mapping, n_classes=K)

    # Ensure target datasets carry EM labels for downstream steps
    e_tgt.targets_em = torch.as_tensor(labels_mapped_pseudo_main, dtype=torch.long)
    tgt_trainset.targets_em = e_tgt.targets_em.cpu().clone()




    X_std = np.asarray(em_res_tgt.get("X"))
    if X_std is not None and X_std.ndim == 2 and X_std.shape[0] == len(e_tgt):
        pca = PCA(n_components=2)
        Z = pca.fit_transform(X_std)

        # Prepare shared items
        y_np      = np.asarray(y_true)                    # ground-truth for left panel
        right_em  = np.asarray(labels_mapped_pseudo_main) # EM-mapped labels (int array)
        right_pl  = np.asarray(pseudolabels)              # raw pseudo labels (int array)
        cmap      = plt.get_cmap('tab10')

        # Project EM means/covs into PCA(2)
        C2 = pca.components_[:2, :]
        m2 = pca.mean_
        mu_src     = em_res_tgt.get("mu")
        Sigma_src  = em_res_tgt.get("Sigma")
        mapping_ps = mapping_pseudo_main  # cluster id -> class id

        # 1) Second panel colored by EM labels
        plot_pca_em_pair_side_by_side(
            X_std,
            y_np,
            mu_src,
            Sigma_src,
            mapping_ps,
            right_labels=right_em,
            right_title="EM-mapped labels",
            save_path=os.path.join(plot_dir, "fr_target_pca_emlabels_side_by_side.png"),
        )

        # 2) Second panel colored by pseudo labels
        plot_pca_em_pair_side_by_side(
            X_std,
            y_np,
            mu_src,
            Sigma_src,
            mapping_ps,
            right_labels=right_pl,
            right_title="Pseudo labels",
            save_path=os.path.join(plot_dir, "fr_target_pca_pseudolabels_side_by_side.png"),
        )


    # Evaluate updated model on target images
    tgt_loader_eval = DataLoader(tgt_trainset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    _, tgt_acc_after = test(tgt_loader_eval, source_model)
    print(f"[Head-Soft] Target accuracy after soft-target tuning: {tgt_acc_after:.4f}")
    # breakpoint()
    if generated_domains <= 0:
        return direct_acc, st_acc, direct_acc_all, st_acc_all, 0.0

    synthetic_domains: List[Dataset] = []
    labels_from_gen_last = None  # generator's pseudo-mapped labels for target
    # choose generator based on flag (default 'fr'); accept common aliases
    _method = (gen_method or "fr").lower()
    if _method in {"fr", "fisher-rao", "fisher_rao"}:
        _gen_fn = generate_fr_domains_between
    elif _method in {"natural", "natureal", "eta", "nat", "np"}:
        _gen_fn = generate_natural_domains_between
    else:
        raise ValueError(f"Unknown gen_method '{gen_method}'. Use 'fr' or 'natural'.")

    for i in range(len(encoded_intersets) - 1):
        # breakpoint()
        out = _gen_fn(
            generated_domains,
            encoded_intersets[i],
            encoded_intersets[i + 1],
            source_model=source_model,
            pseudolabels=pseudolabels,
            visualize=False,
            cov_type="full",
        )
        pair_domains = out[0]
        domain_stats = out[2]

        summarize_domain_stats(domain_stats)
        check_fr_logvar_linearity(domain_stats)
        check_means_between(domain_stats)
        adjacent_distances(domain_stats)

        plot_class_counts(domain_stats, save_path=os.path.join(plot_dir, f"counts_pair{i}.png"))
        plot_size_proxy(domain_stats, mode="trace",   save_path=os.path.join(plot_dir, f"trace_pair{i}.png"))
        plot_size_proxy(domain_stats, mode="logdet", save_path=os.path.join(plot_dir, f"logdet_pair{i}.png"))

        plot_source_real_vs_parametric(
            source_ds=encoded_intersets[i],
            domain_stats=domain_stats,
            classes=(3, 6, 8, 9),                 # or e.g. (3,6,8,9)
            n_per_class="match",          # match real class counts
            pool="auto",                  # GAP if 4D, else no-op
            save_path=os.path.join(plot_dir, f"src_vs_param_pair{i}.png"),
            seed=0,
        )
        plot_real_vs_fullcov_vs_diag_samples(
            source_ds=encoded_intersets[i],
            domain_stats=domain_stats,
            classes=(3, 6, 8, 9),          # or None for all present
            n_per_class="match",           # or an int like 500
            pool="auto",
            pca_dim=2,
            seed=0,
            save_path=os.path.join(plot_dir, f"src_real_vs_full_fullvsdiag_pair{i}.png"),
        )


        # store labels returned by generator (pseudo-mapped labels for target)
        try:
            labels_from_gen_last = out[1]
        except Exception:
            labels_from_gen_last = None
        # include the appended target from the generator; self_train will hold it out
        synthetic_domains += pair_domains



    if not synthetic_domains:
        return direct_acc, st_acc, direct_acc_all, st_acc_all, 0.0

    direct_acc_syn, generated_acc, train_acc_by_domain, test_acc_by_domain = self_train(
        args,
        source_model.mlp,
        synthetic_domains,
        epochs=epochs,
        label_source=args.label_source,
    )
    _save_list(os.path.join(plot_dir, "ours_train_acc_by_domain.json"), train_acc_by_domain)
    _save_list(os.path.join(plot_dir, "ours_test_acc_by_domain.json"), test_acc_by_domain)

    tgt_pl, _ = get_pseudo_labels(
        encoded_intersets[-1],
        getattr(source_model, 'mlp', source_model),
        confidence_q=getattr(args, 'pseudo_confidence_q', 0.0),
    )
    plot_pca_classes_grid(
        encoded_intersets,
        classes=(3, 6, 8, 9),
        save_path=os.path.join(plot_dir, f"pca_classes_real_domains.png"),
        label_source='real'
    )
    # Synthetic chain if available: source (domain 0), then generated steps, then final target
    if synthetic_domains:
        chain = []
        step_len = int(generated_domains) + 1  # per pair: n_inter steps + appended right-side domain
        for k in range(0, len(synthetic_domains), max(step_len, 1)):
            chunk = synthetic_domains[k:k + step_len]
            if not chunk:
                continue
            # exclude the last element (the right-hand endpoint) from each chunk
            chain.extend(chunk[:-1] if step_len > 0 else chunk)
        # prepend source and append final target to show full path
        chain_for_plot = [encoded_intersets[0]] + chain + [encoded_intersets[-1]]
        # Color the target by pseudo labels rather than GT or EM labels
        try:
            encoded_intersets[-1].targets_em = torch.as_tensor(pseudolabels, dtype=torch.long)
        except Exception as _e:
            print(f"[MainAlgo][PCA] Warning: failed to attach pseudo labels to target for coloring: {_e}")
        plot_pca_classes_grid(
            chain_for_plot,
            classes=(3, 6, 8, 9),
            save_path=os.path.join(plot_dir, f"pca_classes_synth_{_method}_{args.label_source}_{args.em_match}.png"),
            label_source=args.label_source,
            pseudolabels=tgt_pl
        )

    return train_acc_by_domain, test_acc_by_domain, st_acc, st_acc_all, generated_acc, acc_em_pseudo

# --------------------- Specific experiments ---------------------




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
    import numpy as np
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


# Backward-compatible wrapper for exactly three series
def _plot_three_series_with_baselines(
    y1, y2, y3,
    lab1, lab2, lab3,
    st1=None, st_all1=None,
    st2=None, st_all2=None,
    st3=None, st_all3=None,
    ref_line_value=None, ref_line_label=None, ref_line_style="--",
    title="",
    ylabel="Accuracy",
    xlabel="Training Domain Index",
    save_path=None
):
    return _plot_series_with_baselines(
        series=[y1, y2, y3],
        labels=[lab1, lab2, lab3],
        baselines=[(st1, st_all1), (st2, st_all2), (st3, st_all3)],
        ref_line_value=ref_line_value,
        ref_line_label=ref_line_label,
        ref_line_style=ref_line_style,
        title=title,
        ylabel=ylabel,
        xlabel=xlabel,
        save_path=save_path,
    )
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
def adjacent_distances(domain_stats: dict, report_bures: bool = True):
    ts = _steps(domain_stats)
    for s in range(len(ts)-1):
        mu_a = _get_mu(domain_stats, s)
        mu_b = _get_mu(domain_stats, s+1)
        present = _present_mask(domain_stats, s) & _present_mask(domain_stats, s+1)
        d_mu = np.linalg.norm((mu_b - mu_a)[present], ord='fro') / max(1, present.sum())

        if not _has_full(domain_stats):
            # diag proxy
            v_a = _get_var(domain_stats, s)[present]
            v_b = _get_var(domain_stats, s+1)[present]
            d_logv = np.mean(np.abs(np.log(np.clip(v_b,1e-12,None)) - np.log(np.clip(v_a,1e-12,None))))
            print(f"[adjacent] t={ts[s]:.2f}→{ts[s+1]:.2f} : ||Δμ||/K={d_mu:.4g}, mean |Δ logvar|={d_logv:.4g}")
        else:
            Sig_a = _get_sigma(domain_stats, s)[present]
            Sig_b = _get_sigma(domain_stats, s+1)[present]
            # logdet delta (volume)
            ld_a = _logdet_cov(Sig_a)
            ld_b = _logdet_cov(Sig_b)
            d_ld = float(np.mean(np.abs(ld_b - ld_a)))
            line = f"[adjacent] t={ts[s]:.2f}→{ts[s+1]:.2f} : ||Δμ||/K={d_mu:.4g}, mean |Δ logdet|={d_ld:.4g}"
            if report_bures:
                def bures2(A,B):
                    wA, VA = np.linalg.eigh(A); wA=np.clip(wA,0, None)
                    As = (VA*np.sqrt(wA))@VA.T
                    M = As@B@As
                    wM, VM = np.linalg.eigh(M); wM=np.clip(wM,0,None)
                    Ms = (VM*np.sqrt(wM))@VM.T
                    return np.trace(A)+np.trace(B)-2*np.trace(Ms)
                b_list=[]
                for k in range(Sig_a.shape[0]):
                    b_list.append(bures2(Sig_a[k], Sig_b[k]))
                line += f", mean Bures^2={float(np.mean(b_list)):.4g}"
            print(line)

def summarize_domain_stats(domain_stats: dict):
    """
    Print a short audit of the domain parameters across steps.
    Now also reports the average per-class logdet(Σ), which is the term
    that evolves (approximately) linearly under Fisher–Rao interpolation
    (log-variances interpolate linearly; for full Σ this tracks the sum of log-eigenvalues).
    """
    import numpy as np

    # ---- helpers expected to exist elsewhere ----
    # _has_full(domain_stats)        -> bool   (whether "Sigma" (full cov) is present)
    # _steps(domain_stats)           -> np.ndarray of shape (S,)
    # _present_mask(domain_stats,s)  -> boolean mask (K,) of classes present at step s
    # _get_sigma(domain_stats,s)     -> covariance for step s:
    #                                   full: (K,d,d), diag: either (K,d) or already expanded
    # _trace_cov(Sig)                -> per-class trace, shape (K,)

    def _logdet_cov_per_class(Sig, eps=1e-12):
        """
        Return per-class logdet(Σ_k) as a vector of length K.
        Accepts:
          - full cov:  Sig shape (K,d,d)
          - diag cov:  Sig shape (K,d)   (interpreted as diagonal entries)
        """
        Sig = np.asarray(Sig)
        if Sig.ndim == 3:  # full
            K = Sig.shape[0]
            out = np.empty(K, dtype=np.float64)
            for k in range(K):
                # eigvalsh is stable/symmetric; clip tiny negatives from numerical noise
                w = np.linalg.eigvalsh(Sig[k])
                w = np.clip(w, eps, None)
                out[k] = np.sum(np.log(w))
            return out
        elif Sig.ndim == 2:  # diag
            w = np.clip(Sig, eps, None)
            return np.sum(np.log(w), axis=1)
        else:
            raise ValueError(f"_logdet_cov_per_class: unexpected Sig.ndim={Sig.ndim}")

    S = len(domain_stats["steps"])
    K = int(domain_stats["K"])
    d = int(domain_stats["d"])
    cov_type = "full" if _has_full(domain_stats) else "diag"
    print(f"[audit:{cov_type.upper()}] S={S}, K={K}, d={d}")

    for s, t in enumerate(_steps(domain_stats)):
        present = _present_mask(domain_stats, s)
        cnt_sum = int(np.asarray(domain_stats["counts"][s])[present].sum())

        Sig = _get_sigma(domain_stats, s)           # (K,d,d) or (K,d)
        tr_all = _trace_cov(Sig)                    # (K,)
        try:
            logdet_all = _logdet_cov_per_class(Sig)     # (K,)
        except Exception as e:
            print(f"[audit] Step {s} logdet computation failed: {e}")
            logdet_all = np.full(K, float("nan"), dtype=float)
            breakpoint()

        tr_avg = tr_all[present].mean() if present.any() else float("nan")
        logdet_avg = logdet_all[present].mean() if present.any() else float("nan")

        # Optional: report geometric mean variance per class (exp(logdet/d))
        geo_var_avg = float(np.exp(logdet_avg / max(d, 1))) if np.isfinite(logdet_avg) else float("nan")

        print(
            f"  step {s} (t={t:.2f}): present={present.sum()}/{K}, "
            f"sum_count={cnt_sum}, avg trace(Σ)={tr_avg:.6g}, "
            f"avg logdet(Σ)={logdet_avg:.6g}, "
            f"avg geo-var={geo_var_avg:.6g}"
        )


def check_means_between(domain_stats: dict):
    """Verify intermediate means lie within segment endpoints per class (component-wise)."""
    ts = _steps(domain_stats)
    mu0 = _get_mu(domain_stats, 0)       # (K,d)
    muT = _get_mu(domain_stats, -1)
    ok_all = True
    for s in range(1, len(ts)-1):
        mus = _get_mu(domain_stats, s)
        lb = np.minimum(mu0, muT)
        ub = np.maximum(mu0, muT)
        present = _present_mask(domain_stats, s)
        inside = ((mus >= lb) & (mus <= ub)).all(axis=1)
        if present.any() and not inside[present].all():
            ok_all = False
            break
    print("[check_means_between] All intermediate means lie within endpoint segments."
          if ok_all else
          "[check_means_between] Some intermediate means fall outside endpoint box.")

def plot_size_proxy(domain_stats: dict, mode: str = "trace", save_path: str = None):
    """mode ∈ {'trace','logdet'}; picks the right formula for diag/full."""
    ts = _steps(domain_stats)
    vals = []
    for s in range(len(ts)):
        present = _present_mask(domain_stats, s)
        Sig = _get_sigma(domain_stats, s)  # full or expanded diag
        if mode == "trace":
            v = _trace_cov(Sig)[present]
        elif mode == "logdet":
            v = _logdet_cov(Sig[present])  # (n_present,)
        else:
            raise ValueError("mode must be 'trace' or 'logdet'")
        vals.append(np.mean(v) if v.size else np.nan)
    vals = np.asarray(vals, dtype=float)

    plt.figure(figsize=(5.2,3.2))
    plt.plot(ts, vals, marker='o')
    plt.title(f"Average {mode} across classes")
    plt.xlabel("t"); plt.ylabel(mode)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150); 
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


import numpy as np
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
    import numpy as np
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(seed)

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
    import numpy as np
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

def plot_pca_classes_grid(
    domains,
    classes=(3, 8),
    save_path=None,
    pool: str = 'gap',
    label_source: str = 'pseudo',     # 'pseudo' | 'em' | 'real'
    pseudolabels=None
):
    """
    Fit ONE PCA on all selected samples across domains, then apply it per-domain.

    - domains: iterable of datasets with .data and labels (.targets or .targets_em)
    - classes: tuple/list of class ids to display
    - save_path: file path to save the figure (directory is created)
    - pool: 'gap' to global-average-pool 4D (N,C,H,W) -> (N,C); 'flatten' to flatten; 'auto' chooses GAP for 4D
    - label_source: 'pseudo' (use provided pseudo labels), 'em' (use .targets_em fallback to .targets), 'real' (use .targets)
    - pseudolabels: dict[id(D)]->labels OR list/tuple aligned with `domains` OR array/tensor of labels per dataset
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    from sklearn.decomposition import PCA

    # ---------- helpers ----------
    def _to_np(y):
        if y is None:
            return None
        if isinstance(y, torch.Tensor):
            return y.detach().cpu().numpy()
        return np.asarray(y)

    def pool_feats(X):
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        if X.ndim == 4:
            if pool == 'gap' or pool == 'auto':
                return X.mean(axis=(2, 3))
            elif pool == 'flatten':
                return X.reshape(X.shape[0], -1)
            else:
                # default to GAP for 4D if an unexpected value is passed
                return X.mean(axis=(2, 3))
        if X.ndim == 3:
            return X.reshape(X.shape[0], -1)
        return X

    def get_labels_for_domain(D, j):
        """Return labels per `label_source`, robust to different pseudolabels types."""
        y = None
        if label_source == 'pseudo':
            if isinstance(pseudolabels, dict):
                y = pseudolabels.get(id(D), None)
            elif isinstance(pseudolabels, (list, tuple)):
                if 0 <= j < len(pseudolabels):
                    y = pseudolabels[j]
            elif isinstance(pseudolabels, (np.ndarray, torch.Tensor)):
                # assume it's aligned with D
                y = pseudolabels
            # fallbacks if pseudo not available
            if y is None:
                y = getattr(D, 'targets_em', None)
                if y is None:
                    y = getattr(D, 'targets', None)

        elif label_source == 'em':
            y = getattr(D, 'targets_em', None)
            if y is None or (isinstance(y, torch.Tensor) and y.numel() > 0 and (y < 0).all()):
                y = getattr(D, 'targets', None)

        elif label_source == 'real':
            y = getattr(D, 'targets', None)

        return _to_np(y)

    # ---------- prepare ----------
    cols = len(domains)
    if cols == 0:
        return

    classes = np.array(list(classes), dtype=int)

    # ---------- 1) Collect data for a single global PCA ----------
    X_all = []
    y_all = []
    pooled_per_domain = []
    masks_per_domain = []

    for j, D in enumerate(domains):
        Xp = pool_feats(D.data)
        pooled_per_domain.append(Xp)
        if j == len(domains) - 1:
            y = get_labels_for_domain(D, j)
        else:
            y = D.targets if label_source =='pseudo' else D.targets_em
        if y is None:
            masks_per_domain.append(None)
            continue

        # keep only requested classes
        m = np.isin(y, classes)
        masks_per_domain.append(m)

        if m.any():
            try:
                X_all.append(Xp[m])
                y_all.append(y[m])
            except Exception as e:
                breakpoint()
                print(f"[plot] Failed to append data for domain {j} ({e})")

    if len(X_all) == 0:
        print("[plot] No samples from requested classes; nothing to plot.")
        return

    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)

    # ---------- PCA fit (shared) ----------
    try:
        pca = PCA(n_components=2)
        pca.fit(X_all)
        use_pca = True
    except Exception as e:
        print(f"[plot] PCA fit failed ({e}); falling back to first two dims.")
        use_pca = False

    # ---------- 2) Plot per domain using the SAME PCA ----------
    fig_w = max(4, 3 * cols)
    fig, axs = plt.subplots(1, cols, figsize=(fig_w, 3.6), squeeze=False)
    axs = axs[0]
    cmap = plt.get_cmap('tab10')

    for j, D in enumerate(domains):
        ax = axs[j]
        Xp = pooled_per_domain[j]
        if j == len(domains) - 1:
            y = get_labels_for_domain(D, j)
        else:
            y = D.targets if label_source =='pseudo' else D.targets_em

        if y is None:
            ax.set_title(f"Domain {j}: no labels")
            ax.axis('off')
            continue

        m = masks_per_domain[j]
        if m is None or not np.any(m):
            ax.set_title(f"Domain {j}: no classes {tuple(classes)}")
            ax.axis('off')
            continue

        Xsel = Xp[m]
        ysel = y[m]

        if use_pca:
            try:
                Z = pca.transform(Xsel)
            except Exception:
                Z = Xsel[:, :2] if Xsel.shape[1] >= 2 else np.pad(
                    Xsel, ((0, 0), (0, max(0, 2 - Xsel.shape[1]))), mode='constant'
                )
        else:
            Z = Xsel[:, :2] if Xsel.shape[1] >= 2 else np.pad(
                Xsel, ((0, 0), (0, max(0, 2 - Xsel.shape[1]))), mode='constant'
            )

        # plot
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
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"[MNIST-EXP] Saved {save_path}")
    plt.close()

# ---------------- Run all three methods and compare on shared plots ----------------
def run_mnist_experiment(target: int, gt_domains: int, generated_domains: int, args=None):
    src_trainset = get_single_rotate(False, 0)
    tgt_trainset = get_single_rotate(False, target)
    model_dir = f"/data/common/yuenchen/GDA/mnist_models/"

    encoder = ENCODER().to(device)
    model_name = f"src0_tgt{target}_ssl{args.ssl_weight}.pth"
    source_model = get_source_model(
        args, src_trainset, tgt_trainset, n_class=10, mode="mnist",
        encoder=encoder, epochs=10, model_path=f"{model_dir}/{model_name}",
        target_dataset=tgt_trainset, force_recompute=False, compress=False
    )
    model_name_smalldim = f"src0_tgt{target}_ssl{args.ssl_weight}_dim{args.small_dim}.pth"
    source_model_smalldim = get_source_model(
        args, src_trainset, tgt_trainset, n_class=10, mode="mnist",
        encoder=encoder, epochs=10, model_path=f"{model_dir}/{model_name_smalldim}",
        target_dataset=tgt_trainset, force_recompute=False, compress=True,
        in_dim=25088, out_dim=args.small_dim
    )

    # SAME reference for all runs
    ref_model   = source_model_smalldim
    our_source = copy.deepcopy(ref_model)
    ours_copy   = copy.deepcopy(ref_model)
    # For ETA/natural path
    our_source_eta = copy.deepcopy(ref_model)
    ours_copy_eta  = copy.deepcopy(ref_model)
    goat_source = copy.deepcopy(ref_model)
    goat_copy   = copy.deepcopy(goat_source)
    goat_cw_src = copy.deepcopy(ref_model)
    goat_cw_cp  = copy.deepcopy(goat_cw_src)

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
    init_acc = test(tgt_loader_eval, ref_model)[1]
    print(f"[MNIST-EXP] Initial target accuracy (pre-adapt): {init_acc:.4f}")
    # set_all_seeds(args.seed)
    # # ---- (1) Ours/FR path (returns EM mapping accuracy too) ----
    # ours_train, ours_test, ours_st, ours_st_all, ours_gen, EM_acc = run_main_algo(
    #     ours_copy, our_source, src_trainset, tgt_trainset, all_sets, deg_idx,
    #     generated_domains, epochs=5, target=target, args=args
    # )


    # # ---- (1b) Ours/ETA (natural-parameter) path ----
    set_all_seeds(args.seed)
    ours_eta_train, ours_eta_test, ours_eta_st, ours_eta_st_all, ours_eta_gen, EM_acc_eta = run_main_algo(
        ours_copy_eta, our_source_eta, src_trainset, tgt_trainset, all_sets, deg_idx,
        generated_domains, epochs=5, target=target, args=args, gen_method="natural"
    )


    set_all_seeds(args.seed)
    # ---- (2) GOAT (pair-wise synthetics) ----
    goat_train, goat_test, goat_st, goat_st_all, goat_gen = run_goat(
        goat_copy, goat_source, src_trainset, tgt_trainset, all_sets, deg_idx,
        generated_domains, epochs=5, target=target, args=args
    )
    set_all_seeds(args.seed)
    # ---- (3) GOAT (class-wise synthetics) ----
    goatcw_train, goatcw_test, goatcw_st, goatcw_st_all, goatcw_gen = run_goat_classwise(
        goat_cw_cp, goat_cw_src, src_trainset, tgt_trainset, all_sets, deg_idx,
        generated_domains, epochs=5, target=target, args=args
    )

    # ---- Persist series ----
    plot_dir = f"plots/target{target}/"
    os.makedirs(plot_dir, exist_ok=True)

    # Aggregate results: store/load one JSON per method, with all returned values.
    ours_path     = os.path.join(plot_dir, f"ours_results_{args.label_source}_{args.em_match}.json")
    ours_eta_path = os.path.join(plot_dir, f"ours_eta_results_{args.label_source}_{args.em_match}.json")
    goat_path     = os.path.join(plot_dir, f"goat_results_{args.label_source}_{args.em_match}.json")
    goatcw_path   = os.path.join(plot_dir, f"goatcw_results_{args.label_source}_{args.em_match}.json")

    if not (ours_train and ours_test):
        # Prefer aggregated files; fall back to legacy per-list files if missing.
        ours_loaded = _load_list(ours_path)
        if isinstance(ours_loaded, dict):
            ours_train = ours_loaded.get("train")
            ours_test  = ours_loaded.get("test")
            ours_st    = ours_loaded.get("st", ours_st)
            ours_st_all= ours_loaded.get("st_all", ours_st_all)
            ours_gen   = ours_loaded.get("gen", ours_gen)
            EM_acc     = ours_loaded.get("em_acc")
        else:
            ours_train = _load_list(os.path.join(plot_dir, f"ours_train_acc_{args.label_source}_{args.em_match}.json"))
            ours_test  = _load_list(os.path.join(plot_dir, f"ours_test_acc_{args.label_source}_{args.em_match}.json"))
            EM_acc     = None

        ours_eta_loaded = _load_list(ours_eta_path)
        if isinstance(ours_eta_loaded, dict):
            ours_eta_train = ours_eta_loaded.get("train")
            ours_eta_test  = ours_eta_loaded.get("test")
            ours_eta_st    = ours_eta_loaded.get("st", ours_eta_st)
            ours_eta_st_all= ours_eta_loaded.get("st_all", ours_eta_st_all)
            ours_eta_gen   = ours_eta_loaded.get("gen", ours_eta_gen)
            EM_acc_eta     = ours_eta_loaded.get("em_acc")
        else:
            ours_eta_train = _load_list(os.path.join(plot_dir, f"ours_eta_train_acc_{args.label_source}_{args.em_match}.json"))
            ours_eta_test  = _load_list(os.path.join(plot_dir, f"ours_eta_test_acc_{args.label_source}_{args.em_match}.json"))
            EM_acc_eta     = None

        goat_loaded = _load_list(goat_path)
        if isinstance(goat_loaded, dict):
            goat_train = goat_loaded.get("train")
            goat_test  = goat_loaded.get("test")
            goat_st    = goat_loaded.get("st", goat_st)
            goat_st_all= goat_loaded.get("st_all", goat_st_all)
            goat_gen   = goat_loaded.get("gen", goat_gen)
        else:
            goat_train = _load_list(os.path.join(plot_dir, f"goat_train_acc_{args.label_source}_{args.em_match}.json"))
            goat_test  = _load_list(os.path.join(plot_dir, f"goat_test_acc_{args.label_source}_{args.em_match}.json"))

        goatcw_loaded = _load_list(goatcw_path)
        if isinstance(goatcw_loaded, dict):
            goatcw_train = goatcw_loaded.get("train")
            goatcw_test  = goatcw_loaded.get("test")
            goatcw_st    = goatcw_loaded.get("st", goatcw_st)
            goatcw_st_all= goatcw_loaded.get("st_all", goatcw_st_all)
            goatcw_gen   = goatcw_loaded.get("gen", goatcw_gen)
        else:
            goatcw_train = _load_list(os.path.join(plot_dir, f"goatcw_train_acc_{args.label_source}_{args.em_match}.json"))
            goatcw_test  = _load_list(os.path.join(plot_dir, f"goatcw_test_acc_{args.label_source}_{args.em_match}.json"))
    else:
        # Save aggregated results per method
        _save_list(ours_path, {
            "train": ours_train,
            "test":  ours_test,
            "st":    ours_st,
            "st_all":ours_st_all,
            "gen":   ours_gen,
            "em_acc": EM_acc,
        })
        _save_list(ours_eta_path, {
            "train": ours_eta_train,
            "test":  ours_eta_test,
            "st":    ours_eta_st,
            "st_all":ours_eta_st_all,
            "gen":   ours_eta_gen,
            "em_acc": EM_acc_eta,
        })
        _save_list(goat_path, {
            "train": goat_train,
            "test":  goat_test,
            "st":    goat_st,
            "st_all":goat_st_all,
            "gen":   goat_gen,
        })
        _save_list(goatcw_path, {
            "train": goatcw_train,
            "test":  goatcw_test,
            "st":    goatcw_st,
            "st_all":goatcw_st_all,
            "gen":   goatcw_gen,
        })


    # load results for plotting if not run just now
# # ---- Plot 1: Training accuracy (now 4-way, includes ETA) ----
    # _plot_series_with_baselines(
    #     series=[ours_train, goat_train, goatcw_train, ours_eta_train],
    #     labels=[
    #         f"Ours-FR-{args.label_source}",
    #         "GOAT",
    #         f"GOAT-Classwise-{args.label_source}",
    #         f"Ours-ETA-{args.label_source}",
    #     ],
    #     baselines=[
    #         (ours_st, ours_st_all),
    #         # (goat_st, goat_st_all),
    #         # (goatcw_st, goatcw_st_all),
    #         # (ours_eta_st, ours_eta_st_all),
    #     ],
    #     ref_line_value=None, ref_line_label=None,
    #     title="Training Accuracy by Domain (Ours-FR/ETA vs GOAT vs GOAT-Classwise)",
    #     ylabel="Accuracy", xlabel="Training Domain Index",
    #     save_path=os.path.join(plot_dir, "compare_train_acc_by_domain_4way.png"),
    # )

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
        save_path=os.path.join(plot_dir, f"compare_test_acc_with_baselines_4way_{args.label_source}_{args.em_match}.png"),
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





def run_portraits_experiment(gt_domains: int, generated_domains: int):
    t0 = time.time()

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

    source_model = get_source_model(
        args,
        src_trainset,
        src_trainset,
        2,
        mode="portraits",
        encoder=encoder,
        epochs=20,
        model_path=f"portraits/cache{args.ssl_weight}/source_model.pth",
        target_dataset=tgt_trainset,
        force_recompute=False,
    )

    all_sets = []
    n2idx = {0: [], 1: [3], 2: [2, 4], 3: [1, 3, 5], 4: [0, 2, 4, 6], 7: [0, 1, 2, 3, 4, 5, 6]}
    for i in n2idx[gt_domains]:
        start, end = i * 2000, (i + 1) * 2000
        all_sets.append(EncodeDataset(inter_x[start:end], inter_y[start:end].astype(int), transforms))
    all_sets.append(tgt_trainset)

    model_copy = copy.deepcopy(source_model)
    _d, _s, _da, _sa, gen_acc = run_goat(
        model_copy, source_model, src_trainset, tgt_trainset, all_sets, deg_idx=list(range(len(all_sets))), generated_domains=generated_domains, epochs=5
    )

    with open("logs/portraits_exp_time.txt", "a") as f:
        elapsed = round(time.time() - t0, 2)
        f.write(f"seed{args.seed}with{gt_domains}gt{generated_domains}generated,{elapsed},{round(gen_acc, 2)}\n")


def run_covtype_experiment(gt_domains: int, generated_domains: int):
    data = make_cov_data(40000, 10000, 400000, 50000, 25000, 20000)
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
    ) = data

    src_trainset = EncodeDataset(torch.from_numpy(src_val_x).float(), src_val_y.astype(int))
    tgt_trainset = EncodeDataset(torch.from_numpy(trg_test_x).float(), torch.tensor(trg_test_y.astype(int)))

    encoder = MLP_Encoder().to(device)
    source_model = get_source_model(args, src_trainset, src_trainset, 2, mode="covtype", encoder=encoder, epochs=5)

    def get_domains(n_domains: int) -> List[Dataset]:
        idx_map = {0: [], 1: [6], 2: [3, 7], 3: [2, 5, 8], 4: [2, 4, 6, 8], 5: [1, 3, 5, 7, 9], 10: range(10), 200: range(200)}
        domain_idx = idx_map[n_domains]
        domains = []
        for i in domain_idx:
            start, end = i * 40000, i * 40000 + 2000
            domains.append(EncodeDataset(torch.from_numpy(inter_x[start:end]).float(), inter_y[start:end].astype(int)))
        return domains

    all_sets = get_domains(gt_domains)
    all_sets.append(tgt_trainset)

    model_copy = copy.deepcopy(source_model)
    _d, _s, _da, _sa, gen_acc = run_goat(
        model_copy, source_model, src_trainset, tgt_trainset, all_sets, deg_idx=list(range(len(all_sets))), generated_domains=generated_domains, epochs=5
    )

    os.makedirs("logs", exist_ok=True)
    with open(f"logs/covtype_exp_{args.log_file}.txt", "a") as f:
        f.write(f"seed{args.seed}with{gt_domains}gt{generated_domains}generated,{round(gen_acc, 2)}\n")


def run_color_mnist_experiment(gt_domains: int, generated_domains: int):
    shift = 1
    total_domains = 20

    (
        src_tr_x,
        src_tr_y,
        src_val_x,
        src_val_y,
        _dir_inter_x,
        _dir_inter_y,
        _dir_inter_x2,
        _dir_inter_y2,
        trg_val_x,
        trg_val_y,
        trg_test_x,
        trg_test_y,
    ) = ColorShiftMNIST(shift=shift)

    inter_x, inter_y = transform_inter_data(_dir_inter_x, _dir_inter_y, 0, shift, interval=len(_dir_inter_x) // total_domains, n_domains=total_domains)

    src_x = np.concatenate([src_tr_x, src_val_x])
    src_y = np.concatenate([src_tr_y, src_val_y])
    tgt_x = np.concatenate([trg_val_x, trg_test_x])
    tgt_y = np.concatenate([trg_val_y, trg_test_y])

    src_trainset = EncodeDataset(src_x, src_y.astype(int), ToTensor())
    tgt_trainset = EncodeDataset(trg_val_x, trg_val_y.astype(int), ToTensor())

    # VAE encoder
    vae = VAE(x_dim=28 * 28, z_dim=16).to(device)
    vae_path = "models/colored_mnist/vae.pt"
    if os.path.exists(vae_path):
        vae.load_state_dict(torch.load(vae_path, map_location=device))
    else:
        # Expect train_vae to be available from dataset/util
        train_vae(vae, None, None, None, vae_path, save=True)  # placeholders if you hook real loaders

    source_model = get_source_model(args, src_trainset, src_trainset, 10, "mnist", encoder=vae.encoder, epochs=20)

    def get_domains(n_domains: int) -> List[Dataset]:
        domain_idx: List[int]
        if n_domains == total_domains:
            domain_idx = list(range(n_domains))
        else:
            domain_idx = [total_domains // (n_domains + 1) * i for i in range(1, n_domains + 1)]
        interval = 42000 // total_domains
        domains = []
        for i in domain_idx:
            start, end = i * interval, (i + 1) * interval
            domains.append(EncodeDataset(inter_x[start:end], inter_y[start:end].astype(int), ToTensor()))
        return domains

    all_sets = get_domains(gt_domains)
    all_sets.append(tgt_trainset)

    model_copy = copy.deepcopy(source_model)
    _d, _s, _da, _sa, gen_acc = run_goat(
        model_copy, source_model, src_trainset, tgt_trainset, all_sets, deg_idx=list(range(len(all_sets))), generated_domains=generated_domains, epochs=10
    )

    os.makedirs("logs", exist_ok=True)
    with open(f"logs/color{args.log_file}.txt", "a") as f:
        f.write(f"seed{args.seed}with{gt_domains}gt{generated_domains}generated,{round(gen_acc, 2)}\n")


# -------------------------------------------------------------
# TensorBoard image logging
# -------------------------------------------------------------

def log_generated_images_tensorboard(writer: SummaryWriter, images, step: int, tag: str = "Generated Images") -> None:
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images)

    if images.ndim == 3 and images.shape[2] == 32:
        H, W, N = images.shape
        images = images.transpose(2, 0, 1).reshape(N, 1, H, W)  # (N,1,H,W)
    elif images.ndim != 4:
        print(f"[TB] Unsupported image shape: {tuple(images.shape)}")
        return

    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)
    elif images.shape[1] > 3:
        images = images[:, :3]

    grid = torchvision.utils.make_grid(images[:16], nrow=4, normalize=True)
    writer.add_image(tag, grid, step)


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

        elif cli_args.mnist_mode == "compare":
            compare_em_vs_pseudo_on_sets(cli_args.rotation_angle, cli_args.gt_domains)
        else:
            raise ValueError(f"Unknown mnist-mode: {cli_args.mnist_mode}")
    elif cli_args.dataset == "portraits":
        run_portraits_experiment(cli_args.gt_domains, cli_args.generated_domains)
    elif cli_args.dataset == "covtype":
        run_covtype_experiment(cli_args.gt_domains, cli_args.generated_domains)
    elif cli_args.dataset == "color_mnist":
        run_color_mnist_experiment(cli_args.gt_domains, cli_args.generated_domains)
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
    parser.add_argument("--em-match",  choices=["pseudo", "prototypes"], default="pseudo", help="For self-training, which labels to use for pseudo-labeling")

    args = parser.parse_args()
    main(args)
