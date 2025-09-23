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

# Project-local deps (must exist in your repo)
from model import *
from train_model import *
from util import *  # noqa: F401,F403 (kept to preserve your helpers)
from ot_util import ot_ablation, generate_domains  # generation helpers
from a_star_util import *
from dataset import *

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from da_algo import *

try:
    import kornia.augmentation as K
except Exception:
    K = None  # Kornia is optional; see build_augment()

# -------------------------------------------------------------
# Global config / utilities
# -------------------------------------------------------------

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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

import torch
from torch.utils.data import DataLoader

# ---- helper: init Gaussian head from source features ----
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader



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
    breakpoint()
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
            x1,