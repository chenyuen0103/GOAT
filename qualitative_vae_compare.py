#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import TensorDataset

from dataset import DomainDataset
from experiment_refrac import (
    ModelConfig,
    build_em_bundles_for_chain,
    build_reference_model,
    encode_real_domains,
    load_encoded_domains,
    run_core_methods,
)
from model import ENCODER, VAE
from ot_util import generate_domains
from util import get_single_rotate
from a_star_util import (
    generate_fr_domains_between_optimized,
    generate_natural_domains_between,
)
from experiment_new import set_all_seeds


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class FeatureToImageVAE(nn.Module):
    def __init__(self, feature_dim: int, latent_dim: int = 256, image_dim: int = 784) -> None:
        super().__init__()
        hidden = 1024
        self.feature_enc = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)
        # Reuse the VAE decoder architecture defined in model.py
        base_vae = VAE(x_dim=image_dim, z_dim=latent_dim, h_dim=hidden)
        self.image_dec = base_vae.decoder
        self.feature_dim = int(feature_dim)
        self.latent_dim = int(latent_dim)
        self.image_dim = int(image_dim)

    def encode_feature(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.feature_enc(feat)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.image_dec(z)
        return x.view(-1, 1, 28, 28)

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode_feature(feat)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    @torch.no_grad()
    def decode_from_features(self, feat: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        mu, logvar = self.encode_feature(feat)
        z = mu if deterministic else self.reparameterize(mu, logvar)
        return self.decode(z)


def _ensure_default_args(args: argparse.Namespace) -> argparse.Namespace:
    defaults = {
        "dataset": "mnist",
        "lr": 1e-4,
        "pseudo_confidence_q": 0.9,
        "use_labels": False,
        "diet": False,
        "mnist_mode": "normal",
        "em_cov_types": ["diag"],
        "em_seeds": [0, 1, 2],
        "em_pca_dims": [None],
        "em_K_list": [10],
    }
    for key, val in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, val)
    return args


def _build_rotated_domains(tgt_trainset, target: int, gt_domains: int):
    all_sets, deg_idx = [], []
    for i in range(1, gt_domains + 1):
        angle = i * target // (gt_domains + 1)
        all_sets.append(get_single_rotate(False, angle))
        deg_idx.append(angle)
    all_sets.append(tgt_trainset)
    deg_idx.append(target)
    return all_sets, deg_idx


def _get_label_tensor(ds: Dataset, use_em: bool) -> torch.Tensor:
    if use_em and hasattr(ds, "targets_em") and ds.targets_em is not None:
        return torch.as_tensor(ds.targets_em).long()
    if hasattr(ds, "targets") and ds.targets is not None:
        return torch.as_tensor(ds.targets).long()
    raise ValueError("Dataset has no valid labels.")


def _subset_domain_by_class(ds: Dataset, cls: int, use_em: bool) -> Optional[DomainDataset]:
    y = _get_label_tensor(ds, use_em=use_em)
    if not hasattr(ds, "data"):
        return None
    x = ds.data if torch.is_tensor(ds.data) else torch.as_tensor(ds.data)
    mask = y == int(cls)
    if mask.sum().item() == 0:
        return None
    x_c = x[mask]
    y_c = y[mask]
    w_c = torch.ones(len(y_c), dtype=torch.float32)
    return DomainDataset(x_c, w_c, y_c, y_c)


def _merge_class_chains(class_chains: List[List[DomainDataset]]) -> List[DomainDataset]:
    if not class_chains:
        return []
    n_steps = min(len(c) for c in class_chains)
    merged: List[DomainDataset] = []
    for step in range(n_steps):
        chunks = [c[step] for c in class_chains if c[step] is not None]
        if not chunks:
            continue
        xs, ws, ys, yems = [], [], [], []
        for ds in chunks:
            xs.append(ds.data if torch.is_tensor(ds.data) else torch.as_tensor(ds.data))
            ws.append(
                ds.weight if hasattr(ds, "weight") else torch.ones(len(ds.targets), dtype=torch.float32)
            )
            ys.append(ds.targets if torch.is_tensor(ds.targets) else torch.as_tensor(ds.targets))
            yems.append(ds.targets_em if torch.is_tensor(ds.targets_em) else torch.as_tensor(ds.targets_em))
        x = torch.cat([t.float().cpu() for t in xs], dim=0)
        w = torch.cat([t.float().cpu() for t in ws], dim=0)
        y = torch.cat([t.long().cpu() for t in ys], dim=0)
        yem = torch.cat([t.long().cpu() for t in yems], dim=0)
        merged.append(DomainDataset(x, w, y, yem))
    return merged


def extract_method_domains(
    args: argparse.Namespace,
    encoded_domains: Sequence[Dataset],
    generated_domains: int,
) -> Dict[str, List[Dataset]]:
    if generated_domains <= 0:
        raise ValueError(
            "generated_domains must be > 0 for qualitative intermediate comparison."
        )
    methods: Dict[str, List[Dataset]] = {
        "goat": [],
        "goat_classwise": [],
        "ours_fr": [],
        "ours_eta": [],
    }

    n_pairs = len(encoded_domains) - 1
    if n_pairs <= 0:
        raise ValueError("Need at least source and target encoded domains.")

    k_classes = int(
        torch.as_tensor(encoded_domains[0].targets).max().item()
    ) + 1

    for i in range(n_pairs):
        left = encoded_domains[i]
        right = encoded_domains[i + 1]

        goat_chain, _, _ = generate_domains(generated_domains, left, right)
        # GOAT generator does not attach labels; inherit left-domain labels by index.
        left_labels = _get_label_tensor(left, use_em=(i > 0))
        for ds in goat_chain[:-1]:
            if len(ds.data) == len(left_labels):
                ds.targets = left_labels.clone().cpu()
                ds.targets_em = left_labels.clone().cpu()
        methods["goat"].extend(goat_chain[:-1])

        class_chains: List[List[DomainDataset]] = []
        for c in range(k_classes):
            src_c = _subset_domain_by_class(left, c, use_em=(i > 0))
            tgt_c = _subset_domain_by_class(right, c, use_em=True)
            if src_c is None or tgt_c is None:
                continue
            chain_c, _, _ = generate_domains(generated_domains, src_c, tgt_c)
            for ds in chain_c:
                labels = torch.full((len(ds.targets),), c, dtype=torch.long)
                ds.targets = labels.clone()
                ds.targets_em = labels.clone()
            class_chains.append(chain_c)
        merged = _merge_class_chains(class_chains)
        methods["goat_classwise"].extend(merged[:-1])

        fr_chain, _, _ = generate_fr_domains_between_optimized(
            generated_domains,
            left,
            right,
            cov_type="full",
            args=args,
        )
        methods["ours_fr"].extend(fr_chain[:-1])

        nat_chain, _, _ = generate_natural_domains_between(
            generated_domains,
            left,
            right,
            cov_type="full",
            args=args,
        )
        methods["ours_eta"].extend(nat_chain[:-1])

    return methods


def _labels_for_domain(ds: Dataset) -> torch.Tensor:
    if hasattr(ds, "targets") and ds.targets is not None:
        return torch.as_tensor(ds.targets).long().view(-1)
    if hasattr(ds, "targets_em") and ds.targets_em is not None:
        return torch.as_tensor(ds.targets_em).long().view(-1)
    return torch.full((len(ds.data),), -1, dtype=torch.long)


def _common_indices_for_class(
    trajectory_chains: Dict[str, Sequence[Dataset]],
    methods: Sequence[str],
    cls: int,
    n_samples: int,
    seed: int,
) -> np.ndarray:
    check_domains: List[Dataset] = []
    for m in methods:
        check_domains.extend(list(trajectory_chains[m]))

    min_len = min(len(ds.data) for ds in check_domains)
    if min_len <= 0:
        return np.array([], dtype=np.int64)

    common = np.ones(min_len, dtype=bool)
    for ds in check_domains:
        y = _labels_for_domain(ds).cpu().numpy()[:min_len]
        common &= (y == int(cls))
    idx = np.where(common)[0]
    if len(idx) == 0:
        return idx

    n = min(n_samples, len(idx))
    rng = np.random.default_rng(seed + 9000 + int(cls))
    picked = np.sort(rng.choice(idx, size=n, replace=False))
    return picked.astype(np.int64)


def _vae_checkpoint_path(args: argparse.Namespace, output_dir: str, latent_dim: int) -> str:
    if args.vae_path:
        return args.vae_path
    return os.path.join(
        output_dir,
        f"feature_vae_target{args.target_angle}_z{latent_dim}_seed{args.seed}.pt",
    )


def train_or_load_vae(
    args: argparse.Namespace,
    train_dataset: Dataset,
    train_features: torch.Tensor,
    latent_dim: int,
    output_dir: str,
) -> FeatureToImageVAE:
    ckpt_path = _vae_checkpoint_path(args, output_dir, latent_dim)
    feature_dim = int(train_features.shape[1])
    vae = FeatureToImageVAE(feature_dim=feature_dim, latent_dim=latent_dim, image_dim=28 * 28).to(DEVICE)

    if os.path.exists(ckpt_path):
        payload = torch.load(ckpt_path, map_location=DEVICE)
        state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
        kind = payload.get("model_kind") if isinstance(payload, dict) else None
        feat_dim_ckpt = payload.get("feature_dim") if isinstance(payload, dict) else None
        latent_ckpt = payload.get("latent_dim") if isinstance(payload, dict) else None
        scope_ckpt = payload.get("train_scope") if isinstance(payload, dict) else None
        decoder_arch_ckpt = payload.get("decoder_arch") if isinstance(payload, dict) else None
        if (
            kind == "feature_to_image_vae"
            and feat_dim_ckpt == feature_dim
            and latent_ckpt == latent_dim
            and scope_ckpt == "source_target"
            and decoder_arch_ckpt == "model.VAE.decoder"
        ):
            vae.load_state_dict(state, strict=True)
            vae.eval()
            print(f"[VAE] Loaded checkpoint: {ckpt_path}")
            return vae
        print("[VAE] Existing checkpoint is incompatible with feature decoder; retraining.")

    feature_tensor = train_features.detach().cpu().float()
    opt = torch.optim.Adam(vae.parameters(), lr=args.vae_lr)
    if feature_tensor.size(0) != len(train_dataset):
        raise ValueError(
            f"Feature/image count mismatch: features={feature_tensor.size(0)} vs images={len(train_dataset)}"
        )

    # Build aligned (feature, image) pairs and split train/val for early stopping.
    all_images = train_dataset.tensors[0].detach().cpu().float()
    if all_images.size(0) != feature_tensor.size(0):
        raise ValueError(
            f"Image/feature pair mismatch: images={all_images.size(0)} vs features={feature_tensor.size(0)}"
        )
    n_all = int(feature_tensor.size(0))
    idx_all = np.arange(n_all)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(idx_all)
    n_val = int(round(n_all * float(args.vae_val_frac)))
    n_val = min(max(n_val, 1), max(1, n_all - 1))
    val_idx = idx_all[:n_val]
    tr_idx = idx_all[n_val:]

    tr_feat = feature_tensor[tr_idx]
    tr_img = all_images[tr_idx]
    va_feat = feature_tensor[val_idx]
    va_img = all_images[val_idx]

    train_pairs = TensorDataset(tr_feat, tr_img)
    val_pairs = TensorDataset(va_feat, va_img)
    train_loader = DataLoader(
        train_pairs,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_pairs,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    best_val = float("inf")
    best_state = copy.deepcopy(vae.state_dict())
    bad_epochs = 0

    for epoch in range(1, args.vae_epochs + 1):
        vae.train()
        running = 0.0
        n_seen = 0
        for feat, x in train_loader:
            feat = feat.to(DEVICE).float()
            x = x.to(DEVICE).float()
            bsz = x.size(0)
            x_flat = x.view(bsz, -1)
            recon, mu, logvar = vae(feat)
            recon_flat = recon.view(x.size(0), -1)
            bce = F.binary_cross_entropy(recon_flat, x_flat, reduction="sum")
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = (bce + kld) / max(1, bsz)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(loss.item()) * bsz
            n_seen += bsz

        # Validation loss for convergence-based stopping.
        vae.eval()
        val_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for feat, x in val_loader:
                feat = feat.to(DEVICE).float()
                x = x.to(DEVICE).float()
                bsz = x.size(0)
                x_flat = x.view(bsz, -1)
                recon, mu, logvar = vae(feat)
                recon_flat = recon.view(bsz, -1)
                bce = F.binary_cross_entropy(recon_flat, x_flat, reduction="sum")
                kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = (bce + kld) / max(1, bsz)
                val_sum += float(loss.item()) * bsz
                val_n += bsz

        tr_loss = running / max(1, n_seen)
        val_loss = val_sum / max(1, val_n)
        print(f"[VAE] epoch={epoch}/{args.vae_epochs} train={tr_loss:.4f} val={val_loss:.4f}")

        if val_loss < (best_val - float(args.vae_min_delta)):
            best_val = val_loss
            best_state = copy.deepcopy(vae.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= int(args.vae_patience):
                print(
                    f"[VAE] Early stopping at epoch {epoch} "
                    f"(best_val={best_val:.4f}, patience={args.vae_patience})"
                )
                break

    vae.load_state_dict(best_state, strict=True)

    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(
        {
            "model_kind": "feature_to_image_vae",
            "state_dict": vae.state_dict(),
            "feature_dim": feature_dim,
            "latent_dim": latent_dim,
            "target_angle": args.target_angle,
            "train_scope": "source_target",
            "decoder_arch": "model.VAE.decoder",
            "best_val_loss": float(best_val),
        },
        ckpt_path,
    )
    print(f"[VAE] Saved checkpoint: {ckpt_path}")
    vae.eval()
    return vae


@torch.no_grad()
def decode_features(vae: FeatureToImageVAE, features: torch.Tensor) -> torch.Tensor:
    feat = features.to(DEVICE).float()
    imgs = vae.decode_from_features(feat, deterministic=True).detach().cpu()
    return imgs.clamp(0.0, 1.0)


def _collect_step_images(
    vae: FeatureToImageVAE,
    methods: Sequence[str],
    method_domains: Dict[str, List[Dataset]],
    step: int,
    n_samples: int,
    seed: int,
) -> Tuple[Dict[str, torch.Tensor], np.ndarray]:
    lengths = []
    for m in methods:
        ds = method_domains[m][step]
        lengths.append(len(ds.data) if hasattr(ds, "data") else 0)
    min_len = min(lengths) if lengths else 0
    if min_len <= 0:
        raise ValueError(f"No data available for step {step}.")

    n = min(n_samples, min_len)
    rng = np.random.default_rng(seed + 1000 + step)
    idx = np.sort(rng.choice(min_len, size=n, replace=False))

    out: Dict[str, torch.Tensor] = {}
    for m in methods:
        ds = method_domains[m][step]
        f = ds.data[idx] if torch.is_tensor(ds.data) else torch.as_tensor(ds.data[idx])
        out[m] = decode_features(vae, f)
    return out, idx


def save_comparison_grid(
    save_path: str,
    methods: Sequence[str],
    decoded: Dict[str, torch.Tensor],
    step: int,
) -> None:
    rows = len(methods)
    cols = decoded[methods[0]].shape[0]
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.4))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if cols == 1:
        axes = np.expand_dims(axes, axis=1)

    for r, m in enumerate(methods):
        imgs = decoded[m]
        for c in range(cols):
            ax = axes[r, c]
            ax.imshow(imgs[c, 0], cmap="gray", vmin=0.0, vmax=1.0)
            ax.axis("off")
            if c == 0:
                ax.set_title(m, fontsize=10, loc="left")
    fig.suptitle(f"Decoded intermediates - step {step}", fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_summary_grid(
    save_path: str,
    methods: Sequence[str],
    method_domains: Dict[str, List[Dataset]],
    vae: FeatureToImageVAE,
    steps: Sequence[int],
) -> None:
    rows = len(methods)
    cols = len(steps)
    fig, axes = plt.subplots(rows, cols, figsize=(max(1, cols) * 1.4, rows * 1.5))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if cols == 1:
        axes = np.expand_dims(axes, axis=1)
    for r, m in enumerate(methods):
        for c, step in enumerate(steps):
            ds = method_domains[m][step]
            f = ds.data[0:1] if torch.is_tensor(ds.data) else torch.as_tensor(ds.data[0:1])
            img = decode_features(vae, f)[0, 0]
            ax = axes[r, c]
            ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
            ax.axis("off")
            if r == 0:
                ax.set_title(f"s{step}", fontsize=9)
            if c == 0:
                ax.set_ylabel(m, fontsize=9)
    fig.suptitle("Method summary (first decoded sample per step)", fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_endpoint_grid(
    save_path: str,
    src_ds: Dataset,
    tgt_ds: Dataset,
    n_samples: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed + 777)
    n_src = min(n_samples, len(src_ds))
    n_tgt = min(n_samples, len(tgt_ds))
    n = min(n_src, n_tgt)
    idx_src = np.sort(rng.choice(len(src_ds), size=n, replace=False))
    idx_tgt = np.sort(rng.choice(len(tgt_ds), size=n, replace=False))

    fig, axes = plt.subplots(2, n, figsize=(n * 1.2, 3))
    if n == 1:
        axes = np.expand_dims(axes, axis=1)
    for i in range(n):
        x_s, _ = src_ds[idx_src[i]]
        x_t, _ = tgt_ds[idx_tgt[i]]
        axes[0, i].imshow(torch.as_tensor(x_s).squeeze().cpu().numpy(), cmap="gray", vmin=0.0, vmax=1.0)
        axes[0, i].axis("off")
        axes[1, i].imshow(torch.as_tensor(x_t).squeeze().cpu().numpy(), cmap="gray", vmin=0.0, vmax=1.0)
        axes[1, i].axis("off")
    axes[0, 0].set_title("Source (0°)", loc="left", fontsize=10)
    axes[1, 0].set_title("Target", loc="left", fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _sample_common_indices_for_trajectories(
    trajectory_chains: Dict[str, Sequence[Dataset]],
    methods: Sequence[str],
    n_samples: int,
    seed: int,
) -> np.ndarray:
    """Pick one shared index set valid for all methods and all trajectory columns."""
    lengths: List[int] = []
    for m in methods:
        for ds in trajectory_chains[m]:
            lengths.append(len(ds.data))

    min_len = int(min(lengths)) if lengths else 0
    if min_len <= 0:
        raise RuntimeError("Cannot sample trajectory indices: no common valid length.")

    n = min(n_samples, min_len)
    rng = np.random.default_rng(seed + 4242)
    idx = np.sort(rng.choice(min_len, size=n, replace=False))
    return idx


def build_labeled_trajectory_chain(
    encoded_domains: Sequence[Dataset],
    method_synth_domains: Sequence[Dataset],
    *,
    generated_domains: int,
) -> Tuple[List[Dataset], List[str], List[bool]]:
    """
    Build trajectory columns with explicit Real/Synthetic structure:
      R0(source), S, ..., S, R1, S, ..., R2, ..., RT(target)
    """
    if generated_domains <= 0:
        raise ValueError("generated_domains must be > 0.")
    n_pairs = len(encoded_domains) - 1
    expected_synth = n_pairs * generated_domains
    if len(method_synth_domains) < expected_synth:
        raise RuntimeError(
            f"Not enough synthetic domains: expected at least {expected_synth}, got {len(method_synth_domains)}"
        )

    chain: List[Dataset] = [encoded_domains[0]]
    labels: List[str] = ["R0 (Source)"]
    is_synth: List[bool] = [False]

    cursor = 0
    for pair_idx in range(n_pairs):
        for j in range(generated_domains):
            chain.append(method_synth_domains[cursor])
            labels.append(f"S{pair_idx+1}.{j+1} (Synthetic)")
            is_synth.append(True)
            cursor += 1

        # Real endpoint of this pair: intermediate or final target
        real_domain_idx = pair_idx + 1
        is_target = real_domain_idx == len(encoded_domains) - 1
        if is_target:
            labels.append(f"R{real_domain_idx} (Target)")
        else:
            labels.append(f"R{real_domain_idx} (Real)")
        chain.append(encoded_domains[real_domain_idx])
        is_synth.append(False)

    return chain, labels, is_synth


def save_method_trajectory_grid(
    save_path: str,
    method_name: str,
    trajectory_domains: Sequence[Dataset],
    trajectory_labels: Sequence[str],
    vae: FeatureToImageVAE,
    indices: np.ndarray,
) -> None:
    """Rows are fixed sample indices; columns are source->...->target trajectory."""
    n_rows = len(indices)
    n_cols = len(trajectory_domains)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.2, n_rows * 1.2))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if n_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    for c, ds in enumerate(trajectory_domains):
        feat = ds.data if torch.is_tensor(ds.data) else torch.as_tensor(ds.data)
        decoded = decode_features(vae, feat[indices])
        for r in range(n_rows):
            ax = axes[r, c]
            ax.imshow(decoded[r, 0], cmap="gray", vmin=0.0, vmax=1.0)
            ax.axis("off")
            if r == 0:
                ax.set_title(trajectory_labels[c], fontsize=8)
            if c == 0:
                ax.set_ylabel(f"idx={int(indices[r])}", fontsize=8)

    fig.suptitle(f"{method_name}: fixed-index trajectory", fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_all_methods_class_trajectory_grid(
    save_path: str,
    methods: Sequence[str],
    display_names: Dict[str, str],
    trajectory_chains: Dict[str, Sequence[Dataset]],
    trajectory_labels: Sequence[str],
    trajectory_is_synth: Sequence[bool],
    vae: FeatureToImageVAE,
    indices: np.ndarray,
    class_id: int,
) -> None:
    if len(indices) == 0:
        return
    n_methods = len(methods)
    n_rows = n_methods
    n_cols = len(trajectory_labels)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.35, n_rows * 1.85), squeeze=False)

    def _to_strip(decoded: torch.Tensor) -> np.ndarray:
        # decoded: (N,1,28,28) -> strip (28, N*28)
        arr = decoded[:, 0].cpu().numpy()
        return np.concatenate([arr[i] for i in range(arr.shape[0])], axis=1)

    for m_idx, m in enumerate(methods):
        chain = trajectory_chains[m]
        for c in range(n_cols):
            ds = chain[c]
            feat = ds.data if torch.is_tensor(ds.data) else torch.as_tensor(ds.data)
            decoded = decode_features(vae, feat[indices])
            strip = _to_strip(decoded)
            ax = axes[m_idx, c]
            ax.imshow(strip, cmap="gray", vmin=0.0, vmax=1.0, aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.8)
                spine.set_color("#202020")
            # Sample separators inside each strip (every 28 px).
            for k in range(1, len(indices)):
                ax.axvline(28 * k - 0.5, color="#f3f3f3", lw=0.8, alpha=0.9)
            if m_idx == 0:
                title_fc = "#fdebd0" if trajectory_is_synth[c] else "#e8f1fb"
                ax.set_title(
                    trajectory_labels[c],
                    fontsize=10,
                    fontweight="bold",
                    pad=8,
                    bbox=dict(boxstyle="round,pad=0.22", fc=title_fc, ec="#777777", lw=0.8),
                )
            if c == 0:
                ax.set_ylabel(
                    display_names[m],
                    fontsize=11,
                    fontweight="bold",
                    rotation=0,
                    labelpad=38,
                    va="center",
                    ha="right",
                    bbox=dict(boxstyle="round,pad=0.28", fc="#f5f5f5", ec="#999999", lw=0.9),
                )

    fig.suptitle(
        f"Class {class_id} Trajectories Across Methods",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.01,
        f"Shared sample indices (same for all methods/columns): {indices.tolist()}",
        ha="center",
        va="bottom",
        fontsize=10,
    )
    fig.tight_layout(rect=(0.02, 0.04, 1.0, 0.96))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=400, bbox_inches="tight")
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def _resolve_output_dir(args: argparse.Namespace) -> str:
    if args.output_dir:
        return args.output_dir
    return os.path.join(
        "analysis_outputs",
        f"qualitative_mnist_target{args.target_angle}",
        f"seed{args.seed}",
    )


def _collect_images_in_order(dataset: Dataset, batch_size: int, num_workers: int) -> torch.Tensor:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    imgs = []
    for x, _ in loader:
        imgs.append(x.detach().cpu().float())
    if not imgs:
        raise RuntimeError("Failed to collect images from dataset.")
    return torch.cat(imgs, dim=0)


def run(args: argparse.Namespace) -> None:
    args = _ensure_default_args(args)
    set_all_seeds(args.seed)

    if args.generated_domains <= 0:
        raise ValueError(
            "generated_domains=0 is unsupported for this script. "
            "Qualitative intermediate comparison requires generated domains."
        )

    output_dir = _resolve_output_dir(args)
    os.makedirs(output_dir, exist_ok=True)

    # Keep a dedicated args object for qualitative artifacts so we never touch
    # the main experiment model/cache/plot paths.
    args_main = copy.deepcopy(args)
    args_main.dataset = "mnist"

    src_trainset = get_single_rotate(False, 0)
    tgt_trainset = get_single_rotate(False, args.target_angle)
    all_sets, deg_idx = _build_rotated_domains(tgt_trainset, args.target_angle, args.gt_domains)

    artifact_tag = f"mnist_qualvae_target{args.target_angle}"
    model_dir = os.path.join(output_dir, "models")
    cache_dir = os.path.join(output_dir, "encoded_cache")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    model_cfg = ModelConfig(
        encoder_builder=ENCODER,
        mode="mnist",
        n_class=10,
        epochs=10,
        model_path=os.path.join(
            model_dir,
            f"src0_tgt{args.target_angle}_ssl{args.ssl_weight}_dim{args.small_dim}.pth",
        ),
        compress=True,
        in_dim=25088,
        out_dim=args.small_dim,
    )

    ref_model, ref_encoder = build_reference_model(args_main, model_cfg, src_trainset, tgt_trainset)
    _, _, _ = encode_real_domains(
        args_main,
        ref_encoder=ref_encoder,
        src_trainset=src_trainset,
        tgt_trainset=tgt_trainset,
        all_sets=all_sets,
        deg_idx=deg_idx,
        cache_dir=cache_dir,
        target_label=args.target_angle,
        force_recompute=False,
    )

    raw_domains = [src_trainset] + all_sets
    angle_keys = [0] + deg_idx
    encoded_domains = load_encoded_domains(cache_dir, angle_keys)

    latent_dim = args.vae_latent_dim if args.vae_latent_dim is not None else min(512, args.small_dim)
    source_features = (
        encoded_domains[0].data
        if torch.is_tensor(encoded_domains[0].data)
        else torch.as_tensor(encoded_domains[0].data)
    )
    target_features = (
        encoded_domains[-1].data
        if torch.is_tensor(encoded_domains[-1].data)
        else torch.as_tensor(encoded_domains[-1].data)
    )
    src_images = _collect_images_in_order(src_trainset, batch_size=args.batch_size, num_workers=args.num_workers)
    tgt_images = _collect_images_in_order(tgt_trainset, batch_size=args.batch_size, num_workers=args.num_workers)
    vae_images = torch.cat([src_images, tgt_images], dim=0)
    vae_features = torch.cat([source_features.detach().cpu().float(), target_features.detach().cpu().float()], dim=0)
    vae_trainset = TensorDataset(vae_images, torch.zeros(len(vae_images), dtype=torch.long))

    _ = train_or_load_vae(args, vae_trainset, vae_features, latent_dim, output_dir)
    if args.vae_only:
        print("[Done] VAE-only mode: reconstruction model trained/loaded. Skipping method runs and trajectory plots.")
        print(f"[Done] Outputs saved under: {output_dir}")
        return

    teacher = copy.deepcopy(ref_model).to(DEVICE).eval()
    _em_bundles, _em_accs, _ = build_em_bundles_for_chain(
        args_main,
        raw_domains=raw_domains,
        encoded_domains=encoded_domains,
        domain_keys=angle_keys,
        teacher_model=teacher,
        n_classes=10,
        description_prefix="qual-mnist-angle-",
    )

    print("[Run] Running standard methods through run_core_methods for consistency...")
    # Method internals write plots/caches under args.dataset; switch to an
    # isolated namespace to avoid overwriting main experiment artifacts.
    method_args = copy.deepcopy(args_main)
    method_args.dataset = artifact_tag
    results = run_core_methods(
        method_args,
        ref_model=ref_model,
        src_trainset=src_trainset,
        tgt_trainset=tgt_trainset,
        all_sets=all_sets,
        deg_idx=deg_idx,
        generated_domains=args.generated_domains,
        target_label=args.target_angle,
    )
    for k in ("goat", "goat_classwise", "ours_fr", "ours_eta"):
        if k in results and results[k].test_curve:
            print(f"[Run] {k}: final test={results[k].test_curve[-1]:.2f}")

    method_domains = extract_method_domains(method_args, encoded_domains, args.generated_domains)
    n_steps_by_method = {k: len(v) for k, v in method_domains.items()}
    print(f"[Run] Extracted synthetic steps per method: {n_steps_by_method}")
    min_steps = min(n_steps_by_method.values())
    if args.max_steps is not None:
        min_steps = min(min_steps, args.max_steps)
    if min_steps <= 0:
        raise RuntimeError("No synthetic intermediate steps were produced.")
    vae = train_or_load_vae(args, vae_trainset, vae_features, latent_dim, output_dir)

    save_endpoint_grid(
        os.path.join(output_dir, "endpoints_source_target.png"),
        src_trainset,
        tgt_trainset,
        n_samples=args.samples_per_step,
        seed=args.seed,
    )

    group_a = ("GOAT", "GOATCW")
    map_a = {"GOAT": "goat", "GOATCW": "goat_classwise"}
    for step in range(min_steps):
        decoded, idx = _collect_step_images(
            vae,
            methods=[map_a[m] for m in group_a],
            method_domains=method_domains,
            step=step,
            n_samples=args.samples_per_step,
            seed=args.seed,
        )
        renamed = {m: decoded[map_a[m]] for m in group_a}
        save_comparison_grid(
            os.path.join(output_dir, f"compare_goat_vs_goatcw_step{step}.png"),
            methods=group_a,
            decoded=renamed,
            step=step,
        )
        print(f"[Save] Group A step {step}, sampled idx={idx.tolist()}")

    summary_steps = list(range(min_steps))
    save_summary_grid(
        os.path.join(output_dir, "compare_goat_vs_goatcw_summary.png"),
        methods=group_a,
        method_domains={k: method_domains[v] for k, v in map_a.items()},
        vae=vae,
        steps=summary_steps,
    )

    group_b = ("GOATCW", "OURS-FR", "OURS-Nat")
    map_b = {"GOATCW": "goat_classwise", "OURS-FR": "ours_fr", "OURS-Nat": "ours_eta"}
    for step in range(min_steps):
        decoded, idx = _collect_step_images(
            vae,
            methods=[map_b[m] for m in group_b],
            method_domains=method_domains,
            step=step,
            n_samples=args.samples_per_step,
            seed=args.seed,
        )
        renamed = {m: decoded[map_b[m]] for m in group_b}
        save_comparison_grid(
            os.path.join(output_dir, f"compare_goatcw_oursfr_oursnat_step{step}.png"),
            methods=group_b,
            decoded=renamed,
            step=step,
        )
        print(f"[Save] Group B step {step}, sampled idx={idx.tolist()}")

    save_summary_grid(
        os.path.join(output_dir, "compare_goatcw_oursfr_oursnat_summary.png"),
        methods=group_b,
        method_domains={k: method_domains[v] for k, v in map_b.items()},
        vae=vae,
        steps=summary_steps,
    )

    # Fixed-index trajectories: same indices across all methods/domains.
    trajectory_methods = ("goat", "goat_classwise", "ours_fr", "ours_eta")
    trajectory_chains: Dict[str, List[Dataset]] = {}
    trajectory_labels: Optional[List[str]] = None
    trajectory_is_synth: Optional[List[bool]] = None
    for m in trajectory_methods:
        chain, labels, is_synth = build_labeled_trajectory_chain(
            encoded_domains=encoded_domains,
            method_synth_domains=method_domains[m][:min_steps],
            generated_domains=args.generated_domains,
        )
        trajectory_chains[m] = chain
        if trajectory_labels is None:
            trajectory_labels = labels
            trajectory_is_synth = is_synth

    assert trajectory_labels is not None and trajectory_is_synth is not None

    shared_idx = _sample_common_indices_for_trajectories(
        trajectory_chains=trajectory_chains,
        methods=trajectory_methods,
        n_samples=args.samples_per_step,
        seed=args.seed,
    )
    np.save(os.path.join(output_dir, "trajectory_indices.npy"), shared_idx)
    print(f"[Trajectory] Shared indices: {shared_idx.tolist()}")

    display_names = {
        "goat": "GOAT",
        "goat_classwise": "GOATCW",
        "ours_fr": "OURS-FR",
        "ours_eta": "OURS-Nat",
    }
    for m in trajectory_methods:
        save_method_trajectory_grid(
            os.path.join(output_dir, f"trajectory_{m}.png"),
            method_name=display_names[m],
            trajectory_domains=trajectory_chains[m],
            trajectory_labels=trajectory_labels,
            vae=vae,
            indices=shared_idx,
        )

    # Combined figure per class: all methods in the same grid.
    n_classes = int(torch.as_tensor(encoded_domains[0].targets).max().item()) + 1
    for cls in range(n_classes):
        class_idx = _common_indices_for_class(
            trajectory_chains=trajectory_chains,
            methods=trajectory_methods,
            cls=cls,
            n_samples=1,
            seed=args.seed,
        )
        if len(class_idx) == 0:
            print(f"[Trajectory] class {cls}: skipped (no common indices across all methods/steps).")
            continue
        save_all_methods_class_trajectory_grid(
            save_path=os.path.join(output_dir, f"trajectory_all_methods_class{cls}.png"),
            methods=trajectory_methods,
            display_names=display_names,
            trajectory_chains=trajectory_chains,
            trajectory_labels=trajectory_labels,
            trajectory_is_synth=trajectory_is_synth,
            vae=vae,
            indices=class_idx,
            class_id=cls,
        )
        print(f"[Trajectory] class {cls}: saved with indices {class_idx.tolist()}")

    print(f"[Done] Outputs saved under: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qualitative VAE decoding comparison for Rotated-MNIST intermediates."
    )
    parser.add_argument("--target-angle", type=int, default=60)
    parser.add_argument("--gt-domains", type=int, default=3)
    parser.add_argument("--generated-domains", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--small-dim", type=int, default=2048)
    parser.add_argument("--ssl-weight", type=float, default=0.1)
    parser.add_argument("--label-source", choices=["pseudo", "em"], default="pseudo")
    parser.add_argument("--em-match", choices=["pseudo", "prototypes", "none"], default="pseudo")
    parser.add_argument("--em-select", choices=["bic", "cost", "ll"], default="bic")
    parser.add_argument("--em-ensemble", action="store_true")
    parser.add_argument("--vae-latent-dim", type=int, default=None)
    parser.add_argument("--vae-epochs", type=int, default=100)
    parser.add_argument("--vae-lr", type=float, default=1e-3)
    parser.add_argument("--vae-val-frac", type=float, default=0.1, help="Validation fraction for VAE early stopping.")
    parser.add_argument("--vae-patience", type=int, default=12, help="Early stopping patience on VAE validation loss.")
    parser.add_argument("--vae-min-delta", type=float, default=1e-3, help="Minimum validation-loss improvement to reset patience.")
    parser.add_argument("--vae-path", type=str, default="")
    parser.add_argument("--samples-per-step", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument(
        "--vae-only",
        action="store_true",
        help="Train/load the feature-to-image VAE only, then exit before EM/method trajectory generation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
