#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import colors as mcolors
from matplotlib.patches import Rectangle
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
from model import ENCODER
from ot_util import generate_domains
from util import get_single_rotate
from a_star_util import (
    generate_fr_domains_between_optimized,
    generate_natural_domains_between,
)
from experiment_new import set_all_seeds


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RMNISTDecoder(nn.Module):
    """Decoder architecture matching ../RMNIST/model.py VAE.decoder."""

    def __init__(self, z_dim: int, x_dim: int = 28 * 28) -> None:
        super().__init__()
        self.z_dim = int(z_dim)
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 64, 3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, 3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, 3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 8, 3),
            nn.ReLU(),
            nn.BatchNorm2d(8),
        )
        # 8 x 9 x 9 = 648 after the transposed-conv stack above.
        self.fc4 = nn.Linear(648, int(x_dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = z.reshape(-1, self.z_dim, 1, 1)
        z = self.decode(z)
        z = z.reshape(-1, 648)
        return torch.sigmoid(self.fc4(z))


class RMNISTVAE(nn.Module):
    """VAE architecture matching ../RMNIST/model.py VAE."""

    def __init__(self, x_dim: int, z_dim: int) -> None:
        super().__init__()
        self.z_dim = int(z_dim)
        self.encode = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Flatten(),
            nn.Linear(576, 128),
            nn.ReLU(),
        )
        self.fc1 = nn.Linear(128, z_dim)
        self.fc2 = nn.Linear(128, z_dim)
        self.decoder = RMNISTDecoder(z_dim=z_dim, x_dim=x_dim)

    def encoder(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encode(x)
        return self.fc1(h), self.fc2(h)

    def sampling(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encoder(x)
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var


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
        self.image_dec = RMNISTDecoder(z_dim=latent_dim, x_dim=image_dim)
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


@dataclass
class ScreenshotStyle:
    source_panel: str = "#f5f5f5"
    target_panel: str = "#f5f5f5"
    mid_panel: str = "#f7f7f7"
    shade_groups: bool = True
    tile_border: str = "#1f1f1f"
    tile_border_lw: float = 1.2
    header_fontsize: int = 14
    header_weight: str = "normal"
    header_fontfamily: str = "serif"
    header_min_fontsize: int = 9
    header_max_fontsize: int = 20
    header_autofit: bool = True
    title_fontsize: int = 15
    title_weight: str = "normal"
    title_fontfamily: str = "serif"
    intermediate_gradient: bool = False
    intermediate_gradient_gamma: float = 1.0
    intermediate_gradient_steps: int = 400
    dpi: int = 300


def _to_numpy_images(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if torch.is_tensor(x):
        x = x.detach().cpu().float().numpy()
    x = np.asarray(x)
    if x.ndim == 4:
        x = x[:, 0]
    if x.ndim != 3:
        raise ValueError(f"Expected (N,1,H,W) or (N,H,W), got {x.shape}")
    return np.clip(x, 0.0, 1.0)


def save_screenshot_style_grid(
    save_path: str,
    source_imgs: torch.Tensor | np.ndarray,
    inter_imgs_list: Sequence[torch.Tensor | np.ndarray],
    target_imgs: torch.Tensor | np.ndarray,
    *,
    headers: Tuple[str, str, str] = ("Source", "Generated Intermediate", "Target"),
    plot_title: str = "",
    style: ScreenshotStyle = ScreenshotStyle(),
    tile_size_in: float = 0.40,
    h_gap_in: float = 0.07,
    v_gap_in: float = 0.07,
    group_gap_in: float = 0.20,
    top_header_in: float = 0.38,
    pad_left_in: float = 0.16,
    pad_right_in: float = 0.16,
    pad_bottom_in: float = 0.14,
    tint_intermediate: bool = False,
    save_pdf_copy: bool = True,
) -> None:
    src = _to_numpy_images(source_imgs)
    tgt = _to_numpy_images(target_imgs)
    inters = [_to_numpy_images(z) for z in inter_imgs_list]
    n_rows = src.shape[0]
    if tgt.shape[0] != n_rows:
        raise ValueError(f"source N={n_rows}, target N={tgt.shape[0]}")
    for k, arr in enumerate(inters):
        if arr.shape[0] != n_rows:
            raise ValueError(f"inter[{k}] N={arr.shape[0]} != {n_rows}")

    n_inter = len(inters)
    n_cols_total = 1 + n_inter + 1
    width_in = (
        pad_left_in
        + pad_right_in
        + n_cols_total * tile_size_in
        + (n_cols_total - 1) * h_gap_in
        + 2 * group_gap_in
    )
    height_in = (
        pad_bottom_in
        + top_header_in
        + n_rows * tile_size_in
        + (n_rows - 1) * v_gap_in
    )

    fig = plt.figure(figsize=(width_in, height_in), dpi=style.dpi)
    fig.patch.set_facecolor("white")

    def xin(v: float) -> float:
        return v / width_in

    def yin(v: float) -> float:
        return v / height_in

    x0 = pad_left_in
    src_x0 = x0
    src_x1 = src_x0 + tile_size_in
    inter_x0 = src_x1 + h_gap_in + group_gap_in
    inter_w = n_inter * tile_size_in + max(0, n_inter - 1) * h_gap_in
    inter_x1 = inter_x0 + inter_w
    tgt_x0 = inter_x1 + h_gap_in + group_gap_in
    tgt_x1 = tgt_x0 + tile_size_in

    tiles_y0 = pad_bottom_in
    tiles_y1 = tiles_y0 + (n_rows * tile_size_in + (n_rows - 1) * v_gap_in)

    ax_bg = fig.add_axes([0, 0, 1, 1], zorder=0)
    ax_bg.axis("off")

    def add_panel(xl: float, xr: float, color: str) -> None:
        ax_bg.add_patch(
            Rectangle(
                (xin(xl - 0.08), yin(tiles_y0 - 0.08)),
                xin((xr - xl) + 0.16),
                yin((tiles_y1 - tiles_y0) + 0.16),
                facecolor=color,
                edgecolor="none",
                zorder=0,
            )
        )

    def add_gradient_panel(xl: float, xr: float, left_color: str, right_color: str) -> None:
        w = max(32, int(style.intermediate_gradient_steps))
        gamma = max(1e-3, float(style.intermediate_gradient_gamma))
        c0 = np.asarray(mcolors.to_rgb(left_color), dtype=np.float32)[None, None, :]
        c1 = np.asarray(mcolors.to_rgb(right_color), dtype=np.float32)[None, None, :]
        t = np.linspace(0.0, 1.0, w, dtype=np.float32) ** gamma
        t = t[None, :, None]
        grad = (1.0 - t) * c0 + t * c1  # (1, W, 3)
        ax_bg.imshow(
            grad,
            extent=(
                xin(xl - 0.08),
                xin(xr + 0.08),
                yin(tiles_y0 - 0.08),
                yin(tiles_y1 + 0.08),
            ),
            origin="lower",
            aspect="auto",
            zorder=0,
        )

    if style.shade_groups:
        add_panel(src_x0, src_x1, style.source_panel)
        if tint_intermediate and n_inter > 0:
            if style.intermediate_gradient:
                add_gradient_panel(inter_x0, inter_x1, style.source_panel, style.target_panel)
            else:
                add_panel(inter_x0, inter_x1, style.mid_panel)
        add_panel(tgt_x0, tgt_x1, style.target_panel)

    # Header boxes (one per group) prevent cross-group text overlap.
    hy0 = tiles_y1 + 0.03
    hh = max(0.16, top_header_in - 0.05)

    def _fit_fs(text: str, width_inches: float) -> float:
        if not style.header_autofit:
            return float(style.header_fontsize)
        chars = max(1, len(text))
        # Heuristic: average glyph width ~0.62 * fontsize points.
        # Convert inches to points (72/in) and reserve small padding.
        max_pts = max(1.0, (width_inches * 72.0 - 10.0) / (0.62 * chars))
        fs = min(float(style.header_fontsize), max_pts, float(style.header_max_fontsize))
        return max(float(style.header_min_fontsize), fs)

    # pick one common fontsize based on smallest group width
    group_widths = [src_x1 - src_x0, inter_x1 - inter_x0, tgt_x1 - tgt_x0]
    min_group_w = min(group_widths)
    common_fs = _fit_fs(max(headers, key=len), min_group_w) if style.header_autofit else float(style.header_fontsize)

    def _header(xl: float, xr: float, text: str) -> None:
        ax_h = fig.add_axes([xin(xl), yin(hy0), xin(max(1e-6, xr - xl)), yin(hh)], zorder=3)
        ax_h.axis("off")
        ax_h.text(
            0.5, 0.08, text,
            transform=ax_h.transAxes,
            ha="center", va="bottom",
            fontsize=common_fs,
            fontweight=style.header_weight,
            fontfamily=style.header_fontfamily,
            clip_on=False,   # <- prevents cutting
        )

    _header(src_x0, src_x1, headers[0])
    _header(inter_x0, inter_x1, headers[1])
    _header(tgt_x0, tgt_x1, headers[2])
    if str(plot_title).strip():
        fig.suptitle(
            str(plot_title),
            fontsize=style.title_fontsize,
            fontweight=style.title_weight,
            fontfamily=style.title_fontfamily,
            y=0.995,
        )

    col_xs: List[float] = [src_x0]
    for j in range(n_inter):
        col_xs.append(inter_x0 + j * (tile_size_in + h_gap_in))
    col_xs.append(tgt_x0)

    for r in range(n_rows):
        y_in = tiles_y1 - tile_size_in - r * (tile_size_in + v_gap_in)
        for c in range(n_cols_total):
            x_in = col_xs[c]
            ax = fig.add_axes([xin(x_in), yin(y_in), xin(tile_size_in), yin(tile_size_in)], zorder=2)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(style.tile_border_lw)
                spine.set_edgecolor(style.tile_border)
            if c == 0:
                img = src[r]
            elif c == n_cols_total - 1:
                img = tgt[r]
            else:
                img = inters[c - 1][r]
            ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0.02)
    if save_pdf_copy:
        pdf_path = os.path.splitext(save_path)[0] + ".pdf"
        fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


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


def _domain_weight_tensor(ds: Dataset) -> torch.Tensor:
    if hasattr(ds, "weight") and ds.weight is not None:
        return torch.as_tensor(ds.weight).float().view(-1)
    n = len(ds.data) if hasattr(ds, "data") else len(ds)
    return torch.ones(int(n), dtype=torch.float32)


def _subset_domain_by_class(
    ds: Dataset,
    cls: int,
    use_em: bool,
    min_count: int = 1,
    confidence_quantile: float = 0.0,
) -> Tuple[Optional[DomainDataset], Dict[str, object]]:
    y = _get_label_tensor(ds, use_em=use_em)
    if not hasattr(ds, "data"):
        return None, {"status": "missing_data", "count": 0}
    x = ds.data if torch.is_tensor(ds.data) else torch.as_tensor(ds.data)
    w = _domain_weight_tensor(ds)
    mask = y == int(cls)
    n_before = int(mask.sum().item())
    if n_before == 0:
        return None, {"status": "no_class_data", "count": 0}

    idx = torch.where(mask)[0]
    if float(confidence_quantile) > 0.0:
        q = float(np.clip(confidence_quantile, 0.0, 0.99))
        w_cls = w[idx]
        thr = float(torch.quantile(w_cls, q).item())
        idx = idx[w_cls >= thr]

    n_after = int(idx.numel())
    if n_after < int(min_count):
        return None, {"status": "below_min_count", "count": n_after, "count_before": n_before}

    x_c = x[idx]
    y_c = y[idx]
    w_c = w[idx].float().cpu()
    return (
        DomainDataset(x_c, w_c, y_c, y_c),
        {"status": "ok", "count": n_after, "count_before": n_before},
    )


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
) -> Tuple[Dict[str, List[Dataset]], List[Dict[str, object]]]:
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
    classwise_quality_report: List[Dict[str, object]] = []

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
            src_c, src_info = _subset_domain_by_class(
                left,
                c,
                use_em=(i > 0),
                min_count=args.cw_min_class_count,
                confidence_quantile=args.cw_confidence_quantile,
            )
            tgt_c, tgt_info = _subset_domain_by_class(
                right,
                c,
                use_em=True,
                min_count=args.cw_min_class_count,
                confidence_quantile=args.cw_confidence_quantile,
            )
            row = {
                "pair_idx": i,
                "class_id": c,
                "src_status": src_info.get("status"),
                "tgt_status": tgt_info.get("status"),
                "src_count": int(src_info.get("count", 0)),
                "tgt_count": int(tgt_info.get("count", 0)),
            }
            if src_c is None or tgt_c is None:
                row["used"] = 0
                classwise_quality_report.append(row)
                continue
            chain_c, _, _ = generate_domains(generated_domains, src_c, tgt_c)
            alpha = float(np.clip(args.cw_prototype_shrink, 0.0, 1.0))
            if alpha > 0.0 and len(chain_c) > 0:
                proto_l = src_c.data.float().mean(dim=0, keepdim=True).cpu()
                proto_r = tgt_c.data.float().mean(dim=0, keepdim=True).cpu()
                denom = max(1, len(chain_c) - 1)
                for step_idx, ds in enumerate(chain_c):
                    t = float(step_idx) / float(denom)
                    target_proto = (1.0 - t) * proto_l + t * proto_r
                    x = ds.data if torch.is_tensor(ds.data) else torch.as_tensor(ds.data)
                    x = x.float().cpu()
                    ds.data = ((1.0 - alpha) * x + alpha * target_proto).float().cpu()
            for ds in chain_c:
                labels = torch.full((len(ds.targets),), c, dtype=torch.long)
                ds.targets = labels.clone()
                ds.targets_em = labels.clone()
            class_chains.append(chain_c)
            row["used"] = 1
            row["proto_shrink"] = alpha
            classwise_quality_report.append(row)
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

    return methods, classwise_quality_report


def _labels_for_domain(ds: Dataset) -> torch.Tensor:
    if hasattr(ds, "targets") and ds.targets is not None:
        return torch.as_tensor(ds.targets).long().view(-1)
    if hasattr(ds, "targets_em") and ds.targets_em is not None:
        return torch.as_tensor(ds.targets_em).long().view(-1)
    return torch.full((len(ds.data),), -1, dtype=torch.long)


def _pick_indices(
    candidates: np.ndarray,
    n: int,
    seed: int,
    strategy: str = "random",
    scores: Optional[np.ndarray] = None,
) -> np.ndarray:
    candidates = np.asarray(candidates, dtype=np.int64)
    if len(candidates) == 0 or n <= 0:
        return np.array([], dtype=np.int64)
    n = min(int(n), len(candidates))
    if strategy == "confidence" and scores is not None:
        scores = np.asarray(scores, dtype=np.float64)
        order = np.lexsort((candidates, -scores))  # high score first, stable on index
        picked = candidates[order[:n]]
        return np.sort(picked.astype(np.int64))
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(candidates, size=n, replace=False).astype(np.int64))


def _common_indices_for_class(
    trajectory_chains: Dict[str, Sequence[Dataset]],
    methods: Sequence[str],
    cls: int,
    n_samples: int,
    seed: int,
    sample_selection: str = "random",
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

    score_arr = None
    if sample_selection == "confidence":
        score_arr = np.zeros(len(idx), dtype=np.float64)
        for ds in check_domains:
            w = _domain_weight_tensor(ds).cpu().numpy()[:min_len]
            score_arr += w[idx]
        score_arr /= max(1, len(check_domains))
    return _pick_indices(
        candidates=idx,
        n=n_samples,
        seed=seed + 9000 + int(cls),
        strategy=sample_selection,
        scores=score_arr,
    )


def _expand_indices(
    idx: np.ndarray,
    n_samples: int,
    seed: int,
) -> np.ndarray:
    """Expand to n_samples with replacement when needed (if idx is non-empty)."""
    idx = np.asarray(idx, dtype=np.int64)
    if len(idx) == 0:
        return idx
    if len(idx) >= int(n_samples):
        return idx[: int(n_samples)]
    rng = np.random.default_rng(seed)
    extra = rng.choice(idx, size=int(n_samples - len(idx)), replace=True).astype(np.int64)
    return np.concatenate([idx, extra], axis=0)


def _per_method_indices_for_class(
    trajectory_chains: Dict[str, Sequence[Dataset]],
    methods: Sequence[str],
    cls: int,
    n_samples: int,
    seed: int,
    sample_selection: str = "random",
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for m in methods:
        chain = trajectory_chains[m]
        min_len = min(len(ds.data) for ds in chain) if chain else 0
        if min_len <= 0:
            out[m] = np.array([], dtype=np.int64)
            continue
        mask = np.ones(min_len, dtype=bool)
        for ds in chain:
            y = _labels_for_domain(ds).cpu().numpy()[:min_len]
            mask &= (y == int(cls))
        idx = np.where(mask)[0]
        if len(idx) == 0:
            out[m] = np.array([], dtype=np.int64)
            continue
        score_arr = None
        if sample_selection == "confidence":
            score_arr = np.zeros(len(idx), dtype=np.float64)
            for ds in chain:
                w = _domain_weight_tensor(ds).cpu().numpy()[:min_len]
                score_arr += w[idx]
            score_arr /= max(1, len(chain))
        picked = _pick_indices(
            candidates=idx,
            n=min(int(n_samples), len(idx)),
            seed=seed + 9100 + int(cls) + (hash(m) % 1000),
            strategy=sample_selection,
            scores=score_arr,
        )
        if len(picked) < int(n_samples):
            rng = np.random.default_rng(seed + 9199 + int(cls) + (hash(m) % 1000))
            extra = rng.choice(picked, size=int(n_samples - len(picked)), replace=True).astype(np.int64)
            picked = np.concatenate([picked, extra], axis=0)
        out[m] = picked.astype(np.int64)
    return out


def _vae_checkpoint_path(args: argparse.Namespace, output_dir: str, latent_dim: int) -> str:
    if args.vae_path:
        return args.vae_path
    ssl_tag = str(getattr(args, "ssl_weight", 0.0)).replace(".", "p")
    return os.path.join(
        output_dir,
        f"{args.vae_input}_vae_target{args.target_angle}_z{latent_dim}_ssl{ssl_tag}_seed{args.seed}.pt",
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
        payload = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
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
            and decoder_arch_ckpt == "RMNISTDecoder"
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
            "decoder_arch": "RMNISTDecoder",
            "best_val_loss": float(best_val),
        },
        ckpt_path,
    )
    print(f"[VAE] Saved checkpoint: {ckpt_path}")
    vae.eval()
    return vae


def train_or_load_image_vae(
    args: argparse.Namespace,
    train_images: torch.Tensor,
    latent_dim: int,
    output_dir: str,
) -> RMNISTVAE:
    ckpt_path = _vae_checkpoint_path(args, output_dir, latent_dim)
    vae = RMNISTVAE(x_dim=28 * 28, z_dim=latent_dim).to(DEVICE)

    if os.path.exists(ckpt_path):
        payload = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
        kind = payload.get("model_kind") if isinstance(payload, dict) else None
        latent_ckpt = payload.get("latent_dim") if isinstance(payload, dict) else None
        scope_ckpt = payload.get("train_scope") if isinstance(payload, dict) else None
        arch_ckpt = payload.get("arch") if isinstance(payload, dict) else None
        if (
            kind == "image_vae"
            and latent_ckpt == latent_dim
            and scope_ckpt == "source_target_images"
            and arch_ckpt == "RMNISTVAE"
        ):
            vae.load_state_dict(state, strict=True)
            vae.eval()
            print(f"[VAE] Loaded checkpoint: {ckpt_path}")
            return vae
        print("[VAE] Existing checkpoint is incompatible with image VAE; retraining.")

    imgs = train_images.detach().cpu().float()
    n_all = int(imgs.size(0))
    idx_all = np.arange(n_all)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(idx_all)
    n_val = int(round(n_all * float(args.vae_val_frac)))
    n_val = min(max(n_val, 1), max(1, n_all - 1))
    val_idx = idx_all[:n_val]
    tr_idx = idx_all[n_val:]

    tr_imgs = imgs[tr_idx]
    va_imgs = imgs[val_idx]
    train_pairs = TensorDataset(tr_imgs, torch.zeros(len(tr_imgs), dtype=torch.long))
    val_pairs = TensorDataset(va_imgs, torch.zeros(len(va_imgs), dtype=torch.long))

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

    opt = torch.optim.Adam(vae.parameters(), lr=args.vae_lr)
    best_val = float("inf")
    best_state = copy.deepcopy(vae.state_dict())
    bad_epochs = 0

    for epoch in range(1, args.vae_epochs + 1):
        vae.train()
        running = 0.0
        n_seen = 0
        for x, _ in train_loader:
            x = x.to(DEVICE).float()
            bsz = x.size(0)
            recon, mu, logvar = vae(x)
            recon_flat = recon.view(bsz, -1)
            x_flat = x.view(bsz, -1)
            bce = F.binary_cross_entropy(recon_flat, x_flat, reduction="sum")
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = (bce + kld) / max(1, bsz)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(loss.item()) * bsz
            n_seen += bsz

        vae.eval()
        val_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(DEVICE).float()
                bsz = x.size(0)
                recon, mu, logvar = vae(x)
                recon_flat = recon.view(bsz, -1)
                x_flat = x.view(bsz, -1)
                bce = F.binary_cross_entropy(recon_flat, x_flat, reduction="sum")
                kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = (bce + kld) / max(1, bsz)
                val_sum += float(loss.item()) * bsz
                val_n += bsz

        tr_loss = running / max(1, n_seen)
        val_loss = val_sum / max(1, val_n)
        print(f"[VAE-image] epoch={epoch}/{args.vae_epochs} train={tr_loss:.4f} val={val_loss:.4f}")

        if val_loss < (best_val - float(args.vae_min_delta)):
            best_val = val_loss
            best_state = copy.deepcopy(vae.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= int(args.vae_patience):
                print(
                    f"[VAE-image] Early stopping at epoch {epoch} "
                    f"(best_val={best_val:.4f}, patience={args.vae_patience})"
                )
                break

    vae.load_state_dict(best_state, strict=True)
    vae.eval()
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(
        {
            "model_kind": "image_vae",
            "state_dict": vae.state_dict(),
            "latent_dim": latent_dim,
            "target_angle": args.target_angle,
            "train_scope": "source_target_images",
            "arch": "RMNISTVAE",
            "best_val_loss": float(best_val),
        },
        ckpt_path,
    )
    print(f"[VAE] Saved checkpoint: {ckpt_path}")
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
    sample_selection: str = "random",
) -> Tuple[Dict[str, torch.Tensor], np.ndarray]:
    lengths = []
    for m in methods:
        ds = method_domains[m][step]
        lengths.append(len(ds.data) if hasattr(ds, "data") else 0)
    min_len = min(lengths) if lengths else 0
    if min_len <= 0:
        raise ValueError(f"No data available for step {step}.")

    candidates = np.arange(min_len, dtype=np.int64)
    score_arr = None
    if sample_selection == "confidence":
        score_arr = np.zeros(min_len, dtype=np.float64)
        for m in methods:
            ds = method_domains[m][step]
            w = _domain_weight_tensor(ds).cpu().numpy()[:min_len]
            score_arr += w
        score_arr /= max(1, len(methods))
    idx = _pick_indices(
        candidates=candidates,
        n=min(n_samples, min_len),
        seed=seed + 1000 + step,
        strategy=sample_selection,
        scores=score_arr,
    )

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
    sample_selection: str = "random",
) -> np.ndarray:
    """Pick one shared index set valid for all methods and all trajectory columns."""
    lengths: List[int] = []
    for m in methods:
        for ds in trajectory_chains[m]:
            lengths.append(len(ds.data))

    min_len = int(min(lengths)) if lengths else 0
    if min_len <= 0:
        raise RuntimeError("Cannot sample trajectory indices: no common valid length.")

    candidates = np.arange(min_len, dtype=np.int64)
    score_arr = None
    if sample_selection == "confidence":
        score_arr = np.zeros(min_len, dtype=np.float64)
        total = 0
        for m in methods:
            for ds in trajectory_chains[m]:
                w = _domain_weight_tensor(ds).cpu().numpy()[:min_len]
                score_arr += w
                total += 1
        score_arr /= max(1, total)
    return _pick_indices(
        candidates=candidates,
        n=min(n_samples, min_len),
        seed=seed + 4242,
        strategy=sample_selection,
        scores=score_arr,
    )


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
    per_method_indices: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    if len(indices) == 0 and not per_method_indices:
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
        idx_m = indices
        if per_method_indices is not None:
            idx_m = per_method_indices.get(m, np.array([], dtype=np.int64))
        if len(idx_m) == 0:
            for c in range(n_cols):
                ax = axes[m_idx, c]
                ax.set_xticks([])
                ax.set_yticks([])
                ax.text(0.5, 0.5, "no class sample", ha="center", va="center", fontsize=8)
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(0.8)
                    spine.set_color("#202020")
            continue
        for c in range(n_cols):
            ds = chain[c]
            feat = ds.data if torch.is_tensor(ds.data) else torch.as_tensor(ds.data)
            decoded = decode_features(vae, feat[idx_m])
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
            for k in range(1, len(idx_m)):
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
    if per_method_indices is None:
        fig.text(
            0.5,
            0.01,
            f"Shared sample indices (same for all methods/columns): {indices.tolist()}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    else:
        summary = ", ".join([f"{display_names[m]}={per_method_indices.get(m, np.array([],dtype=np.int64)).tolist()}" for m in methods])
        fig.text(
            0.5,
            0.01,
            f"Per-method class indices: {summary}",
            ha="center",
            va="bottom",
            fontsize=9,
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


def _save_classwise_quality_report(
    report_rows: Sequence[Dict[str, object]],
    save_path: str,
) -> None:
    if not report_rows:
        return
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    keys = sorted({k for r in report_rows for k in r.keys()})
    with open(save_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in report_rows:
            w.writerow(r)


@torch.no_grad()
def _evaluate_semantic_consistency(
    classifier: nn.Module,
    vae: FeatureToImageVAE,
    method_domains: Dict[str, List[Dataset]],
    max_steps: int,
    sample_selection: str,
    seed: int,
    max_samples_per_step: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    by_step_rows: List[Dict[str, object]] = []
    summary_acc: Dict[str, List[float]] = {k: [] for k in method_domains.keys()}

    classifier.eval()
    for method, domains in method_domains.items():
        n_steps = min(max_steps, len(domains))
        for step in range(n_steps):
            ds = domains[step]
            x = ds.data if torch.is_tensor(ds.data) else torch.as_tensor(ds.data)
            y = _labels_for_domain(ds).long().view(-1)
            n_all = min(len(x), len(y))
            if n_all <= 0:
                by_step_rows.append(
                    {"method": method, "step": step, "n": 0, "acc": float("nan"), "mean_conf": float("nan")}
                )
                continue

            n_take = min(int(max_samples_per_step), n_all)
            candidates = np.arange(n_all, dtype=np.int64)
            scores = None
            if sample_selection == "confidence":
                scores = _domain_weight_tensor(ds).cpu().numpy()[:n_all]
            idx = _pick_indices(
                candidates=candidates,
                n=n_take,
                seed=seed + 7700 + step + (hash(method) % 1000),
                strategy=sample_selection,
                scores=scores,
            )
            if len(idx) == 0:
                by_step_rows.append(
                    {"method": method, "step": step, "n": 0, "acc": float("nan"), "mean_conf": float("nan")}
                )
                continue

            feats = x[idx]
            y_true = y[idx].to(DEVICE)
            imgs = decode_features(vae, feats).to(DEVICE)
            logits = classifier(imgs)
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            acc = float((pred == y_true).float().mean().item())
            mean_conf = float(conf.mean().item())

            by_step_rows.append(
                {
                    "method": method,
                    "step": int(step),
                    "n": int(len(idx)),
                    "acc": acc,
                    "mean_conf": mean_conf,
                }
            )
            summary_acc[method].append(acc)

    summary_rows: List[Dict[str, object]] = []
    for method, vals in summary_acc.items():
        arr = np.asarray(vals, dtype=np.float64)
        summary_rows.append(
            {
                "method": method,
                "n_steps": int(len(vals)),
                "mean_acc": float(np.nanmean(arr)) if len(arr) else float("nan"),
                "std_acc": float(np.nanstd(arr)) if len(arr) else float("nan"),
            }
        )
    return by_step_rows, summary_rows


def _save_rows_csv(rows: Sequence[Dict[str, object]], save_path: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    keys = sorted({k for r in rows for k in r.keys()})
    with open(save_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _pack_domain(ds: Dataset) -> Dict[str, torch.Tensor]:
    x = ds.data if torch.is_tensor(ds.data) else torch.as_tensor(ds.data)
    y = _labels_for_domain(ds)
    yem = (
        torch.as_tensor(ds.targets_em).long().view(-1)
        if hasattr(ds, "targets_em") and ds.targets_em is not None
        else y.clone()
    )
    w = _domain_weight_tensor(ds)
    return {
        "data": x.detach().cpu().float(),
        "targets": y.detach().cpu().long(),
        "targets_em": yem.detach().cpu().long(),
        "weight": w.detach().cpu().float(),
    }


def _save_trajectory_cache(
    output_dir: str,
    args: argparse.Namespace,
    method_domains: Dict[str, List[Dataset]],
    trajectory_chains: Dict[str, Sequence[Dataset]],
    trajectory_labels: Sequence[str],
    trajectory_is_synth: Sequence[bool],
    shared_idx: np.ndarray,
    classwise_report: Sequence[Dict[str, object]],
) -> str:
    payload = {
        "meta": {
            "target_angle": int(args.target_angle),
            "seed": int(args.seed),
            "generated_domains": int(args.generated_domains),
            "sample_selection": str(getattr(args, "sample_selection", "random")),
            "class_plot_num_images": int(getattr(args, "class_plot_num_images", 10)),
        },
        "trajectory_labels": list(trajectory_labels),
        "trajectory_is_synth": list(bool(v) for v in trajectory_is_synth),
        "shared_idx": np.asarray(shared_idx, dtype=np.int64),
        "classwise_report": list(classwise_report),
        "method_domains": {
            m: [_pack_domain(ds) for ds in chain]
            for m, chain in method_domains.items()
        },
        "trajectory_chains": {
            m: [_pack_domain(ds) for ds in chain]
            for m, chain in trajectory_chains.items()
        },
    }
    save_path = os.path.join(output_dir, "trajectory_cache.pt")
    torch.save(payload, save_path)
    return save_path


def _unpack_domain(payload: Dict[str, torch.Tensor]) -> DomainDataset:
    return DomainDataset(
        payload["data"].detach().cpu().float(),
        payload["weight"].detach().cpu().float().view(-1),
        payload["targets"].detach().cpu().long().view(-1),
        payload["targets_em"].detach().cpu().long().view(-1),
    )


def _load_trajectory_cache(cache_path: str) -> Dict[str, object]:
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    required = {"method_domains", "trajectory_chains", "trajectory_labels", "trajectory_is_synth", "shared_idx"}
    missing = required - set(payload.keys())
    if missing:
        raise RuntimeError(f"Invalid trajectory cache at {cache_path}; missing keys: {sorted(missing)}")

    method_domains = {
        m: [_unpack_domain(ds) for ds in chain]
        for m, chain in payload["method_domains"].items()
    }
    trajectory_chains = {
        m: [_unpack_domain(ds) for ds in chain]
        for m, chain in payload["trajectory_chains"].items()
    }
    return {
        "method_domains": method_domains,
        "trajectory_chains": trajectory_chains,
        "trajectory_labels": list(payload["trajectory_labels"]),
        "trajectory_is_synth": [bool(v) for v in payload["trajectory_is_synth"]],
        "shared_idx": np.asarray(payload["shared_idx"], dtype=np.int64),
        "classwise_report": list(payload.get("classwise_report", [])),
        "meta": dict(payload.get("meta", {})),
    }


def _load_feature_vae_for_plotting(
    args: argparse.Namespace,
    output_dir: str,
    feature_dim: int,
) -> FeatureToImageVAE:
    if args.vae_path:
        ckpt_path = args.vae_path
    else:
        if args.vae_latent_dim is None:
            raise ValueError(
                "In --load-trajectory-cache mode, provide --vae-path or --vae-latent-dim so VAE can be loaded."
            )
        ckpt_path = _vae_checkpoint_path(args, output_dir, int(args.vae_latent_dim))

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"VAE checkpoint not found: {ckpt_path}")

    payload = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    latent_ckpt = payload.get("latent_dim") if isinstance(payload, dict) else None
    feature_dim_ckpt = payload.get("feature_dim") if isinstance(payload, dict) else None
    latent_dim = int(args.vae_latent_dim) if args.vae_latent_dim is not None else int(latent_ckpt)

    if feature_dim_ckpt is not None and int(feature_dim_ckpt) != int(feature_dim):
        raise RuntimeError(
            f"VAE feature_dim mismatch: ckpt={feature_dim_ckpt} vs cache={feature_dim}"
        )
    if latent_ckpt is not None and int(latent_ckpt) != int(latent_dim):
        raise RuntimeError(
            f"VAE latent_dim mismatch: ckpt={latent_ckpt} vs requested={latent_dim}"
        )

    vae = FeatureToImageVAE(feature_dim=int(feature_dim), latent_dim=int(latent_dim), image_dim=28 * 28).to(DEVICE)
    vae.load_state_dict(state, strict=True)
    vae.eval()
    print(f"[VAE] Loaded checkpoint: {ckpt_path}")
    return vae


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

    # Cache-only mode: do not train/generate models/domains; only load and plot.
    if args.load_trajectory_cache:
        if args.semantic_eval:
            raise ValueError(
                "--semantic-eval is unsupported with --load-trajectory-cache because it needs a teacher model."
            )
        cache_path = args.trajectory_cache_path.strip() if args.trajectory_cache_path else os.path.join(output_dir, "trajectory_cache.pt")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Trajectory cache not found: {cache_path}")

        cache_obj = _load_trajectory_cache(cache_path)
        method_domains = cache_obj["method_domains"]
        trajectory_chains = cache_obj["trajectory_chains"]
        trajectory_labels = cache_obj["trajectory_labels"]
        trajectory_is_synth = cache_obj["trajectory_is_synth"]
        shared_idx = cache_obj["shared_idx"]
        classwise_report = cache_obj["classwise_report"]
        print(f"[Trajectory] Loaded cache: {cache_path}")

        trajectory_methods = ("goat", "goat_classwise", "ours_fr", "ours_eta")
        n_steps_by_method = {k: len(method_domains[k]) for k in trajectory_methods}
        min_steps = min(n_steps_by_method.values())
        if args.max_steps is not None:
            min_steps = min(min_steps, args.max_steps)
        if min_steps <= 0:
            raise RuntimeError("Trajectory cache is empty.")
        for m in trajectory_methods:
            method_domains[m] = method_domains[m][:min_steps]

        feature_dim = int(method_domains["goat"][0].data.shape[1])
        vae = _load_feature_vae_for_plotting(args, output_dir, feature_dim=feature_dim)

        src_trainset = get_single_rotate(False, 0)
        tgt_trainset = get_single_rotate(False, args.target_angle)
        save_endpoint_grid(
            os.path.join(output_dir, "endpoints_source_target.png"),
            src_trainset,
            tgt_trainset,
            n_samples=args.samples_per_step,
            seed=args.seed,
        )

        group_a = ("GOAT", "C2GDA-Wass")
        map_a = {"GOAT": "goat", "C2GDA-Wass": "goat_classwise"}
        for step in range(min_steps):
            decoded, idx = _collect_step_images(
                vae,
                methods=[map_a[m] for m in group_a],
                method_domains=method_domains,
                step=step,
                n_samples=args.samples_per_step,
                seed=args.seed,
                sample_selection=args.sample_selection,
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

        group_b = ("C2GDA-Wass", "C2GDA-FR", "C2GDA-Nat")
        map_b = {"C2GDA-Wass": "goat_classwise", "C2GDA-FR": "ours_fr", "C2GDA-Nat": "ours_eta"}
        for step in range(min_steps):
            decoded, idx = _collect_step_images(
                vae,
                methods=[map_b[m] for m in group_b],
                method_domains=method_domains,
                step=step,
                n_samples=args.samples_per_step,
                seed=args.seed,
                sample_selection=args.sample_selection,
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

        np.save(os.path.join(output_dir, "trajectory_indices.npy"), shared_idx)
        print(f"[Trajectory] Shared indices: {shared_idx.tolist()}")

        display_names = {
            "goat": "GOAT",
            "goat_classwise": "C2GDA-Wass",
            "ours_fr": "C2GDA-FR",
            "ours_eta": "C2GDA-Nat",
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

        n_classes = int(_labels_for_domain(trajectory_chains["goat"][0]).max().item()) + 1
        n_class_imgs = max(1, int(args.class_plot_num_images))
        skipped_by_threshold = 0
        used_fallback = 0
        for cls in range(n_classes):
            class_idx = _common_indices_for_class(
                trajectory_chains=trajectory_chains,
                methods=trajectory_methods,
                cls=cls,
                n_samples=n_class_imgs,
                seed=args.seed,
                sample_selection=args.sample_selection,
            )
            if len(class_idx) >= int(args.min_class_plot_indices):
                class_idx = _expand_indices(class_idx, n_class_imgs, seed=args.seed + 12000 + cls)
                for img_i, idx_val in enumerate(class_idx[:n_class_imgs]):
                    save_all_methods_class_trajectory_grid(
                        save_path=os.path.join(output_dir, f"trajectory_all_methods_class{cls}_img{img_i}.pdf"),
                        methods=trajectory_methods,
                        display_names=display_names,
                        trajectory_chains=trajectory_chains,
                        trajectory_labels=trajectory_labels,
                        trajectory_is_synth=trajectory_is_synth,
                        vae=vae,
                        indices=np.array([int(idx_val)], dtype=np.int64),
                        class_id=cls,
                    )
                print(f"[Trajectory] class {cls}: saved {n_class_imgs} files with shared indices.")
                continue

            per_method_idx = _per_method_indices_for_class(
                trajectory_chains=trajectory_chains,
                methods=trajectory_methods,
                cls=cls,
                n_samples=n_class_imgs,
                seed=args.seed,
                sample_selection=args.sample_selection,
            )
            non_empty = sum(1 for m in trajectory_methods if len(per_method_idx.get(m, [])) > 0)
            if non_empty == 0:
                skipped_by_threshold += 1
                print(
                    f"[Trajectory] class {cls}: skipped "
                    f"(shared_idx={len(class_idx)}, per_method_nonempty=0)."
                )
                continue
            used_fallback += 1
            for img_i in range(n_class_imgs):
                pm_one: Dict[str, np.ndarray] = {}
                for m in trajectory_methods:
                    arr = per_method_idx.get(m, np.array([], dtype=np.int64))
                    if len(arr) == 0:
                        pm_one[m] = np.array([], dtype=np.int64)
                    else:
                        pm_one[m] = np.array([int(arr[img_i])], dtype=np.int64)
                save_all_methods_class_trajectory_grid(
                    save_path=os.path.join(output_dir, f"trajectory_all_methods_class{cls}_img{img_i}.pdf"),
                    methods=trajectory_methods,
                    display_names=display_names,
                    trajectory_chains=trajectory_chains,
                    trajectory_labels=trajectory_labels,
                    trajectory_is_synth=trajectory_is_synth,
                    vae=vae,
                    indices=np.array([], dtype=np.int64),
                    class_id=cls,
                    per_method_indices=pm_one,
                )
            print(f"[Trajectory] class {cls}: saved {n_class_imgs} files with per-method fallback indices.")

        if skipped_by_threshold > 0:
            print(f"[Trajectory] skipped {skipped_by_threshold} classes by minimum-index threshold.")
        if used_fallback > 0:
            print(f"[Trajectory] used per-method fallback for {used_fallback} classes.")
        if classwise_report:
            report_path = os.path.join(output_dir, "classwise_quality_report.csv")
            _save_classwise_quality_report(classwise_report, report_path)
            print(f"[Classwise] report refreshed: {report_path}")

        print(f"[Done] Outputs saved under: {output_dir}")
        return

    # Image-only VAE mode: skip source-model / feature encoding stage entirely.
    if args.vae_input == "image":
        if not args.vae_only:
            raise ValueError(
                "--vae-input image is currently supported only with --vae-only. "
                "Use --vae-input feature for full method trajectory generation."
            )
        src_trainset = get_single_rotate(False, 0)
        tgt_trainset = get_single_rotate(False, args.target_angle)
        latent_dim = args.vae_latent_dim if args.vae_latent_dim is not None else min(512, args.small_dim)
        src_images = _collect_images_in_order(src_trainset, batch_size=args.batch_size, num_workers=args.num_workers)
        tgt_images = _collect_images_in_order(tgt_trainset, batch_size=args.batch_size, num_workers=args.num_workers)
        vae_images = torch.cat([src_images, tgt_images], dim=0)
        _ = train_or_load_image_vae(args, vae_images, latent_dim, output_dir)
        print("[Done] VAE-only image mode: reconstruction model trained/loaded. Skipping method runs and trajectory plots.")
        print(f"[Done] Outputs saved under: {output_dir}")
        return

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

    vae = train_or_load_vae(args, vae_trainset, vae_features, latent_dim, output_dir)
    if args.vae_only:
        print("[Done] VAE-only mode: reconstruction model trained/loaded. Skipping method runs and trajectory plots.")
        print(f"[Done] Outputs saved under: {output_dir}")
        return

    teacher: Optional[nn.Module] = None
    method_domains: Dict[str, List[Dataset]]
    trajectory_chains: Dict[str, List[Dataset]]
    trajectory_labels: List[str]
    trajectory_is_synth: List[bool]
    classwise_report: List[Dict[str, object]]
    shared_idx: np.ndarray
    trajectory_methods = ("goat", "goat_classwise", "ours_fr", "ours_eta")

    cache_path = args.trajectory_cache_path.strip() if args.trajectory_cache_path else ""
    if not cache_path:
        cache_path = os.path.join(output_dir, "trajectory_cache.pt")
    loaded_cache = False
    if args.load_trajectory_cache and os.path.exists(cache_path):
        cache_obj = _load_trajectory_cache(cache_path)
        method_domains = cache_obj["method_domains"]
        shared_idx = cache_obj["shared_idx"]
        classwise_report = cache_obj["classwise_report"]
        cached_generated = int(cache_obj.get("meta", {}).get("generated_domains", args.generated_domains))
        loaded_cache = True
        print(f"[Trajectory] Loaded cache: {cache_path}")
        n_steps_by_method = {k: len(v) for k, v in method_domains.items()}
        min_steps = min(n_steps_by_method.values())
        if args.max_steps is not None:
            min_steps = min(min_steps, args.max_steps)
        if min_steps <= 0:
            raise RuntimeError("Trajectory cache is empty.")
        for m in trajectory_methods:
            method_domains[m] = method_domains[m][:min_steps]

        trajectory_chains = {}
        trajectory_labels = []
        trajectory_is_synth = []
        for m in trajectory_methods:
            chain, labels, is_synth = build_labeled_trajectory_chain(
                encoded_domains=encoded_domains,
                method_synth_domains=method_domains[m],
                generated_domains=cached_generated,
            )
            trajectory_chains[m] = chain
            if not trajectory_labels:
                trajectory_labels = labels
                trajectory_is_synth = is_synth
    else:
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

        method_domains, classwise_report = extract_method_domains(
            method_args, encoded_domains, args.generated_domains
        )
        report_path = os.path.join(output_dir, "classwise_quality_report.csv")
        _save_classwise_quality_report(classwise_report, report_path)
        used = sum(int(r.get("used", 0)) for r in classwise_report)
        total = len(classwise_report)
        print(f"[Classwise] usable class-pairs: {used}/{total}; report saved: {report_path}")
        n_steps_by_method = {k: len(v) for k, v in method_domains.items()}
        print(f"[Run] Extracted synthetic steps per method: {n_steps_by_method}")
        min_steps = min(n_steps_by_method.values())
        if args.max_steps is not None:
            min_steps = min(min_steps, args.max_steps)
        if min_steps <= 0:
            raise RuntimeError("No synthetic intermediate steps were produced.")

        trajectory_chains = {}
        trajectory_labels = []
        trajectory_is_synth = []
        for m in trajectory_methods:
            chain, labels, is_synth = build_labeled_trajectory_chain(
                encoded_domains=encoded_domains,
                method_synth_domains=method_domains[m][:min_steps],
                generated_domains=args.generated_domains,
            )
            trajectory_chains[m] = chain
            if not trajectory_labels:
                trajectory_labels = labels
                trajectory_is_synth = is_synth

        shared_idx = _sample_common_indices_for_trajectories(
            trajectory_chains=trajectory_chains,
            methods=trajectory_methods,
            n_samples=args.samples_per_step,
            seed=args.seed,
            sample_selection=args.sample_selection,
        )

    if args.semantic_eval:
        if teacher is None:
            teacher = copy.deepcopy(ref_model).to(DEVICE).eval()
        sem_by_step, sem_summary = _evaluate_semantic_consistency(
            classifier=teacher,
            vae=vae,
            method_domains=method_domains,
            max_steps=min_steps,
            sample_selection=args.sample_selection,
            seed=args.seed,
            max_samples_per_step=args.semantic_eval_samples,
        )
        sem_step_path = os.path.join(output_dir, "class_consistency_by_step.csv")
        sem_sum_path = os.path.join(output_dir, "class_consistency_summary.csv")
        _save_rows_csv(sem_by_step, sem_step_path)
        _save_rows_csv(sem_summary, sem_sum_path)
        print(f"[Semantic] Saved: {sem_step_path}")
        print(f"[Semantic] Saved: {sem_sum_path}")

    save_endpoint_grid(
        os.path.join(output_dir, "endpoints_source_target.png"),
        src_trainset,
        tgt_trainset,
        n_samples=args.samples_per_step,
        seed=args.seed,
    )

    group_a = ("GOAT", "C2GDA-Wass")
    map_a = {"GOAT": "goat", "C2GDA-Wass": "goat_classwise"}
    for step in range(min_steps):
        decoded, idx = _collect_step_images(
            vae,
            methods=[map_a[m] for m in group_a],
            method_domains=method_domains,
            step=step,
            n_samples=args.samples_per_step,
            seed=args.seed,
            sample_selection=args.sample_selection,
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

    group_b = ("C2GDA-Wass", "C2GDA-FR", "C2GDA-Nat")
    map_b = {"C2GDA-Wass": "goat_classwise", "C2GDA-FR": "ours_fr", "C2GDA-Nat": "ours_eta"}
    for step in range(min_steps):
        decoded, idx = _collect_step_images(
            vae,
            methods=[map_b[m] for m in group_b],
            method_domains=method_domains,
            step=step,
            n_samples=args.samples_per_step,
            seed=args.seed,
            sample_selection=args.sample_selection,
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
    np.save(os.path.join(output_dir, "trajectory_indices.npy"), shared_idx)
    print(f"[Trajectory] Shared indices: {shared_idx.tolist()}")
    if (not loaded_cache) and args.save_trajectory_cache:
        saved_cache = _save_trajectory_cache(
            output_dir=output_dir,
            args=args,
            method_domains=method_domains,
            trajectory_chains=trajectory_chains,
            trajectory_labels=trajectory_labels,
            trajectory_is_synth=trajectory_is_synth,
            shared_idx=shared_idx,
            classwise_report=classwise_report,
        )
        print(f"[Trajectory] Saved cache: {saved_cache}")

    display_names = {
        "goat": "GOAT",
        "goat_classwise": "C2GDA-Wass",
        "ours_fr": "C2GDA-FR",
        "ours_eta": "C2GDA-Nat",
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
    n_class_imgs = max(1, int(args.class_plot_num_images))
    skipped_by_threshold = 0
    used_fallback = 0
    for cls in range(n_classes):
        class_idx = _common_indices_for_class(
            trajectory_chains=trajectory_chains,
            methods=trajectory_methods,
            cls=cls,
            n_samples=n_class_imgs,
            seed=args.seed,
            sample_selection=args.sample_selection,
        )
        if len(class_idx) >= int(args.min_class_plot_indices):
            class_idx = _expand_indices(class_idx, n_class_imgs, seed=args.seed + 12000 + cls)
            for img_i, idx_val in enumerate(class_idx[:n_class_imgs]):
                save_all_methods_class_trajectory_grid(
                    save_path=os.path.join(output_dir, f"trajectory_all_methods_class{cls}_img{img_i}.pdf"),
                    methods=trajectory_methods,
                    display_names=display_names,
                    trajectory_chains=trajectory_chains,
                    trajectory_labels=trajectory_labels,
                    trajectory_is_synth=trajectory_is_synth,
                    vae=vae,
                    indices=np.array([int(idx_val)], dtype=np.int64),
                    class_id=cls,
                )
            print(f"[Trajectory] class {cls}: saved {n_class_imgs} files with shared indices.")
            continue

        # Fallback: allow per-method class indices so every class can be visualized.
        per_method_idx = _per_method_indices_for_class(
            trajectory_chains=trajectory_chains,
            methods=trajectory_methods,
            cls=cls,
            n_samples=n_class_imgs,
            seed=args.seed,
            sample_selection=args.sample_selection,
        )
        non_empty = sum(1 for m in trajectory_methods if len(per_method_idx.get(m, [])) > 0)
        if non_empty == 0:
            skipped_by_threshold += 1
            print(
                f"[Trajectory] class {cls}: skipped "
                f"(shared_idx={len(class_idx)}, per_method_nonempty=0)."
            )
            continue
        used_fallback += 1
        for img_i in range(n_class_imgs):
            pm_one: Dict[str, np.ndarray] = {}
            for m in trajectory_methods:
                arr = per_method_idx.get(m, np.array([], dtype=np.int64))
                if len(arr) == 0:
                    pm_one[m] = np.array([], dtype=np.int64)
                else:
                    pm_one[m] = np.array([int(arr[img_i])], dtype=np.int64)
            save_all_methods_class_trajectory_grid(
                save_path=os.path.join(output_dir, f"trajectory_all_methods_class{cls}_img{img_i}.pdf"),
                methods=trajectory_methods,
                display_names=display_names,
                trajectory_chains=trajectory_chains,
                trajectory_labels=trajectory_labels,
                trajectory_is_synth=trajectory_is_synth,
                vae=vae,
                indices=np.array([], dtype=np.int64),
                class_id=cls,
                per_method_indices=pm_one,
            )
        print(f"[Trajectory] class {cls}: saved {n_class_imgs} files with per-method fallback indices.")

    if skipped_by_threshold > 0:
        print(f"[Trajectory] skipped {skipped_by_threshold} classes by minimum-index threshold.")
    if used_fallback > 0:
        print(f"[Trajectory] used per-method fallback for {used_fallback} classes.")

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
    parser.add_argument(
        "--vae-input",
        choices=["feature", "image"],
        default="feature",
        help="VAE input type. 'feature' uses feature-conditioned VAE (full pipeline). "
             "'image' trains plain image VAE and skips stage-1 when used with --vae-only.",
    )
    parser.add_argument("--vae-epochs", type=int, default=100)
    parser.add_argument("--vae-lr", type=float, default=1e-3)
    parser.add_argument("--vae-val-frac", type=float, default=0.1, help="Validation fraction for VAE early stopping.")
    parser.add_argument("--vae-patience", type=int, default=12, help="Early stopping patience on VAE validation loss.")
    parser.add_argument("--vae-min-delta", type=float, default=1e-3, help="Minimum validation-loss improvement to reset patience.")
    parser.add_argument("--vae-path", type=str, default="")
    parser.add_argument("--samples-per-step", type=int, default=16)
    parser.add_argument(
        "--sample-selection",
        choices=["random", "confidence"],
        default="random",
        help="How to choose plotted sample indices: random or highest-confidence by domain weights.",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument(
        "--cw-min-class-count",
        type=int,
        default=100,
        help="Minimum per-class sample count required in both adjacent domains for classwise OT.",
    )
    parser.add_argument(
        "--cw-confidence-quantile",
        type=float,
        default=0.0,
        help="Drop low-confidence per-class samples below this weight quantile (0 disables).",
    )
    parser.add_argument(
        "--cw-prototype-shrink",
        type=float,
        default=0.0,
        help="Prototype shrink factor in [0,1] for classwise synthetic features.",
    )
    parser.add_argument(
        "--min-class-plot-indices",
        type=int,
        default=1,
        help="Skip class trajectory figure if common fixed indices across methods are below this threshold.",
    )
    parser.add_argument(
        "--class-plot-num-images",
        type=int,
        default=10,
        help="Number of per-class trajectory files to save (trajectory_all_methods_class{c}_img{i}.pdf).",
    )
    parser.add_argument(
        "--semantic-eval",
        action="store_true",
        help="Evaluate class-semantic consistency on decoded synthetic domains and save CSV reports.",
    )
    parser.add_argument(
        "--semantic-eval-samples",
        type=int,
        default=1024,
        help="Max samples per method/step used in semantic consistency evaluation.",
    )
    parser.add_argument(
        "--vae-only",
        action="store_true",
        help="Train/load the feature-to-image VAE only, then exit before EM/method trajectory generation.",
    )
    parser.add_argument(
        "--save-trajectory-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save serialized synthetic trajectory domains for fast future re-plotting.",
    )
    parser.add_argument(
        "--load-trajectory-cache",
        action="store_true",
        help="Load precomputed trajectory cache and skip synthetic-domain generation.",
    )
    parser.add_argument(
        "--trajectory-cache-path",
        type=str,
        default="",
        help="Optional path to trajectory cache .pt file (default: <output-dir>/trajectory_cache.pt).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
