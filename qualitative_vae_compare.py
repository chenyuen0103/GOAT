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


class MLPVAE(nn.Module):
    def __init__(self, input_dim: int = 784, latent_dim: int = 2048) -> None:
        super().__init__()
        hidden = 1024
        self.enc = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.dec(z)
        return x.view(-1, 1, 28, 28)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


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


def _vae_checkpoint_path(args: argparse.Namespace, output_dir: str, latent_dim: int) -> str:
    if args.vae_path:
        return args.vae_path
    return os.path.join(
        output_dir,
        f"vae_target{args.target_angle}_z{latent_dim}_seed{args.seed}.pt",
    )


def train_or_load_vae(
    args: argparse.Namespace,
    target_dataset: Dataset,
    latent_dim: int,
    output_dir: str,
) -> MLPVAE:
    ckpt_path = _vae_checkpoint_path(args, output_dir, latent_dim)
    vae = MLPVAE(input_dim=28 * 28, latent_dim=latent_dim).to(DEVICE)

    if os.path.exists(ckpt_path):
        payload = torch.load(ckpt_path, map_location=DEVICE)
        state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
        vae.load_state_dict(state, strict=True)
        vae.eval()
        print(f"[VAE] Loaded checkpoint: {ckpt_path}")
        return vae

    loader = DataLoader(
        target_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )
    opt = torch.optim.Adam(vae.parameters(), lr=args.vae_lr)
    vae.train()
    for epoch in range(1, args.vae_epochs + 1):
        running = 0.0
        n_seen = 0
        for x, _ in loader:
            x = x.to(DEVICE).float()
            x_flat = x.view(x.size(0), -1)
            recon, mu, logvar = vae(x_flat)
            recon_flat = recon.view(x.size(0), -1)
            bce = F.binary_cross_entropy(recon_flat, x_flat, reduction="sum")
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = (bce + kld) / max(1, x.size(0))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(loss.item()) * x.size(0)
            n_seen += x.size(0)
        print(f"[VAE] epoch={epoch}/{args.vae_epochs} loss={running / max(1, n_seen):.4f}")

    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(
        {"state_dict": vae.state_dict(), "latent_dim": latent_dim, "target_angle": args.target_angle},
        ckpt_path,
    )
    print(f"[VAE] Saved checkpoint: {ckpt_path}")
    vae.eval()
    return vae


@torch.no_grad()
def decode_latents(vae: MLPVAE, latents: torch.Tensor) -> torch.Tensor:
    z = latents.to(DEVICE).float()
    imgs = vae.decode(z).detach().cpu()
    return imgs.clamp(0.0, 1.0)


def _collect_step_images(
    vae: MLPVAE,
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
        z = ds.data[idx] if torch.is_tensor(ds.data) else torch.as_tensor(ds.data[idx])
        out[m] = decode_latents(vae, z)
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
    vae: MLPVAE,
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
            x = ds.data[0:1] if torch.is_tensor(ds.data) else torch.as_tensor(ds.data[0:1])
            img = decode_latents(vae, x)[0, 0]
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


def _resolve_output_dir(args: argparse.Namespace) -> str:
    if args.output_dir:
        return args.output_dir
    return os.path.join(
        "analysis_outputs",
        f"qualitative_mnist_target{args.target_angle}",
        f"seed{args.seed}",
    )


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

    src_trainset = get_single_rotate(False, 0)
    tgt_trainset = get_single_rotate(False, args.target_angle)
    all_sets, deg_idx = _build_rotated_domains(tgt_trainset, args.target_angle, args.gt_domains)

    model_cfg = ModelConfig(
        encoder_builder=ENCODER,
        mode="mnist",
        n_class=10,
        epochs=10,
        model_path=os.path.join(
            "/data/common/yuenchen/GDA/mnist_models/",
            f"src0_tgt{args.target_angle}_ssl{args.ssl_weight}_dim{args.small_dim}.pth",
        ),
        compress=True,
        in_dim=25088,
        out_dim=args.small_dim,
    )

    ref_model, ref_encoder = build_reference_model(args, model_cfg, src_trainset, tgt_trainset)
    cache_dir = f"cache{args.ssl_weight}/target{args.target_angle}/small_dim{args.small_dim}/"
    _, _, _ = encode_real_domains(
        args,
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

    teacher = copy.deepcopy(ref_model).to(DEVICE).eval()
    _em_bundles, _em_accs, _ = build_em_bundles_for_chain(
        args,
        raw_domains=raw_domains,
        encoded_domains=encoded_domains,
        domain_keys=angle_keys,
        teacher_model=teacher,
        n_classes=10,
        description_prefix="qual-mnist-angle-",
    )

    print("[Run] Running standard methods through run_core_methods for consistency...")
    results = run_core_methods(
        args,
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

    method_domains = extract_method_domains(args, encoded_domains, args.generated_domains)
    n_steps_by_method = {k: len(v) for k, v in method_domains.items()}
    print(f"[Run] Extracted synthetic steps per method: {n_steps_by_method}")
    min_steps = min(n_steps_by_method.values())
    if args.max_steps is not None:
        min_steps = min(min_steps, args.max_steps)
    if min_steps <= 0:
        raise RuntimeError("No synthetic intermediate steps were produced.")

    latent_dim = args.vae_latent_dim if args.vae_latent_dim is not None else args.small_dim
    vae = train_or_load_vae(args, tgt_trainset, latent_dim, output_dir)

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
    parser.add_argument("--vae-path", type=str, default="")
    parser.add_argument("--samples-per-step", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
