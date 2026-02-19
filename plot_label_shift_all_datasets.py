#!/usr/bin/env python3
"""Plot marginal label shifts P(Y_T) - P(Y_S) from cached encoded datasets.

Supports cache layouts used in this repo:
- MNIST:      cache{ssl}/target{T}/small_dim{D}/encoded_0.pt and encoded_{T}.pt
- non-MNIST:  {dataset}/cache{ssl}/small_dim{D}/encoded_source.pt and encoded_target.pt

Examples
--------
python plot_label_shift_all_datasets.py --small-dim 2048 --ssl-weight 0.1
python plot_label_shift_all_datasets.py --datasets mnist portraits --mnist-targets 45 90
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


def _require_torch_available() -> None:
    try:
        import torch  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "PyTorch is required to load encoded_*.pt files. "
            "Please run this script in the same environment used for experiments."
        ) from e


def _load_torch_obj(path: Path):
    import torch

    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def _get_attr(obj: Any, name: str):
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _to_numpy(x) -> Optional[np.ndarray]:
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
    den = max(float(counts.sum()), 1.0)
    return counts / den


@dataclass
class ShiftRecord:
    dataset: str
    run_name: str
    source_path: Path
    target_path: Path
    classes: np.ndarray
    p_source: np.ndarray
    p_target: np.ndarray
    delta: np.ndarray


def _extract_targets(path: Path) -> np.ndarray:
    obj = _load_torch_obj(path)
    y = _to_numpy(_get_attr(obj, "targets"))
    if y is None:
        raise RuntimeError(f"{path} has no .targets field.")
    return np.asarray(y, dtype=int).reshape(-1)


def _mnist_runs(repo_root: Path, ssl_weight: float, small_dim: int, mnist_targets: Optional[Sequence[int]]) -> List[Dict[str, Any]]:
    base = repo_root / f"cache{ssl_weight}"
    runs: List[Dict[str, Any]] = []
    if not base.exists():
        return runs

    if mnist_targets:
        target_dirs = [base / f"target{int(t)}" for t in mnist_targets]
    else:
        target_dirs = sorted(base.glob("target*"))

    for tdir in target_dirs:
        if not tdir.exists() or not tdir.is_dir():
            continue
        target_name = tdir.name
        t_str = target_name.replace("target", "")
        if not t_str.isdigit():
            continue
        t = int(t_str)

        sd = tdir / f"small_dim{small_dim}"
        if not sd.exists():
            continue
        src = sd / "encoded_0.pt"
        tgt = sd / f"encoded_{t}.pt"
        if src.exists() and tgt.exists():
            runs.append(
                {
                    "dataset": "mnist",
                    "run_name": f"target{t}_small_dim{small_dim}",
                    "source": src,
                    "target": tgt,
                }
            )
    return runs


def _other_dataset_runs(repo_root: Path, dataset: str, ssl_weight: float, small_dim: int) -> List[Dict[str, Any]]:
    sd = repo_root / dataset / f"cache{ssl_weight}" / f"small_dim{small_dim}"
    if not sd.exists():
        return []
    src = sd / "encoded_source.pt"
    tgt = sd / "encoded_target.pt"
    if src.exists() and tgt.exists():
        return [
            {
                "dataset": dataset,
                "run_name": f"small_dim{small_dim}",
                "source": src,
                "target": tgt,
            }
        ]
    return []


def discover_runs(
    repo_root: Path,
    datasets: Sequence[str],
    ssl_weight: float,
    small_dim: int,
    mnist_targets: Optional[Sequence[int]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for ds in datasets:
        if ds == "mnist":
            out.extend(_mnist_runs(repo_root, ssl_weight, small_dim, mnist_targets))
        else:
            out.extend(_other_dataset_runs(repo_root, ds, ssl_weight, small_dim))
    return out


def compute_shift(rec: Dict[str, Any]) -> ShiftRecord:
    y_src = _extract_targets(rec["source"])
    y_tgt = _extract_targets(rec["target"])

    max_cls = 0
    if y_src.size:
        max_cls = max(max_cls, int(y_src.max()))
    if y_tgt.size:
        max_cls = max(max_cls, int(y_tgt.max()))
    K = max_cls + 1

    p_s = _bincount_prob(y_src, K)
    p_t = _bincount_prob(y_tgt, K)
    delta = p_t - p_s

    return ShiftRecord(
        dataset=str(rec["dataset"]),
        run_name=str(rec["run_name"]),
        source_path=Path(rec["source"]),
        target_path=Path(rec["target"]),
        classes=np.arange(K, dtype=int),
        p_source=p_s,
        p_target=p_t,
        delta=delta,
    )


def _plot_single(rec: ShiftRecord, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = rec.classes
    colors = np.where(rec.delta >= 0.0, "#1b9e77", "#d95f02")
    ax.bar(x, rec.delta, color=colors, alpha=0.9)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xlabel("class k")
    ax.set_ylabel("P(Y_T=k) - P(Y_S=k)")
    ax.set_title(f"{rec.dataset}: {rec.run_name}")
    fig.tight_layout()

    fname = f"{rec.dataset}__{rec.run_name}__delta_py.png".replace("/", "_")
    out_path = out_dir / fname
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def _plot_combined(records: Sequence[ShiftRecord], out_dir: Path) -> Optional[Path]:
    if not records:
        return None

    n = len(records)
    ncols = 2
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(11, max(3.3 * nrows, 3.6)), squeeze=False)

    for idx, rec in enumerate(records):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r][c]
        x = rec.classes
        colors = np.where(rec.delta >= 0.0, "#1b9e77", "#d95f02")
        ax.bar(x, rec.delta, color=colors, alpha=0.9)
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_xticks(x)
        ax.set_title(f"{rec.dataset} | {rec.run_name}")
        ax.set_xlabel("class")
        ax.set_ylabel("ΔP")

    for idx in range(n, nrows * ncols):
        r = idx // ncols
        c = idx % ncols
        axes[r][c].axis("off")

    fig.tight_layout()
    out_path = out_dir / "all_datasets_delta_py_grid.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["mnist", "portraits", "covtype", "color_mnist"], choices=["mnist", "portraits", "covtype", "color_mnist"])
    p.add_argument("--ssl-weight", type=float, default=0.1)
    p.add_argument("--small-dim", type=int, default=2048)
    p.add_argument("--mnist-targets", nargs="*", type=int, default=None, help="Optional target angles (e.g. 45 90). Default: all discovered target dirs.")
    p.add_argument("--out-dir", default="plots/label_shift")
    args = p.parse_args()
    _require_torch_available()

    repo_root = Path(".").resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(
        repo_root=repo_root,
        datasets=args.datasets,
        ssl_weight=args.ssl_weight,
        small_dim=args.small_dim,
        mnist_targets=args.mnist_targets,
    )
    if not runs:
        raise SystemExit("No matching cached runs found. Check --small-dim, --ssl-weight, and cache paths.")

    records: List[ShiftRecord] = []
    for r in runs:
        try:
            records.append(compute_shift(r))
        except Exception as e:
            print(f"[WARN] Skipping {r['dataset']} ({r['run_name']}): {e}")

    if not records:
        raise SystemExit("No runs could be loaded successfully.")

    print(f"Loaded {len(records)} run(s).")
    for rec in records:
        per_path = _plot_single(rec, out_dir)
        l1 = float(np.abs(rec.delta).sum())
        print(f"- {rec.dataset:12s} {rec.run_name:24s} | K={len(rec.classes):2d} | L1(ΔP)={l1:.6f} | {per_path}")

    grid_path = _plot_combined(records, out_dir)
    if grid_path is not None:
        print(f"Combined grid: {grid_path}")


if __name__ == "__main__":
    main()
