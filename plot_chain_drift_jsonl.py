#!/usr/bin/env python3
"""
Plot interpolation-quality diagnostics from *_drift.jsonl produced by
experiment_chain_diagnostics.py.

Outputs:
1) global drift comparison (mean +/- std over seeds)
2) class-conditional drift comparison (mean and worst; mean +/- std)
3) per-class drift heatmaps (one panel per method, averaged over seeds)

Example:
python plot_chain_drift_jsonl.py \
  --inputs "logs/mnist/s*/target90/chain_diag_gt0_gen3_drift.jsonl" \
  --out_dir figs/target90/chain_compare \
  --methods goat cc_wass cc_fr cc_nat \
  --title "MNIST target=90, gt=0, gen=3"
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt


def _normalize_method_name(name: str) -> str:
    s = str(name).strip().lower().replace("-", "_")
    alias = {
        "cc_goat": "cc_wass",
        "cc_w2": "cc_wass",
        "cc_ot": "cc_wass",
        "cc_wass": "cc_wass",
        "cc_fr": "cc_fr",
        "cc_nat": "cc_nat",
        "nat": "cc_nat",
        "fr": "cc_fr",
    }
    return alias.get(s, s)


@dataclass
class DriftRun:
    method: str
    seed: int
    global_drift: np.ndarray
    mean_cond_drift: np.ndarray
    worst_cond_drift: np.ndarray
    # Shape: (K, E) where K=classes, E=edges
    per_class_drift: np.ndarray


def _to_float_array(xs: Sequence[Any]) -> np.ndarray:
    vals: List[float] = []
    for x in xs:
        if x is None:
            vals.append(np.nan)
        else:
            vals.append(float(x))
    return np.asarray(vals, dtype=np.float64)


def _to_per_class_matrix(raw: Sequence[Sequence[Any]]) -> np.ndarray:
    # raw is edge-major: [edge][class]
    if len(raw) == 0:
        return np.full((0, 0), np.nan, dtype=np.float64)
    edge_rows = []
    for row in raw:
        edge_rows.append(_to_float_array(row))
    mat_edge_class = np.stack(edge_rows, axis=0)  # (E, K)
    return mat_edge_class.T  # (K, E)


def read_jsonl(paths: Sequence[str]) -> List[DriftRun]:
    runs: List[DriftRun] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                method = _normalize_method_name(obj["method"])
                seed = int(obj.get("seed", -1))
                drift = obj["drift"]
                runs.append(
                    DriftRun(
                        method=method,
                        seed=seed,
                        global_drift=_to_float_array(drift["global_drift"]),
                        mean_cond_drift=_to_float_array(drift["mean_cond_drift"]),
                        worst_cond_drift=_to_float_array(drift["worst_cond_drift"]),
                        per_class_drift=_to_per_class_matrix(drift["per_class_drift"]),
                    )
                )
    return runs


def group_by_method(runs: Sequence[DriftRun]) -> Dict[str, List[DriftRun]]:
    out: Dict[str, List[DriftRun]] = {}
    for r in runs:
        out.setdefault(r.method, []).append(r)
    return out


def _stack_truncate(arrs: Sequence[np.ndarray]) -> np.ndarray:
    if len(arrs) == 0:
        raise ValueError("No arrays provided.")
    L = min(a.shape[0] for a in arrs)
    return np.stack([a[:L] for a in arrs], axis=0)


def _mean_std(arrs: Sequence[np.ndarray]) -> Dict[str, np.ndarray]:
    s = _stack_truncate(arrs)
    finite = np.isfinite(s)
    cnt = finite.sum(axis=0)
    sumv = np.where(finite, s, 0.0).sum(axis=0)
    mean = np.divide(sumv, cnt, out=np.full_like(sumv, np.nan, dtype=np.float64), where=(cnt > 0))
    centered = np.where(finite, s - mean[None, :], 0.0)
    var = np.divide(
        (centered ** 2).sum(axis=0),
        cnt,
        out=np.full_like(sumv, np.nan, dtype=np.float64),
        where=(cnt > 0),
    )
    std = np.sqrt(var)
    return dict(mean=mean, std=std, L=np.asarray([s.shape[1]]))


def _aggregate_heatmap(mats: Sequence[np.ndarray]) -> np.ndarray:
    # mats are (K, E), possibly different E. K should match in a method group.
    if len(mats) == 0:
        return np.full((0, 0), np.nan, dtype=np.float64)
    K = mats[0].shape[0]
    E = min(m.shape[1] for m in mats)
    stack = []
    for m in mats:
        if m.shape[0] != K:
            raise ValueError("K mismatch across seeds for the same method.")
        stack.append(m[:, :E])
    s = np.stack(stack, axis=0)
    finite = np.isfinite(s)
    cnt = finite.sum(axis=0)
    sumv = np.where(finite, s, 0.0).sum(axis=0)
    return np.divide(
        sumv,
        cnt,
        out=np.full_like(sumv, np.nan, dtype=np.float64),
        where=(cnt > 0),
    )


def _plot_band(ax, x: np.ndarray, mean: np.ndarray, std: np.ndarray, label: str) -> None:
    ax.plot(x, mean, label=label, linewidth=2.0, marker="o", markersize=3.5)
    ax.fill_between(x, mean - std, mean + std, alpha=0.2)


def _has_any_finite(a: np.ndarray) -> bool:
    return bool(np.isfinite(a).any())


def plot_all(
    grouped: Dict[str, List[DriftRun]],
    out_dir: str,
    title: str,
    methods: Optional[Sequence[str]] = None,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    if methods is None or len(methods) == 0:
        methods = sorted(grouped.keys())
    methods = [_normalize_method_name(m) for m in methods]
    methods = [m for m in methods if m in grouped]
    if len(methods) == 0:
        raise ValueError("No requested methods found in inputs.")

    global_stats: Dict[str, Dict[str, np.ndarray]] = {}
    mean_cond_stats: Dict[str, Dict[str, np.ndarray]] = {}
    worst_cond_stats: Dict[str, Dict[str, np.ndarray]] = {}
    heatmaps: Dict[str, np.ndarray] = {}

    for m in methods:
        rs = grouped[m]
        global_stats[m] = _mean_std([r.global_drift for r in rs])
        mean_cond_stats[m] = _mean_std([r.mean_cond_drift for r in rs])
        worst_cond_stats[m] = _mean_std([r.worst_cond_drift for r in rs])
        heatmaps[m] = _aggregate_heatmap([r.per_class_drift for r in rs])
        print(
            f"[summary] {m}: runs={len(rs)} "
            f"global_finite={int(np.isfinite(global_stats[m]['mean']).sum())} "
            f"mean_cond_finite={int(np.isfinite(mean_cond_stats[m]['mean']).sum())} "
            f"worst_cond_finite={int(np.isfinite(worst_cond_stats[m]['mean']).sum())} "
            f"heatmap_finite={int(np.isfinite(heatmaps[m]).sum())}"
        )

    # Figure 1: global drift comparison
    plt.figure(figsize=(8, 4.8))
    ax = plt.gca()
    plotted = 0
    for m in methods:
        if not _has_any_finite(global_stats[m]["mean"]):
            continue
        L = int(global_stats[m]["L"][0])
        x = np.arange(1, L + 1)
        _plot_band(ax, x, global_stats[m]["mean"], global_stats[m]["std"], label=m)
        plotted += 1
    ax.set_xlabel("Edge index t (step t -> t+1)")
    ax.set_ylabel("Global drift (W2^2)")
    ax.set_title(f"{title} - Global Drift")
    ax.grid(alpha=0.25)
    if plotted > 0:
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No finite global drift data", ha="center", va="center", transform=ax.transAxes)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "drift_global_compare.png"), dpi=200)
    plt.close()

    # Figure 2: class-conditional comparison (mean and worst)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=False)
    plotted_cond = 0
    for m in methods:
        if _has_any_finite(mean_cond_stats[m]["mean"]):
            Lm = int(mean_cond_stats[m]["L"][0])
            x_m = np.arange(1, Lm + 1)
            _plot_band(axes[0], x_m, mean_cond_stats[m]["mean"], mean_cond_stats[m]["std"], label=m)
            plotted_cond += 1

        if _has_any_finite(worst_cond_stats[m]["mean"]):
            Lw = int(worst_cond_stats[m]["L"][0])
            x_w = np.arange(1, Lw + 1)
            _plot_band(axes[1], x_w, worst_cond_stats[m]["mean"], worst_cond_stats[m]["std"], label=m)

    axes[0].set_xlabel("Edge index t")
    axes[0].set_ylabel("Mean class-conditional drift")
    axes[0].set_title("Mean Over Classes")
    axes[0].grid(alpha=0.25)

    axes[1].set_xlabel("Edge index t")
    axes[1].set_ylabel("Worst-class drift")
    axes[1].set_title("Max Over Classes")
    axes[1].grid(alpha=0.25)

    if plotted_cond > 0:
        axes[1].legend(loc="best")
    else:
        axes[0].text(0.5, 0.5, "No finite class-conditional data", ha="center", va="center", transform=axes[0].transAxes)
        axes[1].text(0.5, 0.5, "No finite class-conditional data", ha="center", va="center", transform=axes[1].transAxes)
    fig.suptitle(f"{title} - Class-Conditional Drift", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "drift_conditional_compare.png"), dpi=200)
    plt.close()

    # Figure 2b: worst-class drift only (standalone)
    plt.figure(figsize=(8, 4.8))
    ax = plt.gca()
    plotted_worst = 0
    for m in methods:
        if not _has_any_finite(worst_cond_stats[m]["mean"]):
            continue
        Lw = int(worst_cond_stats[m]["L"][0])
        x_w = np.arange(1, Lw + 1)
        _plot_band(ax, x_w, worst_cond_stats[m]["mean"], worst_cond_stats[m]["std"], label=m)
        plotted_worst += 1
    ax.set_xlabel("Edge index t")
    ax.set_ylabel("Worst-class drift")
    ax.set_title(f"{title} - Maximum Class-wise Drift")
    ax.grid(alpha=0.25)
    if plotted_worst > 0:
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No finite worst-class data", ha="center", va="center", transform=ax.transAxes)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "drift_max_class_compare.png"), dpi=200)
    plt.close()

    # Figure 3: per-class drift heatmaps (averaged over seeds)
    vmax = 0.0
    for m in methods:
        hm = heatmaps[m]
        if hm.size == 0:
            continue
        cur = np.nanmax(hm)
        if np.isfinite(cur):
            vmax = max(vmax, float(cur))
    if vmax <= 0.0:
        vmax = 1.0

    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.8), squeeze=False)
    im = None
    for i, m in enumerate(methods):
        ax = axes[0, i]
        hm = heatmaps[m]
        if hm.size == 0 or not _has_any_finite(hm):
            ax.set_title(f"{m}\n(no data)")
            ax.axis("off")
            continue
        im = ax.imshow(hm, aspect="auto", origin="lower", vmin=0.0, vmax=vmax, cmap="viridis")
        ax.set_title(m)
        ax.set_xlabel("Edge index t")
        if i == 0:
            ax.set_ylabel("Class k")
    fig.suptitle(f"{title} - Per-Class Drift Heatmaps (mean over seeds)", y=1.02)
    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.86)
        cbar.set_label("Per-class drift")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "drift_per_class_heatmaps.png"), dpi=200)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="Paths or globs to *_drift.jsonl files.")
    ap.add_argument("--out_dir", required=True, help="Where to save figures.")
    ap.add_argument("--title", default="Interpolation Diagnostics")
    ap.add_argument(
        "--methods",
        nargs="+",
        default=[],
        help="Optional ordered method list to include (e.g., goat cc_wass cc_fr cc_nat).",
    )
    args = ap.parse_args()

    files: List[str] = []
    for pat in args.inputs:
        files.extend(glob.glob(pat))
    files = sorted(set(files))
    if len(files) == 0:
        raise SystemExit("No input files matched.")

    runs = read_jsonl(files)
    if len(runs) == 0:
        raise SystemExit("No drift records found in inputs.")
    grouped = group_by_method(runs)
    plot_all(grouped, out_dir=args.out_dir, title=args.title, methods=args.methods)
    print(f"[OK] Wrote figures to: {args.out_dir}")


if __name__ == "__main__":
    main()
