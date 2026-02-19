#!/usr/bin/env python3
"""
Plot chain diagnostics for Gradual Domain Adaptation experiments.

What this script expects
------------------------
You save, for each (method, seed, dataset, shift), a JSON file containing per-step
(per domain in the chain) *diagonal-Gaussian* statistics in feature space:

{
  "steps": [
    {
      "name": "src" | "syn_r1_l1" | "real_r1" | ...,
      "classes": [
        {"k": 0, "n": 1234, "mu": [...], "var": [...]},
        ...
      ]
    },
    ...
  ],
  "meta": {"K": 10, "d": 512, "method": "...", "seed": 0, ...}
}

- mu and var are length-d lists (diagonal covariance).
- n is the class count used for weighting and for defining "active" classes.

This format is intentionally light-weight (no full cov matrices) to keep files small.

Outputs
-------
Figure A1: global W2(diag-Gaussian) drift between consecutive steps
Figure A2: mean and max class-conditional drift between consecutive steps
Figure A3: optional heatmap of class-conditional drift over (class, step)

Usage
-----
python plot_chain_diagnostics.py \
  --inputs \
    runs/rotmnist/GOAT/chain_stats_seed0.json \
    runs/rotmnist/CCGDA-Wass/chain_stats_seed0.json \
  --out_dir figs/rotmnist \
  --title "Rotated MNIST (deg=90)"

You can also pass globs:
python plot_chain_diagnostics.py --inputs "runs/rotmnist/*/chain_stats_seed*.json" --out_dir figs/rotmnist
"""
from __future__ import annotations
import argparse, glob, json, os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Core math: W2 for diagonal Gaussians
# -----------------------------
def w2_diag(mu1: np.ndarray, var1: np.ndarray, mu2: np.ndarray, var2: np.ndarray) -> float:
    """
    2-Wasserstein distance between N(mu1, diag(var1)) and N(mu2, diag(var2)).
    Exact formula:
      W2^2 = ||mu1-mu2||^2 + ||sqrt(var1) - sqrt(var2)||^2
    """
    mu1 = np.asarray(mu1, dtype=np.float64)
    mu2 = np.asarray(mu2, dtype=np.float64)
    var1 = np.maximum(np.asarray(var1, dtype=np.float64), 0.0)
    var2 = np.maximum(np.asarray(var2, dtype=np.float64), 0.0)
    dmu2 = np.sum((mu1 - mu2) ** 2)
    dsig2 = np.sum((np.sqrt(var1) - np.sqrt(var2)) ** 2)
    return float(np.sqrt(max(dmu2 + dsig2, 0.0)))

def merge_diag_gaussians(class_stats: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given per-class diagonal Gaussian stats for a step, return:
      (mu_global, var_global, pi)
    where global distribution is the mixture, approximated by a single diagonal Gaussian
    via moment matching:
      mu = sum_k pi_k mu_k
      var = sum_k pi_k (var_k + (mu_k - mu)^2)
    """
    # Filter to classes with n>0 and with finite stats
    usable = [c for c in class_stats if c.get("n", 0) > 0]
    if len(usable) == 0:
        raise ValueError("No active classes with n>0 in this step.")
    ns = np.array([c["n"] for c in usable], dtype=np.float64)
    pi = ns / max(ns.sum(), 1.0)
    mus = np.stack([np.asarray(c["mu"], dtype=np.float64) for c in usable], axis=0)
    vars_ = np.stack([np.asarray(c["var"], dtype=np.float64) for c in usable], axis=0)
    mu = (pi[:, None] * mus).sum(axis=0)
    var = (pi[:, None] * (vars_ + (mus - mu[None, :]) ** 2)).sum(axis=0)
    return mu, var, pi

def per_class_drift(step_a: Dict[str, Any], step_b: Dict[str, Any], K: int) -> Tuple[float, float, np.ndarray]:
    """
    Compute:
      mean_k W2_k, max_k W2_k, and the vector W2_k for k in [0..K-1],
    using only classes active in BOTH steps (n>0 in both).
    """
    # Build dicts for fast lookup
    a = {c["k"]: c for c in step_a["classes"]}
    b = {c["k"]: c for c in step_b["classes"]}
    d_k = np.full((K,), np.nan, dtype=np.float64)
    active = []
    for k in range(K):
        if (k in a) and (k in b) and (a[k].get("n", 0) > 0) and (b[k].get("n", 0) > 0):
            d_k[k] = w2_diag(a[k]["mu"], a[k]["var"], b[k]["mu"], b[k]["var"])
            active.append(k)
    if len(active) == 0:
        return float("nan"), float("nan"), d_k
    mean_k = float(np.nanmean(d_k))
    max_k = float(np.nanmax(d_k))
    return mean_k, max_k, d_k

# -----------------------------
# I/O
# -----------------------------
@dataclass
class ChainStats:
    method: str
    seed: int
    K: int
    steps: List[Dict[str, Any]]

def load_chain_stats(path: str) -> ChainStats:
    with open(path, "r") as f:
        obj = json.load(f)
    meta = obj.get("meta", {})
    method = meta.get("method", os.path.basename(os.path.dirname(path)))
    seed = int(meta.get("seed", -1))
    steps = obj["steps"]
    # Infer K from meta or from max class id + 1
    if "K" in meta:
        K = int(meta["K"])
    else:
        maxk = max((c["k"] for s in steps for c in s["classes"]), default=-1)
        K = maxk + 1
    return ChainStats(method=method, seed=seed, K=K, steps=steps)

def group_by_method(files: List[str]) -> Dict[str, List[ChainStats]]:
    out: Dict[str, List[ChainStats]] = {}
    for p in files:
        cs = load_chain_stats(p)
        out.setdefault(cs.method, []).append(cs)
    return out

# -----------------------------
# Compute curves
# -----------------------------
def compute_curves(cs: ChainStats) -> Dict[str, Any]:
    """
    Returns:
      global_drift[t] for t=0..T-2
      class_mean_drift[t], class_max_drift[t]
      class_drift_matrix[k,t] (K x (T-1))
    """
    K = cs.K
    T = len(cs.steps)
    global_d = []
    mean_d = []
    max_d = []
    mat = np.full((K, max(T - 1, 0)), np.nan, dtype=np.float64)

    for t in range(T - 1):
        s0, s1 = cs.steps[t], cs.steps[t + 1]
        mu0, var0, _ = merge_diag_gaussians(s0["classes"])
        mu1, var1, _ = merge_diag_gaussians(s1["classes"])
        global_d.append(w2_diag(mu0, var0, mu1, var1))

        mk, xk, dk = per_class_drift(s0, s1, K)
        mean_d.append(mk)
        max_d.append(xk)
        mat[:, t] = dk

    return {
        "global": np.asarray(global_d),
        "class_mean": np.asarray(mean_d),
        "class_max": np.asarray(max_d),
        "class_mat": mat,
        "step_names": [s.get("name", f"t{idx}") for idx, s in enumerate(cs.steps)],
    }

def aggregate_over_seeds(curves: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Stack curves of potentially different lengths by truncating to min length.
    """
    if len(curves) == 0:
        raise ValueError("No curves to aggregate.")
    L = min(len(c["global"]) for c in curves)
    def stack(key):
        return np.stack([c[key][:L] for c in curves], axis=0)

    out = {
        "global_mean": stack("global").mean(axis=0),
        "global_std": stack("global").std(axis=0),
        "class_mean_mean": stack("class_mean").mean(axis=0),
        "class_mean_std": stack("class_mean").std(axis=0),
        "class_max_mean": stack("class_max").mean(axis=0),
        "class_max_std": stack("class_max").std(axis=0),
        "L": L,
    }
    return out

# -----------------------------
# Plotting
# -----------------------------
def plot_with_band(x: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray, label: str):
    plt.plot(x, y_mean, label=label)
    plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.2)

def make_plots(grouped: Dict[str, List[ChainStats]], out_dir: str, title: str, heatmap_method: Optional[str]):
    os.makedirs(out_dir, exist_ok=True)
    # Compute aggregated curves per method
    methods = sorted(grouped.keys())
    agg = {}
    raw_for_heatmap = {}

    for m in methods:
        curves = [compute_curves(cs) for cs in grouped[m]]
        agg[m] = aggregate_over_seeds(curves)
        raw_for_heatmap[m] = curves  # keep per-seed for optional heatmap

    # A1: global drift
    plt.figure()
    for m in methods:
        L = agg[m]["L"]
        x = np.arange(L)
        plot_with_band(x, agg[m]["global_mean"], agg[m]["global_std"], m)
    plt.xlabel("chain step t (between consecutive domains)")
    plt.ylabel(r"$W_2(\widehat\mu_t,\widehat\mu_{t+1})$ (diag-Gaussian approx.)")
    plt.title(title + " — Global per-step drift")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "A1_global_drift.png"), dpi=200)
    plt.close()

    # A2: class-conditional drift (mean and max)
    plt.figure()
    for m in methods:
        L = agg[m]["L"]
        x = np.arange(L)
        plot_with_band(x, agg[m]["class_mean_mean"], agg[m]["class_mean_std"], m + " (mean over classes)")
    plt.xlabel("chain step t")
    plt.ylabel(r"$\frac{1}{K}\sum_k W_2(\widehat\mu_{t,k},\widehat\mu_{t+1,k})$")
    plt.title(title + " — Class-conditional per-step drift (mean)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "A2_class_mean_drift.png"), dpi=200)
    plt.close()

    plt.figure()
    for m in methods:
        L = agg[m]["L"]
        x = np.arange(L)
        plot_with_band(x, agg[m]["class_max_mean"], agg[m]["class_max_std"], m + " (max over classes)")
    plt.xlabel("chain step t")
    plt.ylabel(r"$\max_k W_2(\widehat\mu_{t,k},\widehat\mu_{t+1,k})$")
    plt.title(title + " — Class-conditional per-step drift (worst class)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "A2_class_max_drift.png"), dpi=200)
    plt.close()

    # A3: heatmap for one method (pick best / most illustrative)
    if heatmap_method is not None and heatmap_method in raw_for_heatmap:
        # Use the first seed for the heatmap (or average later)
        mat = raw_for_heatmap[heatmap_method][0]["class_mat"]
        # truncate columns to finite where at least one class is finite
        # (this is mostly to avoid all-nan if something went wrong)
        plt.figure()
        plt.imshow(mat, aspect="auto")
        plt.colorbar(label=r"$W_2(\widehat\mu_{t,k},\widehat\mu_{t+1,k})$")
        plt.xlabel("chain step t")
        plt.ylabel("class k")
        plt.title(title + f" — Per-class drift heatmap ({heatmap_method})")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "A3_class_drift_heatmap.png"), dpi=200)
        plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="JSON files (or globs) with chain stats.")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--title", default="Chain diagnostics")
    ap.add_argument("--heatmap_method", default=None, help="Method name to use for the heatmap (optional).")
    args = ap.parse_args()

    files = []
    for pat in args.inputs:
        files.extend(glob.glob(pat))
    files = sorted(set(files))
    if len(files) == 0:
        raise SystemExit("No input files matched.")

    grouped = group_by_method(files)
    make_plots(grouped, args.out_dir, args.title, args.heatmap_method)

if __name__ == "__main__":
    main()
