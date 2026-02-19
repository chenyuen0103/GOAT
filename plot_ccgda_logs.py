#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# Utilities
# ----------------------------

def find_curve_files(log_root: str) -> List[str]:
    out = []
    for root, _, files in os.walk(log_root):
        for fn in files:
            if fn.endswith("_curves.jsonl"):
                out.append(os.path.join(root, fn))
    return sorted(out)

def read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def pad_to_matrix(curves: List[List[float]]) -> np.ndarray:
    """
    Turns variable-length curves into a (n_runs, T_max) matrix with NaNs for padding.
    """
    if not curves:
        return np.zeros((0, 0), dtype=float)
    T = max(len(c) for c in curves)
    mat = np.full((len(curves), T), np.nan, dtype=float)
    for i, c in enumerate(curves):
        mat[i, :len(c)] = np.asarray(c, dtype=float)
    return mat

def nanmean_std(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return np.nanmean(mat, axis=0), np.nanstd(mat, axis=0)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def sanitize_filename(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_.-]+", "_", s)
    return s.strip("_")


# ----------------------------
# Parsing / aggregation
# ----------------------------

@dataclass
class RunRecord:
    dataset: str
    seed: int
    gt_domains: int
    generated_domains: int
    methods: Dict[str, Dict[str, Any]]  # method -> payload (curves, em_acc, ...)

def infer_dataset_from_path(path: str, log_root: str) -> str:
    # expected: logs/<dataset>/s<seed>/..._curves.jsonl
    rel = os.path.relpath(path, log_root)
    parts = rel.split(os.sep)
    if len(parts) >= 2:
        return parts[0]
    return "unknown"

def load_all_runs(log_root: str) -> List[RunRecord]:
    files = find_curve_files(log_root)
    runs: List[RunRecord] = []
    for fp in files:
        dataset = infer_dataset_from_path(fp, log_root)
        rows = read_jsonl(fp)
        for r in rows:
            runs.append(
                RunRecord(
                    dataset=dataset,
                    seed=int(r.get("seed", -1)),
                    gt_domains=int(r.get("gt_domains", -1)),
                    generated_domains=int(r.get("generated_domains", -1)),
                    methods=r.get("methods", {}),
                )
            )
    return runs


# ----------------------------
# Plotting
# ----------------------------

def plot_trajectory(
    out_dir: str,
    dataset: str,
    gt_domains: int,
    generated_domains: int,
    method_to_curves: Dict[str, List[List[float]]],
    method_to_st: Dict[str, List[float]],
    title_suffix: str = "",
) -> None:
    """
    For each method: plot mean±std test_curve over adaptation steps.
    Also draw baselines if method_to_st provides scalars.
    """
    ensure_dir(out_dir)
    plt.figure()

    # Plot trajectories
    for method, curves in sorted(method_to_curves.items()):
        mat = pad_to_matrix(curves)
        if mat.size == 0:
            continue
        m, s = nanmean_std(mat)
        x = np.arange(len(m))
        plt.plot(x, m, label=method)
        plt.fill_between(x, m - s, m + s, alpha=0.2)

    # Plot ST baselines if present (horizontal)
    # We treat st values as scalars logged per run (often length-1 lists).
    for method, st_vals in sorted(method_to_st.items()):
        if not st_vals:
            continue
        st_mean = float(np.mean(st_vals))
        plt.axhline(st_mean, linestyle="--", alpha=0.5)

    plt.xlabel("Adaptation step (domain index)")
    plt.ylabel("Accuracy")
    title = f"{dataset} | Gobs={gt_domains} | Gsyn={generated_domains}"
    if title_suffix:
        title += f" | {title_suffix}"
    plt.title(title)
    plt.legend()

    fn = sanitize_filename(f"traj_{dataset}_gt{gt_domains}_gen{generated_domains}.png")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fn), dpi=200)
    plt.close()

def plot_final_vs_gen(
    out_dir: str,
    dataset: str,
    gt_domains: int,
    method_to_final_by_gen: Dict[str, Dict[int, List[float]]],
) -> None:
    """
    For each method, plot final test accuracy vs generated_domains (mean±std).
    """
    ensure_dir(out_dir)
    plt.figure()

    # sort gens present
    all_gens = sorted({g for m in method_to_final_by_gen.values() for g in m.keys()})
    if not all_gens:
        return

    for method, gen_map in sorted(method_to_final_by_gen.items()):
        xs, ys, es = [], [], []
        for g in all_gens:
            vals = gen_map.get(g, [])
            if not vals:
                continue
            xs.append(g)
            ys.append(float(np.mean(vals)))
            es.append(float(np.std(vals)))
        if xs:
            plt.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=method)

    plt.xlabel("Generated domains per segment (Gsyn)")
    plt.ylabel("Final target accuracy")
    plt.title(f"{dataset} | Gobs={gt_domains}: Final vs Gsyn")
    plt.legend()
    fn = sanitize_filename(f"final_vs_gen_{dataset}_gt{gt_domains}.png")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fn), dpi=200)
    plt.close()

def plot_em_acc_vs_gen(
    out_dir: str,
    dataset: str,
    gt_domains: int,
    em_acc_by_gen: Dict[str, Dict[int, List[float]]],
) -> None:
    """
    em_acc is logged per method/run (if present). Plot mean±std vs generated_domains.
    """
    ensure_dir(out_dir)
    plt.figure()

    all_gens = sorted({g for m in em_acc_by_gen.values() for g in m.keys()})
    if not all_gens:
        return

    for method, gen_map in sorted(em_acc_by_gen.items()):
        xs, ys, es = [], [], []
        for g in all_gens:
            vals = [v for v in gen_map.get(g, []) if v is not None and not math.isnan(v)]
            if not vals:
                continue
            xs.append(g)
            ys.append(float(np.mean(vals)))
            es.append(float(np.std(vals)))
        if xs:
            plt.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=method)

    plt.xlabel("Generated domains per segment (Gsyn)")
    plt.ylabel("EM→class accuracy (if logged)")
    plt.title(f"{dataset} | Gobs={gt_domains}: EM alignment accuracy vs Gsyn")
    plt.legend()
    fn = sanitize_filename(f"emacc_vs_gen_{dataset}_gt{gt_domains}.png")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fn), dpi=200)
    plt.close()


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-root", type=str, default="logs", help="Root dir containing logs/<dataset>/s<seed>/...")
    ap.add_argument("--out-dir", type=str, default="plots_aggregate", help="Where to save aggregated plots")
    ap.add_argument("--dataset", type=str, default=None, help="Filter to a dataset name (e.g., mnist, portraits, covtype, color_mnist)")
    ap.add_argument("--gt-domains", type=int, default=None, help="Filter to a specific Gobs (gt_domains)")
    args = ap.parse_args()

    runs = load_all_runs(args.log_root)

    # filters
    if args.dataset is not None:
        runs = [r for r in runs if r.dataset == args.dataset]
    if args.gt_domains is not None:
        runs = [r for r in runs if r.gt_domains == args.gt_domains]

    if not runs:
        print("No runs found after filtering.")
        return

    # Group by (dataset, gt_domains, generated_domains)
    group = defaultdict(list)  # (dataset, gt, gen) -> [RunRecord]
    for r in runs:
        group[(r.dataset, r.gt_domains, r.generated_domains)].append(r)

    # For final-vs-gen and em-acc-vs-gen we need (dataset, gt) buckets
    final_bucket = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    # final_bucket[(dataset, gt)][method][gen] -> [final_acc]
    em_bucket = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    # em_bucket[(dataset, gt)][method][gen] -> [em_acc]

    for (dataset, gt, gen), recs in group.items():
        # method -> list of curves across runs
        method_to_test = defaultdict(list)
        method_to_st_scalar = defaultdict(list)

        for rr in recs:
            for method, payload in rr.methods.items():
                test_curve = payload.get("test_curve", None)
                if isinstance(test_curve, list) and len(test_curve) > 0:
                    method_to_test[method].append(test_curve)
                    final_bucket[(dataset, gt)][method][gen].append(float(test_curve[-1]))

                st_curve = payload.get("st_curve", None)
                # most of your code logs ST baselines as length-1 lists (scalar). :contentReference[oaicite:2]{index=2}
                if isinstance(st_curve, list) and len(st_curve) > 0:
                    # treat as scalar baseline; store the first element
                    method_to_st_scalar[method].append(float(st_curve[0]))

                em_acc = payload.get("em_acc", None)
                if em_acc is not None:
                    try:
                        em_bucket[(dataset, gt)][method][gen].append(float(em_acc))
                    except Exception:
                        pass

        out_dir = os.path.join(args.out_dir, dataset, f"gt{gt}", f"gen{gen}")
        plot_trajectory(
            out_dir=out_dir,
            dataset=dataset,
            gt_domains=gt,
            generated_domains=gen,
            method_to_curves=method_to_test,
            method_to_st=method_to_st_scalar,
        )

    # Now summary plots: final vs gen (per dataset, gt)
    for (dataset, gt), method_map in final_bucket.items():
        out_dir = os.path.join(args.out_dir, dataset, f"gt{gt}")
        plot_final_vs_gen(out_dir, dataset, gt, method_map)

    for (dataset, gt), method_map in em_bucket.items():
        out_dir = os.path.join(args.out_dir, dataset, f"gt{gt}")
        plot_em_acc_vs_gen(out_dir, dataset, gt, method_map)

    print(f"Saved plots under: {args.out_dir}")

if __name__ == "__main__":
    main()
