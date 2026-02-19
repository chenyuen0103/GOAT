#!/usr/bin/env python3
"""
Plot figures from your existing *_curves.jsonl logs (produced by log_summary()).

This script does NOT compute drift metrics in feature space. It plots:
- Accuracy trajectories (test_curve) over adaptation steps (Figure B-style backbone)
- Baseline horizontal lines if st_curve/st_all_curve exist
- Final accuracy vs generated_domains across seeds/files (optional summary)

Input format
------------
Each line in *_curves.jsonl is a JSON object like:
{
  "generated_domains": 0,
  "gt_domains": 0,
  "seed": 0,
  "methods": {
    "GOAT": {"test_curve": [...], "train_curve": [...], "st_curve": [...], "st_all_curve": [...], "generated_curve": [...]},
    "CCGDA-FR": {...},
    ...
  }
}

This is consistent with log_summary() writing curves + optional em_acc. (See experiment_refrac.py)

Usage
-----
# Plot trajectories for one file
python plot_curves_jsonl.py --inputs /path/to/*_curves.jsonl --out_dir figs/mnist

# Aggregate across many files (globs ok)
python plot_curves_jsonl.py --inputs "logs/mnist/s*/**/*_curves.jsonl" --out_dir figs/mnist --aggregate

"""
from __future__ import annotations
import argparse, glob, json, os
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

def read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def sanitize(s: str) -> str:
    import re
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", s).strip("_")

def pad_to_matrix(curves: List[List[float]]) -> np.ndarray:
    if not curves:
        return np.zeros((0, 0), dtype=float)
    T = max(len(c) for c in curves)
    mat = np.full((len(curves), T), np.nan, dtype=float)
    for i, c in enumerate(curves):
        mat[i, :len(c)] = np.asarray(c, dtype=float)
    return mat

def nanmean_std(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return np.nanmean(mat, axis=0), np.nanstd(mat, axis=0)

def plot_curve_bundle(ax, curves: List[List[float]], label: str):
    mat = pad_to_matrix(curves)
    if mat.size == 0:
        return
    m, s = nanmean_std(mat)
    x = np.arange(len(m))
    ax.plot(x, m, label=label)
    ax.fill_between(x, m - s, m + s, alpha=0.2)

def extract_scalar_baseline(payload: Dict[str, Any], key: str) -> Optional[float]:
    v = payload.get(key, None)
    if isinstance(v, list) and len(v) > 0:
        return float(v[0])
    if isinstance(v, (int, float)):
        return float(v)
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="*_curves.jsonl files or globs")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--aggregate", action="store_true", help="Aggregate across files/seeds")
    ap.add_argument("--only_methods", nargs="*", default=None, help="If set, only plot these methods (exact names)")
    args = ap.parse_args()

    files = []
    for pat in args.inputs:
        files.extend(glob.glob(pat, recursive=True))
    files = sorted(set(files))
    if not files:
        raise SystemExit("No files matched inputs.")

    ensure_dir(args.out_dir)

    # Collect rows across files
    all_rows = []
    for fp in files:
        for row in read_jsonl(fp):
            row["_file"] = fp
            all_rows.append(row)

    if not all_rows:
        raise SystemExit("No JSONL rows found.")

    # Group rows by (gt_domains, generated_domains)
    grouped = defaultdict(list)
    for r in all_rows:
        gt = int(r.get("gt_domains", -1))
        gen = int(r.get("generated_domains", -1))
        grouped[(gt, gen)].append(r)

    # 1) Trajectory plots per (gt, gen)
    for (gt, gen), rows in grouped.items():
        # method -> list of curves across rows
        method_to_test = defaultdict(list)
        method_to_train = defaultdict(list)
        method_to_baselines = defaultdict(list)

        for r in rows:
            methods = r.get("methods", {})
            for m, payload in methods.items():
                if args.only_methods is not None and m not in args.only_methods:
                    continue
                tc = payload.get("test_curve", None)
                tr = payload.get("train_curve", None)
                if isinstance(tc, list) and len(tc) > 0:
                    method_to_test[m].append(tc)
                if isinstance(tr, list) and len(tr) > 0:
                    method_to_train[m].append(tr)

                # baselines (optional)
                b = extract_scalar_baseline(payload, "st_curve")
                if b is not None:
                    method_to_baselines[m].append(b)
                b2 = extract_scalar_baseline(payload, "st_all_curve")
                if b2 is not None:
                    method_to_baselines[m].append(b2)

        if not method_to_test:
            continue

        fig, ax = plt.subplots()
        for m in sorted(method_to_test.keys()):
            plot_curve_bundle(ax, method_to_test[m], m)

        # baseline lines (draw mean baseline per method if present)
        for m, vals in method_to_baselines.items():
            if len(vals) > 0:
                ax.axhline(float(np.mean(vals)), linestyle="--", alpha=0.5)

        ax.set_xlabel("adaptation step")
        ax.set_ylabel("test accuracy")
        ax.set_title(f"Trajectories | Gobs={gt} | Gsyn={gen}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, sanitize(f"traj_gt{gt}_gen{gen}.png")), dpi=200)
        plt.close(fig)

    # 2) Final accuracy vs generated_domains (aggregate view)
    if args.aggregate:
        # method -> gen -> list of final acc
        method_gen_final = defaultdict(lambda: defaultdict(list))
        for r in all_rows:
            gen = int(r.get("generated_domains", -1))
            methods = r.get("methods", {})
            for m, payload in methods.items():
                if args.only_methods is not None and m not in args.only_methods:
                    continue
                tc = payload.get("test_curve", None)
                if isinstance(tc, list) and len(tc) > 0:
                    method_gen_final[m][gen].append(float(tc[-1]))

        gens = sorted({g for m in method_gen_final.values() for g in m.keys()})
        if gens:
            fig, ax = plt.subplots()
            for m in sorted(method_gen_final.keys()):
                xs, ys, es = [], [], []
                for g in gens:
                    vals = method_gen_final[m].get(g, [])
                    if not vals:
                        continue
                    xs.append(g)
                    ys.append(float(np.mean(vals)))
                    es.append(float(np.std(vals)))
                if xs:
                    ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=m)
            ax.set_xlabel("generated_domains (Gsyn)")
            ax.set_ylabel("final target accuracy")
            ax.set_title("Final accuracy vs Gsyn (mean±std across logs)")
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(args.out_dir, "final_vs_gsyn.png"), dpi=200)
            plt.close(fig)

    print(f"Saved plots to: {args.out_dir}")

if __name__ == "__main__":
    main()
