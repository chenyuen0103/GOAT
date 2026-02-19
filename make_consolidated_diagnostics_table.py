#!/usr/bin/env python3
"""Build a consolidated diagnostics LaTeX table from curves/drift JSONL logs.

Columns (method-independent + method outcomes):
- pseudo-label accuracy (from em_acc in curves logs)
- mean conditional drift (from drift logs, selected class-wise method)
- worst conditional drift (from drift logs, selected class-wise method)
- GOAT final target acc
- GOAT-CW final target acc
- GOAT-CW-Oracle final target acc (if available)
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

KNOWN_DATASETS = {"mnist", "color_mnist", "portraits", "covtype"}


def _is_finite(x: Any) -> bool:
    try:
        v = float(x)
    except Exception:
        return False
    return math.isfinite(v)


def _mean(vals: Iterable[float]) -> Optional[float]:
    vals = list(vals)
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _std(vals: Iterable[float]) -> Optional[float]:
    vals = list(vals)
    if len(vals) <= 1:
        return 0.0 if vals else None
    mu = sum(vals) / len(vals)
    var = sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)
    return float(var ** 0.5)


def _fmt_mean_std(vals: List[float], nd: int = 2) -> str:
    if not vals:
        return r"\textemdash"
    mu = _mean(vals)
    sd = _std(vals)
    return f"{mu:.{nd}f} $\\pm$ {sd:.{nd}f}"


def _fmt_scalar(v: Optional[float], nd: int = 4) -> str:
    if v is None or not math.isfinite(float(v)):
        return r"\textemdash"
    return f"{float(v):.{nd}f}"


def _extract_dataset_from_path(path: Path) -> str:
    parts = list(path.parts)
    for i, p in enumerate(parts):
        if p == "logs" and i + 1 < len(parts):
            nxt = parts[i + 1]
            if nxt in KNOWN_DATASETS:
                return nxt
    # fallback heuristics
    s = str(path)
    for d in KNOWN_DATASETS:
        if f"/{d}/" in s:
            return d
    return "mnist"


def _extract_target(path: Path, obj: Dict[str, Any]) -> Optional[int]:
    if "target" in obj:
        try:
            return int(obj["target"])
        except Exception:
            pass
    m = re.search(r"target(\d+)", str(path))
    if m:
        return int(m.group(1))
    return None


def _group_key(dataset: str, target: Optional[int], group_by: str) -> str:
    if group_by == "dataset":
        return dataset
    if dataset == "mnist" and target is not None:
        return f"mnist-t{target}"
    return dataset


def _find_first_finite(payloads: Dict[str, Any], keys: List[str]) -> Optional[float]:
    for k in keys:
        v = payloads.get(k)
        if _is_finite(v):
            return float(v)
    return None


def _pick_method(methods: Dict[str, Dict[str, Any]], candidates: List[str], contains: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
    # exact (case-insensitive)
    lower_map = {k.lower(): v for k, v in methods.items()}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    # fuzzy contains
    if contains:
        for k, v in methods.items():
            lk = k.lower()
            if all(tok in lk for tok in contains):
                return v
    return None


def _final_from_curve(method_payload: Optional[Dict[str, Any]]) -> Optional[float]:
    if not method_payload:
        return None
    tc = method_payload.get("test_curve")
    if isinstance(tc, list) and tc:
        v = tc[-1]
        if _is_finite(v):
            return float(v)
    return None


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves-glob", nargs="+", default=["logs/**/*_curves.jsonl"])
    ap.add_argument("--drift-glob", nargs="+", default=["logs/**/*_drift.jsonl"])
    ap.add_argument("--group-by", choices=["dataset", "setting"], default="dataset")
    ap.add_argument("--gt-domains", type=int, default=0)
    ap.add_argument("--generated-domains", type=int, default=3)
    ap.add_argument("--drift-method", default="cc_wass", help="Which drift record method to use for cond. drift columns.")
    ap.add_argument("--output", default="tables_consolidated_diagnostics.tex")
    ap.add_argument("--caption", default="")
    ap.add_argument("--label", default="tab:consolidated-diagnostics")
    args = ap.parse_args()

    curve_files: List[Path] = []
    for g in args.curves_glob:
        curve_files.extend(Path(p) for p in glob.glob(g, recursive=True))
    curve_files = sorted(set(curve_files))

    drift_files: List[Path] = []
    for g in args.drift_glob:
        drift_files.extend(Path(p) for p in glob.glob(g, recursive=True))
    drift_files = sorted(set(drift_files))

    stats: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    # -------- curves --------
    for fp in curve_files:
        try:
            rows = read_jsonl(fp)
        except Exception:
            continue
        for r in rows:
            gt = r.get("gt_domains")
            gen = r.get("generated_domains")
            if gt is not None and int(gt) != args.gt_domains:
                continue
            if gen is not None and int(gen) != args.generated_domains:
                continue

            ds = _extract_dataset_from_path(fp)
            target = _extract_target(fp, r)
            key = _group_key(ds, target, args.group_by)

            methods = r.get("methods", {}) if isinstance(r.get("methods", {}), dict) else {}

            goat = _pick_method(methods, ["goat"])
            goat_cw = _pick_method(methods, ["goat_classwise", "goat-cw", "goat_cw"], contains=["goat", "class"])
            oracle_cw = _pick_method(methods, ["goat_classwise_oracle", "goat-cw-oracle", "goat_cw_oracle"], contains=["oracle", "class"])

            v_goat = _final_from_curve(goat)
            v_cw = _final_from_curve(goat_cw)
            v_oracle = _final_from_curve(oracle_cw)

            if _is_finite(v_goat):
                stats[key]["goat_final"].append(float(v_goat))
            if _is_finite(v_cw):
                stats[key]["goat_cw_final"].append(float(v_cw))
            if _is_finite(v_oracle):
                stats[key]["goat_cw_oracle_final"].append(float(v_oracle))

            # pseudo-label / EM accuracy: prefer ours_fr.em_acc, else first finite em_acc
            em_v = None
            ours_fr = _pick_method(methods, ["ours_fr", "ours-fr"])
            if ours_fr is not None:
                em_v = _find_first_finite(ours_fr, ["em_acc"])
            if em_v is None:
                for p in methods.values():
                    if isinstance(p, dict):
                        em_v = _find_first_finite(p, ["em_acc"])
                        if em_v is not None:
                            break
            if _is_finite(em_v):
                stats[key]["pseudo_acc"].append(100.0 * float(em_v))

    # -------- drift --------
    for fp in drift_files:
        try:
            rows = read_jsonl(fp)
        except Exception:
            continue
        for r in rows:
            try:
                if int(r.get("gt_domains", args.gt_domains)) != args.gt_domains:
                    continue
                if int(r.get("generated_domains", args.generated_domains)) != args.generated_domains:
                    continue
            except Exception:
                continue

            if str(r.get("method", "")) != str(args.drift_method):
                continue

            ds = str(r.get("dataset") or _extract_dataset_from_path(fp))
            target = _extract_target(fp, r)
            key = _group_key(ds, target, args.group_by)

            drift = r.get("drift", {}) if isinstance(r.get("drift", {}), dict) else {}
            mean_cond = [float(v) for v in drift.get("mean_cond_drift", []) if _is_finite(v)]
            worst_cond = [float(v) for v in drift.get("worst_cond_drift", []) if _is_finite(v)]
            if mean_cond:
                stats[key]["mean_cond_avg"].append(float(sum(mean_cond) / len(mean_cond)))
            if worst_cond:
                stats[key]["worst_cond_max"].append(float(max(worst_cond)))

    keys = sorted(stats.keys())
    if not keys:
        raise SystemExit("No matching records found for the requested filters.")

    caption = args.caption
    if not caption:
        caption = (
            "Consolidated diagnostics by dataset/setting. "
            "Pseudo-label accuracy is reported as mean$\\pm$std (\\%). "
            "Conditional drift metrics are computed from the selected drift method "
            f"(\\texttt{{{args.drift_method}}}) and reported as means over matched runs. "
            "Final accuracies are the last values of test curves."
        )

    lines: List[str] = []
    lines.append("% Auto-generated by make_consolidated_diagnostics_table.py")
    lines.append("% Requires: \\usepackage{booktabs}")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{" + caption + "}")
    lines.append(r"\label{" + args.label + "}")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.08}")
    lines.append(r"\begin{tabular}{@{}lcccccc@{}}")
    lines.append(r"\toprule")
    lines.append(r"Setting & Pseudo-Acc ($\%$) & Mean cond. drift & Worst cond. drift & GOAT & GOAT-CW & GOAT-CW-Oracle \\")
    lines.append(r"\midrule")

    for k in keys:
        row = stats[k]
        lines.append(
            "{} & {} & {} & {} & {} & {} & {} \\\\".format(
                k.replace("_", r"\_"),
                _fmt_mean_std(row.get("pseudo_acc", []), nd=2),
                _fmt_scalar(_mean(row.get("mean_cond_avg", [])), nd=4),
                _fmt_scalar(_mean(row.get("worst_cond_max", [])), nd=4),
                _fmt_mean_std(row.get("goat_final", []), nd=2),
                _fmt_mean_std(row.get("goat_cw_final", []), nd=2),
                _fmt_mean_std(row.get("goat_cw_oracle_final", []), nd=2),
            )
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    out = Path(args.output)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote: {out}")
    print(f"Rows: {len(keys)}")


if __name__ == "__main__":
    main()
