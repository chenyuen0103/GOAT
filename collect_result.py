import os
import re
import argparse
from typing import Optional, Dict, List, Tuple

os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.makedirs("/tmp/matplotlib", exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker

# ----------------------------
# CONFIG
# ----------------------------
DEFAULT_LOG_BASES = ["logs","logs_rerun"]  # search these roots for logs, in order. If a run appears in multiple, keep the most recent one.

DATASETS = {
    "rotated_mnist": {"log_dir": "mnist", "has_target": True},
    "colored_mnist": {"log_dir": "color_mnist", "has_target": False},
    "portraits": {"log_dir": "portraits", "has_target": False},
    "covtype": {"log_dir": "covtype", "has_target": False},
}

# Main text: hardest regime and selected degrees for Rotated MNIST
MAIN_GT = 0
MAIN_GEN_ORDER = [0, 1, 2, 3]
ROT_MNIST_MAIN_DEGREES = [30, 45, 90]

# Parse gt/gen from filename: test_acc_dim2048_int3_gen2_....
NAME_RE = re.compile(r"test_acc_dim\d+_int(\d+)_gen(\d+)_")
FULL_NAME_RE = re.compile(
    r"^test_acc_dim(?P<dim>\d+)_int(?P<gt>\d+)_gen(?P<gen>\d+)(?:_(?P<suffix>.*))?$",
    re.IGNORECASE,
)
SEED_DIR_RE = re.compile(r"^s(\d+)$", re.IGNORECASE)
TARGET_DIR_RE = re.compile(r"^target(\d+)$", re.IGNORECASE)

# Parse method:value pairs like "OursFR:66.3"
PAIR_RE = re.compile(r"([A-Za-z][A-Za-z0-9_]*)\s*:\s*([0-9]*\.?[0-9]+)")

METHODS: List[Tuple[str, str]] = [
    ("goat", "GOAT"),
    ("goatcw", "C2GDA-Wass"),
    ("eta", "C2GDA-Nat"),
    ("ours_fr", "C2GDA-FR"),
]

# Match notebook selection logic for reporting/plots:
# (~em_ensemble) & em_match contains "prototype" & em_select contains "bic",
# then pick best setting per (dataset, degree, gt, gen, method) by acc_mean.
REPORT_EM_ENSEMBLE = False
REPORT_EM_MATCH_SUBSTR = "prototype"
REPORT_EM_SELECT_SUBSTR = "bic"

APPENDIX_COMPARISON_GROUPS: List[Dict[str, object]] = [
    {
        "name": "goat_vs_wass",
        "title": "GOAT vs C2GDA-Wass (marginal vs conditional interpolation)",
        "label_suffix": "goat-vs-wass",
        "methods": [("goat", "GOAT"), ("goatcw", "C2GDA-Wass")],
    },
    {
        "name": "c2gda_variants",
        "title": "C2GDA variants (conditional interpolation family)",
        "label_suffix": "c2gda-variants",
        "methods": [("goatcw", "C2GDA-Wass"), ("eta", "C2GDA-Nat"), ("ours_fr", "C2GDA-FR")],
    },
]

MAIN_PLOT_FAMILIES: List[Dict[str, object]] = [
    {
        "name": "goat_vs_wass",
        "title": "GOAT vs C2GDA-Wass",
        "methods": [("goat", "GOAT"), ("goatcw", "C2GDA-Wass")],
        "outfile": "main_plot_goat_vs_c2gda_wass",
    },
    {
        "name": "c2gda_variants",
        "title": "C2GDA variants",
        "methods": [("goatcw", "C2GDA-Wass"), ("eta", "C2GDA-Nat"), ("ours_fr", "C2GDA-FR")],
        "outfile": "main_plot_c2gda_variants",
    },
]

PLOT_STYLE = {
    "goat": dict(color="#1f77b4", marker="o", linestyle="-"),
    "goatcw": dict(color="#d62728", marker="s", linestyle="-"),
    "eta": dict(color="#2ca02c", marker="^", linestyle="-"),
    "ours_fr": dict(color="#000000", marker="D", linestyle="-"),
}

# For gen=0 baseline, which method column to use if multiple exist
GEN0_BASELINE_PREFERENCE = ["eta", "ours_fr", "goat", "goatcw"]

# ---- Layout settings for the 2-col x 3-row composite tables ----
PANEL_COL_WIDTH = "0.49\\textwidth"
PANEL_GAP = "0.02\\textwidth"
TITLE_BOX_HEIGHT = "1.6em"  # fixed title height so panels align vertically
ROW_VGAP = "0.8em"          # vertical gap between rows of panels

# Match your template
TABCOLSEP_MAIN = 2          # (you showed 2pt in the template)
ARRAYSTRETCH_MAIN = 1.05

# Appendix: same grid layout, but one table* per GT
APPENDIX_TABCOLSEP = 2
APPENDIX_ARRAYSTRETCH = 1.05


# ----------------------------
# HELPERS
# ----------------------------
def _extract_seed_from_parts(parts: List[str]) -> Optional[int]:
    for p in parts:
        m = SEED_DIR_RE.match(p)
        if m:
            return int(m.group(1))
    return None


def _extract_target_degree_from_parts(parts: List[str]) -> Optional[int]:
    for p in parts:
        m = TARGET_DIR_RE.match(p)
        if m:
            return int(m.group(1))
    return None


def _mean_std(vals: pd.Series) -> Tuple[float, float, int]:
    vals = vals.dropna()
    n = int(len(vals))
    if n == 0:
        return (np.nan, np.nan, 0)
    mean = float(vals.mean())
    std = float(vals.std(ddof=1)) if n > 1 else 0.0
    return (mean, std, n)


def _format_mean_std(vals: pd.Series) -> str:
    """
    Format exactly like your template: $mean{\scriptstyle\pm std}$.
    """
    mean, std, n = _mean_std(vals)
    if n == 0 or np.isnan(mean):
        return "--"
    return f"${mean:.1f}{{\\scriptstyle\\pm {std:.1f}}}$"


def _render_cell(vals: pd.Series) -> Tuple[str, float, int]:
    """
    Returns (cell_string, score_for_bolding, n).
    score is the mean (higher is better).
    """
    vals = vals.dropna()
    n = int(len(vals))
    if n == 0:
        return ("--", -np.inf, 0)
    mean = float(vals.mean())
    std = float(vals.std(ddof=1)) if n > 1 else 0.0
    cell = f"${mean:.1f}{{\\scriptstyle\\pm {std:.1f}}}$"
    return (cell, mean, n)


def _bold_best_row(cells: List[Tuple[str, float]]) -> List[str]:
    """
    Bold all entries achieving the best score (ties allowed).
    IMPORTANT: Use \\boldmath so math ($...$) truly bolds, including \\pm term.
    """
    scores = [s for _c, s in cells]
    best = max(scores) if scores else -np.inf
    out: List[str] = []
    for c, s in cells:
        if s == best and s != -np.inf and c != "--":
            out.append("{\\boldmath " + c + "}")
        else:
            out.append(c)
    return out


def _latex_preamble_comment() -> str:
    return (
        "% Auto-generated tables.\n"
        "% Required packages (in your main.tex):\n"
        "% \\usepackage{booktabs}\n"
        "% \\usepackage{multirow}\n"
        "% \\usepackage{caption}\n"
    )


def _gt_caption_phrase(gt: int) -> str:
    # You can reword this to match your paper style.
    if gt == 0:
        return "when no intermediate domains are given"
    if gt == 1:
        return "when 1 intermediate domain is given"
    return f"when {gt} intermediate domains are given"


# ----------------------------
# DATA COLLECTION
# ----------------------------
def _parse_filename_settings(fname: str) -> Optional[Dict[str, object]]:
    stem, _ext = os.path.splitext(fname)
    m = FULL_NAME_RE.match(stem)
    if not m:
        return None

    suffix = m.group("suffix") or ""
    tokens = [t for t in suffix.split("_") if t] if suffix else []
    em_ensemble = any(t.lower() in {"em-ensemble", "emensemble"} for t in tokens)

    return {
        "dim": int(m.group("dim")),
        "gt": int(m.group("gt")),
        "gen": int(m.group("gen")),
        "setting_suffix": suffix if suffix else None,
        "suffix_tokens": "|".join(tokens) if tokens else None,
        "label_source": tokens[0] if len(tokens) > 0 else None,
        "em_match": tokens[1] if len(tokens) > 1 else None,
        "em_select": tokens[2] if len(tokens) > 2 else None,
        "em_ensemble": em_ensemble,
        "extra_tokens": "|".join(tokens[3:]) if len(tokens) > 3 else None,
    }


def _to_table_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()

    # When gen=0, the updated experiment runner may only emit a single GOAT entry.
    # For reporting, we want the "gen=0 baseline" row to be identical across all methods.
    if "gen" in out.columns:
        gen0 = out["gen"] == 0
        if gen0.any():
            baseline = None
            # Prefer GOAT if present; otherwise fall back to any available method.
            for col in ["goat", "eta", "ours_fr", "goatcw"]:
                if col in out.columns:
                    s = out.loc[gen0, col].dropna()
                    if len(s) > 0:
                        baseline = out.loc[gen0, col]
                        break
            if baseline is not None:
                for col in ["goat", "goatcw", "eta", "ours_fr"]:
                    if col in out.columns:
                        out.loc[gen0, col] = out.loc[gen0, col].fillna(baseline)

    keep = ["dataset", "seed", "degree", "gt", "gen"] + [c for c, _ in METHODS]
    return out[keep]


def build_selected_seed_long(df_detailed: pd.DataFrame) -> pd.DataFrame:
    """
    Build seed-level long table after applying the same filtering/ranking policy
    used in analyze_results_summary.ipynb:
      1) filter settings by em_ensemble/em_match/em_select
      2) compute mean accuracy per setting and method
      3) pick best setting per (dataset, degree, gt, gen, method)
      4) keep seed-level runs only from those selected settings
    Output columns include:
      [dataset, seed, degree, gt, gen, method, acc, label_source, em_match,
       em_select, em_ensemble, file_path]
    """
    if df_detailed.empty:
        return df_detailed

    id_cols = [
        "dataset",
        "seed",
        "degree",
        "gt",
        "gen",
        "label_source",
        "em_match",
        "em_select",
        "em_ensemble",
        "file_path",
    ]
    val_cols = [c for c, _ in METHODS]

    long_df = df_detailed[id_cols + val_cols].melt(
        id_vars=id_cols,
        value_vars=val_cols,
        var_name="method",
        value_name="acc",
    )
    long_df = long_df.dropna(subset=["acc"]).copy()

    setting_keys = [
        "dataset",
        "degree",
        "gt",
        "gen",
        "label_source",
        "em_match",
        "em_select",
        "em_ensemble",
        "method",
    ]
    setting_summary = (
        long_df.groupby(setting_keys, dropna=False)["acc"]
        .agg(acc_mean="mean", acc_std="std", n_rows="size")
        .reset_index()
    )

    filt = (
        (setting_summary["em_ensemble"] == REPORT_EM_ENSEMBLE)
        & setting_summary["em_match"].astype(str).str.contains(REPORT_EM_MATCH_SUBSTR, case=False, na=False)
        & setting_summary["em_select"].astype(str).str.contains(REPORT_EM_SELECT_SUBSTR, case=False, na=False)
    )
    cand = setting_summary[filt].copy()
    if cand.empty:
        return long_df.iloc[0:0].copy()

    rank_df = cand.sort_values(
        ["dataset", "degree", "gt", "gen", "method", "acc_mean"],
        ascending=[True, True, True, True, True, False],
        na_position="last",
    )
    best = rank_df.groupby(["dataset", "degree", "gt", "gen", "method"], dropna=False).head(1).copy()

    selected = long_df.merge(best[setting_keys], on=setting_keys, how="inner")
    return selected


def _dedupe_prefer_latest(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    # Same logical run may appear in multiple log roots (e.g., logs + logs_rerun).
    # Keep the most recently modified file as the canonical record.
    dedupe_keys = [
        "dataset",
        "seed",
        "degree",
        "dim",
        "gt",
        "gen",
        "setting_suffix",
        "suffix_tokens",
        "label_source",
        "em_match",
        "em_select",
        "em_ensemble",
        "extra_tokens",
    ]
    return (
        df.sort_values(by=["file_mtime", "file_path"], ascending=[False, False], na_position="last")
        .drop_duplicates(subset=dedupe_keys, keep="first")
        .reset_index(drop=True)
    )


def collect_results_detailed(dataset_name: str, log_bases: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Collect accuracies into a DataFrame with columns:
    [dataset, seed, degree, gt, gen, dim, file_path, setting fields, goat, goatcw, eta, ours_fr].

    For datasets with target directories (rotated_mnist), degree is parsed
    from the directory name target{deg}. Otherwise degree is None.
    """
    config = DATASETS.get(dataset_name)
    if config is None:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    roots = log_bases if log_bases else DEFAULT_LOG_BASES
    base_paths = [os.path.join(root, config["log_dir"]) for root in roots]
    existing_paths = [p for p in base_paths if os.path.isdir(p)]
    if not existing_paths:
        raise FileNotFoundError(
            f"No log directory found for dataset={dataset_name}. Tried: {base_paths}"
        )

    rows = []

    for base_path in existing_paths:
        for root, _dirs, files in os.walk(base_path):
            parts = root.split(os.sep)

            seed = _extract_seed_from_parts(parts)
            if seed is None:
                continue

            degree = _extract_target_degree_from_parts(parts) if config["has_target"] else None

            for fname in files:
                if not fname.endswith(".txt"):
                    continue

                parsed = _parse_filename_settings(fname)
                if parsed is None:
                    continue
                # Ignore non-canonical variants (e.g., *_old, *_copy, etc.).
                # Keep only the expected setting triplet:
                #   label_source_em_match_em_select[_em-ensemble]
                if parsed.get("extra_tokens") is not None:
                    continue

                fp = os.path.join(root, fname)
                last_parts: Dict[str, float] = {}

                try:
                    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            pairs = PAIR_RE.findall(line)
                            if not pairs:
                                continue
                            for name, val in pairs:
                                last_parts[name.strip().lower()] = float(val)
                except OSError:
                    continue

                if not last_parts:
                    continue

                rows.append(
                    dict(
                        dataset=dataset_name,
                        log_dataset_dir=config["log_dir"],
                        log_root=os.path.normpath(base_path),
                        seed=seed,
                        degree=degree,
                        dim=parsed["dim"],
                        gt=parsed["gt"],
                        gen=parsed["gen"],
                        setting_suffix=parsed["setting_suffix"],
                        suffix_tokens=parsed["suffix_tokens"],
                        label_source=parsed["label_source"],
                        em_match=parsed["em_match"],
                        em_select=parsed["em_select"],
                        em_ensemble=parsed["em_ensemble"],
                        extra_tokens=parsed["extra_tokens"],
                        file_name=fname,
                        file_path=fp,
                        file_mtime=int(os.path.getmtime(fp)),
                        goat=last_parts.get("goat"),
                        goatcw=last_parts.get("goatcw"),
                        eta=last_parts.get("eta"),
                        ours_fr=last_parts.get("oursfr")
                        if "oursfr" in last_parts
                        else last_parts.get("ours_fr"),
                    )
                )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return _dedupe_prefer_latest(df)


def collect_results(dataset_name: str, log_bases: Optional[List[str]] = None) -> pd.DataFrame:
    detailed = collect_results_detailed(dataset_name, log_bases=log_bases)
    if detailed.empty:
        return detailed
    return _to_table_df(detailed)


def build_results_summary(df_detailed: pd.DataFrame) -> pd.DataFrame:
    if df_detailed.empty:
        return df_detailed

    group_cols = [
        "dataset",
        "degree",
        "gt",
        "gen",
        "label_source",
        "em_match",
        "em_select",
        "em_ensemble",
    ]
    agg: Dict[str, Tuple[str, str]] = {
        "n_rows": ("file_path", "size"),
        "n_files": ("file_path", "nunique"),
        "n_seeds": ("seed", "nunique"),
    }
    for col, _ in METHODS:
        agg[f"{col}_mean"] = (col, "mean")
        agg[f"{col}_std"] = (col, "std")
        agg[f"{col}_count"] = (col, "count")

    out = df_detailed.groupby(group_cols, dropna=False).agg(**agg).reset_index()
    seed_lists = (
        df_detailed.groupby(group_cols, dropna=False)["seed"]
        .apply(lambda s: "|".join(str(int(v)) for v in sorted(set(s.dropna().astype(int).tolist()))))
        .reset_index(name="seeds")
    )
    out = out.merge(seed_lists, on=group_cols, how="left")
    out = out.sort_values(
        by=["dataset", "degree", "gt", "gen", "label_source", "em_match", "em_select", "em_ensemble"],
        na_position="last",
    ).reset_index(drop=True)
    return out


# ----------------------------
# GEN=0 BASELINE NORMALIZATION
# ----------------------------
def _baseline_series_for_gen0(sub_df: pd.DataFrame) -> pd.Series:
    """
    Pick a baseline run-series for gen=0 from available columns using preference.
    Returns a Series of per-run values.
    """
    for col in GEN0_BASELINE_PREFERENCE:
        if col in sub_df.columns:
            s = sub_df[col].dropna()
            if len(s) > 0:
                return s
    return pd.Series(dtype=float)


def _emit_gen0_row_same_for_all_methods(sg: pd.DataFrame, methods: List[Tuple[str, str]]) -> List[str]:
    """
    Return row cells for gen=0 where all method entries are identical.
    """
    base = _baseline_series_for_gen0(sg)
    cell = _format_mean_std(base)
    return [cell] * len(methods)


# ----------------------------
# PANEL RENDERER (shared by main + appendix)
# ----------------------------
def _panel_minipage(
    block_title: str,
    sub_df: pd.DataFrame,
    gens: List[int],
    methods: List[Tuple[str, str]],
    tabcolsep_pt: int,
    arraystretch: float,
    gen_header: str = "\\# gen.",
) -> str:
    r"""
    One panel as a minipage containing a fixed-height title and a width-filled tabular*.
    Always includes the leftmost "# gen." column (c|cccc), matching your template.
    """
    lines: List[str] = []
    lines.append(f"\\begin{{minipage}}[t]{{{PANEL_COL_WIDTH}}}")
    lines.append("\\vspace{0pt}")
    lines.append("\\centering")
    lines.append(
        f"\\parbox[t][{TITLE_BOX_HEIGHT}][c]{{\\linewidth}}"
        f"{{\\centering\\textbf{{{block_title}}}}}\\\\[-0.25em]"
    )

    lines.append("\\setlength{\\tabcolsep}{" + str(tabcolsep_pt) + "pt}")
    lines.append("\\renewcommand{\\arraystretch}{" + str(arraystretch) + "}")

    method_cols = "c" * len(methods)
    colspec = "@{\\extracolsep{\\fill}}c@{\\hspace{6pt}}" + method_cols
    header = [gen_header] + [lab for _col, lab in methods]

    lines.append(f"\\begin{{tabular*}}{{\\linewidth}}{{{colspec}}}")
    lines.append("\\toprule")
    is_main_template_header = (
        len(methods) == 4
        and methods[0][0] == "goat"
        and methods[1][0] == "goatcw"
        and methods[2][0] == "eta"
        and methods[3][0] == "ours_fr"
    )
    if is_main_template_header:
        lines.append("\\multirow{2}{*}{$G_{\\text{syn}}$} & \\multirow{2}{*}{GOAT} & \\multicolumn{3}{c}{C2GDA} \\\\ \\cmidrule(lr){3-5} & & Wass & Nat & FR \\\\")
    else:
        lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")

    if sub_df.empty:
        lines.append(f"\\multicolumn{{{len(methods) + 1}}}{{c}}{{--}} \\\\")
        lines.append("\\bottomrule")
        lines.append("\\end{tabular*}")
        lines.append("\\end{minipage}")
        return "\n".join(lines)

    for gen in gens:
        sg = sub_df[sub_df["gen"] == gen]
        row = [str(gen)]

        if gen == 0:
            base = pd.Series(dtype=float)
            for pref in GEN0_BASELINE_PREFERENCE:
                s_pref = sg[sg["method"] == pref]["acc"].dropna()
                if len(s_pref) > 0:
                    base = s_pref
                    break
            cell = _format_mean_std(base)
            row += [cell] * len(methods)
            lines.append(" & ".join(row) + " \\\\")
            continue

        scored: List[Tuple[str, float]] = []
        for col, _lab in methods:
            vals = sg[sg["method"] == col]["acc"].dropna()
            cell, score, _n = _render_cell(vals)
            scored.append((cell, score))
        row += _bold_best_row(scored)

        lines.append(" & ".join(row) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular*}")
    lines.append("\\end{minipage}")
    return "\n".join(lines)


# ----------------------------
# MAIN TABLE (GT = MAIN_GT)
# ----------------------------
def make_main_table_gt0_grid_2x3(df_all_long: pd.DataFrame) -> str:
    r"""
    Main-text composite table, arranged as 2 columns x 3 rows for GT=MAIN_GT.
    """
    if df_all_long.empty:
        return "% No data found.\n"

    df = df_all_long[df_all_long["gt"] == MAIN_GT].copy()
    if df.empty:
        return f"% No data for G_gt={MAIN_GT}.\n"

    existing_gens = sorted(df["gen"].unique())
    gens = [g for g in MAIN_GEN_ORDER if g in existing_gens] + [
        g for g in existing_gens if g not in MAIN_GEN_ORDER
    ]

    rot = df[df["dataset"] == "rotated_mnist"]
    rot_panels: Dict[int, pd.DataFrame] = {deg: rot[rot["degree"] == deg] for deg in ROT_MNIST_MAIN_DEGREES}
    portraits = df[df["dataset"] == "portraits"]
    covtype = df[df["dataset"] == "covtype"]
    color_mnist = df[df["dataset"] == "colored_mnist"]

    p11 = _panel_minipage("Rotated MNIST (degree $=30^\\circ$)", rot_panels.get(30, pd.DataFrame()), gens, METHODS,
                          TABCOLSEP_MAIN, ARRAYSTRETCH_MAIN)
    p12 = _panel_minipage("Rotated MNIST (degree $=45^\\circ$)", rot_panels.get(45, pd.DataFrame()), gens, METHODS,
                          TABCOLSEP_MAIN, ARRAYSTRETCH_MAIN)
    p21 = _panel_minipage("Rotated MNIST (degree $=90^\\circ$)", rot_panels.get(90, pd.DataFrame()), gens, METHODS,
                          TABCOLSEP_MAIN, ARRAYSTRETCH_MAIN)
    p22 = _panel_minipage("Portraits", portraits, gens, METHODS, TABCOLSEP_MAIN, ARRAYSTRETCH_MAIN)
    p31 = _panel_minipage("Covtype", covtype, gens, METHODS, TABCOLSEP_MAIN, ARRAYSTRETCH_MAIN)
    p32 = _panel_minipage("Color MNIST", color_mnist, gens, METHODS, TABCOLSEP_MAIN, ARRAYSTRETCH_MAIN)

    lines: List[str] = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\footnotesize")
    lines.append("\\centering")
    lines.append(
        "\\caption{Target accuracy (\\%) in the hardest regime ($G_{\\text{obs}}=0$), "
        "varying the number of generated intermediate domains $G_{\\text{syn}}$.}"
    )
    lines.append("\\label{tab:main-gt0-gen-sweep-grid}")
    lines.append("% \\resizebox{\\textwidth}{!}{%")

    lines.append("\\setlength{\\tabcolsep}{0pt}")
    lines.append("\\renewcommand{\\arraystretch}{1.0}")
    lines.append(
        "\\begin{tabular*}{\\textwidth}{@{}c@{\\hspace{0.03\\textwidth}}c@{}}"
    )
    lines.append(p11 + " & " + p12 + " \\\\[" + ROW_VGAP + "]")
    lines.append(p21 + " & " + p22 + " \\\\[" + ROW_VGAP + "]")
    lines.append(p31 + " & " + p32 + " \\\\")
    lines.append("\\end{tabular*}")
    lines.append("% }")
    lines.append("\\end{table*}")
    return "\n".join(lines)


# ----------------------------
# APPENDIX TABLES: ONE 2x3 GRID PER G_gt
# ----------------------------
def make_appendix_tables_by_gt_2x3(df_all_long: pd.DataFrame) -> str:
    r"""
    Appendix: for each G_gt value, produce ONE table* that has the same 2x3 panel layout
    as the main table (RotMNIST 30/45/90, Portraits, Covtype, ColorMNIST), with rows over #gen.
    """
    if df_all_long.empty:
        return "% No appendix tables (no data).\n"

    chunks: List[str] = []
    gt_vals = sorted(df_all_long["gt"].dropna().unique())
    for gt in gt_vals:
        df = df_all_long[df_all_long["gt"] == gt].copy()
        if df.empty:
            continue

        existing_gens = sorted(df["gen"].unique())
        gens = [g for g in MAIN_GEN_ORDER if g in existing_gens] + [
            g for g in existing_gens if g not in MAIN_GEN_ORDER
        ]

        rot = df[df["dataset"] == "rotated_mnist"]
        rot_panels: Dict[int, pd.DataFrame] = {deg: rot[rot["degree"] == deg] for deg in ROT_MNIST_MAIN_DEGREES}
        portraits = df[df["dataset"] == "portraits"]
        covtype = df[df["dataset"] == "covtype"]
        color_mnist = df[df["dataset"] == "colored_mnist"]

        for group in APPENDIX_COMPARISON_GROUPS:
            methods = group["methods"]  # type: ignore[assignment]
            group_title = str(group["title"])
            label_suffix = str(group["label_suffix"])

            p11 = _panel_minipage(
                "Rotated MNIST (degree $=30^\\circ$)",
                rot_panels.get(30, pd.DataFrame()),
                gens,
                methods,  # type: ignore[arg-type]
                APPENDIX_TABCOLSEP,
                APPENDIX_ARRAYSTRETCH,
            )
            p12 = _panel_minipage(
                "Rotated MNIST (degree $=45^\\circ$)",
                rot_panels.get(45, pd.DataFrame()),
                gens,
                methods,  # type: ignore[arg-type]
                APPENDIX_TABCOLSEP,
                APPENDIX_ARRAYSTRETCH,
            )
            p21 = _panel_minipage(
                "Rotated MNIST (degree $=90^\\circ$)",
                rot_panels.get(90, pd.DataFrame()),
                gens,
                methods,  # type: ignore[arg-type]
                APPENDIX_TABCOLSEP,
                APPENDIX_ARRAYSTRETCH,
            )
            p22 = _panel_minipage(
                "Portraits",
                portraits,
                gens,
                methods,  # type: ignore[arg-type]
                APPENDIX_TABCOLSEP,
                APPENDIX_ARRAYSTRETCH,
            )
            p31 = _panel_minipage(
                "Covtype",
                covtype,
                gens,
                methods,  # type: ignore[arg-type]
                APPENDIX_TABCOLSEP,
                APPENDIX_ARRAYSTRETCH,
            )
            p32 = _panel_minipage(
                "Color MNIST",
                color_mnist,
                gens,
                methods,  # type: ignore[arg-type]
                APPENDIX_TABCOLSEP,
                APPENDIX_ARRAYSTRETCH,
            )

            chunks.append("% ----------------------------")
            chunks.append(f"% Appendix grid for G_gt = {int(gt)} ({group_title})")
            chunks.append("% ----------------------------")
            chunks.append("\\begin{table*}[t]")
            chunks.append("\\small")
            chunks.append("\\centering")
            chunks.append(
                "\\caption{"
                + group_title
                + ": target accuracy (\\%) "
                + _gt_caption_phrase(int(gt))
                + ", varying the number of generated intermediate domains \\# gen. "
                "The mean and standard deviation are computed over runs.}"
            )
            chunks.append(f"\\label{{tab:app-gt{int(gt)}-{label_suffix}-gen-sweep-grid}}")

            chunks.append("\\setlength{\\tabcolsep}{0pt}")
            chunks.append("\\renewcommand{\\arraystretch}{1.0}")
            chunks.append(
                f"\\begin{{tabular}}{{@{{}}p{{{PANEL_COL_WIDTH}}}@{{\\hspace{{{PANEL_GAP}}}}}p{{{PANEL_COL_WIDTH}}}@{{}}}}"
            )
            chunks.append(p11 + " & " + p12 + " \\\\[" + ROW_VGAP + "]")
            chunks.append(p21 + " & " + p22 + " \\\\[" + ROW_VGAP + "]")
            chunks.append(p31 + " & " + p32 + " \\\\")
            chunks.append("\\end{tabular}")
            chunks.append("\\end{table*}")
            chunks.append("")  # spacing

    return "\n".join(chunks)


def _dataset_panel_order(df: pd.DataFrame) -> List[Tuple[str, Optional[int], str]]:
    items: List[Tuple[str, Optional[int], str]] = []
    rot = df[df["dataset"] == "rotated_mnist"]
    for deg in ROT_MNIST_MAIN_DEGREES:
        items.append(("rotated_mnist", deg, f"Rot-MNIST {deg}$^\\circ$"))
    items.append(("portraits", None, "Portraits"))
    items.append(("covtype", None, "Covtype"))
    items.append(("colored_mnist", None, "Color-MNIST"))
    # Keep only panels that have at least one row.
    out: List[Tuple[str, Optional[int], str]] = []
    for ds, deg, title in items:
        if ds == "rotated_mnist":
            if len(rot[rot["degree"] == deg]) > 0:
                out.append((ds, deg, title))
        else:
            if len(df[df["dataset"] == ds]) > 0:
                out.append((ds, deg, title))
    return out


def _subset_for_panel(df: pd.DataFrame, dataset: str, degree: Optional[int]) -> pd.DataFrame:
    sub = df[df["dataset"] == dataset]
    if dataset == "rotated_mnist":
        sub = sub[sub["degree"] == degree]
    return sub


def _plot_method_family(df_all_long: pd.DataFrame, family: Dict[str, object], out_dir: str) -> List[str]:
    df = df_all_long[df_all_long["gt"] == MAIN_GT].copy()
    if df.empty:
        return []

    methods: List[Tuple[str, str]] = family["methods"]  # type: ignore[assignment]
    title = str(family["title"])
    outfile = str(family["outfile"])
    panels = _dataset_panel_order(df)
    if not panels:
        return []

    n_panels = len(panels)
    ncols = 3
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 3.8 * nrows), sharex=True)
    axes_arr = np.array(axes).reshape(-1)

    all_gens = sorted(df["gen"].dropna().unique())
    if 0 in all_gens:
        all_gens = [0] + [g for g in all_gens if g != 0]

    for ax_i, (dataset, degree, panel_title) in enumerate(panels):
        ax = axes_arr[ax_i]
        sub = _subset_for_panel(df, dataset, degree)
        plotted = 0
        for m_key, m_label in methods:
            means = []
            stds = []
            ns = []
            xs = []
            for g in all_gens:
                vals = sub[(sub["gen"] == g) & (sub["method"] == m_key)]["acc"].dropna()
                if len(vals) == 0:
                    continue
                xs.append(g)
                means.append(float(vals.mean()))
                stds.append(float(vals.std(ddof=1)) if len(vals) > 1 else 0.0)
                ns.append(len(vals))
            if not xs:
                continue
            xs_np = np.asarray(xs, dtype=float)
            means_np = np.asarray(means, dtype=float)
            stds_np = np.asarray(stds, dtype=float)
            style = PLOT_STYLE.get(m_key, {})
            ax.plot(xs_np, means_np, linewidth=2.4, label=m_label, **style)
            ax.fill_between(xs_np, means_np - stds_np, means_np + stds_np, alpha=0.16, color=style.get("color"))
            plotted += 1

        ax.set_title(panel_title, fontsize=11)
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.8)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.set_xlabel("$G_{gen}$")
        ax.set_ylabel("Target acc. (%)")
        if plotted == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

    for j in range(n_panels, len(axes_arr)):
        axes_arr[j].axis("off")

    handles, labels = axes_arr[0].get_legend_handles_labels() if len(axes_arr) > 0 else ([], [])
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"{title} (main text, $G_{{gt}}={MAIN_GT}$)", y=1.06, fontsize=14, fontweight="bold")
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, outfile + ".png")
    pdf_path = os.path.join(out_dir, outfile + ".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]


def generate_main_plots(df_all_long: pd.DataFrame, out_dir: str = "figures_main") -> List[str]:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 300,
        }
    )
    written: List[str] = []
    for fam in MAIN_PLOT_FAMILIES:
        written.extend(_plot_method_family(df_all_long, fam, out_dir))
    return written


# ----------------------------
# DRIVER
# ----------------------------
def generate_tables(log_bases: Optional[List[str]] = None) -> Dict[str, str]:
    dfs = []
    dfs_detailed = []
    for dataset in DATASETS.keys():
        df_detailed = collect_results_detailed(dataset, log_bases=log_bases)
        if df_detailed.empty:
            continue
        dfs_detailed.append(df_detailed)
        dfs.append(_to_table_df(df_detailed))

    df_all = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    df_all_detailed = pd.concat(dfs_detailed, ignore_index=True) if dfs_detailed else pd.DataFrame()
    df_selected_long = build_selected_seed_long(df_all_detailed)

    main_tex = _latex_preamble_comment() + "\n" + make_main_table_gt0_grid_2x3(df_selected_long)
    appendix_tex = _latex_preamble_comment() + "\n" + make_appendix_tables_by_gt_2x3(df_selected_long)

    raw_dump = ""
    if not df_selected_long.empty:
        raw_dump = df_selected_long.to_string(index=False)

    plots = generate_main_plots(df_selected_long, out_dir="figures_main")
    summary_csv = build_results_summary(df_all_detailed)
    detailed_csv = df_all_detailed.sort_values(
        by=["dataset", "degree", "seed", "gt", "gen", "file_mtime", "file_name"],
        na_position="last",
    ).reset_index(drop=True)

    return {
        "main_tables_tex": main_tex,
        "appendix_tables_tex": appendix_tex,
        "raw_dump_txt": raw_dump,
        "plot_paths_txt": "\n".join(plots),
        "results_detailed_csv": detailed_csv.to_csv(index=False),
        "results_summary_csv": summary_csv.to_csv(index=False),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect and summarize experiment logs.")
    parser.add_argument(
        "--log-bases",
        type=str,
        default=",".join(DEFAULT_LOG_BASES),
        help="Comma-separated log root directories (e.g., logs,logs_rerun).",
    )
    args = parser.parse_args()
    log_bases = [x.strip() for x in args.log_bases.split(",") if x.strip()]
    out = generate_tables(log_bases=log_bases)

    with open("tables_main.tex", "w", encoding="utf-8") as f:
        f.write(out["main_tables_tex"] + "\n")
        
    with open("tables_appendix.tex", "w", encoding="utf-8") as f:
        f.write(out["appendix_tables_tex"] + "\n")

    with open("tables_debug_dump.txt", "w", encoding="utf-8") as f:
        f.write(out["raw_dump_txt"] + "\n")

    with open("results_all_with_settings.csv", "w", encoding="utf-8") as f:
        f.write(out["results_detailed_csv"])

    with open("results_summary_by_settings.csv", "w", encoding="utf-8") as f:
        f.write(out["results_summary_csv"])

    os.makedirs("figures_main", exist_ok=True)
    with open("figures_main/manifest.txt", "w", encoding="utf-8") as f:
        f.write(out["plot_paths_txt"] + "\n")

    print(
        "Wrote: tables_main.tex, tables_appendix.tex, tables_debug_dump.txt, "
        "results_all_with_settings.csv, results_summary_by_settings.csv, figures_main/*"
    )
