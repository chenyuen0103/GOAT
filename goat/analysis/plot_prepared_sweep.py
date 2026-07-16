from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from goat.analysis.prepared_sweep import discover_run_records, method_rows


def plot_run_curves(record, *, plot_root: Path, methods: set[str] | None, dpi: int) -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = {
        name: result
        for name, result in record.methods.items()
        if result.test_curve and (methods is None or name in methods)
    }
    if not selected:
        return None
    config = record.config
    target = config.get("target")
    target_token = "target-none" if target is None else f"target{int(target)}"
    directory = (
        plot_root
        / "runs"
        / record.dataset
        / target_token
        / f"s{record.seed}"
        / f"gobs{int(config.get('gt_domains', 0))}"
        / f"gsyn{int(config.get('generated_domains', 0))}"
        / f"match-{config.get('em_match', 'unknown')}"
        / str(record.run_id or "unknown")
    )
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "accuracy_curves.png"

    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    for name, result in sorted(selected.items()):
        axis.plot(range(len(result.test_curve)), result.test_curve, marker="o", label=name)
    axis.set_xlabel("Adaptation step")
    axis.set_ylabel("Target accuracy")
    axis.set_title(
        f"{record.dataset} seed={record.seed}, Gobs={config.get('gt_domains')}, "
        f"Gsyn={config.get('generated_domains')}, map={config.get('em_match')}"
    )
    axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=dpi)
    plt.close(figure)
    return output


def plot_aggregate_final_accuracy(
    records,
    *,
    plot_root: Path,
    methods: set[str] | None,
    dpi: int,
) -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    grouped = defaultdict(list)
    for row in method_rows(records):
        if methods is not None and row["method"] not in methods:
            continue
        if row["final_accuracy"] is None:
            continue
        key = (row["method"], row["em_match"], int(row["generated_domains"]))
        grouped[key].append(float(row["final_accuracy"]))
    if not grouped:
        return None

    output_dir = plot_root / "aggregate"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "final_accuracy_by_gsyn.png"
    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    series = defaultdict(dict)
    for (method, mapping, gsyn), values in grouped.items():
        series[(method, mapping)][gsyn] = (
            float(np.mean(values)),
            float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        )
    for (method, mapping), points in sorted(series.items()):
        xs = sorted(points)
        ys = [points[x][0] for x in xs]
        errors = [points[x][1] for x in xs]
        axis.errorbar(xs, ys, yerr=errors, marker="o", capsize=3, label=f"{method}/{mapping}")
    axis.set_xlabel("Generated intermediate domains (Gsyn)")
    axis.set_ylabel("Final target accuracy")
    axis.set_title("Prepared-sweep final accuracy")
    axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(output, dpi=dpi)
    plt.close(figure)
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate plots from canonical prepared-sweep run records."
    )
    parser.add_argument("--log-root", required=True)
    parser.add_argument("--plot-root", required=True)
    parser.add_argument("--dataset", default="")
    parser.add_argument("--methods", nargs="+", default=[])
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    records = discover_run_records(args.log_root)
    if args.dataset:
        records = [record for record in records if record.dataset == args.dataset]
    methods = set(args.methods) if args.methods else None
    plot_root = Path(args.plot_root).expanduser()
    written = [
        path
        for record in records
        if (path := plot_run_curves(record, plot_root=plot_root, methods=methods, dpi=args.dpi))
        is not None
    ]
    aggregate_path = plot_aggregate_final_accuracy(
        records,
        plot_root=plot_root,
        methods=methods,
        dpi=args.dpi,
    )
    if aggregate_path is not None:
        written.append(aggregate_path)
    print(f"[plots] written={len(written)} root={plot_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

