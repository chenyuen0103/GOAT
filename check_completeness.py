#!/usr/bin/env python3
import argparse
import glob
import os
import shlex
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple


def _parse_int_list(text: str) -> List[int]:
    # Accept "0,1,2" or "0 1 2"
    parts = [p for p in text.replace(",", " ").split() if p]
    return [int(p) for p in parts]


def _parse_str_list(text: str) -> List[str]:
    parts = [p for p in text.replace(",", " ").split() if p]
    return [p.strip() for p in parts if p.strip()]


def _completion_marker(seed: int, gt_domains: int, generated_domains: int) -> str:
    return f"seed{seed}with{gt_domains}gt{generated_domains}generated,"


def _file_contains(path: str, needle: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if needle in line:
                    return True
    except OSError:
        return False
    return False


def _any_completed_log(log_path: str, marker: str) -> Tuple[bool, Optional[str]]:
    if os.path.isfile(log_path) and _file_contains(log_path, marker):
        return True, log_path
    for bak in glob.glob(log_path + ".bak.*"):
        if os.path.isfile(bak) and _file_contains(bak, marker):
            return True, bak
    return False, None


def _any_file_exists(path: str) -> bool:
    if os.path.isfile(path):
        return True
    if glob.glob(path + ".bak.*"):
        return True
    return False


@dataclass(frozen=True)
class Setting:
    dataset: str
    seed: int
    gt_domains: int
    generated_domains: int
    label_source: str
    em_match: str
    em_select: str
    em_ensemble: bool
    small_dim: int
    target: Optional[int] = None

    def base_name(self) -> str:
        suffix = "_em-ensemble" if self.em_ensemble else ""
        return (
            f"test_acc_dim{self.small_dim}_int{self.gt_domains}_gen{self.generated_domains}_"
            f"{self.label_source}_{self.em_match}_{self.em_select}{suffix}"
        )

    def log_dir(self) -> str:
        base = os.path.join("logs", self.dataset, f"s{self.seed}")
        if self.dataset == "mnist":
            if self.target is None:
                raise ValueError("MNIST requires --targets / Setting.target")
            return os.path.join(base, f"target{self.target}")
        return base

    def paths(self) -> Tuple[str, str]:
        base = self.base_name()
        log_path = os.path.join(self.log_dir(), f"{base}.txt")
        curves_path = os.path.join(self.log_dir(), f"{base}_curves.jsonl")
        return log_path, curves_path


def iter_settings(
    *,
    dataset: str,
    seeds: Sequence[int],
    gt_domains: Sequence[int],
    generated_domains: Sequence[int],
    label_sources: Sequence[str],
    em_matches: Sequence[str],
    targets: Optional[Sequence[int]],
    em_select: str,
    em_ensemble_values: Sequence[bool],
    small_dim: int,
    canonical_gen0: bool,
) -> Iterable[Setting]:
    canonical_label_source = (
        "pseudo" if "pseudo" in label_sources else (label_sources[0] if label_sources else "pseudo")
    )
    canonical_em_match = (
        "prototypes" if "prototypes" in em_matches else (em_matches[0] if em_matches else "prototypes")
    )
    for seed in seeds:
        for gt in gt_domains:
            for gd in generated_domains:
                for ls in label_sources:
                    for em in em_matches:
                        if canonical_gen0 and gd == 0:
                            # For generated_domains=0, experiment_refrac collapses to the GOAT baseline, and the
                            # run scripts intentionally run only one canonical config to avoid duplicates.
                            if dataset == "color_mnist":
                                if ls != canonical_label_source or em != canonical_em_match:
                                    continue
                            else:
                                if em != canonical_em_match:
                                    continue
                        for em_ensemble in em_ensemble_values:
                            if dataset == "mnist":
                                assert targets is not None
                                for t in targets:
                                    yield Setting(
                                        dataset=dataset,
                                        seed=seed,
                                        target=t,
                                        gt_domains=gt,
                                        generated_domains=gd,
                                        label_source=ls,
                                        em_match=em,
                                        em_select=em_select,
                                        em_ensemble=em_ensemble,
                                        small_dim=small_dim,
                                    )
                            else:
                                yield Setting(
                                    dataset=dataset,
                                    seed=seed,
                                    target=None,
                                    gt_domains=gt,
                                    generated_domains=gd,
                                    label_source=ls,
                                    em_match=em,
                                    em_select=em_select,
                                    em_ensemble=em_ensemble,
                                    small_dim=small_dim,
                                )


def _fmt_setting(s: Setting) -> str:
    if s.dataset == "mnist":
        return f"{s.dataset} seed={s.seed} target={s.target} gt={s.gt_domains} gen={s.generated_domains} {s.label_source}/{s.em_match} {s.em_select}{' em-ens' if s.em_ensemble else ''}"
    return f"{s.dataset} seed={s.seed} gt={s.gt_domains} gen={s.generated_domains} {s.label_source}/{s.em_match} {s.em_select}{' em-ens' if s.em_ensemble else ''}"


def _python_command(s: Setting) -> str:
    cmd: List[str] = ["python", "experiment_refrac.py"]
    if s.dataset != "mnist":
        cmd += ["--dataset", s.dataset]
    if s.dataset == "mnist":
        assert s.target is not None
        cmd += ["--rotation-angle", str(s.target)]
    cmd += [
        "--seed",
        str(s.seed),
        "--gt-domains",
        str(s.gt_domains),
        "--generated-domains",
        str(s.generated_domains),
        "--label-source",
        s.label_source,
        "--em-match",
        s.em_match,
        "--em-select",
        s.em_select,
        "--small-dim",
        str(s.small_dim),
    ]
    if s.em_ensemble:
        cmd.append("--em-ensemble")
    # Keep ColorMNIST log filenames stable (run_color_mnist.sh passes --log-file).
    if s.dataset == "color_mnist":
        cmd += ["--log-file", s.base_name()]
    return " ".join(shlex.quote(x) for x in cmd)


def main() -> int:
    p = argparse.ArgumentParser(description="Check completeness of GOAT experiments from log files.")
    p.add_argument("--dataset", choices=["mnist", "color_mnist", "portraits", "covtype"], required=True)
    p.add_argument("--seeds", default="0,1,2", help="Comma/space list, e.g. '0,1,2'")
    p.add_argument("--gt-domains", default="0,1,2,3")
    p.add_argument("--generated-domains", default="0,1,2,3")
    p.add_argument("--label-sources", default="pseudo", help="Comma/space list, e.g. 'pseudo,em'")
    p.add_argument("--em-matches", default="prototypes,pseudo", help="Comma/space list")
    p.add_argument("--em-select", default="bic")
    p.add_argument(
        "--em-ensemble",
        action="store_true",
        help="Alias for --em-ensemble-mode=on (expect *_em-ensemble* logs).",
    )
    p.add_argument(
        "--em-ensemble-mode",
        choices=["off", "on", "both"],
        default="off",
        help="Check one or both EM-ensemble settings.",
    )
    p.add_argument("--small-dim", type=int, default=2048, help="Expected dim in filename (CovType typically 54).")
    p.add_argument("--targets", default="30,45,60,90", help="MNIST only: comma/space list of target degrees.")
    p.add_argument("--show-ok", action="store_true", help="Print completed settings too.")
    p.add_argument(
        "--no-canonical-gen0",
        action="store_true",
        help="Also require all (label_source/em_match) combos when generated_domains=0.",
    )
    p.add_argument(
        "--emit-commands",
        choices=["missing", "incomplete", "not-ok"],
        default=None,
        help="Print experiment_refrac.py commands for the selected subset.",
    )
    args = p.parse_args()

    if args.em_ensemble and args.em_ensemble_mode != "off":
        raise SystemExit("Use either --em-ensemble or --em-ensemble-mode, not both.")
    em_ensemble_mode = "on" if args.em_ensemble else args.em_ensemble_mode
    if em_ensemble_mode == "off":
        em_ensemble_values: List[bool] = [False]
    elif em_ensemble_mode == "on":
        em_ensemble_values = [True]
    else:
        em_ensemble_values = [False, True]

    dataset = args.dataset
    seeds = _parse_int_list(args.seeds)
    gt_domains = _parse_int_list(args.gt_domains)
    generated_domains = _parse_int_list(args.generated_domains)
    label_sources = _parse_str_list(args.label_sources)
    em_matches = _parse_str_list(args.em_matches)
    targets: Optional[List[int]] = None
    if dataset == "mnist":
        targets = _parse_int_list(args.targets)

    total = 0
    ok = 0
    missing = 0
    incomplete = 0

    for setting in iter_settings(
        dataset=dataset,
        seeds=seeds,
        gt_domains=gt_domains,
        generated_domains=generated_domains,
        label_sources=label_sources,
        em_matches=em_matches,
        targets=targets,
        em_select=args.em_select,
        em_ensemble_values=em_ensemble_values,
        small_dim=int(args.small_dim),
        canonical_gen0=not bool(args.no_canonical_gen0),
    ):
        total += 1
        log_path, curves_path = setting.paths()
        marker = _completion_marker(setting.seed, setting.gt_domains, setting.generated_domains)

        done, done_path = _any_completed_log(log_path, marker)
        if done:
            ok += 1
            if args.show_ok:
                print(f"OK: {_fmt_setting(setting)} ({done_path})")
            continue

        # Not done: distinguish "missing" vs "incomplete".
        has_any_log = _any_file_exists(log_path)
        has_any_curves = _any_file_exists(curves_path)
        if not has_any_log and not has_any_curves:
            missing += 1
            print(f"MISSING: {_fmt_setting(setting)}")
            if args.emit_commands in ("missing", "not-ok"):
                print(_python_command(setting))
        else:
            incomplete += 1
            print(f"INCOMPLETE: {_fmt_setting(setting)} (found log/curves but no completion marker)")
            if args.emit_commands in ("incomplete", "not-ok"):
                print(_python_command(setting))

    print(f"\nSummary: total={total} ok={ok} missing={missing} incomplete={incomplete}")
    return 0 if (missing == 0 and incomplete == 0) else 2


if __name__ == "__main__":
    raise SystemExit(main())
