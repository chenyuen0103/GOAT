"""
Experiment orchestration utilities extracted from ``experiment_new.py``.

The goal of this module is to concentrate the boilerplate that every
dataset-specific experiment shared (model loading, encoding/caching,
EM fitting, running baselines, plotting, logging, ...).  Each concrete
experiment now only has to describe *what* data it needs, while *how*
to run the standard pipeline lives here.
"""
from __future__ import annotations

import argparse
import copy
import os
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
from dataset import (
    ColorShiftMNIST,
    EncodeDataset,
    make_cov_data,
    make_portraits_data,
    transform_inter_data,
)

from util import *
from model import ENCODER, MLP_Encoder
from da_algo import *
from train_model import test
from torchvision.transforms import ToTensor

from experiment_new import (
    _plot_series_with_baselines,
    apply_em_bundle_to_target,
    build_em_bundle,
    encode_all_domains,
    fit_global_pca,
    fit_many_em_on_target,
    fit_source_gaussian_params,
    get_source_model,
    plot_pca_classes_grid,
    print_em_model_accuracies,
    run_goat,
    run_goat_classwise,
    run_mnist_ablation,
    set_all_seeds,
)
from a_star_util import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Dataclasses describing reusable artifacts
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Configuration required to (re)train the reference classifier."""

    encoder_builder: Callable[[], nn.Module]
    mode: str
    n_class: int
    epochs: int
    model_path: str
    compress: bool = True
    in_dim: Optional[int] = None
    out_dim: Optional[int] = None


@dataclass
class MethodResult:
    """Records the curves returned by a training routine."""

    name: str
    train_curve: Sequence[float]
    test_curve: Sequence[float]
    st_curve: Sequence[float]
    st_all_curve: Sequence[float]
    generated_curve: Sequence[float]
    em_acc: Optional[float] = None
    duration_sec: Optional[float] = None


def _require_args(args):
    if args is None:
        raise ValueError("Expected an argparse.Namespace 'args', but received None.")
    return args


# ---------------------------------------------------------------------------
# Generic helpers (model loading, encoding, EM fitting, plotting, logging)
# ---------------------------------------------------------------------------


def _ensure_parent_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def _seed_tag(args) -> str:
    return f"s{getattr(args, 'seed', 0)}"


def _plots_base_dir(args) -> str:
    return os.path.join("plots", args.dataset, _seed_tag(args))


def _logs_base_dir(args) -> str:
    return os.path.join(args.log_root, args.dataset, _seed_tag(args))


def _snapshot_target_em_labels(args, target_ds) -> None:
    """Persist canonical target EM labels so later method runs can restore them."""
    if target_ds is None or not hasattr(target_ds, "targets_em"):
        return
    labels_em = getattr(target_ds, "targets_em", None)
    if labels_em is None:
        return
    args._canonical_target_em = torch.as_tensor(labels_em).view(-1).long().cpu().clone()


def _restore_target_em_labels(args, target_ds, all_sets=None) -> None:
    """Restore canonical target EM labels before each method to avoid cross-method drift."""
    canonical = getattr(args, "_canonical_target_em", None)
    bundle_canonical = getattr(args, "_canonical_target_em_from_bundle", None)
    if bundle_canonical is not None:
        canonical = torch.as_tensor(bundle_canonical).view(-1).long().cpu().clone()
    if canonical is None or target_ds is None or not hasattr(target_ds, "targets_em"):
        return

    target_ds.targets_em = canonical.clone()

    # Keep the final real-domain alias in sync when provided separately.
    if all_sets and len(all_sets) > 0:
        last_real = all_sets[-1]
        if hasattr(last_real, "targets_em"):
            last_real.targets_em = canonical.clone()


def _apply_canonical_target_em(
    args,
    target_ds,
    *,
    all_sets=None,
    e_tgt=None,
    encoded_intersets=None,
) -> Optional[torch.Tensor]:
    """
    Apply one canonical target EM labeling everywhere it can be consumed:
    raw target dataset, last real-domain alias, encoded target, and encoded chain tail.
    """
    canonical = getattr(args, "_canonical_target_em", None)
    bundle_canonical = getattr(args, "_canonical_target_em_from_bundle", None)
    if bundle_canonical is not None:
        canonical = torch.as_tensor(bundle_canonical).view(-1).long().cpu().clone()
    elif canonical is not None:
        canonical = torch.as_tensor(canonical).view(-1).long().cpu().clone()
    else:
        return None

    if target_ds is not None and hasattr(target_ds, "targets_em"):
        target_ds.targets_em = canonical.clone()

    if all_sets and len(all_sets) > 0:
        last_real = all_sets[-1]
        if hasattr(last_real, "targets_em"):
            last_real.targets_em = canonical.clone()

    if e_tgt is not None:
        if hasattr(e_tgt, "data") and torch.is_tensor(e_tgt.data):
            e_tgt.targets_em = canonical.to(e_tgt.data.device)
        else:
            e_tgt.targets_em = canonical.clone()

    if encoded_intersets and len(encoded_intersets) > 0:
        tail = encoded_intersets[-1]
        if hasattr(tail, "data") and torch.is_tensor(tail.data):
            tail.targets_em = canonical.to(tail.data.device)
        else:
            tail.targets_em = canonical.clone()

    return canonical


def build_reference_model(
    args,
    config: ModelConfig,
    src_trainset,
    tgt_trainset,
) -> Tuple[nn.Module, nn.Module]:
    """Train or load the compressed classifier and expose its encoder."""

    _ensure_parent_dir(config.model_path)
    encoder = config.encoder_builder().to(device)
    model = get_source_model(
        args,
        src_trainset,
        tgt_trainset,
        n_class=config.n_class,
        mode=config.mode,
        encoder=encoder,
        epochs=config.epochs,
        model_path=config.model_path,
        target_dataset=tgt_trainset,
        force_recompute=False,
        compress=config.compress,
        in_dim=config.in_dim,
        out_dim=config.out_dim,
    )
    ref_encoder = nn.Sequential(
        model.encoder,
        nn.Flatten(start_dim=1),
        getattr(model, "compressor", nn.Identity()),
    ).eval()
    return model, ref_encoder


def _ensure_long_tensor(x):
    if x is None:
        return None
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    return x.view(-1).long().cpu()


def _normalize_dataset_targets(dataset):
    if dataset is None:
        return None
    if hasattr(dataset, "targets"):
        dataset.targets = _ensure_long_tensor(dataset.targets)
    if hasattr(dataset, "targets_em") and dataset.targets_em is not None:
        dataset.targets_em = _ensure_long_tensor(dataset.targets_em)
    if hasattr(dataset, "targets_pseudo") and dataset.targets_pseudo is not None:
        dataset.targets_pseudo = _ensure_long_tensor(dataset.targets_pseudo)
    return dataset


def _deg_idx_to_tuple(deg_idx):
    """Normalize deg_idx into a tuple so callers can pass ints or sequences."""
    if deg_idx is None:
        return None
    if isinstance(deg_idx, tuple):
        return deg_idx
    if isinstance(deg_idx, list):
        return tuple(deg_idx)
    try:
        return tuple(deg_idx)
    except TypeError:
        return (deg_idx,)


def encode_real_domains(
    args,
    *,
    ref_encoder: nn.Module,
    src_trainset,
    tgt_trainset,
    all_sets: Sequence,
    deg_idx,
    cache_dir: str,
    target_label: int,
    force_recompute: bool = False,
):
    """Encode source/intermediate/target datasets exactly once."""

    os.makedirs(cache_dir, exist_ok=True)
    e_src, e_tgt, encoded_intersets = encode_all_domains(
        src_trainset,
        tgt_trainset,
        list(all_sets),
        deg_idx,
        ref_encoder,
        cache_dir=cache_dir,
        target=target_label,
        force_recompute=force_recompute,
        args=args,
    )
    e_src = _normalize_dataset_targets(e_src)
    e_tgt = _normalize_dataset_targets(e_tgt)
    encoded_intersets = [_normalize_dataset_targets(ds) for ds in encoded_intersets]
    return e_src, e_tgt, encoded_intersets


def attach_shared_pca(args, encoded_intersets, *, seed: Optional[int] = None):
    """Fit (or reuse) a 2-D PCA used purely for visualization."""

    shared_pca = fit_global_pca(
        domains=encoded_intersets,
        classes=None,
        pool="auto",
        n_components=2,
        per_domain_cap=10000,
        random_state=seed if seed is not None else getattr(args, "seed", 0),
    )
    args.shared_pca = shared_pca
    return shared_pca


def maybe_cache_source_stats(args, encoded_src):
    """Store source Gaussian stats once if prototype matching is requested."""

    if getattr(args, "em_match", "pseudo") != "prototypes":
        return None
    mu_s, Sigma_s, priors_s = fit_source_gaussian_params(
        X=encoded_src.data,
        y=encoded_src.targets,
    )
    args._cached_source_stats = (mu_s, Sigma_s, priors_s)
    return args._cached_source_stats


def compute_pseudo_labels(model: nn.Module, dataset, args) -> np.ndarray:
    """Run pseudo-labeling with the provided (frozen) teacher."""

    with torch.no_grad():
        pseudo_labels, _ = get_pseudo_labels(
            dataset,
            model,
            confidence_q=getattr(args, "pseudo_confidence_q", 0.9),
            device_override=device,
        )
    return pseudo_labels.cpu().numpy()


def fit_em_bundle_for_dataset(
    args,
    encoded_dataset,
    raw_dataset,
    *,
    n_classes: int,
    description: str,
    cov_types: Optional[Sequence[str]] = None,
    seeds: Optional[Sequence[int]] = None,
):
    """Fit many EMs on ``encoded_dataset`` and cache the selected bundle."""

    k_list = getattr(args, "em_K_list", [n_classes])
    cov_types = cov_types or getattr(args, "em_cov_types", ["diag"])
    seeds = seeds or getattr(args, "em_seeds", [0, 1, 2])
    pca_dims = getattr(args, "em_pca_dims", [None])

    print(f"[EM] Fitting bundles for {description} ...")
    em_models = fit_many_em_on_target(
        encoded_dataset,
        K_list=k_list,
        cov_types=cov_types,
        seeds=seeds,
        pool="gap",
        pca_dims=pca_dims,
        reg=1e-4,
        max_iter=300,
        rng_base=getattr(args, "seed", 0),
        args=args,
    )
    rows = print_em_model_accuracies(em_models, raw_dataset.targets)
    rows = [{**r, "original_idx": i} for i, r in enumerate(rows)]
    for r in rows:
        bic = r.get("bic")
        print(
            f"[EM {description} idx={r['original_idx']:02d}] "
            f"acc={r['acc']*100:6.2f}% | seed={r.get('seed')} | cov={r.get('cov')} | "
            f"K={r.get('K')} | BIC={bic if bic is not None else 'NA'} | "
            f"final_ll={r.get('final_ll')}"
        )

    em_bundle = build_em_bundle(em_models, args)
    apply_em_bundle_to_target(em_bundle, encoded_dataset, raw_dataset)
    args._shared_em = em_bundle
    return em_bundle


def _wrap_result(name: str, payload, duration: float, has_em: bool = True):
    if has_em:
        train_curve, test_curve, st_curve, st_all_curve, generated_curve, em_acc = payload
    else:
        train_curve, test_curve, st_curve, st_all_curve, generated_curve = payload
        em_acc = None
    return MethodResult(
        name=name,
        train_curve=train_curve,
        test_curve=test_curve,
        st_curve=st_curve,
        st_all_curve=st_all_curve,
        generated_curve=generated_curve,
        em_acc=em_acc,
        duration_sec=duration,
    )


def run_core_methods(
    args,
    *,
    ref_model: nn.Module,
    src_trainset,
    tgt_trainset,
    all_sets,
    deg_idx,
    generated_domains: int,
    target_label: int,
) -> Dict[str, MethodResult]:
    """Run the standard suite (Ours-FR, Ours-ETA, GOAT, GOAT-Classwise)."""

    results: Dict[str, MethodResult] = {}

    def _clone():
        return copy.deepcopy(ref_model)

    # For generated_domains = 0, all "methods" collapse to the same real-domain self-training
    # baseline. To save time, run a single method and log it under GOAT.
    if generated_domains <= 0:
        set_all_seeds(args.seed)
        start = time.time()
        goat_src = _clone()
        _direct_acc, st_acc_all, train_acc_list_all, test_acc_list_all, _ = self_train(
            args,
            goat_src,
            all_sets,
            epochs=5,
            label_source="pseudo",
        )
        payload = (
            train_acc_list_all,
            test_acc_list_all,
            [float(st_acc_all)],  # kept for plotting/logging compatibility
            [float(st_acc_all)],  # kept for plotting/logging compatibility
            [0.0],
        )
        results["goat"] = _wrap_result("GOAT", payload, time.time() - start, has_em=False)
        print("[run_core_methods] generated_domains <= 0: running GOAT baseline only.")
        return results

    # ---------------- Ours-FR ----------------
    _restore_target_em_labels(args, tgt_trainset, all_sets)
    set_all_seeds(args.seed)
    start = time.time()
    ours_src = _clone()
    ours_cp = _clone()
    payload = run_main_algo_cached(
        ours_cp,
        ours_src,
        src_trainset,
        tgt_trainset,
        all_sets,
        deg_idx,
        generated_domains,
        epochs=5,
        target=target_label,
        args=args,
        gen_method="fr",
    )
    results["ours_fr"] = _wrap_result("Ours-FR", payload, time.time() - start)

    # ---------------- Ours-ETA ----------------
    _restore_target_em_labels(args, tgt_trainset, all_sets)
    set_all_seeds(args.seed)
    start = time.time()
    ours_eta_src = _clone()
    ours_eta_cp = _clone()
    payload = run_main_algo_cached(
        ours_eta_cp,
        ours_eta_src,
        src_trainset,
        tgt_trainset,
        all_sets,
        deg_idx,
        generated_domains,
        epochs=5,
        target=target_label,
        args=args,
        gen_method="natural",
    )
    results["ours_eta"] = _wrap_result("Ours-ETA", payload, time.time() - start)

    # ---------------- GOAT / GOAT-Classwise ----------------
    # Only meaningful (and only safe) when we actually generate synthetic domains
    if generated_domains > 0:
        # GOAT
        _restore_target_em_labels(args, tgt_trainset, all_sets)
        set_all_seeds(args.seed)
        start = time.time()
        goat_src = _clone()
        goat_cp = copy.deepcopy(goat_src)
        payload = run_goat(
            goat_cp,
            goat_src,
            src_trainset,
            tgt_trainset,
            all_sets,
            deg_idx,
            generated_domains,
            epochs=5,
            target=target_label,
            args=args,
        )
        results["goat"] = _wrap_result("GOAT", payload, time.time() - start, has_em=False)

        # GOAT-Classwise
        _restore_target_em_labels(args, tgt_trainset, all_sets)
        set_all_seeds(args.seed)
        start = time.time()
        goatcw_src = _clone()
        goatcw_cp = copy.deepcopy(goatcw_src)
        payload = run_goat_classwise(
            goatcw_cp,
            goatcw_src,
            src_trainset,
            tgt_trainset,
            all_sets,
            deg_idx,
            generated_domains,
            epochs=5,
            target=target_label,
            args=args,
        )
        results["goat_classwise"] = _wrap_result(
            "GOAT-Classwise", payload, time.time() - start
        )
    else:
        print(
            "[run_core_methods] generated_domains <= 0: "
            "skipping GOAT and GOAT-Classwise (no synthetic domains to generate)."
        )

    return results


def _main_algo_cache_key(
    args,
    *,
    src_trainset,
    tgt_trainset,
    all_sets,
    deg_idx,
    generated_domains,
    epochs,
    target,
):
    dataset_ids = tuple(id(ds) for ds in all_sets)
    return (
        getattr(args, "dataset", None),
        getattr(args, "seed", None),
        getattr(args, "ssl_weight", None),
        getattr(args, "small_dim", None),
        getattr(args, "pseudo_confidence_q", None),
        getattr(args, "gt_domains", None),
        getattr(args, "label_source", None),
        target,
        _deg_idx_to_tuple(deg_idx),
        epochs,
        generated_domains,
        id(src_trainset),
        id(tgt_trainset),
        dataset_ids,
    )

def run_goat_classwise(
    model_copy: Classifier,
    source_model: Classifier,
    src_trainset: Dataset,
    tgt_trainset: Dataset,
    all_sets: List[Dataset],
    deg_idx: List[int],
    generated_domains: int,
    epochs: int = 10,
    target: int = 60,
    args=None,
):
    """
    GOAT baseline with class-wise synthetic generation.

    Returns:
        train_curve, test_curve, st_curve, st_all_curve, generated_curve, em_acc
    so it can be wrapped by _wrap_result(..., has_em=True).
    """
    device = next(source_model.parameters()).device

    # -------------------- helpers --------------------
    def _labels_for_split(ds: Dataset, is_source: bool) -> torch.Tensor:
        if is_source and hasattr(ds, "targets") and ds.targets is not None:
            return torch.as_tensor(ds.targets).long().cpu()
        return torch.as_tensor(ds.targets_em).long().cpu()

    def _subset_by_class(ds: Dataset, cls: int, is_source: bool) -> Optional[Dataset]:
        labels = _labels_for_split(ds, is_source=is_source)
        X = ds.data if torch.is_tensor(getattr(ds, "data", None)) else torch.as_tensor(ds.data)
        X = X.cpu()
        m = (labels == int(cls))
        if m.sum().item() == 0:
            return None
        Xc = X[m]
        yc = labels[m]
        w  = torch.ones(len(yc))
        return DomainDataset(Xc, w, yc)

    def _merge_domains_per_step(list_of_lists: List[List[Dataset]]) -> List[Dataset]:
        """Merge step j across classes into a single DomainDataset."""
        if not list_of_lists:
            return []
        n_steps = min(len(L) for L in list_of_lists)   # expected: n_inter + 1 incl. right endpoint
        merged: List[Dataset] = []
        for j in range(n_steps):
            parts = [L[j] for L in list_of_lists if L[j] is not None]
            if not parts:
                continue
            Xs, Ws, Ys = [], [], []
            for D in parts:
                Xs.append(D.data if torch.is_tensor(D.data) else torch.as_tensor(D.data))
                ws = getattr(D, "weights", None)
                if ws is None:
                    ws = torch.ones(len(D.targets))
                Ws.append(ws if torch.is_tensor(ws) else torch.as_tensor(ws))
                Ys.append(D.targets if torch.is_tensor(D.targets) else torch.as_tensor(D.targets))
            X = torch.cat([x.cpu().float() for x in Xs], dim=0)
            W = torch.cat([w.cpu().float() for w in Ws], dim=0)
            Y = torch.cat([y.cpu().long() for y in Ys], dim=0)
            merged.append(DomainDataset(X, W, Y, Y))
        return merged

    # -------------------- Case 1: GST baseline (no synthetic domains, no EM) --------------------
    if generated_domains <= 0:
        set_all_seeds(args.seed)

        # Baseline on target only (GST-style)
        direct_acc, st_acc, train_acc_by_domain0, test_acc_by_domain0, _ = self_train(
            args,
            model_copy,
            [tgt_trainset],
            epochs=epochs,
            label_source="pseudo",
        )

        # Baseline on full chain of real domains
        set_all_seeds(args.seed)
        direct_acc_all, st_acc_all, train_acc_list_all, test_acc_list_all, _ = self_train(
            args,
            source_model,
            all_sets,
            epochs=epochs,
            label_source="pseudo",
        )

        generated_acc = 0.0
        acc_em_pseudo = float("nan")  # no EM mapping in this regime

        # Return length-1 curves to keep plotting/logging consistent
        return (
            [float(train_acc_by_domain0[-1])],   # train_curve
            [float(test_acc_by_domain0[-1])],    # test_curve
            [float(st_acc)],                     # st_curve
            [float(st_acc_all)],                 # st_all_curve
            [float(generated_acc)],              # generated_curve
            acc_em_pseudo,
        )

    # -------------------- Case 2: generated_domains > 0 (full GOAT-CW with EM available) --------------------

    # Reuse cached baselines + encodings when possible
    cache_key = _main_algo_cache_key(
        args,
        src_trainset=src_trainset,
        tgt_trainset=tgt_trainset,
        all_sets=all_sets,
        deg_idx=deg_idx,
        generated_domains=generated_domains,
        epochs=epochs,
        target=target,
    )
    if not hasattr(args, "_refrac_main_cache"):
        args._refrac_main_cache = {}
    cached_setup = args._refrac_main_cache.get(cache_key)

    if cached_setup is not None:
        direct_acc        = cached_setup["direct_acc"]
        st_acc            = cached_setup["st_acc"]
        direct_acc_all    = cached_setup["direct_acc_all"]
        st_acc_all        = cached_setup["st_acc_all"]
        e_src             = cached_setup["e_src"]
        e_tgt             = cached_setup["e_tgt"]
        encoded_intersets = cached_setup["encoded_intersets"]
        pseudolabels      = cached_setup["pseudolabels"]
    else:
        # 1) Baselines
        set_all_seeds(args.seed)
        direct_acc, st_acc, train_acc_by_domain0, test_acc_by_domain0, _ = self_train(
            args,
            model_copy,
            [tgt_trainset],
            epochs=epochs,
            label_source="pseudo",
        )

        set_all_seeds(args.seed)
        direct_acc_all, st_acc_all, train_acc_list_all, test_acc_list_all, _ = self_train(
            args,
            source_model,
            all_sets,
            epochs=epochs,
            label_source="pseudo",
        )

        if abs(st_acc - st_acc_all) > 1e-4:
            print(f"[GOAT-CW] Warning: st_acc ({st_acc}) != st_acc_all ({st_acc_all})")

        # 2) Encode domains once (encoder → flatten → compressor)
        if args.dataset != "mnist":
            cache_dir = f"{args.dataset}/cache{args.ssl_weight}/small_dim{args.small_dim}/"
            plot_dir  = _plots_base_dir(args)
        else:
            cache_dir = f"cache{args.ssl_weight}/target{target}/small_dim{args.small_dim}/"
            plot_dir  = os.path.join(_plots_base_dir(args), f"target{target}")
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(plot_dir,  exist_ok=True)

        e_src, e_tgt, encoded_intersets = encode_all_domains(
            src_trainset,
            tgt_trainset,
            all_sets,
            deg_idx,
            nn.Sequential(
                source_model.encoder,
                nn.Flatten(start_dim=1),
                getattr(source_model, "compressor", nn.Identity()),
            ),
            cache_dir,
            target,
            force_recompute=False,
            args=args,
        )

        # 3) Teacher pseudo-labels on TARGET (diagnostics only; not EM)
        with torch.no_grad():
            teacher = copy.deepcopy(source_model).to(device).eval()
            pseudo_labels, _ = get_pseudo_labels(
                tgt_trainset,
                teacher,
                confidence_q=getattr(args, "pseudo_confidence_q", 0.9),
                device_override=device,
            )
        pseudolabels = pseudo_labels.cpu().numpy()

        args._refrac_main_cache[cache_key] = {
            "direct_acc": direct_acc,
            "st_acc": st_acc,
            "direct_acc_all": direct_acc_all,
            "st_acc_all": st_acc_all,
            "e_src": e_src,
            "e_tgt": e_tgt,
            "encoded_intersets": encoded_intersets,
            "pseudolabels": pseudolabels,
        }

    # Plot/cache directories (for synthetic runs only)
    if args.dataset != "mnist":
        cache_dir = f"{args.dataset}/cache{args.ssl_weight}/small_dim{args.small_dim}/"
        plot_dir  = _plots_base_dir(args)
    else:
        cache_dir = f"cache{args.ssl_weight}/target{target}/small_dim{args.small_dim}/"
        plot_dir  = os.path.join(_plots_base_dir(args), f"target{target}")
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(plot_dir,  exist_ok=True)

    # Make sure encoded domains carry the right labels for class-wise splits
    raw_domains = [src_trainset] + all_sets
    enc_domains = encoded_intersets
    if len(raw_domains) != len(enc_domains):
        raise RuntimeError(
            f"raw/encoded domains count mismatch ({len(raw_domains)=} vs {len(enc_domains)=}); "
            "encoded_intersets should include source + intermediates + target."
        )

    # Re-apply canonical/shared target EM labels before GOAT-CW.
    # This prevents prior method runs from mutating target labels seen by class-wise generation.
    em_bundle_target = None
    if hasattr(args, "_shared_em_per_domain") and target in getattr(args, "_shared_em_per_domain", {}):
        em_bundle_target = args._shared_em_per_domain[target]
    elif hasattr(args, "_shared_em"):
        em_bundle_target = args._shared_em

    if em_bundle_target is not None:
        apply_em_bundle_to_target(em_bundle_target, e_tgt, tgt_trainset)
    else:
        _restore_target_em_labels(args, tgt_trainset, all_sets)
        if hasattr(tgt_trainset, "targets_em") and tgt_trainset.targets_em is not None:
            e_tgt.targets_em = torch.as_tensor(tgt_trainset.targets_em).to(e_tgt.data.device).long()
    # Final hard guard: enforce canonical target EM labels on raw + encoded targets.
    _apply_canonical_target_em(
        args,
        tgt_trainset,
        all_sets=all_sets,
        e_tgt=e_tgt,
        encoded_intersets=encoded_intersets,
    )

    for raw_ds, enc_ds in zip(raw_domains, enc_domains):
        if hasattr(raw_ds, "targets_em"):
            labels_em = torch.as_tensor(raw_ds.targets_em)
        else:
            labels_em = torch.as_tensor(raw_ds.targets)
        enc_ds.targets_em = labels_em.to(enc_ds.data.device)
        enc_ds.targets    = torch.as_tensor(raw_ds.targets).to(enc_ds.data.device)

    # Attach pseudo-labels on target
    setattr(args, "_cached_pseudolabels", pseudolabels)
    # Here we assume tgt_trainset.targets_em was set upstream by EM mapping
    e_tgt.targets_em = tgt_trainset.targets_em.to(device)

    # K: number of classes from source
    K = int(e_src.targets.max().item()) + 1
    y_true = e_tgt.targets.cpu().numpy() if torch.is_tensor(e_tgt.targets) else np.asarray(e_tgt.targets)
    e_tgt.targets_pseudo = torch.as_tensor(pseudolabels, dtype=torch.long)
    tgt_trainset.targets_pseudo = e_tgt.targets_pseudo.cpu().clone()

    acc_em_pseudo = (
        tgt_trainset.targets_em == torch.as_tensor(
            y_true,
            device=tgt_trainset.targets_em.device,
            dtype=tgt_trainset.targets_em.dtype,
        )
    ).to(torch.float32).mean().item()
    print(f"[GOAT-CW] EM→class (pseudo mapping) accuracy: {acc_em_pseudo:.4f}")

    # -------------------- class-wise generation --------------------
    generated_acc = 0.0
    all_domains: List[Dataset] = []

    for i in range(len(encoded_intersets) - 1):
        s_ds = encoded_intersets[i]
        t_ds = encoded_intersets[i + 1]
        is_source_left = (i == 0)

        per_class_chains: List[List[Dataset]] = []
        for c in range(K):
            s_c = _subset_by_class(s_ds, c, is_source=is_source_left)
            t_c = _subset_by_class(t_ds, c, is_source=False)
            if s_c is None or t_c is None:
                continue

            chain_c, _, _ = generate_domains(
                generated_domains,
                s_c,
                t_c,
            )

            # force global class id c
            for D in chain_c:
                y_global = torch.full((len(D.targets),), c, dtype=torch.long)
                D.targets = y_global
                D.targets_em = y_global.clone()

            if chain_c:
                for step_ds in chain_c:
                    labs = step_ds.targets if torch.is_tensor(step_ds.targets) else torch.as_tensor(step_ds.targets)
                    assert (labs.cpu().numpy() == c).all()
                per_class_chains.append(chain_c)

        merged_chain = _merge_domains_per_step(per_class_chains)
        all_domains += merged_chain

    # ensure last training domain is the full encoded target
    if len(all_domains) > 0:
        all_domains[-1] = DomainDataset(
            e_tgt.data if torch.is_tensor(e_tgt.data) else torch.as_tensor(e_tgt.data),
            torch.ones(len(e_tgt.targets)),
            e_tgt.targets,
            e_tgt.targets_em,
        )

    # -------------------- train on merged synthetic chain --------------------
    set_all_seeds(args.seed)
    _, generated_acc, train_acc_by_domain, test_acc_by_domain, domain_stats, last_prediction = self_train(
        args,
        source_model.mlp,
        all_domains,
        epochs=epochs,
        label_source=getattr(args, "label_source", "pseudo"),
        use_labels=getattr(args, "use_labels", False),
        return_stats=True,
    )

    # plotting code unchanged ...

    return (
        train_acc_by_domain,
        test_acc_by_domain,
        st_acc,
        st_acc_all,
        generated_acc,
        acc_em_pseudo,
    )



def run_main_algo_cached(
    model_copy: nn.Module,
    source_model: nn.Module,
    src_trainset,
    tgt_trainset,
    all_sets,
    deg_idx,
    generated_domains: int,
    epochs: int = 3,
    target: int = 60,
    args=None,
    gen_method: str = "fr",
    # multi-EM controls (kept for API compatibility; not used here)
    use_multi_em: bool = False,
    em_cov_types: Tuple[str, ...] = ("diag",),
    em_K_list: Optional[List[int]] = None,
    em_seeds: Tuple[int, ...] = (0, 1, 2, 3, 4),
    em_pca_dims: Tuple[Optional[int], ...] = (None,),
    em_select: str = "bic",
    em_ensemble_weights: str = "bic",
    # sharing controls; EM bundles are built outside (e.g., in run_mnist_experiment)
    use_shared_em: bool = True,
    shared_em_cfg: dict = None,
):
    """
    Copy of experiment_new.run_main_algo with lightweight caching so repeated calls
    (e.g., FR vs ETA variants) reuse the expensive setup artifacts.
    """
    if generated_domains <= 0:
        set_all_seeds(args.seed)

        # Baseline on target only (GST-style on target)
        direct_acc, st_acc, train_acc_by_domain0, test_acc_by_domain0, _ = self_train(
            args,
            model_copy,
            [tgt_trainset],
            epochs=epochs,
            label_source="pseudo",
        )

        # Baseline on full chain of real domains
        set_all_seeds(args.seed)
        direct_acc_all, st_acc_all, train_acc_list_all, test_acc_list_all, _ = self_train(
            args,
            source_model,
            all_sets,
            epochs=epochs,
            label_source="pseudo",
        )

        # No synthetic domains, no EM accuracy needed
        generated_acc = 0.0
        acc_em_pseudo = float("nan")

        # In this regime, the meaningful "trajectory" is the self-training path over REAL domains.
        # Keep st/st_all as scalar baselines (horizontal lines), but return full curves for plotting/logging.
        return (
            train_acc_list_all,                  # train_curve (per-domain train accs)
            test_acc_list_all,                   # test_curve (includes initial direct acc)
            [float(st_acc)],                     # st_curve (target-only GST baseline)
            [float(st_acc_all)],                 # st_all_curve (real-chain GST baseline)
            [float(generated_acc)],              # generated_curve
            acc_em_pseudo,
        )

    def _sanitize_targets_em_field(ds):
        if ds is None or not hasattr(ds, "targets_em"):
            return None
        val = getattr(ds, "targets_em")
        if val is None:
            return None
        tensor = val if torch.is_tensor(val) else torch.as_tensor(val)
        tensor = tensor.view(-1).long()
        if tensor.numel() == 0 or not torch.any(tensor >= 0):
            ds.targets_em = None
            return None
        ds.targets_em = tensor.to(ds.data.device) if hasattr(ds, "data") and torch.is_tensor(ds.data) else tensor
        return ds.targets_em

    device = next(source_model.parameters()).device
    cache_key = _main_algo_cache_key(
        args,
        src_trainset=src_trainset,
        tgt_trainset=tgt_trainset,
        all_sets=all_sets,
        deg_idx=deg_idx,
        generated_domains=generated_domains,
        epochs=epochs,
        target=target,
    )
    if not hasattr(args, "_refrac_main_cache"):
        args._refrac_main_cache = {}
    cached_setup = args._refrac_main_cache.get(cache_key)

    # 0) Sanity: identical initialization?
    for p1, p2 in zip(model_copy.parameters(), source_model.parameters()):
        if not torch.equal(p1, p2):
            print(
                "[run_main_algo_cached] Warning: model_copy and source_model have different initial weights."
            )
            break

    # Cache + plot directories are always ensured per call (plots differ per method)
    if args.dataset != "mnist":
        cache_dir = f"{args.dataset}/cache{args.ssl_weight}/small_dim{args.small_dim}"
        plot_dir = _plots_base_dir(args)
    else:
        cache_dir = f"cache{args.ssl_weight}/target{target}/small_dim{args.small_dim}/"
        plot_dir = os.path.join(_plots_base_dir(args), f"target{target}")
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    if cached_setup is not None:
        direct_acc = cached_setup["direct_acc"]
        st_acc = cached_setup["st_acc"]
        direct_acc_all = cached_setup["direct_acc_all"]
        st_acc_all = cached_setup["st_acc_all"]
        e_src = cached_setup["e_src"]
        e_tgt = cached_setup["e_tgt"]
        encoded_intersets = cached_setup["encoded_intersets"]
        pseudolabels = cached_setup["pseudolabels"]
    else:
        # 1) Baselines
        set_all_seeds(args.seed)
        direct_acc, st_acc, train_acc_by_domain0, test_acc_by_domain0, _ = self_train(
            args, model_copy, [tgt_trainset], epochs=epochs, label_source="pseudo"
        )

        set_all_seeds(args.seed)
        direct_acc_all, st_acc_all, train_acc_list_all, test_acc_list_all, _ = self_train(
            args, source_model, all_sets, epochs=epochs, label_source="pseudo"
        )

        if abs(st_acc - st_acc_all) > 1e-4:
            print(
                f"[run_main_algo_cached] Warning: st_acc ({st_acc}) != st_acc_all ({st_acc_all})"
            )

        # 2) Teacher pseudo-labels on TARGET (diagnostics only)
        em_teacher = copy.deepcopy(source_model).to(device).eval()
        with torch.no_grad():
            pseudo_labels, _ = get_pseudo_labels(
                tgt_trainset,
                em_teacher,
                confidence_q=getattr(args, "pseudo_confidence_q", 0.9),
                device_override=device,
            )
        pseudolabels = pseudo_labels.cpu().numpy()

        # 4) Encode domains once (encoder → flatten → compressor)
        e_src, e_tgt, encoded_intersets = encode_all_domains(
            src_trainset,
            tgt_trainset,
            all_sets,
            deg_idx,
            nn.Sequential(
                source_model.encoder,
                nn.Flatten(start_dim=1),
                source_model.compressor if hasattr(source_model, "compressor") else nn.Identity(),
            ),
            cache_dir,
            target,
            force_recompute=False,
            args=args,
        )

        # Ensure EM labels for target
        tgt_em = _sanitize_targets_em_field(tgt_trainset)
        if tgt_em is not None:
            e_tgt.targets_em = tgt_em.clone()
        else:
            print(
                "[run_main_algo_cached] Warning: tgt_trainset.targets_em is missing; EM labels not set for target."
            )

        # Ensure EVERY encoded domain has .targets_em before calling generators
        for ds in encoded_intersets:
            if _sanitize_targets_em_field(ds) is not None:
                continue

            if getattr(ds, "targets", None) is not None:
                t = ds.targets
                ds.targets_em = (
                    t.clone().long() if torch.is_tensor(t) else torch.as_tensor(t, dtype=torch.long)
                )
                continue

            if getattr(ds, "targets_pseudo", None) is not None:
                t = ds.targets_pseudo
                ds.targets_em = (
                    t.clone().long() if torch.is_tensor(t) else torch.as_tensor(t, dtype=torch.long)
                )
                continue

            raise ValueError(
                "[run_main_algo_cached] Encoded domain is missing targets, targets_em, and targets_pseudo; "
                "natural / FR generators cannot proceed."
        )

        args._refrac_main_cache[cache_key] = {
            "direct_acc": direct_acc,
            "st_acc": st_acc,
            "direct_acc_all": direct_acc_all,
            "st_acc_all": st_acc_all,
            "e_src": e_src,
            "e_tgt": e_tgt,
            "encoded_intersets": encoded_intersets,
            "pseudolabels": pseudolabels,
        }

    # Force canonical target EM labels across raw/encoded views before diagnostics or training.
    _apply_canonical_target_em(
        args,
        tgt_trainset,
        all_sets=all_sets,
        e_tgt=e_tgt,
        encoded_intersets=encoded_intersets,
    )

    _sanitize_targets_em_field(e_src)
    _sanitize_targets_em_field(e_tgt)
    for ds in encoded_intersets:
        _sanitize_targets_em_field(ds)

    # 5) Diagnostics: pseudo-label accuracy on target
    y_true = e_tgt.targets
    if torch.is_tensor(y_true):
        y_true = y_true.cpu()
    preds = torch.as_tensor(pseudolabels, device=y_true.device)
    acc_pl = (preds == y_true).float().mean().item()
    print(f"Pseudo-label accuracy (teacher on target): {acc_pl:.4f}")

    # 6) Determine K from SOURCE (stable)
    try:
        K_infer = int(e_src.targets.max().item()) + 1
    except Exception:
        K_infer = 10
    if not em_K_list:
        em_K_list = [K_infer]

    # 7) Attach pseudo labels to target datasets (for plotting / diagnostics)
    e_tgt.targets_pseudo = torch.as_tensor(pseudolabels, dtype=torch.long)
    tgt_trainset.targets_pseudo = e_tgt.targets_pseudo.cpu().clone()

    # 8) EM→class accuracy on TARGET from precomputed labels
    if hasattr(e_tgt, "targets_em") and e_tgt.targets_em is not None:
        acc_em_pseudo = (
            e_tgt.targets_em.cpu() == torch.as_tensor(y_true, dtype=e_tgt.targets_em.dtype)
        ).to(torch.float32).mean().item()
    else:
        acc_em_pseudo = float("nan")
    print(f"[MainAlgo] EM→class (mapped) accuracy on target: {acc_em_pseudo:.4f}")

    # Optional: check presence of target bundle
    if use_shared_em:
        if hasattr(args, "_shared_em_per_domain") and target in getattr(args, "_shared_em_per_domain", {}):
            em_bundle_target = args._shared_em_per_domain[target]
        else:
            em_bundle_target = getattr(args, "_shared_em", None)
        if em_bundle_target is None:
            print("[run_main_algo_cached] Warning: no shared EM bundle found for target domain.")

    # 9) Evaluate current source_model on target images (sanity)
    tgt_loader_eval = DataLoader(
        tgt_trainset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    _, tgt_acc_after = test(tgt_loader_eval, source_model)
    print(f"[MainAlgo] Baseline target accuracy before synthetic training: {tgt_acc_after:.4f}")


    synthetic_domains: List = []
    _method = (gen_method or "fr").lower()
    if _method in {"fr", "fisher-rao", "fisher_rao"}:
        _gen_fn = generate_fr_domains_between_optimized
    elif _method in {"natural", "natureal", "eta", "nat", "np"}:
        _gen_fn = generate_natural_domains_between
    else:
        raise ValueError(f"Unknown gen_method '{gen_method}'. Use 'fr' or 'natural'.")

    for i in range(len(encoded_intersets) - 1):
        # breakpoint()
        out = _gen_fn(
            generated_domains,
            encoded_intersets[i],
            encoded_intersets[i + 1],
            cov_type="full",
            save_path=plot_dir,
            args=args,
        )
        pair_domains = out[0]
        domain_stats = out[2]
        synthetic_domains += pair_domains

    if not synthetic_domains:
        return direct_acc, st_acc, direct_acc_all, st_acc_all, 0.0, acc_em_pseudo

    # 11) Self-training on synthetic domains (EM labels drive training)
    set_all_seeds(args.seed)
    direct_acc_syn, generated_acc, train_acc_by_domain, test_acc_by_domain, last_predictions = self_train(
        args,
        source_model.mlp,
        synthetic_domains,
        epochs=epochs,
        label_source=getattr(args, "label_source", "em"),
    )

    # 12) Plots
    plot_pca_classes_grid(
        encoded_intersets,
        classes=(3, 6, 8, 9) if "mnist" in args.dataset else (0, 1),
        save_path=os.path.join(
            plot_dir,
            f"pca_dim{args.small_dim}_int{args.gt_domains}_gen{args.generated_domains}_real_domains.png",
        ),
        label_source="real",
        ground_truths=True,
        pca=getattr(args, "shared_pca", None),
    )

    # --- build chain with real + synthetic domains interleaved ---
    if synthetic_domains:
        step_len = int(generated_domains) + 1
        chain_for_plot: List = []
        n_pairs = len(encoded_intersets) - 1

        for i in range(n_pairs):
            if i == 0:
                chain_for_plot.append(encoded_intersets[0])  # first real domain

            start = i * step_len
            chunk = synthetic_domains[start : start + step_len]
            if not chunk:
                continue

            if step_len > 1:
                chain_for_plot.extend(chunk[:-1])  # synthetic only (drop appended endpoint)
            chain_for_plot.append(encoded_intersets[i + 1])  # right real endpoint

        # (a) labels used during self_train (typically 'em')
        plot_pca_classes_grid(
            chain_for_plot,
            classes=(3, 6, 8, 9) if "mnist" in args.dataset else (0, 1),
            save_path=os.path.join(
                plot_dir,
                f"pca_dim{args.small_dim}_int{args.gt_domains}_gen{args.generated_domains}_"
                f"{args.label_source}_{getattr(args, 'em_match', 'pseudo')}_"
                f"{args.em_select}{'_em-ensemble' if args.em_ensemble else ''}_{_method}.png",
            ),
            label_source=getattr(args, "label_source", "em"),
            pseudolabels=last_predictions,
            pca=getattr(args, "shared_pca", None),
        )

        # (b) explicit EM labels
        plot_pca_classes_grid(
            chain_for_plot,
            classes=(3, 6, 8, 9) if "mnist" in args.dataset else (0, 1),
            save_path=os.path.join(
                plot_dir,
                f"pca_dim{args.small_dim}_int{args.gt_domains}_gen{args.generated_domains}_"
                f"emlabels_{args.em_select}{'_em-ensemble' if args.em_ensemble else ''}_{_method}.png",
            ),
            label_source="em",
            pca=getattr(args, "shared_pca", None),
        )

        # (c) teacher pseudo labels along chain
        plot_pca_classes_grid(
            chain_for_plot,
            classes=(3, 6, 8, 9) if "mnist" in args.dataset else (0, 1),
            save_path=os.path.join(
                plot_dir,
                f"pca_dim{args.small_dim}_int{args.gt_domains}_gen{args.generated_domains}_source_pseudo_{_method}.png",
            ),
            label_source="pseudo",
            pseudolabels=pseudolabels,
            pca=getattr(args, "shared_pca", None),
        )

    return (
        train_acc_by_domain,
        test_acc_by_domain,
        st_acc,
        st_acc_all,
        generated_acc,
        acc_em_pseudo,
    )

def _display_name(key: str) -> str:
    mapping = {
        "ours_fr": "Ours-FR",
        "ours_eta": "Ours-ETA",
        "goat": "GOAT",
        "goat_classwise": "GOAT-Classwise",
    }
    return mapping.get(key, key)


# ---------------- Generic plotting helper: N series + per-method baselines ----------------
def _plot_series_with_baselines(
    series,
    labels,
    baselines=None,  # list of (st, st_all) per series (optional)
    ref_line_value=None,
    ref_line_label=None,
    ref_line_style="--",
    title="",
    ylabel="Accuracy",
    xlabel="Training Domain Index",
    save_path=None,
    # New: distinguish real vs synthetic domains
    synth_per_segment: int = None,   # number of synthetic domains between two real domains
    n_real_segments: int = None,     # number of real gaps (src→inter1, inter1→inter2, ..., →tgt)
):
    import numbers

    def _to_array(v):
        # No data
        if v is None:
            return np.array([], dtype=float)

        # Torch tensor
        try:
            import torch
            if isinstance(v, torch.Tensor):
                if v.ndim == 0:
                    return np.array([float(v.item())], dtype=float)
                return v.detach().cpu().flatten().numpy().astype(float)
        except Exception:
            pass

        # Scalar (float, int, numpy scalar, etc.)
        if isinstance(v, numbers.Number):
            return np.array([float(v)], dtype=float)

        # Try as a generic iterable
        try:
            return np.array(
                [np.nan if x is None else float(x) for x in v],
                dtype=float,
            )
        except TypeError:
            # Fallback: treat as scalar
            return np.array([float(v)], dtype=float)

    S = [_to_array(s) for s in (series or [])]
    if not S:
        print(f"[plot] Skip {title}: no data.")
        return

    L = max(len(s) for s in S)
    if L == 0:
        print(f"[plot] Skip {title}: empty series.")
        return
    S = [
        (np.pad(s, (0, L - len(s)), constant_values=np.nan) if len(s) < L else s)
        for s in S
    ]
    x = np.arange(0, L, dtype=int)

    plt.figure()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    markers = ['o', 's', '^', 'D', 'v', '>', '<', 'P', 'X']

    n = len(S)
    # Compute boundaries of real domains along the x-axis if info provided
    boundaries = None
    if (
        synth_per_segment is not None
        and n_real_segments is not None
        and isinstance(synth_per_segment, int)
        and isinstance(n_real_segments, int)
        and synth_per_segment >= 0
        and n_real_segments >= 0
    ):
        step = synth_per_segment + 1
        boundaries = [k * step for k in range(n_real_segments + 1)]
        boundaries = sorted(set(boundaries))
        boundaries = [b for b in boundaries if b < L]

    for i, s in enumerate(S):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        label = labels[i] if labels and i < len(labels) else f"Series {i+1}"
        plt.plot(x, s, marker=marker, linewidth=1.8, label=label, color=color)

        if boundaries:
            idxs = [b for b in boundaries if b < len(s) and np.isfinite(s[b])]
            if idxs:
                plt.scatter(
                    idxs,
                    s[idxs],
                    s=36,
                    facecolors='white',
                    edgecolors=color,
                    linewidths=1.2,
                    zorder=4,
                )

        if baselines and i < len(baselines) and baselines[i] is not None:
            st, st_all = baselines[i]
            if st is not None:
                plt.axhline(
                    float(st),
                    linestyle=':',
                    linewidth=1.5,
                    color=color,
                    alpha=0.9,
                    label="st",
                )
            if st_all is not None:
                plt.axhline(
                    float(st_all),
                    linestyle='--',
                    linewidth=1.5,
                    color=color,
                    alpha=0.9,
                    label="st_all",
                )

    if boundaries:
        for b in boundaries:
            plt.axvline(b, linestyle=':', linewidth=1.0, color='gray', alpha=0.7)

    if ref_line_value is not None:
        plt.axhline(
            float(ref_line_value),
            linestyle=ref_line_style,
            linewidth=1.6,
            color='k',
            alpha=0.7,
            label=(ref_line_label or "reference"),
        )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)

    handles, labels_ = plt.gca().get_legend_handles_labels()
    if boundaries:
        from matplotlib.lines import Line2D
        extra = [
            Line2D([0], [0], color='gray', linestyle=':', linewidth=1.0, label='Real boundary'),
            Line2D(
                [0],
                [0],
                marker='o',
                markerfacecolor='white',
                markeredgecolor='gray',
                linestyle='None',
                label='Real domain',
            ),
        ]
        handles += extra
        labels_ += [h.get_label() for h in extra]
    plt.legend(handles, labels_)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[MNIST-EXP] Saved {save_path}")
    plt.close()

def _last_as_float(curve):
    """Take a curve (list/np/torch/scalar) and return its last value as float."""
    if curve is None:
        return None

    # torch tensor
    if torch.is_tensor(curve):
        if curve.numel() == 0:
            return None
        return float(curve.view(-1)[-1].item())

    # list / tuple / np array
    import numpy as np
    if isinstance(curve, (list, tuple, np.ndarray)):
        if len(curve) == 0:
            return None
        return float(curve[-1])

    # scalar
    return float(curve)

def _prepare_plot_kwargs(
    args,
    results: Dict[str, MethodResult],
    *,
    plot_dir: str,
    filename: str,
    title: str,
    extra_kwargs: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Collect common plotting inputs for _plot_series_with_baselines."""

    os.makedirs(plot_dir, exist_ok=True)
    series, labels = [], []
    for key in ("ours_fr", "goat", "goat_classwise", "ours_eta"):
        if key in results:
            series.append(results[key].test_curve)
            labels.append(_display_name(key))

    baselines = []
    ref_line_value = None

    # Use Ours-FR as the reference method for baselines
    if "ours_fr" in results:
        base = results["ours_fr"]

        st_scalar = _last_as_float(base.st_curve)
        st_all_scalar = _last_as_float(base.st_all_curve)

        baselines.append((st_scalar, st_all_scalar))

        if base.em_acc is not None:
            ref_line_value = base.em_acc * 100.0

    kwargs: Dict[str, object] = dict(
        series=series,
        labels=labels,
        baselines=baselines,
        ref_line_value=ref_line_value,
        ref_line_label=f"EM ({args.em_match})",
        ref_line_style="--",
        title=title,
        ylabel="Accuracy",
        xlabel="Domain Index",
        save_path=os.path.join(plot_dir, filename),
    )
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    return kwargs

def _to_float_list(seq) -> List[float]:
    """Return a list of floats, accepting scalars, tensors, or sequences."""
    if seq is None:
        return []

    # Handle torch tensors explicitly if you use them
    try:
        if isinstance(seq, torch.Tensor):
            if seq.ndim == 0:
                return [float(seq.item())]
            return [float(x) for x in seq.detach().cpu().flatten()]
    except ImportError:
        pass

    # Scalars (float, int, numpy scalar, etc.)
    import numbers
    if isinstance(seq, numbers.Number):
        return [float(seq)]

    # Try treating as a sequence
    try:
        return [float(x) for x in seq]
    except TypeError:
        # Fallback: last resort, treat as scalar
        return [float(seq)]

def _append_curves_jsonl(
    curves_path: str,
    *,
    args,
    gt_domains: int,
    generated_domains: int,
    results: Dict[str, MethodResult],
    elapsed: float,
):
    """
    Append one JSON record with full train/test/ST curves for all methods.
    Each element in the curves corresponds to one adaptation step (domain).
    """
    os.makedirs(os.path.dirname(curves_path), exist_ok=True)

    methods_payload = {}
    for name, res in results.items():
        methods_payload[name] = {
            "train_curve": _to_float_list(res.train_curve),
            "test_curve": _to_float_list(res.test_curve),
            "st_curve": _to_float_list(res.st_curve),
            "st_all_curve": _to_float_list(res.st_all_curve),
            "generated_curve": _to_float_list(res.generated_curve),
            "em_acc": None if res.em_acc is None else float(res.em_acc),
            "duration_sec": None if res.duration_sec is None else float(res.duration_sec),
        }

    record = {
        "seed": int(getattr(args, "seed", -1)),
        "gt_domains": int(gt_domains),
        "generated_domains": int(generated_domains),
        "elapsed": float(elapsed),
        "methods": methods_payload,
    }

    with open(curves_path, "a") as f:
        f.write(json.dumps(record) + "\n")

def log_summary(
    log_path: str,
    *,
    args,
    gt_domains: int,
    generated_domains: int,
    results: Dict[str, MethodResult],
    elapsed: float,
):
    """Persist a short CSV row with the final accuracies AND the full curves."""

    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def _final_value(key: str) -> Optional[float]:
        if key not in results or not results[key].test_curve:
            return None
        return float(results[key].test_curve[-1])

    # 1) final-summary CSV-style line (unchanged behavior)
    parts: List[str] = []
    val = _final_value("ours_fr")
    if val is not None:
        parts.append(f"OursFR:{round(val, 2)}")
    val = _final_value("goat")
    if val is not None:
        parts.append(f"GOAT:{round(val, 2)}")
    val = _final_value("goat_classwise")
    if val is not None:
        parts.append(f"GOATCW:{round(val, 2)}")
    val = _final_value("ours_eta")
    if val is not None:
        parts.append(f"ETA:{round(val, 2)}")

    with open(log_path, "a") as f:
        f.write(
            f"seed{args.seed}with{gt_domains}gt{generated_domains}generated,{elapsed},"
            f"{','.join(parts)}\n"
        )

    # 2) full adaptation progression as JSONL
    curves_path = log_path.replace(".txt", "_curves.jsonl")
    _append_curves_jsonl(
        curves_path,
        args=args,
        gt_domains=gt_domains,
        generated_domains=generated_domains,
        results=results,
        elapsed=elapsed,
    )

# ---------------------------------------------------------------------------
# Specialized helpers (MNIST per-angle EM, cached encodings, etc.)
# ---------------------------------------------------------------------------


def load_encoded_domains(cache_dir: str, keys: Sequence[str]):
    """Utility to reload cached encodings by name/angle."""

    datasets = []
    for key in keys:
        path = os.path.join(cache_dir, f"encoded_{key}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Expected cached encoding at {path}, but the file is missing."
            )
        ds = torch.load(path, weights_only=False)
        datasets.append(_normalize_dataset_targets(ds))
    return datasets


def build_em_bundles_for_chain(
    args,
    *,
    raw_domains: Sequence,
    encoded_domains: Sequence,
    domain_keys: Sequence,
    teacher_model: nn.Module,
    n_classes: int,
    description_prefix: str = "domain",
    final_alias=None,
) -> Tuple[Dict[object, object], Dict[object, float], np.ndarray]:
    """
    Fit EM bundles for every non-source real domain in the chain (source, inter..., target).
    Mirrors the MNIST per-angle behavior but works for all datasets.
    """

    if len(raw_domains) != len(encoded_domains):
        raise ValueError("raw_domains and encoded_domains must have equal length.")
    if len(domain_keys) != len(raw_domains):
        raise ValueError("domain_keys must align with provided domains.")

    em_bundles: Dict[object, object] = {}
    em_accs: Dict[object, float] = {}

    # Teacher pseudo-labels on the final target serve as canonical mapping later.
    final_domain = raw_domains[-1]
    teacher_target_np = compute_pseudo_labels(teacher_model, final_domain, args)

    for key, raw_ds, enc_ds in zip(domain_keys[1:], raw_domains[1:], encoded_domains[1:]):
        pseudo_np = compute_pseudo_labels(teacher_model, raw_ds, args)
        args._cached_pseudolabels = pseudo_np

        description = f"{description_prefix}{key}"
        em_bundle = fit_em_bundle_for_dataset(
            args,
            enc_ds,
            raw_ds,
            n_classes=n_classes,
            description=description,
            cov_types=getattr(args, "em_cov_types", None),
            seeds=getattr(args, "em_seeds", None),
        )
        apply_em_bundle_to_target(em_bundle, enc_ds, raw_ds)

        em_bundles[key] = em_bundle
        if hasattr(enc_ds, "targets_em") and enc_ds.targets_em is not None:
            raw_targets = raw_ds.targets
            if not torch.is_tensor(raw_targets):
                raw_targets = torch.as_tensor(raw_targets, device=enc_ds.targets_em.device)
            else:
                raw_targets = raw_targets.to(enc_ds.targets_em.device)
            acc = (
                enc_ds.targets_em == raw_targets.long()
            ).float().mean().item()
            em_accs[key] = acc
            print(f"[EM] Ensemble accuracy at {key}: {acc * 100:.2f}%")

    args._cached_pseudolabels = teacher_target_np
    args._shared_em_per_domain = em_bundles
    last_key = domain_keys[-1]
    if last_key in em_bundles:
        args._shared_em = em_bundles[last_key]
        em_accs[last_key] = em_accs.get(last_key, None)

    if final_alias is not None and last_key in em_bundles:
        em_bundles[final_alias] = em_bundles[last_key]
        em_accs[final_alias] = em_accs.get(last_key, None)

    # Snapshot canonical target EM labels once; later method calls restore from this.
    _snapshot_target_em_labels(args, final_domain)
    if last_key in em_bundles and getattr(em_bundles[last_key], "labels_em", None) is not None:
        args._canonical_target_em_from_bundle = np.asarray(
            em_bundles[last_key].labels_em, dtype=np.int64
        ).copy()

    return em_bundles, em_accs, teacher_target_np


# Dataset-specific entry points will be added below.


def _build_rotated_domains(tgt_trainset, target: int, gt_domains: int):
    """Construct evenly spaced rotation angles between 0 and ``target``."""

    all_sets, deg_idx = [], []
    for i in range(1, gt_domains + 1):
        angle = i * target // (gt_domains + 1)
        all_sets.append(get_single_rotate(False, angle))
        deg_idx.append(angle)
    all_sets.append(tgt_trainset)
    deg_idx.append(target)
    return all_sets, deg_idx


def run_mnist_experiment(target: int, gt_domains: int, generated_domains: int, args=None):
    """Refactored MNIST pipeline that reuses the shared helpers above."""

    args = _require_args(args)
    t0 = time.time()

    # ------------ data + cache dirs ------------
    src_trainset = get_single_rotate(False, 0)
    tgt_trainset = get_single_rotate(False, target)
    all_sets, deg_idx = _build_rotated_domains(tgt_trainset, target, gt_domains)
    cache_dir = f"cache{args.ssl_weight}/target{target}/small_dim{args.small_dim}/"

    # ------------ reference model ------------
    model_cfg = ModelConfig(
        encoder_builder=ENCODER,
        mode="mnist",
        n_class=10,
        epochs=10,
        model_path=os.path.join(
            "/data/common/yuenchen/GDA/mnist_models/",
            f"src0_tgt{target}_ssl{args.ssl_weight}_dim{args.small_dim}.pth",
        ),
        compress=True,
        in_dim=25088,
        out_dim=args.small_dim,
    )

    ref_model, ref_encoder = build_reference_model(
        args, model_cfg, src_trainset, tgt_trainset
    )

    # ------------ encode real domains ------------
    e_src, e_tgt, encoded_intersets = encode_real_domains(
        args,
        ref_encoder=ref_encoder,
        src_trainset=src_trainset,
        tgt_trainset=tgt_trainset,
        all_sets=all_sets,
        deg_idx=deg_idx,
        cache_dir=cache_dir,
        target_label=target,
    )

    shared_pca = fit_global_pca(
        domains=encoded_intersets,
        classes=None,
        pool="auto",
        n_components=2,
        per_domain_cap=10000,
        random_state=args.seed if hasattr(args, "seed") else 0,
    )
    args.shared_pca = shared_pca

    # Optional: cache source Gaussian params for prototype matching
    if getattr(args, "em_match", "pseudo") == "prototypes":
        mu_s, Sigma_s, priors_s = fit_source_gaussian_params(
            X=e_src.data, y=e_src.targets
        )
        args._cached_source_stats = (mu_s, Sigma_s, priors_s)

    # ------------ EM on each real domain (only if we will actually use it) ------------
    if generated_domains > 0:
        with torch.no_grad():
            teacher = copy.deepcopy(ref_model).to(device).eval()

        raw_domains = [src_trainset] + all_sets
        angle_list = [0] + deg_idx
        encoded_domains = load_encoded_domains(cache_dir, angle_list)

        for ang, enc_ds in zip(angle_list, encoded_domains):
            print(
                f"[DEBUG] angle={ang} encoded shape={enc_ds.data.shape}, "
                f"first_norm={enc_ds.data[0].norm().item():.4f}"
            )

        em_bundles, em_accs, _ = build_em_bundles_for_chain(
            args,
            raw_domains=raw_domains,
            encoded_domains=encoded_domains,
            domain_keys=angle_list,
            teacher_model=teacher,
            n_classes=model_cfg.n_class,
            description_prefix="mnist-angle-",
        )
        em_acc_target = em_accs.get(target, 0.0)
        print(
            f"[MNIST-EXP] Initial target EM accuracy ({args.em_match}): "
            f"{em_acc_target * 100:.2f}%"
        )
    else:
        # No EM fitting in the GST / gen=0 case
        em_acc_target = float("nan")

    # ------------ run main methods (ours_fr, ours_eta, GOAT, GOAT-CW) ------------
    results = run_core_methods(
        args,
        ref_model=ref_model,
        src_trainset=src_trainset,
        tgt_trainset=tgt_trainset,
        all_sets=all_sets,
        deg_idx=deg_idx,
        generated_domains=generated_domains,
        target_label=target,
    )

    # ------------ unified naming for plots + logs ------------
    base_name = (
        f"test_acc_dim{args.small_dim}_int{gt_domains}_gen{generated_domains}_"
        f"{args.label_source}_{args.em_match}_{args.em_select}"
        f"{'_em-ensemble' if args.em_ensemble else ''}"
    )
    plot_filename = f"{base_name}.png"
    log_filename = f"{base_name}.txt"

    # For generated_domains = 0, we disable synthetic/real boundary markers
    if generated_domains > 0:
        synth_per_segment = int(generated_domains)
        n_real_segments = len(all_sets)
    else:
        synth_per_segment = None
        n_real_segments = None

    # ------------ plot ------------
    _plot_series_with_baselines(
        **_prepare_plot_kwargs(
            args,
            results,
            plot_dir=os.path.join(_plots_base_dir(args), f"target{target}"),
            filename=plot_filename,
            title=(
                f"Target Accuracy (ST: {args.label_source} labels; "
                f"Cluster Map: {args.em_match})"
            ),
            extra_kwargs={
                "synth_per_segment": synth_per_segment,
                "n_real_segments": n_real_segments,
            },
        )
    )

    # ------------ logs (always, including gen=0) ------------
    log_dir = os.path.join(_logs_base_dir(args), f"target{target}")
    os.makedirs(log_dir, exist_ok=True)
    log_summary(
        os.path.join(log_dir, log_filename),
        args=args,
        gt_domains=gt_domains,
        generated_domains=generated_domains,
        results=results,
        elapsed=time.time() - t0,
    )

    return results


def run_portraits_experiment(gt_domains: int, generated_domains: int, args=None):
    """Portraits pipeline with shared orchestration utilities."""

    args = _require_args(args)
    t0 = time.time()
    (
        src_tr_x,
        src_tr_y,
        src_val_x,
        src_val_y,
        inter_x,
        inter_y,
        _,
        _,
        trg_val_x,
        trg_val_y,
        trg_test_x,
        trg_test_y,
    ) = make_portraits_data(1000, 1000, 14000, 2000, 1000, 1000)

    tr_x = np.concatenate([src_tr_x, src_val_x])
    tr_y = np.concatenate([src_tr_y, src_val_y])
    ts_x = np.concatenate([trg_val_x, trg_test_x])
    ts_y = np.concatenate([trg_val_y, trg_test_y])

    transforms = ToTensor()
    src_trainset = EncodeDataset(tr_x, tr_y.astype(int), transforms)
    tgt_trainset = EncodeDataset(ts_x, ts_y.astype(int), transforms)

    model_dir = f"portraits/cache{args.ssl_weight}/"
    model_cfg = ModelConfig(
        encoder_builder=ENCODER,
        mode="portraits",
        n_class=2,
        epochs=20,
        model_path=os.path.join(
            model_dir,
            f"ssl{args.ssl_weight}_dim{args.small_dim}.pth",
        ),
        compress=True,
        in_dim=32768,
        out_dim=args.small_dim,
    )

    def get_domains(n_domains):
        domain_set = []
        n2idx = {0: [], 1: [3], 2: [2, 4], 3: [1, 3, 5], 4: [0, 2, 4, 6], 7: [0, 1, 2, 3, 4, 5, 6]}
        domain_idx = n2idx.get(n_domains, [])
        for i in domain_idx:
            start, end = i * 2000, (i + 1) * 2000
            domain_set.append(EncodeDataset(inter_x[start:end], inter_y[start:end].astype(int), transforms))
        return domain_set

    all_sets = get_domains(gt_domains)
    all_sets.append(tgt_trainset)
    ref_model, ref_encoder = build_reference_model(args, model_cfg, src_trainset, tgt_trainset)
    e_src, e_tgt, encoded_intersets = encode_real_domains(
        args,
        ref_encoder=ref_encoder,
        src_trainset=src_trainset,
        tgt_trainset=tgt_trainset,
        all_sets=all_sets,
        deg_idx=0,
        cache_dir=f"portraits/cache{args.ssl_weight}/small_dim{args.small_dim}/",
        target_label=1,
    )
    attach_shared_pca(args, encoded_intersets)
    maybe_cache_source_stats(args, e_src)

    # Fit EM bundles only when we will actually generate/use synthetic domains.
    if generated_domains > 0:
        teacher = copy.deepcopy(ref_model).to(device).eval()
        raw_domains = [src_trainset] + all_sets
        domain_keys = ["source"] + [f"real{i}" for i in range(len(all_sets) - 1)] + ["target"]
        em_bundles, em_accs, teacher_target_np = build_em_bundles_for_chain(
            args,
            raw_domains=raw_domains,
            encoded_domains=encoded_intersets,
            domain_keys=domain_keys,
            teacher_model=teacher,
            n_classes=model_cfg.n_class,
            description_prefix="portraits-",
            final_alias=1,
        )
        args._cached_pseudolabels = teacher_target_np
        em_acc_now = em_accs.get("target") or em_accs.get(1) or 0.0
        print(f"[Portraits] EM→class accuracy: {em_acc_now:.4f}")
    else:
        print("[Portraits] generated_domains=0: skipping EM bundle fitting.")

    results = run_core_methods(
        args,
        ref_model=ref_model,
        src_trainset=src_trainset,
        tgt_trainset=tgt_trainset,
        all_sets=all_sets,
        deg_idx=0,
        generated_domains=generated_domains,
        target_label=1,
    )

    # ---------- unified naming ----------
    base_name = (
        f"test_acc_dim{args.small_dim}_int{gt_domains}_gen{generated_domains}_"
        f"{args.label_source}_{args.em_match}_{args.em_select}"
        f"{'_em-ensemble' if args.em_ensemble else ''}"
    )

    # ---------- plot ----------
    _plot_series_with_baselines(
        **_prepare_plot_kwargs(
            args,
            results,
            plot_dir=_plots_base_dir(args),
            filename=f"{base_name}.png",
            title=(
                f"Portraits: Target Accuracy (ST: {args.label_source}; "
                f"Cluster Map: {args.em_match})"
            ),
            extra_kwargs={
                "synth_per_segment": int(generated_domains),
                "n_real_segments": len(all_sets),
            },
        )
    )

    # ---------- logs ----------
    log_dir = _logs_base_dir(args)
    os.makedirs(log_dir, exist_ok=True)

    log_summary(
        os.path.join(log_dir, f"{base_name}.txt"),
        args=args,
        gt_domains=gt_domains,
        generated_domains=generated_domains,
        results=results,
        elapsed=time.time() - t0,
    )

    return results


def run_covtype_experiment(gt_domains: int, generated_domains: int, args=None):
    """CovType pipeline refactored to use the shared helpers."""

    args = _require_args(args)
    t0 = time.time()
    (
        src_tr_x,
        src_tr_y,
        src_val_x,
        src_val_y,
        inter_x,
        inter_y,
        _,
        _,
        trg_val_x,
        trg_val_y,
        trg_test_x,
        trg_test_y,
    ) = make_cov_data(40000, 10000, 400000, 50000, 25000, 20000)

    src_x = np.concatenate([src_tr_x, src_val_x])
    src_y = np.concatenate([src_tr_y, src_val_y])
    tgt_x = np.concatenate([trg_val_x, trg_test_x])
    tgt_y = np.concatenate([trg_val_y, trg_test_y])

    src_trainset = EncodeDataset(torch.from_numpy(src_x).float(), src_y.astype(int))
    tgt_trainset = EncodeDataset(torch.from_numpy(tgt_x).float(), tgt_y.astype(int))

    encoder = MLP_Encoder().to(device)

    def infer_out_dim(encoder_module, dataset_sample):
        encoder_module = encoder_module.to(device).eval()
        with torch.no_grad():
            z = encoder_module(dataset_sample.to(device))
            z = z.view(z.size(0), -1)
        return int(z.size(1))

    enc_out_dim = infer_out_dim(encoder, torch.as_tensor(src_trainset.data[:2]).float())
    if enc_out_dim < args.small_dim:
        args.small_dim = enc_out_dim

    model_dir = f"covtype/cache{args.ssl_weight}/"
    model_cfg = ModelConfig(
        encoder_builder=MLP_Encoder,
        mode="covtype",
        n_class=2,
        epochs=10,
        model_path=os.path.join(
            model_dir,
            f"ssl{args.ssl_weight}_dim{args.small_dim}.pth",
        ),
        compress=args.small_dim < enc_out_dim,
        in_dim=enc_out_dim,
        out_dim=args.small_dim,
    )

    def get_domains(n_domains: int) -> List[EncodeDataset]:
        idx_map = {
            0: [],
            1: [6],
            2: [3, 7],
            3: [2, 5, 8],
            4: [2, 4, 6, 8],
            5: [1, 3, 5, 7, 9],
            10: list(range(10)),
            200: list(range(200)),
        }
        domain_idx = idx_map.get(n_domains, [])
        domains = []
        for i in domain_idx:
            start, end = i * 40000, i * 40000 + 2000
            Xi = torch.from_numpy(inter_x[start:end]).float()
            yi = inter_y[start:end].astype(int)
            domains.append(EncodeDataset(Xi, yi))
        return domains

    all_sets = get_domains(gt_domains)
    all_sets.append(tgt_trainset)

    ref_model, ref_encoder = build_reference_model(args, model_cfg, src_trainset, tgt_trainset)
    e_src, e_tgt, encoded_intersets = encode_real_domains(
        args,
        ref_encoder=ref_encoder,
        src_trainset=src_trainset,
        tgt_trainset=tgt_trainset,
        all_sets=all_sets,
        deg_idx=0,
        cache_dir=f"covtype/cache{args.ssl_weight}/small_dim{args.small_dim}/",
        target_label=1,
    )
    attach_shared_pca(args, encoded_intersets)
    maybe_cache_source_stats(args, e_src)

    # Fit EM bundles only when we will actually generate/use synthetic domains.
    if generated_domains > 0:
        teacher = copy.deepcopy(ref_model).to(device).eval()
        raw_domains = [src_trainset] + all_sets
        domain_keys = ["source"] + [f"real{i}" for i in range(len(all_sets) - 1)] + ["target"]
        em_bundles, em_accs, teacher_target_np = build_em_bundles_for_chain(
            args,
            raw_domains=raw_domains,
            encoded_domains=encoded_intersets,
            domain_keys=domain_keys,
            teacher_model=teacher,
            n_classes=model_cfg.n_class,
            description_prefix="covtype-",
            final_alias=1,
        )
        args._cached_pseudolabels = teacher_target_np
        em_acc_now = em_accs.get("target") or em_accs.get(1) or 0.0
        print(f"[CovType] EM→class accuracy: {em_acc_now:.4f}")
    else:
        print("[CovType] generated_domains=0: skipping EM bundle fitting.")

    results = run_core_methods(
        args,
        ref_model=ref_model,
        src_trainset=src_trainset,
        tgt_trainset=tgt_trainset,
        all_sets=all_sets,
        deg_idx=0,
        generated_domains=generated_domains,
        target_label=1,
    )

    # ---------- unified naming ----------
    base_name = (
        f"test_acc_dim{args.small_dim}_int{gt_domains}_gen{generated_domains}_"
        f"{args.label_source}_{args.em_match}_{args.em_select}"
        f"{'_em-ensemble' if args.em_ensemble else ''}"
    )

    # ---------- plot ----------
    _plot_series_with_baselines(
        **_prepare_plot_kwargs(
            args,
            results,
            plot_dir=_plots_base_dir(args),
            filename=f"{base_name}.png",
            title=(
                f"CovType: Target Accuracy (ST: {args.label_source}; "
                f"Cluster Map: {args.em_match})"
            ),
            extra_kwargs={
                "synth_per_segment": int(generated_domains),
                "n_real_segments": len(all_sets),
            },
        )
    )

    # ---------- log ----------
    log_dir = _logs_base_dir(args)
    os.makedirs(log_dir, exist_ok=True)

    log_summary(
        os.path.join(log_dir, f"{base_name}.txt"),
        args=args,
        gt_domains=gt_domains,
        generated_domains=generated_domains,
        results=results,
        elapsed=time.time() - t0,
    )

    return results


def run_color_mnist_experiment(gt_domains: int, generated_domains: int, args=None):
    """Colored-MNIST experiment with shared infrastructure."""

    args = _require_args(args)
    t0 = time.time()
    shift = 1
    total_domains = 20
    (
        src_tr_x,
        src_tr_y,
        src_val_x,
        src_val_y,
        dir_inter_x,
        dir_inter_y,
        _,
        _,
        trg_val_x,
        trg_val_y,
        trg_test_x,
        trg_test_y,
    ) = ColorShiftMNIST(shift=shift)

    inter_x, inter_y = transform_inter_data(
        dir_inter_x,
        dir_inter_y,
        0,
        shift,
        interval=len(dir_inter_x) // total_domains,
        n_domains=total_domains,
    )
    src_x = np.concatenate([src_tr_x, src_val_x])
    src_y = np.concatenate([src_tr_y, src_val_y])
    tgt_x = np.concatenate([trg_val_x, trg_test_x])
    tgt_y = np.concatenate([trg_val_y, trg_test_y])

    to_tensor = ToTensor()
    src_trainset = EncodeDataset(src_x, src_y.astype(int), to_tensor)
    tgt_trainset = EncodeDataset(tgt_x, tgt_y.astype(int), to_tensor)

    encoder = ENCODER().to(device)

    def infer_out_dim(dataset, encoder_module):
        encoder_module = encoder_module.to(device).eval()
        with torch.no_grad():
            xb = torch.stack([dataset[i][0] for i in range(1)], dim=0).to(device).float()
            z = encoder_module(xb).reshape(xb.size(0), -1)
        return int(z.size(1))

    enc_out_dim = infer_out_dim(src_trainset, encoder)
    if enc_out_dim < args.small_dim:
        args.small_dim = enc_out_dim

    model_dir = f"color_mnist/cache{args.ssl_weight}"
    model_cfg = ModelConfig(
        encoder_builder=ENCODER,
        mode="mnist",
        n_class=int(np.unique(src_trainset.targets).shape[0]),
        epochs=20,
        model_path=os.path.join(
            model_dir,
            f"ssl{args.ssl_weight}_dim{args.small_dim}.pth",
        ),
        compress=True,
        in_dim=enc_out_dim,
        out_dim=args.small_dim,
    )

    def get_domains(n_domains: int) -> List[EncodeDataset]:
        if n_domains == total_domains:
            domain_idx = list(range(n_domains))
        else:
            domain_idx = [
                total_domains // (n_domains + 1) * i for i in range(1, n_domains + 1)
            ]
        interval = len(inter_x) // total_domains
        domains = []
        for i in domain_idx:
            start, end = i * interval, (i + 1) * interval
            domains.append(EncodeDataset(inter_x[start:end], inter_y[start:end].astype(int), to_tensor))
        return domains

    all_sets = get_domains(gt_domains)
    all_sets.append(tgt_trainset)

    ref_model, ref_encoder = build_reference_model(args, model_cfg, src_trainset, tgt_trainset)
    e_src, e_tgt, encoded_intersets = encode_real_domains(
        args,
        ref_encoder=ref_encoder,
        src_trainset=src_trainset,
        tgt_trainset=tgt_trainset,
        all_sets=all_sets,
        deg_idx=0,
        cache_dir=f"color_mnist/cache{args.ssl_weight}/small_dim{args.small_dim}/",
        target_label=1,
    )
    shared_pca = fit_global_pca(
        domains=encoded_intersets,
        classes=None,
        pool="auto",
        n_components=2,
        per_domain_cap=10000,
        random_state=args.seed if hasattr(args, "seed") else 0,
    )
    args.shared_pca = shared_pca

    # Optional: cache source Gaussian params for prototype matching
    if getattr(args, "em_match", "pseudo") == "prototypes":
        mu_s, Sigma_s, priors_s = fit_source_gaussian_params(
            X=e_src.data, y=e_src.targets
        )
        args._cached_source_stats = (mu_s, Sigma_s, priors_s)
    # Fit EM bundles only when we will actually generate/use synthetic domains.
    if generated_domains > 0:
        # ---- (E) Frozen teacher for pseudo-labels on every REAL domain ----
        with torch.no_grad():
            teacher = copy.deepcopy(ref_model).to(device).eval()

        raw_domains = [src_trainset] + all_sets
        domain_keys = ["source"] + [f"real{i}" for i in range(len(all_sets) - 1)] + ["target"]
        em_bundles, em_accs, teacher_target_np = build_em_bundles_for_chain(
            args,
            raw_domains=raw_domains,
            encoded_domains=encoded_intersets,
            domain_keys=domain_keys,
            teacher_model=teacher,
            n_classes=model_cfg.n_class,
            description_prefix="color-mnist-",
            final_alias=1,
        )
        args._cached_pseudolabels = teacher_target_np
        em_acc_now = em_accs.get("target") or em_accs.get(1) or 0.0
        print(f"[ColorMNIST] EM→class accuracy: {em_acc_now:.4f}")
    else:
        print("[ColorMNIST] generated_domains=0: skipping EM bundle fitting.")
    src_trainset.targets_em = src_trainset.targets  # for plotting
    results = run_core_methods(
        args,
        ref_model=ref_model,
        src_trainset=src_trainset,
        tgt_trainset=tgt_trainset,
        all_sets=all_sets,
        deg_idx=0,
        generated_domains=generated_domains,
        target_label=1,
    )

    _plot_series_with_baselines(
        **_prepare_plot_kwargs(
            args,
            results,
            plot_dir=_plots_base_dir(args),
            filename=(
                f"test_acc_dim{args.small_dim}_int{gt_domains}_gen{generated_domains}_"
                f"{args.label_source}_{args.em_match}_{args.em_select}"
                f"{'_em-ensemble' if args.em_ensemble else ''}.png"
            ),
            title=(
                f"Colored-MNIST: Target Accuracy (ST: {args.label_source}; "
                f"Cluster Map: {args.em_match})"
            ),
            extra_kwargs={
                "synth_per_segment": int(generated_domains),
                "n_real_segments": len(all_sets),
            },
        )
    )
    log_dir = _logs_base_dir(args)
    os.makedirs(log_dir, exist_ok=True)
    base_name = (
        args.log_file
        if args.log_file
        else (
            f"test_acc_dim{args.small_dim}_int{gt_domains}_gen{generated_domains}_"
            f"{args.label_source}_{args.em_match}_{args.em_select}"
            f"{'_em-ensemble' if args.em_ensemble else ''}"
        )
    )
    log_summary(
        os.path.join(log_dir, f"{base_name}.txt"),
        args=args,
        gt_domains=gt_domains,
        generated_domains=generated_domains,
        results=results,
        elapsed=time.time() - t0,
    )
    return results


# -------------------------------------------------------------
# Main / CLI
# -------------------------------------------------------------


def main(cli_args: argparse.Namespace) -> None:
    set_all_seeds(cli_args.seed)
    print(cli_args)

    if cli_args.dataset == "mnist":
        if cli_args.mnist_mode == "normal":
            run_mnist_experiment(
                cli_args.rotation_angle,
                cli_args.gt_domains,
                cli_args.generated_domains,
                args=cli_args,
            )
        elif cli_args.mnist_mode == "ablation":
            run_mnist_ablation(
                cli_args.rotation_angle,
                cli_args.gt_domains,
                cli_args.generated_domains,
            )
        else:
            raise ValueError(f"Unknown mnist-mode: {cli_args.mnist_mode}")
    elif cli_args.dataset == "portraits":
        run_portraits_experiment(
            cli_args.gt_domains,
            cli_args.generated_domains,
            args=cli_args,
        )
    elif cli_args.dataset == "covtype":
        run_covtype_experiment(
            cli_args.gt_domains,
            cli_args.generated_domains,
            args=cli_args,
        )
    elif cli_args.dataset == "color_mnist":
        run_color_mnist_experiment(
            cli_args.gt_domains,
            cli_args.generated_domains,
            args=cli_args,
        )
    else:
        raise ValueError(f"Unknown dataset: {cli_args.dataset}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GOAT experiments (refactored)")
    parser.add_argument(
        "--dataset",
        choices=["mnist", "portraits", "covtype", "color_mnist"],
        default="mnist",
    )
    parser.add_argument("--gt-domains", type=int, default=0)
    parser.add_argument("--generated-domains", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mnist-mode",
        choices=["normal", "ablation", "sweep", "compare"],
        default="normal",
    )
    parser.add_argument("--rotation-angle", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--log-root",
        type=str,
        default=os.environ.get("LOG_ROOT", "logs_rerun"),
        help="Root directory for experiment logs.",
    )
    parser.add_argument("--log-file", type=str, default="")
    parser.add_argument("--ssl-weight", type=float, default=0.1)
    parser.add_argument(
        "--use-labels",
        action="store_true",
        help="Use true labels when available in generators",
    )
    parser.add_argument(
        "--diet",
        action="store_true",
        help="Run DIET to refine encoder before CE training",
    )
    parser.add_argument(
        "--small-dim",
        type=int,
        default=2048,
        help="Add a small-dim compressor before the head (0 to disable)",
    )
    parser.add_argument(
        "--label-source",
        choices=["pseudo", "em"],
        default="pseudo",
        help="For self-training, which labels to use for pseudo-labeling",
    )
    parser.add_argument(
        "--em-match",
        choices=["pseudo", "prototypes", "none"],
        default="pseudo",
        help="For self-training, which labels to use for pseudo-labeling",
    )
    parser.add_argument(
        "--em-ensemble",
        action="store_true",
        help="Whether to ensemble multiple EM models",
    )
    parser.add_argument(
        "--em-select",
        choices=["bic", "cost", "ll"],
        default="bic",
        help="Criterion to select best EM model",
    )
    args = parser.parse_args()
    main(args)
