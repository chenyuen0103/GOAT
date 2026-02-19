from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple, Union
import math
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.mixture import GaussianMixture
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable=None, total=None, **kwargs):
        return iterable if iterable is not None else range(total or 0)
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import base
from sklearn.cluster import kmeans_plusplus
# Global in-memory cache; key -> EM bundle
_EM_REGISTRY: Dict[str, "EMBundle"] = {}
@dataclass
class EMBundle:
    key: str
    em_res: Dict[str, Any]                 # raw run_em_on_encoded(...) result
    mapping: Dict[int, int]                # cluster_id -> class_id mapping
    labels_em: np.ndarray                  # (N,) mapped hard labels on target
    P_soft: Optional[np.ndarray] = None    # (N,K_classes) mapped soft class posteriors (optional)
    info: Dict[str, Any] = None            # any diagnostics (BIC, ll, etc.)

def _em_cache_key(
    dataset: str,
    target_angle: int,
    small_dim: int,
    pool: str,
    do_pca: bool,
    pca_dim: int,
    cov_type: str,
    K: int,
    seed: int,
    dtype: str = "float32",
) -> str:
    # include knobs that affect the fitted model / feature space
    return (f"{dataset}|t={target_angle}|d={small_dim}|pool={pool}"
            f"|pca={int(do_pca)}:{pca_dim}|cov={cov_type}|K={K}"
            f"|seed={seed}|dtype={dtype}")

def ensure_shared_em_for_target(
    e_tgt,                       # encoded target domain (same object used by all methods)
    *,
    args,
    K: Optional[int] = None,     # classes; None -> infer
    cov_type: str = "diag",
    pool: str = "gap",
    do_pca: bool = False,
    pca_dim: int = 64,
    reg: float = 1e-6,
    max_iter: int = 150,
    tol: float = 1e-5,
    n_init: int = 5,
    dtype: str = "float32",
) -> EMBundle:
    """
    Fit EM exactly once on e_tgt (or reuse from cache), map clusters to classes per args.em_match,
    and return a reusable EMBundle.
    """
    # 1) infer K from labels (fallback=10)
    if K is None:
        try:
            y_true = e_tgt.targets.cpu().numpy() if torch.is_tensor(e_tgt.targets) else np.asarray(e_tgt.targets)
            K = int(y_true.max()) + 1
        except Exception:
            K = 10

    # 2) cache key
    dataset = getattr(args, "dataset", "mnist")
    key = _em_cache_key(
        dataset=dataset,
        target_angle=getattr(args, "target_angle", getattr(args, "target", 0)),
        small_dim=getattr(args, "small_dim", 2048),
        pool=pool, do_pca=do_pca, pca_dim=pca_dim,
        cov_type=cov_type, K=K,
        seed=getattr(args, "seed", 0),
        dtype=dtype,
    )
    if key in _EM_REGISTRY:
        bundle = _EM_REGISTRY[key]
        # attach for convenience
        args._shared_em = bundle
        return bundle

    # 3) prepare standardized features ONCE and fit EM ONCE
    X_std, scaler, pca, _ = prepare_em_representation(
        e_tgt, pool=pool, do_pca=do_pca, pca_dim=pca_dim,
        dtype=dtype, rng=getattr(args, "seed", 0), verbose=False
    )
    em_res = run_em_on_encoded_fast(
        X_std,
        K=K, cov_type=cov_type, reg=reg, max_iter=max_iter, tol=tol,
        n_init=n_init, subsample_init=len(e_tgt),
        warm_start=None, rng=getattr(args, "seed", 0), verbose=False
    )
    em_res["scaler"] = scaler
    em_res["pca"] = pca

    # 4) map clusters -> classes (shared for all methods)
    #    Note: we reuse your existing mapping choices.
    method = getattr(args, "em_match", "pseudo")
    if method == "prototypes":
        mu_s, Sigma_s, priors_s = fit_source_gaussian_params(X=e_tgt.data, y=e_tgt.targets)
        mapping, labels_mapped, _ = map_em_clusters(
            em_res, method="prototypes", n_classes=K,
            mus_s=mu_s, Sigma_s=Sigma_s, priors_s=priors_s
        )
    else:
        # Expect args._cached_pseudolabels to be set by caller once (teacher PL)
        if not hasattr(args, "_cached_pseudolabels"):
            raise RuntimeError("args._cached_pseudolabels missing; compute teacher pseudo-labels once, then call ensure_shared_em_for_target.")
        mapping, labels_mapped, _ = map_em_clusters(
            em_res, method="pseudo", n_classes=K,
            pseudo_labels=args._cached_pseudolabels, metric='FR'
        )

    P_soft = _map_cluster_posts_to_classes(np.asarray(em_res["gamma"]), mapping, K)

    bundle = EMBundle(
        key=key, em_res=em_res, mapping=mapping,
        labels_em=np.asarray(labels_mapped, dtype=int),
        P_soft=P_soft, info={"K": K, "cov_type": cov_type, "pool": pool, "pca_dim": pca_dim, "dtype": dtype},
    )
    _EM_REGISTRY[key] = bundle
    # attach to args so children don’t have to compute the key
    args._shared_em = bundle
    return bundle


def build_em_bundle(em_models, args):
    """
    Returns an EMBundle either from a trimmed BIC-weighted ensemble
    or from the single best model selected by `args.em_select`.

    Ensemble path:
      - Trim by ΔBIC
      - Weight with exp(-0.5 * ΔBIC)
      - Average class-space posteriors
      - Use the top-weighted trimmed model as a metadata anchor

    Single-model path:
      - Use the best model under `args.em_select`
    """
    # Always select a 'best' for the single-model fallback
    best = select_best_em(em_models, criterion=args.em_select)

    use_ensemble = bool(args.em_ensemble) and len(em_models) > 1
    if use_ensemble:
        # Modify the ensemble logic to skip trimming and assign equal weights
        trimmed = em_models  # Use all models without trimming
        w = np.ones(len(trimmed)) / len(trimmed)  # Assign equal weights to all models

        P_ens, H_ens = ensemble_posteriors_trimmed(trimmed, w)
        labels_ens = P_ens.argmax(axis=1).astype(int)
        # Deterministic anchor: top-weight model among trimmed
        anchor_idx = int(np.argmax(w))
        anchor = trimmed[anchor_idx]
        # breakpoint()
        return EMBundle(
            key="multi_em_trimmed",
            em_res=anchor["em_res"],          # carry scaler/PCA/diagnostics from an actual fitted model
            mapping=None,                     # ensemble already in class space; no per-model mapping needed
            labels_em=labels_ens,
            P_soft=P_ens,
            info={
                "criterion": "bic+trim_ensemble",
                "bic_best": float(min(m["bic"] for m in trimmed)),
                "weights": [float(wi) for wi in w.tolist()],
                "bics": [float(m["bic"]) for m in trimmed],
                "anchor_idx": anchor_idx,
            },
        )

    # Single-model fallback
    return EMBundle(
        key="multi_em_best",
        em_res=best["em_res"],
        mapping=best.get("mapping", None),
        labels_em=np.asarray(best["labels_mapped"], dtype=int),
        P_soft=np.asarray(best["mapped_soft"], dtype=float),
        info={"criterion": args.em_select, "bic": float(best["bic"]), "ll": float(best["final_ll"]), "cost": float(best.get("cost", math.nan))},
    )


def _map_cluster_posts_to_classes(gamma: np.ndarray, mapping: Dict[int, int], K: int) -> np.ndarray:
    """
    Map cluster responsibilities (N x K_clusters) to class posteriors (N x K_classes)
    using a cluster->class mapping. If multiple clusters map to same class, sum them.
    """
    n = gamma.shape[0]
    out = np.zeros((n, K), dtype=float)
    for c_from, c_to in mapping.items():
        if 0 <= c_from < gamma.shape[1] and 0 <= c_to < K:
            out[:, c_to] += gamma[:, c_from]
    # Normalize row-wise to guard against numerical drift
    row_sums = out.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return out / row_sums

def _soft_average(p_list: List[np.ndarray], weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Weighted average of a list of (N x K) probability matrices."""
    assert len(p_list) > 0
    N, K = p_list[0].shape
    W = np.ones(len(p_list)) if weights is None else np.asarray(weights, dtype=float)
    W = W / W.sum()
    acc = np.zeros((N, K), dtype=float)
    for w, P in zip(W, p_list):
        acc += w * P
    # normalize for safety (should already be normalized)
    acc /= acc.sum(axis=1, keepdims=True)
    return acc

def _entropy(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise entropy for (N x K) probs."""
    q = np.clip(p, eps, 1.0)
    return -(q * np.log(q)).sum(axis=1)




# ---------- Ensemble trimming, entropy, and soft-target finetune utilities ----------

def trim_em_models_by_bic(em_models, max_delta_bic: float = 10.0):
    """
    Keep only EM runs whose BIC is within max_delta_bic of the best (lowest) BIC.
    Returns (trimmed_models, weights) where weights are BIC-based (exp(-0.5 * ΔBIC)).
    """
    bics = np.array([m["bic"] for m in em_models], dtype=float)
    if not np.all(np.isfinite(bics)):
        # Guard: drop NaN/inf BIC runs first
        keep = np.isfinite(bics)
        em_models = [m for i, m in enumerate(em_models) if keep[i]]
        bics = bics[keep]
        if len(em_models) == 0:
            raise RuntimeError("All EM runs have non-finite BIC; fix EM numerics first.")

    best = float(np.min(bics))
    deltas = bics - best
    keep = deltas <= max_delta_bic
    trimmed = [m for i, m in enumerate(em_models) if keep[i]]
    kept_deltas = deltas[keep]

    # BIC weights ~ exp(-0.5 * ΔBIC)
    w = np.exp(-0.5 * kept_deltas)
    w = w / w.sum()

    return trimmed, w


def ensemble_posteriors_trimmed(em_models_trimmed, weights=None):
    """
    Weighted average of mapped class posteriors from trimmed EM models.
    em_models_trimmed: list of dicts with 'mapped_soft' (N x K).
    weights: numpy array summing to 1 (len == len(em_models_trimmed)); if None, uniform.
    Returns P_ens (N x K) and per-sample entropy (N,).
    """
    P_list = [np.asarray(m["mapped_soft"], dtype=float) for m in em_models_trimmed]
    N, K = P_list[0].shape
    if weights is None:
        weights = np.ones(len(P_list), dtype=float) / len(P_list)
    else:
        weights = np.asarray(weights, dtype=float)
        weights = weights / weights.sum()

    P_ens = np.zeros((N, K), dtype=float)
    for w, P in zip(weights, P_list):
        P_ens += w * P

    # normalize defensively
    P_ens /= np.clip(P_ens.sum(axis=1, keepdims=True), 1e-12, None)

    # entropy (natural log)
    Q = np.clip(P_ens, 1e-12, 1.0)
    H = -(Q * np.log(Q)).sum(axis=1)

    return P_ens, H


def _lowest_entropy_mask(H: np.ndarray, p_keep: float):
    """
    Return boolean mask selecting the lowest-entropy p% samples.
    p_keep in [0, 100]. Example: p_keep=60 selects the lowest 60% entropy samples.
    """
    assert 0.0 <= p_keep <= 100.0
    if p_keep >= 100.0:
        return np.ones_like(H, dtype=bool)
    if p_keep <= 0.0:
        return np.zeros_like(H, dtype=bool)
    q = np.percentile(H, p_keep)
    return H <= q


class _SoftTargetSubset(Dataset):
    """
    Wrap a base dataset (images, labels) to provide only a subset of indices,
    and to attach a soft target distribution per sample.
    """
    def __init__(self, base_dataset, indices, soft_targets: np.ndarray):
        self.base = base_dataset
        self.indices = np.asarray(indices, dtype=int)
        # soft_targets assumed to be N x K over the FULL target set; we slice by indices
        self.soft = soft_targets[self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        base_idx = int(self.indices[i])
        x, _y = self.base[base_idx]  # ignore original hard label for this fine-tune
        # Return x and soft target (as float32 tensor)
        return x, torch.from_numpy(self.soft[i]).float()


def _soft_ce_loss_from_probs(logits: torch.Tensor, soft_targets: torch.Tensor):
    """
    Soft cross-entropy (a.k.a. KL with one-hot replaced by soft probs, temperature=1).
    logits: (B, K); soft_targets: (B, K), sums to 1.
    Returns scalar loss.
    """
    log_probs = torch.log_softmax(logits, dim=1)
    # mean over batch of - sum_k p_k * log q_k
    return -(soft_targets * log_probs).sum(dim=1).mean()


@torch.no_grad()
def _freeze_except_head(model):
    """
    Freeze all parameters except those in the classifier head 'mlp' (change if your head name differs).
    """
    for p in model.parameters():
        p.requires_grad_(False)
    # Unfreeze head
    head = getattr(model, "mlp", None)
    if head is None:
        raise AttributeError("Expected source_model to have attribute 'mlp' for the head.")
    for p in head.parameters():
        p.requires_grad_(True)


def finetune_head_with_soft_targets_on_low_entropy(
    args,
    source_model: nn.Module,
    tgt_trainset,
    P_ens: np.ndarray,
    H: np.ndarray,
    p_keep: float = 60.0,
    epochs: int = 1,
    batch_size: int = None,
    lr: float = None,
    weight_decay: float = 0.0,
    num_workers: int = None,
    device: torch.device = None,
):
    """
    Freeze encoder, fine-tune classifier head on the lowest-entropy p% samples using soft targets.

    - P_ens: (N x K) ensemble class probs for each sample in tgt_trainset (in the SAME order as indexing).
    - H:     (N,) entropy for each sample.
    """
    if device is None:
        device = next(source_model.parameters()).device
    if batch_size is None:
        batch_size = getattr(args, "batch_size", 256)
    if lr is None:
        lr = getattr(args, "soft_ft_lr", 1e-3)
    if num_workers is None:
        num_workers = getattr(args, "num_workers", 2)

    # Build mask and subset
    mask = _lowest_entropy_mask(H, p_keep=float(p_keep))
    idx = np.nonzero(mask)[0]
    if idx.size == 0:
        print("[SoftFT] No samples selected by entropy mask; skipping soft fine-tune.")
        return 0.0

    ds = _SoftTargetSubset(tgt_trainset, idx, P_ens)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    source_model.train()
    _freeze_except_head(source_model)
    head = source_model.mlp

    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)

    running_loss = 0.0
    n_batches = 0
    for ep in range(int(epochs)):
        for x, p in loader:
            x = x.to(device, non_blocking=True)
            p = p.to(device, non_blocking=True)
            logits = source_model(x)  # forward through full model; only head has grads
            loss = _soft_ce_loss_from_probs(logits, p)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running_loss += float(loss.item())
            n_batches += 1

    mean_loss = (running_loss / max(n_batches, 1)) if n_batches > 0 else 0.0
    print(f"[SoftFT] Done: p_keep={p_keep:.1f}%, epochs={epochs}, batches={n_batches}, mean_loss={mean_loss:.4f}, kept={idx.size}/{len(tgt_trainset)}")
    return mean_loss

# --- log-likelihood extraction

# ---------- numerically stable mixture LL for init scoring ----------

def estimate_log_likelihood(X, mu, Sigma, pi, *, cov_type="diag", reg=1e-6) -> float:
    X  = np.asarray(X,  dtype=np.float64, order="C")
    mu = np.asarray(mu, dtype=np.float64, order="C")
    pi = np.asarray(pi, dtype=np.float64, order="C")
    N, D = X.shape; K = mu.shape[0]
    pi = np.clip(pi, 1e-12, None); pi = pi / pi.sum(); logpi = np.log(pi)
    LOG2PI = np.log(2.0 * np.pi)
    comp = np.empty((K, N), dtype=np.float64)

    cov_type = str(cov_type).lower()
    if cov_type == "diag":
        Sigma = np.asarray(Sigma, dtype=np.float64, order="C") + reg
        invv = 1.0 / Sigma; ldet = np.log(Sigma).sum(axis=1)
        for k in range(K):
            r = X - mu[k]; quad = (r * invv[k] * r).sum(axis=1)
            comp[k] = -0.5 * (D*LOG2PI + ldet[k] + quad) + logpi[k]
    elif cov_type == "spherical":
        Sigma = np.asarray(Sigma, dtype=np.float64, order="C").reshape(K) + reg
        invv = 1.0 / Sigma; ldet = D * np.log(Sigma)
        for k in range(K):
            r = X - mu[k]; quad = invv[k] * (r*r).sum(axis=1)
            comp[k] = -0.5 * (D*LOG2PI + ldet[k] + quad) + logpi[k]
    elif cov_type == "full":
        Sigma = np.asarray(Sigma, dtype=np.float64, order="C")
        for k in range(K):
            Sk = Sigma[k] + reg*np.eye(D)
            sign, ldet = np.linalg.slogdet(Sk)
            if sign <= 0:
                w, V = np.linalg.eigh(Sk); w = np.clip(w, 1e-12, None)
                Sk = (V * w) @ V.T; ldet = np.log(w).sum()
            iSk = np.linalg.inv(Sk)
            r = X - mu[k]; quad = (r @ iSk * r).sum(axis=1)
            comp[k] = -0.5 * (D*LOG2PI + ldet + quad) + logpi[k]
    else:
        raise ValueError(f"Unsupported cov_type={cov_type}")

    m = comp.max(axis=0)                             # (N,)
    ll = m + np.log(np.exp(comp - m).sum(axis=0))    # (N,)
    return float(ll.sum())


def _safe_last_ll(em_res) -> float:
    llc = em_res.get("ll_curve", None)
    if isinstance(llc, (list, tuple)) and len(llc) > 0:
        return float(llc[-1])
    if "X" in em_res and all(k in em_res for k in ("mu", "Sigma", "pi")):
        # Fallback: compute LL with your earlier helper
        return float(estimate_log_likelihood(
            em_res["X"], em_res["mu"], em_res["Sigma"], em_res["pi"],
            cov_type=em_res.get("cov_type", "diag"),
            reg=1e-6
        ))
    return float("nan")

# --- parameter count for GMM
def _count_gmm_params(K: int, D: int, cov_type: str) -> int:
    cov_type = str(cov_type).lower()
    means = K * D
    if cov_type == "diag":
        covs = K * D
    elif cov_type == "spherical":
        covs = K
    elif cov_type == "full":
        covs = K * (D * (D + 1) // 2)
    else:
        raise ValueError(f"Unknown cov_type {cov_type}")
    weights = (K - 1)  # simplex
    return means + covs + weights

# --- information criteria
def _bic(ll: float, n_params: int, n_samples: int) -> float:
    return -2.0 * float(ll) + float(n_params) * math.log(max(1, n_samples))

def _icl(bic: float, gamma: np.ndarray) -> float:
    # ICL = BIC - 2 * sum_i H(gamma_i)
    eps = 1e-12
    Pi = np.clip(gamma, eps, 1.0)
    ent = -(Pi * np.log(Pi)).sum(axis=1).sum()
    return float(bic - 2.0 * ent)

# --- selection among fitted models
def _model_simplicity_key(cfg: dict) -> tuple:
    # simpler first: diag < spherical < full ; smaller K ; smaller pca_dim (None treated as 0)
    cov_rank = {"diag": 0, "spherical": 1, "full": 2}.get(str(cfg["cov_type"]).lower(), 3)
    k = int(cfg["K"])
    pd = cfg.get("pca_dim", None)
    pca_rank = 0 if (pd in [None, 0]) else int(pd)
    return (cov_rank, k, pca_rank)


# --- add near your imports ---
from scipy.optimize import linear_sum_assignment
import numpy as np
import math

# --- new helper: proper scalarization of mapping cost ---
def _scalarize_mapping_cost(cost_obj, mode: str) -> float:
    """
    Convert a mapping 'cost object' into a scalar suitable for model selection.

    mode = 'pseudo'      : cost_obj is a contingency matrix C (K_clusters x n_classes).
                           We maximize matches -> minimize (max(C) - C) via Hungarian.
                           Returns the minimized sum, i.e., assignment cost.
                           A smaller value is better (fewer misassignments).
    mode = 'prototypes'  : cost_obj is a distance matrix D (K_clusters x n_classes).
                           We minimize D via Hungarian and return D[r,c].sum().
    """
    A = np.asarray(cost_obj, dtype=float)
    if A.size == 0 or not np.all(np.isfinite(A)):
        return float("nan")

    if mode == "pseudo":
        # Convert contingency to a cost matrix where lower is better
        m = float(np.max(A)) if A.size > 0 else 0.0
        Ccost = m - A
        r, c = linear_sum_assignment(Ccost)
        return float(Ccost[r, c].sum())
    elif mode == "prototypes":
        # Already a distance/cost matrix
        r, c = linear_sum_assignment(A)
        return float(A[r, c].sum())
    else:
        raise ValueError(f"unknown mode for scalarization: {mode}")


def fit_many_em_on_target(
    e_tgt,
    K_list: List[int],
    cov_types: List[str],
    seeds: List[int],
    pool: str = "gap",
    pca_dims: Optional[List[Optional[int]]] = None,
    reg: float = 1e-4,
    max_iter: int = 300,
    rng_base: int = 0,
    args=None,
) -> List[Dict]:
    """
    Run a grid of EM configurations on encoded target embeddings.
    Returns a list of dicts (one per config):
      {
        'cfg': {'K', 'cov_type', 'pca_dim', 'seed},
        'em_res': dict,
        'final_ll': float,
        'bic': float,
        'mapped_soft': np.ndarray,   # (N x K_classes)
        'labels_mapped': np.ndarray, # (N,)
        'mapping': Dict[int,int],    # cluster -> class
      }
    tqdm is used for grid progress; falls back to a no-op if unavailable.
    """
    results: List[Dict] = []
    N = len(e_tgt)
    if pca_dims is None:
        pca_dims = [None]

    # Inferred class count for mapped outputs
    try:
        y_true = e_tgt.targets.cpu().numpy() if torch.is_tensor(e_tgt.targets) else np.asarray(e_tgt.targets)
        K_default = int(max(int(y_true.max()), 0)) + 1
    except Exception:
        K_default = 10
    K_for_classes = max([K_default] + [int(k) for k in (K_list or [])]) if K_list else K_default

    # Mapping prerequisites
    use_proto = (getattr(args, "em_match", "pseudo") == "prototypes")
    if use_proto and not hasattr(args, "_cached_source_stats"):
        raise RuntimeError("Prototype mapping requested but args._cached_source_stats is missing.")
    if (not use_proto) and not hasattr(args, "_cached_pseudolabels"):
        raise RuntimeError("Pseudo mapping requested but args._cached_pseudolabels is missing.")
    # no interactive breakpoints in library code
    mu_s = Sigma_s = priors_s = None
    if use_proto:
        mu_s, Sigma_s, priors_s = args._cached_source_stats

    Ks        = [int(k) for k in (K_list if K_list else [K_default])]
    Covs      = [str(c) for c in cov_types]
    Pcas      = [None if (p in [None, 0]) else int(p) for p in pca_dims]
    Seeds     = [int(s) for s in seeds]
    total_cfg = len(Ks) * len(Covs) * len(Pcas) * len(Seeds)

    # >>> NEW: compute & print total GMM count at the start
    total_cfg = len(Ks) * len(Covs) * len(Pcas) * len(Seeds)
    print(f"[EM-grid] Total GMMs to fit: {total_cfg}  "
          f"(K={Ks}, cov={Covs}, pca={Pcas}, seeds={Seeds})")

    # Grid progress bar (degrades to no-op if tqdm is the shim)
    bar = tqdm(total=total_cfg, desc="Fitting EM grid", leave=True)

    for K in Ks:
        for cov_type in Covs:
            for pca_dim in Pcas:
                for seed in Seeds:
                    cfg = dict(K=K, cov_type=cov_type, pca_dim=pca_dim, seed=seed)
                    # Update caption (safe for no-op tqdm)
                    try:
                        bar.set_postfix_str(f"K={K}, cov={cov_type}, pca={pca_dim}, seed={seed}")
                    except Exception:
                        pass
                    # breakpoint()
                    # ---- Fit EM for this configuration ----
                    em_res = run_em_on_encoded(
                        e_tgt,
                        K=K,
                        cov_type=cov_type,
                        pool=pool,
                        do_pca=(pca_dim is not None),
                        pca_dim=pca_dim,
                        reg=reg,
                        max_iter=max_iter,
                        return_transforms=True,
                        verbose=False,                 # avoid clashing prints with tqdm
                        subsample_init=N,
                        rng=seed if seed is not None else rng_base,
                        # Optional: if your EM supports per-iteration tqdm via a hook, wire it:
                        # show_iter_progress=getattr(args, "em_show_iter_progress", False),
                    )

                    # ---- Score: LL + BIC ----
                    final_ll = _safe_last_ll(em_res)
                    if "X" in em_res and em_res["X"] is not None:
                        D_eff = int(em_res["X"].shape[1]); n_samples = int(em_res["X"].shape[0])
                    else:
                        # fallback dimensionality
                        D_eff = getattr(e_tgt, "dim", None)
                        if D_eff is None and hasattr(e_tgt, "data"):
                            arr = np.asarray(e_tgt.data)
                            D_eff = int(arr.reshape(arr.shape[0], -1).shape[1])
                        D_eff = int(D_eff) if D_eff is not None else 64
                        n_samples = N
                    n_params = _count_gmm_params(K, D_eff, cov_type)
                    bic_val  = _bic(final_ll if final_ll is not None and not math.isnan(final_ll) else -1e12,
                                    n_params, n_samples)
                    # breakpoint()
                    # ---- Map clusters -> classes ----
                    if use_proto:
                        mapping, labels_mapped, cost_obj = map_em_clusters(
                            em_res, method="prototypes", n_classes=K_for_classes,
                            mus_s=mu_s, Sigma_s=Sigma_s, priors_s=priors_s
                        )
                        cost = _scalarize_mapping_cost(cost_obj, mode="prototypes")
                    else:
                        mapping, labels_mapped, cost_obj = map_em_clusters(
                            em_res, method="pseudo", n_classes=K_for_classes,
                            pseudo_labels=args._cached_pseudolabels, metric='FR'
                        )
                        cost = _scalarize_mapping_cost(cost_obj, mode="pseudo")

                    gamma = np.asarray(em_res["gamma"])      # (N x K_clusters)
                    mapped_soft = _map_cluster_posts_to_classes(gamma, mapping, K=K_for_classes)

                
                    if isinstance(cost, dict):
                        cost_scalar = float(np.nansum(list(cost.values())))
                    else:
                        cost_scalar = float(np.nansum(np.asarray(cost, dtype=float)))


                    results.append(dict(
                        cfg=cfg,
                        em_res=em_res,
                        final_ll=(float(final_ll) if final_ll is not None else float("nan")),
                        bic=float(bic_val),
                        cost=float(cost_scalar),              # <-- scalar, comparable across runs
                        cost_matrix=np.asarray(cost_obj),     # <-- keep matrix for diagnostics
                        mapped_soft=mapped_soft,
                        labels_mapped=np.asarray(labels_mapped, dtype=int),
                        mapping=mapping,
                    ))

                    # advance the grid bar
                    try:
                        bar.update(1)
                    except Exception:
                        pass

    try:
        bar.close()
    except Exception:
        pass

    return results

def score_em_by_cost(cost_matrix, cluster_sizes=None, class_priors=None, tau=1.0):
    C = np.asarray(cost_matrix, dtype=float)  # shape (Kc, K)
    # Hungarian assignment on cost
    r, c = linear_sum_assignment(C)
    J_assign = float(C[r, c].sum())

    # Optional weighting of assignment cost
    if cluster_sizes is not None:
        w = np.asarray(cluster_sizes, dtype=float)[r]
        w = w / (w.sum() + 1e-12)
        J_assign = float((w * C[r, c]).sum())
    elif class_priors is not None:
        w = np.asarray(class_priors, dtype=float)[c]
        w = w / (w.sum() + 1e-12)
        J_assign = float((w * C[r, c]).sum())

    # Confidence (margin) & entropy (for tie-break)
    best = C.min(axis=1)
    second = np.partition(C, 1, axis=1)[:,1]
    margin = float(np.nansum(second - best))  # larger is better

    A = np.exp(-C / max(tau, 1e-6))
    A = A / (A.sum(axis=1, keepdims=True) + 1e-12)
    ent = float(-(A * (np.log(A + 1e-12))).sum())  # smaller is better

    return dict(J_assign=J_assign, margin=margin, entropy=ent, assignment=(r, c))


def _safe(v, bad_for_min=math.inf, bad_for_max=-math.inf):
    # Replace None/NaN with sentinels depending on whether we will min or max it.
    if v is None:
        return bad_for_min  # default assumption; pass bad_for_max explicitly when needed
    try:
        return v if not math.isnan(v) else bad_for_min
    except (TypeError, ValueError):
        return bad_for_min

def select_best_em(results: List[Dict], criterion: str = "bic") -> Dict:
    """
    Pick the single best EM model.

    criterion:
      - 'bic'  : minimize BIC; tie-break by larger final_ll
      - 'll'   : maximize final log-likelihood
      - 'cost' : minimize assignment cost from score_em_by_cost(C); tie-break by larger margin, then smaller entropy, then BIC
      - 'pm'/'match'/'perfmatch': maximize perfect-matching accuracy
    """
    assert len(results) > 0, "No EM results provided."

    if criterion == "bic":
        # Lower BIC is better; tie-break with higher LL
        return min(
            results,
            key=lambda r: (_safe(r.get("bic"), bad_for_min=math.inf),
                           -_safe(r.get("final_ll"), bad_for_min=-math.inf, bad_for_max=-math.inf))
        )

    if criterion == "ll":
        # Higher LL is better
        return max(
            results,
            key=lambda r: _safe(r.get("final_ll"), bad_for_min=-math.inf, bad_for_max=-math.inf)
        )

    if criterion == "cost":
        # Requires a cost matrix per result
        best = None
        best_key = None
        for r in results:
            C = r.get("cost_matrix", None)
            if C is None:
                # If absent, treat as very bad (infinite cost)
                key = (math.inf, 0.0, math.inf, _safe(r.get("bic"), bad_for_min=math.inf))
            else:
                sc = score_em_by_cost(
                    C,
                    cluster_sizes=r.get("cluster_sizes"),
                    class_priors=r.get("class_priors"),
                    tau=1.0
                )
                # Persist the diagnostics on the original dict
                r["cost"]       = sc["J_assign"]
                r["margin"]     = sc["margin"]
                r["entropy"]    = sc["entropy"]
                r["assignment"] = sc["assignment"]

                # Primary: minimize cost; Secondary: maximize margin; Tertiary: minimize entropy; Quaternary: minimize BIC
                key = ( _safe(r["cost"],    bad_for_min=math.inf),
                       -_safe(r["margin"],  bad_for_min=-math.inf, bad_for_max=-math.inf),
                        _safe(r["entropy"], bad_for_min=math.inf),
                        _safe(r.get("bic"), bad_for_min=math.inf) )

            if best is None or key < best_key:
                best, best_key = r, key
        return best

    if criterion in ("pm", "match", "perfmatch"):
        # Higher perfect-matching accuracy is better; if missing/NaN, treat as -inf
        return max(
            results,
            key=lambda r: _safe(r.get("pm_acc"), bad_for_min=-math.inf, bad_for_max=-math.inf)
        )

    raise ValueError(f"Unknown selection criterion: {criterion}")


_EM_CACHE = {}

def run_em_on_encoded(
    enc_ds,
    K=10,
    cov_type='diag',          # 'diag' | 'spherical' | 'full'
    pool='gap',
    do_pca=True,              # default: turn PCA on
    pca_dim=64,               # compact projection for speed
    reg=1e-6,
    max_iter=100,             # lower than 300 by default
    tol=1e-3,                 # slightly looser -> earlier stop
    freeze_sigma_iters=0,
    rng: Union[int, np.random.Generator, None] = 0,
    *,
    scaler: Optional[StandardScaler] = None,
    pca: Optional[PCA] = None,
    return_transforms: bool = False,
    subsample_init: Optional[int] = 20000,   # for KMeans init
    warm_start: Optional[dict] = None,       # may be dict or (mu,Sigma,pi)
    verbose: bool = False,
    cache_key: Optional[str] = None,
    dtype: str = "float32",                  # compute in float32 for speed
    n_init: int = 2,                         # fewer restarts for speed
    max_points: Optional[int] = 20000,       # NEW: cap for the main EM
):
    """
    High-level wrapper:
      - prepares X (scaler + optional PCA; float32 contiguous)
      - caps N for EM (max_points)
      - runs a faster EM with fewer restarts
    """
    X_std, _scaler, _pca, key = prepare_em_representation(
        enc_ds, pool=pool, do_pca=do_pca, pca_dim=pca_dim,
        scaler=scaler, pca=pca, cache_key=cache_key, dtype=dtype, rng=rng, verbose=verbose
    )
    out = run_em_on_encoded_fast(
        X_std,
        K=K, cov_type=cov_type, reg=reg, max_iter=max_iter, tol=tol,
        n_init=n_init, subsample_init=subsample_init,
        warm_start=warm_start, rng=rng, verbose=verbose,
        max_points=subsample_init,
    )

    if return_transforms:
        out["scaler"] = _scaler
        out["pca"] = _pca
    return out


def run_em_on_encoded_fast(
    X_std: np.ndarray,
    *,
    K: int = 10,
    cov_type: str = "diag",
    reg: float = 1e-6,
    max_iter: int = 100,
    tol: float = 1e-3,
    n_init: int = 2,
    subsample_init: Optional[int] = 20000,
    warm_start: Optional[dict] = None,   # dict or (mu,Sigma,pi)
    rng: Union[int, np.random.Generator, None] = 0,
    verbose: bool = False,
    return_params: bool = True,
    max_points: Optional[int] = 20000,   # NEW: cap for main EM fit
):
    """
    Faster EM:
      - float32 end-to-end (with regularization)
      - cap N before EM to bound runtime
      - early stopping with tol relative to LL
    """
    # RNG
    if rng is None:
        rng = np.random.default_rng()
    elif not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(int(rng))

    # keep dtype from upstream (float32 recommended)
    X = np.asarray(X_std, dtype=np.float32, order="C")
    N, D = X.shape

    # Slightly higher reg helps stability in float32 and in 'full' covs
    reg_used = max(reg, 1e-4) if cov_type == "full" else reg

    # ---- cap N for main EM
    if (max_points is not None) and (N > max_points):
        idx = rng.choice(N, size=max_points, replace=False)
        X = X[idx]
        N = X.shape[0]
        if verbose:
            print(f"[em] capped N to {N} for speed")

    # ---- subset for init
    if subsample_init is None or N <= subsample_init:
        X_init = X
    else:
        X_init = X[rng.choice(N, size=subsample_init, replace=False)]

    # ---- build inits in ARRAY form
    inits = _make_inits(
        X_init, K=K, cov_type=cov_type, n_init=n_init, reg=reg_used, rng=rng
    )

    # optional warm start replaces first init
    if warm_start is not None:
        ws_mu, ws_Sig, ws_pi = _normalize_gmm_params(warm_start, cov_type=cov_type, reg=reg_used)
        inits[0] = (ws_mu.astype(np.float32, copy=False),
                    _cast_cov(ws_Sig, cov_type).astype(np.float32, copy=False),
                    ws_pi.astype(np.float32, copy=False))

    # choose best init by quick LL (ARRAY form)
    def quick_ll(theta):
        mu0, Sig0, pi0 = theta
        return estimate_log_likelihood(X, mu0, Sig0, pi0, cov_type=cov_type, reg=reg_used)

    best_mu, best_Sig, best_pi = max(inits, key=quick_ll)

    # convert winner to DICT form (what your EM expects)
    mu_d, Sig_d, pi_d = _arrays_to_component_dicts(best_mu, best_Sig, best_pi, cov_type=cov_type)

    # ---- run your EM (unchanged API)
    mu, Sigma, pi, gamma, ll_curve = em_k_gaussians_sklearn(
        X,
        mu_init=mu_d, Sigma_init=Sig_d, pi_init=pi_d,
        cov_type=cov_type, max_iter=max_iter, tol=tol, reg=reg_used, verbose=verbose
    )

    z = gamma.argmax(axis=1)
    out = {"X": X, "mu": mu, "Sigma": Sigma, "pi": pi, "gamma": gamma, "labels": z, "ll_curve": ll_curve}
    return out if return_params else {"gamma": gamma, "labels": z, "ll_curve": ll_curve}

def best_mapping_accuracy(pred_clusters, true_labels):
    """
    Compute the maximum achievable accuracy by one-to-one mapping of cluster IDs to class IDs
    using Hungarian assignment on the confusion matrix.

    Returns:
      acc   : float          # best achievable accuracy in [0,1]
      mapping : dict[int,int]# cluster_id -> class_id
      C    : np.ndarray      # confusion matrix used for the assignment (K x C)
    """
    z = np.asarray(pred_clusters, dtype=int)
    y = np.asarray(true_labels, dtype=int)

    # Build confusion matrix over the actually present labels
    uniq_clusters = np.unique(z)
    uniq_classes  = np.unique(y)
    k_to_row = {k:i for i,k in enumerate(uniq_clusters)}
    c_to_col = {c:j for j,c in enumerate(uniq_classes)}

    K, Cc = len(uniq_clusters), len(uniq_classes)
    C = np.zeros((K, Cc), dtype=int)
    for k in uniq_clusters:
        mk = (z == k)
        if mk.any():
            counts = np.bincount(y[mk], minlength=uniq_classes.max()+1)[uniq_classes]
            C[k_to_row[k], :] = counts

    # Pad to square so Hungarian assigns every row (cluster)
    n = max(K, Cc)
    Cp = np.zeros((n, n), dtype=int)
    Cp[:K, :Cc] = C

    # Hungarian on cost = max - counts (maximize matches)
    cost = Cp.max() - Cp
    rows, cols = linear_sum_assignment(cost)

    # Compute accuracy from real assignments only
    correct = 0
    for r, c in zip(rows, cols):
        if r < K and c < Cc:
            correct += C[r, c]
    acc = correct / len(z)

    # Build mapping cluster_id -> class_id (only for real assignments)
    mapping = {}
    for r, c in zip(rows, cols):
        if r < K and c < Cc:
            k = uniq_clusters[r]
            cls = uniq_classes[c]
            mapping[int(k)] = int(cls)
    print(f"Best mapping accuracy: {acc:.4f} over {len(z)} samples with {K} clusters and {Cc} classes")
    return acc, mapping, C


def eval_mapped_labels_against_gt(labels_mapped, tgt_enc, n_classes=None):
    """Compute accuracy and confusion matrix vs ground-truth target labels."""
    y_true = np.asarray(tgt_enc.targets)
    y_pred = np.asarray(labels_mapped, dtype=int)
    if n_classes is None:
        n_classes = int(max(y_true.max(), y_pred.max())) + 1

    acc = accuracy_score(y_true, y_pred)
    cm  = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))
    return acc, cm

# ---------- helpers you already have; add small casts ----------
def _cast_cov(Sigma, cov_type):
    if cov_type == "diag":
        return np.asarray(Sigma, dtype=np.float32)
    elif cov_type == "spherical":
        return np.asarray(Sigma, dtype=np.float32)
    elif cov_type == "full":
        return np.asarray(Sigma, dtype=np.float32)
    else:
        raise ValueError(cov_type)

def to_features_from_encoded(enc_ds, pool='gap'):
    """
    enc_ds: encoded dataset with .data (torch.Tensor or np.ndarray)
    pool: 'gap' (global avg pool) or 'flatten'
    returns X (N, d)
    """
    X = enc_ds.data
    if torch.is_tensor(X):
        X = X.detach().cpu().numpy()
    if X.ndim == 4:
        # N,C,H,W
        if pool == 'gap':
            X = X.mean(axis=(2, 3))           # (N,C)
        elif pool == 'flatten':
            X = X.reshape(X.shape[0], -1)     # (N, C*H*W)
        else:
            raise ValueError("pool must be 'gap' or 'flatten'")
    elif X.ndim == 3:
        X = X.reshape(X.shape[0], -1)
    elif X.ndim != 2:
        raise ValueError(f"Unexpected data ndim={X.ndim}")
    return X.astype(np.float64)



def prepare_em_representation(
    enc_ds,
    *,
    pool: str = "gap",
    do_pca: bool = True,
    pca_dim: int = 64,
    scaler: Optional[StandardScaler] = None,
    pca: Optional[PCA] = None,
    cache_key: Optional[str] = None,
    dtype: str = "float32",
    rng: Union[int, np.random.Generator, None] = 0,
    verbose: bool = False,
):
    """
    Produces standardized (and optionally PCA-compressed) features for EM.
    Ensures C-contiguous float32 by default (fast BLAS).
    Caching is now *per-enc_ds* to avoid mixing domains.
    """

    # Make cache key depend on the specific encoded dataset object
    # so angle 22 and angle 45 do NOT share the same entry.
    if cache_key is None:
        cache_key = (
            f"id={id(enc_ds)}|"
            f"{pool}|pca={int(do_pca)}|dim={pca_dim}|dtype={dtype}"
        )

    # Try cache
    if cache_key in _EM_CACHE:
        if verbose:
            print(f"[prep] cache hit for key={cache_key}")
        c = _EM_CACHE[cache_key]
        return c["X_std"], c["scaler"], c["pca"], cache_key

    # --- Compute features from scratch ---
    X = to_features_from_encoded(enc_ds, pool=pool)  # expected (N,D)
    X = X.astype(np.float32 if dtype == "float32" else np.float64, copy=False)
    if not X.flags.c_contiguous:
        X = np.ascontiguousarray(X)
    if verbose:
        print(f"[prep] Features: {X.shape} dtype={X.dtype}")

    if scaler is None:
        scaler = StandardScaler(with_mean=True, with_std=True)
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    _pca = None
    if do_pca and X.shape[1] > pca_dim:
        if not isinstance(rng, np.random.Generator):
            rng = np.random.default_rng(int(rng) if rng is not None else None)
        _pca = PCA(
            n_components=pca_dim,
            svd_solver="randomized",
            random_state=int(rng.integers(2**31 - 1)),
        )
        X = _pca.fit_transform(X)
        if verbose:
            print(f"[prep] PCA -> {X.shape}")

    if verbose:
        print(f"[prep] caching key={cache_key}")
    _EM_CACHE[cache_key] = {"X_std": X, "scaler": scaler, "pca": _pca}
    return X, scaler, _pca, cache_key

def _make_inits(X_init, K: int, cov_type: str, n_init: int, reg: float, rng):
    """Return list of ARRAY-form tuples (mu, Sigma, pi), regardless of initializer's native format."""
    inits = []
    for _ in range(max(1, int(n_init))):
        ini = kmeanspp_init_params(X_init, K=K, cov_type=cov_type, reg=reg, rng=rng)
        if isinstance(ini, (tuple, list)) and len(ini) == 3 and isinstance(ini[0], dict):
            mu0, Sigma0, pi0 = _normalize_init_tuple_of_dicts(ini, K=K, cov_type=cov_type, reg=reg)
        elif isinstance(ini, dict):
            mu0, Sigma0, pi0 = _normalize_gmm_params(ini, cov_type=cov_type, reg=reg)
        elif isinstance(ini, (tuple, list)) and len(ini) == 3:
            mu0, Sigma0, pi0 = _normalize_gmm_params(ini, cov_type=cov_type, reg=reg)
        else:
            raise TypeError(f"Unsupported init format from kmeanspp_init_params: {type(ini)}")
        inits.append((mu0, Sigma0, pi0))
    return inits

def _stack_from_component_dict(d: dict, K: int, dtype=np.float64) -> np.ndarray:
    keys = sorted(d.keys())
    if len(keys) != K or any(i != k for i, k in enumerate(keys)):
        raise ValueError(f"Expected component keys 0..{K-1}, got {keys}")
    vals = [np.asarray(d[k], dtype=dtype) for k in keys]
    return np.stack(vals, axis=0)



def _normalize_init_tuple_of_dicts(ini, K: int, cov_type: str, reg: float):
    """ini = (mu_dict, Sigma_dict, pi_dict)  -> arrays (mu, Sigma, pi)."""
    mu_dict, Sigma_dict, pi_dict = ini
    mu = _stack_from_component_dict(mu_dict, K, dtype=np.float64)  # (K,D)
    cov_type = str(cov_type).lower()
    if cov_type == "diag":
        Sigma = _stack_from_component_dict(Sigma_dict, K, dtype=np.float64) + reg       # (K,D)
    elif cov_type == "spherical":
        Sigma = np.array([np.asarray(Sigma_dict[k], dtype=np.float64).squeeze() for k in range(K)])
        Sigma = Sigma + reg                                                             # (K,)
    elif cov_type == "full":
        Sigma_vals = [np.asarray(Sigma_dict[k], dtype=np.float64) for k in range(K)]
        if Sigma_vals[0].ndim == 1:
            Sigma = np.stack([np.diag(s + reg) for s in Sigma_vals], axis=0)            # (K,D,D)
        elif Sigma_vals[0].ndim == 2:
            Sigma = np.stack([Sk + reg*np.eye(Sk.shape[0]) for Sk in Sigma_vals], axis=0)
        else:
            raise ValueError(f"Unsupported Sigma entry ndim={Sigma_vals[0].ndim}")
    else:
        raise ValueError(f"Unsupported cov_type={cov_type}")
    pi = np.array([float(pi_dict[k]) for k in range(K)], dtype=np.float64)
    pi = np.clip(pi, 1e-12, None); pi = pi / pi.sum()
    return mu, Sigma, pi

def _normalize_gmm_params(params, cov_type: str, *, reg: float = 1e-6):
    """
    Accept dict{'mu','Sigma','pi'} or tuple(mu,Sigma,pi) → arrays (mu,Sigma,pi).
    """
    if isinstance(params, dict):
        mu  = params.get("mu",  params.get("means"))
        Sig = params.get("Sigma", params.get("covariances"))
        pi  = params.get("pi",  params.get("weights"))
    elif isinstance(params, (tuple, list)) and len(params) == 3:
        mu, Sig, pi = params
    else:
        raise TypeError(f"Unsupported params type: {type(params)}")

    mu = np.asarray(mu, dtype=np.float64, order="C")
    pi = np.asarray(pi, dtype=np.float64, order="C")
    K, D = mu.shape
    cov_type = str(cov_type).lower()

    if cov_type == "diag":
        Sig = np.asarray(Sig, dtype=np.float64, order="C")
        if Sig.shape != (K, D): raise ValueError(f"diag Sigma shape {Sig.shape} != {(K,D)}")
        Sig = Sig + reg
    elif cov_type == "spherical":
        Sig = np.asarray(Sig, dtype=np.float64, order="C").reshape(K)
        Sig = Sig + reg
    elif cov_type == "full":
        Sig = np.asarray(Sig, dtype=np.float64, order="C")
        if Sig.ndim == 2 and Sig.shape == (K, D):   # promote diag vector per comp
            Sig = np.stack([np.diag(Sig[k]) for k in range(K)], axis=0)
        if Sig.shape != (K, D, D): raise ValueError(f"full Sigma shape {Sig.shape} != {(K,D,D)}")
    else:
        raise ValueError(f"Unsupported cov_type={cov_type}")

    pi = np.clip(pi, 1e-12, None); pi = pi / pi.sum()
    return mu, Sig, pi

# ---------- helper: arrays → dicts (what your EM expects) ----------

def _arrays_to_component_dicts(
    mu: np.ndarray, Sigma, pi: np.ndarray, *, cov_type: str
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, float]]:
    mu = np.asarray(mu)
    K, D = mu.shape
    cov_type = str(cov_type).lower()
    mu_d  = {i: mu[i].copy() for i in range(K)}
    if cov_type == "diag":
        Sig = np.asarray(Sigma); assert Sig.shape == (K, D)
        Sig_d = {i: Sig[i].copy() for i in range(K)}
    elif cov_type == "spherical":
        Sig = np.asarray(Sigma).reshape(K)
        Sig_d = {i: float(Sig[i]) for i in range(K)}
    elif cov_type == "full":
        Sig = np.asarray(Sigma); assert Sig.shape == (K, D, D)
        Sig_d = {i: Sig[i].copy() for i in range(K)}
    else:
        raise ValueError(f"Unsupported cov_type={cov_type}")
    pi = np.asarray(pi).reshape(K); pi_d = {i: float(pi[i]) for i in range(K)}
    return mu_d, Sig_d, pi_d


def em_k_gaussians(
    X: np.ndarray,
    mu_init: Dict[int, np.ndarray],                 # {k: (d,)}
    Sigma_init: Dict[int, np.ndarray],              # {k: (d,d)} (full/diag as 2D)
    pi_init: Dict[int, float],                      # {k: float}, sum≈1
    max_iter: int = 100,
    tol: float = 1e-5,
    reg: float = 1e-6,                              # ridge added to Σₖ
    cov_type: str = "full",                         # 'full' | 'diag' | 'spherical'
    min_Nk: float = 1e-3,                           # re-init threshold
    verbose: bool = True,
    rng: Optional[np.random.Generator] = None
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, float], np.ndarray]:
    """
    for a K-component Gaussian mixture with *component-specific* covariances.

    Returns:
      mu    : {k: (d,)}        means
      Sigma : {k: (d,d)}       per-component covariance (diag/spherical stored as (d,d))
      pi    : {k: float}       mixing proportions
      gamma : (n, K)           responsibilities
    """
    assert cov_type in {"full", "diag", "spherical"}
    rng = np.random.default_rng() if rng is None else rng

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.float32

    n, d = X.shape
    classes = sorted(mu_init.keys())
    K = len(classes)

    Xt = torch.as_tensor(X, dtype=dtype, device=device)
    pi_t = torch.zeros(K, dtype=dtype, device=device)
    mu_t = torch.zeros(K, d, dtype=dtype, device=device)
    if cov_type == "full":
        Sigma_t = torch.zeros(K, d, d, dtype=dtype, device=device)
    elif cov_type == "diag":
        Sigma_t = torch.zeros(K, d, dtype=dtype, device=device)
    else:
        Sigma_t = torch.zeros(K, dtype=dtype, device=device)

    ssum = 0.0
    for j, k in enumerate(classes):
        pi_t[j] = max(float(pi_init[k]), 0.0)
        mu_t[j] = torch.as_tensor(mu_init[k], dtype=dtype, device=device)
        S = np.asarray(Sigma_init[k])
        if cov_type == "full":
            Sigma_t[j] = torch.as_tensor(S, dtype=dtype, device=device)
        elif cov_type == "diag":
            v = np.diag(S) if S.ndim == 2 else S
            Sigma_t[j] = torch.as_tensor(np.maximum(v, reg), dtype=dtype, device=device)
        else:
            s2 = float(np.trace(S) / d) if S.ndim == 2 else float(np.mean(S))
            Sigma_t[j] = torch.tensor(max(s2, reg), dtype=dtype, device=device)
        ssum += float(pi_t[j])
    if ssum <= 0:
        pi_t.fill_(1.0 / K)
    else:
        pi_t /= ssum

    Xc = Xt - Xt.mean(0, keepdim=True)
    if cov_type == "full":
        glob = (Xc.T @ Xc) / max(n-1, 1) + reg * torch.eye(d, device=device, dtype=dtype)
    elif cov_type == "diag":
        glob = Xc.pow(2).mean(0).clamp_min(reg)
    else:
        glob = Xc.pow(2).mean() / d

    log2pi = float(d) * np.log(2.0 * np.pi)

    def logpdf_full(x, mu_k, Sig_k):
        L = torch.linalg.cholesky(Sig_k + reg * torch.eye(d, device=device, dtype=dtype))
        diff = x - mu_k
        sol = torch.cholesky_solve(diff.unsqueeze(-1), L)
        mah = (diff.unsqueeze(1) @ sol).squeeze()
        logdet = 2.0 * torch.log(torch.diag(L)).sum()
        return -0.5 * (log2pi + logdet + mah)

    def logpdf_diag(x, mu_k, var_k):
        diff2 = (x - mu_k).pow(2)
        mah = (diff2 / var_k.clamp_min(reg)).sum(dim=1)
        logdet = torch.log(var_k.clamp_min(reg)).sum()
        return -0.5 * (log2pi + logdet + mah)

    def logpdf_sph(x, mu_k, s2_k):
        diff2 = (x - mu_k).pow(2).sum(dim=1)
        s2 = torch.clamp(s2_k, min=reg)
        mah = diff2 / s2
        logdet = d * torch.log(s2)
        return -0.5 * (log2pi + logdet + mah)

    def e_step():
        L = torch.empty(n, K, dtype=dtype, device=device)
        for j in range(K):
            if cov_type == "full":
                lp = logpdf_full(Xt, mu_t[j], Sigma_t[j])
            elif cov_type == "diag":
                lp = logpdf_diag(Xt, mu_t[j], Sigma_t[j])
            else:
                lp = logpdf_sph(Xt, mu_t[j], Sigma_t[j])
            L[:, j] = torch.log(pi_t[j] + 1e-12) + lp
        m, _ = torch.max(L, dim=1, keepdim=True)
        L = L - m
        R = torch.exp(L)
        denom = R.sum(dim=1, keepdim=True) + 1e-12
        gamma_t = R / denom
        ll = torch.sum(torch.log(denom) + m).item()
        return gamma_t, ll

    def m_step(gamma_t):
        Nk = gamma_t.sum(dim=0)
        for j in range(K):
            if Nk[j].item() < min_Nk:
                idx = int(np.random.randint(n))
                mu_t[j] = Xt[idx]
                if cov_type == "full":
                    Sigma_t[j] = glob.clone()
                elif cov_type == "diag":
                    Sigma_t[j] = glob.clone()
                else:
                    Sigma_t[j] = torch.tensor(float(glob), device=device, dtype=dtype)
                pi_t[j] = 1.0 / K
                Nk[j] = 1.0

        Nk = torch.clamp(Nk, min=1e-12)
        pi_t.copy_(Nk / float(n))
        GtX = gamma_t.t() @ Xt
        mu_t[:] = GtX / Nk.unsqueeze(1)

        for j in range(K):
            Z = Xt - mu_t[j]
            if cov_type == "full":
                WZ = Z * torch.sqrt(gamma_t[:, j:j+1])
                Sk = (WZ.T @ WZ) / Nk[j]
                Sigma_t[j] = Sk + reg * torch.eye(d, device=device, dtype=dtype)
            elif cov_type == "diag":
                var = (gamma_t[:, j:j+1] * (Z**2)).sum(dim=0) / Nk[j]
                Sigma_t[j] = torch.clamp(var, min=reg)
            else:
                s2 = float((gamma_t[:, j:j+1] * (Z**2)).sum() / (Nk[j] * d))
                Sigma_t[j] = torch.tensor(max(s2, reg), device=device, dtype=dtype)

    ll_old = -np.inf
    ll_history: List[float] = []
    for it in range(max_iter):
        with torch.no_grad():
            gamma_t, ll = e_step()
            ll_history.append(ll)
            m_step(gamma_t)
            if verbose and (it % 10 == 0 or it == 0):
                print(f"[EM] iter={it:03d} ll={ll:.3f}")
            if ll - ll_old < tol:
                break
            ll_old = ll

    with torch.no_grad():
        gamma_t, ll = e_step()
        ll_history.append(ll)

    mu_out: Dict[int, np.ndarray] = {}
    Sigma_out: Dict[int, np.ndarray] = {}
    pi_out: Dict[int, float] = {}
    for j, k in enumerate(classes):
        mu_out[k] = mu_t[j].detach().cpu().numpy().astype(np.float64)
        if cov_type == "full":
            Sigma_out[k] = Sigma_t[j].detach().cpu().numpy().astype(np.float64)
        elif cov_type == "diag":
            v = Sigma_t[j].detach().cpu().numpy().astype(np.float64)
            Sigma_out[k] = np.diag(v)
        else:
            s2 = float(Sigma_t[j].detach().cpu().item())
            Sigma_out[k] = np.eye(d) * s2
        pi_out[k] = float(pi_t[j].detach().cpu().item())

    gamma_out = gamma_t.detach().cpu().numpy().astype(np.float64)
    return mu_out, Sigma_out, pi_out, gamma_out, ll_history


def kmeanspp_init_params(
    X: np.ndarray,
    K: int,
    cov_type: str = "full",            # 'full' | 'diag' | 'spherical'
    rng: Optional[np.random.Generator] = None,
    reg: float = 1e-6,
    reorder_by_dim: Optional[int] = None,  # e.g., 0 to sort by x-dim
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, float]]:
    """
    Initialize a K-class Gaussian mixture using k-means++ centers.

    Returns:
      mu_init    : {k: (d,)}      initial means
      Sigma_init : {k: (d,d)}     initial covariances (diag/spherical stored as (d,d))
      pi_init    : {k: float}     mixing proportions, sum≈1
    """
    assert cov_type in {"full", "diag", "spherical"}
    rng = np.random.default_rng() if rng is None else rng

    n, d = X.shape

    # scikit-learn expects an int seed (or None), not a Generator
    if hasattr(rng, "integers"):
        seed = int(rng.integers(2**31 - 1))
    else:
        seed = None

    # --- 1) k-means++ centers
    centers, _ = kmeans_plusplus(X, n_clusters=K, random_state=seed)   # (K, d)

    # Optional: reorder components by a specific coordinate (e.g., x-axis)
    if reorder_by_dim is not None:
        order = np.argsort(centers[:, reorder_by_dim])
        centers = centers[order]
        inv_order = np.empty_like(order)
        inv_order[order] = np.arange(K)

    # --- 2) Hard assignment to nearest center
    # squared Euclidean distances to centers -> labels
    d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)      # (n, K)
    labels = d2.argmin(axis=1)                                         # (n,)

    # --- 3) Global covariance for fallback
    glob = np.cov(X, rowvar=False) + reg * np.eye(d)

    def make_cov(Xk):
        """Compute covariance per cov_type with SVD-safe ridge."""
        nk = len(Xk)
        if nk <= 1:
            # ill-posed; fallback to global shape
            if cov_type == "full":
                return glob.copy()
            elif cov_type == "diag":
                return np.diag(np.clip(np.diag(glob), reg, None))
            else:  # spherical
                s2 = float(np.trace(glob) / d)
                return np.eye(d) * max(s2, reg)

        Z = Xk - Xk.mean(axis=0)
        if cov_type == "full":
            S = (Z.T @ Z) / (nk - 1) + reg * np.eye(d)
            return S
        elif cov_type == "diag":
            # breakpoint()
            # return the diagonal as a (d,) vector
            vars = (Z ** 2).sum(axis=0) / (nk - 1)
            return vars

            # return np.diag(np.maximum(vars, reg))

            # var = (Z ** 2).sum(axis=0) / (nk - 1)
            # return np.diag(np.maximum(var, reg))
        
        else:  # spherical
            s2 = float((Z ** 2).sum() / ((nk - 1) * d))
            return np.eye(d) * max(s2, reg)

    # --- 4) Build dicts
    mu_init: Dict[int, np.ndarray] = {}
    Sigma_init: Dict[int, np.ndarray] = {}
    pi_init: Dict[int, float] = {}

    for k in range(K):
        Xk = X[labels == k]
        nk = len(Xk)

        if nk == 0:
            # Empty cluster: re-init at a random data point; covariance from global
            idx = int(rng.integers(n))
            mu_k = X[idx].copy()
            if cov_type == "full":
                Sig_k = glob.copy()
            elif cov_type == "diag":
                Sig_k = np.diag(np.clip(np.diag(glob), reg, None))
            else:
                s2 = float(np.trace(glob) / d)
                Sig_k = np.eye(d) * max(s2, reg)
            pi_k = 1.0 / K
        else:
            mu_k = Xk.mean(axis=0)
            Sig_k = make_cov(Xk)
            pi_k = nk / n

        mu_init[k] = mu_k
        Sigma_init[k] = Sig_k
        pi_init[k] = float(pi_k)

    return mu_init, Sigma_init, pi_init


def _compute_source_prototypes(src_enc, pool='gap'):
    """Compute per-class mean vectors from a source encoded dataset."""
    Xs = _pool_features(src_enc.data, pool=pool)
    ys = np.asarray(src_enc.targets, dtype=int)
    C = ys.max() + 1
    d = Xs.shape[1]
    mu_src = np.zeros((C, d), dtype=float)
    for c in range(C):
        mask = (ys == c)
        if not mask.any():
            # leave zeros; Hungarian still works with large distances
            continue
        mu_src[c] = Xs[mask].mean(axis=0)
    return mu_src  # (C, d)


def _match_with_pseudolabels(cluster_ids, pseudo_labels, n_classes):
    """Hungarian match maximizing overlap counts between clusters and pseudo-labels."""
    cluster_ids = np.asarray(cluster_ids, dtype=int)
    pseudo_labels = np.asarray(pseudo_labels, dtype=int)

    # Align lengths defensively
    if cluster_ids.shape[0] != pseudo_labels.shape[0]:
        m = int(min(cluster_ids.shape[0], pseudo_labels.shape[0]))
        print(f"[EM-map] Warning: len(cluster_ids)={cluster_ids.shape[0]} != len(pseudo_labels)={pseudo_labels.shape[0]}; truncating to {m}.")
        cluster_ids = cluster_ids[:m]
        pseudo_labels = pseudo_labels[:m]

    # Classes present in pseudo_labels may exceed n_classes
    if n_classes is None:
        n_classes = int(pseudo_labels.max()) + 1
    else:
        n_classes = max(int(n_classes), int(pseudo_labels.max()) + 1)

    uniq_clusters = np.unique(cluster_ids)              # actual cluster keys
    R = len(uniq_clusters)
    row_of = {k: i for i, k in enumerate(uniq_clusters)}

    C = np.zeros((R, n_classes), dtype=np.int64)        # contingency
    for k in uniq_clusters:
        mk = (cluster_ids == k)
        if mk.any():
            C[row_of[k]] = np.bincount(pseudo_labels[mk], minlength=n_classes)

    # maximize matches => minimize per-row (max - C)
    row_max = C.max(axis=1, keepdims=True)              # rowwise, not global
    cost = row_max - C
    r, c = linear_sum_assignment(cost)

    # Map true cluster IDs -> assigned class IDs
    mapping = {int(uniq_clusters[ri]): int(ci) for ri, ci in zip(r, c)}
    return mapping, C

# --- distance helpers you already have ---
# fisher_rao_gaussian, wasserstein2_gaussian, eta_distance_gaussian

def _stack_src_means(mus_s):
    """mus_s can be dict {c: (d,)} or array (C,d). Returns (C,d) and class order."""
    if isinstance(mus_s, dict):
        classes = sorted(mus_s.keys())
        mu_src = np.stack([np.asarray(mus_s[c], dtype=float) for c in classes], axis=0)
    else:
        mu_src = np.asarray(mus_s, dtype=float)
        classes = list(range(mu_src.shape[0]))
    return mu_src, classes

def _get_src_cov_for_class(Sigma_s, c, d):
    """
    Sigma_s can be:
      - a single (d,d) shared covariance (np.ndarray)
      - a dict {c: (d,d)} of per-class covariances
      - None (fall back to identity)
    """
    if Sigma_s is None:
        return np.eye(d)
    if isinstance(Sigma_s, dict):
        return np.asarray(Sigma_s[c], dtype=float)
    return np.asarray(Sigma_s, dtype=float)  # shared


def _stack_means(means):
    """
    means: either (K,d) ndarray OR dict {k: (d,)}
    returns: (M, d) ndarray and the ordered keys
    """
    if isinstance(means, dict):
        keys = sorted(means.keys())
        arr = np.stack([np.asarray(means[k], dtype=float) for k in keys], axis=0)  # (M,d)
        return arr, keys
    m = np.asarray(means, dtype=float)
    return m, list(range(m.shape[0]))

def _stack_sigmas(sigmas):
    """
    sigmas: (K,d) diag OR (K,d,d) full OR dict {k: (d,)} OR {k: (d,d)}
    returns: stacked ndarray and a tag "diag" or "full"
    """
    if isinstance(sigmas, dict):
        keys = sorted(sigmas.keys())
        sample = np.asarray(sigmas[keys[0]])
        if sample.ndim == 1:  # diag
            arr = np.stack([np.asarray(sigmas[k], dtype=float) for k in keys], axis=0)  # (K,d)
            return arr, "diag"
        elif sample.ndim == 2:  # full
            arr = np.stack([np.asarray(sigmas[k], dtype=float) for k in keys], axis=0)  # (K,d,d)
            return arr, "full"
        else:
            raise ValueError(f"Unexpected Sigma ndim {sample.ndim} in dict")
    S = np.asarray(sigmas, dtype=float)
    if S.ndim == 2:
        return S, "diag"
    if S.ndim == 3:
        return S, "full"
    raise ValueError(f"Unexpected Sigma shape {S.shape}")

def _spd_logm(A: torch.Tensor, rel_floor: float = 1e-6, abs_floor: float = 1e-10) -> torch.Tensor:
    w, V = _spd_eigh(A, rel_floor, abs_floor)  # float64
    logw = torch.log(w)
    M = V @ torch.diag_embed(logw) @ V.mT
    return _sym(M).to(dtype=A.dtype, device=A.device)

def _spd_inv(A: torch.Tensor, rel_floor: float = 1e-6, abs_floor: float = 1e-10) -> torch.Tensor:
    w, V = _spd_eigh(A, rel_floor, abs_floor)
    invw = 1.0 / w
    M = V @ torch.diag_embed(invw) @ V.mT
    return _sym(M).to(dtype=A.dtype, device=A.device)

def fisher_rao_gaussian_diag(mu1, s1, mu2, s2, eps=1e-8):
    mu1 = np.asarray(mu1, float); mu2 = np.asarray(mu2, float)
    s1  = np.maximum(np.asarray(s1,  float), eps)
    s2  = np.maximum(np.asarray(s2,  float), eps)
    g   = np.sqrt(s1 * s2)
    quad = np.sum((mu2 - mu1)**2 / g)
    logp = np.sum(np.log(s2 / s1)**2)
    return np.sqrt(quad + logp)


# ---------- Fisher–Rao distance ----------
def fisher_rao_gaussian(mu1, Sigma1, mu2, Sigma2, eps=1e-12):
    """
    Fisher–Rao geodesic distance on the Gaussian manifold.
    d_FR^2 = (mu2-mu1)^T (Sigma1 # Sigma2)^{-1} (mu2-mu1)
             + 0.5 * || log( Sigma1^{-1/2} Sigma2 Sigma1^{-1/2} ) ||_F^2
    where '#' is the affine-invariant geometric mean.
    In 1D this reduces to: sqrt( (Δμ)^2 / sqrt(σ1^2 σ2^2) + 0.5*(log(σ2^2/σ1^2))^2 ).
    """
    mu1 = np.atleast_1d(mu1).astype(np.float64)
    mu2 = np.atleast_1d(mu2).astype(np.float64)
    Sigma1 = np.atleast_2d(Sigma1).astype(np.float64)
    Sigma2 = np.atleast_2d(Sigma2).astype(np.float64)

    # mean part with geometric-mean covariance
    Sgeom = _spd_geometric_mean(Sigma1, Sigma2, eps)
    Sgeom_inv = _spd_inv(Sgeom, eps)
    dmu = (mu2 - mu1).reshape(-1, 1)
    mean_term = float(dmu.T @ Sgeom_inv @ dmu)

    # covariance part (affine-invariant / Fisher on SPD)
    S1ih = _spd_invsqrt(Sigma1, eps)
    A = S1ih @ Sigma2 @ S1ih
    cov_term = 0.5 * np.linalg.norm(_spd_logm(A, eps), 'fro')**2

    return np.sqrt(mean_term + cov_term)

# ---------- “eta” (natural-parameter Euclidean) distance ----------
def eta_distance_gaussian(mu1, Sigma1, mu2, Sigma2, eps=1e-12):
    """
    Euclidean distance in natural-parameter space:
      eta1 = Sigma^{-1} mu   (vector)
      eta2 = -1/2 Sigma^{-1} (matrix)
    d_eta = sqrt( ||eta1_1 - eta1_2||_2^2 + ||eta2_1 - eta2_2||_F^2 )
    In 1D this matches your cost_eta: sqrt( (μ/λ diff)^2 + ( -1/(2λ) diff )^2 ).
    """
    mu1 = np.atleast_1d(mu1).astype(np.float64)
    mu2 = np.atleast_1d(mu2).astype(np.float64)
    Sigma1 = np.atleast_2d(Sigma1).astype(np.float64)
    Sigma2 = np.atleast_2d(Sigma2).astype(np.float64)

    S1i = _spd_inv(Sigma1, eps)
    S2i = _spd_inv(Sigma2, eps)

    eta1_1 = S1i @ mu1
    eta1_2 = S2i @ mu2
    eta2_1 = -0.5 * S1i
    eta2_2 = -0.5 * S2i

    vpart = float(np.linalg.norm(eta1_1 - eta1_2)**2)
    mpart = float(np.linalg.norm(eta2_1 - eta2_2, ord='fro')**2)
    return np.sqrt(vpart + mpart)

def _match_by_prototypes_metric(mu_clusters_in,
                                Sigma_clusters_in,
                                mus_s_in, Sigma_s_in,
                                metric="euclidean"):
    # Means and keys
    mu_clusters, cluster_keys = _stack_means(mu_clusters_in)   # (K,d)
    mus_s,       class_keys   = _stack_means(mus_s_in)         # (C,d)
    K, d = mu_clusters.shape
    C, d2 = mus_s.shape
    assert d == d2, f"d mismatch: {d} vs {d2}"

    # Covariances (allow None => identity)
    Sigma_clusters, kindA = _stack_sigmas(Sigma_clusters_in)   # (K,d) or (K,d,d) or None
    Sigma_s,       kindB  = _stack_sigmas(Sigma_s_in)
    if Sigma_clusters is None:
        Sigma_clusters = np.tile(np.eye(d)[None, ...], (K, 1, 1))
        kindA = "full"
    if Sigma_s is None:
        Sigma_s = np.tile(np.eye(d)[None, ...], (C, 1, 1))
        kindB = "full"

    def to_full(S, is_diag):
        if is_diag:
            return np.diag(S) if S.ndim == 1 else np.diag(np.asarray(S).ravel())
        return S

    def get_pair_cov(i, j, need_full: bool):
        Sig_k = Sigma_clusters[i]
        Sig_c = Sigma_s[j]
        if need_full:
            Sig_k = to_full(Sig_k, kindA == "diag")
            Sig_c = to_full(Sig_c, kindB == "diag")
        # jitter for SPD safety
        eps = 1e-8
        if Sig_k.ndim == 2:
            Sig_k = 0.5 * (Sig_k + Sig_k.T) + eps * np.eye(d)
        if Sig_c.ndim == 2:
            Sig_c = 0.5 * (Sig_c + Sig_c.T) + eps * np.eye(d)
        return Sig_k, Sig_c

    D = np.zeros((K, C), dtype=float)
    m = metric.lower()

    for i in range(K):
        mu_k = mu_clusters[i]
        for j in range(C):
            mu_c = mus_s[j]
            if m == "euclidean":
                D[i, j] = float(np.sum((mu_k - mu_c) ** 2))
            elif m == "w2":
                Sig_k, Sig_c = get_pair_cov(i, j, need_full=True)
                D[i, j] = wasserstein2_gaussian(mu_k, Sig_k, mu_c, Sig_c, squared=False)
            elif m == "fr":
                if kindA == "diag" and kindB == "diag":
                    D[i, j] = fisher_rao_gaussian_diag(mu_k, Sigma_clusters[i], mu_c, Sigma_s[j])
                else:
                    Sig_k, Sig_c = get_pair_cov(i, j, need_full=True)
                    D[i, j] = fisher_rao_gaussian(mu_k, Sig_k, mu_c, Sig_c)
            elif m == "eta":
                Sig_k, Sig_c = get_pair_cov(i, j, need_full=True)
                D[i, j] = eta_distance_gaussian(mu_k, Sig_k, mu_c, Sig_c)
            elif m in {"kl", "symkl"}:
                Sig_k, Sig_c = get_pair_cov(i, j, need_full=True)
                de = 1e-8
                S0 = 0.5 * (np.asarray(Sig_k) + np.asarray(Sig_k).T) + de * np.eye(d)
                S1 = 0.5 * (np.asarray(Sig_c) + np.asarray(Sig_c).T) + de * np.eye(d)
                invS1 = np.linalg.inv(S1)
                diff = (mu_c - mu_k).reshape(-1, 1)
                tr_term = float(np.trace(invS1 @ S0))
                quad = float((diff.T @ invS1 @ diff).squeeze())
                _, logdet0 = np.linalg.slogdet(S0)
                _, logdet1 = np.linalg.slogdet(S1)
                kl_kc = 0.5 * (tr_term + quad - d + (logdet1 - logdet0))
                if m == "kl":
                    D[i, j] = kl_kc
                else:
                    invS0 = np.linalg.inv(S0)
                    diff2 = (mu_k - mu_c).reshape(-1, 1)
                    tr_term2 = float(np.trace(invS0 @ S1))
                    quad2 = float((diff2.T @ invS0 @ diff2).squeeze())
                    _, logdet0b = np.linalg.slogdet(S0)  # already computed; kept for clarity
                    kl_ck = 0.5 * (tr_term2 + quad2 - d + (logdet0 - logdet1))
                    D[i, j] = 0.5 * (kl_kc + kl_ck)
            else:
                raise ValueError(f"Unknown metric '{metric}'.")

    r, c = linear_sum_assignment(D)  # minimize
    mapping = {int(cluster_keys[i]): int(class_keys[j]) for i, j in zip(r, c)}
    return mapping, D


def map_em_clusters(
    res,
    method: str = "pseudo",
    n_classes: int | None = None,
    pseudo_labels: np.ndarray | None = None,
    src_enc=None,
    pool: str = "gap",
    metric: str = "euclidean",
    mus_s=None,
    Sigma_s=None,
    priors_s=None,  # kept for API compatibility; unused here
):
    """
    Map EM clusters to class ids.

    Returns:
      mapping_dict   : {cluster_id -> class_id}
      labels_mapped  : (N) array of mapped class ids for each sample (from res["labels"])
      cost_obj       : For 'pseudo'      -> contingency matrix C (K_clusters x n_classes).
                       For 'prototypes'  -> distance matrix D (K_clusters x n_classes).
    """
    # ---- normalize method flag
    m = str(method).lower()
    if m == "pseudolabels":
        m = "pseudo"
    if m not in {"pseudo", "prototypes"}:
        raise ValueError("method must be one of {'pseudo','prototypes'}")

    # ---- get hard cluster ids from EM
    if "labels" not in res:
        raise ValueError("res must contain 'labels' (hard cluster assignments).")
    cluster_ids = np.asarray(res["labels"], dtype=int)
    if cluster_ids.ndim != 1:
        raise ValueError("res['labels'] must be a 1D array of cluster ids.")
    K_clusters = int(cluster_ids.max()) + 1 if cluster_ids.size > 0 else 0

    # ------------------------------------------------------------------
    # PSEUDO mode: construct contingency C and Hungarian match on (max(C)-C)
    # ------------------------------------------------------------------
    if m == "pseudo":
        if pseudo_labels is None:
            raise ValueError("pseudo_labels must be provided for method='pseudo'.")

        pseudo_labels = np.asarray(pseudo_labels, dtype=int)
        # length guard: align both arrays
        if pseudo_labels.shape[0] != cluster_ids.shape[0]:
            n = min(pseudo_labels.shape[0], cluster_ids.shape[0])
            # We truncate to the common length; prevents shape mismatch.
            pseudo_labels = pseudo_labels[:n]
            cluster_ids   = cluster_ids[:n]
            K_clusters    = int(cluster_ids.max()) + 1 if cluster_ids.size > 0 else 0

        if n_classes is None:
            n_classes = int(pseudo_labels.max()) + 1 if pseudo_labels.size > 0 else 0
        n_classes = int(n_classes)

        # Contingency counts C[k, c] = #samples assigned to cluster k with pseudo label c
        mapping_dict, contingency = _match_with_pseudolabels(
            cluster_ids, pseudo_labels, n_classes
        )
        # Build fast array mapping for label projection
        # Ensure coverage for any cluster id that may appear (even if unmapped)
        max_id = max(cluster_ids.max(initial=-1), max(mapping_dict.keys(), default=-1))
        mapping_array = np.arange(max_id + 1, dtype=int)
        # default identity doesn't make sense if unmapped cluster ids appear;
        # but _match_with_pseudolabels covers all present clusters, so this is safe.
        for cid, cls in mapping_dict.items():
            mapping_array[cid] = int(cls)

        labels_mapped = mapping_array[cluster_ids]
        # cost_obj = contingency (for later scalarization by caller)
        return mapping_dict, labels_mapped, contingency

    # ------------------------------------------------------------------
    # PROTOTYPES mode: compute distance matrix D and Hungarian on D
    # ------------------------------------------------------------------
    # Here 'cost' is a distance matrix D; smaller is better.
    if mus_s is None:
        if src_enc is None:
            raise ValueError("Provide either mus_s (+ optional Sigma_s) or src_enc for method='prototypes'.")
        mus_s = _compute_source_prototypes(src_enc, pool=pool)
        Sigma_s = None

    if "mu" not in res:
        raise ValueError("res['mu'] (cluster mean dict) is required for method='prototypes'.")

    mu_clusters = res["mu"]
    Sigma_clusters = res.get("Sigma", None)

    mapping_dict, D = _match_by_prototypes_metric(
        mu_clusters, Sigma_clusters, mus_s, Sigma_s, metric=metric
    )

    # map labels via array lookup
    max_id = max(cluster_ids.max(initial=-1), max(mapping_dict.keys(), default=-1))
    mapping_array = np.arange(max_id + 1, dtype=int)
    for cid, cls in mapping_dict.items():
        mapping_array[cid] = int(cls)
    labels_mapped = mapping_array[cluster_ids]

    # cost_obj = distance matrix D (for later scalarization by caller)
    return mapping_dict, labels_mapped, D

def fit_source_gaussian_params(X, y, pool: str = "gap"):
    y = np.asarray(y, dtype=int)
    if pool == "gap":
        X = _pool_features(X, pool=pool)
    else:
        X = X.reshape(X.shape[0], -1)
    classes = np.unique(y)
    d = X.shape[1]
    mus, priors, Sigma_s = {}, {}, {}
    for c in classes:
        Xc = X[y == c]
        if len(Xc) == 0:
            # fallback: zeros with small ridge to avoid singulars
            mus[c] = np.zeros(d, dtype=float)
            Sigma_s[c] = np.eye(d, dtype=float)
            priors[c] = 0.0
        else:
            mu_c = Xc.mean(axis=0)
            Z = Xc - mu_c
            # unbiased sample covariance + ridge
            S = (Z.T @ Z) / max(len(Xc) - 1, 1) + 1e-6 * np.eye(d)
            mus[c] = mu_c
            Sigma_s[c] = S
            priors[c] = float(len(Xc)) / float(len(X))
    return mus, Sigma_s, priors

# ---- helpers ---------------------------------------------------------------

def _pool_features(arr, pool='gap'):
    """Accepts (N,D) or (N,C,H,W). Returns (N,D')."""
    X = np.asarray(arr)
    if X.ndim == 4:
        if pool != 'gap':
            raise ValueError(f"Only 'gap' pooling implemented for 4D tensors, got {pool}.")
        return X.mean(axis=(2, 3))
    elif X.ndim == 2:
        return X
    else:
        raise ValueError(f"Unexpected feature shape {X.shape}. Expected (N,D) or (N,C,H,W).")




def _sigma_to_precision_init(Sigma_init, cov_type, reg=1e-6):
    """Convert dict of Σ_k to precisions_init in the shapes sklearn expects."""
    keys = sorted(Sigma_init.keys())
    if cov_type == "full":
        P = []
        for k in keys:
            S = np.asarray(Sigma_init[k], dtype=float)
            S = 0.5*(S+S.T) + reg*np.eye(S.shape[0])
            P.append(np.linalg.inv(S))
        return np.stack(P, axis=0)                             # (K,d,d)
    elif cov_type == "diag":
        P = []
        for k in keys:
            S = np.asarray(Sigma_init[k], dtype=float)
            if S.ndim == 2:
                v = np.diag(S)
            else:
                v = S
            v = np.maximum(v, reg)
            P.append(1.0 / v)
        return np.stack(P, axis=0)                             # (K,d)
    elif cov_type == "spherical":
        P = []
        for k in keys:
            S = np.asarray(Sigma_init[k], dtype=float)
            if S.ndim == 2:
                s2 = float(np.trace(S)/S.shape[0])
            else:
                s2 = float(np.mean(S))
            s2 = max(s2, reg)
            P.append(1.0 / s2)
        return np.asarray(P)                                   # (K,)
    else:
        raise ValueError("cov_type must be one of {'full','diag','spherical'}")



def em_k_gaussians_sklearn(
    X: np.ndarray,
    mu_init: dict,                 # {k: (d,)}
    Sigma_init: dict,              # {k: (d,d)} or diag/spherical as 2D diag
    pi_init: dict,                 # {k: float}
    max_iter: int = 100,
    tol: float = 1e-5,
    reg: float = 1e-6,             # -> reg_covar
    cov_type: str = "full",        # 'full' | 'diag' | 'spherical'
    verbose: bool = False,
    rng=None
):
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    
    keys = sorted(mu_init.keys())
    K = len(keys)   

    # pack initial means / weights
    means_init = np.stack([np.asarray(mu_init[k], dtype=float) for k in keys], axis=0)   # (K,d)
    weights_init = np.asarray([max(float(pi_init[k]), 0.0) for k in keys], dtype=float)
    s = weights_init.sum()
    if s <= 0:
        weights_init[:] = 1.0 / K
    else:
        weights_init /= s

    # Build GaussianMixture kwargs, guarding against invalid precision inits
    gm_kwargs = dict(
        n_components=K,
        covariance_type=cov_type,
        tol=tol,
        reg_covar=reg,
        max_iter=max_iter,
        init_params="random",
        means_init=means_init,
        weights_init=weights_init,
        random_state=None if rng is None else int(np.random.RandomState().randint(2**31 - 1)),
        verbose=2 if verbose else 0,
    )

    def _valid_full(P):
        try:
            # symmetric positive-definite check via Cholesky for every component
            for k in range(P.shape[0]):
                # symmetrize numerically
                M = 0.5 * (P[k] + P[k].T)
                # tiny jitter to avoid borderline failures
                M = M + reg * np.eye(M.shape[0])
                np.linalg.cholesky(M)
            return True
        except Exception:
            return False

    def _valid_diag(p):
        return np.all(np.isfinite(p)) and np.all(p > 0)

    def _valid_spherical(p):
        return np.all(np.isfinite(p)) and np.all(p > 0)

    try:
        precisions_init = _sigma_to_precision_init(Sigma_init, cov_type, reg=reg)
        if cov_type == "full":
            # Skip providing precisions_init for 'full' to avoid strict SPD rejections.
            # We rely on sklearn's internal initialization with our means/weights and reg_covar.
            if verbose:
                print("[EM] Skipping precisions_init for 'full' (use sklearn init)")
        elif cov_type == "diag":
            p = np.maximum(precisions_init, reg)
            if _valid_diag(p):
                gm_kwargs["precisions_init"] = p
        elif cov_type == "spherical":
            p = np.maximum(precisions_init, reg)
            if _valid_spherical(p):
                gm_kwargs["precisions_init"] = p
        else:
            pass
    except Exception as _e:
        if verbose:
            print(f"[EM] Could not build/sanitize precisions_init ({_e}); using sklearn init")

    gm = GaussianMixture(**gm_kwargs)
    gm.fit(X)

    # posteriors (responsibilities)
    gamma = gm.predict_proba(X)                          # (n,K)

    # unpack params back to your dict format
    mu = {k: gm.means_[i].copy() for i, k in enumerate(keys)}
    pi = {k: float(gm.weights_[i]) for i, k in enumerate(keys)}

    # get covariances; for 'diag' return per-dimension variances (K,d)
    if cov_type == "full":
        covs = gm.covariances_                          # (K,d,d)
    elif cov_type == "diag":
        covs = np.asarray(gm.covariances_, dtype=float)                 # (K,d)
    elif cov_type == "spherical":
        covs = np.stack([np.eye(d) * s2 for s2 in gm.covariances_], axis=0)  # (K,d,d)
    else:
        raise ValueError("cov_type must be one of {'full','diag','spherical'}")

    Sigma = {k: covs[i].copy() for i, k in enumerate(keys)}

    # sklearn exposes average log-likelihood per sample; emulate a single-point history
    ll_final = float(gm.lower_bound_) * n
    ll_history = [ll_final]

    return mu, Sigma, pi, gamma, ll_history

def apply_em_bundle_to_target(bundle, e_tgt, tgt_trainset):
    """
    Put EM labels and mapped soft posteriors on the target dataset so *all* methods
    can read them without refitting EM.
    """
    # hard labels
    labels_em = np.asarray(bundle.labels_em, dtype=np.int64)
    e_tgt.targets_em = torch.as_tensor(labels_em, dtype=torch.long)
    tgt_trainset.targets_em = e_tgt.targets_em.cpu().clone()

    # optional soft posteriors (class posteriors after cluster->class mapping)
    if getattr(bundle, "P_soft", None) is not None:
        tgt_trainset.soft_targets = torch.from_numpy(bundle.P_soft).float()