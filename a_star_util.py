import numpy as np
from typing import Tuple, Dict, List, Optional
from sklearn import base
from sklearn.cluster import kmeans_plusplus
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import os
# If you kept the generalized EM/K-means++ from earlier:
# from unsup_exp import gmm_em_k, kmeanspp_init  # K-class EM + kmeans++ init
# (names below assume the ones I provided earlier; adjust to your file)
from sklearn.metrics import accuracy_score, confusion_matrix
import time
from dataset import *
import torch.nn as nn

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

@torch.no_grad()
def class_stats_diag(Z: torch.Tensor, y: torch.Tensor, K: int, eps: float = 1e-6):
    """Per-class mean and diagonal var on embeddings Z (N×d)."""
    d = Z.shape[1]
    mus = torch.zeros(K, d, device=Z.device)
    vars_ = torch.ones(K, d, device=Z.device)
    priors = torch.zeros(K, device=Z.device)
    for k in range(K):
        Zk = Z[y == k]
        n = Zk.size(0)
        priors[k] = n
        if n == 0: 
            continue
        mus[k] = Zk.mean(0)
        vars_[k] = Zk.var(0, unbiased=False).clamp_min(eps)
    priors = (priors + 1.0) / (priors.sum() + K)  # Laplace
    return mus, vars_, priors

def fr_interp_diag(mu_s, var_s, mu_t, var_t, t, eps=1e-8):
    """Fisher–Rao geodesic for diagonal Gaussians (element-wise) with safe ratios.

    Avoids Inf/NaN when some variances are near zero by clamping both numerator
    and denominator before division.
    """
    var_s_safe = var_s.clamp_min(eps)
    var_t_safe = var_t.clamp_min(eps)
    r = (var_t_safe / var_s_safe)              # ratio
    var_mid = var_s_safe * torch.pow(r, t)
    num = torch.pow(r, t) - 1.0
    den = r - 1.0
    w = torch.where(den.abs() < 1e-6, torch.full_like(den, t), num / den)
    mu_mid = mu_s + w * (mu_t - mu_s)
    return mu_mid, var_mid

def sample_diag(mu, var, n):
    std = var.clamp_min(1e-8).sqrt()
    return torch.randn(n, mu.numel(), device=mu.device) * std + mu


@torch.no_grad()
def class_stats_full(Z: torch.Tensor, y: torch.Tensor, K: int, reg: float, ddof: int):
    """Per-class mean and full covariance (K×d×d), with eigenvalue regularization."""
    N, d = Z.shape
    mus   = torch.zeros(K, d, device=Z.device, dtype=Z.dtype)
    Sig   = torch.zeros(K, d, d, device=Z.device, dtype=Z.dtype)
    prior = torch.zeros(K, device=Z.device, dtype=Z.dtype)
    for k in range(K):
        Zk = Z[y == k]
        n  = Zk.size(0)
        prior[k] = n
        if n == 0:
            continue
        mu = Zk.mean(0, keepdim=True)                   # (1,d)
        Xc = Zk - mu                                    # (n,d)
        # empirical covariance (ddof matches caller)
        denom = max(1, n - ddof)
        C = (Xc.T @ Xc) / float(denom)                  # (d,d)
        # make SPD via eigen-regularization
        w, V = torch.linalg.eigh(C)
        w = torch.clamp(w, min=0.0) + reg
        C = (V * w) @ V.T
        mus[k] = mu.squeeze(0)
        Sig[k] = C
    prior = (prior + 1.0) / (prior.sum() + K)           # Laplace
    return mus, Sig, prior

def sample_full(mu: torch.Tensor, Sigma: torch.Tensor, n: int, jitter: float=1e-6) -> torch.Tensor:
    """Sample from N(mu, Sigma) with SPD safety."""
    with torch.no_grad():
        d = mu.numel()
        w, V = torch.linalg.eigh(Sigma)
        w = torch.clamp(w, min=0.0)
        eps = max(jitter, 1e-12 * (float(torch.trace(Sigma)) / d + 1e-12))
        w = w + eps
        A = (V * torch.sqrt(w)) @ V.T                   # Sigma^{1/2}
        Z = torch.randn(n, d, device=mu.device, dtype=mu.dtype)
        return Z @ A.T + mu                             # (n,d)

def fr_interp_full(mu_s, Sig_s, mu_t, Sig_t, t: float, eps: float = 1e-8, reg: float = 1e-6):
    """Fisher–Rao geodesic for full SPD covariances + same mean rule as diag case."""
    # Σ_mid = Σ_s^{1/2} ( Σ_s^{-1/2} Σ_t Σ_s^{-1/2} )^t Σ_s^{1/2}
    ws, Vs = torch.linalg.eigh(Sig_s)
    ws = torch.clamp(ws, min=0.0) + reg
    Sig_s_invh = (Vs * (ws.clamp_min(eps) ** -0.5)) @ Vs.T    # Σ_s^{-1/2}
    Sig_s_h    = (Vs * torch.sqrt(ws)) @ Vs.T                 # Σ_s^{1/2}

    M = Sig_s_invh @ Sig_t @ Sig_s_invh
    wm, Vm = torch.linalg.eigh(M)
    wm = torch.clamp(wm, min=0.0) + reg
    M_pow_t = (Vm * (wm ** t)) @ Vm.T

    Sig_mid = Sig_s_h @ M_pow_t @ Sig_s_h

    # mean rule consistent with your diag version (element-wise ratio reduces to scalar cap)
    # We use coordinate-wise rule with denominators guarded by eps:
    #   w = ((r^t - 1) / (r - 1)) where r is per-dim variance ratio
    # For full covariances we approximate with diag ratios of Σ:
    var_s = torch.diag(Sig_s)
    var_t = torch.diag(Sig_t)
    r = (var_t.clamp_min(eps) / var_s.clamp_min(eps))
    num = torch.pow(r, t) - 1.0
    den = r - 1.0
    w = torch.where(den.abs() < 1e-6, torch.full_like(den, t), num / den)
    # project this per-dim weight to a single scalar α via average (stable, matches diag if isotropic)
    alpha = float(w.mean().item())
    mu_mid = mu_s + alpha * (mu_t - mu_s)
    return mu_mid, Sig_mid


def generate_fr_domains_between(
    n_inter,
    dataset_s,
    dataset_t,
    plan=None,
    entry_cutoff: int = 0,
    conf: float = 0.0,
    source_model: Optional[torch.nn.Module] = None,
    pseudolabels: Optional[torch.Tensor] = None,
    visualize: bool = False,
    save_path: Optional[str] = None,
    *,
    cov_type: str = "diag",     # NEW: 'diag' or 'full'
    reg: float = 1e-6,          # NEW: covariance regularization (added to eigenvalues)
    ddof: int = 0,              # NEW: 0 → population covariance (matches class_stats_diag)
    jitter: float = 1e-6,       # NEW: SPD safety margin for sampling/inversion
):
    """
    Generate FR-interpolated domains assuming dataset_t.targets_em already set.

    Returns:
      all_domains   : List[DomainDataset]
      target_em     : np.ndarray of target EM labels
      domain_params : dict with per-step parameters (source, intermediates, target)

    When cov_type='full', we estimate per-class full Σ and interpolate along the
    Fisher–Rao geodesic:
       Σ(t) = Σ_s^{1/2} ( Σ_s^{-1/2} Σ_t Σ_s^{-1/2} )^t Σ_s^{1/2}
    Means follow the same scalar weight rule you used for the diagonal case.
    """
    assert cov_type in {"diag", "full"}, "cov_type must be 'diag' or 'full'"

    print("------------Generate Intermediate domains (FR)----------")
    _t_total = time.time()

    xs = dataset_s.data
    xt = dataset_t.data
    ys = dataset_s.targets
    if len(xs.shape) > 2:
        xs, xt = nn.Flatten()(xs), nn.Flatten()(xt)
    device = xs.device if torch.is_tensor(xs) else torch.device("cpu")

    if ys is None:
        raise ValueError("Source dataset must provide targets for class statistics.")
    if not hasattr(dataset_t, "targets_em") or dataset_t.targets_em is None:
        raise ValueError("Target dataset must provide targets_em (EM-derived labels).")

    Zs = xs if torch.is_tensor(xs) else torch.as_tensor(xs)
    Zs = Zs.to(device)
    Ys = ys if torch.is_tensor(ys) else torch.as_tensor(ys)
    Ys = Ys.to(device, dtype=torch.long)
    if Ys.numel() == 0:
        return [], torch.empty(0, dtype=torch.long)

    Zt = xt if torch.is_tensor(xt) else torch.as_tensor(xt)
    Zt = Zt.to(device)
    Yt_em = dataset_t.targets_em if torch.is_tensor(dataset_t.targets_em) else torch.as_tensor(dataset_t.targets_em)
    Yt_em = Yt_em.to(device, dtype=torch.long)

    # ---------- helpers ----------
    def _to_np(x):
        return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


    # ---------- basic class stats ----------
    K = int(max(Ys.max(), Yt_em.max()).item()) + 1
    if cov_type == "diag":
        mus_s, vars_s, _ = class_stats_diag(Zs, Ys, K)
        mus_t, vars_t, _ = class_stats_diag(Zt, Yt_em, K)
        Sig_s = Sig_t = None
    else:
        mus_s, Sig_s, _ = class_stats_full(Zs, Ys, K, reg=reg, ddof=ddof)
        mus_t, Sig_t, _ = class_stats_full(Zt, Yt_em, K, reg=reg, ddof=ddof)
        # also keep diagonals for logging / compatibility
        vars_s = torch.stack([torch.diag(Sig_s[k]) for k in range(K)], dim=0)
        vars_t = torch.stack([torch.diag(Sig_t[k]) for k in range(K)], dim=0)

    counts_s = torch.bincount(Ys, minlength=K)
    counts_t = torch.bincount(Yt_em, minlength=K)

    d = mus_s.shape[1]
    total_s = int(counts_s.sum().item())
    total_t = int(counts_t.sum().item())
    total_min = int(torch.minimum(counts_s, counts_t).sum().item())
    if total_min <= 0:
        total_min = max(total_s, total_t)

    # ---------- containers (consistent schema) ----------
    steps, mu_list, var_list, counts_list, pi_list = [], [], [], [], []
    Sigma_list = [] if cov_type == "full" else None

    def _pad_kd(mu_like, var_like, present_mask):
        mu_out  = np.full((K, d), np.nan, dtype=np.float64)
        var_out = np.full((K, d), np.nan, dtype=np.float64)
        mu_np   = _to_np(mu_like)
        var_np  = _to_np(var_like)
        mu_out[present_mask]  = mu_np[present_mask]
        var_out[present_mask] = var_np[present_mask]
        return mu_out, var_out

    def _pad_kdd(Sig_like, present_mask):
        Sig_out = np.full((K, d, d), np.nan, dtype=np.float64)
        Sig_np  = _to_np(Sig_like)
        Sig_out[present_mask] = Sig_np[present_mask]
        return Sig_out

    present_s = (counts_s.cpu().numpy() > 0)
    present_t = (counts_t.cpu().numpy() > 0)

    # ---------- Source (t=0) ----------
    pi_s = (counts_s.float() / max(1, total_s)).cpu().numpy()
    mu0, var0 = _pad_kd(mus_s, vars_s, np.ones(K, dtype=bool))
    steps.append(0.0); mu_list.append(mu0); var_list.append(var0)
    counts_list.append(counts_s.cpu().numpy().astype(np.int64)); pi_list.append(pi_s)
    if cov_type == "full":
        Sigma0 = _pad_kdd(Sig_s, np.ones(K, dtype=bool))
        Sigma_list.append(Sigma0)

    all_domains: List[DomainDataset] = []

    # ---------- Intermediates ----------
    for i in range(1, n_inter + 1):
        _t_step = time.time()
        t = i / (n_inter + 1)

        # class allocation (keeps your current behavior)
        pi_mid = (counts_s.float() / max(1, total_s))
        desired = (pi_mid * float(total_min)).clamp_min(0.0)
        base = torch.floor(desired)
        frac = desired - base
        present = ((counts_s > 0) & (counts_t > 0)).float()
        base = base * present
        frac = frac * present
        n_alloc = int(base.sum().item())
        rem = int(max(0, total_min - n_alloc))
        if rem > 0:
            k_take = min(rem, K)
            if k_take > 0:
                _, idx = torch.topk(frac, k=k_take)
                add = torch.zeros_like(base); add[idx] = 1.0
                base = base + add
        n_per_class = base.long()

        mu_mid_full  = np.full((K, d), np.nan, dtype=np.float64)
        var_mid_full = np.full((K, d), np.nan, dtype=np.float64)
        Sig_mid_full = np.full((K, d, d), np.nan, dtype=np.float64) if cov_type == "full" else None

        Zm_list: List[torch.Tensor] = []
        Ym_list: List[torch.Tensor] = []

        for k_idx in range(K):
            n_k = int(n_per_class[k_idx].item())
            if n_k <= 0:
                continue
            if counts_s[k_idx].item() == 0 or counts_t[k_idx].item() == 0:
                continue

            if cov_type == "diag":
                mu_s_k, var_s_k = mus_s[k_idx], vars_s[k_idx]
                mu_t_k, var_t_k = mus_t[k_idx], vars_t[k_idx]
                mu_mid, var_mid = fr_interp_diag(mu_s_k, var_s_k, mu_t_k, var_t_k, t)
                Zk = sample_diag(mu_mid, var_mid, n_k)
            else:
                mu_s_k, Sig_s_k = mus_s[k_idx], Sig_s[k_idx]
                mu_t_k, Sig_t_k = mus_t[k_idx], Sig_t[k_idx]
                mu_mid, Sig_mid = fr_interp_full(mu_s_k, Sig_s_k, mu_t_k, Sig_t_k, t, eps=1e-8)
                Zk = sample_full(mu_mid, Sig_mid, n_k)
            mu_mid_full[k_idx]  = _to_np(mu_mid)
            var_mid_full[k_idx] = _to_np(torch.diag(Sig_mid))
            Sig_mid_full[k_idx] = _to_np(Sig_mid)

            Yk = torch.full((n_k,), k_idx, device=device, dtype=torch.long)
            Zm_list.append(Zk); Ym_list.append(Yk)

        # record mid-step params (and check for NaNs; fallback if needed)
        steps.append(float(t))
        mu_list.append(mu_mid_full)
        var_list.append(var_mid_full)
        counts_list.append(n_per_class)
        pi_list.append(_to_np(pi_mid))

        if cov_type == "full":
            # if a class stayed NaN (e.g., absent), replace with diag(var)
            for k_idx in range(K):
                if not np.isfinite(Sig_mid_full[k_idx]).all():
                    v = var_mid_full[k_idx]
                    if np.isfinite(v).all():
                        Sig_mid_full[k_idx] = np.diag(np.clip(v, 1e-12, None))
            Sigma_list.append(Sig_mid_full)

        if not Zm_list:
            continue
        Zm = torch.cat(Zm_list, 0).cpu().float()
        Ym = torch.cat(Ym_list, 0).cpu().long()
        weights = torch.ones(len(Ym))
        all_domains.append(DomainDataset(Zm, weights, Ym, Ym))
        print(f"[FR] Step {i}/{n_inter}: generated {len(Ym)} samples with d={Zm.shape[1]} in {time.time()-_t_step:.2f}s")

    # ---------- Target (t=1) ----------
    try:
        X_tgt_final = xt if torch.is_tensor(xt) else torch.as_tensor(xt)
        X_tgt_final = X_tgt_final.cpu()
        Y_tgt_final = dataset_t.targets if torch.is_tensor(dataset_t.targets) else torch.as_tensor(dataset_t.targets, dtype=torch.long)
        Y_tgt_final = Y_tgt_final.cpu().long()
        Y_em_final  = Yt_em.cpu().long()
        W_tgt_final = torch.ones(len(Y_em_final))
        all_domains.append(DomainDataset(X_tgt_final, W_tgt_final, Y_tgt_final, Y_em_final))
    except Exception as e:
        print(f"[FR] Warning: failed to wrap target with targets_em ({e}); appending raw dataset.")
        all_domains.append(dataset_t)

    pi_t = (counts_t.float() / max(1, total_t)).cpu().numpy()
    muT, varT = _pad_kd(mus_t, vars_t, np.ones(K, dtype=bool))
    steps.append(1.0); mu_list.append(muT); var_list.append(varT)
    counts_list.append(counts_t.cpu().numpy().astype(np.int64)); pi_list.append(pi_t)
    if cov_type == "full":
        SigT = _pad_kdd(Sig_t, np.ones(K, dtype=bool))
        Sigma_list.append(SigT)

    print(f"[FR] Total generation time: {time.time()-_t_total:.2f}s")
    generated_size = len(all_domains[-2].data) if len(all_domains) > 1 else 0
    print(f"Total data for each intermediate domain: {generated_size}")

    # ---------- pack domain params ----------
    domain_params = {
        "K": int(K),
        "d": int(d),
        "cov_type": cov_type,
        "steps": np.asarray(steps, dtype=np.float64),            # (S,)
        "mu":    np.asarray(mu_list,  dtype=np.float64),         # (S, K, d)
        "var":   np.asarray(var_list, dtype=np.float64),         # (S, K, d)  (always provided)
        "counts": np.asarray(counts_list, dtype=np.int64),       # (S, K)
        "pi":      np.asarray(pi_list,    dtype=np.float64),     # (S, K)
        "present_source": present_s.astype(np.bool_),
        "present_target": present_t.astype(np.bool_),
    }
    if cov_type == "full":
        domain_params["Sigma"] = np.asarray(Sigma_list, dtype=np.float64)  # (S, K, d, d)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez_compressed(save_path, **domain_params)
        print(f"[FR] Saved domain parameters -> {save_path}")

    return all_domains, Yt_em.cpu().numpy(), domain_params


@torch.no_grad()
def natural_interp_full(mu_s: torch.Tensor,
                        Sigma_s: torch.Tensor,
                        mu_t: torch.Tensor,
                        Sigma_t: torch.Tensor,
                        t: float,
                        eps: float = 1e-6):
    """
    Natural-parameter interpolation for full-covariance Gaussians.

    Inputs (per class; no batching assumed):
      mu_s     : (d,)   source mean
      Sigma_s  : (d,d)  source covariance (SPD)
      mu_t     : (d,)   target mean
      Sigma_t  : (d,d)  target covariance (SPD)
      t        : float in [0,1]
      eps      : jitter added to covariances for SPD safety

    Returns:
      mu_m     : (d,)   interpolated mean
      Sigma_m  : (d,d)  interpolated covariance (SPD)
    """
    d = mu_s.numel()
    I = torch.eye(d, dtype=Sigma_s.dtype, device=Sigma_s.device)

    def _spd_prec(S):
        # make SPD (jitter) and compute precision via Cholesky solve (no explicit inverse)
        S = 0.5 * (S + S.T) + eps * I
        L = torch.linalg.cholesky(S)
        # Solve S^{-1} = (L L^T)^{-1} = L^{-T} L^{-1}
        # Use identity: A^{-1} B by solving A X = B
        # Here we want full matrix: solve for columns of I
        inv_S = torch.cholesky_inverse(L)  # stable, symmetric
        return inv_S

    # Precisions (Λ = Σ^{-1})
    Lambda_s = _spd_prec(Sigma_s)
    Lambda_t = _spd_prec(Sigma_t)

    # Natural parameters: η1 = Λ μ, η2 = -1/2 Λ
    eta1_s = Lambda_s @ mu_s
    eta2_s = -0.5 * Lambda_s
    eta1_t = Lambda_t @ mu_t
    eta2_t = -0.5 * Lambda_t

    # Linear interpolation in natural-parameter space
    eta1_m = (1.0 - t) * eta1_s + t * eta1_t
    eta2_m = (1.0 - t) * eta2_s + t * eta2_t

    # Back to (μ, Σ): Λ_m = -2 η2_m, Σ_m = Λ_m^{-1}, μ_m = Λ_m^{-1} η1_m
    Lambda_m = -2.0 * eta2_m
    Lambda_m = 0.5 * (Lambda_m + Lambda_m.T) + eps * I  # symmetrize + SPD safety

    # Invert Λ_m stably and solve for μ_m
    Lm = torch.linalg.cholesky(Lambda_m)
    Sigma_m = torch.cholesky_inverse(Lm)
    # μ_m = Λ_m^{-1} η1_m  -> solve Λ_m μ = η1
    mu_m = torch.cholesky_solve(eta1_m.unsqueeze(1), Lm).squeeze(1)

    # final symmetry clean-up
    Sigma_m = 0.5 * (Sigma_m + Sigma_m.T)

    return mu_m, Sigma_m


@torch.no_grad()
def natural_interp_diag(mu_s_k, var_s_k, mu_t_k, var_t_k, t, eps: float = 1e-8):
    # precision vectors
    prec_s = (1.0 / var_s_k.clamp_min(eps))
    prec_t = (1.0 / var_t_k.clamp_min(eps))
    # natural params
    eta1_s = prec_s * mu_s_k
    eta2_s = -0.5 * prec_s
    eta1_t = prec_t * mu_t_k
    eta2_t = -0.5 * prec_t
    # interpolate
    eta1_m = (1.0 - t) * eta1_s + t * eta1_t
    eta2_m = (1.0 - t) * eta2_s + t * eta2_t
    # back to mean/var
    prec_m = (-2.0) * eta2_m
    prec_m = prec_m.clamp_min(eps)
    var_m = 1.0 / prec_m
    mu_m = var_m * eta1_m
    return mu_m, var_m


def generate_natural_domains_between(
    n_inter,
    dataset_s,
    dataset_t,
    plan=None,
    entry_cutoff: int = 0,
    conf: float = 0.0,
    source_model: Optional[torch.nn.Module] = None,
    pseudolabels: Optional[torch.Tensor] = None,
    visualize: bool = False,
    save_path: Optional[str] = None,
    cov_type = "diag",
    reg: float = 1e-6,          # NEW: covariance regularization (added to eigenvalues)
    ddof: int = 0,              # NEW: 0 → population covariance (matches class_stats_diag)
    jitter: float = 1e-6,       # NEW: SPD safety margin for sampling/inversion
):
    """
    Generate intermediate domains by linear interpolation in the natural parameter
    space of diagonal Gaussian class conditionals.

    For a Gaussian N(mu, Sigma) with diagonal variance vector `var`, the natural
    parameters are:
      eta1 = inv(var) * mu
      eta2 = -0.5 * inv(var)

    We interpolate eta = (1-t) * eta_source + t * eta_target and then map back:
      precision_mid = -2 * eta2_mid  ⇒  var_mid = 1 / precision_mid
      mu_mid        = var_mid * eta1_mid

    Input and output follow generate_fr_domains_between.
    """
    print("------------Generate Intermediate domains (NATURAL)----------")
    _t_total = time.time()

    xs = dataset_s.data
    xt = dataset_t.data
    ys = dataset_s.targets
    if len(xs.shape) > 2:
        xs, xt = nn.Flatten()(xs), nn.Flatten()(xt)
    device = xs.device if torch.is_tensor(xs) else torch.device("cpu")

    if ys is None:
        raise ValueError("Source dataset must provide targets for class statistics.")
    if not hasattr(dataset_t, "targets_em") or dataset_t.targets_em is None:
        raise ValueError("Target dataset must provide targets_em (EM-derived labels).")

    Zs = xs if torch.is_tensor(xs) else torch.as_tensor(xs)
    Zs = Zs.to(device)
    Ys = ys if torch.is_tensor(ys) else torch.as_tensor(ys)
    Ys = Ys.to(device, dtype=torch.long)
    if Ys.numel() == 0:
        return [], torch.empty(0, dtype=torch.long)

    Zt = xt if torch.is_tensor(xt) else torch.as_tensor(xt)
    Zt = Zt.to(device)
    Yt_em = dataset_t.targets_em if torch.is_tensor(dataset_t.targets_em) else torch.as_tensor(dataset_t.targets_em)
    Yt_em = Yt_em.to(device, dtype=torch.long)

    K = int(max(Ys.max(), Yt_em.max()).item()) + 1
    if cov_type == "diag":
        mus_s, vars_s, _ = class_stats_diag(Zs, Ys, K)
        mus_t, vars_t, _ = class_stats_diag(Zt, Yt_em, K)
        Sig_s = Sig_t = None
    else:
        mus_s, Sig_s, _ = class_stats_full(Zs, Ys, K, reg=reg, ddof=ddof)
        mus_t, Sig_t, _ = class_stats_full(Zt, Yt_em, K, reg=reg, ddof=ddof)
        # also keep diagonals for logging / compatibility
        vars_s = torch.stack([torch.diag(Sig_s[k]) for k in range(K)], dim=0)
        vars_t = torch.stack([torch.diag(Sig_t[k]) for k in range(K)], dim=0)

    counts_s = torch.bincount(Ys, minlength=K)
    counts_t = torch.bincount(Yt_em, minlength=K)
    d = mus_s.shape[1]
    total_s = int(counts_s.sum().item())
    total_t = int(counts_t.sum().item())
    total_min = int(torch.minimum(counts_s, counts_t).sum().item())
    if total_min <= 0:
        total_min = max(total_s, total_t)
    # ---------- containers (consistent schema) ----------
    steps, mu_list, var_list, counts_list, pi_list = [], [], [], [], []
    Sigma_list = [] if cov_type == "full" else None

    # ---------- helpers ----------
    def _to_np(x):
        return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


    def _pad_kd(mu_like, var_like, present_mask):
        mu_out  = np.full((K, d), np.nan, dtype=np.float64)
        var_out = np.full((K, d), np.nan, dtype=np.float64)
        mu_np   = _to_np(mu_like)
        var_np  = _to_np(var_like)
        mu_out[present_mask]  = mu_np[present_mask]
        var_out[present_mask] = var_np[present_mask]
        return mu_out, var_out

    def _pad_kdd(Sig_like, present_mask):
        Sig_out = np.full((K, d, d), np.nan, dtype=np.float64)
        Sig_np  = _to_np(Sig_like)
        Sig_out[present_mask] = Sig_np[present_mask]
        return Sig_out

    def _to_spd(S: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        # Symmetrize + clamp eigenvalues
        S = 0.5 * (S + S.T)
        w, V = torch.linalg.eigh(S)
        w = torch.clamp(w, min=eps)
        return (V * w) @ V.T

    def _diag_from_full(S):
        return np.diag(S).copy()

    present_s = (counts_s.cpu().numpy() > 0)
    present_t = (counts_t.cpu().numpy() > 0)
    # ---------- Source (t=0) ----------
    pi_s = (counts_s.float() / max(1, total_s)).cpu().numpy()
    mu0, var0 = _pad_kd(mus_s, vars_s, np.ones(K, dtype=bool))
    steps.append(0.0); mu_list.append(mu0); var_list.append(var0)
    counts_list.append(counts_s.cpu().numpy().astype(np.int64)); pi_list.append(pi_s)
    if cov_type == "full":
        Sigma0 = _pad_kdd(Sig_s, np.ones(K, dtype=bool))
        Sigma_list.append(Sigma0)

    all_domains: List[DomainDataset] = []
    for i in range(1, n_inter + 1):
        _t_step = time.time()
        t = i / (n_inter + 1)
        # class allocation (keeps your current behavior)
        pi_mid = (counts_s.float() / max(1, total_s))
        desired = (pi_mid * float(total_min)).clamp_min(0.0)
        base = torch.floor(desired)
        frac = desired - base
        present = ((counts_s > 0) & (counts_t > 0)).float()
        base = base * present
        frac = frac * present
        n_alloc = int(base.sum().item())
        rem = int(max(0, total_min - n_alloc))

        if rem > 0:
            k_take = min(rem, K)
            if k_take > 0:
                _, idx = torch.topk(frac, k=k_take)
                add = torch.zeros_like(base); add[idx] = 1.0
                base = base + add
        n_per_class = base.long()

        mu_mid_full  = np.full((K, d), np.nan, dtype=np.float64)
        var_mid_full = np.full((K, d), np.nan, dtype=np.float64)
        Sig_mid_full = np.full((K, d, d), np.nan, dtype=np.float64) if cov_type == "full" else None
        Zm_list: List[torch.Tensor] = []
        Ym_list: List[torch.Tensor] = []
        n_per_class = base.long().cpu().numpy().astype(np.int64)

        for k_idx in range(K):
            if counts_s[k_idx].item() == 0 or counts_t[k_idx].item() == 0:
                continue
            mu_s_k = mus_s[k_idx]
            var_s_k = vars_s[k_idx]
            mu_t_k = mus_t[k_idx]
            var_t_k = vars_t[k_idx]
            # n_k = int(max(8, min(counts_s[k_idx].item(), counts_t[k_idx].item())))
            n_k = int(counts_s[k_idx].item())
            if n_k <= 0:
                continue
            if cov_type == "full":
                Sig_s_k = Sig_s[k_idx]
                Sig_t_k = Sig_t[k_idx]
                mu_mid, Sig_mid = natural_interp_full(mu_s_k, Sig_s_k, mu_t_k, Sig_t_k, t, eps=1e-8)
                if not np.isfinite(Sig_mid).all():
                    Sig_mid = _to_spd(np.diag(np.maximum(np.diag(Sig_s_k), 1e-8)))
                else:
                    Sig_mid = _to_spd(Sig_mid)
                mu_mid_full[k_idx]  = mu_mid
                var_mid_full[k_idx] = np.clip(_diag_from_full(Sig_mid), 1e-12, None)
                Sig_mid_full[k_idx] = Sig_mid
                Zk = sample_full(mu_mid, Sig_mid, n_k)
                # sample
                # w, V = np.linalg.eigh(Sig_mid)
                # Z = np.random.standard_normal(size=(n_k, d))
                # A = (V * np.sqrt(np.clip(w, 1e-12, None))) @ V.T
                # Zk = torch.from_numpy(Z @ A.T + mu_mid).to(device).float()

            else:
                mu_mid, var_mid = natural_interp_diag(mu_s_k, var_s_k, mu_t_k, var_t_k, t)
                Zk = sample_diag(mu_mid, var_mid, n_k)
            Yk = torch.full((n_k,), k_idx, device=device, dtype=torch.long)
            Zm_list.append(Zk)
            Ym_list.append(Yk)


        if not Zm_list:
            continue
        Zm = torch.cat(Zm_list, 0).cpu().float()
        Ym = torch.cat(Ym_list, 0).cpu().long()
        weights = torch.ones(len(Ym))
        all_domains.append(DomainDataset(Zm, weights, Ym, Ym))
        print(f"[NATURAL] Step {i}/{n_inter}: generated {len(Ym)} samples with d={Zm.shape[1]} in {time.time()-_t_step:.2f}s")
        steps.append(float(t))
        mu_list.append(mu_mid_full)
        var_list.append(var_mid_full)
        counts_list.append(n_per_class.astype(np.int64))
        pi_list.append(_to_np(pi_mid))
        if cov_type == "full":
            Sigma_list.append(Sig_mid_full)
    try:
        X_tgt_final = xt if torch.is_tensor(xt) else torch.as_tensor(xt)
        X_tgt_final = X_tgt_final.cpu()
        Y_tgt_final = dataset_t.targets if torch.is_tensor(dataset_t.targets) else torch.as_tensor(dataset_t.targets, dtype=torch.long)
        Y_tgt_final = Y_tgt_final.cpu().long()
        Y_em_final = Yt_em.cpu().long()
        W_tgt_final = torch.ones(len(Y_em_final))
        all_domains.append(DomainDataset(X_tgt_final, W_tgt_final, Y_tgt_final, Y_em_final))
    except Exception as e:
        print(f"[NATURAL] Warning: failed to wrap target with targets_em ({e}); appending raw dataset.")
        all_domains.append(dataset_t)
    pi_t = (counts_t.float() / max(1, total_t)).cpu().numpy()
    muT, varT = _pad_kd(mus_t, vars_t, np.ones(K, dtype=bool))
    steps.append(1.0); mu_list.append(muT); var_list.append(varT)
    counts_list.append(counts_t.cpu().numpy().astype(np.int64)); pi_list.append(pi_t)
    if cov_type == "full":
        SigT = _pad_kdd(Sig_t, np.ones(K, dtype=bool))
        Sigma_list.append(SigT)

    print(f"[NATURAL] Total generation time: {time.time()-_t_total:.2f}s")
    generated_size = len(all_domains[-2].data) if len(all_domains) > 1 else 0
    print(f"Total data for each intermediate domain: {generated_size}")

    # ---------- pack domain params ----------
    domain_params = {
        "K": int(K),
        "d": int(d),
        "cov_type": cov_type,
        "steps": np.asarray(steps, dtype=np.float64),            # (S,)
        "mu":    np.asarray(mu_list,  dtype=np.float64),         # (S, K, d)
        "var":   np.asarray(var_list, dtype=np.float64),         # (S, K, d)  (always provided)
        "counts": np.asarray(counts_list, dtype=np.int64),       # (S, K)
        "pi":      np.asarray(pi_list,    dtype=np.float64),     # (S, K)
        "present_source": present_s.astype(np.bool_),
        "present_target": present_t.astype(np.bool_),
    }

    if cov_type == "full":
        domain_params["Sigma"] = np.asarray(Sigma_list, dtype=np.float64)  # (S, K, d, d)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez_compressed(save_path, **domain_params)
        print(f"[FR] Saved domain parameters -> {save_path}")

    return all_domains, Yt_em.cpu().numpy(), domain_params


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

def run_em_on_encoded(
    enc_ds,
    K=10,
    cov_type='diag',          # 'diag' | 'spherical' | 'full'
    pool='gap',               # 'gap' or 'flatten'
    do_pca=False,
    pca_dim=64,
    reg=1e-6,
    max_iter=100,
    tol=1e-5,
    freeze_sigma_iters=0,     # unused (API compat)
    rng=np.random.default_rng(0),
    *,
    scaler: Optional[StandardScaler] = None,
    pca: Optional[PCA] = None,
    return_transforms: bool = False,
    subsample_init: Optional[int] = 20000,
    warm_start: Optional[Dict[str, dict]] = None,
    verbose: bool = False,
):
    if not do_pca:
        pca_dim =0
    # 1) make features (N,d)
    X = to_features_from_encoded(enc_ds, pool=pool)
    if X.dtype != np.float64:
        X = X.astype(np.float64, copy=False)
    if not X.flags.c_contiguous:
        X = np.ascontiguousarray(X)
    if verbose:
        print("Features:", X.shape)

    # 2) standardize
    if scaler is None:
        _scaler = StandardScaler(with_mean=True, with_std=True)
        X = _scaler.fit_transform(X)
    else:
        _scaler = scaler
        X = _scaler.transform(X)

    # 3) optional PCA for stability/speed (auto-enable for 'full' in high dim)
    _pca = None
    auto_pca = (cov_type == 'full' and not do_pca and X.shape[1] > pca_dim)
    # use_pca = do_pca or auto_pca
    use_pca = do_pca 
    if use_pca and X.shape[1] > pca_dim:
        if pca is None:
            _pca = PCA(n_components=pca_dim, svd_solver='randomized', random_state=int(rng.integers(2**31-1)))
            X = _pca.fit_transform(X)
            if verbose:
                print(f"PCA -> {X.shape} (auto={auto_pca})")
        else:
            _pca = pca
            X = _pca.transform(X)

    # Use stronger reg for full covariance early on
    reg_used = max(reg, 1e-4) if cov_type == 'full' else reg

    # 4) k-means++ init
    n = X.shape[0]
    if subsample_init is not None and n > subsample_init:
        idx = rng.choice(n, size=subsample_init, replace=False)
        X_init = X[idx]
    else:
        X_init = X
    mu0, Sigma0, pi0 = kmeanspp_init_params(X_init, K=K, cov_type=cov_type, reg=reg_used, rng=rng)

    if warm_start is not None:
        mu0 = warm_start.get('mu', mu0)
        Sigma0 = warm_start.get('Sigma', Sigma0)
        pi0 = warm_start.get('pi', pi0)

    # 5) EM
    # mu, Sigma, pi, gamma, ll_curve = em_k_gaussians(
    #     X,
    #     mu_init=mu0, Sigma_init=Sigma0, pi_init=pi0,
    #     cov_type="diag", max_iter=max_iter, tol=tol, reg=reg, verbose=True
    # )   
    
    # strengthen regularization for 'full' covariance for numerical stability
    reg_used = max(reg, 1e-4) if cov_type == 'full' else reg

    mu, Sigma, pi, gamma, ll_curve = em_k_gaussians_sklearn(
        X,
        mu_init=mu0, Sigma_init=Sigma0, pi_init=pi0,
        cov_type=cov_type, max_iter=max_iter, tol=tol, reg=reg_used, verbose=verbose
    )

    

    # responsibilities → hard labels if you want
    z = gamma.argmax(axis=1)
    out = {
        "X": X,
        "mu": mu,
        "Sigma": Sigma,
        "pi": pi,
        "gamma": gamma,
        "labels": z,
        "ll_curve": ll_curve,
    }
    if return_transforms:
        out["scaler"] = _scaler
        out["pca"] = _pca
    return out



def gaussian_e_geodesic(mu_s: np.ndarray, Sigma_s: np.ndarray,
                        mu_t: np.ndarray, Sigma_t: np.ndarray,
                        t: float) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate class-conditional Gaussian via e-geodesic (linear in natural params)."""
    Js = np.linalg.inv(Sigma_s); Jt = np.linalg.inv(Sigma_t)
    eta1_s, eta2_s = Js @ mu_s, -0.5 * Js
    eta1_t, eta2_t = Jt @ mu_t, -0.5 * Jt
    eta1 = (1-t) * eta1_s + t * eta1_t
    eta2 = (1-t) * eta2_s + t * eta2_t
    J_interp = -2.0 * eta2
    Sigma_interp = np.linalg.inv(J_interp)
    mu_interp = Sigma_interp @ eta1
    return mu_interp, Sigma_interp
def sample_dataset(mu: np.ndarray, Sigma: np.ndarray,
                      n: int, rng):
    """Draw n samples from a 2-class Gaussian with shared covariance Sigma (d×d)."""
    d = len(mu)
    X = rng.multivariate_normal(mean=mu, cov=Sigma, size=n) if n else np.empty((0, d))
    return X


# ---------- small SPD helpers ----------
def _sym(A):
    return 0.5 * (A + A.T)




def _spd_logm(A, eps=1e-12):
    w, V = _spd_eigh(A, eps)
    return (V * np.log(w)) @ V.T

def _spd_inv(A, eps=1e-12):
    w, V = _spd_eigh(A, eps)
    return (V * (1.0/w)) @ V.T

def _spd_geometric_mean(A, B, eps=1e-12):
    """
    Affine-invariant geometric mean:  A # B = A^{1/2} (A^{-1/2} B A^{-1/2})^{1/2} A^{1/2}
    """
    Ah = _spd_sqrt(A, eps)
    Aih = _spd_invsqrt(A, eps)
    mid = _spd_sqrt(Aih @ B @ Aih, eps)
    return Ah @ mid @ Ah



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

def _diag_to_full(S_diag):
    # (K,d) -> (K,d,d)
    return np.stack([np.diag(s) for s in np.asarray(S_diag)], axis=0)

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


# ---------- 2-Wasserstein (Bures) distance ----------
def wasserstein2_gaussian(mu1, Sigma1, mu2, Sigma2, squared=False, eps=1e-12):
    """
    W2 distance between Gaussians:
    W2^2 = ||mu1 - mu2||^2 + Tr(Sigma1 + Sigma2 - 2 * (Sigma1^{1/2} Sigma2 Sigma1^{1/2})^{1/2})
    """
    mu1 = np.atleast_1d(mu1).astype(np.float64)
    mu2 = np.atleast_1d(mu2).astype(np.float64)
    Sigma1 = np.atleast_2d(Sigma1).astype(np.float64)
    Sigma2 = np.atleast_2d(Sigma2).astype(np.float64)

    dmu2 = float(np.dot(mu1 - mu2, mu1 - mu2))
    S1h = _spd_sqrt(Sigma1, eps)
    mid = S1h @ Sigma2 @ S1h
    mid_h = _spd_sqrt(mid, eps)
    trace_term = float(np.trace(Sigma1 + Sigma2 - 2.0 * mid_h))
    w2_sq = dmu2 + trace_term
    return w2_sq if squared else np.sqrt(max(w2_sq, 0.0))


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


def _compute_source_gaussians(src_enc, pool='gap', reg=1e-6):
    """
    Returns:
      mu_src: (C,d)
      Sig_src: (C,d,d)
    """
    X = _pool_features(src_enc.data, pool=pool).astype(float)
    y = np.asarray(src_enc.targets, dtype=int)
    C = y.max() + 1
    d = X.shape[1]
    mu = np.zeros((C, d), dtype=float)
    Sig = np.zeros((C, d, d), dtype=float)
    for c in range(C):
        mk = (y == c)
        if not mk.any():
            # empty class: leave zeros; matcher will penalize via distances
            Sig[c] = np.eye(d) * reg
            continue
        Xc = X[mk]
        mu[c] = Xc.mean(axis=0)
        Z = Xc - mu[c]
        if len(Xc) > 1:
            Sig[c] = (Z.T @ Z) / (len(Xc) - 1) + reg * np.eye(d)
        else:
            Sig[c] = np.eye(d) * reg
    return mu, Sig


def _match_with_pseudolabels(cluster_ids, pseudo_labels, n_classes):
    """Hungarian match that maximizes overlap counts between clusters and pseudo-labels."""
    cluster_ids = np.asarray(cluster_ids, dtype=int)
    pseudo_labels = np.asarray(pseudo_labels, dtype=int)
    K = int(cluster_ids.max()) + 1

    C = np.zeros((K, n_classes), dtype=np.int64)  # contingency
    for k in range(K):
        mk = (cluster_ids == k)
        if mk.any():
            C[k] = np.bincount(pseudo_labels[mk], minlength=n_classes)

    # maximize matches => minimize (max - C)
    cost = C.max() - C
    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = {int(r): int(c) for r, c in zip(row_ind, col_ind)}
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

def _match_by_prototypes_metric(mu_clusters_in,
                                Sigma_clusters_in,
                                mus_s_in, Sigma_s_in,
                                metric="euclidean"):
    """
    Inputs can be dicts {k: (d,) or (d,d)} or stacked arrays.
    Returns: mapping {cluster_key -> class_key}, cost matrix D (K×C).
    """
    # ---- normalize to arrays + keep original key orders ----
    mu_clusters, cluster_keys = _stack_means(mu_clusters_in)       # (K,d)
    mus_s,       class_keys   = _stack_means(mus_s_in)             # (C,d)

    Sigma_clusters, kindA = _stack_sigmas(Sigma_clusters_in)       # (K,d) or (K,d,d)
    Sigma_s,       kindB  = _stack_sigmas(Sigma_s_in)              # (C,d) or (C,d,d)

    K, d = mu_clusters.shape
    C, d2 = mus_s.shape
    # breakpoint()
    assert d == d2, f"d mismatch: {d} vs {d2}"

    # helpers to get per-(i,j) covs in the right form for each metric
    def get_pair_cov(i, j, need_full: bool):
        # cluster
        Sig_k = Sigma_clusters[i]
        if kindA == "diag" and need_full:
            Sig_k = np.diag(Sig_k)
        # class
        Sig_c = Sigma_s[j]
        if kindB == "diag" and need_full:
            Sig_c = np.diag(Sig_c)
        return Sig_k, Sig_c

    D = np.zeros((K, C), dtype=float)

    for i in range(K):
        mu_k = mu_clusters[i]
        for j in range(C):
            mu_c = mus_s[j]

            if metric.lower() == "euclidean":
                D[i, j] = float(np.sum((mu_k - mu_c) ** 2))

            elif metric.lower() == "w2":
                # W2 requires full SPD inputs
                Sig_k, Sig_c = get_pair_cov(i, j, need_full=True)
                D[i, j] = wasserstein2_gaussian(mu_k, Sig_k, mu_c, Sig_c, squared=False)

            elif metric.lower() == "fr":
                if kindA == "diag" and kindB == "diag":
                    D[i, j] = fisher_rao_gaussian_diag(mu_clusters[i], Sigma_clusters[i],
                                                       mus_s[j],       Sigma_s[j])
                else:
                    Sig_k, Sig_c = get_pair_cov(i, j, need_full=True)
                    D[i, j] = fisher_rao_gaussian(mu_k, Sig_k, mu_c, Sig_c)

            elif metric.lower() == "eta":
                Sig_k, Sig_c = get_pair_cov(i, j, need_full=True)  # if η needs full
                D[i, j] = eta_distance_gaussian(mu_k, Sig_k, mu_c, Sig_c)

            elif metric.lower() in {"kl", "symkl"}:
                # KL between Gaussians; use symmetrized KL if requested.
                # Convert to full SPD for numerical stability.
                Sig_k, Sig_c = get_pair_cov(i, j, need_full=True)
                d_eps = 1e-8
                S0 = np.asarray(Sig_k, dtype=float)
                S1 = np.asarray(Sig_c, dtype=float)
                S0 = 0.5 * (S0 + S0.T) + d_eps * np.eye(d)
                S1 = 0.5 * (S1 + S1.T) + d_eps * np.eye(d)
                invS1 = np.linalg.inv(S1)
                diff = (mu_c - mu_k).reshape(-1, 1)
                tr_term = float(np.trace(invS1 @ S0))
                quad = float((diff.T @ invS1 @ diff).squeeze())
                sign0, logdet0 = np.linalg.slogdet(S0)
                sign1, logdet1 = np.linalg.slogdet(S1)
                kl_kc = 0.5 * (tr_term + quad - d + (logdet1 - logdet0))
                if metric.lower() == "kl":
                    D[i, j] = kl_kc
                else:
                    invS0 = np.linalg.inv(S0)
                    diff2 = (mu_k - mu_c).reshape(-1, 1)
                    tr_term2 = float(np.trace(invS0 @ S1))
                    quad2 = float((diff2.T @ invS0 @ diff2).squeeze())
                    _, logdet0b = sign0, logdet0
                    # recompute slogdet for S0,S1 already available
                    kl_ck = 0.5 * (tr_term2 + quad2 - d + (logdet0 - logdet1))
                    D[i, j] = 0.5 * (kl_kc + kl_ck)

            else:
                raise ValueError(f"Unknown metric '{metric}'.")

    # Hungarian on costs
    row_ind, col_ind = linear_sum_assignment(D)
    mapping = {int(cluster_keys[i]): int(class_keys[j]) for i, j in zip(row_ind, col_ind)}
    return mapping, D


def map_em_clusters(
    res,
    method: str = "pseudo",
    n_classes: int = None,
    pseudo_labels: np.ndarray = None,
    src_enc=None,
    pool: str = "gap",
    metric: str = "euclidean",
    mus_s=None,
    Sigma_s=None,
    priors_s=None,  # unused but kept for compatibility
):
    method = method.lower()
    cluster_ids = np.asarray(res["labels"], dtype=int)

    if method == "pseudo":
        if pseudo_labels is None:
            raise ValueError("pseudo_labels must be provided for method='pseudolabels'.")

        pseudo_labels = np.asarray(pseudo_labels, dtype=int)
        if n_classes is None:
            n_classes = int(pseudo_labels.max()) + 1
        mapping_dict, contingency = _match_with_pseudolabels(cluster_ids, pseudo_labels, n_classes)

        # fast mapping using an array lookup
        max_id = max(cluster_ids.max(), max(mapping_dict.keys(), default=-1))
        mapping_array = np.arange(max_id + 1, dtype=int)
        for cid, cls in mapping_dict.items():
            mapping_array[cid] = cls

        labels_mapped = mapping_array[cluster_ids]

        return mapping_dict, labels_mapped, contingency

    if method == "prototypes":
        # breakpoint()
        if mus_s is None:
            if src_enc is None:
                raise ValueError("Provide either mus_s (+ optional Sigma_s) or src_enc.")
            mus_s = _compute_source_prototypes(src_enc, pool=pool)
            Sigma_s = None

        if "mu" not in res:
            raise ValueError("res['mu'] (cluster mean dict) is required for method='prototypes'.")

        mu_clusters = res["mu"]
        Sigma_clusters = res.get("Sigma", None)

        mapping_dict, cost = _match_by_prototypes_metric(mu_clusters, Sigma_clusters, mus_s, Sigma_s, metric=metric)

        max_id = max(cluster_ids.max(), max(mapping_dict.keys(), default=-1))
        mapping_array = np.arange(max_id + 1, dtype=int)
        for cid, cls in mapping_dict.items():
            mapping_array[cid] = cls

        labels_mapped = mapping_array[cluster_ids]

        return mapping_dict, labels_mapped, cost

    raise ValueError("method must be one of {'pseudolabels','prototypes'}")


def em_soft_targets_from_mapping(gamma: np.ndarray, mapping: Dict[int, int], n_classes: int) -> np.ndarray:
    """
    Convert EM responsibilities over clusters (N×K_clusters) into soft class targets (N×C)
    using a cluster->class mapping.

    Args:
      gamma   : (N, K) responsibilities for each cluster
      mapping : {cluster_id -> class_id}
      n_classes: total number of classes C

    Returns:
      em_soft : (N, C) row-normalized soft targets in class space
    """
    G = np.asarray(gamma, dtype=float)
    N, _K = G.shape
    C = int(n_classes)
    T = np.zeros((N, C), dtype=float)
    for k, c in mapping.items():
        if 0 <= k < G.shape[1] and 0 <= c < C:
            T[:, c] += G[:, k]
    row_sum = T.sum(axis=1, keepdims=True)
    # safe normalize
    T = np.divide(T, np.maximum(row_sum, 1e-12), out=T, where=row_sum > 0)
    return T


def _spd_eigh(A, eps=1e-12):
    """Symmetric eigendecomp with small ridge; returns eigenvalues (clipped) and eigenvectors."""
    A = 0.5 * (A + A.T)
    w, V = np.linalg.eigh(A)
    w = np.clip(w, eps, None)
    return w, V

def _spd_sqrt(A, eps=1e-12):
    w, V = _spd_eigh(A, eps)
    return (V * np.sqrt(w)) @ V.T

def _spd_invsqrt(A, eps=1e-12):
    w, V = _spd_eigh(A, eps)
    return (V * (1.0 / np.sqrt(w))) @ V.T

def _bures_geodesic(S0, S1, t, eps=1e-12):
    """
    2-Wasserstein/Bures covariance geodesic:
      S_t = ((1-t)I + t Γ) S0 ((1-t)I + t Γ)^T,
      Γ = S0^{-1/2} (S0^{1/2} S1 S0^{1/2})^{1/2} S0^{-1/2}
    """
    S0 = 0.5 * (S0 + S0.T)
    S1 = 0.5 * (S1 + S1.T)
    S0h  = _spd_sqrt(S0, eps)
    S0ih = _spd_invsqrt(S0, eps)
    inner = S0h @ S1 @ S0h
    inner_h = _spd_sqrt(inner, eps)
    Gamma = S0ih @ inner_h @ S0ih
    A = (1.0 - t) * np.eye(S0.shape[0]) + t * Gamma
    St = A @ S0 @ A.T
    return 0.5 * (St + St.T)

def _airm_geodesic(S0, S1, t, eps=1e-12):
    """
    Affine-invariant (Rao/AIRM) covariance geodesic:
      S_t = S0^{1/2} ( S0^{-1/2} S1 S0^{-1/2} )^t S0^{1/2}
    """
    S0 = 0.5 * (S0 + S0.T)
    S1 = 0.5 * (S1 + S1.T)
    S0h  = _spd_sqrt(S0, eps)
    S0ih = _spd_invsqrt(S0, eps)
    M = S0ih @ S1 @ S0ih
    # M^t via eigendecomp
    w, V = _spd_eigh(M, eps)
    Mt = (V * (w**t)) @ V.T
    St = S0h @ Mt @ S0h
    return 0.5 * (St + St.T)

def _eta_geodesic(mu0, S0, mu1, S1, t, eps=1e-12):
    """
    Exponential-family (natural-param) straight line:
      η1 = Σ^{-1} μ,   η2 = -1/2 Σ^{-1}
      η(t) = (1-t)η(0) + tη(1), then invert back to (μ_t, Σ_t).
    """
    S0 = 0.5 * (S0 + S0.T)
    S1 = 0.5 * (S1 + S1.T)
    S0inv = np.linalg.inv(S0 + eps*np.eye(S0.shape[0]))
    S1inv = np.linalg.inv(S1 + eps*np.eye(S1.shape[0]))
    eta10, eta20 = S0inv @ mu0, -0.5 * S0inv
    eta11, eta21 = S1inv @ mu1, -0.5 * S1inv
    eta1 = (1.0 - t) * eta10 + t * eta11
    eta2 = (1.0 - t) * eta20 + t * eta21
    J = -2.0 * eta2                  # J = Σ^{-1}
    # ensure SPD (small ridge)
    J = 0.5 * (J + J.T)
    w, V = np.linalg.eigh(J + eps*np.eye(J.shape[0]))
    w = np.clip(w, eps, None)
    Sinv = (V * w) @ V.T
    S = np.linalg.inv(Sinv)
    mu = S @ eta1
    return mu, 0.5 * (S + S.T)

def build_grid_mu_Sigma(mu0, Sigma0, mu1, Sigma1,
                        n_mu: int, n_sigma: int,
                        metric: str = "W2",
                        eps: float = 1e-12):
    """
    Multivariate generalization of build_grid_mu_lambda.

    Args:
      mu0, mu1     : (d,)         endpoints for the mean
      Sigma0, Sigma1 : (d,d)       SPD endpoints for covariance
      n_mu, n_sigma: ints          grid lengths for mean and covariance
      metric       : 'W2' | 'FR' | 'eta'
        - 'W2'  : μ_t = (1-t)μ0 + tμ1; Σ_t via Bures–Wasserstein geodesic
        - 'FR'  : μ_t = (1-t)μ0 + tμ1; Σ_t via affine-invariant (Rao/AIRM) geodesic
        - 'eta' : exponential-family line in natural params (μ and Σ *coupled*).
                  Here we still return two independent grids by sampling t for μ and Σ.
    Returns:
      mus    : (n_mu, d)
      Sigmas : (n_sigma, d, d)
    """
    mu0 = np.asarray(mu0, dtype=float)
    mu1 = np.asarray(mu1, dtype=float)
    Sigma0 = np.asarray(Sigma0, dtype=float)
    Sigma1 = np.asarray(Sigma1, dtype=float)

    t_mu = np.linspace(0.0, 1.0, n_mu)
    t_s  = np.linspace(0.0, 1.0, n_sigma)

    # --- mean path
    if metric.lower() in {"w2", "fr"}:
        mus = np.stack([(1.0 - t) * mu0 + t * mu1 for t in t_mu], axis=0)
    elif metric.lower() == "eta":
        # couple μ and Σ along the same construction but sampled on t_mu
        mus = []
        for t in t_mu:
            mu_t, _ = _eta_geodesic(mu0, Sigma0, mu1, Sigma1, t, eps)
            mus.append(mu_t)
        mus = np.stack(mus, axis=0)
    else:
        raise ValueError("metric must be one of {'W2','FR','eta'}")

    # --- covariance path
    Sigmas = []
    if metric.lower() == "w2":
        for t in t_s:
            Sigmas.append(_bures_geodesic(Sigma0, Sigma1, t, eps))
    elif metric.lower() == "fr":
        for t in t_s:
            Sigmas.append(_airm_geodesic(Sigma0, Sigma1, t, eps))
    elif metric.lower() == "eta":
        for t in t_s:
            _, S_t = _eta_geodesic(mu0, Sigma0, mu1, Sigma1, t, eps)
            Sigmas.append(S_t)
    Sigmas = np.stack(Sigmas, axis=0)

    return mus, Sigmas


def fit_source_gaussian_params(X, y,  pool: str = "") -> Tuple[Dict[int, np.ndarray], np.ndarray, Dict[int, float]]:
    """Estimate per-class means and shared covariance (pooled) + class priors from labeled source."""
    # classes = np.unique(y)
    # classes = np.unique(y).tolist()
    classes = y.unique().tolist()
    
    mus = {}
    priors = {}
    Sigma_s = {}
    if pool == "gap":
        X = _pool_features(X, pool=pool)
    else:
        X = X.reshape(X.shape[0], -1)
    # breakpoint()
    d = X.shape[1]
    for c in classes:
        Sigma = np.zeros((d, d))
        Xc = X[y == c]
        mus[c] = Xc.mean(axis=0)
        priors[c] = float(len(Xc)) / len(X)
        Z = Xc - mus[c]
        Sigma_s[c] = Sigma
    return mus, Sigma_s, priors






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


def fr_distance_diag(muA, varA, muB, varB, eps=1e-12):
    """
    Rough class-wise FR distance for diagonal Gaussians:
    d^2 ≈ ||μA-μB||^2_{Σ^{-1}} + ||log σ_A^2 - log σ_B^2||^2  (summed over dims)
    Here we use σ^{-2} from A midway (symmetric variants also fine for diagnostics).
    """
    import torch
    varA = varA.clamp_min(eps); varB = varB.clamp_min(eps)
    invA = 1.0 / varA
    term_mu = ((muA - muB)**2 * invA).sum(dim=1)       # per class
    term_sig = (varA.log() - varB.log()).pow(2).sum(dim=1)
    return term_mu + term_sig



def fr_interp_diag(mu_s: torch.Tensor, var_s: torch.Tensor,
                   mu_t: torch.Tensor, var_t: torch.Tensor,
                   t: float, eps: float = 1e-8):
    """
    Fisher–Rao geodesic between two diagonal Gaussians, applied element-wise.
    mu_*: (d,), var_*: (d,) with var_*>0
    Returns (mu_t, var_t) at time t in [0,1].
    """
    # variance path: geometric (AIRM)
    r = (var_t / var_s).clamp_min(eps)   # ratio of variances
    var_mid = var_s * torch.pow(r, t)

    # mean path: coupled with variance ratio
    # handle r≈1 stably: (r^t - 1)/(r - 1) -> t when r→1
    num = torch.pow(r, t) - 1.0
    den = r - 1.0
    w = torch.where(den.abs() < 1e-6, torch.full_like(den, t), num / den)
    mu_mid = mu_s + w * (mu_t - mu_s)
    return mu_mid, var_mid

def sample_diag_gauss(mu: torch.Tensor, var: torch.Tensor, n: int, device=None):
    """
    Sample n vectors from N(mu, diag(var)).
    mu,var: (d,)
    """
    if device is None: device = mu.device
    d = mu.numel()
    std = (var.clamp_min(1e-8)).sqrt()
    z = torch.randn(n, d, device=device) * std + mu
    return z
