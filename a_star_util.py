import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from typing import Tuple, Dict, List, Optional, Union

import torch
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
# If you kept the generalized EM/K-means++ from earlier:
# from unsup_exp import gmm_em_k, kmeanspp_init  # K-class EM + kmeans++ init
# (names below assume the ones I provided earlier; adjust to your file)

import time
from dataset import *
import torch.nn as nn
from em_utils import *
def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)
@torch.no_grad()
def check_natural_linearity_full(domain_stats, eps=1e-6, steps=None, verbose=True):
    """
    Verifies e-geodesic (natural) linearity for FULL covariance:
      η2(t) == (1-t)η2_s + t η2_T,  where η2 = -1/2 Λ, Λ=Σ^{-1}
      η1(t) == (1-t)η1_s + t η1_T,  where η1 = Λ μ
    Uses the SAME ε as the generator when forming Λ.
    """


    def _to_t(x):  # numpy -> torch (float64 for stability)
        return torch.from_numpy(np.asarray(x)).to(dtype=torch.float64)

    assert "Sigma" in domain_stats and domain_stats["Sigma"] is not None, \
        "Need full covariances in domain_stats['Sigma'] for this check."

    S = len(domain_stats["steps"])
    K = int(domain_stats["K"])
    d = int(domain_stats["d"])

    if steps is None:
        steps = list(range(S))
    steps = list(steps)

    # endpoints
    Sig_s = _to_t(domain_stats["Sigma"][0])      # (K,d,d)
    Sig_T = _to_t(domain_stats["Sigma"][-1])     # (K,d,d)
    mu_s  = _to_t(domain_stats["mu"][0])         # (K,d)
    mu_T  = _to_t(domain_stats["mu"][-1])        # (K,d)

    I = torch.eye(d, dtype=torch.float64)

    def prec(S):
        S = 0.5*(S + S.transpose(-1,-2)) + eps*I
        L = torch.linalg.cholesky(S)
        return torch.cholesky_inverse(L)  # symmetric Λ

    # endpoint natural params (with SAME eps)
    Λ_s = prec(Sig_s)
    Λ_T = prec(Sig_T)
    η2_s = -0.5 * Λ_s
    η2_T = -0.5 * Λ_T
    η1_s = Λ_s @ mu_s.unsqueeze(-1)                      # (K,d,1)
    η1_T = Λ_T @ mu_T.unsqueeze(-1)

    bad_pairs = []
    for si in steps[1:-1]:  # only intermediate steps
        t = float(domain_stats["steps"][si])
        Sig_m = _to_t(domain_stats["Sigma"][si])         # (K,d,d)
        mu_m  = _to_t(domain_stats["mu"][si])            # (K,d)

        Λ_m  = prec(Sig_m)
        η2_m = -0.5 * Λ_m
        η1_m = Λ_m @ mu_m.unsqueeze(-1)

        # expected linear combos
        η2_exp = (1.0 - t)*η2_s + t*η2_T
        η1_exp = (1.0 - t)*η1_s + t*η1_T

        # errors per class (Frobenius)
        e2 = torch.linalg.matrix_norm(η2_m - η2_exp, ord='fro', dim=(1,2))
        e1 = torch.linalg.matrix_norm(η1_m - η1_exp, ord='fro', dim=(1,2))

        # scale by norms to get relative errors
        d2 = torch.linalg.matrix_norm(η2_exp, ord='fro', dim=(1,2)).clamp_min(1e-12)
        d1 = torch.linalg.matrix_norm(η1_exp, ord='fro', dim=(1,2)).clamp_min(1e-12)
        rel2 = (e2 / d2).cpu().numpy()
        rel1 = (e1 / d1).cpu().numpy()

        # mark large deviations
        thr = 1e-6  # e-geodesic should hit machine-precision if all steps use same eps
        bad = np.where((rel1 > thr) | (rel2 > thr))[0].tolist()
        if bad:
            bad_pairs.append((si, t, bad, rel1[bad], rel2[bad]))

    if verbose:
        if not bad_pairs:
            print("[NAT-CHK/FULL] All intermediate steps satisfy natural linearity (within tolerance).")
        else:
            total = sum(len(b[2]) for b in bad_pairs)
            print(f"[NAT-CHK/FULL] {total} class–step mismatches. Examples:")
            for si, t, ks, r1, r2 in bad_pairs[:3]:
                desc = ", ".join([f"k={k} rel|η1|={r1[i]:.2e} rel|η2|={r2[i]:.2e}" for i,k in enumerate(ks[:5])])
                print(f"  step {si} (t={t:.2f}): {desc}")


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


def _safe_slogdet_sym(S: np.ndarray, eps: float = 1e-12) -> float:
    # robust logdet for a symmetric matrix
    Ssym = 0.5*(S + S.T)
    w = np.linalg.eigvalsh(Ssym)
    w = np.clip(w, eps, None)
    return float(np.log(w).sum())

def _chol_prec_np(S: np.ndarray, eps: float = 0.0) -> np.ndarray:
    # precision via Cholesky with optional jitter
    Ssym = 0.5*(S + S.T) + eps*np.eye(S.shape[0], dtype=S.dtype)
    L = np.linalg.cholesky(Ssym)
    Linv = np.linalg.inv(L)
    return Linv.T @ Linv  # (Ssym)^{-1}

def _fro(A: np.ndarray) -> float:
    A = np.asarray(A)
    if A.ndim <= 1:
        return float(np.linalg.norm(A))          # 2-norm for vectors
    else:
        return float(np.linalg.norm(A, ord='fro'))  # Frobenius for matrices


def _rel(a: float, b: float, eps: float = 1e-12) -> float:
    return a / (abs(b) + eps)


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

@torch.no_grad()
# def fr_interp_full(mu_s: torch.Tensor,
#                          Sig_s: torch.Tensor,
#                          mu_t: torch.Tensor,
#                          Sig_t: torch.Tensor,
#                          t: float,
#                          eps: float = 1e-8):
#     """
#     Fisher–Rao geodesic between two Gaussians with full covariance.
#     Returns (mu_mid, Sig_mid) as torch.Tensors on the same device as inputs.
#     Assumes no gradients needed (no_grad).
#     """
#     device = Sig_s.device
#     dtype  = torch.float64  # for stable eigendecompositions
#     mu_s = mu_s.to(dtype=dtype, device=device)
#     mu_t = mu_t.to(dtype=dtype, device=device)
#     Sig_s = 0.5 * (Sig_s.to(dtype=dtype, device=device) + Sig_s.mH)  # symmetrize
#     Sig_t = 0.5 * (Sig_t.to(dtype=dtype, device=device) + Sig_t.mH)

#     # Σ_s^{±1/2} via eigen
#     ws, Vs = torch.linalg.eigh(Sig_s)
#     ws = torch.clamp(ws, min=eps)
#     Sig_s_half     = (Vs * torch.sqrt(ws)) @ Vs.mH
#     Sig_s_inv_half = (Vs * (1.0 / torch.sqrt(ws))) @ Vs.mH

#     # A = Σ_s^{-1/2} Σ_t Σ_s^{-1/2}
#     A = Sig_s_inv_half @ Sig_t @ Sig_s_inv_half
#     A = 0.5 * (A + A.mH)
#     wa, Va = torch.linalg.eigh(A)
#     wa = torch.clamp(wa, min=eps)
#     A_t = (Va * (wa**t)) @ Va.mH

#     Sig_mid = Sig_s_half @ A_t @ Sig_s_half
#     Sig_mid = 0.5 * (Sig_mid + Sig_mid.mH)

#     # mean path (simple Euclidean interpolation; change if you use a different rule)
#     mu_mid = (1.0 - t) * mu_s + t * mu_t
#     return mu_mid, Sig_mid

@torch.no_grad()
def fr_interp_full(mu_s: torch.Tensor,
                   Sig_s: torch.Tensor,
                   mu_t: torch.Tensor,
                   Sig_t: torch.Tensor,
                   t: float,
                   eps: float = 1e-8):
    """
    Fisher–Rao geodesic between two Gaussians with full covariance.
    Returns (mu_mid, Sig_mid). Uses float64 internally for stability.
    """
    device = Sig_s.device
    in_dtype = mu_s.dtype

    # work in float64 for eigens
    mu_s = mu_s.to(dtype=torch.float64, device=device)
    mu_t = mu_t.to(dtype=torch.float64, device=device)
    Sig_s = 0.5 * (Sig_s.to(dtype=torch.float64, device=device) + Sig_s.mH)
    Sig_t = 0.5 * (Sig_t.to(dtype=torch.float64, device=device) + Sig_t.mH)

    # Σ_s^{±1/2} via eigen
    ws, Vs = torch.linalg.eigh(Sig_s)
    ws = torch.clamp(ws, min=eps)
    Sig_s_half     = (Vs * ws.sqrt()) @ Vs.mH
    Sig_s_inv_half = (Vs * (1.0 / ws.sqrt())) @ Vs.mH

    # A = Σ_s^{-1/2} Σ_t Σ_s^{-1/2} = V diag(λ) V^T
    A = Sig_s_inv_half @ Sig_t @ Sig_s_inv_half
    A = 0.5 * (A + A.mH)
    lam, V = torch.linalg.eigh(A)
    lam = torch.clamp(lam, min=eps)

    # --- covariance along AIRM geodesic ---
    A_t = (V * lam.pow(t)) @ V.mH
    Sig_mid = Sig_s_half @ A_t @ Sig_s_half
    Sig_mid = 0.5 * (Sig_mid + Sig_mid.mH)

    # # --- mean along FR geodesic (closed form) ---
    # w(λ,t) = (λ^t - 1)/(λ - 1), with stable fallback w≈t when λ≈1
    num = lam.pow(t) - 1.0
    den = lam - 1.0
    w = torch.where(den.abs() < 1e-6, torch.full_like(den, t), num / den)

    # M(t) = Σ_s^{1/2} V diag(w) V^T Σ_s^{-1/2}
    M = Sig_s_half @ (V * w) @ V.mH @ Sig_s_inv_half
    mu_mid = mu_s + M @ (mu_t - mu_s)
    # mu_mid = (1.0 - t) * mu_s + t * mu_t  # simple Euclidean interp for mean

    # cast outputs back to input dtype
    return mu_mid.to(in_dtype), Sig_mid.to(in_dtype)



def _slogdet(S, eps=1e-12):
    S = 0.5*(S+S.T)
    s, ld = np.linalg.slogdet(S + eps*np.eye(S.shape[0]))
    return float(ld)

def assert_logdet_linearity(domain_params, tol=1e-5):
    if "Sigma" not in domain_params: 
        return  # diag case: check per-dim log-var if you want
    Sig = domain_params["Sigma"]   # (S,K,d,d)
    ts  = np.asarray(domain_params["steps"])
    S, K = Sig.shape[0], Sig.shape[1]
    for k in range(K):
        ld0 = _slogdet(Sig[0,k]); ldT = _slogdet(Sig[-1,k])
        for s in range(1, S-1):
            lhs = _slogdet(Sig[s,k])
            rhs = (1.0 - ts[s]) * ld0 + ts[s] * ldT
            if not np.isfinite(lhs) or not np.isfinite(rhs) or abs(lhs - rhs) > tol*max(1.0, abs(rhs)):
                print(f"[FR-CHK] class {k} t={ts[s]:.2f} logdet mismatch: lhs={lhs:.6f}, rhs={rhs:.6f}, Δ={lhs-rhs:.2e}")

def slogdet(S):
    S = 0.5*(S+S.T)
    sgn, ld = np.linalg.slogdet(S + eps*np.eye(S.shape[0]))
    return float(ld)

def _logdet_per_class_full(SigK: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """SigK: (K,d,d). Return per-class logdet with SPD safety."""
    out = np.full(SigK.shape[0], np.nan, dtype=np.float64)
    for k in range(SigK.shape[0]):
        S = 0.5*(SigK[k] + SigK[k].T)
        sgn, ld = np.linalg.slogdet(S + eps*np.eye(S.shape[0]))
        if sgn <= 0:
            # fall back to eigen-clipped
            w = np.linalg.eigvalsh(S)
            w = np.clip(w, eps, None)
            ld = float(np.sum(np.log(w)))
        out[k] = float(ld)
    return out

def _logdet_per_class_diag(VarK: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """VarK: (K,d). Return per-class sum(log(var))."""
    V = np.asarray(VarK, dtype=np.float64)
    V = np.clip(V, eps, None)
    return np.sum(np.log(V), axis=1)

def _np(a):
    return a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a, dtype=np.float64)
def _to_np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

def _eta_diag_from_mu_var(mu_kd: np.ndarray, var_kd: np.ndarray, eps: float = 1e-12):
    var = np.clip(var_kd, eps, None)
    prec = 1.0 / var
    eta1 = prec * mu_kd            # (K,d)
    eta2 = -0.5 * prec             # (K,d)
    return eta1, eta2

def _eta_full_from_mu_Sig(mu_kd: np.ndarray, Sig_kdd: np.ndarray, eps: float = 1e-12):
    """
    Compute eta1_k = Λ_k μ_k and the DIAGONAL of eta2_k = -1/2 Λ_k for each class.
    Returns:
    eta1_kd      : (K,d)
    eta2_diagnostic_classd : (K,d)  (diag of the matrix -1/2 Λ_k)
    """
    K, d = mu_kd.shape
    eta1 = np.full((K, d), np.nan, dtype=np.float64)
    eta2_diag = np.full((K, d), np.nan, dtype=np.float64)
    for k in range(K):
        S = Sig_kdd[k]
        if not np.isfinite(S).all(): 
            continue
        S = 0.5 * (S + S.T)
        w, V = np.linalg.eigh(S)
        w = np.clip(w, eps, None)
        Lam = (V * (1.0 / w)) @ V.T       # Λ_k
        eta1[k] = Lam @ mu_kd[k]
        eta2_diag[k] = -0.5 * np.diag(Lam)
    return eta1, eta2_diag

def generate_fr_domains_between(
    n_inter,
    dataset_s,
    dataset_t,
    save_path: Optional[str] = None,
    *,
    cov_type: str = "diag",     # NEW: 'diag' or 'full'
    reg: float = 1e-6,          # NEW: covariance regularization (added to eigenvalues)
    ddof: int = 0,              # NEW: 0 → population covariance (matches class_stats_diag)
    jitter: float = 1e-6,       # NEW: SPD safety margin for sampling/inversion
    args=None
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

    Zs = xs if torch.is_tensor(xs) else torch.as_tensor(xs)
    Zs = Zs.to(device)
    Ys = ys if torch.is_tensor(ys) else torch.as_tensor(ys)
    Ys = Ys.to(device, dtype=torch.long)
    if Ys.numel() == 0:
        return [], torch.empty(0, dtype=torch.long)

    Zt = xt if torch.is_tensor(xt) else torch.as_tensor(xt)
    Zt = Zt.to(device)
    if args.em_match != "none":
        Yt_em = dataset_t.targets_em if torch.is_tensor(dataset_t.targets_em) else torch.as_tensor(dataset_t.targets_em)
    else:
        Yt_em = dataset_t.targets_pseudo if torch.is_tensor(dataset_t.targets_pseudo) else torch.as_tensor(dataset_t.targets_pseudo)
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


    # endpoint per-class logdet for FR linearity check
    # if cov_type == "full":
    #     ld0 = _logdet_per_class_full(_np(Sig_s))
    #     ldT = _logdet_per_class_full(_np(Sig_t))
    # else:
    #     ld0 = _logdet_per_class_diag(_np(vars_s))
    #     ldT = _logdet_per_class_diag(_np(vars_t))

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
    eta1_list, eta2d_list = [], []   # NEW
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
    present_both = present_s & present_t
    # ---------- Source (t=0) ----------
    pi_s = (counts_s.float() / max(1, total_s)).cpu().numpy()
    mu0, var0 = _pad_kd(mus_s, vars_s, present_s)
    steps.append(0.0); mu_list.append(mu0); var_list.append(var0)
    counts_list.append(counts_s.cpu().numpy().astype(np.int64)); pi_list.append(pi_s)
    if cov_type == "full":
        Sigma0 = _pad_kdd(Sig_s, present_s)
        Sigma_list.append(Sigma0)
        e1, e2d = _eta_full_from_mu_Sig(mu0, Sigma0)
    else:
        e1, e2d = _eta_diag_from_mu_var(mu0, var0)
    eta1_list.append(e1); eta2d_list.append(e2d)

    all_domains: List[DomainDataset] = []
    def _logdet_spd(S, eps=1e-12):
        """Robust logdet for (d,d) SPD; works for np arrays or torch tensors."""
        if isinstance(S, np.ndarray):
            S = 0.5 * (S + S.T)
            w = np.linalg.eigvalsh(S)
            w = np.clip(w, eps, None)
            return float(np.log(w).sum())
        else:
            S = 0.5 * (S + S.transpose(-2, -1))
            w = torch.linalg.eigvalsh(S).clamp_min(eps)
            return float(torch.log(w).sum().item())

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
                mu_mid_full[k_idx]  = _to_np(mu_mid)
                var_mid_full[k_idx] = _to_np(var_mid)
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

        # NEW: natural params at this step
        if cov_type == "full":
            # Ensure any NaN Sig gets replaced by diag(var) for η as well
            for k_idx in range(K):
                if not np.isfinite(Sig_mid_full[k_idx]).all():
                    v = var_mid_full[k_idx]
                    if np.isfinite(v).all():
                        Sig_mid_full[k_idx] = np.diag(np.clip(v, 1e-12, None))
            Sigma_list.append(Sig_mid_full)
            e1, e2d = _eta_full_from_mu_Sig(mu_mid_full, Sig_mid_full)
        else:
            e1, e2d = _eta_diag_from_mu_var(mu_mid_full, var_mid_full)
        eta1_list.append(e1); eta2d_list.append(e2d)

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
    muT, varT = _pad_kd(mus_t, vars_t, present_t)
    steps.append(1.0); mu_list.append(muT); var_list.append(varT)
    counts_list.append(counts_t.cpu().numpy().astype(np.int64)); pi_list.append(pi_t)
    if cov_type == "full":
        SigT = _pad_kdd(Sig_t, present_t)
        Sigma_list.append(SigT)

        muT_t = torch.as_tensor(muT,  device=device, dtype=mus_s.dtype)   # (K,d)
        SigT_t = torch.as_tensor(SigT, device=device, dtype=mus_s.dtype)   # (K,d,d)
        K, d = muT_t.shape
        I = torch.eye(d, device=device, dtype=mus_s.dtype)

        e1 = np.full((K, d), np.nan, dtype=np.float64)
        e2d = np.full((K, d), np.nan, dtype=np.float64)

        for k in range(K):
            if not present_t[k]:
                continue
            Sigk = 0.5 * (SigT_t[k] + SigT_t[k].T) + jitter * I  # SAME jitter
            L    = torch.linalg.cholesky(Sigk)
            Lam  = torch.cholesky_inverse(L)
            eta1_k  = Lam @ muT_t[k]
            eta2d_k = -0.5 * torch.diag(Lam)

            e1[k]  = eta1_k.detach().cpu().numpy()
            e2d[k] = eta2d_k.detach().cpu().numpy()
    else:
        e1, e2d = _eta_diag_from_mu_var(muT, varT)

    eta1_list.append(e1); eta2d_list.append(e2d)

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
        "eta1":      np.asarray(eta1_list,  dtype=np.float64),   # (S,K,d)
        "eta2_diag": np.asarray(eta2d_list, dtype=np.float64),   # (S,K,d)
    }
    if cov_type == "full":
        domain_params["Sigma"] = np.asarray(Sigma_list, dtype=np.float64)  # (S, K, d, d)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez_compressed(save_path, **domain_params)
        print(f"[FR] Saved domain parameters -> {save_path}")
    # breakpoint()

    # ---------- FR logdet linearity audit (before returning) ----------
    strict_fr_check = False          # set True to raise if badly violated
    rtol, atol = 1e-5, 1e-6          # tolerances for np.isclose
    K = int(domain_params["K"])
    ts = np.asarray(domain_params["steps"], dtype=np.float64)  # (S,)
    S = len(ts)

    def _sym_logdet(Sig, eps=1e-12):
        """Return logdet for one covariance; Sig can be (d,d) array."""
        if Sig.ndim != 2:
            raise ValueError("Sig must be (d,d).")
        Ssym = 0.5*(Sig + Sig.T)
        w = np.linalg.eigvalsh(Ssym)
        w = np.clip(w, eps, None)
        return float(np.log(w).sum())


    has_full = ("Sigma" in domain_params) and (domain_params["Sigma"] is not None)
    present_s = domain_params["present_source"].astype(bool)
    present_t = domain_params["present_target"].astype(bool)
    present_both = present_s & present_t

    # collect per-step per-class logdet
    LD = []  # list of (K,)
    for s in range(S):
        if has_full:
            Sig_s = np.asarray(domain_params["Sigma"][s])  # (K,d,d)
            LD.append(_logdet_per_class_full(Sig_s))
        else:
            var_s = np.asarray(domain_params["var"][s])    # (K,d)
            LD.append(_logdet_per_class_diag(var_s))
    LD = np.stack(LD, axis=0)  # (S,K)

    ld0 = LD[0]    # (K,)
    ldT = LD[-1]   # (K,)

    # check each intermediate step
    bad_total = 0
    for s in range(1, S-1):
        t = ts[s]
        rhs = (1.0 - t) * ld0 + t * ldT                    # (K,)
        lhs = LD[s]                                        # (K,)
        # only classes present at both ends and finite at this step
        mask = present_both & np.isfinite(rhs) & np.isfinite(lhs)
        if not np.any(mask):
            continue
        diff = np.abs(lhs[mask] - rhs[mask])
        ok = np.isclose(lhs[mask], rhs[mask], rtol=rtol, atol=atol)
        n_bad = int((~ok).sum())
        bad_total += n_bad
        if n_bad > 0:
            # show up to 5 worst offenders for this step
            idx = np.where(mask)[0]
            offenders = idx[~ok]
            worst = offenders[np.argsort(-diff[~ok])[:5]]
            summary = ", ".join([f"k={int(k)} Δ={float(abs(lhs[k]-rhs[k])):.3e}"
                                for k in worst])
            print(f"[FR-CHK] t={t:.2f}: {n_bad} class(es) off linear logdet. Worst: {summary}")

    print(f"[FR-CHK] Overall: {bad_total} non-linear class-step(s) out of {(S-2)*int(present_both.sum())} checked.")

    if strict_fr_check and bad_total > 0:
        raise RuntimeError("FR logdet linearity check failed; see [FR-CHK] messages above.")

    return all_domains, Yt_em.cpu().numpy(), domain_params



# def generate_fr_domains_between(
#     n_inter,
#     dataset_s,
#     dataset_t,
#     visualize: bool = False,
#     save_path: Optional[str] = None,
#     *,
#     cov_type: str = "diag",     # 'diag' or 'full'
#     reg: float = 1e-6,          # covariance regularization (added to eigenvalues)
#     ddof: int = 0,              # 0 → population covariance
#     jitter: float = 1e-6,       # SPD safety margin for sampling/inversion
#     args=None
# ):
#     """
#     Generate FR-interpolated domains assuming dataset_t.targets_em (or .targets_pseudo) set.

#     Returns:
#       all_domains   : List[DomainDataset]
#       target_em     : np.ndarray of target EM labels
#       domain_params : dict with per-step parameters (source, intermediates, target)

#     FR on SPD (full-cov): Σ(t) = Σ_s^{1/2} ( Σ_s^{-1/2} Σ_t Σ_s^{-1/2} )^t Σ_s^{1/2}
#     Means follow your existing (diag-case) scalar weight rule.
#     """
#     assert cov_type in {"diag", "full"}, "cov_type must be 'diag' or 'full'"

#     # -------------------- setup & tensors on device --------------------
#     print("------------Generate Intermediate domains (FR)----------")
#     _t_total = time.time()

#     xs = dataset_s.data
#     xt = dataset_t.data
#     ys = dataset_s.targets
#     if len(xs.shape) > 2:
#         flat = nn.Flatten()
#         xs, xt = flat(xs), flat(xt)
#     device = xs.device if torch.is_tensor(xs) else torch.device("cpu")

#     if ys is None:
#         raise ValueError("Source dataset must provide targets for class statistics.")
#     has_em = hasattr(dataset_t, "targets_em") and (dataset_t.targets_em is not None)
#     has_pseudo = hasattr(dataset_t, "targets_pseudo") and (dataset_t.targets_pseudo is not None)
#     if args is None:
#         em_mode = "em"
#     else:
#         em_mode = "em" if getattr(args, "em_match", "em") != "none" else "pseudo"
#     if em_mode == "em" and not has_em:
#         raise ValueError("Target dataset must provide targets_em (EM-derived labels).")
#     if em_mode == "pseudo" and not has_pseudo:
#         raise ValueError("Target dataset must provide targets_pseudo.")

#     Zs = xs if torch.is_tensor(xs) else torch.as_tensor(xs, device=device)
#     Zs = Zs.to(device)
#     Ys = ys if torch.is_tensor(ys) else torch.as_tensor(ys, device=device)
#     Ys = Ys.to(device, dtype=torch.long)
#     if Ys.numel() == 0:
#         return [], torch.empty(0, dtype=torch.long), {}

#     Zt = xt if torch.is_tensor(xt) else torch.as_tensor(xt, device=device)
#     Zt = Zt.to(device)
#     if em_mode == "em":
#         Yt_em = dataset_t.targets_em if torch.is_tensor(dataset_t.targets_em) else torch.as_tensor(dataset_t.targets_em, device=device)
#     else:
#         Yt_em = dataset_t.targets_pseudo if torch.is_tensor(dataset_t.targets_pseudo) else torch.as_tensor(dataset_t.targets_pseudo, device=device)
#     Yt_em = Yt_em.to(device, dtype=torch.long)

#     # -------------------- class stats --------------------
#     K = int(max(Ys.max(), Yt_em.max()).item()) + 1
#     if cov_type == "diag":
#         mus_s, vars_s, _ = class_stats_diag(Zs, Ys, K)                      # (K,d), (K,d)
#         mus_t, vars_t, _ = class_stats_diag(Zt, Yt_em, K)
#         Sig_s = Sig_t = None
#         d = mus_s.shape[1]
#     else:
#         mus_s, Sig_s, _ = class_stats_full(Zs, Ys, K, reg=reg, ddof=ddof)  # (K,d), (K,d,d)
#         mus_t, Sig_t, _ = class_stats_full(Zt, Yt_em, K, reg=reg, ddof=ddof)
#         d = mus_s.shape[1]
#         # also keep diagonals for compatibility/logging
#         vars_s = torch.diagonal(Sig_s, dim1=-2, dim2=-1)                   # (K,d)
#         vars_t = torch.diagonal(Sig_t, dim1=-2, dim2=-1)                   # (K,d)

#     # counts and priors
#     counts_s = torch.bincount(Ys, minlength=K)
#     counts_t = torch.bincount(Yt_em, minlength=K)
#     total_s = int(counts_s.sum().item())
#     total_t = int(counts_t.sum().item())
#     total_min = int(torch.minimum(counts_s, counts_t).sum().item())
#     if total_min <= 0:
#         total_min = max(total_s, total_t)

#     present_s = (counts_s > 0).detach().cpu().numpy()
#     present_t = (counts_t > 0).detach().cpu().numpy()
#     present_both = present_s & present_t

#     # -------------------- caches for FULL covariance FR --------------------
#     # Precompute Σ_s^{±1/2} and eig(A) once per class. Avoids per-step eigendecompositions.
#     if cov_type == "full":
#         eye = torch.eye(d, device=device, dtype=Sig_s.dtype)[None, :, :]
#         Sig_s_spd = 0.5 * (Sig_s + Sig_s.transpose(-1, -2)) + jitter * eye
#         Sig_t_spd = 0.5 * (Sig_t + Sig_t.transpose(-1, -2)) + jitter * eye

#         eval_s, U_s = torch.linalg.eigh(Sig_s_spd)            # (K,d), (K,d,d)
#         eval_s = eval_s.clamp_min(jitter)
#         S_s_half     = U_s @ torch.diag_embed(eval_s.sqrt())  @ U_s.transpose(-1, -2)  # (K,d,d)
#         S_s_inv_half = U_s @ torch.diag_embed(eval_s.rsqrt()) @ U_s.transpose(-1, -2)  # (K,d,d)

#         A = S_s_inv_half @ Sig_t_spd @ S_s_inv_half          # (K,d,d)
#         A = 0.5 * (A + A.transpose(-1, -2)) + jitter * eye
#         eval_A, U_A = torch.linalg.eigh(A)                   # (K,d), (K,d,d)
#         eval_A = eval_A.clamp_min(jitter)
#         log_eval_A = torch.log(eval_A)                       # (K,d)

#         def fr_cov_at_t(k: int, t: float) -> torch.Tensor:
#             """Σ_k(t) via cached U_A/log_eval_A & S_s^{1/2}."""
#             lam_t = torch.exp(t * log_eval_A[k])                             # (d,)
#             mid   = U_A[k] @ torch.diag(lam_t) @ U_A[k].T                    # (d,d)
#             return S_s_half[k] @ mid @ S_s_half[k].T                         # (d,d)

#     # -------------------- containers (Torch first; NumPy only at the end) --------------------
#     steps = [0.0]
#     mu_list  = [mus_s.detach().clone().to(torch.float32)]                   # each: (K,d)
#     var_list = [vars_s.detach().clone().to(torch.float32)]                  # each: (K,d)
#     Sigma_list = [Sig_s.detach().clone().to(torch.float32)] if cov_type == "full" else None
#     counts_list = [counts_s.detach().cpu().numpy().astype(np.int64)]        # keep counts as numpy to match your schema
#     pi_list  = [(counts_s.float() / max(1, total_s)).detach().cpu().numpy()]# (K,)
#     eta1_list, eta2d_list = [], []

#     # natural params at source
#     if cov_type == "full":
#         # batched Σ^{-1} μ and diag(Σ^{-1}) using Cholesky
#         L = torch.linalg.cholesky(Sig_s_spd)                                 # (K,d,d)
#         eta1_src = torch.cholesky_solve(mus_s.unsqueeze(-1), L).squeeze(-1)  # (K,d)
#         Linv = torch.linalg.solve(L, eye)                                    # (K,d,d) ~ L^{-1}
#         diag_inv = (Linv ** 2).sum(dim=-1)                                   # (K,d) since Σ^{-1} = L^{-T} L^{-1}
#         eta2_src = -0.5 * diag_inv
#     else:
#         # diag case: Σ^{-1} μ and diag easily
#         inv_var = (vars_s + jitter).reciprocal()
#         eta1_src = inv_var * mus_s
#         eta2_src = -0.5 * inv_var
#     eta1_list.append(eta1_src.detach().clone().to(torch.float32))
#     eta2d_list.append(eta2_src.detach().clone().to(torch.float32))

#     all_domains: List[DomainDataset] = []

#     # -------------------- Intermediates --------------------
#     for i in range(1, n_inter + 1):
#         _t_step = time.time()
#         t = i / (n_inter + 1)

#         # class allocation (unchanged)
#         pi_mid = (counts_s.float() / max(1, total_s))
#         desired = (pi_mid * float(total_min)).clamp_min(0.0)
#         base = torch.floor(desired)
#         frac = desired - base
#         present = ((counts_s > 0) & (counts_t > 0)).float()
#         base = base * present
#         frac = frac * present
#         n_alloc = int(base.sum().item())
#         rem = int(max(0, total_min - n_alloc))
#         if rem > 0:
#             k_take = min(rem, K)
#             if k_take > 0:
#                 _, idx = torch.topk(frac, k=k_take)
#                 add = torch.zeros_like(base); add[idx] = 1.0
#                 base = base + add
#         n_per_class = base.long()

#         # per-step parameter holders (Torch tensors; convert later)
#         mu_mid_full  = torch.full((K, d), float("nan"), device=device, dtype=mus_s.dtype)
#         var_mid_full = torch.full((K, d), float("nan"), device=device, dtype=mus_s.dtype)
#         if cov_type == "full":
#             Sig_mid_full = torch.full((K, d, d), float("nan"), device=device, dtype=mus_s.dtype)
#         else:
#             Sig_mid_full = None

#         Zm_list: List[torch.Tensor] = []
#         Ym_list: List[torch.Tensor] = []

#         # per-class synthesis
#         for k_idx in range(K):
#             n_k = int(n_per_class[k_idx].item())
#             if n_k <= 0 or counts_s[k_idx].item() == 0 or counts_t[k_idx].item() == 0:
#                 continue

#             if cov_type == "diag":
#                 # you can keep your fr_interp_diag; here’s a direct version:
#                 # log(var)_mid = (1-t) log(var_s) + t log(var_t)
#                 mu_s_k, var_s_k = mus_s[k_idx], vars_s[k_idx]
#                 mu_t_k, var_t_k = mus_t[k_idx], vars_t[k_idx]
#                 mu_mid = (1.0 - t) * mu_s_k + t * mu_t_k
#                 var_mid = torch.exp((1.0 - t) * torch.log(var_s_k + jitter) + t * torch.log(var_t_k + jitter))
#                 std_mid = torch.sqrt(var_mid + 0.0)  # ensure non-neg
#                 Zk = torch.randn(n_k, d, device=device, dtype=mus_s.dtype).mul_(std_mid).add_(mu_mid)

#                 mu_mid_full[k_idx]  = mu_mid
#                 var_mid_full[k_idx] = var_mid

#             else:
#                 # FULL: use the cached bridge for Σ(t)
#                 mu_s_k = mus_s[k_idx]
#                 mu_t_k = mus_t[k_idx]
#                 mu_mid = (1.0 - t) * mu_s_k + t * mu_t_k
#                 Sig_mid = fr_cov_at_t(k_idx, float(t))                              # (d,d)

#                 # sample with Cholesky
#                 L_mid = torch.linalg.cholesky(0.5*(Sig_mid + Sig_mid.T) + jitter * torch.eye(d, device=device, dtype=Sig_mid.dtype))
#                 Zk = (torch.randn(n_k, d, device=device, dtype=mus_s.dtype) @ L_mid.T) + mu_mid

#                 mu_mid_full[k_idx]  = mu_mid
#                 var_mid_full[k_idx] = torch.diagonal(Sig_mid)
#                 Sig_mid_full[k_idx] = Sig_mid

#             Yk = torch.full((n_k,), k_idx, device=device, dtype=torch.long)
#             Zm_list.append(Zk); Ym_list.append(Yk)

#         # record mid-step params
#         steps.append(float(t))
#         mu_list.append(mu_mid_full.detach().clone().to(torch.float32))
#         var_list.append(var_mid_full.detach().clone().to(torch.float32))
#         counts_list.append(n_per_class.detach().cpu().numpy().astype(np.int64))
#         pi_list.append(pi_mid.detach().cpu().numpy())

#         # natural params at this step (batched)
#         if cov_type == "full":
#             # fix any NaN class by falling back to diag(var)
#             # (rare; ensures η well-defined)
#             if torch.isnan(Sig_mid_full).any():
#                 diag_fallback = torch.diag_embed(torch.clamp(var_mid_full, min=jitter))
#                 mask_bad = ~torch.isfinite(Sig_mid_full).all(dim=(-1, -2))
#                 Sig_mid_full[mask_bad] = diag_fallback[mask_bad]

#             L = torch.linalg.cholesky(0.5*(Sig_mid_full + Sig_mid_full.transpose(-1, -2)) + jitter * eye)
#             eta1_mid = torch.cholesky_solve(mu_mid_full.unsqueeze(-1), L).squeeze(-1)     # (K,d)
#             Linv = torch.linalg.solve(L, eye)                                             # (K,d,d)
#             diag_inv = (Linv ** 2).sum(dim=-1)                                            # (K,d)
#             eta2_mid = -0.5 * diag_inv

#             Sigma_list.append(Sig_mid_full.detach().clone().to(torch.float32))
#             eta1_list.append(eta1_mid.detach().clone().to(torch.float32))
#             eta2d_list.append(eta2_mid.detach().clone().to(torch.float32))
#         else:
#             inv_var_mid = (var_mid_full + jitter).reciprocal()
#             eta1_mid = inv_var_mid * mu_mid_full
#             eta2_mid = -0.5 * inv_var_mid
#             eta1_list.append(eta1_mid.detach().clone().to(torch.float32))
#             eta2d_list.append(eta2_mid.detach().clone().to(torch.float32))

#         if Zm_list:
#             Zm = torch.cat(Zm_list, 0).to("cpu", dtype=torch.float32)
#             Ym = torch.cat(Ym_list, 0).to("cpu", dtype=torch.long)
#             weights = torch.ones(len(Ym), dtype=torch.float32)
#             all_domains.append(DomainDataset(Zm, weights, Ym, Ym))
#         print(f"[FR] Step {i}/{n_inter}: generated {len(Ym_list) and sum([len(_y) for _y in Ym_list]) or 0} samples with d={d} in {time.time()-_t_step:.2f}s")

#     # -------------------- Target (t=1) --------------------
#     try:
#         X_tgt_final = xt if torch.is_tensor(xt) else torch.as_tensor(xt)
#         X_tgt_final = X_tgt_final.cpu().to(torch.float32)
#         Y_tgt_final = dataset_t.targets if torch.is_tensor(dataset_t.targets) else torch.as_tensor(dataset_t.targets, dtype=torch.long)
#         Y_tgt_final = Y_tgt_final.cpu().long()
#         Y_em_final  = Yt_em.cpu().long()
#         W_tgt_final = torch.ones(len(Y_em_final), dtype=torch.float32)
#         all_domains.append(DomainDataset(X_tgt_final, W_tgt_final, Y_tgt_final, Y_em_final))
#     except Exception as e:
#         print(f"[FR] Warning: failed to wrap target with targets_em ({e}); appending raw dataset.")
#         all_domains.append(dataset_t)

#     pi_t = (counts_t.float() / max(1, total_t)).detach().cpu().numpy()
#     steps.append(1.0)
#     mu_list.append(mus_t.detach().clone().to(torch.float32))
#     var_list.append(vars_t.detach().clone().to(torch.float32))
#     counts_list.append(counts_t.detach().cpu().numpy().astype(np.int64))
#     pi_list.append(pi_t)

#     if cov_type == "full":
#         # for target
#         Sig_t_spd = 0.5 * (Sig_t + Sig_t.transpose(-1, -2)) + jitter * eye
#         Sigma_list.append(Sig_t.detach().clone().to(torch.float32))

#         L = torch.linalg.cholesky(Sig_t_spd)
#         eta1_t = torch.cholesky_solve(mus_t.unsqueeze(-1), L).squeeze(-1)
#         Linv = torch.linalg.solve(L, eye)
#         diag_inv = (Linv ** 2).sum(dim=-1)
#         eta2_t = -0.5 * diag_inv
#     else:
#         inv_var_t = (vars_t + jitter).reciprocal()
#         eta1_t = inv_var_t * mus_t
#         eta2_t = -0.5 * inv_var_t
#     eta1_list.append(eta1_t.detach().clone().to(torch.float32))
#     eta2d_list.append(eta2_t.detach().clone().to(torch.float32))

#     print(f"[FR] Total generation time: {time.time()-_t_total:.2f}s")
#     generated_size = len(all_domains[-2].data) if len(all_domains) > 1 else 0
#     print(f"Total data for each intermediate domain: {generated_size}")

#     # -------------------- pack domain params (NumPy conversion ONCE) --------------------
#     def _to_np32(x: torch.Tensor) -> np.ndarray:
#         return x.detach().cpu().to(torch.float32).numpy()

#     steps_np = np.asarray(steps, dtype=np.float32)
#     mu_np    = np.stack([_to_np32(t) for t in mu_list], axis=0)         # (S,K,d)
#     var_np   = np.stack([_to_np32(t) for t in var_list], axis=0)        # (S,K,d)
#     eta1_np  = np.stack([_to_np32(t) for t in eta1_list], axis=0)       # (S,K,d)
#     eta2_np  = np.stack([_to_np32(t) for t in eta2d_list], axis=0)      # (S,K,d)
#     if cov_type == "full":
#         Sigma_np = np.stack([_to_np32(t) for t in Sigma_list], axis=0)  # (S,K,d,d)
#     else:
#         Sigma_np = None

#     domain_params = {
#         "K": int(K),
#         "d": int(d),
#         "cov_type": cov_type,
#         "steps": steps_np,                              # (S,)
#         "mu":    mu_np,                                 # (S, K, d)
#         "var":   var_np,                                # (S, K, d)
#         "counts": np.asarray(counts_list, dtype=np.int64),  # (S, K)
#         "pi":      np.asarray(pi_list, dtype=np.float32),   # (S, K)
#         "present_source": present_s.astype(np.bool_),
#         "present_target": present_t.astype(np.bool_),
#         "eta1":      eta1_np,                           # (S,K,d)
#         "eta2_diag": eta2_np,                           # (S,K,d)
#     }
#     if cov_type == "full":
#         domain_params["Sigma"] = Sigma_np              # (S,K,d,d)

#     if save_path is not None:
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         np.savez_compressed(save_path, **domain_params)
#         print(f"[FR] Saved domain parameters -> {save_path}")

#     # -------------------- Optional lightweight FR audit --------------------
#     # Exact identity: logdet Σ(t) = (1-t) logdet Σ_s + t logdet Σ_t. Sample-check a few pairs if visualize=True.
#     if visualize and cov_type == "full":
#         with torch.no_grad():
#             idx_k = torch.arange(min(K, 3), device=device)
#             ts_samp = torch.tensor([0.25, 0.5, 0.75], device=device)
#             eye_d = torch.eye(d, device=device, dtype=mus_s.dtype)
#             for k in idx_k:
#                 ld0 = torch.logdet(0.5*(Sig_s[k]+Sig_s[k].T)+jitter*eye_d)
#                 ld1 = torch.logdet(0.5*(Sig_t[k]+Sig_t[k].T)+jitter*eye_d)
#                 for t_ in ts_samp:
#                     Sig_mid = fr_cov_at_t(int(k), float(t_))
#                     lhs = torch.logdet(0.5*(Sig_mid+Sig_mid.T)+jitter*eye_d)
#                     rhs = (1.0 - t_) * ld0 + t_ * ld1
#                     if not torch.isclose(lhs, rhs, rtol=1e-4, atol=1e-6):
#                         print(f"[FR-CHK] k={int(k)} t={float(t_):.2f} Δ={float((lhs-rhs).abs())}")

#     return all_domains, Yt_em.cpu().numpy(), domain_params

def eig_stats(Sigma: torch.Tensor, name="Σ", eps: float = 1e-8):
    # Make SPD numerically
    S = 0.5 * (Sigma + Sigma.T)
    S = S + eps * torch.eye(S.shape[0], dtype=S.dtype, device=S.device)

    # Symmetric eigen-decomposition (ascending)
    w = torch.linalg.eigvalsh(S)

    # Basic summaries
    lam_min = w[0].item()
    lam_max = w[-1].item()
    trace   = torch.sum(w).item()
    fro     = torch.linalg.norm(S, ord='fro').item()
    cond    = lam_max / lam_min
    sign, logabsdet = torch.linalg.slogdet(S)  # sign should be +1 for SPD

    print(f"[EIG] {name}: "
          f"λ_min={lam_min:.6g}, λ_max={lam_max:.6g}, cond={cond:.6g}, "
          f"trace={trace:.6g}, ||·||_F={fro:.6g}, logdet={logabsdet:.6g}")
    return w  # return full spectrum if you want to plot/quantile


# @torch.no_grad()
# def natural_interp_full(mu_s: torch.Tensor,
#                         Sigma_s: torch.Tensor,
#                         mu_t: torch.Tensor,
#                         Sigma_t: torch.Tensor,
#                         t: float,
#                         eps: float = 0,
#                         debug: bool = True,
#                         tag: str = "",
#                         eig_checks: bool = True,              # <-- NEW: toggle eigen checks
#                         pct: tuple = (1, 5, 10, 25, 50, 75, 90, 95, 99),  # <-- NEW: percentiles
#                         warn_tol: float = 1e-12               # <-- NEW: tiny tolerance for comparisons
#                         ):
#     """
#     Natural-parameter interpolation (e-geodesic) for full-cov Gaussians.
#     Returns:
#       mu_m, Sigma_m, eta1_m, eta2_m
#     """
#     d = mu_s.numel()
#     I = torch.eye(d, dtype=Sigma_s.dtype, device=Sigma_s.device)

#     # ---- utilities ----------------------------------------------------------
#     def _spd_prec(S):
#         # SPD polish + jitter
#         # S = 0.5 * (S + S.T) + eps * I
#         L = torch.linalg.cholesky(S)
#         inv_S = torch.cholesky_inverse(L)   # symmetric precision
#         return inv_S, L

#     def _eigvals(S):
#         # assumes S is symmetric; add a tiny jitter to be safe
#         S = 0.5 * (S + S.T) + eps * I
#         return torch.linalg.eigvalsh(S)  # ascending

#     def _cov_stats(S: torch.Tensor, name: str):
#         S = 0.5 * (S + S.T)
#         w = torch.linalg.eigvalsh(S)
#         w_clamped = torch.clamp(w, min=1e-20)
#         tr = torch.sum(w_clamped)
#         fro = torch.linalg.norm(S, ord='fro')
#         lam_min = torch.min(w_clamped)
#         lam_max = torch.max(w_clamped)
#         cond = (lam_max / lam_min).item()
#         sign, logabsdet = torch.linalg.slogdet(S)
#         print(f"[NAT-DBG{(':' + tag) if tag else ''}] {name}: "
#               f"trace={tr.item():.6g}  ||·||_F={fro.item():.6g}  "
#               f"λ_min={lam_min.item():.6g}  λ_max={lam_max.item():.6g}  "
#               f"cond={cond:.6g}  logdet={logabsdet.item():.6g}")
#         return w  # return raw spectrum (ascending)

#     def _percentiles(w: torch.Tensor, ps: tuple):
#         qs = torch.quantile(w, torch.tensor([p/100.0 for p in ps], device=w.device))
#         return {p: qs[i].item() for i, p in enumerate(ps)}

#     def _print_percentiles(w: torch.Tensor, name: str):
#         q = _percentiles(w, pct)
#         q_str = "  ".join([f"p{p}={q[p]:.6g}" for p in pct])
#         print(f"[EIG-PCT{(':' + tag) if tag else ''}] {name}: {q_str}")

#     # ---- precisions / natural params ---------------------------------------
#     breakpoint()
#     Lambda_s, Ls = _spd_prec(Sigma_s)
#     Lambda_t, Lt = _spd_prec(Sigma_t)

#     eta1_s = Lambda_s @ mu_s
#     eta2_s = -0.5 * Lambda_s
#     eta1_t = Lambda_t @ mu_t
#     eta2_t = -0.5 * Lambda_t

#     # e-geodesic: linear in natural parameters
#     eta1_m = (1.0 - t) * eta1_s + t * eta1_t
#     eta2_m = (1.0 - t) * eta2_s + t * eta2_t
#     # eta1_m = t * eta1_t
#     # eta2_m = t * eta2_t
#     # back to (mu, Sigma)
#     Lambda_m = -2.0 * eta2_m
#     # Lambda_m = 0.5 * (Lambda_m + Lambda_m.T) + eps * I
#     Lm = torch.linalg.cholesky(Lambda_m)
#     Sigma_m = torch.cholesky_inverse(Lm)
#     mu_m = torch.cholesky_solve(eta1_m.unsqueeze(1), Lm).squeeze(1)
#     # Sigma_m = 0.5 * (Sigma_m + Sigma_m.T)

#     # ---- diagnostics --------------------------------------------------------
#     if debug:
#         # Basic magnitude stats
#         w_s = _cov_stats(Sigma_s + eps * I, "Σ_source")
#         w_t = _cov_stats(Sigma_t + eps * I, "Σ_target")
#         w_m = _cov_stats(Sigma_m + eps * I, f"Σ_interp(t={t:.3f})")

#         # Precisions (informative to see linearity/conditioning there)
#         def _prec_stats(P: torch.Tensor, name: str):
#             P = 0.5 * (P + P.T)
#             w = torch.linalg.eigvalsh(P)
#             w_clamped = torch.clamp(w, min=1e-20)
#             tr = torch.sum(w_clamped)
#             fro = torch.linalg.norm(P, ord='fro')
#             lam_min = torch.min(w_clamped)
#             lam_max = torch.max(w_clamped)
#             cond = (lam_max / lam_min).item()
#             sign, logabsdet = torch.linalg.slogdet(P)
#             print(f"[NAT-DBG{(':' + tag) if tag else ''}] {name}: "
#                   f"trace={tr.item():.6g}  ||·||_F={fro.item():.6g}  "
#                   f"λ_min={lam_min.item():.6g}  λ_max={lam_max.item():.6g}  "
#                   f"cond={cond:.6g}  logdet={logabsdet.item():.6g}")
#             return w
#         wp_s = _prec_stats(Lambda_s, "Λ_source")
#         wp_t = _prec_stats(Lambda_t, "Λ_target")
#         wp_m = _prec_stats(Lambda_m, f"Λ_interp(t={t:.3f})")

#         # Consistency: ΛΣ ≈ I
#         resid_inv = torch.linalg.norm(Lambda_m @ Sigma_m - I, ord='fro') / d
#         print(f"[CHK{(':' + tag) if tag else ''}] inverse residual ‖Λ(t)Σ(t)-I‖_F/d = {resid_inv:.3e}")

#         # Consistency: harmonic-mean linearity in precision
#         Lam_lin = (1.0 - t) * Lambda_s + t * Lambda_t
#         rel_lin = (torch.linalg.norm(Lambda_m - Lam_lin, ord='fro') /
#                    (torch.linalg.norm(Lam_lin, ord='fro') + 1e-20))
#         print(f"[CHK{(':' + tag) if tag else ''}] precision linearity ‖Λ(t) - [(1-t)Λ_s+tΛ_t]‖_F / ‖·‖_F = {rel_lin:.3e}")

#         # Non-commutativity measure (0 iff commute exactly)
#         comm = Sigma_s @ Sigma_t - Sigma_t @ Sigma_s
#         comm_norm = torch.linalg.norm(comm, ord='fro')
#         print(f"[CHK{(':' + tag) if tag else ''}] commutator ‖Σ_s Σ_t - Σ_t Σ_s‖_F = {comm_norm:.6g}")

#         # ---- eigen value checks --------------------------------------------
#         if eig_checks:
#             # Percentiles (bulk movement)
#             _print_percentiles(w_s, "Σ_source")
#             _print_percentiles(w_t, "Σ_target")
#             _print_percentiles(w_m, f"Σ_interp(t={t:.3f})")

#             # "Between endpoints" checks for extreme eigenvalues
#             lam_min_s, lam_max_s = w_s[0].item(), w_s[-1].item()
#             lam_min_t, lam_max_t = w_t[0].item(), w_t[-1].item()
#             lam_min_m, lam_max_m = w_m[0].item(), w_m[-1].item()

#             lo_end = min(lam_min_s, lam_min_t)
#             hi_end = max(lam_max_s, lam_max_t)

#             below_hi = (lam_max_m < hi_end - warn_tol)
#             above_lo = (lam_min_m > lo_end + warn_tol)

#             print(f"[EIG-CHK{(':' + tag) if tag else ''}] endpoints: "
#                   f"λ_min^ends={lo_end:.6g}, λ_max^ends={hi_end:.6g}; "
#                   f"interp: λ_min={lam_min_m:.6g}, λ_max={lam_max_m:.6g}")

#             if below_hi:
#                 print(f"[EIG-OBS{(':' + tag) if tag else ''}] λ_max at t={t:.3f} is "
#                       f"below the larger endpoint (expected under non-commuting harmonic mean).")
#             if above_lo:
#                 print(f"[EIG-OBS{(':' + tag) if tag else ''}] λ_min at t={t:.3f} "
#                       f"exceeds the smaller endpoint (conditioning improves).")

#             # Optional: report median and a couple of high/low percentiles difference vs endpoints
#             # for p in (10, 50, 90):
#             #     qm = torch.quantile(w_m, torch.tensor(p/100.0, device=w_m.device)).item()
#             #     qs = torch.quantile(w_s, torch.tensor(p/100.0, device=w_s.device)).item()
#             #     qt = torch.quantile(w_t, torch.tensor(p/100.0, device=w_t.device)).item()
#             #     print(f"[EIG-ΔPCT{(':' + tag) if tag else ''}] p{p}: "
#             #           f"interp={qm:.6g}  source={qs:.6g}  target={qt:.6g}")

#     return mu_m, Sigma_m, eta1_m, eta2_m


# @torch.no_grad()
# def natural_interp_full(mu_s: torch.Tensor,
#                         Sigma_s: torch.Tensor,
#                         mu_t: torch.Tensor,
#                         Sigma_t: torch.Tensor,
#                         t: float,
#                         eps: float = 0.0,                 # base jitter for robust SPD (B)
#                         debug: bool = True,
#                         tag: str = "",
#                         eig_checks: bool = True,
#                         pct: tuple = (1, 5, 10, 25, 50, 75, 90, 95, 99),
#                         warn_tol: float = 1e-12,
#                         *,
#                         compare: bool = False,            # <<< NEW: run A/B comparison
#                         scale_alpha: float = 1e5,         # <<< scaling factor for method A
#                         return_report: bool = True        # <<< when compare=True, also return a report dict
#                         ):
#     """
#     Natural-parameter interpolation (e-geodesic) for full-cov Gaussians.

#     Returns (default): mu_m, Sigma_m, eta1_m, eta2_m          # robust SPD (method B)
#     If compare=True and return_report=True, returns a 5th item: dict with metrics.
#     """
#     device = Sigma_s.device
#     dtype  = Sigma_s.dtype
#     d = mu_s.numel()
#     I = torch.eye(d, dtype=dtype, device=device)

#     # ---------- small helpers ----------
#     def _sym(S):
#         return 0.5 * (S + S.T)

#     def _cond(S):
#         # Fro-safe condition number (clamp tiny eigs)
#         w = torch.linalg.eigvalsh(_sym(S)).clamp_min(1e-20)
#         return (w[-1] / w[0]).item()

#     def _cov_stats(S: torch.Tensor, name: str):
#         S = _sym(S)
#         w = torch.linalg.eigvalsh(S)
#         w_clamped = torch.clamp(w, min=1e-20)
#         tr = torch.sum(w_clamped)
#         fro = torch.linalg.norm(S, ord='fro')
#         lam_min = torch.min(w_clamped)
#         lam_max = torch.max(w_clamped)
#         cond = (lam_max / lam_min).item()
#         sign, logabsdet = torch.linalg.slogdet(S)
#         print(f"[NAT-DBG{(':' + tag) if tag else ''}] {name}: "
#               f"trace={tr.item():.6g}  ||·||_F={fro.item():.6g}  "
#               f"λ_min={lam_min.item():.6g}  λ_max={lam_max.item():.6g}  "
#               f"cond={cond:.6g}  logdet={logabsdet.item():.6g}")
#         return w

#     def _percentiles(w: torch.Tensor, ps: tuple):
#         qs = torch.quantile(w, torch.tensor([p/100.0 for p in ps], device=w.device))
#         return {p: qs[i].item() for i, p in enumerate(ps)}

#     def _print_percentiles(w: torch.Tensor, name: str):
#         q = _percentiles(w, pct)
#         q_str = "  ".join([f"p{p}={q[p]:.6g}" for p in pct])
#         print(f"[EIG-PCT{(':' + tag) if tag else ''}] {name}: {q_str}")

#     # ---------- Method A: scale–then–unscale precision ----------
#     def _prec_scale_unscale(S: torch.Tensor, alpha: float):
#         # Compute inv(S) via inv(alpha*S) * alpha
#         Ssym = _sym(S)
#         L = torch.linalg.cholesky(alpha * Ssym)
#         Lam_scaled = torch.cholesky_inverse(L)      # (alpha*S)^{-1}
#         Lam = Lam_scaled * alpha                    # recover S^{-1}
#         return Lam

#     # ---------- Method B: robust SPD precision (adaptive jitter, fallback eig-clip) ----------
#     def _prec_robust(S: torch.Tensor, base_eps: float):
#         Ssym = _sym(S)
#         # start with relative jitter based on mean diag scale
#         scale = torch.mean(torch.diag(Ssym)).abs().item()
#         jitter = max(base_eps, 1e-12 * (scale if np.isfinite(scale) else 1.0))
#         tries = 0
#         while True:
#             try:
#                 L = torch.linalg.cholesky(Ssym + jitter * I)
#                 Lam = torch.cholesky_inverse(L)
#                 return Lam, L, jitter, False  # no eig-clip
#             except RuntimeError:
#                 tries += 1
#                 if tries >= 5:
#                     # fallback: eig-clip
#                     w, V = torch.linalg.eigh(Ssym)
#                     w = torch.clamp(w, min=jitter if jitter > 0 else 1e-12)
#                     Sp = (V * w) @ V.T
#                     L = torch.linalg.cholesky(Sp)
#                     Lam = torch.cholesky_inverse(L)
#                     return Lam, L, jitter, True
#                 jitter *= 10.0

#     # ---------- Build endpoint precisions: A and B ----------
#     # A endpoints

#     # breakpoint()
#     Lambda_s_A = _prec_scale_unscale(Sigma_s, scale_alpha)
#     Lambda_t_A = _prec_scale_unscale(Sigma_t, scale_alpha)

#     # B endpoints
#     Lambda_s_B, Ls_B, jit_s, clip_s = _prec_robust(Sigma_s, eps)
#     Lambda_t_B, Lt_B, jit_t, clip_t = _prec_robust(Sigma_t, eps)

#     # ---------- Natural parameters (A and B) ----------
#     def _etas(Lam, mu):
#         eta1 = Lam @ mu
#         eta2 = -0.5 * Lam
#         return eta1, eta2

#     eta1_s_A, eta2_s_A = _etas(Lambda_s_A, mu_s)
#     eta1_t_A, eta2_t_A = _etas(Lambda_t_A, mu_t)

#     eta1_s_B, eta2_s_B = _etas(Lambda_s_B, mu_s)
#     eta1_t_B, eta2_t_B = _etas(Lambda_t_B, mu_t)

#     # Interpolate in natural space
#     def _interp_eta(e1_s, e1_t, e2_s, e2_t, t):
#         eta1_m = (1.0 - t) * e1_s + t * e1_t
#         eta2_m = (1.0 - t) * e2_s + t * e2_t
#         Lam_m  = -2.0 * eta2_m
#         Lm     = torch.linalg.cholesky(_sym(Lam_m))
#         Sig_m  = torch.cholesky_inverse(Lm)
#         mu_m   = torch.cholesky_solve(eta1_m.unsqueeze(1), Lm).squeeze(1)
#         return mu_m, Sig_m, eta1_m, eta2_m, Lam_m

#     # Mid-point (or general t) for A and B
#     mu_m_A, Sigma_m_A, eta1_m_A, eta2_m_A, Lambda_m_A = _interp_eta(eta1_s_A, eta1_t_A, eta2_s_A, eta2_t_A, t)
#     mu_m_B, Sigma_m_B, eta1_m_B, eta2_m_B, Lambda_m_B = _interp_eta(eta1_s_B, eta1_t_B, eta2_s_B, eta2_t_B, t)

#     # ---------- Diagnostics ----------
#     def _inv_resid(Lam, Sig):
#         return (torch.linalg.norm(Lam @ Sig - I, ord='fro') / d).item()

#     report = None
#     if compare:
#         # Endpoint residuals
#         res_s_A = _inv_resid(Lambda_s_A, Sigma_s)
#         res_t_A = _inv_resid(Lambda_t_A, Sigma_t)
#         res_s_B = _inv_resid(Lambda_s_B, Sigma_s + jit_s * I)
#         res_t_B = _inv_resid(Lambda_t_B, Sigma_t + jit_t * I)

#         # Mid residuals
#         res_m_A = _inv_resid(Lambda_m_A, Sigma_m_A)
#         res_m_B = _inv_resid(Lambda_m_B, Sigma_m_B)

#         # Condition numbers
#         kS_s = _cond(Sigma_s); kS_t = _cond(Sigma_t)
#         kS_mA = _cond(Sigma_m_A); kS_mB = _cond(Sigma_m_B)

#         # Relative diffs between A and B at t
#         rel_mu = (torch.linalg.norm(mu_m_A - mu_m_B) /
#                   (torch.linalg.norm(mu_m_B) + 1e-20)).item()
#         rel_S  = (torch.linalg.norm(Sigma_m_A - Sigma_m_B, ord='fro') /
#                   (torch.linalg.norm(Sigma_m_B, ord='fro') + 1e-20)).item()

#         print("\n==== [A/B comparison @ natural_interp_full] ====")
#         print(f"Endpoints:  κ(Σ_s)={kS_s:.3e}  κ(Σ_t)={kS_t:.3e}")
#         print(f"A endpoint residuals:  ‖Λ_s Σ_s−I‖/d={res_s_A:.3e},  ‖Λ_t Σ_t−I‖/d={res_t_A:.3e}")
#         print(f"B endpoint residuals:  ‖Λ_s Σ_s−I‖/d={res_s_B:.3e},  ‖Λ_t Σ_t−I‖/d={res_t_B:.3e}  "
#               f"(jit_s={jit_s:.1e}{' eig-clip' if clip_s else ''}, jit_t={jit_t:.1e}{' eig-clip' if clip_t else ''})")
#         print(f"Midpoint residuals:    A: {res_m_A:.3e}   B: {res_m_B:.3e}")
#         print(f"Midpoint κ(Σ):         A: {kS_mA:.3e}     B: {kS_mB:.3e}")
#         print(f"Rel. diff @ t in μ:    {rel_mu:.3e}")
#         print(f"Rel. diff @ t in Σ(F): {rel_S:.3e}")
#         print("=================================================\n")

#         report = {
#             "endpoint_residuals": {
#                 "A": {"source": res_s_A, "target": res_t_A},
#                 "B": {"source": res_s_B, "target": res_t_B, "jit_s": jit_s, "jit_t": jit_t,
#                       "eigclip_source": bool(clip_s), "eigclip_target": bool(clip_t)},
#             },
#             "mid_residuals": {"A": res_m_A, "B": res_m_B},
#             "kappa": {"Sigma_s": kS_s, "Sigma_t": kS_t, "Sigma_mid_A": kS_mA, "Sigma_mid_B": kS_mB},
#             "relative_diff": {"mu": rel_mu, "Sigma_fro": rel_S},
#         }

#     # ---- optional detailed debug on B (robust) ----
#     if debug:
#         w_s = _cov_stats(Sigma_s + eps * I, "Σ_source")
#         w_t = _cov_stats(Sigma_t + eps * I, "Σ_target")
#         w_m = _cov_stats(Sigma_m_B + eps * I, f"Σ_interp(t={t:.3f}) [B]")
#         def _prec_stats(P: torch.Tensor, name: str):
#             P = _sym(P)
#             w = torch.linalg.eigvalsh(P)
#             w_clamped = torch.clamp(w, min=1e-20)
#             tr = torch.sum(w_clamped)
#             fro = torch.linalg.norm(P, ord='fro')
#             lam_min = torch.min(w_clamped)
#             lam_max = torch.max(w_clamped)
#             cond = (lam_max / lam_min).item()
#             sign, logabsdet = torch.linalg.slogdet(P)
#             print(f"[NAT-DBG{(':' + tag) if tag else ''}] {name}: "
#                   f"trace={tr.item():.6g}  ||·||_F={fro.item():.6g}  "
#                   f"λ_min={lam_min.item():.6g}  λ_max={lam_max.item():.6g}  "
#                   f"cond={cond:.6g}  logdet={logabsdet.item():.6g}")
#             return w
#         _ = _prec_stats(Lambda_s_B, "Λ_source [B]")
#         _ = _prec_stats(Lambda_t_B, "Λ_target [B]")
#         _ = _prec_stats(Lambda_m_B, f"Λ_interp(t={t:.3f}) [B]")

#         resid_inv = torch.linalg.norm(Lambda_m_B @ Sigma_m_B - torch.eye(d, device=device, dtype=dtype), ord='fro') / d
#         print(f"[CHK{(':' + tag) if tag else ''}] inverse residual (B) ‖Λ(t)Σ(t)-I‖_F/d = {resid_inv:.3e}")

#         # non-commutativity
#         comm = Sigma_s @ Sigma_t - Sigma_t @ Sigma_s
#         comm_norm = torch.linalg.norm(comm, ord='fro')
#         print(f"[CHK{(':' + tag) if tag else ''}] commutator ‖Σ_s Σ_t - Σ_t Σ_s‖_F = {comm_norm:.6g}")

#         if eig_checks:
#             # Percentiles
#             def _pp(w, name): 
#                 q = _percentiles(w, pct)
#                 q_str = "  ".join([f"p{p}={q[p]:.6g}" for p in pct]); 
#                 print(f"[EIG-PCT{(':' + tag) if tag else ''}] {name}: {q_str}")
#             _pp(w_s, "Σ_source")
#             _pp(w_t, "Σ_target")
#             _pp(w_m, f"Σ_interp(t={t:.3f}) [B]")

#             lam_min_s, lam_max_s = w_s[0].item(), w_s[-1].item()
#             lam_min_t, lam_max_t = w_t[0].item(), w_t[-1].item()
#             lam_min_m, lam_max_m = w_m[0].item(), w_m[-1].item()
#             lo_end = min(lam_min_s, lam_min_t)
#             hi_end = max(lam_max_s, lam_max_t)
#             below_hi = (lam_max_m < hi_end - warn_tol)
#             above_lo = (lam_min_m > lo_end + warn_tol)
#             print(f"[EIG-CHK{(':' + tag) if tag else ''}] endpoints: "
#                   f"λ_min^ends={lo_end:.6g}, λ_max^ends={hi_end:.6g}; "
#                   f"interp[B]: λ_min={lam_min_m:.6g}, λ_max={lam_max_m:.6g}")
#             if below_hi:
#                 print(f"[EIG-OBS{(':' + tag) if tag else ''}] λ_max at t={t:.3f} is below the larger endpoint (harmonic-mean effect).")
#             if above_lo:
#                 print(f"[EIG-OBS{(':' + tag) if tag else ''}] λ_min at t={t:.3f} exceeds the smaller endpoint (conditioning improves).")
#     # breakpoint()
#     # Return the robust (B) path by default, to preserve existing callers
#     if compare and return_report:
#         return mu_m_B, Sigma_m_B, eta1_m_B, eta2_m_B, report
#     return mu_m_B, Sigma_m_B, eta1_m_B, eta2_m_B


# @torch.no_grad()
# def natural_interp_diag(mu_s_k, var_s_k, mu_t_k, var_t_k, t, eps: float = 1e-8):
#     # precision vectors
#     prec_s = (1.0 / var_s_k.clamp_min(eps))
#     prec_t = (1.0 / var_t_k.clamp_min(eps))
#     # natural params
#     eta1_s = prec_s * mu_s_k
#     eta2_s = -0.5 * prec_s
#     eta1_t = prec_t * mu_t_k
#     eta2_t = -0.5 * prec_t
#     # interpolate
#     eta1_m = (1.0 - t) * eta1_s + t * eta1_t
#     eta2_m = (1.0 - t) * eta2_s + t * eta2_t
#     # back to mean/var
#     prec_m = (-2.0) * eta2_m
#     prec_m = prec_m.clamp_min(eps)
#     var_m = 1.0 / prec_m
#     mu_m = var_m * eta1_m
#     return mu_m, var_m, eta1_m, eta2_m




import torch
from typing import Tuple, Optional

@torch.no_grad()
def natural_interp_full(
    mu_s: torch.Tensor,          # (K, d)
    Sigma_s: torch.Tensor,       # (K, d, d)
    mu_t: torch.Tensor,          # (K, d)
    Sigma_t: torch.Tensor,       # (K, d, d)
    t: float,
    *,
    base_eps: float = 1e-8,      # base jitter as a fraction of avg diag scale
    max_tries: int = 3,          # Cholesky retry passes (jitter *= 10 each pass)
    rel_floor: float = 1e-7,     # eigen clip floor: rel * (trace/d)
    abs_floor: float = 1e-12,    # eigen clip floor: absolute
    compute_dtype: Optional[torch.dtype] = torch.float32,
    linalg_dtype: Optional[torch.dtype]  = torch.float64,
    return_eta: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Vectorized natural-parameter interpolation across K classes on GPU.

    Interpolation in natural space:
        Λ_s = Σ_s^{-1}, Λ_t = Σ_t^{-1}
        η1_s = Λ_s μ_s,  η2_s = -½ Λ_s
        η1_t = Λ_t μ_t,  η2_t = -½ Λ_t
        η1(t) = (1-t)η1_s + tη1_t
        Λ(t)  = (1-t)Λ_s  + tΛ_t
        μ(t)  = Λ(t)^{-1} η1(t),  Σ(t) = Λ(t)^{-1}

    Returns:
        mu_m:   (K, d)
        Sigma_m:(K, d, d)
        eta1_m:(K, d) if return_eta else None
        eta2_m:(K, d, d) if return_eta else None
    """

    assert mu_s.ndim == 2 and mu_t.ndim == 2
    assert Sigma_s.ndim == 3 and Sigma_t.ndim == 3
    K, d = mu_s.shape
    device = mu_s.device
    I = torch.eye(d, device=device)

    def _sym(M: torch.Tensor) -> torch.Tensor:
        return 0.5 * (M + M.transpose(-1, -2))

    # ---- (0) Dtype strategy: promote fragile LA to linalg_dtype, keep results in compute_dtype
    if compute_dtype is not None:
        mu_s = mu_s.to(compute_dtype)
        mu_t = mu_t.to(compute_dtype)
        Sigma_s = Sigma_s.to(compute_dtype)
        Sigma_t = Sigma_t.to(compute_dtype)

    S_s = _sym(Sigma_s).to(linalg_dtype)
    S_t = _sym(Sigma_t).to(linalg_dtype)

    # ---- (1) Batched Cholesky with adaptive jitter (vectorized)
    # jitter_k = base_eps * mean(diag(Σ_k)); retry only for failing classes
    def chol_inv_batched(S: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # S: (K,d,d) linalg_dtype
        diag_mean = torch.diagonal(S, dim1=-2, dim2=-1).mean(-1).abs()  # (K,)
        # avoid zero scale; fallback to 1.0 when needed
        scale = torch.where(diag_mean > 0, diag_mean, torch.ones_like(diag_mean))
        jitter = (base_eps * scale).clamp_min(torch.finfo(S.dtype).eps)  # (K,)

        # current matrices: S + jitter*I per class
        L = torch.empty_like(S)
        ok = torch.zeros((K,), dtype=torch.bool, device=device)

        S_cur = S.clone()
        for _try in range(max_tries):
            S_try = S_cur + jitter.view(K, 1, 1) * I
            # Attempt batched Cholesky; failures will raise on a per-class basis only if we index them,
            # so we use try/except on the whole batch by masking.
            try:
                L_try = torch.linalg.cholesky(S_try)
                L = L_try
                ok = torch.ones_like(ok, dtype=torch.bool)
                break
            except RuntimeError:
                # Need per-class handling; do a masked loop over the failing subset (small mask size typically)
                ok_mask = torch.zeros_like(ok)
                L_buf = torch.empty_like(S)
                for k in range(K):
                    if ok[k]:
                        continue
                    try:
                        Lk = torch.linalg.cholesky(S_try[k])
                        L_buf[k] = Lk
                        ok_mask[k] = True
                    except RuntimeError:
                        # will retry this class with larger jitter
                        pass
                # write successes
                L[ok_mask] = L_buf[ok_mask]
                ok |= ok_mask
                if ok.all():
                    break
                # increase jitter for the failures
                jitter = torch.where(ok, jitter, jitter * 10.0)
                # no need to update S_cur; we add jitter anew next round

        # Fallback: eigen-clip only for remaining failures
        if not ok.all():
            fail_idx = (~ok).nonzero(as_tuple=False).flatten()
            # eigen decomposition for the failed subset
            Sf = S[fail_idx]
            w, V = torch.linalg.eigh(_sym(Sf))
            # floors per class: max(rel_floor*trace/d, abs_floor)
            trace = w.sum(-1, keepdim=True)  # (F,1)
            rel_thr = rel_floor * (trace / d)
            thr = torch.maximum(rel_thr, torch.full_like(rel_thr, abs_floor, dtype=w.dtype))
            w_clipped = torch.maximum(w, thr)
            Sp = (V * w_clipped) @ V.transpose(-1, -2)
            Lp = torch.linalg.cholesky(Sp)
            L[fail_idx] = Lp
            ok[fail_idx] = True

        Lam = torch.cholesky_solve(I.expand_as(S), L)  # (K,d,d), precision
        return Lam.to(compute_dtype), L.to(compute_dtype), ok.to(torch.bool), jitter.to(compute_dtype)

    Lambda_s, Ls, ok_s, _ = chol_inv_batched(S_s)
    Lambda_t, Lt, ok_t, _ = chol_inv_batched(S_t)

    # ---- (2) Natural parameters at endpoints (batched)
    # η1 = Λ μ,  η2 = -½ Λ
    def matvec_batch(A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # A: (K,d,d), x: (K,d) -> (K,d)
        return torch.bmm(A, x.unsqueeze(-1)).squeeze(-1)

    eta1_s = matvec_batch(Lambda_s, mu_s)
    eta2_s = -0.5 * Lambda_s
    eta1_t = matvec_batch(Lambda_t, mu_t)
    eta2_t = -0.5 * Lambda_t

    # ---- (3) Interpolate in natural space (batched)
    eta1_m = (1.0 - t) * eta1_s + t * eta1_t                   # (K,d)
    # Λ(t) = (1-t)Λ_s + tΛ_t  -> then η2(t) = -½ Λ(t)
    Lambda_m = (1.0 - t) * Lambda_s + t * Lambda_t             # (K,d,d)
    # symmetrize + tiny safety jitter proportional to mean diag
    Lambda_m = _sym(Lambda_m).to(linalg_dtype)
    diag_mean_m = torch.diagonal(Lambda_m, dim1=-2, dim2=-1).mean(-1).abs()
    lam_eps = (base_eps * diag_mean_m).clamp_min(torch.finfo(Lambda_m.dtype).eps)
    # Cholesky for Λ(t)
    Lm = torch.linalg.cholesky(Lambda_m + lam_eps.view(K, 1, 1) * I.to(linalg_dtype))
    # Σ(t) = Λ(t)^{-1}, μ(t) = Λ(t)^{-1} η1(t)
    eyeKd = I.to(linalg_dtype).expand(K, d, d)
    Sigma_m = torch.cholesky_solve(eyeKd, Lm).to(compute_dtype)          # (K,d,d)
    mu_m    = torch.cholesky_solve(eta1_m.to(linalg_dtype).unsqueeze(-1), Lm).squeeze(-1).to(compute_dtype)
    eta2_m  = (-0.5 * Lambda_m).to(compute_dtype)

    # Cast final eta1_m to compute_dtype
    eta1_m = eta1_m.to(compute_dtype)

    if return_eta:
        return mu_m, Sigma_m, eta1_m, eta2_m
    return mu_m, Sigma_m, None, None

@torch.no_grad()
def natural_interp_diag(
    mu_s: torch.Tensor,     # (K,d)
    var_s: torch.Tensor,    # (K,d)
    mu_t: torch.Tensor,     # (K,d)
    var_t: torch.Tensor,    # (K,d)
    t: float,
    eps: float = 1e-8,
):
    var_s = var_s.clamp_min(eps)
    var_t = var_t.clamp_min(eps)
    prec_s = var_s.reciprocal()
    prec_t = var_t.reciprocal()

    eta1_s = prec_s * mu_s
    eta1_t = prec_t * mu_t
    # η2 = -½ Λ, but we only need Λ(t) for μ(t), var(t)
    prec_m = (1.0 - t) * prec_s + t * prec_t
    prec_m = prec_m.clamp_min(eps)
    eta1_m = (1.0 - t) * eta1_s + t * eta1_t

    var_m = prec_m.reciprocal()
    mu_m  = var_m * eta1_m
    # If you need eta2_m as well:
    eta2_m = -0.5 * prec_m
    return mu_m, var_m, eta1_m, eta2_m


# def generate_natural_domains_between(
#     n_inter,
#     dataset_s,
#     dataset_t,
#     plan=None,
#     entry_cutoff: int = 0,
#     conf: float = 0.0,
#     source_model: Optional[torch.nn.Module] = None,
#     pseudolabels: Optional[torch.Tensor] = None,
#     save_path: Optional[str] = None,
#     cov_type: str = "full",
#     reg: float = 1e-6,
#     ddof: int = 0,
#     jitter: float = 0.0,
#     diagnostic_class: Optional[int] = 1,  # kept in signature but unused now
#     visualize: bool = True,
#     args=None,
# ):
#     """
#     Generate intermediate domains by linear interpolation in the natural-parameter space
#     for Gaussian class-conditionals.

#     Natural parameters: η₁ = Λ μ,  η₂ = -½ Λ, where Λ = Σ⁻¹.
#     (Diagnostics disabled in this version.)
#     """
#     import time, os
    
#     import torch
#     import torch.nn as nn
#     # import matplotlib.pyplot as plt  # (unused now that diagnostics are off)

#     # ---- small local helpers -----------------------------------------------
#     def _to_np(x):
#         return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

#     def _pad_kd(mu_like, var_like, present_mask, K, d):
#         mu_out = np.full((K, d), np.nan, dtype=np.float64)
#         var_out = np.full((K, d), np.nan, dtype=np.float64)
#         mu_np, var_np = _to_np(mu_like), _to_np(var_like)
#         mu_out[present_mask] = mu_np[present_mask]
#         var_out[present_mask] = var_np[present_mask]
#         return mu_out, var_out

#     def _pad_kdd(Sig_like, present_mask, K, d):
#         Sig_out = np.full((K, d, d), np.nan, dtype=np.float64)
#         Sig_np = _to_np(Sig_like)
#         Sig_out[present_mask] = Sig_np[present_mask]
#         return Sig_out

#     print("------------Generate Intermediate domains (NATURAL)----------")
#     _t_total = time.time()

#     xs, xt = dataset_s.data, dataset_t.data
#     ys = dataset_s.targets
#     if len(xs.shape) > 2:
#         xs, xt = nn.Flatten()(xs), nn.Flatten()(xt)

#     if ys is None:
#         raise ValueError("Source dataset must provide targets for class statistics.")
#     if not hasattr(dataset_t, "targets_em") or dataset_t.targets_em is None:
#         raise ValueError("Target dataset must provide targets_em (EM-derived labels).")

#     device = xs.device if torch.is_tensor(xs) else torch.device("cpu")
#     Zs = xs if torch.is_tensor(xs) else torch.as_tensor(xs)
#     Zt = xt if torch.is_tensor(xt) else torch.as_tensor(xt)
#     Ys = ys if torch.is_tensor(ys) else torch.as_tensor(ys)
#     if args.em_match != "none":
#         Yt_em = dataset_t.targets_em if torch.is_tensor(dataset_t.targets_em) else torch.as_tensor(dataset_t.targets_em)
#     else:
#         Yt_em = dataset_t.targets_pesudo if torch.is_tensor(dataset_t.targets_pseudo) else torch.as_tensor(dataset_t.targets_pseudo)

#     Zs, Zt = Zs.to(device), Zt.to(device)
#     Ys, Yt_em = Ys.to(device, dtype=torch.long), Yt_em.to(device, dtype=torch.long)
#     if Ys.numel() == 0:
#         return [], torch.empty(0, dtype=torch.long)

#     K = int(max(Ys.max(), Yt_em.max()).item()) + 1

#     # --- class stats (source / target) ---
#     if cov_type == "diag":
#         mus_s, vars_s, _ = class_stats_diag(Zs, Ys, K)
#         mus_t, vars_t, _ = class_stats_diag(Zt, Yt_em, K)
#         Sig_s = Sig_t = None
#     else:
#         mus_s, Sig_s, _ = class_stats_full(Zs, Ys, K, reg=reg, ddof=ddof)
#         mus_t, Sig_t, _ = class_stats_full(Zt, Yt_em, K, reg=reg, ddof=ddof)
#         vars_s = torch.stack([torch.diag(Sig_s[k]) for k in range(K)], dim=0)
#         vars_t = torch.stack([torch.diag(Sig_t[k]) for k in range(K)], dim=0)

#     counts_s = torch.bincount(Ys, minlength=K)
#     counts_t = torch.bincount(Yt_em, minlength=K)
#     d = mus_s.shape[1]
#     total_s = int(counts_s.sum().item())
#     total_t = int(counts_t.sum().item())
#     total_min = int(torch.minimum(counts_s, counts_t).sum().item())
#     if total_min <= 0:
#         total_min = max(total_s, total_t)

#     present_s = (counts_s.detach().cpu().numpy() > 0)
#     present_t = (counts_t.detach().cpu().numpy() > 0)
#     present_both = present_s & present_t

#     # ---------------- containers ----------------
#     steps, mu_list, var_list = [], [], []
#     counts_list, pi_list = [], []
#     Sigma_list = [] if cov_type == "full" else None
#     eta1_steps, eta2d_steps = [], []

#     # ---------------- source (t=0) ----------------
#     pi_s = (counts_s.float() / max(1, total_s)).cpu().numpy()
#     mu0, var0 = _pad_kd(mus_s, vars_s, present_s, K, d)
#     steps.append(0.0)
#     mu_list.append(mu0)
#     var_list.append(var0)
#     counts_list.append(counts_s.cpu().numpy().astype(np.int64))
#     pi_list.append(pi_s)

#     # Endpoint η arrays computed ONCE and reused
#     eta1_src = np.full((K, d), np.nan, dtype=np.float64)
#     eta2d_src = np.full((K, d), np.nan, dtype=np.float64)
#     eta1_tgt = np.full((K, d), np.nan, dtype=np.float64)
#     eta2d_tgt = np.full((K, d), np.nan, dtype=np.float64)

#     if cov_type == "full":
#         Sigma0 = _pad_kdd(Sig_s, present_s, K, d)
#         Sigma_list.append(Sigma0)

#         # η(source) via SAME path (jitter + Cholesky)
#         mu0_t = torch.as_tensor(mu0, device=device, dtype=mus_s.dtype)
#         Sig0_t = torch.as_tensor(Sigma0, device=device, dtype=mus_s.dtype)
#         I = torch.eye(d, device=device, dtype=mus_s.dtype)

#         for k in range(K):
#             if not present_s[k]:
#                 continue
#             Sigk = 0.5 * (Sig0_t[k] + Sig0_t[k].T) + float(jitter) * I
#             L = torch.linalg.cholesky(Sigk)
#             Lam = torch.cholesky_inverse(L)
#             eta1_src[k]  = (Lam @ mu0_t[k]).detach().cpu().numpy()
#             eta2d_src[k] = (-0.5 * torch.diag(Lam)).detach().cpu().numpy()

#         eta1_steps.append(eta1_src.copy())
#         eta2d_steps.append(eta2d_src.copy())
#     else:
#         v = np.clip(var0, float(jitter), None)
#         eta1_src = mu0 / v
#         eta2d_src = -0.5 / v
#         eta1_steps.append(eta1_src.copy())
#         eta2d_steps.append(eta2d_src.copy())

#     # ---- Compute η(target) ONCE (for RHS); do NOT append yet ----
#     if cov_type == "full":
#         for k in range(K):
#             if not present_t[k]:
#                 continue
#             S1 = 0.5 * (_to_np(Sig_t[k]) + _to_np(Sig_t[k]).T)
#             L1 = np.linalg.cholesky(S1 + float(jitter) * np.eye(d))
#             Linv1 = np.linalg.inv(L1)
#             Lam1 = Linv1.T @ Linv1
#             eta1_tgt[k]  = Lam1 @ _to_np(mus_t[k])
#             eta2d_tgt[k] = -0.5 * np.diag(Lam1)
#     else:
#         for k in range(K):
#             if not present_t[k]:
#                 continue
#             v1 = np.clip(_to_np(vars_t[k]), 1e-12 if jitter == 0 else float(jitter), None)
#             eta1_tgt[k]  = _to_np(mus_t[k]) / v1
#             eta2d_tgt[k] = -0.5 / v1

#     # ---------------- intermediates ----------------
#     all_domains: List[DomainDataset] = []
#     for i in range(1, n_inter + 1):
#         _t_step = time.time()
#         t = i / (n_inter + 1)

#         # allocate class counts proportionally to source (masked by presence in both)
#         pi_mid = (counts_s.float() / max(1, total_s))
#         desired = (pi_mid * float(total_min)).clamp_min(0.0)
#         base = torch.floor(desired)
#         frac = desired - base
#         present = ((counts_s > 0) & (counts_t > 0)).float()
#         base *= present
#         frac *= present
#         n_alloc = int(base.sum().item())
#         rem = int(max(0, total_min - n_alloc))
#         if rem > 0:
#             k_take = min(rem, K)
#             if k_take > 0:
#                 _, idx = torch.topk(frac, k=k_take)
#                 add = torch.zeros_like(base); add[idx] = 1.0
#                 base = base + add
#         n_per_class = base.long().cpu().numpy().astype(np.int64)

#         mu_mid_full = np.full((K, d), np.nan, dtype=np.float64)
#         var_mid_full = np.full((K, d), np.nan, dtype=np.float64)
#         Sig_mid_full = np.full((K, d, d), np.nan, dtype=np.float64) if cov_type == "full" else None
#         step_eta1 = np.full((K, d), np.nan, dtype=np.float64)
#         step_eta2d = np.full((K, d), np.nan, dtype=np.float64)

#         Zm_list, Ym_list = [], []

#         for k_idx in range(K):
#             if counts_s[k_idx].item() == 0 or counts_t[k_idx].item() == 0:
#                 continue

#             n_k = int(counts_s[k_idx].item())
#             if n_k <= 0:
#                 continue

#             if cov_type == "full":
#                 mu_mid, Sig_mid, eta1_mid, eta2_mid = natural_interp_full(
#                     mus_s[k_idx], Sig_s[k_idx],
#                     mus_t[k_idx], Sig_t[k_idx],
#                     t, eps=jitter, debug=False,
#                     compare=False, scale_alpha=1e5, return_report=False
#                 )

#                 mu_mid_full[k_idx] = _to_np(mu_mid)
#                 Sig_mid_np = _to_np(Sig_mid)
#                 var_mid_full[k_idx] = np.clip(np.diag(Sig_mid_np), 1e-12, None)
#                 Sig_mid_full[k_idx] = Sig_mid_np
#                 step_eta1[k_idx] = _to_np(eta1_mid)
#                 step_eta2d[k_idx] = _to_np(torch.diag(eta2_mid))

#                 try:
#                     Zk = sample_full(mu_mid, Sig_mid, n_k)
#                 except RuntimeError:
#                     dK = Sig_mid.shape[0]
#                     Zk = sample_full(mu_mid, Sig_mid + (1e-6) * torch.eye(dK, device=Sig_mid.device), n_k)

#             else:
#                 mu_mid, var_mid, _, _ = natural_interp_diag(
#                     mus_s[k_idx], vars_s[k_idx],
#                     mus_t[k_idx], vars_t[k_idx],
#                     t
#                 )
#                 Zk = sample_diag(mu_mid, var_mid, n_k)

#                 mu_mid_full[k_idx] = _to_np(mu_mid)
#                 var_mid_full[k_idx] = _to_np(var_mid)
#                 prec = 1.0 / np.clip(var_mid_full[k_idx], 1e-12, None)
#                 step_eta1[k_idx] = prec * mu_mid_full[k_idx]
#                 step_eta2d[k_idx] = -0.5 * prec

#             # append samples
#             Yk = torch.full((n_k,), k_idx, device=device, dtype=torch.long)
#             Zm_list.append(Zk); Ym_list.append(Yk)

#         if not Zm_list:
#             continue

#         Zm = torch.cat(Zm_list, 0).cpu().float()
#         Ym = torch.cat(Ym_list, 0).cpu().long()
#         all_domains.append(DomainDataset(Zm, torch.ones(len(Ym)), Ym, Ym))

#         print(f"[NATURAL] Step {i}/{n_inter}: generated {len(Ym)} samples with d={Zm.shape[1]} in {time.time()-_t_step:.2f}s")

#         steps.append(float(t))
#         mu_list.append(mu_mid_full)
#         var_list.append(var_mid_full)
#         counts_list.append(n_per_class.astype(np.int64))
#         pi_list.append(pi_mid.cpu().numpy())
#         if cov_type == "full":
#             Sigma_list.append(Sig_mid_full)
#         eta1_steps.append(step_eta1)
#         eta2d_steps.append(step_eta2d)

#     # ---------------- wrap target & append target η --------------------------
#     try:
#         X_tgt_final = xt if torch.is_tensor(xt) else torch.as_tensor(xt)
#         X_tgt_final = X_tgt_final.cpu()
#         Y_tgt_final = dataset_t.targets if torch.is_tensor(dataset_t.targets) \
#                       else torch.as_tensor(dataset_t.targets, dtype=torch.long)
#         Y_tgt_final = Y_tgt_final.cpu().long()
#         Y_em_final = Yt_em.cpu().long()
#         all_domains.append(DomainDataset(X_tgt_final, torch.ones(len(Y_em_final)), Y_tgt_final, Y_em_final))
#     except Exception as e:
#         print(f"[NATURAL] Warning: failed to wrap target with targets_em ({e}); appending raw dataset.")
#         all_domains.append(dataset_t)

#     pi_t = (counts_t.float() / max(1, total_t)).cpu().numpy()
#     muT, varT = _pad_kd(mus_t, vars_t, present_t, K, d)
#     steps.append(1.0)
#     mu_list.append(muT)
#     var_list.append(varT)
#     counts_list.append(counts_t.cpu().numpy().astype(np.int64))
#     pi_list.append(pi_t)

#     if cov_type == "full":
#         SigT = _pad_kdd(Sig_t, present_t, K, d)
#         Sigma_list.append(SigT)

#     # append target endpoint η (precomputed above)
#     eta1_steps.append(eta1_tgt.copy())
#     eta2d_steps.append(eta2d_tgt.copy())

#     # ---------------- pack outputs ----------------
#     domain_params = {
#         "K": int(K),
#         "d": int(d),
#         "cov_type": cov_type,
#         "steps": np.asarray(steps, dtype=np.float64),
#         "mu": np.asarray(mu_list, dtype=np.float64),
#         "var": np.asarray(var_list, dtype=np.float64),
#         "counts": np.asarray(counts_list, dtype=np.int64),
#         "pi": np.asarray(pi_list, dtype=np.float64),
#         "present_source": present_s.astype(np.bool_),
#         "present_target": present_t.astype(np.bool_),
#         "eta1": np.asarray(eta1_steps, dtype=np.float64),
#         "eta2_diag": np.asarray(eta2d_steps, dtype=np.float64),
#     }
#     if cov_type == "full":
#         domain_params["Sigma"] = np.asarray(Sigma_list, dtype=np.float64)


#     # ---------------- diagnostics plot (optional) ----------------
#     if visualize:
#         # cls = int(diagnostic_class if diagnostic_class is not None else 0)
#         cls = K - 1
#         steps_np = np.asarray(steps, dtype=np.float64)

#         # pick the class row k=cls at each step
#         mu_steps   = domain_params["mu"]      # (S, K, d)
#         var_steps  = domain_params["var"]     # (S, K, d)
#         has_full   = (cov_type == "full")
#         if has_full:
#             Sig_steps = domain_params["Sigma"]  # (S, K, d, d)

#         # --- (1) total variance = trace(Σ) ---
#         if has_full:
#             # trace = sum of diagonal entries
#             trSigma = np.array([np.trace(Sig_steps[s, cls]) for s in range(len(steps_np))], dtype=np.float64)
#         else:
#             trSigma = np.array([np.nansum(var_steps[s, cls]) for s in range(len(steps_np))], dtype=np.float64)

#         # --- (2) mean norm ||μ||_2 ---
#         mu_norm = np.array([np.linalg.norm(mu_steps[s, cls], ord=2) for s in range(len(steps_np))], dtype=np.float64)

#         # --- (3) condition number κ(Σ) (full only) ---
#         if has_full:
#             def _kappa(S):
#                 # symmetric safeguard + small jitter for numerical stability
#                 Ssym = 0.5 * (S + S.T)
#                 w = np.linalg.eigvalsh(Ssym)
#                 w = np.clip(w, 1e-20, None)
#                 return float(w.max() / w.min())
#             kappa = np.array([_kappa(Sig_steps[s, cls]) for s in range(len(steps_np))], dtype=np.float64)

#         # ---- draw the figure ----
#         ncols = 3 if has_full else 2
#         fig, axes = plt.subplots(1, ncols, figsize=(16 if has_full else 12, 3.8))
#         if ncols == 2:
#             ax1, ax2 = axes
#         else:
#             ax1, ax2, ax3 = axes

#         ax1.plot(steps_np, trSigma, marker='o')
#         ax1.set_title("Trace(Σ)")
#         ax1.set_xlabel("Interpolation step (t)")
#         ax1.set_ylabel("Total Variance")
#         ax1.grid(True, ls='--', alpha=0.4)

#         ax2.plot(steps_np, mu_norm, marker='o')
#         ax2.set_title("‖μ‖₂")
#         ax2.set_xlabel("Interpolation step (t)")
#         ax2.set_ylabel("Norm")
#         ax2.grid(True, ls='--', alpha=0.4)

#         if has_full:
#             ax3.plot(steps_np, kappa, marker='o')
#             ax3.set_yscale('log')
#             ax3.set_title("Condition number (κ(Σ))")
#             ax3.set_xlabel("Interpolation step (t)")
#             ax3.set_ylabel("Condition Number (log scale)")
#             ax3.grid(True, ls='--', alpha=0.4)

#         fig.suptitle(f"The Effect of Natural Parameter Interpolation (Class {cls}, Dim={d})", y=1.05)
#         fig.tight_layout()

#         if save_path is not None:
#             out_png = os.path.join(save_path, f"natural_interp_diag_class{cls}_d{d}_gen{(steps_np.max()-1):.0f}.png")
#             os.makedirs(os.path.dirname(out_png), exist_ok=True)
#             plt.savefig(out_png, dpi=160, bbox_inches="tight")
#             print(f"[NATURAL] Saved diagnostics to {out_png}")
#         else:
#             plt.show()
#         plt.close(fig)


#     return all_domains, Yt_em.cpu().numpy(), domain_params




import os
from typing import Optional, Tuple, List
import numpy as np
import torch
import torch.nn as nn

# Assumed available in your codebase:
# - class_stats_diag(X, y, K) -> (mu_kd, var_kd, counts_k)
# - class_stats_full(X, y, K, reg=1e-6, ddof=0) -> (mu_kd, Sigma_kdd, counts_k)
# - DomainDataset(tensor_X, tensor_W, tensor_y_gt, tensor_y_em)

@torch.no_grad()
def generate_natural_domains_between(
    n_inter: int,
    dataset_s,
    dataset_t,
    *,
    plan=None,                         # unused here (kept for API parity)
    entry_cutoff: int = 0,            # unused here (kept for API parity)
    conf: float = 0.0,                # unused here (kept for API parity)
    source_model: Optional[torch.nn.Module] = None,   # unused (kept for API parity)
    pseudolabels: Optional[torch.Tensor] = None,      # optional fallback for Yt_em
    save_path: Optional[str] = None,
    cov_type: str = "full",           # "diag" or "full" (what you store; interpolation is natural in either case)
    reg: float = 1e-6,
    ddof: int = 0,
    jitter: float = 1e-6,
    diagnostic_class: Optional[int] = None,           # kept for API parity
    visualize: bool = False,                          # kept for API parity
    args=None,
    # New optional knobs (safe defaults):
    dtype_policy: str = "fp32",       # "fp32" (default) | "keep"
    compile_steps: bool = False,      # set True if torch.compile is available and you want to compile the inner step
    jit: bool = True                  # if True, applies SPD jitter in precision interpolation
):
    """
    Natural-parameter interpolation between source/target Gaussian class-conditionals.

    Core math:
      Λ_s = Σ_s^{-1},  Λ_t = Σ_t^{-1},       η1_s = Λ_s μ_s,  η1_t = Λ_t μ_t
      Λ(t)  = (1-t)Λ_s + tΛ_t,               η1(t) = (1-t)η1_s + tη1_t
      μ(t)  = Λ(t)^{-1} η1(t),               Σ(t)  = Λ(t)^{-1}

    Implementation notes:
      - Device-resident (GPU if available), single CPU conversion at the end.
      - Batched Cholesky for (M,d,d) over present classes; no per-class Python loops in LA.
      - Sampling uses precision Cholesky solves; vectorized by class counts.
      - API compatible with your previous function; extra args are optional and do not change defaults.
    """

    # ------------------ I/O, device, dtype ------------------
    xs, xt = dataset_s.data, dataset_t.data
    ys     = dataset_s.targets
    if xs.ndim > 2:
        flatten = nn.Flatten()
        xs, xt = flatten(xs), flatten(xt)

    if ys is None:
        raise ValueError("Source dataset must provide targets for class statistics.")

    # choose EM labels or pseudolabels
    if hasattr(dataset_t, "targets_em") and dataset_t.targets_em is not None:
        ytem = dataset_t.targets_em
    elif pseudolabels is not None:
        ytem = pseudolabels
    elif hasattr(dataset_t, "targets_pseudo") and dataset_t.targets_pseudo is not None:
        ytem = dataset_t.targets_pseudo
    else:
        raise ValueError("Target dataset must provide targets_em or pseudolabels.")

    # device and dtype normalization
    prefer_gpu = torch.cuda.is_available()
    device = xs.device if torch.is_tensor(xs) else torch.device("cuda" if prefer_gpu else "cpu")

    def _to(x):
        if torch.is_tensor(x):
            t = x
        else:
            t = torch.as_tensor(x)
        if dtype_policy == "fp32" and t.dtype in (torch.float16, torch.float32, torch.bfloat16):
            t = t.to(device=device, dtype=torch.float32, non_blocking=True)
        else:
            t = t.to(device=device, non_blocking=True)
        return t

    Zs = _to(xs)
    Zt = _to(xt)
    Ys = _to(ys).long().view(-1)
    Yt_em = _to(ytem).long().view(-1)

    mask_s = Ys >= 0
    mask_t = Yt_em >= 0
    if mask_s.sum() != Ys.numel():
        print(f"[generate_nat] Dropping {(~mask_s).sum().item()} source samples with negative labels.")
    if mask_t.sum() != Yt_em.numel():
        print(f"[generate_nat] Dropping {(~mask_t).sum().item()} target samples with negative labels.")
    Ys = Ys[mask_s]
    Yt_em = Yt_em[mask_t]
    Zs = Zs[mask_s]
    Zt = Zt[mask_t]

    if Ys.numel() == 0:
        return [], torch.empty(0, dtype=torch.long), {}
    if Yt_em.numel() == 0:
        print("[generate_nat] Warning: target endpoint has no valid samples after filtering; skipping generation.")
        return [], torch.empty(0, dtype=torch.long), {}

    K = int(torch.maximum(Ys.max(), Yt_em.max()).item()) + 1
    d = int(Zs.shape[1])
    I = torch.eye(d, device=device, dtype=Zs.dtype)

    # ------------------ class stats ------------------
    if cov_type == "diag":
        mus_s, vars_s, _ = class_stats_diag(Zs, Ys, K)              # (K,d), (K,d)
        mus_t, vars_t, _ = class_stats_diag(Zt, Yt_em, K)
        Sig_s = Sig_t = None
    else:
        mus_s, Sig_s, _ = class_stats_full(Zs, Ys, K, reg=reg, ddof=ddof)   # (K,d),(K,d,d)
        mus_t, Sig_t, _ = class_stats_full(Zt, Yt_em, K, reg=reg, ddof=ddof)
        vars_s = torch.diagonal(Sig_s, dim1=-2, dim2=-1)                    # (K,d)
        vars_t = torch.diagonal(Sig_t, dim1=-2, dim2=-1)

    counts_s = torch.bincount(Ys, minlength=K)                  # (K,)
    counts_t = torch.bincount(Yt_em, minlength=K)               # (K,)
    present_s = counts_s > 0
    present_t = counts_t > 0
    present_both = present_s & present_t
    idx_pres = present_both.nonzero(as_tuple=False).squeeze(-1)  # (M,)
    M = int(idx_pres.numel())

    total_s = int(counts_s.sum().item())
    total_t = int(counts_t.sum().item())
    total_min = int(torch.minimum(counts_s, counts_t).sum().item())
    if total_min <= 0:
        total_min = max(total_s, total_t)

    # ------------------ helpers (modular) ------------------

    def _prec_and_etas_from_cov(
        mu_kd: torch.Tensor,
        Sigma_kdd: Optional[torch.Tensor],
        var_kd: Optional[torch.Tensor],
        jitter_eps: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given endpoint (μ, Σ) OR (μ, var) per class, return:
          Λ_kdd (K,d,d),  η1_kd (K,d),  η2diag_kd (K,d).
        """
        if Sigma_kdd is not None:
            # symmetrize + jitter, batched Cholesky
            Sig_spd = 0.5 * (Sigma_kdd + Sigma_kdd.transpose(-1, -2)) + (jitter_eps * I)
            L = torch.linalg.cholesky(Sig_spd)                           # (K,d,d)
            eyeK = I.expand(K, d, d)
            Lam = torch.cholesky_solve(eyeK, L)                          # (K,d,d)
        else:
            Lam = torch.diag_embed((var_kd.clamp_min(jitter_eps)).reciprocal())  # (K,d,d)

        eta1 = torch.bmm(Lam, mu_kd.unsqueeze(-1)).squeeze(-1)           # (K,d)
        eta2d = -0.5 * torch.diagonal(Lam, dim1=-2, dim2=-1)             # (K,d)
        return Lam, eta1, eta2d

    def _allocate_counts_once() -> torch.Tensor:
        """
        Allocate per-class sample counts proportional to source priors,
        masked by presence in both domains; fill remainder by largest frac.
        """
        pi_src = counts_s.float() / max(1, total_s)
        desired = (pi_src * float(total_min)).clamp_min(0.0)
        base = torch.floor(desired)
        frac = desired - base
        base = (base * present_both).long()
        frac = frac * present_both
        rem = int(max(0, total_min - int(base.sum().item())))
        if rem > 0:
            k_take = min(rem, K)
            if k_take > 0:
                _, idx = torch.topk(frac, k=k_take)
                add = torch.zeros_like(base)
                add[idx] = 1
                base = (base + add).long()
        return base  # (K,)

    def _pad_kd(mu_like: torch.Tensor, var_like: torch.Tensor, mask_bool: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out_mu = torch.full((K, d), float('nan'), device=device, dtype=mu_like.dtype)
        out_vr = torch.full((K, d), float('nan'), device=device, dtype=var_like.dtype)
        out_mu[mask_bool] = mu_like[mask_bool]
        out_vr[mask_bool] = var_like[mask_bool]
        return out_mu, out_vr

    def _pad_kdd(sig_like: torch.Tensor, mask_bool: torch.Tensor) -> torch.Tensor:
        out = torch.full((K, d, d), float('nan'), device=device, dtype=sig_like.dtype)
        out[mask_bool] = sig_like[mask_bool]
        return out

    def _batched_natural_step(
        t: float,
        Lam_s_m: torch.Tensor,     # (M,d,d)
        Lam_t_m: torch.Tensor,     # (M,d,d)
        eta1_s_m: torch.Tensor,    # (M,d)
        eta1_t_m: torch.Tensor,    # (M,d)
        jitter_eps: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        For present classes (M of them), compute μ(t), Σ(t), η1(t), η2diag(t)
        using natural-parameter interpolation, batched.
        """
        # Interpolate precision and eta1; SPD symmetrization + jitter
        Lam_t = (1.0 - t) * Lam_s_m + t * Lam_t_m                          # (M,d,d)
        Lam_t = 0.5 * (Lam_t + Lam_t.transpose(-1, -2))
        if jit:
            Lam_t = Lam_t + (jitter_eps * I.expand_as(Lam_t))
        L = torch.linalg.cholesky(Lam_t)                                    # (M,d,d)

        eta1_t = (1.0 - t) * eta1_s_m + t * eta1_t_m                        # (M,d)
        mu_m = torch.cholesky_solve(eta1_t.unsqueeze(-1), L).squeeze(-1)    # (M,d)
        Sig_m = torch.cholesky_solve(I.expand_as(Lam_t), L)                 # (M,d,d)

        # natural params for logging
        eta1_m = torch.einsum("mij,mj->mi", Lam_t, mu_m)                    # (M,d)
        eta2d_m = -0.5 * torch.diagonal(Lam_t, dim1=-2, dim2=-1)            # (M,d)
        return mu_m, Sig_m, eta1_m, eta2d_m

    def _sample_from_precision_cholesky(
        L_m: torch.Tensor,          # (M,d,d), Cholesky of precision Λ(t)
        mu_m_fullK: torch.Tensor,   # (K,d), μ(t) with NaNs on absent classes
        n_per_class: torch.Tensor   # (K,), counts per class
    ):
        """
        Vectorized sampling: for each class k with nk>0, draw z ~ N(μ_k, Σ_k)
        using y = solve(L_k^T, ε^T)^T with ε ~ N(0,I); Σ = Λ^{-1}.
        Returns tensors on CPU for DomainDataset wrapper.
        """
        nk = n_per_class
        if int(nk.sum().item()) == 0:
            return None

        # map class index -> row m in the present-batch (or -1 if absent)
        map_k_to_m = torch.full((K,), -1, device=device, dtype=torch.long)
        map_k_to_m[idx_pres] = torch.arange(M, device=device, dtype=torch.long)

        Z_chunks, Y_chunks = [], []
        active = (nk > 0).nonzero(as_tuple=False).squeeze(-1).tolist()
        for k_idx in active:
            m_idx = int(map_k_to_m[k_idx].item())
            if m_idx < 0:
                continue  # should not happen with nk>0
            n_k = int(nk[k_idx].item())
            Lk  = L_m[m_idx]                            # (d,d)
            muk = mu_m_fullK[k_idx]                     # (d,)
            eps = torch.randn(n_k, d, device=device, dtype=muk.dtype)
            yT  = torch.linalg.solve(Lk.transpose(-1, -2), eps.T)
            zk  = yT.T + muk
            Z_chunks.append(zk)
            Y_chunks.append(torch.full((n_k,), k_idx, device=device, dtype=torch.long))
        if not Z_chunks:
            return None
        Zm = torch.cat(Z_chunks, 0).float().cpu()
        Ym = torch.cat(Y_chunks, 0).long().cpu()
        Wm = torch.ones(len(Ym))
        return DomainDataset(Zm, Wm, Ym, Ym)

    # Optionally compile the batched step (PyTorch 2.x). Safe no-op if unavailable.
    if compile_steps and hasattr(torch, "compile"):
        _batched_natural_step = torch.compile(_batched_natural_step, mode="reduce-overhead", fullgraph=False)

    # ------------------ endpoints → precisions & etas ------------------
    Lam_s, eta1_s, eta2d_s = _prec_and_etas_from_cov(mus_s, Sig_s, vars_s if cov_type == "diag" else None, jitter)
    Lam_t, eta1_t, eta2d_t = _prec_and_etas_from_cov(mus_t, Sig_t, vars_t if cov_type == "diag" else None, jitter)

    # Slice to present classes once
    Lam_s_m  = Lam_s.index_select(0, idx_pres)      # (M,d,d)
    Lam_t_m  = Lam_t.index_select(0, idx_pres)      # (M,d,d)
    eta1_s_m = eta1_s.index_select(0, idx_pres)     # (M,d)
    eta1_t_m = eta1_t.index_select(0, idx_pres)     # (M,d)

    # Per-class sample allocation (constant across steps)
    n_per_class = _allocate_counts_once()           # (K,)

    # ------------------ containers (on device; CPU once at end) ------------------
    steps: List[float] = []
    mu_list: List[torch.Tensor] = []
    var_list: List[torch.Tensor] = []
    counts_list: List[torch.Tensor] = []
    pi_list: List[torch.Tensor] = []
    Sigma_list: Optional[List[torch.Tensor]] = [] if (cov_type == "full") else None
    eta1_steps: List[torch.Tensor] = []
    eta2d_steps: List[torch.Tensor] = []

    # source endpoint (t=0)
    steps.append(0.0)
    mu0, var0 = _pad_kd(mus_s, vars_s, present_s)
    mu_list.append(mu0)
    var_list.append(var0)
    counts_list.append(counts_s.clone())
    pi_list.append(counts_s.float() / max(1, total_s))
    eta1_steps.append(eta1_s.clone())
    eta2d_steps.append(eta2d_s.clone())
    if cov_type == "full":
        Sigma_list.append(_pad_kdd(Sig_s, present_s))

    # ------------------ intermediates ------------------
    all_domains: List = []
    I_m = I.expand(M, d, d)  # used only to match shapes inside cholesky_solve when compiling

    for i in range(1, n_inter + 1):
        t = i / (n_inter + 1)
        steps.append(float(t))

        # Batched natural-parameter step over present classes
        mu_m, Sig_m, eta1_m, eta2d_m = _batched_natural_step(t, Lam_s_m, Lam_t_m, eta1_s_m, eta1_t_m, jitter)

        # Scatter back to K with NaNs on absent classes
        mu_full  = torch.full((K, d), float('nan'), device=device, dtype=mu_m.dtype)
        var_full = torch.full((K, d), float('nan'), device=device, dtype=mu_m.dtype)
        mu_full[idx_pres]  = mu_m
        var_full[idx_pres] = torch.diagonal(Sig_m, dim1=-2, dim2=-1)

        eta1_full  = torch.full((K, d), float('nan'), device=device, dtype=mu_m.dtype)
        eta2d_full = torch.full((K, d), float('nan'), device=device, dtype=mu_m.dtype)
        eta1_full[idx_pres]  = eta1_m
        eta2d_full[idx_pres] = eta2d_m

        mu_list.append(mu_full)
        var_list.append(var_full)
        counts_list.append(n_per_class.clone())
        pi_list.append((counts_s.float() / max(1, total_s)))   # same prior as source

        if cov_type == "full":
            Sig_full = torch.full((K, d, d), float('nan'), device=device, dtype=mu_m.dtype)
            Sig_full[idx_pres] = Sig_m
            Sigma_list.append(Sig_full)

        eta1_steps.append(eta1_full)
        eta2d_steps.append(eta2d_full)

        # Sampling (vectorized by per-class counts) — uses precision cholesky from this step
        # Reuse the already computed cholesky factor by recomputing quickly here (kept local for clarity)
        Lam_t_cur = (1.0 - t) * Lam_s_m + t * Lam_t_m
        Lam_t_cur = 0.5 * (Lam_t_cur + Lam_t_cur.transpose(-1, -2))
        if jit:
            Lam_t_cur = Lam_t_cur + (jitter * I_m)
        L_cur = torch.linalg.cholesky(Lam_t_cur)
        dom = _sample_from_precision_cholesky(L_cur, mu_full, n_per_class)
        if dom is not None:
            all_domains.append(dom)

    # target endpoint (t=1)
    steps.append(1.0)
    muT, varT = _pad_kd(mus_t, vars_t, present_t)
    mu_list.append(muT)
    var_list.append(varT)
    counts_list.append(counts_t.clone())
    pi_list.append((counts_t.float() / max(1, total_t)))
    eta1_steps.append(eta1_t.clone())
    eta2d_steps.append(eta2d_t.clone())
    if cov_type == "full":
        Sigma_list.append(_pad_kdd(Sig_t, present_t))

    # wrap the provided target dataset (mirror prior behavior; move payloads to CPU for DomainDataset)
    X_tgt_final = (xt if torch.is_tensor(xt) else torch.as_tensor(xt)).cpu()
    Y_tgt_final = (dataset_t.targets if torch.is_tensor(dataset_t.targets)
                   else torch.as_tensor(dataset_t.targets, dtype=torch.long)).cpu().long()
    Y_em_final  = Yt_em.cpu().long()
    all_domains.append(DomainDataset(X_tgt_final, torch.ones(len(Y_em_final)), Y_tgt_final, Y_em_final))

    # ------------------ pack outputs (single CPU conversion) ------------------
    def _stack_np(lst: List[torch.Tensor]) -> np.ndarray:
        return torch.stack(lst, dim=0).detach().cpu().numpy()

    domain_params = {
        "K": int(K),
        "d": int(d),
        "cov_type": cov_type,
        "steps": np.asarray(steps, dtype=np.float64),                                      # (S,)
        "mu":   _stack_np(mu_list),                                                        # (S,K,d)
        "var":  _stack_np(var_list),                                                       # (S,K,d)
        "counts": torch.stack(counts_list, dim=0).cpu().numpy().astype(np.int64),         # (S,K)
        "pi":     torch.stack(pi_list,    dim=0).cpu().numpy().astype(np.float64),        # (S,K)
        "present_source": present_s.detach().cpu().numpy().astype(np.bool_),
        "present_target": present_t.detach().cpu().numpy().astype(np.bool_),
        "eta1":      _stack_np(eta1_steps),                                                # (S,K,d)
        "eta2_diag": _stack_np(eta2d_steps),                                               # (S,K,d)
    }
    if cov_type == "full":
        domain_params["Sigma"] = _stack_np(Sigma_list)                                     # (S,K,d,d)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez_compressed(save_path, **domain_params)

    return all_domains, Yt_em.detach().cpu().numpy(), domain_params


def inv_via_scaling(S: torch.Tensor, alpha= None):
    """
    Returns Λ = S^{-1} using scale→inverse→unscale.
    If alpha is None, choose alpha = trace(S)/d (or max eigenvalue for tighter conditioning).
    """
    S = 0.5 * (S + S.transpose(-2, -1))  # symmetrize
    d = S.shape[-1]
    if alpha is None:
        alpha = (torch.trace(S) / d).clamp_min(torch.finfo(S.dtype).eps).item()
    S_tilde = S * alpha
    # Factorize scaled matrix (better conditioning)
    L = torch.linalg.cholesky(S_tilde)         # if this still fails, fall back to tiny jitter or eig-clip
    Lam_tilde = torch.cholesky_inverse(L)      # = (α S)^{-1}
    Lam = Lam_tilde * alpha                    # = α * (α S)^{-1} = S^{-1}
    return Lam



@torch.no_grad()
def generate_fr_domains_between_optimized(
    n_inter: int,
    dataset_s,
    dataset_t,
    *,
    cov_type: str = "diag",     # 'diag' | 'full'
    reg: float = 1e-6,
    ddof: int = 0,
    jitter: float = 1e-6,
    save_path: Optional[str] = None,
    args=None
):
    """
    Generate intermediate domains along the *true* Fisher–Rao geodesic between
    class-conditional Gaussians:
      Full-cov:  Σ(t) = Σ_s^{1/2} (Σ_s^{-1/2} Σ_t Σ_s^{-1/2})^t Σ_s^{1/2}
                  μ(t) = μ_s + M(t) (μ_t - μ_s),   M(t) = Σ_s^{1/2} V diag(w(λ,t)) V^T Σ_s^{-1/2},
                  w(λ,t) = (λ^t - 1) / (λ - 1), with w≈t when λ≈1.
      Diag:      var_i(t) = var_{s,i}^{1-t} var_{t,i}^t
                 μ_i(t)   = μ_{s,i} + w(λ_i,t) (μ_{t,i} - μ_{s,i}),  λ_i = var_{t,i}/var_{s,i}.
    The function also samples from each intermediate Gaussian per class.

    Returns:
      all_domains   : List[DomainDataset]
      target_em     : np.ndarray of target EM labels
      domain_params : dict with per-step parameters (source, intermediates, target)
    """
    assert cov_type in {"diag", "full"}
    print("---- FR domains (true geodesic) ----")

    # ---------- 0) I/O tensors, shapes, device ----------
    xs, xt = dataset_s.data, dataset_t.data
    ys     = dataset_s.targets
    if xs.ndim > 2:
        flatten = torch.nn.Flatten()
        xs, xt = flatten(xs), flatten(xt)

    device = xs.device if torch.is_tensor(xs) else torch.device("cpu")
    Xs = xs if torch.is_tensor(xs) else torch.as_tensor(xs, device=device)
    Xt = xt if torch.is_tensor(xt) else torch.as_tensor(xt, device=device)
    Ys = ys if torch.is_tensor(ys) else torch.as_tensor(ys, device=device, dtype=torch.long)
    # EM labels are required
    Yt = dataset_t.targets_em if torch.is_tensor(dataset_t.targets_em) else torch.as_tensor(dataset_t.targets_em, device=device, dtype=torch.long)

    if Ys.numel() == 0:
        return [], np.empty((0,), dtype=np.int64), {}

    K = int(max(Ys.max(), Yt.max()).item()) + 1
    d = int(Xs.shape[1])

    # ---------- 1) Endpoints stats (once) ----------
    if cov_type == "diag":
        mu_s, var_s, _ = class_stats_diag(Xs, Ys, K)              # (K,d)
        mu_t, var_t, _ = class_stats_diag(Xt, Yt, K)
        Sig_s = Sig_t = None
    else:
        mu_s, Sig_s, _ = class_stats_full(Xs, Ys, K, reg=reg, ddof=ddof)    # (K,d),(K,d,d)
        mu_t, Sig_t, _ = class_stats_full(Xt, Yt, K, reg=reg, ddof=ddof)
        # Keep diagonals for logging/compat
        var_s = torch.diagonal(Sig_s, dim1=-2, dim2=-1)  # (K,d)
        var_t = torch.diagonal(Sig_t, dim1=-2, dim2=-1)

    counts_s = torch.bincount(Ys, minlength=K)
    counts_t = torch.bincount(Yt, minlength=K)
    total_s  = int(counts_s.sum().item())
    total_t  = int(counts_t.sum().item())
    present_s = (counts_s > 0)
    present_t = (counts_t > 0)
    present_both = (present_s & present_t)

    # total_min used below → constant across steps
    total_min = int(torch.minimum(counts_s, counts_t).sum().item())
    if total_min <= 0:
        total_min = max(total_s, total_t)

    # class allocation used at all steps (use source priors, masked by present_both)
    pi_src = counts_s.float() / max(1, total_s)
    desired = (pi_src * float(total_min)).clamp_min(0.0)
    base    = torch.floor(desired)
    frac    = desired - base
    base    = (base * present_both).long()
    frac    = frac * present_both
    rem     = int(max(0, total_min - int(base.sum().item())))
    if rem > 0:
        k_take = min(rem, K)
        if k_take > 0:
            _, idx = torch.topk(frac, k=k_take)
            base.index_add_(0, idx, torch.ones_like(idx, dtype=base.dtype, device=base.device))
    n_per_class = base.long()                                     # (K,)

    # ---------- 2) Preallocate outputs (S = n_inter+2; src + inter + tgt) ----------
    S = n_inter + 2
    steps   = torch.empty(S, dtype=torch.float64)
    mu_out  = torch.full((S, K, d), float("nan"), dtype=torch.float64)
    var_out = torch.full((S, K, d), float("nan"), dtype=torch.float64)
    cnt_out = torch.empty((S, K), dtype=torch.int64)
    pi_out  = torch.empty((S, K), dtype=torch.float64)
    eta1_out  = torch.full((S, K, d), float("nan"), dtype=torch.float64)
    eta2d_out = torch.full((S, K, d), float("nan"), dtype=torch.float64)
    if cov_type == "full":
        Sig_out = torch.full((S, K, d, d), float("nan"), dtype=torch.float64)
    else:
        Sig_out = None

    # ---------- helpers ----------
    def _eta_diag(mu_kd: torch.Tensor, var_kd: torch.Tensor, eps=1e-12):
        var = var_kd.clamp_min(eps)
        prec = 1.0 / var
        eta1 = prec * mu_kd
        eta2d = -0.5 * prec
        return eta1, eta2d

    def _eta_full(mu_kd: torch.Tensor, Sig_kdd: torch.Tensor, eps=1e-12):
        Sig_sym = 0.5*(Sig_kdd + Sig_kdd.transpose(-2, -1)) + eps*torch.eye(d, device=Sig_kdd.device)[None]
        L = torch.linalg.cholesky(Sig_sym)                   # (K,d,d)
        Lam = torch.cholesky_inverse(L)                      # (K,d,d)
        eta1 = torch.einsum("kdd,kd->kd", Lam, mu_kd)        # (K,d)
        eta2d = -0.5 * Lam.diagonal(dim1=-2, dim2=-1)       # (K,d)
        return eta1, eta2d

    # FR mean/cov (diag) — stable
    def _fr_step_diag(mu_s, var_s, mu_t, var_t, t, eps=1e-8):
        mu_s64  = mu_s.to(torch.float64)
        mu_t64  = mu_t.to(torch.float64)
        var_s64 = var_s.to(torch.float64).clamp_min(eps)
        var_t64 = var_t.to(torch.float64).clamp_min(eps)
        r = var_t64 / var_s64
        var_mid64 = var_s64 * r.pow(t)
        num = r.pow(t) - 1.0
        den = r - 1.0
        w   = torch.where(den.abs() < 1e-6, torch.full_like(den, t), num/den)
        mu_mid64 = mu_s64 + w * (mu_t64 - mu_s64)
        out_dtype = mu_s.dtype if mu_s.dtype == mu_t.dtype else torch.float32
        return mu_mid64.to(out_dtype), var_mid64.to(out_dtype)

    # ---------- 3) Cache FR spectral pieces for full cov ----------
    cache = None
    if cov_type == "full":
        # Per class: Σs^{±1/2}, and eig of A = Σs^{-1/2} Σt Σs^{-1/2} = V diag(λ) V^T
        cache = []
        I = torch.eye(d, device=device, dtype=Sig_s.dtype)
        # Symmetrize + jitter once
        Sig_s_spd = 0.5*(Sig_s + Sig_s.transpose(-1,-2)) + jitter*I
        Sig_t_spd = 0.5*(Sig_t + Sig_t.transpose(-1,-2)) + jitter*I

        # Work in float64 for spectral accuracy, then cast back
        to64 = lambda x: x.to(torch.float64)
        back = lambda x: x.to(Sig_s.dtype)

        Sig_s64 = to64(Sig_s_spd)
        Sig_t64 = to64(Sig_t_spd)

        for k in range(K):
            if not present_both[k]:
                cache.append(None)
                continue
            ws, Vs = torch.linalg.eigh(Sig_s64[k])
            ws = ws.clamp_min(jitter)
            sqrt_ws     = ws.sqrt()
            inv_sqrt_ws = 1.0 / sqrt_ws
            S_half  = (Vs * sqrt_ws)     @ Vs.mH          # Σs^{1/2}
            S_mhalf = (Vs * inv_sqrt_ws) @ Vs.mH          # Σs^{-1/2}

            A  = S_mhalf @ Sig_t64[k] @ S_mhalf
            A  = 0.5*(A + A.mH) + jitter*torch.eye(d, device=A.device, dtype=A.dtype)
            la, Va = torch.linalg.eigh(A)
            la = la.clamp_min(jitter)

            cache.append((back(S_half), back(S_mhalf), back(Va), back(la)))

    # ---------- 4) Fill source (s=0) ----------
    steps[0] = 0.0
    mu_out[0]  = mu_s.double().cpu()
    var_out[0] = var_s.double().cpu()
    cnt_out[0] = counts_s.cpu()
    pi_out[0]  = (counts_s.double() / max(1, total_s)).cpu()
    if cov_type == "full":
        Sig_out[0] = Sig_s.double().cpu()
        e1, e2d = _eta_full(mu_s, Sig_s)
    else:
        e1, e2d = _eta_diag(mu_s, var_s)
    eta1_out[0]  = e1.double().cpu()
    eta2d_out[0] = e2d.double().cpu()

    all_domains = []

    # ---------- 5) Intermediates (s=1..n_inter) ----------
    for i in range(1, n_inter+1):
        t = i / (n_inter + 1)
        steps[i]   = t

        if cov_type == "diag":
            # TRUE FR mean + var
            mu_mid, var_mid = _fr_step_diag(mu_s, var_s, mu_t, var_t, t, eps=jitter)
            e1, e2d = _eta_diag(mu_mid, var_mid)

            mu_out[i]  = mu_mid.double().cpu()
            var_out[i] = var_mid.double().cpu()
            cnt_out[i] = n_per_class.cpu()
            pi_out[i]  = (counts_s.double() / max(1, total_s)).cpu()
            eta1_out[i]  = e1.double().cpu()
            eta2d_out[i] = e2d.double().cpu()

            # sampling (vectorized per class on CPU for portability)
            n_k = n_per_class.cpu().numpy()
            if n_k.sum() > 0:
                zs, ys_cls = [], []
                for k in range(K):
                    nk = int(n_k[k])
                    if nk <= 0: continue
                    m  = mu_mid[k].cpu()
                    s2 = var_mid[k].clamp_min(jitter).cpu()
                    z  = torch.randn(nk, d) * s2.sqrt() + m
                    zs.append(z); ys_cls.append(torch.full((nk,), k, dtype=torch.long))
                if zs:
                    Zm = torch.cat(zs, 0).float()
                    Ym = torch.cat(ys_cls, 0).long()
                    Wm = torch.ones(len(Ym))
                    all_domains.append(DomainDataset(Zm, Wm, Ym, Ym))

        else:
            # TRUE FR mean + covariance using cached spectra
            Sig_mid = torch.empty_like(Sig_s)
            mu_mid  = torch.empty_like(mu_s)
            I = torch.eye(d, device=device, dtype=Sig_s.dtype)

            for k in range(K):
                if not present_both[k]:
                    # keep NaNs for absent classes
                    Sig_mid[k].fill_(float("nan"))
                    mu_mid[k].fill_(float("nan"))
                    continue
                S_half, S_mhalf, Va, la = cache[k]

                # Σ(t) = Σs^{1/2} (V diag(λ^t) V^T) Σs^{1/2}
                Lt = (Va * (la ** t)) @ Va.mH
                Sk = S_half @ Lt @ S_half
                Sk = 0.5*(Sk + Sk.mH) + jitter*I
                Sig_mid[k] = Sk

                # μ(t) = μs + Σs^{1/2} V diag(w(λ,t)) V^T Σs^{-1/2} (μt - μs)
                num = (la ** t) - 1.0
                den = la - 1.0
                w   = torch.where(den.abs() < 1e-6, torch.full_like(den, t), num/den)
                M   = S_half @ (Va * w) @ Va.mH @ S_mhalf
                mu_mid[k] = mu_s[k] + (M @ (mu_t[k] - mu_s[k]))

            # natural params
            e1, e2d = _eta_full(mu_mid, Sig_mid)

            mu_out[i]   = mu_mid.double().cpu()
            var_out[i]  = torch.diagonal(Sig_mid, dim1=-2, dim2=-1).double().cpu()
            Sig_out[i]  = Sig_mid.double().cpu()
            cnt_out[i]  = n_per_class.cpu()
            pi_out[i]   = (counts_s.double() / max(1, total_s)).cpu()
            eta1_out[i]  = e1.double().cpu()
            eta2d_out[i] = e2d.double().cpu()

            # sampling (per class via Cholesky on CPU for portability)
            n_k = n_per_class.cpu().numpy()
            if n_k.sum() > 0:
                zs, ys_cls = [], []
                for k in range(K):
                    nk = int(n_k[k])
                    if nk <= 0: continue
                    Sk = Sig_mid[k].cpu()
                    mk = mu_mid[k].cpu()
                    Lk = torch.linalg.cholesky(Sk)
                    z  = (torch.randn(nk, d) @ Lk.T) + mk
                    zs.append(z); ys_cls.append(torch.full((nk,), k, dtype=torch.long))
                if zs:
                    Zm = torch.cat(zs, 0).float()
                    Ym = torch.cat(ys_cls, 0).long()
                    Wm = torch.ones(len(Ym))
                    all_domains.append(DomainDataset(Zm, Wm, Ym, Ym))

    # ---------- 6) Target (s=S-1) ----------
    steps[-1]  = 1.0
    mu_out[-1] = mu_t.double().cpu()
    var_out[-1]= (var_t if cov_type == "diag" else torch.diagonal(Sig_t, dim1=-2, dim2=-1)).double().cpu()
    cnt_out[-1]= counts_t.cpu()
    pi_out[-1] = (counts_t.double() / max(1, total_t)).cpu()
    if cov_type == "full":
        Sig_out[-1] = Sig_t.double().cpu()
        e1, e2d     = _eta_full(mu_t, Sig_t)
    else:
        e1, e2d     = _eta_diag(mu_t, var_t)
    eta1_out[-1]  = e1.double().cpu()
    eta2d_out[-1] = e2d.double().cpu()

    # Append target domain wrapper (targets + targets_em)
    X_tgt = Xt.cpu()
    Y_gt  = (dataset_t.targets if torch.is_tensor(dataset_t.targets)
             else torch.as_tensor(dataset_t.targets, dtype=torch.long)).cpu()
    Y_em  = Yt.cpu()
    all_domains.append(DomainDataset(X_tgt, torch.ones(len(Y_em)), Y_gt, Y_em))

    # ---------- 7) Pack domain_params (NumPy once) ----------
    domain_params = {
        "K": int(K), "d": int(d), "cov_type": cov_type,
        "steps": steps.numpy(),
        "mu":    mu_out.numpy(),
        "var":   var_out.numpy(),
        "counts": cnt_out.numpy(),
        "pi":      pi_out.numpy(),
        "present_source": present_s.cpu().numpy().astype(bool),
        "present_target": present_t.cpu().numpy().astype(bool),
        "eta1":      eta1_out.numpy(),
        "eta2_diag": eta2d_out.numpy(),
    }
    if cov_type == "full":
        domain_params["Sigma"] = Sig_out.numpy()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez_compressed(save_path, **domain_params)
        print(f"[FR] Saved domain parameters -> {save_path}")

    return all_domains, Yt.cpu().numpy(), domain_params



@torch.no_grad()
def generate_fr_domains_between_optimized(
    n_inter: int,
    dataset_s,
    dataset_t,
    *,
    cov_type: str = "diag",     # 'diag' | 'full'
    reg: float = 1e-6,
    ddof: int = 0,
    jitter: float = 1e-6,
    save_path: Optional[str] = None,
    args=None
):
    """
    Fisher–Rao (Bures) interpolation between class-conditional Gaussians.

    - Numerically robust: eigenvalue flooring -> SPD, adaptive jitter, Cholesky fallback.
    - Batch-safe linear algebra (K x d x d).
    - Keeps tensors on device; single CPU hop when packaging outputs.
    """


    # ------------------------- helpers (self-contained) -------------------------
    def _sym(A: torch.Tensor) -> torch.Tensor:
        return 0.5 * (A + A.transpose(-1, -2))

    def _project_spd(A: torch.Tensor,
                     rel_floor: float = 1e-8,
                     abs_floor: float = 1e-12) -> torch.Tensor:
        """
        Eig-project A to SPD: V diag(clamp(w, λ_min)) V^T with
        λ_min = max(abs_floor, rel_floor * trace(A)/d). Batch-safe.
        """
        A = _sym(A)
        d = A.shape[-1]
        w, V = torch.linalg.eigh(A)                       # (.., d), (.., d, d)
        # compute average eigenvalue per batch item to scale the floor
        avg = (w.sum(dim=-1, keepdim=True) / d).clamp_min(torch.finfo(w.dtype).eps)
        lam_min = torch.maximum(
            torch.as_tensor(abs_floor, dtype=w.dtype, device=w.device),
            rel_floor * avg
        )
        w = torch.maximum(w, lam_min)
        return (V * w) @ V.transpose(-1, -2)

    def _chol_inv_spd(S: torch.Tensor,
                      base_rel: float = 1e-12,
                      abs_floor: float = 1e-12,
                      max_tries: int = 5):
        """
        Compute (S^{-1}, L) robustly for batched SPD S (K x d x d).
        Try Cholesky on S + jitter*I with exponentially increasing jitter.
        On failure, eigen-project to SPD and try once more.
        """
        S = _sym(S)
        d = S.shape[-1]
        I = torch.eye(d, device=S.device, dtype=S.dtype)
        # ---- FIX: batch-safe scale estimate ----
        # Use mean of all diagonal entries across the whole batch to form a scalar jitter scale.
        diag_mean = torch.diagonal(S, dim1=-2, dim2=-1).mean()     # <-- FIX: batch-safe diag
        scale = float(diag_mean.abs().item())
        if not math.isfinite(scale) or scale <= 0:
            scale = 1.0
        jitter_here = max(abs_floor, base_rel * scale)
        for _ in range(max_tries):
            try:
                L = torch.linalg.cholesky(S + jitter_here * I)     # batched Cholesky
                Lam = torch.cholesky_inverse(L)                    # batched inverse
                return Lam, L
            except RuntimeError:
                jitter_here *= 10.0
        # Fallback: project and do one more Cholesky
        Sp = _project_spd(S, rel_floor=1e-8, abs_floor=max(abs_floor, jitter))
        L = torch.linalg.cholesky(Sp)
        Lam = torch.cholesky_inverse(L)
        return Lam, L

    def _chol_sqrt_spd(S: torch.Tensor,
                       base_rel: float = 1e-12,
                       abs_floor: float = 1e-12,
                       max_tries: int = 5):
        """
        Robust matrix square-root for SPD S (d x d).
        Returns (F, used_eig). If Cholesky succeeds, F=L (lower-tri).
        Else uses eigen sqrt on SPD-projected matrix.
        """
        S = _sym(S)
        d = S.shape[-1]
        I = torch.eye(d, device=S.device, dtype=S.dtype)
        # ---- FIX: batch-safe scale estimate for 2D S (single class) ----
        scale = float(torch.diagonal(S, dim1=-2, dim2=-1).mean().abs().item())  # <-- FIX
        if not math.isfinite(scale) or scale <= 0:
            scale = 1.0
        jitter_here = max(abs_floor, base_rel * scale)
        for _ in range(max_tries):
            try:
                L = torch.linalg.cholesky(S + jitter_here * I)
                return L, False
            except RuntimeError:
                jitter_here *= 10.0
        # Fallback: eigen sqrt on SPD-projected matrix
        Sp = _project_spd(S, rel_floor=1e-8, abs_floor=max(abs_floor, jitter))
        w, V = torch.linalg.eigh(Sp)
        F = (V * w.clamp_min(0).sqrt()) @ V.transpose(-1, -2)
        return F, True

    def _eta_diag(mu_kd: torch.Tensor, var_kd: torch.Tensor, eps: float = 1e-12):
        var = var_kd.clamp_min(eps)
        prec = var.reciprocal()
        eta1 = prec * mu_kd
        eta2d = -0.5 * prec
        return eta1, eta2d

    # ------------------------- 0) I/O tensors, device ---------------------------
    xs, xt = dataset_s.data, dataset_t.data
    ys = dataset_s.targets
    if xs.ndim > 2:
        flatten = nn.Flatten()
        xs, xt = flatten(xs), flatten(xt)

    device = xs.device if torch.is_tensor(xs) else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xs = xs if torch.is_tensor(xs) else torch.as_tensor(xs, device=device)
    Xt = xt if torch.is_tensor(xt) else torch.as_tensor(xt, device=device)
    Ys = ys if torch.is_tensor(ys) else torch.as_tensor(ys, device=device, dtype=torch.long)
    Yt = dataset_t.targets_em if torch.is_tensor(dataset_t.targets_em) else torch.as_tensor(dataset_t.targets_em, device=device, dtype=torch.long)

    if Ys.numel() == 0:
        return [], np.empty((0,), dtype=np.int64), {}

    fp_dtype = torch.float32 if Xs.dtype in (torch.float16, torch.float32, torch.bfloat16) else Xs.dtype
    Xs = Xs.to(device=device, dtype=fp_dtype, non_blocking=True)
    Xt = Xt.to(device=device, dtype=fp_dtype, non_blocking=True)
    Ys = Ys.to(device=device, dtype=torch.long, non_blocking=True).view(-1)
    Yt = Yt.to(device=device, dtype=torch.long, non_blocking=True).view(-1)

    mask_s = Ys >= 0
    mask_t = Yt >= 0
    if mask_s.sum() != Ys.numel():
        breakpoint()
        print(f"[generate_fr] Dropping {(~mask_s).sum().item()} source samples with negative labels.")
    if mask_t.sum() != Yt.numel():
        breakpoint()
        print(f"[generate_fr] Dropping {(~mask_t).sum().item()} target samples with negative labels.")
    Ys = Ys[mask_s]
    Yt = Yt[mask_t]
    Xs = Xs[mask_s]
    Xt = Xt[mask_t]

    if Ys.numel() == 0 or Yt.numel() == 0:
        print("[generate_fr] Warning: one endpoint has zero valid samples after filtering; skipping generation.")
        return [], np.empty((0,), dtype=np.int64), {}

    K = int(torch.max(torch.stack([Ys.max(), Yt.max()])).item()) + 1
    d = int(Xs.shape[1])

    # ------------------------- 1) Endpoints stats -------------------------------
    if cov_type == "diag":
        mu_s, var_s, _ = class_stats_diag(Xs, Ys, K)             # (K,d)
        mu_t, var_t, _ = class_stats_diag(Xt, Yt, K)
        Sig_s = Sig_t = None
    else:
        mu_s, Sig_s, _ = class_stats_full(Xs, Ys, K, reg=reg, ddof=ddof)    # (K,d),(K,d,d)
        mu_t, Sig_t, _ = class_stats_full(Xt, Yt, K, reg=reg, ddof=ddof)
        Sig_s = _sym(Sig_s)
        Sig_t = _sym(Sig_t)
        var_s = torch.diagonal(Sig_s, dim1=-2, dim2=-1)
        var_t = torch.diagonal(Sig_t, dim1=-2, dim2=-1)

    counts_s = torch.bincount(Ys, minlength=K)
    counts_t = torch.bincount(Yt, minlength=K)
    total_s = int(counts_s.sum().item())
    total_t = int(counts_t.sum().item())
    present_s = (counts_s > 0)
    present_t = (counts_t > 0)
    present_both = (present_s & present_t)

    total_min = int(torch.minimum(counts_s, counts_t).sum().item())
    if total_min <= 0:
        total_min = max(total_s, total_t)

    pi_src = counts_s.to(dtype=fp_dtype) / max(1, total_s)
    desired = (pi_src * float(total_min)).clamp_min(0.0)
    base = torch.floor(desired)
    frac = desired - base
    base = (base * present_both).long()
    frac = frac * present_both
    rem = int(max(0, total_min - int(base.sum().item())))
    if rem > 0:
        k_take = min(rem, K)
        if k_take > 0:
            _, idx = torch.topk(frac, k=k_take)
            base.index_add_(0, idx, torch.ones_like(idx, dtype=base.dtype, device=base.device))
    n_per_class = base  # (K,)

    # ------------------------- 2) Preallocate outputs --------------------------
    S_steps = n_inter + 2
    steps = torch.empty(S_steps, dtype=fp_dtype, device=device)
    mu_out = torch.full((S_steps, K, d), float("nan"), dtype=fp_dtype, device=device)
    var_out = torch.full((S_steps, K, d), float("nan"), dtype=fp_dtype, device=device)
    cnt_out = torch.empty((S_steps, K), dtype=torch.int64, device=device)
    pi_out = torch.empty((S_steps, K), dtype=fp_dtype, device=device)
    eta1_out = torch.full((S_steps, K, d), float("nan"), dtype=fp_dtype, device=device)
    eta2d_out = torch.full((S_steps, K, d), float("nan"), dtype=fp_dtype, device=device)
    Sig_out = torch.full((S_steps, K, d, d), float("nan"), dtype=fp_dtype, device=device) if cov_type == "full" else None

    # ------------------------- 3) Cache FR factors (full) ----------------------
    if cov_type == "full":
        cache = [None] * K
        eps = torch.finfo(fp_dtype).eps
        for k in range(K):
            if not present_both[k]:
                cache[k] = None
                continue
            S_k = _sym(Sig_s[k])
            T_k = _sym(Sig_t[k])
            alpha_k = (torch.trace(S_k) / d).clamp_min(eps).item()
            beta_k  = (torch.trace(T_k) / d).clamp_min(eps).item()
            S_hat = S_k / alpha_k
            T_hat = T_k / beta_k
            ws, Vs = torch.linalg.eigh(S_hat)
            ws = ws.clamp_min(eps)
            ws_sqrt = ws.sqrt()
            inv_ws_sqrt = ws_sqrt.reciprocal()
            S_half  = (Vs * ws_sqrt) @ Vs.transpose(-1, -2)
            S_mhalf = (Vs * inv_ws_sqrt) @ Vs.transpose(-1, -2)
            A_hat = S_mhalf @ T_hat @ S_mhalf
            la, Va = torch.linalg.eigh(_sym(A_hat))
            la = la.clamp_min(eps)
            cache[k] = (S_half, S_mhalf, Va, la, alpha_k, beta_k)
    else:
        cache = None

    # ------------------------- 4) Fill source (s=0) ----------------------------
    steps[0] = 0.0
    mu_out[0] = mu_s
    var_out[0] = var_s
    cnt_out[0] = counts_s
    pi_out[0] = pi_src
    if cov_type == "full":
        Sig_out[0] = Sig_s
        Lam0, _ = _chol_inv_spd(Sig_s, base_rel=1e-12, abs_floor=jitter, max_tries=5)  # <-- uses batch-safe diag
        eta1_out[0] = torch.einsum("kdd,kd->kd", Lam0, mu_s)
        eta2d_out[0] = -0.5 * torch.diagonal(Lam0, dim1=-2, dim2=-1)
    else:
        e1, e2d = _eta_diag(mu_s, var_s)
        eta1_out[0] = e1
        eta2d_out[0] = e2d

    all_domains = []

    # ------------------------- 5) Intermediates (s=1..n_inter) -----------------
    for i in range(1, n_inter + 1):
        t = i / (n_inter + 1)
        steps[i] = t

        if cov_type == "diag":
            mu_mid = (1.0 - t) * mu_s + t * mu_t
            var_mid = ((var_s.clamp_min(jitter)).log() * (1.0 - t) +
                       (var_t.clamp_min(jitter)).log() * t).exp()
            e1, e2d = _eta_diag(mu_mid, var_mid)
            mu_out[i] = mu_mid
            var_out[i] = var_mid
            cnt_out[i] = n_per_class
            pi_out[i] = pi_src
            eta1_out[i] = e1
            eta2d_out[i] = e2d

            n_k = n_per_class
            nk_sum = int(n_k.sum().item())
            if nk_sum > 0:
                class_ids = torch.repeat_interleave(torch.arange(K, device=device), n_k)
                m = mu_mid[class_ids]
                s2 = var_mid[class_ids].clamp_min(jitter)
                eps_samp = torch.randn(nk_sum, d, device=device, dtype=fp_dtype)
                z = eps_samp * s2.sqrt() + m
                y = class_ids
                w = torch.ones(nk_sum, device=device, dtype=fp_dtype)
                all_domains.append(DomainDataset(z, w, y, y))

        else:
            mu_mid = (1.0 - t) * mu_s + t * mu_t
            Sig_mid_i = torch.empty_like(Sig_s)
            Lam_diag = torch.empty_like(var_s)
            eta1_i = torch.empty_like(mu_s)

            for k in range(K):
                if not present_both[k]:
                    Sig_mid_i[k].fill_(float('nan'))
                    Lam_diag[k].fill_(float('nan'))
                    eta1_i[k].fill_(float('nan'))
                    continue
                S_half, S_mhalf, Va, la, alpha_k, beta_k = cache[k]
                c_t = (alpha_k ** (1.0 - t)) * (beta_k ** t)
                Lt = (Va * la.pow(t)) @ Va.transpose(-1, -2)
                Sig_hat = _sym(S_half @ Lt @ S_half)
                Sig_k = _project_spd(Sig_hat * c_t, rel_floor=1e-8, abs_floor=jitter)
                Sig_mid_i[k] = Sig_k
                Lam_hat = (Va * la.pow(-t)) @ Va.transpose(-1, -2)
                Prec_t = _sym(S_mhalf @ Lam_hat @ S_mhalf) / c_t
                eta1_i[k] = Prec_t @ mu_mid[k]
                Lam_diag[k] = torch.diagonal(Prec_t)

            mu_out[i] = mu_mid
            Sig_out[i] = Sig_mid_i
            var_out[i] = torch.diagonal(Sig_mid_i, dim1=-2, dim2=-1)
            cnt_out[i] = n_per_class
            pi_out[i] = pi_src
            eta1_out[i] = eta1_i
            eta2d_out[i] = -0.5 * Lam_diag

            n_k = n_per_class
            nk_sum = int(n_k.sum().item())
            if nk_sum > 0:
                zs, ys_cls = [], []
                for k in range(K):
                    nk = int(n_k[k].item())
                    if nk <= 0 or not present_both[k]:
                        continue
                    Sk = Sig_mid_i[k]
                    mk = mu_mid[k]
                    Fk, _ = _chol_sqrt_spd(Sk, base_rel=1e-12, abs_floor=jitter, max_tries=5)  # <-- uses batch-safe diag
                    eps_samp = torch.randn(nk, d, device=device, dtype=fp_dtype)
                    z_k = eps_samp @ Fk.transpose(-1, -2) + mk
                    zs.append(z_k)
                    ys_cls.append(torch.full((nk,), k, device=device, dtype=torch.long))
                if zs:
                    Zm = torch.cat(zs, dim=0)
                    Ym = torch.cat(ys_cls, dim=0)
                    Wm = torch.ones(len(Ym), device=device, dtype=fp_dtype)
                    all_domains.append(DomainDataset(Zm, Wm, Ym, Ym))

    # ------------------------- 6) Target (s=S-1) --------------------------------
    steps[-1] = 1.0
    mu_out[-1] = mu_t
    var_out[-1] = var_t
    cnt_out[-1] = counts_t
    pi_out[-1] = counts_t.to(dtype=fp_dtype) / max(1, total_t)
    if cov_type == "full":
        Sig_out[-1] = Sig_t
        LamT, _ = _chol_inv_spd(Sig_t, base_rel=1e-12, abs_floor=jitter, max_tries=5)
        eta1_out[-1] = torch.einsum("kdd,kd->kd", LamT, mu_t)
        eta2d_out[-1] = -0.5 * torch.diagonal(LamT, dim1=-2, dim2=-1)
    else:
        e1, e2d = _eta_diag(mu_t, var_t)
        eta1_out[-1] = e1
        eta2d_out[-1] = e2d

    # Wrap target domain
    X_tgt = Xt
    if hasattr(dataset_t, "targets") and dataset_t.targets is not None:
        Y_gt_full = dataset_t.targets
    else:
        Y_gt_full = torch.full((len(dataset_t.data),), -1, device=device, dtype=torch.long)
    if not torch.is_tensor(Y_gt_full):
        Y_gt_full = torch.as_tensor(Y_gt_full, dtype=torch.long, device=device)
    else:
        Y_gt_full = Y_gt_full.to(device=device, dtype=torch.long)
    Y_gt_full = Y_gt_full.view(-1)
    Y_gt = Y_gt_full[mask_t]
    Y_em = Yt
    all_domains.append(DomainDataset(X_tgt, torch.ones(len(Y_em), device=device, dtype=fp_dtype), Y_gt, Y_em))

    # ------------------------- 7) Pack outputs (single CPU hop) -----------------
    domain_params = {
        "K": int(K), "d": int(d), "cov_type": cov_type,
        "steps": steps.to("cpu", dtype=torch.float64).numpy(),
        "mu":    mu_out.to("cpu", dtype=torch.float64).numpy(),
        "var":   var_out.to("cpu", dtype=torch.float64).numpy(),
        "counts": cnt_out.to("cpu").numpy(),
        "pi":     pi_out.to("cpu", dtype=torch.float64).numpy(),
        "present_source": present_s.to("cpu").numpy().astype(bool),
        "present_target": present_t.to("cpu").numpy().astype(bool),
        "eta1":      eta1_out.to("cpu", dtype=torch.float64).numpy(),
        "eta2_diag": eta2d_out.to("cpu", dtype=torch.float64).numpy(),
    }
    if cov_type == "full":
        domain_params["Sigma"] = Sig_out.to("cpu", dtype=torch.float64).numpy()

    if save_path is not None:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez_compressed(save_path, **domain_params)
        print(f"[FR/GPU] Saved domain parameters -> {save_path}")

    return all_domains, Yt.to("cpu").numpy(), domain_params




# ---- Cache for prepared representations






# ---------- helper for covariance flooring on full-covariance mixes ----------
def _eigen_floor_full_covs(Sigma, eps: float = 1e-6):
    """
    Floor eigenvalues of full covariance matrices to ensure PSD and better conditioning.
    Sigma: array-like of shape (K, D, D)
    """
    Sigma = np.asarray(Sigma)
    K, D, _ = Sigma.shape
    out = np.empty_like(Sigma)
    for k in range(K):
        S = Sigma[k]
        # symmetric safety
        S = 0.5 * (S + S.T)
        w, V = np.linalg.eigh(S)
        w = np.clip(w, eps, None)
        out[k] = (V * w) @ V.T
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
def _sym(A: torch.Tensor) -> torch.Tensor:
    # Works for 2D, 3D, and higher: only swap the last two axes.
    return 0.5 * (A + A.transpose(-1, -2))

def _spd_eigh(A: torch.Tensor,
              rel_floor: float = 1e-6,
              abs_floor: float = 1e-10) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Eigendecompose a (batched) symmetric matrix robustly in float64 and floor eigenvalues.
    Returns (w, V) in float64; caller may cast back to original dtype.
    """
    A64 = _sym(A).to(torch.float64)
    w, V = torch.linalg.eigh(A64)  # (.., d), (.., d, d)
    d = A64.shape[-1]
    # per-batch scale = trace/d (robust to overall magnitude)
    tr_over_d = torch.clamp(A64.diagonal(dim1=-2, dim2=-1).sum(dim=-1) / float(d), min=abs_floor)
    floor = torch.clamp(rel_floor * tr_over_d, min=abs_floor)[..., None]  # (...,1)
    w = torch.maximum(w, floor.expand_as(w))
    return w, V

def _spd_from_eig(A: torch.Tensor,
                  rel_floor: float = 1e-6,
                  abs_floor: float = 1e-10) -> torch.Tensor:
    """
    Project symmetric matrix to SPD using robust eig floor; returns same dtype/device as A.
    """
    dtype, device = A.dtype, A.device
    w, V = _spd_eigh(A, rel_floor=rel_floor, abs_floor=abs_floor)  # float64
    A_spd = V @ torch.diag_embed(w) @ V.mT
    A_spd = _sym(A_spd)
    return A_spd.to(dtype=dtype, device=device)

def _chol_inv_spd(S: torch.Tensor,
                  base_rel: float = 1e-12,
                  max_tries: int = 7) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Robust Cholesky-inverse with adaptive per-batch diagonal loading tied to trace/d.
    Returns (Lam = S^{-1}, L) with L = chol factor of successful S_try.
    """
    K = S.shape[0]
    d = S.shape[-1]
    dtype, device = S.dtype, S.device
    eye = torch.eye(d, dtype=dtype, device=device)
    # scale per class for loading
    tr_over_d = torch.clamp(S.diagonal(dim1=-2, dim2=-1).sum(dim=-1) / float(d),
                            min=torch.finfo(dtype).eps)  # (K,)
    lam = base_rel * tr_over_d  # (K,)

    for _ in range(max_tries):
        S_try = S + lam.view(K, 1, 1) * eye
        L, info = torch.linalg.cholesky_ex(S_try)
        if (info == 0).all():
            Lam = torch.cholesky_inverse(L)
            return Lam, L
        # escalate only where it failed
        lam = lam * torch.where(info != 0, torch.tensor(10.0, device=device, dtype=dtype),
                                torch.tensor(1.0, device=device, dtype=dtype))

    # Final safeguard: hard project and factor
    S_proj = _spd_from_eig(S, rel_floor=1e-4, abs_floor=1e-8)
    L = torch.linalg.cholesky(S_proj)
    Lam = torch.cholesky_inverse(L)
    return Lam, L


def _spd_sqrt(A: torch.Tensor, rel_floor: float = 1e-6, abs_floor: float = 1e-10) -> torch.Tensor:
    w, V = _spd_eigh(A, rel_floor, abs_floor)
    sqrtw = torch.sqrt(w)
    M = V @ torch.diag_embed(sqrtw) @ V.mT
    return _sym(M).to(dtype=A.dtype, device=A.device)

def _spd_invsqrt(A: torch.Tensor, rel_floor: float = 1e-6, abs_floor: float = 1e-10) -> torch.Tensor:
    w, V = _spd_eigh(A, rel_floor, abs_floor)
    invsqrtw = 1.0 / torch.sqrt(w)
    M = V @ torch.diag_embed(invsqrtw) @ V.mT
    return _sym(M).to(dtype=A.dtype, device=A.device)

def _spd_geometric_mean(A: torch.Tensor, B: torch.Tensor,
                        rel_floor: float = 1e-6, abs_floor: float = 1e-10) -> torch.Tensor:
    """
    A # B = A^{1/2} (A^{-1/2} B A^{-1/2})^{1/2} A^{1/2}
    """
    Ah  = _spd_sqrt(A, rel_floor, abs_floor)
    Aih = _spd_invsqrt(A, rel_floor, abs_floor)
    mid = _spd_sqrt(_sym(Aih @ B @ Aih), rel_floor, abs_floor)
    M = Ah @ mid @ Ah
    return _sym(M).to(dtype=A.dtype, device=A.device)



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




@torch.no_grad()
def fr_interp_diag(mu_s: torch.Tensor, var_s: torch.Tensor,
                   mu_t: torch.Tensor, var_t: torch.Tensor,
                   t: float, eps: float = 1e-8):
    """
    Fisher–Rao geodesic between two diagonal Gaussians (element-wise).
    Returns (mu_mid, var_mid).
    """
    # variance path (AIRM): var(t) = var_s^{1-t} * var_t^{t}
    var_s = var_s.to(torch.float64)
    var_t = var_t.to(torch.float64)
    r = (var_t.clamp_min(eps) / var_s.clamp_min(eps))            # λ_i
    var_mid = var_s * r.pow(t)

    # mean path: μ_i(t) = μ_s,i + ((λ_i^t - 1)/(λ_i - 1)) (μ_t,i - μ_s,i)
    num = r.pow(t) - 1.0
    den = r - 1.0
    w = torch.where(den.abs() < 1e-6, torch.full_like(den, t), num / den)

    mu_s = mu_s.to(torch.float64)
    mu_t = mu_t.to(torch.float64)
    mu_mid = mu_s + w * (mu_t - mu_s)

    # back to original dtype
    out_dtype = mu_s.dtype if mu_s.dtype == mu_t.dtype else torch.float32
    return mu_mid.to(out_dtype), var_mid.to(out_dtype)

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
