import numpy as np
from typing import Tuple, Dict, List, Optional, Union
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
    if cov_type == "full":
        ld0 = _logdet_per_class_full(_np(Sig_s))
        ldT = _logdet_per_class_full(_np(Sig_t))
    else:
        ld0 = _logdet_per_class_diag(_np(vars_s))
        ldT = _logdet_per_class_diag(_np(vars_t))

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


# @torch.no_grad()
# def natural_interp_full(mu_s: torch.Tensor,
#                         Sigma_s: torch.Tensor,
#                         mu_t: torch.Tensor,
#                         Sigma_t: torch.Tensor,
#                         t: float,
#                         eps: float = 1e-6):
#     """
#     Natural-parameter interpolation for full-covariance Gaussians.

#     Inputs (per class; no batching assumed):
#       mu_s     : (d,)   source mean
#       Sigma_s  : (d,d)  source covariance (SPD)
#       mu_t     : (d,)   target mean
#       Sigma_t  : (d,d)  target covariance (SPD)
#       t        : float in [0,1]
#       eps      : jitter added to covariances for SPD safety

#     Returns:
#       mu_m     : (d,)   interpolated mean
#       Sigma_m  : (d,d)  interpolated covariance (SPD)
#     """
#     d = mu_s.numel()
#     I = torch.eye(d, dtype=Sigma_s.dtype, device=Sigma_s.device)

#     def _spd_prec(S):
#         # make SPD (jitter) and compute precision via Cholesky solve (no explicit inverse)
#         S = 0.5 * (S + S.T) + eps * I
#         L = torch.linalg.cholesky(S)
#         # Solve S^{-1} = (L L^T)^{-1} = L^{-T} L^{-1}
#         # Use identity: A^{-1} B by solving A X = B
#         # Here we want full matrix: solve for columns of I
#         inv_S = torch.cholesky_inverse(L)  # stable, symmetric
#         return inv_S

#     # Precisions (Λ = Σ^{-1})
#     Lambda_s = _spd_prec(Sigma_s)
#     Lambda_t = _spd_prec(Sigma_t)

#     # Natural parameters: η1 = Λ μ, η2 = -1/2 Λ
#     eta1_s = Lambda_s @ mu_s
#     eta2_s = -0.5 * Lambda_s
#     eta1_t = Lambda_t @ mu_t
#     eta2_t = -0.5 * Lambda_t

#     # Linear interpolation in natural-parameter space
#     eta1_m = (1.0 - t) * eta1_s + t * eta1_t
#     eta2_m = (1.0 - t) * eta2_s + t * eta2_t

#     # Back to (μ, Σ): Λ_m = -2 η2_m, Σ_m = Λ_m^{-1}, μ_m = Λ_m^{-1} η1_m
#     Lambda_m = -2.0 * eta2_m
#     Lambda_m = 0.5 * (Lambda_m + Lambda_m.T) + eps * I  # symmetrize + SPD safety

#     # Invert Λ_m stably and solve for μ_m
#     Lm = torch.linalg.cholesky(Lambda_m)
#     Sigma_m = torch.cholesky_inverse(Lm)
#     # μ_m = Λ_m^{-1} η1_m  -> solve Λ_m μ = η1
#     mu_m = torch.cholesky_solve(eta1_m.unsqueeze(1), Lm).squeeze(1)

#     # final symmetry clean-up
#     Sigma_m = 0.5 * (Sigma_m + Sigma_m.T)

#     return mu_m, Sigma_m, eta1_m, eta2_m  # return natural params for logging if needed




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


@torch.no_grad()
def natural_interp_full(mu_s: torch.Tensor,
                        Sigma_s: torch.Tensor,
                        mu_t: torch.Tensor,
                        Sigma_t: torch.Tensor,
                        t: float,
                        eps: float = 0.0,                 # base jitter for robust SPD (B)
                        debug: bool = True,
                        tag: str = "",
                        eig_checks: bool = True,
                        pct: tuple = (1, 5, 10, 25, 50, 75, 90, 95, 99),
                        warn_tol: float = 1e-12,
                        *,
                        compare: bool = False,            # <<< NEW: run A/B comparison
                        scale_alpha: float = 1e5,         # <<< scaling factor for method A
                        return_report: bool = True        # <<< when compare=True, also return a report dict
                        ):
    """
    Natural-parameter interpolation (e-geodesic) for full-cov Gaussians.

    Returns (default): mu_m, Sigma_m, eta1_m, eta2_m          # robust SPD (method B)
    If compare=True and return_report=True, returns a 5th item: dict with metrics.
    """
    device = Sigma_s.device
    dtype  = Sigma_s.dtype
    d = mu_s.numel()
    I = torch.eye(d, dtype=dtype, device=device)

    # ---------- small helpers ----------
    def _sym(S):
        return 0.5 * (S + S.T)

    def _cond(S):
        # Fro-safe condition number (clamp tiny eigs)
        w = torch.linalg.eigvalsh(_sym(S)).clamp_min(1e-20)
        return (w[-1] / w[0]).item()

    def _cov_stats(S: torch.Tensor, name: str):
        S = _sym(S)
        w = torch.linalg.eigvalsh(S)
        w_clamped = torch.clamp(w, min=1e-20)
        tr = torch.sum(w_clamped)
        fro = torch.linalg.norm(S, ord='fro')
        lam_min = torch.min(w_clamped)
        lam_max = torch.max(w_clamped)
        cond = (lam_max / lam_min).item()
        sign, logabsdet = torch.linalg.slogdet(S)
        print(f"[NAT-DBG{(':' + tag) if tag else ''}] {name}: "
              f"trace={tr.item():.6g}  ||·||_F={fro.item():.6g}  "
              f"λ_min={lam_min.item():.6g}  λ_max={lam_max.item():.6g}  "
              f"cond={cond:.6g}  logdet={logabsdet.item():.6g}")
        return w

    def _percentiles(w: torch.Tensor, ps: tuple):
        qs = torch.quantile(w, torch.tensor([p/100.0 for p in ps], device=w.device))
        return {p: qs[i].item() for i, p in enumerate(ps)}

    def _print_percentiles(w: torch.Tensor, name: str):
        q = _percentiles(w, pct)
        q_str = "  ".join([f"p{p}={q[p]:.6g}" for p in pct])
        print(f"[EIG-PCT{(':' + tag) if tag else ''}] {name}: {q_str}")

    # ---------- Method A: scale–then–unscale precision ----------
    def _prec_scale_unscale(S: torch.Tensor, alpha: float):
        # Compute inv(S) via inv(alpha*S) * alpha
        Ssym = _sym(S)
        L = torch.linalg.cholesky(alpha * Ssym)
        Lam_scaled = torch.cholesky_inverse(L)      # (alpha*S)^{-1}
        Lam = Lam_scaled * alpha                    # recover S^{-1}
        return Lam

    # ---------- Method B: robust SPD precision (adaptive jitter, fallback eig-clip) ----------
    def _prec_robust(S: torch.Tensor, base_eps: float):
        Ssym = _sym(S)
        # start with relative jitter based on mean diag scale
        scale = torch.mean(torch.diag(Ssym)).abs().item()
        jitter = max(base_eps, 1e-12 * (scale if np.isfinite(scale) else 1.0))
        tries = 0
        while True:
            try:
                L = torch.linalg.cholesky(Ssym + jitter * I)
                Lam = torch.cholesky_inverse(L)
                return Lam, L, jitter, False  # no eig-clip
            except RuntimeError:
                tries += 1
                if tries >= 5:
                    # fallback: eig-clip
                    w, V = torch.linalg.eigh(Ssym)
                    w = torch.clamp(w, min=jitter if jitter > 0 else 1e-12)
                    Sp = (V * w) @ V.T
                    L = torch.linalg.cholesky(Sp)
                    Lam = torch.cholesky_inverse(L)
                    return Lam, L, jitter, True
                jitter *= 10.0

    # ---------- Build endpoint precisions: A and B ----------
    # A endpoints

    # breakpoint()
    Lambda_s_A = _prec_scale_unscale(Sigma_s, scale_alpha)
    Lambda_t_A = _prec_scale_unscale(Sigma_t, scale_alpha)

    # B endpoints
    Lambda_s_B, Ls_B, jit_s, clip_s = _prec_robust(Sigma_s, eps)
    Lambda_t_B, Lt_B, jit_t, clip_t = _prec_robust(Sigma_t, eps)

    # ---------- Natural parameters (A and B) ----------
    def _etas(Lam, mu):
        eta1 = Lam @ mu
        eta2 = -0.5 * Lam
        return eta1, eta2

    eta1_s_A, eta2_s_A = _etas(Lambda_s_A, mu_s)
    eta1_t_A, eta2_t_A = _etas(Lambda_t_A, mu_t)

    eta1_s_B, eta2_s_B = _etas(Lambda_s_B, mu_s)
    eta1_t_B, eta2_t_B = _etas(Lambda_t_B, mu_t)

    # Interpolate in natural space
    def _interp_eta(e1_s, e1_t, e2_s, e2_t, t):
        eta1_m = (1.0 - t) * e1_s + t * e1_t
        eta2_m = (1.0 - t) * e2_s + t * e2_t
        Lam_m  = -2.0 * eta2_m
        Lm     = torch.linalg.cholesky(_sym(Lam_m))
        Sig_m  = torch.cholesky_inverse(Lm)
        mu_m   = torch.cholesky_solve(eta1_m.unsqueeze(1), Lm).squeeze(1)
        return mu_m, Sig_m, eta1_m, eta2_m, Lam_m

    # Mid-point (or general t) for A and B
    mu_m_A, Sigma_m_A, eta1_m_A, eta2_m_A, Lambda_m_A = _interp_eta(eta1_s_A, eta1_t_A, eta2_s_A, eta2_t_A, t)
    mu_m_B, Sigma_m_B, eta1_m_B, eta2_m_B, Lambda_m_B = _interp_eta(eta1_s_B, eta1_t_B, eta2_s_B, eta2_t_B, t)

    # ---------- Diagnostics ----------
    def _inv_resid(Lam, Sig):
        return (torch.linalg.norm(Lam @ Sig - I, ord='fro') / d).item()

    report = None
    if compare:
        # Endpoint residuals
        res_s_A = _inv_resid(Lambda_s_A, Sigma_s)
        res_t_A = _inv_resid(Lambda_t_A, Sigma_t)
        res_s_B = _inv_resid(Lambda_s_B, Sigma_s + jit_s * I)
        res_t_B = _inv_resid(Lambda_t_B, Sigma_t + jit_t * I)

        # Mid residuals
        res_m_A = _inv_resid(Lambda_m_A, Sigma_m_A)
        res_m_B = _inv_resid(Lambda_m_B, Sigma_m_B)

        # Condition numbers
        kS_s = _cond(Sigma_s); kS_t = _cond(Sigma_t)
        kS_mA = _cond(Sigma_m_A); kS_mB = _cond(Sigma_m_B)

        # Relative diffs between A and B at t
        rel_mu = (torch.linalg.norm(mu_m_A - mu_m_B) /
                  (torch.linalg.norm(mu_m_B) + 1e-20)).item()
        rel_S  = (torch.linalg.norm(Sigma_m_A - Sigma_m_B, ord='fro') /
                  (torch.linalg.norm(Sigma_m_B, ord='fro') + 1e-20)).item()

        print("\n==== [A/B comparison @ natural_interp_full] ====")
        print(f"Endpoints:  κ(Σ_s)={kS_s:.3e}  κ(Σ_t)={kS_t:.3e}")
        print(f"A endpoint residuals:  ‖Λ_s Σ_s−I‖/d={res_s_A:.3e},  ‖Λ_t Σ_t−I‖/d={res_t_A:.3e}")
        print(f"B endpoint residuals:  ‖Λ_s Σ_s−I‖/d={res_s_B:.3e},  ‖Λ_t Σ_t−I‖/d={res_t_B:.3e}  "
              f"(jit_s={jit_s:.1e}{' eig-clip' if clip_s else ''}, jit_t={jit_t:.1e}{' eig-clip' if clip_t else ''})")
        print(f"Midpoint residuals:    A: {res_m_A:.3e}   B: {res_m_B:.3e}")
        print(f"Midpoint κ(Σ):         A: {kS_mA:.3e}     B: {kS_mB:.3e}")
        print(f"Rel. diff @ t in μ:    {rel_mu:.3e}")
        print(f"Rel. diff @ t in Σ(F): {rel_S:.3e}")
        print("=================================================\n")

        report = {
            "endpoint_residuals": {
                "A": {"source": res_s_A, "target": res_t_A},
                "B": {"source": res_s_B, "target": res_t_B, "jit_s": jit_s, "jit_t": jit_t,
                      "eigclip_source": bool(clip_s), "eigclip_target": bool(clip_t)},
            },
            "mid_residuals": {"A": res_m_A, "B": res_m_B},
            "kappa": {"Sigma_s": kS_s, "Sigma_t": kS_t, "Sigma_mid_A": kS_mA, "Sigma_mid_B": kS_mB},
            "relative_diff": {"mu": rel_mu, "Sigma_fro": rel_S},
        }

    # ---- optional detailed debug on B (robust) ----
    if debug:
        w_s = _cov_stats(Sigma_s + eps * I, "Σ_source")
        w_t = _cov_stats(Sigma_t + eps * I, "Σ_target")
        w_m = _cov_stats(Sigma_m_B + eps * I, f"Σ_interp(t={t:.3f}) [B]")
        def _prec_stats(P: torch.Tensor, name: str):
            P = _sym(P)
            w = torch.linalg.eigvalsh(P)
            w_clamped = torch.clamp(w, min=1e-20)
            tr = torch.sum(w_clamped)
            fro = torch.linalg.norm(P, ord='fro')
            lam_min = torch.min(w_clamped)
            lam_max = torch.max(w_clamped)
            cond = (lam_max / lam_min).item()
            sign, logabsdet = torch.linalg.slogdet(P)
            print(f"[NAT-DBG{(':' + tag) if tag else ''}] {name}: "
                  f"trace={tr.item():.6g}  ||·||_F={fro.item():.6g}  "
                  f"λ_min={lam_min.item():.6g}  λ_max={lam_max.item():.6g}  "
                  f"cond={cond:.6g}  logdet={logabsdet.item():.6g}")
            return w
        _ = _prec_stats(Lambda_s_B, "Λ_source [B]")
        _ = _prec_stats(Lambda_t_B, "Λ_target [B]")
        _ = _prec_stats(Lambda_m_B, f"Λ_interp(t={t:.3f}) [B]")

        resid_inv = torch.linalg.norm(Lambda_m_B @ Sigma_m_B - torch.eye(d, device=device, dtype=dtype), ord='fro') / d
        print(f"[CHK{(':' + tag) if tag else ''}] inverse residual (B) ‖Λ(t)Σ(t)-I‖_F/d = {resid_inv:.3e}")

        # non-commutativity
        comm = Sigma_s @ Sigma_t - Sigma_t @ Sigma_s
        comm_norm = torch.linalg.norm(comm, ord='fro')
        print(f"[CHK{(':' + tag) if tag else ''}] commutator ‖Σ_s Σ_t - Σ_t Σ_s‖_F = {comm_norm:.6g}")

        if eig_checks:
            # Percentiles
            def _pp(w, name): 
                q = _percentiles(w, pct)
                q_str = "  ".join([f"p{p}={q[p]:.6g}" for p in pct]); 
                print(f"[EIG-PCT{(':' + tag) if tag else ''}] {name}: {q_str}")
            _pp(w_s, "Σ_source")
            _pp(w_t, "Σ_target")
            _pp(w_m, f"Σ_interp(t={t:.3f}) [B]")

            lam_min_s, lam_max_s = w_s[0].item(), w_s[-1].item()
            lam_min_t, lam_max_t = w_t[0].item(), w_t[-1].item()
            lam_min_m, lam_max_m = w_m[0].item(), w_m[-1].item()
            lo_end = min(lam_min_s, lam_min_t)
            hi_end = max(lam_max_s, lam_max_t)
            below_hi = (lam_max_m < hi_end - warn_tol)
            above_lo = (lam_min_m > lo_end + warn_tol)
            print(f"[EIG-CHK{(':' + tag) if tag else ''}] endpoints: "
                  f"λ_min^ends={lo_end:.6g}, λ_max^ends={hi_end:.6g}; "
                  f"interp[B]: λ_min={lam_min_m:.6g}, λ_max={lam_max_m:.6g}")
            if below_hi:
                print(f"[EIG-OBS{(':' + tag) if tag else ''}] λ_max at t={t:.3f} is below the larger endpoint (harmonic-mean effect).")
            if above_lo:
                print(f"[EIG-OBS{(':' + tag) if tag else ''}] λ_min at t={t:.3f} exceeds the smaller endpoint (conditioning improves).")
    # breakpoint()
    # Return the robust (B) path by default, to preserve existing callers
    if compare and return_report:
        return mu_m_B, Sigma_m_B, eta1_m_B, eta2_m_B, report
    return mu_m_B, Sigma_m_B, eta1_m_B, eta2_m_B


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
    return mu_m, var_m, eta1_m, eta2_m


def generate_natural_domains_between(
    n_inter,
    dataset_s,
    dataset_t,
    plan=None,
    entry_cutoff: int = 0,
    conf: float = 0.0,
    source_model: Optional[torch.nn.Module] = None,
    pseudolabels: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None,
    cov_type: str = "full",
    reg: float = 1e-6,
    ddof: int = 0,
    jitter: float = 0.0,
    diagnostic_class: Optional[int] = 1,  # kept in signature but unused now
    visualize: bool = True,
    args=None,
):
    """
    Generate intermediate domains by linear interpolation in the natural-parameter space
    for Gaussian class-conditionals.

    Natural parameters: η₁ = Λ μ,  η₂ = -½ Λ, where Λ = Σ⁻¹.
    (Diagnostics disabled in this version.)
    """
    import time, os
    
    import torch
    import torch.nn as nn
    # import matplotlib.pyplot as plt  # (unused now that diagnostics are off)

    # ---- small local helpers -----------------------------------------------
    def _to_np(x):
        return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

    def _pad_kd(mu_like, var_like, present_mask, K, d):
        mu_out = np.full((K, d), np.nan, dtype=np.float64)
        var_out = np.full((K, d), np.nan, dtype=np.float64)
        mu_np, var_np = _to_np(mu_like), _to_np(var_like)
        mu_out[present_mask] = mu_np[present_mask]
        var_out[present_mask] = var_np[present_mask]
        return mu_out, var_out

    def _pad_kdd(Sig_like, present_mask, K, d):
        Sig_out = np.full((K, d, d), np.nan, dtype=np.float64)
        Sig_np = _to_np(Sig_like)
        Sig_out[present_mask] = Sig_np[present_mask]
        return Sig_out

    print("------------Generate Intermediate domains (NATURAL)----------")
    _t_total = time.time()

    xs, xt = dataset_s.data, dataset_t.data
    ys = dataset_s.targets
    if len(xs.shape) > 2:
        xs, xt = nn.Flatten()(xs), nn.Flatten()(xt)

    if ys is None:
        raise ValueError("Source dataset must provide targets for class statistics.")
    if not hasattr(dataset_t, "targets_em") or dataset_t.targets_em is None:
        raise ValueError("Target dataset must provide targets_em (EM-derived labels).")

    device = xs.device if torch.is_tensor(xs) else torch.device("cpu")
    Zs = xs if torch.is_tensor(xs) else torch.as_tensor(xs)
    Zt = xt if torch.is_tensor(xt) else torch.as_tensor(xt)
    Ys = ys if torch.is_tensor(ys) else torch.as_tensor(ys)
    if args.em_match != "none":
        Yt_em = dataset_t.targets_em if torch.is_tensor(dataset_t.targets_em) else torch.as_tensor(dataset_t.targets_em)
    else:
        Yt_em = dataset_t.targets_pesudo if torch.is_tensor(dataset_t.targets_pseudo) else torch.as_tensor(dataset_t.targets_pseudo)

    Zs, Zt = Zs.to(device), Zt.to(device)
    Ys, Yt_em = Ys.to(device, dtype=torch.long), Yt_em.to(device, dtype=torch.long)
    if Ys.numel() == 0:
        return [], torch.empty(0, dtype=torch.long)

    K = int(max(Ys.max(), Yt_em.max()).item()) + 1

    # --- class stats (source / target) ---
    if cov_type == "diag":
        mus_s, vars_s, _ = class_stats_diag(Zs, Ys, K)
        mus_t, vars_t, _ = class_stats_diag(Zt, Yt_em, K)
        Sig_s = Sig_t = None
    else:
        mus_s, Sig_s, _ = class_stats_full(Zs, Ys, K, reg=reg, ddof=ddof)
        mus_t, Sig_t, _ = class_stats_full(Zt, Yt_em, K, reg=reg, ddof=ddof)
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

    present_s = (counts_s.detach().cpu().numpy() > 0)
    present_t = (counts_t.detach().cpu().numpy() > 0)
    present_both = present_s & present_t

    # ---------------- containers ----------------
    steps, mu_list, var_list = [], [], []
    counts_list, pi_list = [], []
    Sigma_list = [] if cov_type == "full" else None
    eta1_steps, eta2d_steps = [], []

    # ---------------- source (t=0) ----------------
    pi_s = (counts_s.float() / max(1, total_s)).cpu().numpy()
    mu0, var0 = _pad_kd(mus_s, vars_s, present_s, K, d)
    steps.append(0.0)
    mu_list.append(mu0)
    var_list.append(var0)
    counts_list.append(counts_s.cpu().numpy().astype(np.int64))
    pi_list.append(pi_s)

    # Endpoint η arrays computed ONCE and reused
    eta1_src = np.full((K, d), np.nan, dtype=np.float64)
    eta2d_src = np.full((K, d), np.nan, dtype=np.float64)
    eta1_tgt = np.full((K, d), np.nan, dtype=np.float64)
    eta2d_tgt = np.full((K, d), np.nan, dtype=np.float64)

    if cov_type == "full":
        Sigma0 = _pad_kdd(Sig_s, present_s, K, d)
        Sigma_list.append(Sigma0)

        # η(source) via SAME path (jitter + Cholesky)
        mu0_t = torch.as_tensor(mu0, device=device, dtype=mus_s.dtype)
        Sig0_t = torch.as_tensor(Sigma0, device=device, dtype=mus_s.dtype)
        I = torch.eye(d, device=device, dtype=mus_s.dtype)

        for k in range(K):
            if not present_s[k]:
                continue
            Sigk = 0.5 * (Sig0_t[k] + Sig0_t[k].T) + float(jitter) * I
            L = torch.linalg.cholesky(Sigk)
            Lam = torch.cholesky_inverse(L)
            eta1_src[k]  = (Lam @ mu0_t[k]).detach().cpu().numpy()
            eta2d_src[k] = (-0.5 * torch.diag(Lam)).detach().cpu().numpy()

        eta1_steps.append(eta1_src.copy())
        eta2d_steps.append(eta2d_src.copy())
    else:
        v = np.clip(var0, float(jitter), None)
        eta1_src = mu0 / v
        eta2d_src = -0.5 / v
        eta1_steps.append(eta1_src.copy())
        eta2d_steps.append(eta2d_src.copy())

    # ---- Compute η(target) ONCE (for RHS); do NOT append yet ----
    if cov_type == "full":
        for k in range(K):
            if not present_t[k]:
                continue
            S1 = 0.5 * (_to_np(Sig_t[k]) + _to_np(Sig_t[k]).T)
            L1 = np.linalg.cholesky(S1 + float(jitter) * np.eye(d))
            Linv1 = np.linalg.inv(L1)
            Lam1 = Linv1.T @ Linv1
            eta1_tgt[k]  = Lam1 @ _to_np(mus_t[k])
            eta2d_tgt[k] = -0.5 * np.diag(Lam1)
    else:
        for k in range(K):
            if not present_t[k]:
                continue
            v1 = np.clip(_to_np(vars_t[k]), 1e-12 if jitter == 0 else float(jitter), None)
            eta1_tgt[k]  = _to_np(mus_t[k]) / v1
            eta2d_tgt[k] = -0.5 / v1

    # ---------------- intermediates ----------------
    all_domains: List[DomainDataset] = []
    for i in range(1, n_inter + 1):
        _t_step = time.time()
        t = i / (n_inter + 1)

        # allocate class counts proportionally to source (masked by presence in both)
        pi_mid = (counts_s.float() / max(1, total_s))
        desired = (pi_mid * float(total_min)).clamp_min(0.0)
        base = torch.floor(desired)
        frac = desired - base
        present = ((counts_s > 0) & (counts_t > 0)).float()
        base *= present
        frac *= present
        n_alloc = int(base.sum().item())
        rem = int(max(0, total_min - n_alloc))
        if rem > 0:
            k_take = min(rem, K)
            if k_take > 0:
                _, idx = torch.topk(frac, k=k_take)
                add = torch.zeros_like(base); add[idx] = 1.0
                base = base + add
        n_per_class = base.long().cpu().numpy().astype(np.int64)

        mu_mid_full = np.full((K, d), np.nan, dtype=np.float64)
        var_mid_full = np.full((K, d), np.nan, dtype=np.float64)
        Sig_mid_full = np.full((K, d, d), np.nan, dtype=np.float64) if cov_type == "full" else None
        step_eta1 = np.full((K, d), np.nan, dtype=np.float64)
        step_eta2d = np.full((K, d), np.nan, dtype=np.float64)

        Zm_list, Ym_list = [], []

        for k_idx in range(K):
            if counts_s[k_idx].item() == 0 or counts_t[k_idx].item() == 0:
                continue

            n_k = int(counts_s[k_idx].item())
            if n_k <= 0:
                continue

            if cov_type == "full":
                mu_mid, Sig_mid, eta1_mid, eta2_mid = natural_interp_full(
                    mus_s[k_idx], Sig_s[k_idx],
                    mus_t[k_idx], Sig_t[k_idx],
                    t, eps=jitter, debug=False,
                    compare=False, scale_alpha=1e5, return_report=False
                )

                mu_mid_full[k_idx] = _to_np(mu_mid)
                Sig_mid_np = _to_np(Sig_mid)
                var_mid_full[k_idx] = np.clip(np.diag(Sig_mid_np), 1e-12, None)
                Sig_mid_full[k_idx] = Sig_mid_np
                step_eta1[k_idx] = _to_np(eta1_mid)
                step_eta2d[k_idx] = _to_np(torch.diag(eta2_mid))

                try:
                    Zk = sample_full(mu_mid, Sig_mid, n_k)
                except RuntimeError:
                    dK = Sig_mid.shape[0]
                    Zk = sample_full(mu_mid, Sig_mid + (1e-6) * torch.eye(dK, device=Sig_mid.device), n_k)

            else:
                mu_mid, var_mid, _, _ = natural_interp_diag(
                    mus_s[k_idx], vars_s[k_idx],
                    mus_t[k_idx], vars_t[k_idx],
                    t
                )
                Zk = sample_diag(mu_mid, var_mid, n_k)

                mu_mid_full[k_idx] = _to_np(mu_mid)
                var_mid_full[k_idx] = _to_np(var_mid)
                prec = 1.0 / np.clip(var_mid_full[k_idx], 1e-12, None)
                step_eta1[k_idx] = prec * mu_mid_full[k_idx]
                step_eta2d[k_idx] = -0.5 * prec

            # append samples
            Yk = torch.full((n_k,), k_idx, device=device, dtype=torch.long)
            Zm_list.append(Zk); Ym_list.append(Yk)

        if not Zm_list:
            continue

        Zm = torch.cat(Zm_list, 0).cpu().float()
        Ym = torch.cat(Ym_list, 0).cpu().long()
        all_domains.append(DomainDataset(Zm, torch.ones(len(Ym)), Ym, Ym))

        print(f"[NATURAL] Step {i}/{n_inter}: generated {len(Ym)} samples with d={Zm.shape[1]} in {time.time()-_t_step:.2f}s")

        steps.append(float(t))
        mu_list.append(mu_mid_full)
        var_list.append(var_mid_full)
        counts_list.append(n_per_class.astype(np.int64))
        pi_list.append(pi_mid.cpu().numpy())
        if cov_type == "full":
            Sigma_list.append(Sig_mid_full)
        eta1_steps.append(step_eta1)
        eta2d_steps.append(step_eta2d)

    # ---------------- wrap target & append target η --------------------------
    try:
        X_tgt_final = xt if torch.is_tensor(xt) else torch.as_tensor(xt)
        X_tgt_final = X_tgt_final.cpu()
        Y_tgt_final = dataset_t.targets if torch.is_tensor(dataset_t.targets) \
                      else torch.as_tensor(dataset_t.targets, dtype=torch.long)
        Y_tgt_final = Y_tgt_final.cpu().long()
        Y_em_final = Yt_em.cpu().long()
        all_domains.append(DomainDataset(X_tgt_final, torch.ones(len(Y_em_final)), Y_tgt_final, Y_em_final))
    except Exception as e:
        print(f"[NATURAL] Warning: failed to wrap target with targets_em ({e}); appending raw dataset.")
        all_domains.append(dataset_t)

    pi_t = (counts_t.float() / max(1, total_t)).cpu().numpy()
    muT, varT = _pad_kd(mus_t, vars_t, present_t, K, d)
    steps.append(1.0)
    mu_list.append(muT)
    var_list.append(varT)
    counts_list.append(counts_t.cpu().numpy().astype(np.int64))
    pi_list.append(pi_t)

    if cov_type == "full":
        SigT = _pad_kdd(Sig_t, present_t, K, d)
        Sigma_list.append(SigT)

    # append target endpoint η (precomputed above)
    eta1_steps.append(eta1_tgt.copy())
    eta2d_steps.append(eta2d_tgt.copy())

    # ---------------- pack outputs ----------------
    domain_params = {
        "K": int(K),
        "d": int(d),
        "cov_type": cov_type,
        "steps": np.asarray(steps, dtype=np.float64),
        "mu": np.asarray(mu_list, dtype=np.float64),
        "var": np.asarray(var_list, dtype=np.float64),
        "counts": np.asarray(counts_list, dtype=np.int64),
        "pi": np.asarray(pi_list, dtype=np.float64),
        "present_source": present_s.astype(np.bool_),
        "present_target": present_t.astype(np.bool_),
        "eta1": np.asarray(eta1_steps, dtype=np.float64),
        "eta2_diag": np.asarray(eta2d_steps, dtype=np.float64),
    }
    if cov_type == "full":
        domain_params["Sigma"] = np.asarray(Sigma_list, dtype=np.float64)


    # ---------------- diagnostics plot (optional) ----------------
    if visualize:
        # cls = int(diagnostic_class if diagnostic_class is not None else 0)
        cls = K - 1
        steps_np = np.asarray(steps, dtype=np.float64)

        # pick the class row k=cls at each step
        mu_steps   = domain_params["mu"]      # (S, K, d)
        var_steps  = domain_params["var"]     # (S, K, d)
        has_full   = (cov_type == "full")
        if has_full:
            Sig_steps = domain_params["Sigma"]  # (S, K, d, d)

        # --- (1) total variance = trace(Σ) ---
        if has_full:
            # trace = sum of diagonal entries
            trSigma = np.array([np.trace(Sig_steps[s, cls]) for s in range(len(steps_np))], dtype=np.float64)
        else:
            trSigma = np.array([np.nansum(var_steps[s, cls]) for s in range(len(steps_np))], dtype=np.float64)

        # --- (2) mean norm ||μ||_2 ---
        mu_norm = np.array([np.linalg.norm(mu_steps[s, cls], ord=2) for s in range(len(steps_np))], dtype=np.float64)

        # --- (3) condition number κ(Σ) (full only) ---
        if has_full:
            def _kappa(S):
                # symmetric safeguard + small jitter for numerical stability
                Ssym = 0.5 * (S + S.T)
                w = np.linalg.eigvalsh(Ssym)
                w = np.clip(w, 1e-20, None)
                return float(w.max() / w.min())
            kappa = np.array([_kappa(Sig_steps[s, cls]) for s in range(len(steps_np))], dtype=np.float64)

        # ---- draw the figure ----
        ncols = 3 if has_full else 2
        fig, axes = plt.subplots(1, ncols, figsize=(16 if has_full else 12, 3.8))
        if ncols == 2:
            ax1, ax2 = axes
        else:
            ax1, ax2, ax3 = axes

        ax1.plot(steps_np, trSigma, marker='o')
        ax1.set_title("Trace(Σ)")
        ax1.set_xlabel("Interpolation step (t)")
        ax1.set_ylabel("Total Variance")
        ax1.grid(True, ls='--', alpha=0.4)

        ax2.plot(steps_np, mu_norm, marker='o')
        ax2.set_title("‖μ‖₂")
        ax2.set_xlabel("Interpolation step (t)")
        ax2.set_ylabel("Norm")
        ax2.grid(True, ls='--', alpha=0.4)

        if has_full:
            ax3.plot(steps_np, kappa, marker='o')
            ax3.set_yscale('log')
            ax3.set_title("Condition number (κ(Σ))")
            ax3.set_xlabel("Interpolation step (t)")
            ax3.set_ylabel("Condition Number (log scale)")
            ax3.grid(True, ls='--', alpha=0.4)

        fig.suptitle(f"The Effect of Natural Parameter Interpolation (Class {cls}, Dim={d})", y=1.05)
        fig.tight_layout()

        if save_path is not None:
            out_png = os.path.join(save_path, f"natural_interp_diag_class{cls}_d{d}_gen{(steps_np.max()-1):.0f}.png")
            os.makedirs(os.path.dirname(out_png), exist_ok=True)
            plt.savefig(out_png, dpi=160, bbox_inches="tight")
            print(f"[NATURAL] Saved diagnostics to {out_png}")
        else:
            plt.show()
        plt.close(fig)


    return all_domains, Yt_em.cpu().numpy(), domain_params


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
    assert cov_type in {"diag", "full"}
    print("---- FR domains (optimized) ----")

    # ---------- 0) I/O tensors, shapes, device ----------
    xs, xt = dataset_s.data, dataset_t.data
    ys     = dataset_s.targets
    if xs.ndim > 2:
        flatten = nn.Flatten()
        xs, xt = flatten(xs), flatten(xt)

    device = xs.device if torch.is_tensor(xs) else torch.device("cpu")
    Xs = xs if torch.is_tensor(xs) else torch.as_tensor(xs, device=device)
    Xt = xt if torch.is_tensor(xt) else torch.as_tensor(xt, device=device)
    Ys = ys if torch.is_tensor(ys) else torch.as_tensor(ys, device=device, dtype=torch.long)
    Yt = dataset_t.targets_em if torch.is_tensor(dataset_t.targets_em) else torch.as_tensor(dataset_t.targets_em, device=device, dtype=torch.long)

    if Ys.numel() == 0:
        return [], np.empty((0,), dtype=np.int64), {}

    K = int(max(Ys.max(), Yt.max()).item()) + 1
    d = int(Xs.shape[1])
    # breakpoint()
    # ---------- 1) Endpoints stats (once) ----------
    if cov_type == "diag":
        mu_s, var_s, _ = class_stats_diag(Xs, Ys, K)              # (K,d)
        mu_t, var_t, _ = class_stats_diag(Xt, Yt, K)
        Sig_s = Sig_t = None
    else:
        mu_s, Sig_s, _ = class_stats_full(Xs, Ys, K, reg=reg, ddof=ddof)    # (K,d),(K,d,d)
        mu_t, Sig_t, _ = class_stats_full(Xt, Yt, K, reg=reg, ddof=ddof)
        var_s = torch.stack([torch.diag(Sig_s[k]) for k in range(K)], dim=0)
        var_t = torch.stack([torch.diag(Sig_t[k]) for k in range(K)], dim=0)

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

    # class allocation used at all steps (your behavior: use source priors)
    pi_mid = counts_s.float() / max(1, total_s)
    desired = (pi_mid * float(total_min)).clamp_min(0.0)
    base    = torch.floor(desired)
    frac    = desired - base
    base    = base * present_both
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

    # helper to compute natural params from (μ, var) or (μ, Σ)
    def _eta_diag(mu_kd: torch.Tensor, var_kd: torch.Tensor, eps=1e-12):
        var = var_kd.clamp_min(eps)
        prec = 1.0 / var
        eta1 = prec * mu_kd
        eta2d = -0.5 * prec
        return eta1, eta2d

    def _eta_full(mu_kd: torch.Tensor, Sig_kdd: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Robust natural params for full-cov Gaussians, batched over classes.
        Sig_kdd may be only PSD; we project to SPD and invert stably.
        Returns:
            eta1: (K,d), eta2d: (K,d) with eta2d = diag(-0.5 * Λ).
        """
        # 1) symmetrize once
        Sig = _sym(Sig_kdd)

        # 2) project to SPD using relative floor in eig-space (float64), then cast back
        Sig_spd = _spd_from_eig(Sig, rel_floor=1e-8, abs_floor=1e-12)

        # 3) cholesky inverse with adaptive diagonal nudge if needed
        Lam, L = _chol_inv_spd(Sig_spd, base_rel=1e-12, max_tries=5)

        # 4) natural parameters
        eta1 = torch.einsum("kdd,kd->kd", Lam, mu_kd)
        eta2d = -0.5 * Lam.diagonal(dim1=-2, dim2=-1)
        return eta1, eta2d


    # ---------- 3) Cache FR eigen structure for full cov ----------
    if cov_type == "full":
        # per class: Σs^{1/2} and Σs^{-1/2} via eig; A = S^{-1/2} Σt S^{-1/2} = V diag(λ) V^T
        # cache: (Vs, sqrt_ws), (inv_sqrt), and (V_A, lam_A)
        cache = []
        I = torch.eye(d, device=device, dtype=Sig_s.dtype)
        # for k in range(K):
        #     if not present_both[k]:
        #         cache.append(None)
        #         continue
        #     # eig Σs
        #     ws, Vs = torch.linalg.eigh(0.5*(Sig_s[k]+Sig_s[k].T) + jitter*I)
        #     ws = ws.clamp_min(jitter)
        #     sqrt_ws = ws.sqrt()
        #     inv_sqrt_ws = 1.0 / sqrt_ws
        #     S_half      = (Vs * sqrt_ws) @ Vs.T
        #     S_mhalf     = (Vs * inv_sqrt_ws) @ Vs.T
        #     A = S_mhalf @ Sig_t[k] @ S_mhalf
        #     la, Va = torch.linalg.eigh(0.5*(A+A.T) + jitter*I)
        #     la = la.clamp_min(jitter)
        #     cache.append((S_half, S_mhalf, Va, la))
        # ----- Cache (per class) -----
        for k in range(K):
            if not present_both[k]:
                cache.append(None); continue

            S_k = 0.5*(Sig_s[k] + Sig_s[k].T)
            T_k = 0.5*(Sig_t[k] + Sig_t[k].T)

            # Choose scalars (unit trace is simple & robust)
            alpha_k = (torch.trace(S_k) / d).clamp_min(torch.finfo(S_k.dtype).eps).item()
            beta_k  = (torch.trace(T_k) / d).clamp_min(torch.finfo(T_k.dtype).eps).item()

            S_hat = S_k / alpha_k
            T_hat = T_k / beta_k

            # Eig of S_hat
            ws, Vs = torch.linalg.eigh(S_hat)
            ws = ws.clamp_min(torch.finfo(ws.dtype).eps)  # keep strictly positive
            S_half  = (Vs * ws.sqrt()) @ Vs.T            # S_hat^{1/2}
            S_mhalf = (Vs * (1.0/ws.sqrt())) @ Vs.T      # S_hat^{-1/2}

            # A_hat = S_hat^{-1/2} T_hat S_hat^{-1/2}
            A_hat = S_mhalf @ T_hat @ S_mhalf
            la, Va = torch.linalg.eigh(0.5*(A_hat + A_hat.T))
            la = la.clamp_min(torch.finfo(la.dtype).eps)

            # stash everything + (alpha_k, beta_k)
            cache.append((S_half, S_mhalf, Va, la, alpha_k, beta_k))

    else:
        cache = None

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
            mu_mid = (1.0 - t) * mu_s + t * mu_t
            # FR diag: Σ(t) = exp((1-t)log Σs + t log Σt) ⇒ var_mid = var_s^{1-t} * var_t^{t}
            var_mid = (var_s.clamp_min(jitter).log() * (1.0 - t) + var_t.clamp_min(jitter).log() * t).exp()
            # natural params
            e1, e2d = _eta_diag(mu_mid, var_mid)

            mu_out[i]  = mu_mid.double().cpu()
            var_out[i] = var_mid.double().cpu()
            cnt_out[i] = n_per_class.cpu()
            pi_out[i]  = (counts_s.double() / max(1, total_s)).cpu()
            eta1_out[i]  = e1.double().cpu()
            eta2d_out[i] = e2d.double().cpu()

            # sampling (vectorized)
            n_k = n_per_class.cpu().numpy()
            if n_k.sum() > 0:
                # sample per class in one pass
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
            # full FR: Σ(t) = S^{1/2} (V diag(la^t) V^T) S^{1/2}
            mu_mid = (1.0 - t) * mu_s + t * mu_t
            Sig_mid = torch.empty_like(Sig_s)
            for k in range(K):
                if not present_both[k]:
                    Sig_mid[k].fill_(float("nan")); continue

                S_half, S_mhalf, Va, la, alpha_k, beta_k = cache[k]

                # normalized FR in the hat-space
                la_t = la ** t
                Lt = (Va * la_t) @ Va.T
                Sig_hat = S_half @ Lt @ S_half
                Sig_hat = 0.5*(Sig_hat + Sig_hat.T)

                # exact rescale back to original: c(t) = alpha^{1-t} * beta^t
                c_t = (alpha_k ** (1.0 - t)) * (beta_k ** t)
                Sig_k = Sig_hat * c_t

                Sig_mid[k] = 0.5*(Sig_k + Sig_k.T)
            # Sig_mid = torch.empty_like(Sig_s)
            # for k in range(K):
            #     if not present_both[k]:
            #         Sig_mid[k].fill_(float("nan"))
            #         continue
            #     S_half, S_mhalf, Va, la = cache[k]
            #     Lt = (Va * (la ** t)) @ Va.T
            #     Sig_k = S_half @ Lt @ S_half
            #     Sig_k = 0.5*(Sig_k + Sig_k.T) + jitter*torch.eye(d, device=device)
            #     Sig_mid[k] = Sig_k
            # natural params
            e1, e2d = _eta_full(mu_mid, Sig_mid)

            mu_out[i]   = mu_mid.double().cpu()
            var_out[i]  = torch.stack([torch.diag(Sig_mid[k]) for k in range(K)], dim=0).double().cpu()
            Sig_out[i]  = Sig_mid.double().cpu()
            cnt_out[i]  = n_per_class.cpu()
            pi_out[i]   = (counts_s.double() / max(1, total_s)).cpu()
            eta1_out[i]  = e1.double().cpu()
            eta2d_out[i] = e2d.double().cpu()

            # sampling (per class with cached Cholesky)
            n_k = n_per_class.cpu().numpy()
            if n_k.sum() > 0:
                zs, ys_cls = [], []
                for k in range(K):
                    nk = int(n_k[k])
                    if nk <= 0: continue
                    Sk = Sig_mid[k].cpu()
                    mk = mu_mid[k].cpu()
                    # Robust sampling for (potentially PSD) Sigma via eigen sqrt + jitter
                    z  = sample_full(mk, Sk, nk, jitter=jitter)
                    zs.append(z); ys_cls.append(torch.full((nk,), k, dtype=torch.long))
                if zs:
                    Zm = torch.cat(zs, 0).float()
                    Ym = torch.cat(ys_cls, 0).long()
                    Wm = torch.ones(len(Ym))
                    all_domains.append(DomainDataset(Zm, Wm, Ym, Ym))

    # ---------- 6) Target (s=S-1) ----------
    steps[-1]  = 1.0
    mu_out[-1] = mu_t.double().cpu()
    var_out[-1]= var_t.double().cpu()
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
    return all_domains, Yt.cpu().numpy(), domain_params



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

# --- Module-level cache (simple) ---
_EM_CACHE = {}  # key -> dict with {'X_std','scaler','pca'}


from typing import Optional, Tuple, Dict, Union

# ---------- helpers: normalize various init formats to ARRAYS ----------

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


# ---------- build inits (ARRAY form), pick best by LL, convert to DICT for EM ----------

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

# ---------- your fast EM, now wiring formats correctly ----------

# def run_em_on_encoded_fast(
#     X_std: np.ndarray,
#     *,
#     K: int = 10,
#     cov_type: str = "diag",
#     reg: float = 1e-6,
#     max_iter: int = 150,
#     tol: float = 1e-5,
#     n_init: int = 5,
#     subsample_init: Optional[int] = 20000,
#     warm_start: Optional[dict] = None,   # dict or (mu,Sigma,pi)
#     rng: Union[int, np.random.Generator, None] = 0,
#     verbose: bool = False,
#     return_params: bool = True,
# ):
#     # RNG
#     if rng is None:
#         rng = np.random.default_rng()
#     elif not isinstance(rng, np.random.Generator):
#         rng = np.random.default_rng(int(rng))

#     X = np.asarray(X_std, dtype=np.float64, order="C")
#     N, D = X.shape
#     reg_used = max(reg, 1e-4) if cov_type == "full" else reg

#     # subset for init
#     X_init = X if (subsample_init is None or N <= subsample_init) else X[rng.choice(N, size=subsample_init, replace=False)]

#     # build inits (ARRAY form)
#     inits = _make_inits(X_init, K=K, cov_type=cov_type, n_init=n_init, reg=reg_used, rng=rng)

#     # optional warm start (normalize to arrays) replaces first init
#     if warm_start is not None:
#         ws_mu, ws_Sig, ws_pi = _normalize_gmm_params(warm_start, cov_type=cov_type, reg=reg_used)
#         inits[0] = (ws_mu, ws_Sig, ws_pi)

#     # choose best init by quick LL (ARRAY form)
#     def quick_ll(theta):
#         mu0, Sig0, pi0 = theta
#         return estimate_log_likelihood(X, mu0, Sig0, pi0, cov_type=cov_type, reg=reg_used)

#     best_mu, best_Sig, best_pi = max(inits, key=quick_ll)

#     # convert winner to DICT form (what your EM expects)
#     mu_d, Sig_d, pi_d = _arrays_to_component_dicts(best_mu, best_Sig, best_pi, cov_type=cov_type)

#     # run your EM (unchanged implementation)
#     mu, Sigma, pi, gamma, ll_curve = em_k_gaussians_sklearn(
#         X,
#         mu_init=mu_d, Sigma_init=Sig_d, pi_init=pi_d,
#         cov_type=cov_type, max_iter=max_iter, tol=tol, reg=reg_used, verbose=verbose
#     )

#     z = gamma.argmax(axis=1)
#     out = {"X": X, "mu": mu, "Sigma": Sigma, "pi": pi, "gamma": gamma, "labels": z, "ll_curve": ll_curve}
#     return out if return_params else {"gamma": gamma, "labels": z, "ll_curve": ll_curve}

# def prepare_em_representation(
#     enc_ds,
#     *,
#     pool: str = "gap",
#     do_pca: bool = False,
#     pca_dim: int = 128,
#     scaler: Optional[StandardScaler] = None,
#     pca: Optional[PCA] = None,
#     cache_key: Optional[str] = None,
#     dtype: str = "float32",
#     rng: Union[int, np.random.Generator, None] = 0,
#     verbose: bool = False,
# ):
#     if cache_key is None:
#         cache_key = f"{pool}|pca={int(do_pca)}|dim={pca_dim}|dtype={dtype}"

#     if cache_key in _EM_CACHE:
#         c = _EM_CACHE[cache_key]
#         return c["X_std"], c["scaler"], c["pca"], cache_key

#     X = to_features_from_encoded(enc_ds, pool=pool)
#     X = X.astype(np.float32 if dtype == "float32" else np.float64, copy=False)
#     if not X.flags.c_contiguous:
#         X = np.ascontiguousarray(X)
#     if verbose:
#         print(f"[prep] Features: {X.shape} dtype={X.dtype}")

#     if scaler is None:
#         scaler = StandardScaler(with_mean=True, with_std=True)
#         X = scaler.fit_transform(X)
#     else:
#         X = scaler.transform(X)

#     _pca = None
#     if do_pca and X.shape[1] > pca_dim:
#         if not isinstance(rng, np.random.Generator):
#             rng = np.random.default_rng(int(rng) if rng is not None else None)
#         _pca = PCA(n_components=pca_dim, svd_solver="randomized",
#                    random_state=int(rng.integers(2**31-1)))
#         X = _pca.fit_transform(X)
#         if verbose:
#             print(f"[prep] PCA -> {X.shape}")

#     _EM_CACHE[cache_key] = {"X_std": X, "scaler": scaler, "pca": _pca}
#     return X, scaler, _pca, cache_key


# def run_em_on_encoded(
#     enc_ds,
#     K=10,
#     cov_type='diag',          # 'diag' | 'spherical' | 'full'
#     pool='gap',
#     do_pca=False,
#     pca_dim=64,
#     reg=1e-6,
#     max_iter=100,
#     tol=1e-5,
#     freeze_sigma_iters=0,
#     rng: Union[int, np.random.Generator, None] = 0,
#     *,
#     scaler: Optional[StandardScaler] = None,
#     pca: Optional[PCA] = None,
#     return_transforms: bool = False,
#     subsample_init: Optional[int] = 20000,
#     warm_start: Optional[dict] = None,   # may be dict or (mu,Sigma,pi)
#     verbose: bool = False,
#     cache_key: Optional[str] = None,
#     dtype: str = "float32",
#     n_init: int = 5,
#     ):

#     X_std, _scaler, _pca, key = prepare_em_representation(
#         enc_ds, pool=pool, do_pca=do_pca, pca_dim=pca_dim,
#         scaler=scaler, pca=pca, cache_key=cache_key, dtype=dtype, rng=rng, verbose=verbose
#     )

#     out = run_em_on_encoded_fast(
#         X_std,
#         K=K, cov_type=cov_type, reg=reg, max_iter=max_iter, tol=tol,
#         n_init=n_init, subsample_init=subsample_init,
#         warm_start=warm_start, rng=rng, verbose=verbose
#     )

#     if return_transforms:
#         out["scaler"] = _scaler
#         out["pca"] = _pca
#     return out



from typing import Optional, Union

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ---- Cache for prepared representations
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
        max_points=max_points
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
    """
    if cache_key is None:
        cache_key = f"{pool}|pca={int(do_pca)}|dim={pca_dim}|dtype={dtype}"

    if cache_key in _EM_CACHE:
        c = _EM_CACHE[cache_key]
        return c["X_std"], c["scaler"], c["pca"], cache_key

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
        _pca = PCA(n_components=pca_dim, svd_solver="randomized",
                   random_state=int(rng.integers(2**31-1)))
        X = _pca.fit_transform(X)
        if verbose:
            print(f"[prep] PCA -> {X.shape}")

    _EM_CACHE[cache_key] = {"X_std": X, "scaler": scaler, "pca": _pca}
    return X, scaler, _pca, cache_key


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
