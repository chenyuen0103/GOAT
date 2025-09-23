import numpy as np
from typing import Tuple, Dict, List, Optional
from sklearn.cluster import kmeans_plusplus
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import GaussianMixture
# If you kept the generalized EM/K-means++ from earlier:
# from unsup_exp import gmm_em_k, kmeanspp_init  # K-class EM + kmeans++ init
# (names below assume the ones I provided earlier; adjust to your file)
from sklearn.metrics import accuracy_score, confusion_matrix


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

    # 3) optional PCA for stability/speed
    _pca = None
    if do_pca and X.shape[1] > pca_dim:
        if pca is None:
            _pca = PCA(n_components=pca_dim, svd_solver='randomized', random_state=int(rng.integers(2**31-1)))
            X = _pca.fit_transform(X)
            if verbose: print("PCA ->", X.shape)
        else:
            _pca = pca
            X = _pca.transform(X)

    # 4) k-means++ init
    n = X.shape[0]
    if subsample_init is not None and n > subsample_init:
        idx = rng.choice(n, size=subsample_init, replace=False)
        X_init = X[idx]
    else:
        X_init = X
    mu0, Sigma0, pi0 = kmeanspp_init_params(X_init, K=K, cov_type=cov_type, reg=reg, rng=rng)

    if warm_start is not None:
        mu0 = warm_start.get('mu', mu0)
        Sigma0 = warm_start.get('Sigma', Sigma0)
        pi0 = warm_start.get('pi', pi0)

    # 5) EM
    mu, Sigma, pi, gamma, ll_curve = em_k_gaussians(
        X,
        mu_init=mu0, Sigma_init=Sigma0, pi_init=pi0,
        cov_type="diag", max_iter=max_iter, tol=tol, reg=reg, verbose=True
    )   
    # mu, Sigma, pi, gamma, ll_curve = em_k_gaussians_sklearn(
    #     X,
    #     mu_init=mu0, Sigma_init=Sigma0, pi_init=pi0,
    #     cov_type=cov_type, max_iter=max_iter, tol=tol, reg=reg, verbose=verbose
    # )

    

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

            else:
                raise ValueError(f"Unknown metric '{metric}'.")

    # Hungarian on costs
    row_ind, col_ind = linear_sum_assignment(D)
    mapping = {int(cluster_keys[i]): int(class_keys[j]) for i, j in zip(row_ind, col_ind)}
    return mapping, D


def map_em_clusters(
    res,
    *,
    method: str = "pseudolabels",
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

    if method == "pseudolabels":
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


def fit_source_gaussian_params(X, y,  pool: str = "gap") -> Tuple[Dict[int, np.ndarray], np.ndarray, Dict[int, float]]:
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

    # precisions init from Sigma_init
    precisions_init = _sigma_to_precision_init(Sigma_init, cov_type, reg=reg)

    gm = GaussianMixture(
        n_components=K,
        covariance_type=cov_type,
        tol=tol,
        reg_covar=reg,
        max_iter=max_iter,
        init_params="random",            # we provide init explicitly
        means_init=means_init,
        weights_init=weights_init,
        precisions_init=precisions_init,
        random_state=None if rng is None else int(np.random.RandomState().randint(2**31-1)),
        verbose=2 if verbose else 0
    )
    gm.fit(X)

    # posteriors (responsibilities)
    gamma = gm.predict_proba(X)                          # (n,K)

    # unpack params back to your dict format
    mu = {k: gm.means_[i].copy() for i, k in enumerate(keys)}
    pi = {k: float(gm.weights_[i]) for i, k in enumerate(keys)}

    # get covariances in a consistent (K,d,d) form then map to dict
    if cov_type == "full":
        covs = gm.covariances_                          # (K,d,d)
    elif cov_type == "diag":
        covs = np.stack([np.diag(v) for v in gm.covariances_], axis=0)  # (K,d,d)
    elif cov_type == "spherical":
        covs = np.stack([np.eye(d) * s2 for s2 in gm.covariances_], axis=0)  # (K,d,d)
    else:
        raise ValueError("cov_type must be one of {'full','diag','spherical'}")

    Sigma = {k: covs[i].copy() for i, k in enumerate(keys)}

    # sklearn exposes average log-likelihood per sample; emulate a single-point history
    ll_final = float(gm.lower_bound_) * n
    ll_history = [ll_final]

    return mu, Sigma, pi, gamma, ll_history




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


@torch.no_grad()
def class_stats_diag(Z: torch.Tensor, y: torch.Tensor, K: int):
    """
    Z: (N,d) pooled/flattened features; y: (N,)
    Returns dict k -> (mu_k (d,), var_k (d,)), and counts per class.
    """
    d = Z.shape[1]
    stats = {}
    counts = {}
    eps = 1e-6
    for k in range(K):
        Zk = Z[y == k]
        if Zk.numel() == 0:
            stats[k] = (torch.zeros(d, device=Z.device), torch.ones(d, device=Z.device))
            counts[k] = 0
        else:
            mu = Zk.mean(0)
            var = Zk.var(0, unbiased=False).clamp_min(eps)
            stats[k] = (mu, var)
            counts[k] = Zk.size(0)
    return stats, counts
