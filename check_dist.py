import math, numpy as np, torch
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ----------------------------- BBSE (label shift) -----------------------------
@dataclass
class BBSEOut:
    priors_source: np.ndarray
    priors_target_hat: np.ndarray
    conf_matrix: np.ndarray
    ok: bool
    chi2: float
    chi2_p: float

def bbse_label_shift(
    y_true_source: np.ndarray, y_pred_source: np.ndarray,
    y_pred_target: np.ndarray,
) -> BBSEOut:
    """
    Black-box shift estimator (BBSE):
      p_T(y) ≈ C^{-1} * p_T(ŷ)
    where C[i,j] = P(ŷ=i | y=j) estimated on source.
    Returns estimated target priors and a simple χ^2 GOF test vs source priors.
    """
    K = int(max(y_true_source.max(), y_pred_source.max(), y_pred_target.max()) + 1)
    # confusion: rows = predicted class i, cols = true class j
    C = np.zeros((K, K), dtype=float)
    for yi, yj in zip(y_pred_source, y_true_source):
        C[int(yi), int(yj)] += 1.0
    col_sums = C.sum(axis=0, keepdims=True) + 1e-12
    C = C / col_sums  # normalize columns -> P(ŷ=i | y=j)

    # empirical p_T(ŷ)
    p_hat_yhat_T = np.bincount(y_pred_target, minlength=K).astype(float)
    p_hat_yhat_T /= p_hat_yhat_T.sum() + 1e-12

    # solve for priors via least squares (stable if C is near singular)
    # C * p ≈ p_hat_yhat_T  => p ≈ argmin ||C p - p_hat||
    p_T = np.linalg.lstsq(C, p_hat_yhat_T, rcond=None)[0]
    p_T = np.clip(p_T, 0, None)
    if p_T.sum() > 0: p_T /= p_T.sum()
    ok = bool(np.all(np.isfinite(p_T)))

    # source priors from y_true_source
    p_S = np.bincount(y_true_source, minlength=K).astype(float)
    p_S /= p_S.sum() + 1e-12

    # Pearson χ^2 comparing p_T vs p_S (rough heuristic; small-sample caveats)
    with np.errstate(divide='ignore', invalid='ignore'):
        chi2 = np.nansum((p_T - p_S)**2 / (p_S + 1e-12))
    # very rough p-value using χ^2_{K-1}; for diagnostics only
    from scipy.stats import chi2 as chi2_dist  # if SciPy not available, set chi2_p=np.nan
    chi2_p = 1.0 - chi2_dist.cdf(chi2, df=max(K-1,1)) if 'scipy' in chi2_dist.__module__ else np.nan

    return BBSEOut(
        priors_source=p_S, priors_target_hat=p_T,
        conf_matrix=C, ok=ok, chi2=chi2, chi2_p=chi2_p
    )

# ----------------------------- MMD per class ---------------------------------
def _median_heuristic(X: np.ndarray) -> float:
    # robust bandwidth for RBF kernels
    X = X.astype(np.float64)
    if len(X) > 2000:
        idx = np.random.RandomState(0).choice(len(X), size=2000, replace=False)
        X = X[idx]
    d2 = np.sum((X[None, :] - X[:, None])**2, axis=-1)
    med = np.median(d2[d2 > 0])
    return math.sqrt(0.5 * med) if np.isfinite(med) and med > 0 else 1.0

def _rbf_kernel(X: np.ndarray, Y: np.ndarray, sigma: Optional[float]) -> np.ndarray:
    if sigma is None:
        Z = np.vstack([X, Y])
        sigma = _median_heuristic(Z)
    gamma = 1.0 / (2.0 * (sigma**2) + 1e-12)
    # ||x-y||^2 = x^2 + y^2 - 2xy
    X2 = np.sum(X*X, axis=1)[:, None]
    Y2 = np.sum(Y*Y, axis=1)[None, :]
    K = np.exp(-gamma * (X2 + Y2 - 2.0 * (X @ Y.T)))
    return K

def mmd2_unbiased(X: np.ndarray, Y: np.ndarray, sigma: Optional[float]=None) -> float:
    Kxx = _rbf_kernel(X, X, sigma)
    Kyy = _rbf_kernel(Y, Y, sigma)
    Kxy = _rbf_kernel(X, Y, sigma)
    n = X.shape[0]; m = Y.shape[0]
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)
    term_x = Kxx.sum() / (n*(n-1) + 1e-12)
    term_y = Kyy.sum() / (m*(m-1) + 1e-12)
    term_xy = 2.0 * Kxy.mean()
    return term_x + term_y - term_xy

def mmd_permutation_pvalue(X: np.ndarray, Y: np.ndarray, B: int = 500, sigma: Optional[float]=None, rng: Optional[np.random.RandomState]=None) -> Tuple[float,float]:
    """Returns (statistic, pvalue)."""
    rng = rng or np.random.RandomState(0)
    stat = mmd2_unbiased(X, Y, sigma)
    Z = np.vstack([X, Y])
    n = len(X)
    cnt = 0
    for _ in range(B):
        idx = rng.permutation(len(Z))
        Xb, Yb = Z[idx[:n]], Z[idx[n:]]
        sb = mmd2_unbiased(Xb, Yb, sigma)
        cnt += (sb >= stat)
    p = (cnt + 1) / (B + 1)
    return float(stat), float(p)

# ----------------------------- C2ST per class --------------------------------
def c2st_auc_bootstrap(
    Xs: np.ndarray, Xt: np.ndarray, B: int = 300, seed: int = 0
) -> Tuple[float, Tuple[float,float]]:
    """
    Train a simple LR domain classifier to separate source vs target.
    Return held-out AUC and a bootstrap 95% CI.
    """
    rng = np.random.RandomState(seed)
    X = np.vstack([Xs, Xt]).astype(np.float64)
    y = np.array([0]*len(Xs) + [1]*len(Xt))
    # single split (stratified), LR with scaling
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(X, y):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr]); Xte = scaler.transform(X[te])
        clf = LogisticRegression(max_iter=200, n_jobs=1)
        clf.fit(Xtr, y[tr])
        proba = clf.predict_proba(Xte)[:,1]
        aucs.append(roc_auc_score(y[te], proba))
    auc = float(np.mean(aucs))

    # bootstrap CI over the pooled set, with stratified resampling
    nS, nT = len(Xs), len(Xt)
    auc_bs = []
    for _ in range(B):
        idxS = rng.choice(nS, nS, replace=True)
        idxT = rng.choice(nT, nT, replace=True)
        Xb = np.vstack([Xs[idxS], Xt[idxT]])
        yb = np.array([0]*nS + [1]*nT)
        scaler = StandardScaler()
        Xb = scaler.fit_transform(Xb)
        clf = LogisticRegression(max_iter=200, n_jobs=1)
        # simple holdout inside bootstrap
        mask = rng.rand(nS+nT) < 0.7
        clf.fit(Xb[mask], yb[mask])
        proba = clf.predict_proba(Xb[~mask])[:,1]
        auc_bs.append(roc_auc_score(yb[~mask], proba))
    lo, hi = np.percentile(auc_bs, [2.5, 97.5])
    return auc, (float(lo), float(hi))

# ------------------------ Energy distance per class ---------------------------
def energy_statistic(X: np.ndarray, Y: np.ndarray) -> float:
    # E = 2 E||X-Y|| - E||X-X'|| - E||Y-Y'||  (empirical)
    from numpy.linalg import norm
    X = X.astype(np.float64); Y = Y.astype(np.float64)
    n, m = len(X), len(Y)
    dxy = np.mean([norm(X[i]-Y[j]) for i in range(n) for j in range(m)])
    dxx = np.mean([norm(X[i]-X[j]) for i in range(n) for j in range(n) if i!=j])
    dyy = np.mean([norm(Y[i]-Y[j]) for i in range(m) for j in range(m) if i!=j])
    return 2*dxy - dxx - dyy

def energy_permutation_pvalue(X: np.ndarray, Y: np.ndarray, B: int = 500, seed: int = 0) -> Tuple[float,float]:
    rng = np.random.RandomState(seed)
    stat = energy_statistic(X, Y)
    Z = np.vstack([X, Y])
    n = len(X)
    cnt = 0
    for _ in range(B):
        idx = rng.permutation(len(Z))
        Xb, Yb = Z[idx[:n]], Z[idx[n:]]
        sb = energy_statistic(Xb, Yb)
        cnt += (sb >= stat)
    p = (cnt + 1) / (B + 1)
    return float(stat), float(p)

# ------------------------ Wrapper to run all classwise tests ------------------
@dataclass
class ClasswiseTestResult:
    nS: int; nT: int
    mmd_stat: float; mmd_p: float
    c2st_auc: float; c2st_ci: Tuple[float,float]
    energy_stat: float; energy_p: float

def run_classwise_tests(
    Zs: np.ndarray, ys: np.ndarray, Zt: np.ndarray, yt: np.ndarray,
    classes: Optional[List[int]] = None,
    B_perm: int = 500, seed: int = 0, use_pca: Optional[object] = None
) -> Dict[int, ClasswiseTestResult]:
    """
    Zs, Zt : feature arrays (encoded; optionally transformed by use_pca)
    ys, yt : integer labels
    """
    if use_pca is not None:
        Zs = use_pca.transform(Zs)
        Zt = use_pca.transform(Zt)

    if classes is None:
        classes = sorted(set(ys.tolist()) & set(yt.tolist()))
    out: Dict[int, ClasswiseTestResult] = {}
    rng = np.random.RandomState(seed)

    for c in classes:
        Xs = Zs[ys == c]
        Xt = Zt[yt == c]
        if len(Xs) < 5 or len(Xt) < 5:
            # too few to test robustly
            out[c] = ClasswiseTestResult(
                nS=len(Xs), nT=len(Xt),
                mmd_stat=np.nan, mmd_p=np.nan,
                c2st_auc=np.nan, c2st_ci=(np.nan, np.nan),
                energy_stat=np.nan, energy_p=np.nan
            )
            continue
        # MMD
        mmd_stat, mmd_p = mmd_permutation_pvalue(Xs, Xt, B=B_perm, sigma=None, rng=rng)
        # C2ST
        auc, ci = c2st_auc_bootstrap(Xs, Xt, B=200, seed=seed)
        # Energy
        e_stat, e_p = energy_permutation_pvalue(Xs, Xt, B=B_perm, seed=seed)
        out[c] = ClasswiseTestResult(
            nS=len(Xs), nT=len(Xt),
            mmd_stat=mmd_stat, mmd_p=mmd_p,
            c2st_auc=auc, c2st_ci=ci,
            energy_stat=e_stat, energy_p=e_p
        )
    return out
