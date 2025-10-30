# ============================================================
# Unified: Expansion & Self-Training (Parametric / OT / ExpFam)
# ============================================================

import math
import heapq
import copy
from typing import Tuple, Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

import ot  # POT (Python Optimal Transport)

# =========================================
# Globals / Config
# =========================================
METRICS = ("FR", "W2", "eta")
GRID_N = 201                 # λ-grid resolution for A*
NEIGHBOR_SPAN = 2            # 4/8-neighborhood span
D = 1
RNG = np.random.default_rng(7)



def eval_final_on_target(run, X_test, Y_test):
    """Use the last snapshot's classifier (or fall back to source clf if none)."""
    if run.get("snapshots"):
        clf = run["snapshots"][-1]["clf"]
    else:
        raise ValueError("Run has no snapshots to evaluate.")
    P = clf.predict_proba(X_test)[:,1]
    return {
        "acc": accuracy_score(Y_test, (P >= 0.5).astype(int)),
        "ce":  log_loss(Y_test, P, labels=[0,1]),
        "auc": roc_auc_score(Y_test, P)
    }

def summarize_final(runs_dict, X_test, Y_test):
    rows = []
    for name, run in runs_dict.items():
        rows.append((name, eval_final_on_target(run, X_test, Y_test)))
    return rows

# Example after you’ve computed your runs:
# runs_parametric: dict like {"FR": run_FR, "W2": run_W2, "eta": run_eta}
# run_ot: a single run dict
# run_expFam: a single run dict


# =========================================
# Sinkhorn helpers
# =========================================
def sinkhorn_barycentric_map(X_src: np.ndarray,
                             X_tgt: np.ndarray,
                             reg: float = 0.05,
                             a: Optional[np.ndarray] = None,
                             b: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute barycentric map T(x_i) ≈ sum_j pi_ij y_j / sum_j pi_ij
    using entropic-regularized OT (Sinkhorn) between two empirical clouds.
    """
    n, m = len(X_src), len(X_tgt)
    if a is None: a = np.full(n, 1.0/n)
    if b is None: b = np.full(m, 1.0/m)
    M = ot.dist(X_src, X_tgt, metric='euclidean')**2
    P = ot.sinkhorn(a, b, M, reg)
    denom = P.sum(axis=1, keepdims=True) + 1e-12
    return (P @ X_tgt) / denom

def sinkhorn_displacement_interpolate(X_src: np.ndarray,
                                      X_tgt: np.ndarray,
                                      t: float,
                                      reg: float = 0.05,
                                      a: Optional[np.ndarray] = None,
                                      b: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Return {x_i^(t)} = (1-t) x_i + t T(x_i), where T is barycentric map.
    """
    T = sinkhorn_barycentric_map(X_src, X_tgt, reg=reg, a=a, b=b)
    return (1.0 - t) * X_src + t * T


import numpy as np
import ot
from typing import Tuple, Optional

def interpolate_classwise_sinkhorn_unlabeled(
    X0: np.ndarray, y0: np.ndarray, X1: np.ndarray,
    t: float, reg: float = 0.05,
    clf=None,                      # any prob. classifier with predict_proba
    temperature: float = 1.0,      # >1 softens, <1 sharpens weights
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Classwise Sinkhorn without target labels:
    - build two OT plans from each source class to the whole unlabeled target,
      using soft target weights from current classifier probabilities.
    - return interpolated samples and keep source labels.
    """
    assert clf is not None, "Provide a classifier to score target samples."

    # soft target weights from the current model
    p1 = np.clip(clf.predict_proba(X1)[:, 1], 1e-6, 1-1e-6)
    if temperature != 1.0:
        p1 = p1**(1.0/temperature);  p1 /= p1.sum()
        p0 = (1.0-p1); p0 /= p0.sum()
    else:
        p0 = 1.0 - p1
        p1 = p1 / p1.sum()
        p0 = p0 / p0.sum()

    Xt_list, yt_list = [], []
    for c, bw in [(0, p0), (1, p1)]:
        X0c = X0[y0 == c]
        if len(X0c) == 0:
            continue
        a = np.full(len(X0c), 1.0/len(X0c))
        # cost = squared Euclidean (W2)
        M = ot.dist(X0c, X1, metric='euclidean')**2
        P = ot.sinkhorn(a, bw, M, reg)

        denom = P.sum(axis=1, keepdims=True) + 1e-12
        T = (P @ X1) / denom                       # barycentric map per source point
        Xtc = (1.0 - t) * X0c + t * T              # displacement interpolation
        Xt_list.append(Xtc)
        yt_list.append(np.full(len(X0c), c, dtype=y0.dtype))

    if not Xt_list:
        return np.empty((0, X0.shape[1])), np.empty((0,), dtype=y0.dtype)
    return np.vstack(Xt_list), np.concatenate(yt_list)

def interpolate_classwise_sinkhorn(X0: np.ndarray, y0: np.ndarray,
                                   X1: np.ndarray, y1: np.ndarray,
                                   t: float, reg: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate source→target per class using Sinkhorn; keep source labels.
    """
    classes = np.unique(np.concatenate([y0, y1]))
    outX, outy = [], []
    for c in classes:
        X0c = X0[y0 == c]; X1c = X1[y1 == c]
        if len(X0c) == 0 or len(X1c) == 0:
            continue
        Tc = sinkhorn_barycentric_map(X0c, X1c, reg=reg)
        Xtc = (1.0 - t) * X0c + t * Tc
        outX.append(Xtc)
        outy.append(np.full(len(Xtc), c, dtype=y0.dtype))
    if not outX:
        return np.empty((0, X0.shape[1])), np.empty((0,), dtype=y0.dtype)
    return np.vstack(outX), np.concatenate(outy)


# =========================================
# A* on (mu, lambda) grid (1D Normal param pathing)
# =========================================
def build_grid_mu_lambda(mu0, mu1, lam0, lam1, n_mu, n_lam):
    mus  = np.linspace(min(mu0, mu1), max(mu0, mu1), n_mu)
    lams = np.linspace(min(lam0, lam1), max(lam0, lam1), n_lam)
    return mus, lams

def a_star_mu_lambda_path(
    metric: str,
    mu0: float, lam0: float,        # start
    mu1: float, lam1: float,        # goal
    mus: np.ndarray,                # 1D grid for mu (length n_mu)
    lams: np.ndarray,               # 1D grid for lambda (length n_lam; values > 0)
    K: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    A* in (mu, lambda)-space on a 2D grid using the chosen metric.
    Returns (mu_path, lam_path, t_arc, total_length).

    metric in {"FR", "W2", "eta", "OT"}:
      - FR (1D Normal): local length^2 = (dmu^2)/lam_bar + 0.5*(dloglam^2)
      - W2 (1D Normal): sqrt( (Δmu)^2 + (sqrt(lam')-sqrt(lam))^2 )
      - eta: Euclidean in natural params theta=(mu/lam, -1/(2*lam))
      - OT: alias of W2 here (pathing only; sampling may use Sinkhorn)
    """
    from math import sqrt, log, hypot

    n_mu, n_lam = len(mus), len(lams)

    def snap_idx(arr, x):
        return int(np.argmin(np.abs(arr - x)))

    def neighbors(i, j):
        for di in range(-K, K+1):
            for dj in range(-K, K+1):
                if di == 0 and dj == 0: continue
                ii, jj = i + di, j + dj
                if 0 <= ii < n_mu and 0 <= jj < n_lam:
                    yield ii, jj

    def cost_FR(mu, lam, mu2, lam2):
        lam = max(lam, 1e-300); lam2 = max(lam2, 1e-300)
        lam_bar = sqrt(lam * lam2)
        dmu = mu2 - mu
        dloglam = log(lam2) - log(lam)
        return sqrt((dmu * dmu) / lam_bar + 0.5 * (dloglam * dloglam))

    def cost_W2(mu, lam, mu2, lam2):
        return hypot(mu2 - mu, sqrt(max(lam2, 0.0)) - sqrt(max(lam, 0.0)))

    def cost_eta(mu, lam, mu2, lam2):
        lam = max(lam, 1e-300); lam2 = max(lam2, 1e-300)
        th1, th2 = (mu / lam), (-0.5 / lam)
        th1b, th2b = (mu2 / lam2), (-0.5 / lam2)
        return hypot(th1b - th1, th2b - th2)

    if metric == "FR":
        edge_cost = cost_FR
    elif metric == "W2":
        edge_cost = cost_W2
    elif metric == "eta":
        edge_cost = cost_eta
    elif metric == "OT":
        edge_cost = cost_W2
    else:
        raise ValueError(f"Unknown metric: {metric}")

    si, sj = snap_idx(mus, mu0), snap_idx(lams, lam0)
    gi, gj = snap_idx(mus, mu1), snap_idx(lams, lam1)
    mu_goal, lam_goal = mus[gi], lams[gj]

    def h(mu, lam):
        return edge_cost(mu, lam, mu_goal, lam_goal)

    INF = float("inf")
    g_cost = np.full((n_mu, n_lam), INF, dtype=float)
    parent = { (si, sj): None }
    g_cost[si, sj] = 0.0
    start_f = g_cost[si, sj] + h(mus[si], lams[sj])
    pq: List[Tuple[float, int, int]] = [(start_f, si, sj)]
    closed = np.zeros((n_mu, n_lam), dtype=bool)

    while pq:
        f_u, ui, uj = heapq.heappop(pq)
        if closed[ui, uj]: continue
        closed[ui, uj] = True
        if (ui, uj) == (gi, gj): break
        mu_u, lam_u = mus[ui], lams[uj]
        for vi, vj in neighbors(ui, uj):
            mu_v, lam_v = mus[vi], lams[vj]
            w = edge_cost(mu_u, lam_u, mu_v, lam_v)
            tentative = g_cost[ui, uj] + w
            if tentative < g_cost[vi, vj]:
                g_cost[vi, vj] = tentative
                parent[(vi, vj)] = (ui, uj)
                f_v = tentative + h(mu_v, lam_v)
                heapq.heappush(pq, (f_v, vi, vj))

    assert (gi, gj) in parent or (gi, gj) == (si, sj), "A*: path not found"
    idx_path = []
    cur = (gi, gj)
    while cur is not None:
        idx_path.append(cur)
        cur = parent.get(cur, None)
    idx_path.reverse()

    mu_path = np.array([mus[i] for i, j in idx_path], dtype=float)
    lam_path = np.array([lams[j] for i, j in idx_path], dtype=float)

    seg = [0.0]
    for k in range(1, len(idx_path)):
        (i0, j0), (i1, j1) = idx_path[k-1], idx_path[k]
        seg.append(seg[-1] + edge_cost(mus[i0], lams[j0], mus[i1], lams[j1]))
    seg = np.asarray(seg, dtype=float)
    total = float(seg[-1]) if len(seg) else 1.0
    t_arc = seg / total if total > 0 else seg
    return mu_path, lam_path, t_arc, total


# =========================================
# 1D Gaussian generator & FR geodesic (analytic)
# =========================================
def fr_geodesic_mu_lambda(t: float, mu0: float, lam0: float, mu1: float, lam1: float):
    """Exact Fisher–Rao geodesic for 1D Normal between (mu0, lam0) and (mu1, lam1)."""
    def to_xy(mu, lam):
        sigma = math.sqrt(lam)
        x = mu
        y = math.sqrt(2.0) * sigma
        return x, y

    x0, y0 = to_xy(mu0, lam0)
    x1, y1 = to_xy(mu1, lam1)

    if abs(x0 - x1) < 1e-15:
        x_t = x0
        y_t = y0 * (y1 / y0) ** t
        mu_t = x_t
        sigma_t = y_t / math.sqrt(2.0)
        lam_t = sigma_t ** 2
        return mu_t, lam_t

    c = (x0 * x0 + y0 * y0 - x1 * x1 - y1 * y1) / (2.0 * (x0 - x1))
    R = math.hypot(x0 - c, y0)
    theta0 = math.atan2(y0, x0 - c)
    theta1 = math.atan2(y1, x1 - c)
    T0 = max(math.tan(0.5 * theta0), 1e-300)
    T1 = max(math.tan(0.5 * theta1), 1e-300)
    Tt = (T0 ** (1.0 - t)) * (T1 ** t)
    theta_t = 2.0 * math.atan(Tt)
    x_t = c + R * math.cos(theta_t)
    y_t = R * math.sin(theta_t)
    mu_t = x_t
    sigma_t = y_t / math.sqrt(2.0)
    lam_t = sigma_t ** 2
    return mu_t, lam_t


# =========================================
# Sampling & source training (1D)
# =========================================
def sample_dataset_with_means(mu_left: float, mu_right: float, lam_val: float,
                              n: int, d: int = D, rng=None, pi: float = 0.5,
                              ensure_both: bool = True):
    rng = np.random.default_rng() if rng is None else rng
    if lam_val <= 0:
        raise ValueError("lam_val must be > 0.")
    y = (rng.random(n) < pi).astype(int)
    if ensure_both and n >= 2 and (y.min() == y.max()):
        y[0], y[1] = 0, 1
    means = np.zeros((n, d), dtype=float)
    means[:, 0] = np.where(y == 0, mu_left, mu_right)
    Sigma = np.eye(d)
    Sigma[0, 0] = lam_val
    X = rng.multivariate_normal(mean=np.zeros(d), cov=Sigma, size=n) + means
    return X.astype(float), y

def train_source_model(n_source: int,
                       mu_left0: float,
                       mu_right0: float,
                       lam_source: float,
                       rng=RNG) -> Tuple[np.ndarray, np.ndarray, LogisticRegression]:
    Xs, Ys = sample_dataset_with_means(mu_left=mu_left0, mu_right=mu_right0,
                                       lam_val=lam_source, n=n_source, rng=rng)
    clf_src = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=2000)
    clf_src.fit(Xs, Ys)
    return Xs, Ys, clf_src


# =========================================
# Exponential-family (Gaussian shared-cov) helpers
# =========================================
def fit_source_gaussian_params(X: np.ndarray, y: np.ndarray) -> Tuple[Dict[int, np.ndarray], np.ndarray, Dict[int, float]]:
    """Estimate per-class means and shared covariance (pooled) + class priors from labeled source."""
    classes = np.unique(y)
    d = X.shape[1]
    mus = {}
    priors = {}
    for c in classes:
        Xc = X[y == c]
        mus[c] = Xc.mean(axis=0)
        priors[c] = float(len(Xc)) / len(X)
    Sigma = np.zeros((d, d))
    for c in classes:
        Xc = X[y == c]
        Z = Xc - mus[c]
        Sigma += (Z.T @ Z)
    Sigma /= (len(X) - len(classes))
    return mus, Sigma, priors

import numpy as np

def init_means_kmeanspp(X, rng=None, reorder_dim=0):
    """
    Return k-means++ initial means for K=2 as a dict {0: mu_left, 1: mu_right}.
    Also returns pi_init estimated from one hard assignment to these centers.
    """
    from sklearn.cluster import kmeans_plusplus

    n, d = X.shape
    # make an int seed that sklearn accepts
    if hasattr(rng, "integers"):
        seed = int(rng.integers(2**31 - 1))
    elif isinstance(rng, (int, np.integer)) or rng is None:
        seed = rng
    else:
        seed = None

    centers, indices = kmeans_plusplus(X, n_clusters=2, random_state=seed)

    # optional: sort by a coordinate so "left" < "right"
    order = np.argsort(centers[:, reorder_dim])
    centers_sorted = centers[order]

    # estimate priors by one k-means assignment
    d2 = ((X[:, None, :] - centers[None, :, :])**2).sum(axis=2)
    labels = d2.argmin(axis=1)                    # labels in original center order
    counts = np.bincount(labels, minlength=2).astype(float)
    counts_sorted = counts[order]

    mu_init = {0: centers_sorted[0], 1: centers_sorted[1]}
    pi_init = {0: counts_sorted[0] / n, 1: counts_sorted[1] / n}
    return mu_init, pi_init


def em_two_gaussians_shared_cov(X: np.ndarray,
                                mu_init: Dict[int, np.ndarray],
                                Sigma_init: np.ndarray,
                                pi_init: Dict[int, float],
                                max_iter: int = 100,
                                freeze_sigma_iters=25,                        # <-- important
                                tol: float = 1e-5,
                                reg: float = 1e-6) -> Tuple[Dict[int, np.ndarray], np.ndarray, Dict[int, float]]:
    """
    EM for 2-component Gaussian mixture with shared covariance.
    Returns estimated {mu_c}, shared Sigma, and priors {pi_c}.
    """
    n, d = X.shape
    classes = sorted(mu_init.keys())
    c0, c1 = classes[0], classes[1]

    mu = {c0: mu_init[c0].copy(), c1: mu_init[c1].copy()}
    Sigma = Sigma_init.copy()
    pi = {c0: float(pi_init[c0]), c1: float(pi_init[c1])}

    def logpdf_gauss_shared(X_, mu_c, Sigma_, reg_=1e-6):
        U, s, Vt = np.linalg.svd(Sigma_ + reg_*np.eye(d), full_matrices=False)
        logdet = np.sum(np.log(s))
        Sinv = (Vt.T * (1.0 / s)) @ U.T
        Z = X_ - mu_c
        q = np.einsum('nd,dd,nd->n', Z, Sinv, Z)
        return -0.5*(d*np.log(2*np.pi) + logdet + q)

    ll_old = -np.inf
    for it in range(max_iter):
        l0 = np.log(pi[c0] + 1e-12) + logpdf_gauss_shared(X, mu[c0], Sigma, reg)
        l1 = np.log(pi[c1] + 1e-12) + logpdf_gauss_shared(X, mu[c1], Sigma, reg)
        m = np.maximum(l0, l1)
        r0 = np.exp(l0 - m); r1 = np.exp(l1 - m)
        ssum = r0 + r1 + 1e-12
        gamma0 = r0 / ssum
        gamma1 = r1 / ssum

        N0 = gamma0.sum(); N1 = gamma1.sum()
        pi[c0] = float(N0 / n); pi[c1] = float(N1 / n)
        mu[c0] = (gamma0[:, None] * X).sum(axis=0) / max(N0, 1e-12)
        mu[c1] = (gamma1[:, None] * X).sum(axis=0) / max(N1, 1e-12)
        Z0 = X - mu[c0]; Z1 = X - mu[c1]
        S0 = (gamma0[:, None] * Z0).T @ Z0
        S1 = (gamma1[:, None] * Z1).T @ Z1
        Sigma = (S0 + S1) / max(n, 1e-12) + reg * np.eye(d)
        if it >= freeze_sigma_iters:
            Sigma = Sigma_init

        ll = np.sum(np.log(np.exp(l0 - m) + np.exp(l1 - m)) + m)
        if ll - ll_old < tol:
            break
        ll_old = ll
        print(f"iter={it:03d}  N0={N0:.1f}  N1={N1:.1f}  mu0={mu[c0][0]:.3f}  mu1={mu[c1][0]:.3f}  ll={ll:.1f}")

    return mu, Sigma, pi


# If these live in unsup_exp.py, import them (unchanged helpers):
# from unsup_exp import fit_source_gaussian_params, init_means_kmeanspp, em_two_gaussians_shared_cov

def _posteriors_two_gauss_shared(X, mu, Sigma, pi, reg=1e-6):
    """E-step only: return gamma0, gamma1 (posterior probs) for K=2 shared-cov Gaussians."""
    n, d = X.shape
    U, s, Vt = np.linalg.svd(Sigma + reg * np.eye(d), full_matrices=False)
    logdet = np.sum(np.log(s))
    Sinv = (Vt.T * (1.0 / s)) @ U.T

    def logpdf(mu_c):
        Z = X - mu_c
        q = np.einsum('nd,dd,nd->n', Z, Sinv, Z)
        return -0.5 * (d * np.log(2*np.pi) + logdet + q)

    l0 = np.log(pi[0] + 1e-12) + logpdf(mu[0])
    l1 = np.log(pi[1] + 1e-12) + logpdf(mu[1])
    m  = np.maximum(l0, l1)
    r0 = np.exp(l0 - m); r1 = np.exp(l1 - m)
    ssum = r0 + r1 + 1e-12
    return (r0 / ssum), (r1 / ssum)  # gamma0, gamma1


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

def sample_gaussian(mu: np.ndarray, Sigma: np.ndarray, n: int, rng) -> np.ndarray:
    return rng.multivariate_normal(mean=mu, cov=Sigma, size=n)


# =========================================
# Self-training: Parametric/OT (1D generator) 
# =========================================
def self_train_along_expansion(metric: str,
                               n_per_step: int,
                               X_source: np.ndarray,
                               Y_source: np.ndarray,
                               clf_src: LogisticRegression,
                               mu_left0: float,
                               mu_right0: float,
                               mu_left1: float,
                               mu_right1: float,
                               lam0: float,
                               lam1: float,
                               grid_n: int = GRID_N,
                               neighbor_span: int = NEIGHBOR_SPAN,
                               stride: int = 1,
                               rng=RNG,
                               conf_threshold_low=0.6,
                               conf_threshold_high=0.9,
                               min_batch=100,
                               replay_ratio=0.0,
                               snapshot_k: int = 6,
                               snapshot_n: int = 3000,
                               max_iter_steps: int = 5,
                               X_target: Optional[np.ndarray] = None,
                               Y_target: Optional[np.ndarray] = None,
                               supervised: bool = False,   # <— NEW
                               em_max_iter: int = 10,
                               em_tol: float = 1e-5,
                               em_reg: float = 1e-6,
                               em_init: str = "kmeans++",
                               ) -> Dict[str, np.ndarray]:
    """Self-train along A* arc; if sample_ot=True, generate X_t via Sinkhorn displacement interpolation."""
    # Build A* paths and aligned schedule

    # --- if UNSUPERVISED: estimate target endpoints via EM on unlabeled target batch
    if not supervised:
        if X_target is None:
            raise ValueError("X_target must be provided when supervised=False.")

        # 1) Fit source params (for EM init + alignment)
        mus_s, Sigma_s, priors_s = fit_source_gaussian_params(X_source, Y_source)  # from your earlier helper
        # breakpoint()
        # 2) EM with shared covariance on unlabeled target; init at source to reduce label-swap
        q30, q70 = np.quantile(X_target[:,0], [0.30, 0.70])
        # mu_init = {0: np.array([q30]), 1: np.array([q70])}
        # pi_init = {0: 0.5, 1: 0.5}

        # Unlabeled target init via k-means++
        mu_init, pi_init = init_means_kmeanspp(X_target, rng=RNG, reorder_dim=0)

        mu_t, Sigma_t, priors_t = em_two_gaussians_shared_cov(
            X_target, mu_init, Sigma_s, pi_init, max_iter=em_max_iter, tol=em_tol, reg=1e-6
        )

        # 3) Align target components (avoid label flip): choose mapping that preserves left/right on x-dim
        #    We'll order by the first coordinate to match "left" vs "right".
        #    If your 1D feature is not the first column, change idx.
        idx = 0
        src_left_is_class0 = (mus_s[0][idx] <= mus_s[1][idx])
        tgt_class0_is_left  = (mu_t[0][idx]  <= mu_t[1][idx])

        if src_left_is_class0 != tgt_class0_is_left:
            # swap target component labels to align with source convention
            mu_t = {0: mu_t[1], 1: mu_t[0]}
            priors_t = {0: priors_t[1], 1: priors_t[0]}

        # 4) Set the *estimated* target endpoints for A*
        mu_left1  = float(mu_t[0][idx])
        mu_right1 = float(mu_t[1][idx])
        lam1      = float(Sigma_t[idx, idx])   # use shared covariance's variance along x

    mus_neg, lams_neg = build_grid_mu_lambda(mu_left0,  mu_left1,  lam0, lam1, grid_n, grid_n)
    mus_pos, lams_pos = build_grid_mu_lambda(mu_right0, mu_right1, lam0, lam1, grid_n, grid_n)

    mu_path_pos, lam_path_pos, t_arc_pos, _ = a_star_mu_lambda_path(
        metric=metric, mu0=mu_right0, mu1=mu_right1, lam0=lam0, lam1=lam1, mus=mus_pos, lams=lams_pos, K=1)
    mu_path_neg, lam_path_neg, t_arc_neg, _ = a_star_mu_lambda_path(
        metric=metric, mu0=mu_left0,  mu1=mu_left1,  lam0=lam0, lam1=lam1, mus=mus_neg, lams=lams_neg, K=1)

    mu_neg_on_pos = interp1d(t_arc_neg, mu_path_neg, kind="linear",
                             bounds_error=False, fill_value="extrapolate")(t_arc_pos)

    mu_path_pos = mu_path_pos[::stride]
    mu_path_neg = mu_neg_on_pos[::stride]
    lam_path    = lam_path_pos[::stride]
    t_arc       = t_arc_pos[::stride]

    # Classifier
    clf = SGDClassifier(loss="log_loss", penalty="l2",
                        alpha=1e-4, learning_rate="constant", eta0=0.05,
                        random_state=0)
    clf.partial_fit(X_source, Y_source, classes=np.array([0, 1]))

    # Bookkeeping
    snapshots = []
    accs, ces, counts, accs_b, ces_b = [], [], [], [], []
    accs_target, ces_target = [], []
    snap_idx = np.linspace(0, len(t_arc) - 1, snapshot_k, dtype=int)


    for step, (t, mu_pos_t, mu_neg_t, lam_t) in enumerate(zip(t_arc, mu_path_pos, mu_path_neg, lam_path)):
        # Build unlabeled batch
        # after you set X_t, y_true

        X_t, y_true = sample_dataset_with_means(
            mu_left=mu_neg_t, mu_right=mu_pos_t, lam_val=lam_t, n=n_per_step, rng=rng
        )

        n_batch = len(X_t)
        if n_batch == 0:
            counts.append(0)
            continue  # nothing to do this step

        # --------- 2) Pseudo labels computed ALWAYS (safe for debug/metrics) ---------
        proba = clf.predict_proba(X_t)[:, 1]
        y_hat = (proba >= 0.5).astype(int)
        conf  = np.maximum(proba, 1 - proba)
        tau_hi, tau_lo = conf_threshold_high, conf_threshold_low
        tau  = tau_hi + (tau_lo - tau_hi) * (step / max(1, len(t_arc) - 1))
        mask = conf >= tau

        if supervised:
            counts.append(len(X_t))
            for _ in range(max_iter_steps):
                clf.partial_fit(X_t, y_true)
        else:
            # balanced selection
            idx_pos = np.where(mask & (y_hat == 1))[0]
            idx_neg = np.where(mask & (y_hat == 0))[0]
            k = min(len(idx_pos), len(idx_neg))
            counts.append(2 * k)

            if k >= max(min_batch // 2, 1):
                sel = np.concatenate([
                    rng.choice(idx_pos, k, replace=False),
                    rng.choice(idx_neg, k, replace=False)
                ])
                X_pl, y_pl = X_t[sel], y_hat[sel]
                w_pl       = conf[sel]

                # optional replay...
                k_src = int(replay_ratio * len(sel))
                if k_src > 0:
                    idx_s = rng.choice(len(X_source), size=k_src, replace=False)
                    X_mix  = np.vstack([X_pl, X_source[idx_s]])
                    y_mix  = np.concatenate([y_pl, Y_source[idx_s]])
                    w_mix  = np.concatenate([w_pl, np.ones(k_src)])
                else:
                    X_mix, y_mix, w_mix = X_pl, y_pl, w_pl

                for _ in range(max_iter_steps):
                    clf.partial_fit(X_mix, y_mix, sample_weight=w_mix)

        # metrics & debug (safe: y_hat/mask always defined)
        P = clf.predict_proba(X_t)[:, 1]
        accs.append(accuracy_score(y_true, (P >= 0.5).astype(int)))
        ces.append(log_loss(y_true, P, labels=[0, 1]))

        P_b = clf_src.predict_proba(X_t)[:, 1]
        accs_b.append(accuracy_score(y_true, (P_b >= 0.5).astype(int)))
        ces_b.append(log_loss(y_true, P_b, labels=[0, 1]))

        # breakpoint()
        P_target = clf.predict_proba(X_target)[:, 1]
        accs_target.append(accuracy_score(Y_target, (P_target >= 0.5).astype(int)))
        ces_target.append(log_loss(Y_target, P_target, labels=[0, 1]))

        # Debug prints for parametric path
        if (step % 50 == 0):
            w = float(clf.coef_[0, 0]); b = float(clf.intercept_[0])
            x_star = -b / max(abs(w), 1e-12)
            mid = 0.5 * (mu_pos_t + mu_neg_t)
            pl_rate = mask.mean()
            err_pl = (y_hat[mask] != y_true[mask]).mean() if mask.any() else np.nan
            pl_pos_frac = (y_hat[mask] == 1).mean() if mask.any() else np.nan
            true_pos_frac = (y_true[mask] == 1).mean() if mask.any() else np.nan
            print(f"step {step:03d} | mid={mid: .3f} | x*={x_star: .3f} | "
                  f"acc={accuracy_score(y_true, (P >= 0.5).astype(int)):.3f} | ce={log_loss(y_true, P, labels=[0,1]):.3f} | "
                  f"PL used={pl_rate: .2f} | PL err={err_pl: .3f} | "
                  f"PL +frac={pl_pos_frac: .2f} | true +frac={true_pos_frac: .2f}")

        if step in set(snap_idx):
            snapshots.append({
                "step": int(step),
                "mu_left": float(mu_path_neg[min(step, len(mu_path_neg)-1)]),
                "mu_right": float(mu_path_pos[min(step, len(mu_path_pos)-1)]),
                "lam": float(lam_path[min(step, len(lam_path)-1)]),
                "clf": copy.deepcopy(clf)
            })

    return {
        "mu_path_pos": mu_path_pos,
        "mu_path_neg": mu_path_neg,
        "lam_path":    lam_path,
        "t_arc":       t_arc,
        "accs":        np.array(accs),
        "ces":         np.array(ces),
        "accs_target": np.array(accs_target),
        "ces_target": np.array(ces_target),
        "counts":      np.array(counts),
        "accs_base":   np.array(accs_b),
        "ces_base":    np.array(ces_b),
        "snapshots":   snapshots
    }


def self_train_on_ot_interpolation(
    n_per_step: int,
    X_source: np.ndarray,
    Y_source: np.ndarray,
    clf_src: LogisticRegression,             # unlabeled target pool used to build the OT map
    t_arc: Optional[np.ndarray] = None,        # schedule in [0,1]; default = np.linspace(0,1,201)
    stride: int = 1,
    rng=np.random.default_rng(0),
    # training hyperparams
    label_mode: str = "pl",                    # "pl" (prev classifier) or "ot" (labels transported from source)
    conf_threshold_low: float = 0.6,
    conf_threshold_high: float = 0.9,
    min_batch: int = 100,
    replay_ratio: float = 0.0,
    max_iter_steps: int = 5,

    # Sinkhorn / OT
    sinkhorn_reg: float = 0.05,
    a_weights: Optional[np.ndarray] = None,    # optional source weights
    b_weights: Optional[np.ndarray] = None,    # optional target weights

    # optional per-step evaluation on a held-out target test set (not used in training)
    X_target: Optional[np.ndarray] = None,
    Y_target: Optional[np.ndarray] = None,

    snapshot_k: int = 6,
    em_max_iter: int = 100,
    em_tol: float = 1e-5,
    em_reg: float = 1e-6,
    em_init: str = "kmeans++",  
) -> Dict[str, np.ndarray]:
    """
    Self-train along the Sinkhorn OT displacement interpolation between X_source and X_target_pool.
    - Builds a single global barycentric map T(x_i) from the OT plan.
    - At each t in t_arc, forms X_t = (1-t) X_source + t T(X_source).
    - Pseudo labels:
        * 'pl': use current classifier predictions with confidence gating + class balance.
        * 'ot': attach the source label Y_source[i] to the interpolated sample for source i
                (labels are transported along the OT flow; no target labels are used).
    Returns a dict mirroring your other runners, with train-time metrics and optional target-test metrics.
    """

    # ---- schedule
    if t_arc is None:
        t_arc = np.linspace(0.0, 1.0, 201)
    t_arc = t_arc[::stride]
    Tn = len(t_arc)

    if label_mode.lower() in {"pl", "ot"}:
        # ----- global barycentric map (same as before)
        if X_target is None:
            raise ValueError("X_target is required to build the OT map.")
        T_global = sinkhorn_barycentric_map(
            X_src=X_source, X_tgt=X_target, reg=sinkhorn_reg, a=a_weights, b=b_weights
        )
        classwise = False
    elif label_mode.lower() == "em":
        # ----- classwise maps guided by EM posteriors on the TARGET
        if X_target is None:
            raise ValueError("X_target is required for label_mode='em'.")

        # Source stats (for 'source' init or Sigma_init)
        mus_s, Sigma_s, priors_s = fit_source_gaussian_params(X_source, Y_source)

        # EM init for target
        if em_init.lower() == "source":
            mu_init = {0: mus_s[0].copy(), 1: mus_s[1].copy()}
            pi_init = {0: priors_s[0], 1: priors_s[1]}
        else:  # kmeans++ on target
            mu_init, pi_init = init_means_kmeanspp(X_target, rng=rng, reorder_dim=0)

        mu_t, Sigma_t, priors_t = em_two_gaussians_shared_cov(
            X_target, mu_init, Sigma_s, pi_init, max_iter=em_max_iter, tol=em_tol, reg=em_reg
        )


        # Get posteriors on target
        gamma0, gamma1 = _posteriors_two_gauss_shared(X_target, mu_t, Sigma_t, priors_t, reg=em_reg)
        # normalize to prob. weights over target per class
        w0 = gamma0 / (gamma0.sum() + 1e-12)
        w1 = gamma1 / (gamma1.sum() + 1e-12)

        # Build classwise barycentric maps: source class c → whole target with weights w_c
        X0 = X_source[Y_source == 0]
        X1 = X_source[Y_source == 1]
        a0 = np.full(len(X0), 1.0 / max(len(X0), 1))
        a1 = np.full(len(X1), 1.0 / max(len(X1), 1))

        # cost = squared Euclidean
        M0 = ot.dist(X0, X_target, metric='euclidean')**2
        M1 = ot.dist(X1, X_target, metric='euclidean')**2
        P0 = ot.sinkhorn(a0, w0, M0, sinkhorn_reg) if len(X0) else None
        P1 = ot.sinkhorn(a1, w1, M1, sinkhorn_reg) if len(X1) else None

        def bary_map(P, X_tgt):
            denom = P.sum(axis=1, keepdims=True) + 1e-12
            return (P @ X_tgt) / denom

        T0 = bary_map(P0, X_target) if P0 is not None else np.zeros_like(X0)
        T1 = bary_map(P1, X_target) if P1 is not None else np.zeros_like(X1)

        classwise = True
    else:
        raise ValueError("label_mode must be one of {'pl','ot','em'}.")

    # ----- classifier warm-started from source
    clf = SGDClassifier(loss="log_loss", penalty="l2",
                        alpha=1e-4, learning_rate="constant", eta0=0.05,
                        random_state=0)
    clf.partial_fit(X_source, Y_source, classes=np.array([0, 1]))

    # bookkeeping
    n_src = len(X_source)
    idx_all = np.arange(n_src)
    snapshots = []
    accs, ces, accs_b, ces_b, counts = [], [], [], [], []
    accs_target, ces_target = [], []
    snap_idx = np.linspace(0, Tn - 1, snapshot_k, dtype=int)

    # ---- main loop over the OT path
    for step, t in enumerate(t_arc):
        # Intermediate cloud along OT geodesic
        if not classwise:
            # global map (pl / ot)
            X_full = (1.0 - t) * X_source + t * T_global
            y_full = Y_source
        else:
            # classwise interpolation (em)
            X0 = X_source[Y_source == 0]; X1 = X_source[Y_source == 1]
            X0_t = (1.0 - t) * X0 + t * T0 if len(X0) else np.empty((0, X_source.shape[1]))
            X1_t = (1.0 - t) * X1 + t * T1 if len(X1) else np.empty((0, X_source.shape[1]))
            X_full = np.vstack([X0_t, X1_t])
            y_full = np.concatenate([np.zeros(len(X0_t), dtype=int), np.ones(len(X1_t), dtype=int)])

        # subsample
        k = min(n_per_step, len(X_full))
        sel = rng.choice(len(X_full), size=k, replace=False) if k > 0 else np.array([], dtype=int)
        X_t = X_full[sel]
        y_true = y_full[sel]  # labels move with mass for evaluation & (in 'ot'/'em') training

        if len(X_t) == 0:
            counts.append(0)
            continue
        if label_mode.lower() in {"ot", "em"}:
            # OT-transported labels: attach source label to its moving mass (no model predictions)
            X_pl, y_pl = X_t, y_true
            w_pl = np.ones(len(X_pl))
            counts.append(len(X_pl))

            for _ in range(max_iter_steps):
                clf.partial_fit(X_pl, y_pl, sample_weight=w_pl)

        elif label_mode.lower() == "pl":
            # Pseudo-labels from the current classifier, with confidence gating + class balance
            proba = clf.predict_proba(X_t)[:, 1]
            y_hat = (proba >= 0.5).astype(int)
            conf  = np.maximum(proba, 1 - proba)

            # anneal threshold hi→low along the path
            tau = conf_threshold_high + (conf_threshold_low - conf_threshold_high) * (step / max(1, Tn - 1))
            mask = conf >= tau

            idx_pos = np.where(mask & (y_hat == 1))[0]
            idx_neg = np.where(mask & (y_hat == 0))[0]
            bsz = min(len(idx_pos), len(idx_neg))
            counts.append(2 * bsz)

            if bsz >= max(min_batch // 2, 1):
                keep = np.concatenate([
                    rng.choice(idx_pos, bsz, replace=False),
                    rng.choice(idx_neg, bsz, replace=False)
                ])
                X_pl, y_pl = X_t[keep], y_hat[keep]
                w_pl       = conf[keep]

                # tiny replay (optional)
                k_src = int(replay_ratio * len(keep))
                if k_src > 0:
                    idx_s = rng.choice(n_src, size=k_src, replace=False)
                    X_mix = np.vstack([X_pl, X_source[idx_s]])
                    y_mix = np.concatenate([y_pl, Y_source[idx_s]])
                    w_mix = np.concatenate([w_pl, np.ones(k_src)])
                else:
                    X_mix, y_mix, w_mix = X_pl, y_pl, w_pl

                for _ in range(max_iter_steps):
                    clf.partial_fit(X_mix, y_mix, sample_weight=w_mix)
        else:
            raise ValueError("label_mode must be one of {'pl','ot','em'}.")

        # ---- training-batch metrics (vs. y_true defined above)
        P = clf.predict_proba(X_t)[:, 1]
        accs.append(accuracy_score(y_true, (P >= 0.5).astype(int)))
        ces.append(log_loss(y_true, P, labels=[0, 1]))

        # baseline (frozen source model) on the same batch
        Pb = clf_src.predict_proba(X_t)[:, 1]
        accs_b.append(accuracy_score(y_true, (Pb >= 0.5).astype(int)))
        ces_b.append(log_loss(y_true, Pb, labels=[0, 1]))

        # optional target-test metrics each step
        if (X_target is not None) and (Y_target is not None):
            Pt = clf.predict_proba(X_target)[:, 1]
            accs_target.append(accuracy_score(Y_target, (Pt >= 0.5).astype(int)))
            ces_target.append(log_loss(Y_target, Pt, labels=[0, 1]))

        # store snapshot models periodically (for later evaluation/plots)
        if step in set(snap_idx):
            snapshots.append({
                "step": int(step),
                "mu_left": np.nan,     # not relevant in OT-only runner, keep for API consistency
                "mu_right": np.nan,
                "lam": np.nan,
                "clf": copy.deepcopy(clf)
            })

    return {
        # dummy placeholders kept for API parity with your parametric runner
        "mu_path_pos": np.array([]),
        "mu_path_neg": np.array([]),
        "lam_path":    np.array([]),

        "t_arc":       t_arc,
        "accs":        np.array(accs),
        "ces":         np.array(ces),
        "accs_base":   np.array(accs_b),
        "ces_base":    np.array(ces_b),
        "counts":      np.array(counts),
        "snapshots":   snapshots,

        # optional per-step target test series
        "accs_target": np.array(accs_target) if len(accs_target) > 0 else np.array([]),
        "ces_target":  np.array(ces_target)  if len(ces_target)  > 0 else np.array([]),
    }


# =========================================
# Self-training: Unsupervised ExpFam (shared-cov Gaussians)
# =========================================
def self_train_along_expansion_expFam_unsup(
    n_per_step: int,
    X_source: np.ndarray,
    Y_source: np.ndarray,
    X_target: np.ndarray,
    Y_target: np.ndarray, # for target metrics
    clf_src: LogisticRegression,
    t_arc: np.ndarray,                      # schedule (e.g., np.linspace)
    stride: int = 1,
    rng=np.random.default_rng(0),
    min_batch: int = 100,                   # unused here (supervised synthetic)
    max_iter_steps: int = 3,
    snapshot_k: int = 6
) -> Dict[str, np.ndarray]:
    """
    Unsupervised gradual DA: source/target share exponential family.
    Here: K=2 Gaussians with shared covariance (estimated by EM).
    """
    t_arc = t_arc[::stride]
    T = len(t_arc)

    # 1) source params
    mus_s, Sigma_s, priors_s = fit_source_gaussian_params(X_source, Y_source)

    # 2) target params via EM on unlabeled (init at source to avoid swaps)
    # mu_init = {0: mus_s[0], 1: mus_s[1]}
    # pi_init = {0: priors_s[0], 1: priors_s[1]}

    # Unlabeled target init via k-means++
    mu_init, pi_init = init_means_kmeanspp(X_target, rng=RNG, reorder_dim=0)

    mu_t, Sigma_t, priors_t = em_two_gaussians_shared_cov(
        X_target, mu_init, Sigma_s, pi_init, max_iter=100, tol=1e-5, reg=1e-6
    )

    # 3) classifier
    clf = SGDClassifier(loss="log_loss", penalty="l2",
                        alpha=1e-4, learning_rate="constant", eta0=0.05,
                        random_state=0)
    clf.partial_fit(X_source, Y_source, classes=np.array([0, 1]))

    # 4) loop over t (supervised synthetic intermediates)
    snapshots = []
    accs, ces, accs_b, ces_b, counts = [], [], [], [], []
    accs_target, ces_target = [], []
    snap_idx = np.linspace(0, T-1, snapshot_k, dtype=int)

    for step, t in enumerate(t_arc):
        mu0_t, Sig_t = gaussian_e_geodesic(mus_s[0], Sigma_s, mu_t[0], Sigma_t, t)
        mu1_t, _     = gaussian_e_geodesic(mus_s[1], Sigma_s, mu_t[1], Sigma_t, t)
        pi0_t = (1-t) * priors_s[0] + t * priors_t[0]
        n0 = int(round(pi0_t * n_per_step)); n1 = n_per_step - n0
        X0 = sample_gaussian(mu0_t, Sig_t, n0, rng)
        X1 = sample_gaussian(mu1_t, Sig_t, n1, rng)
        X_t = np.vstack([X0, X1])
        y_t = np.concatenate([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)])
        counts.append(len(X_t))

        for _ in range(max_iter_steps):
            clf.partial_fit(X_t, y_t)

        P = clf.predict_proba(X_t)[:, 1]
        accs.append(accuracy_score(y_t, (P >= 0.5).astype(int)))
        ces.append(log_loss(y_t, P, labels=[0, 1]))

        P_b = clf_src.predict_proba(X_t)[:, 1]
        accs_b.append(accuracy_score(y_t, (P_b >= 0.5).astype(int)))
        ces_b.append(log_loss(y_t, P_b, labels=[0, 1]))

        P_target = clf.predict_proba(X_target)[:, 1]
        accs_target.append(accuracy_score(Y_target, (P_target >= 0.5).astype(int)))
        ces_target.append(log_loss(Y_target, P_target, labels=[0, 1]))

        if step in set(snap_idx):
            snapshots.append({
                "step": int(step),
                "mu_left": float(mu0_t[0]),
                "mu_right": float(mu1_t[0]),
                "lam": float(np.trace(Sig_t)/Sig_t.shape[0]),
                "clf": copy.deepcopy(clf)
            })

    return {
        "mu_path_pos": np.array([s["mu_right"] for s in snapshots]) if snapshots else np.array([]),
        "mu_path_neg": np.array([s["mu_left"]  for s in snapshots]) if snapshots else np.array([]),
        "lam_path":    np.array([s["lam"]      for s in snapshots]) if snapshots else np.array([]),
        "t_arc":       t_arc,
        "accs":        np.array(accs),
        "ces":         np.array(ces),
        "counts":      np.array(counts),
        "accs_base":   np.array(accs_b),
        "ces_base":    np.array(ces_b),
        "snapshots":   snapshots,
        "accs_target": np.array(accs_target),
        "ces_target": np.array(ces_target)
    }


# =========================================
# Plotting helpers
# =========================================
def decision_boundary_1d(clf):
    if not hasattr(clf, "coef_") or not hasattr(clf, "intercept_"):
        return None
    w = float(np.ravel(clf.coef_)[0])
    b = float(np.ravel(clf.intercept_)[0])
    if abs(w) < 1e-12:
        return None
    return -b / w

def plot_evolution_snapshots(snapshots, clf_src=None, n_vis=3000, bins=80, rng=None, title="evolution_snapshots"):
    rng = np.random.default_rng() if rng is None else rng
    if len(snapshots) == 0:
        print("No snapshots to plot.")
        return

    xs = []
    for s in snapshots:
        sd = np.sqrt(s["lam"])
        xs += [s["mu_left"] - 4*sd, s["mu_left"] + 4*sd,
               s["mu_right"] - 4*sd, s["mu_right"] + 4*sd]
    xmin, xmax = min(xs), max(xs)
    B = np.linspace(xmin, xmax, bins)

    xstar_base = decision_boundary_1d(clf_src) if clf_src is not None else None

    fig, axes = plt.subplots(1, len(snapshots), figsize=(16, 3.6), sharey=True)
    if len(snapshots) == 1:
        axes = [axes]

    for ax, s in zip(axes, snapshots):
        X, Y = sample_dataset_with_means(mu_left=s["mu_left"], mu_right=s["mu_right"],
                                         lam_val=s["lam"], n=n_vis, rng=rng)
        ax.hist(X[Y==0].ravel(), bins=B, alpha=0.6, density=True, label="class 0")
        ax.hist(X[Y==1].ravel(), bins=B, alpha=0.6, density=True, label="class 1")

        xstar = decision_boundary_1d(s["clf"])
        if xstar is not None:
            ax.axvline(xstar, color="k", linestyle="--", linewidth=2, label="self-train x*")
        if xstar_base is not None:
            ax.axvline(xstar_base, color="k", linestyle=":", linewidth=2, label="source x*")

        ax.set_title(f"step {s['step']}\nμL={s['mu_left']:.2f}, μR={s['mu_right']:.2f}, λ={s['lam']:.1f}")
        ax.set_xlabel("x")
    axes[0].set_ylabel("density")
    axes[0].legend(loc="upper right")
    plt.tight_layout()
    plt.show()
    plt.savefig(f"{title}.png")

    xgrid = np.linspace(xmin, xmax, 400).reshape(-1, 1)
    fig, axes = plt.subplots(1, len(snapshots), figsize=(10, 3.2), sharey=True)
    if len(snapshots) == 1:
        axes = [axes]
    for ax, s in zip(axes, snapshots):
        p = s["clf"].predict_proba(xgrid)[:, 1]
        ax.plot(xgrid, p, linewidth=2, label="p(y=1|x)")
        ax.axhline(0.5, color="k", linestyle=":", linewidth=1)
        xstar = decision_boundary_1d(s["clf"])
        if xstar is not None:
            ax.axvline(xstar, color="k", linestyle="--", linewidth=1)
        if xstar_base is not None:
            ax.axvline(xstar_base, color="k", linestyle=":", linewidth=1)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"step {s['step']}")
        ax.set_xlabel("x")
    axes[0].set_ylabel("probability")
    axes[0].legend(loc="upper left")
    plt.tight_layout()
    plt.show()

def plot_expansion_both(runs: Dict[str, Dict[str, np.ndarray]],
                        key: str, ylabel: str, title: str,
                        marker_stride: int = 10):
    plt.figure(figsize=(12,4))

    # --- (A) Normalized: t in [0,1]
    ax1 = plt.subplot(1,2,1)
    for label, run in runs.items():
        t = run["t_arc"]
        y_self = run[key]
        y_base = run["accs_base"] if key == "accs" else run["ces_base"]
        L = min(len(t), len(y_self), len(y_base))
        ax1.plot(t[:L:marker_stride], y_self[:L:marker_stride], label=f"{label} — self-train")
        ax1.plot(t[:L:marker_stride], y_base[:L:marker_stride], linestyle=':', linewidth=2, label=f"{label} — baseline")
    ax1.set_xlabel("t (normalized A* arc-length)")
    ax1.set_ylabel(ylabel)
    ax1.set_title(f"{title} — normalized")
    ax1.legend()

    # --- (B) Unnormalized: step index
    ax2 = plt.subplot(1,2,2)
    for label, run in runs.items():
        y_self = run[key]
        y_base = run["accs_base"] if key == "accs" else run["ces_base"]
        steps = np.arange(len(y_self))
        ax2.plot(steps[::marker_stride], y_self[::marker_stride], label=f"{label} — self-train")
        ax2.plot(steps[::marker_stride], y_base[::marker_stride], linestyle=':', linewidth=2, label=f"{label} — baseline")
    ax2.set_xlabel("step index")
    ax2.set_ylabel(ylabel)
    ax2.set_title(f"{title} — unnormalized (steps)")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f"{title}.png")
    plt.show()








def plot_compare_runs(runs_dict, key="accs", ylabel="Accuracy", title="", marker_stride=10, save=True):
    """
    Overlay curves from multiple runs on two panels:
      (left)  y vs normalized arc-length t (run['t_arc'])
      (right) y vs unnormalized step index (0..T-1)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import re

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)

    for name, run in runs_dict.items():
        if key not in run:
            continue
        y = np.asarray(run[key])
        if y.size == 0:
            continue

        # Normalized arc-length; if missing, fall back to uniform [0,1]
        t = np.asarray(run.get("t_arc", np.linspace(0, 1, len(y))))
        if t.size != y.size:
            # best-effort fallback if lengths mismatch
            t = np.linspace(0, 1, len(y))

        # Left: normalized t
        ax1.plot(t[::marker_stride], y[::marker_stride], marker='o', linewidth=2, ms=3, label=name)

        # Right: unnormalized step index
        steps = np.arange(len(y))
        ax2.plot(steps[::marker_stride], y[::marker_stride], marker='o', linewidth=2, ms=3, label=name)

    # Labels/titles
    ax1.set_xlabel("t (normalized path progress)")
    ax2.set_xlabel("step index")
    for ax in (ax1, ax2):
        ax.set_ylabel(ylabel)
        ax.grid(False)

    ax1.set_title(f"{title} — normalized")
    ax2.set_title(f"{title} — unnormalized (steps)")

    # Single legend (left panel) to avoid clutter
    ax1.legend()

    plt.tight_layout()

    if save:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", title.strip()) or "compare"
        plt.savefig(f"{safe}.png", dpi=120)

    plt.show()


def main():
    # ----- Config
    LAM0, LAM1 = 1.0, 2.0
    MU_LEFT0, MU_RIGHT0  = -2.0, +2.0        # source endpoint
    MU_LEFT1, MU_RIGHT1  = +2.0, +5.0        # target endpoint (for synthetic data)
    GRID_N = 201
    RNG = np.random.default_rng(7)

    # ----- Source data + source model
    X_source, Y_source, clf_src = train_source_model(
        n_source=5000, lam_source=LAM0,
        mu_left0=MU_LEFT0, mu_right0=MU_RIGHT0, rng=RNG
    )

    # ----- Target samples (only used to *evaluate* and for EM in ExpFam)
    X_target_all, Y_target_all = sample_dataset_with_means(
        mu_left=MU_LEFT1, mu_right=MU_RIGHT1, lam_val=LAM1,
        n=16000, rng=RNG
    )
    # split a held-out test set for fair comparison
    X_target_test, Y_target_test = X_target_all[:4000], Y_target_all[:4000]

    # ================
    # POINT 1: “Metric matters”
    # Run parametric path with the three metrics {FR, W2, eta}, both
    # (a) supervised-oracle (uses the synthetic labels along the path),
    # (b) unsupervised pseudo-labeling (PL).
    # ================
    metrics = ("FR", "W2", "eta")

    runs_sup = {}
    runs_unsup = {}
    for m in metrics:
        runs_sup[m] = self_train_along_expansion(
            metric=m, n_per_step=4000,
            X_source=X_source, Y_source=Y_source, clf_src=clf_src,
            X_target=X_target_test, Y_target=Y_target_test,
            mu_left0=MU_LEFT0,  mu_right0=MU_RIGHT0,
            mu_left1=MU_LEFT1,  mu_right1=MU_RIGHT1,
            lam0=LAM0, lam1=LAM1,
            grid_n=GRID_N, neighbor_span=NEIGHBOR_SPAN,
            stride=1, rng=RNG,
            supervised=True,         # parametric generator (not Sinkhorn)
            max_iter_steps=5
        )

        runs_unsup[m] = self_train_along_expansion(
            metric=m, n_per_step=4000,
            X_source=X_source, Y_source=Y_source, Y_target=Y_target_test,
            clf_src=clf_src,
            mu_left0=MU_LEFT0,  mu_right0=MU_RIGHT0,
            mu_left1=MU_LEFT1,  mu_right1=MU_RIGHT1,
            lam0=LAM0, lam1=LAM1,
            grid_n=GRID_N, neighbor_span=NEIGHBOR_SPAN,
            stride=1, rng=RNG,
            supervised=False,            # pseudo-labeling
            X_target=X_target_test,
            conf_threshold_low=0.6, conf_threshold_high=0.9,
            replay_ratio=0.0, max_iter_steps=5
        )
        run_ot_pl = self_train_on_ot_interpolation(
            n_per_step=4000,
            X_source=X_source, Y_source=Y_source, clf_src=clf_src,            # unlabeled target cloud for OT
            t_arc=np.linspace(0,1,201),
            label_mode="pl",                             # pseudo-labels from model
            sinkhorn_reg=0.05,
            X_target=X_target_test, Y_target=Y_target_test
        )

        run_ot_transport = self_train_on_ot_interpolation(
            n_per_step=4000,
            X_source=X_source, Y_source=Y_source, clf_src=clf_src,
            t_arc=np.linspace(0,1,201),
            label_mode="ot",                             # labels from OT transport (source-attached)
            sinkhorn_reg=0.05,
            X_target=X_target_test, Y_target=Y_target_test
        )

        run_ot_em = self_train_on_ot_interpolation(
            n_per_step=4000,
            X_source=X_source, Y_source=Y_source, clf_src=clf_src,
            t_arc=np.linspace(0,1,201),
            label_mode="em",                             # labels from EM
            sinkhorn_reg=0.05,
            X_target=X_target_test, Y_target=Y_target_test
        )

    # Visualization for Point 1 (training curves)
    plot_compare_runs({f"{m} (supervised)": runs_sup[m] for m in metrics},
                      key="accs", ylabel="Accuracy",
                      title="Class-wise Expansion with Ground Truth Labels")
    plot_compare_runs({f"A* {m} ": runs_unsup[m] for m in metrics} | {"OT (PL)": run_ot_pl, "OT (transport)": run_ot_transport, "OT (em)": run_ot_em},
                      key="accs", ylabel="Accuracy",
                      title="Class-wise Expansion with Pseudo-labels")

    plot_compare_runs({f"{m} (supervised)": runs_sup[m] for m in metrics},
                      key="accs_target", ylabel="Accuracy",
                      title="Test Accuracy, Ground Truth Labels")
    plot_compare_runs({f"A* {m}": runs_unsup[m] for m in metrics} | {"OT (PL)": run_ot_pl, "OT (transport)": run_ot_transport, "OT (EM)": run_ot_em},
                      key="accs_target", ylabel="Accuracy",
                      title="Test Accuracy, Pseudo-labels")

    def final_target_metrics(run):
        accs = run["accs_target"]
        ces = run["ces_target"]

        return float(accs[-1]), float(ces[-1])

    print("\n=== POINT 1: Final target metrics (last snapshot) ===")
    for m in metrics:
        a_sup, c_sup = final_target_metrics(runs_sup[m])
        a_uns, c_uns = final_target_metrics(runs_unsup[m])
        print(f"{m:>3s} | Supervised: Acc={a_sup:.3f}  CE={c_sup:.3f} | Unsupervised (PL): Acc={a_uns:.3f}  CE={c_uns:.3f}")

    # ============================
    # POINT 2: “EM labels > self-training”
    # Compare ExpFam (unsup; EM + e-geodesic, supervised on synthetic intermediates)
    # vs unsupervised PL on a fixed geometry (choose W2 for clarity).
    # ============================
    # Use the same schedule length as one of the parametric runs for visual alignment
    t_arc = runs_sup["W2"]["t_arc"] if "W2" in runs_sup else np.linspace(0, 1, 201)

    run_expfam = self_train_along_expansion_expFam_unsup(
        n_per_step=4000,
        X_source=X_source, Y_source=Y_source,
        X_target=X_target_test,   # unlabeled target for EM
        Y_target=Y_target_test,
        clf_src=clf_src,
        t_arc=t_arc, stride=1,
        rng=RNG, max_iter_steps=3, snapshot_k=6
    )

    # Training curves (accuracy / risk) for Point 2
    plot_compare_runs({
        "ExpFam (EM + e-geodesic, supervised synthetic)": run_expfam,
        "Parametric W2 — Unsupervised (PL)": runs_unsup["W2"]
    }, key="accs", ylabel="Accuracy", title="EM-based labeling vs Pseudo-labeling")

    plot_compare_runs({
        "ExpFam (EM + e-geodesic, supervised synthetic)": run_expfam,
        "Parametric W2 — Unsupervised (PL)": runs_unsup["W2"]
    }, key="ces", ylabel="Cross-entropy ↓", title="Risk: EM vs PL")

    plot_compare_runs({
        "ExpFam (EM + e-geodesic, supervised synthetic)": run_expfam,
        "Parametric W2 — Unsupervised (PL)": runs_unsup["W2"]
    }, key="accs_target", ylabel="Accuracy", title="Target Accuracy: EM-based labeling vs Pseudo-labeling")

    plot_compare_runs({
        "ExpFam (EM + e-geodesic, supervised synthetic)": run_expfam,
        "Parametric W2 — Unsupervised (PL)": runs_unsup["W2"]
    }, key="ces_target", ylabel="Cross-entropy ↓", title="Target Risk: EM vs PL")


    # # Target test curves for Point 2
    # plot_compare_on_target({
    #     "ExpFam (EM + e-geodesic)": run_expfam,
    #     "Parametric W2 — PL": runs_unsup["W2"]
    # }, X_target_test, Y_target_test, what="acc",
    #    title="POINT 2 — Target Accuracy vs progress (EM vs PL)")

    # Final target metrics (last snapshot)
    # breakpoint()
    a_em, c_em = final_target_metrics(run_expfam)
    a_pl, c_pl = final_target_metrics(runs_unsup["W2"])
    print("\n=== POINT 2: Final target metrics (last snapshot) ===")
    print(f"ExpFam (EM): Acc={a_em:.3f}  CE={c_em:.3f}")
    print(f"Parametric W2 — PL: Acc={a_pl:.3f}  CE={c_pl:.3f}")


if __name__ == "__main__":
    main()
