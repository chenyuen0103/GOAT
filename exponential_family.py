# =========================================
# Expansion & Self-Training on 1D Gaussian
# (only first covariance eigenvalue λ varies)
# =========================================

# ----- Imports
import math
import heapq
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
from scipy.interpolate import interp1d
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, log_loss
import copy

# =========================================
# Section 1: Metric helpers
# =========================================

METRICS = ("FR", "W2", "eta")
GRID_N = 201                 # λ-grid resolution for A*
NEIGHBOR_SPAN = 2            # connect to immediate left/right nodes
D = 1
RNG = np.random.default_rng(7)


# =========================================
# Section 2: A* on λ-grid (expansion)
# =========================================


import ot  # POT

def sinkhorn_barycentric_map(X_src, X_tgt, reg=0.05, a=None, b=None):
    """
    Compute barycentric map T(x_i) ≈ sum_j pi_ij y_j / sum_j pi_ij
    using entropic-regularized OT (Sinkhorn) between two empirical clouds.

    X_src: (n,d), Y_tgt: (m,d)
    reg:   entropic ε; larger = smoother plan (more mass spreading)
    a,b:   weights on source/target (uniform if None)
    returns T: (n,d) mapped targets for each x_i
    """
    n, d = X_src.shape
    m = Y_tgt.shape[0]
    if a is None: a = np.full(n, 1.0/n)
    if b is None: b = np.full(m, 1.0/m)

    # cost matrix (squared Euclidean = W2)
    M = ot.dist(X_src, Y_tgt, metric='euclidean')**2

    # Sinkhorn transport plan π_ε (n×m)
    P = ot.sinkhorn(a, b, M, reg)

    # barycentric map
    denom = P.sum(axis=1, keepdims=True) + 1e-12
    T = (P @ Y_tgt) / denom
    return T

def sinkhorn_displacement_interpolate(X_src: np.ndarray,
                                      Y_tgt: np.ndarray,
                                      t: float,
                                      reg: float = 0.05,
                                      a = None,
                                      b= None) -> np.ndarray:
    """
    Return interpolated samples {x_i^(t)} = (1-t) x_i + t T(x_i)
    where T is the Sinkhorn barycentric map from X_src to Y_tgt.
    """
    T = sinkhorn_barycentric_map(X_src, Y_tgt, reg=reg, a=a, b=b)
    return (1.0 - t) * X_src + t * T

def build_grid_mu_lambda(mu0, mu1, lam0, lam1, n_mu, n_lam):
    mus  = np.linspace(min(mu0, mu1), max(mu0, mu1), n_mu)
    lams = np.linspace(min(lam0, lam1), max(lam0, lam1), n_lam)
    return mus, lams  # states indexed by (i,j)

def interpolate_classwise_sinkhorn(X0: np.ndarray, y0: np.ndarray,
                                   X1: np.ndarray, y1: np.ndarray,
                                   t: float,
                                   reg: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """
    Interpolate source→target separately per class using Sinkhorn,
    then concatenate. Keeps class labels from source.
    Assumes each class exists in both sets (or handle missing with skip/fallback).
    """
    classes = np.unique(np.concatenate([y0, y1]))
    Xt_list, yt_list = [], []
    for c in classes:
        X0c = X0[y0 == c]
        X1c = X1[y1 == c]
        if len(X0c) == 0 or len(X1c) == 0:
            # fallback: no matching class in one domain → skip or map globally
            continue
        Xtc = sinkhorn_displacement_interpolate(X0c, X1c, t=t, reg=reg)
        Xt_list.append(Xtc)
        yt_list.append(np.full(len(Xtc), c, dtype=y0.dtype))
    Xt = np.vstack(Xt_list)
    yt = np.concatenate(yt_list)
    return Xt, yt



def build_adj_2d(n_mu, n_lam, K=1):
    adj = {}
    for i in range(n_mu):
        for j in range(n_lam):
            nbrs = []
            for di in range(-K, K+1):
                for dj in range(-K, K+1):
                    if di == 0 and dj == 0: continue
                    ii, jj = i+di, j+dj
                    if 0 <= ii < n_mu and 0 <= jj < n_lam:
                        nbrs.append((ii, jj))
            adj[(i,j)] = nbrs
    return adj


def a_star_mu_lambda_path(
    metric: str,
    mu0: float, lam0: float,        # start
    mu1: float, lam1: float,        # goal
    mus: np.ndarray,                # 1D grid for mu (length n_mu)
    lams: np.ndarray,               # 1D grid for lambda (length n_lam; values > 0)
    K: int = 1                      # neighbor span on each axis (1 => 4/8-neighborhood)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    A* in (mu, lambda)-space on a 2D grid using the chosen metric.
    Returns (mu_path, lam_path, t_arc, total_length).

    metric in {"FR", "W2", "eta"}:
      - FR (1D Normal): local length^2 = (dmu^2)/lam_bar + 0.5*(dloglam^2),   lam_bar = sqrt(lam*lam')
      - W2 (1D Normal): length = sqrt( (mu'-mu)^2 + (sqrt(lam')-sqrt(lam))^2 )
      - eta (natural coords): theta=(mu/lam, -1/(2*lam)), Euclidean in theta
    """
    import math
    from math import sqrt, log, hypot

    n_mu, n_lam = len(mus), len(lams)

    # --- index helpers
    def snap_idx(arr, x):
        return int(np.argmin(np.abs(arr - x)))

    def neighbors(i, j):
        for di in range(-K, K+1):
            for dj in range(-K, K+1):
                if di == 0 and dj == 0:
                    continue
                ii, jj = i + di, j + dj
                if 0 <= ii < n_mu and 0 <= jj < n_lam:
                    yield ii, jj

    # --- edge costs
    def cost_FR(mu, lam, mu2, lam2):
        # guard against tiny/zero lambdas
        lam = max(lam, 1e-300); lam2 = max(lam2, 1e-300)
        lam_bar = sqrt(lam * lam2)  # geometric mean
        dmu = mu2 - mu
        dloglam = log(lam2) - log(lam)
        return sqrt((dmu * dmu) / lam_bar + 0.5 * (dloglam * dloglam))

    def cost_W2(mu, lam, mu2, lam2):
        return hypot(mu2 - mu, sqrt(max(lam2, 0.0)) - sqrt(max(lam, 0.0)))

    def cost_eta(mu, lam, mu2, lam2):
        # Euclidean in natural parameters theta = (mu/lam, -1/(2*lam))
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

    # Heuristic: same metric to the goal (admissible by triangle inequality)
    def h(mu, lam):
        return edge_cost(mu, lam, mu_goal, lam_goal)

    # --- snap start/goal to grid
    si, sj = snap_idx(mus, mu0), snap_idx(lams, lam0)
    gi, gj = snap_idx(mus, mu1), snap_idx(lams, lam1)
    mu_goal, lam_goal = mus[gi], lams[gj]

    # --- A* tables
    INF = float("inf")
    g_cost = np.full((n_mu, n_lam), INF, dtype=float)
    parent = { (si, sj): None }

    g_cost[si, sj] = 0.0
    start_f = g_cost[si, sj] + h(mus[si], lams[sj])

    # priority queue of (f, i, j)
    pq: List[Tuple[float, int, int]] = [(start_f, si, sj)]
    closed = np.zeros((n_mu, n_lam), dtype=bool)

    while pq:
        f_u, ui, uj = heapq.heappop(pq)
        if closed[ui, uj]:
            continue
        closed[ui, uj] = True
        if (ui, uj) == (gi, gj):
            break

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

    # --- reconstruct path
    assert (gi, gj) in parent or (gi, gj) == (si, sj), "A*: path not found"
    idx_path = []
    cur = (gi, gj)
    while cur is not None:
        idx_path.append(cur)
        cur = parent.get(cur, None)
    idx_path.reverse()

    mu_path = np.array([mus[i] for i, j in idx_path], dtype=float)
    lam_path = np.array([lams[j] for i, j in idx_path], dtype=float)

    # --- arc-length parameterization t_arc in [0,1]
    seg = [0.0]
    for k in range(1, len(idx_path)):
        (i0, j0), (i1, j1) = idx_path[k-1], idx_path[k]
        seg.append(seg[-1] + edge_cost(mus[i0], lams[j0], mus[i1], lams[j1]))
    seg = np.asarray(seg, dtype=float)
    total = float(seg[-1]) if len(seg) else 1.0
    t_arc = seg / total if total > 0 else seg

    return mu_path, lam_path, t_arc, total


# =========================================
# Section 4: Gaussian generator & paths (analytic)
# =========================================


def fr_geodesic_mu_lambda(t: float, mu0: float, lam0: float, mu1: float, lam1: float):
    """
    Exact Fisher–Rao geodesic for 1D Normal between (mu0, lam0) and (mu1, lam1).
    Returns (mu(t), lam(t)) with t in [0,1], parameterized by FR arc-length fraction.
    """
    import math

    # Map to upper half-plane: x = mu, y = sqrt(2) * sigma
    def to_xy(mu, lam):
        sigma = math.sqrt(lam)
        x = mu
        y = math.sqrt(2.0) * sigma
        return x, y

    x0, y0 = to_xy(mu0, lam0)
    x1, y1 = to_xy(mu1, lam1)

    # Vertical-line geodesic (x constant)
    if abs(x0 - x1) < 1e-15:  # same x -> vertical
        x_t = x0
        # along vertical geodesic: y(t) geometric interpolate for equal hyperbolic length
        y_t = y0 * (y1 / y0) ** t
        mu_t = x_t
        sigma_t = y_t / math.sqrt(2.0)
        lam_t = sigma_t ** 2
        return mu_t, lam_t

    # Otherwise, circle orthogonal to boundary: find center c on real axis and radius R
    c = (x0 * x0 + y0 * y0 - x1 * x1 - y1 * y1) / (2.0 * (x0 - x1))
    R = math.hypot(x0 - c, y0)  # = sqrt((x0-c)^2 + y0^2) = sqrt((x1-c)^2 + y1^2)

    # Angles at endpoints
    theta0 = math.atan2(y0, x0 - c)  # in (0, pi)
    theta1 = math.atan2(y1, x1 - c)

    # Equal FR-arc parameterization via tan(theta/2) geometric interpolation
    T0 = math.tan(0.5 * theta0)
    T1 = math.tan(0.5 * theta1)
    # guard against numerical edge cases
    T0 = max(T0, 1e-300)
    T1 = max(T1, 1e-300)

    Tt = (T0 ** (1.0 - t)) * (T1 ** t)
    theta_t = 2.0 * math.atan(Tt)

    # Back to (x,y) on the circle
    x_t = c + R * math.cos(theta_t)
    y_t = R * math.sin(theta_t)

    # Map back to (mu, lambda)
    mu_t = x_t
    sigma_t = y_t / math.sqrt(2.0)
    lam_t = sigma_t ** 2
    return mu_t, lam_t





def sample_dataset_with_means(mu_left: float, mu_right: float, lam_val: float,
                              n: int, d: int = D, rng=None, pi: float = 0.5,
                              ensure_both: bool = True):
    rng = np.random.default_rng() if rng is None else rng
    if lam_val <= 0:
        raise ValueError("lam_val must be > 0.")

    # labels: 0 with prob (1-pi), 1 with prob pi
    y = (rng.random(n) < pi).astype(int)
    if ensure_both and n >= 2 and (y.min() == y.max()):
        y[0], y[1] = 0, 1  # force presence of both classes

    # means per sample
    means = np.zeros((n, d), dtype=float)
    means[:, 0] = np.where(y == 0, mu_left, mu_right)

    # shared covariance diag(lam_val, 1, ..., 1)
    Sigma = np.eye(d)
    Sigma[0, 0] = lam_val

    # sample noise once and shift by means (vectorized)
    X = rng.multivariate_normal(mean=np.zeros(d), cov=Sigma, size=n) + means
    return X.astype(float), y





# =========================================
# Section 5: Self-training routines
# =========================================

def train_source_model(n_source: int,
                       mu_left0: float,
                       mu_right0: float,
                       lam_source: float,
                       rng=RNG) -> Tuple[np.ndarray, np.ndarray, LogisticRegression]:
    # Xs, Ys = sample_dataset_at_lambda(lam_source, n_source, rng=rng)
    Xs, Ys = sample_dataset_with_means(mu_left=mu_left0, mu_right=mu_right0, lam_val=lam_source, n=n_source, rng=rng)
    clf_src = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=2000)
    clf_src.fit(Xs, Ys)
    return Xs, Ys, clf_src



# def self_train_along_expansion(metric: str,
#                                n_per_step: int,
#                                X_source: np.ndarray,
#                                Y_source: np.ndarray,
#                                clf_src: LogisticRegression,
#                                mu_left0: float,
#                                mu_right0: float,
#                                mu_left1: float,
#                                mu_right1: float,
#                                lam0: float ,
#                                lam1: float ,
#                                grid_n: int = GRID_N,
#                                neighbor_span: int = NEIGHBOR_SPAN,
#                                stride: int = 1,
#                                rng=RNG,
#                                conf_threshold_low=0.6,
#                                conf_threshold_high=0.9,
#                                min_batch=100,
#                                replay_ratio=0.0,
#                                 snapshot_k: int = 6,    # how many snapshots to keep
#                                snapshot_n: int = 3000, # samples per snapshot figure
#                                max_iter_steps=5,
#                                sample_ot = False,
#                                classwise_ot = False) -> Dict[str, np.ndarray]:
#     """Self-train along A* (expansion) λ-trajectory."""


#     if sample_ot:
#         for step in range(GRID_N):
#             t = step / GRID_N
#             # Produce unlabeled batch at time t by OT displacement interpolation
#             if classwise_ot:
#                 X_pool, y_pool_dbg = interpolate_classwise_sinkhorn(
#                     X0=X_source, y0=Y_source, X1=X_target, y1=Y_target,
#                     t=t, reg=sinkhorn_reg
#                 )
#             else:
#                 # global OT (ignores labels); labels only for eval
#                 T = sinkhorn_barycentric_map(X_source, X_target, reg=sinkhorn_reg)
#                 X_pool = (1.0 - t) * X_source + t * T
#                 y_pool_dbg = Y_source  # for evaluation only, optional

#             # Subsample to n_per_step (keeps runtime bounded)
#             if len(X_pool) > n_per_step:
#                 idx = rng.choice(len(X_pool), size=n_per_step, replace=False)
#                 X_t = X_pool[idx]
#                 y_true = y_pool_dbg[idx]  # only for metrics; not used in training
#             else:
#                 X_t = X_pool
#                 y_true = y_pool_dbg

#             # --- pseudo labels from current model
#             proba = clf.predict_proba(X_t)[:, 1]
#             y_hat = (proba >= 0.5).astype(int)
#             conf  = np.maximum(proba, 1 - proba)

#             # confidence gating (anneal from high→low over the path)
#             tau_hi, tau_lo = conf_threshold_high, conf_threshold_low
#             tau = tau_hi + (tau_lo - tau_hi) * (step / max(1, len(t_arc)-1))
#             mask = conf >= tau

#             # balance by predicted class to avoid collapse
#             idx_pos = np.where(mask & (y_hat == 1))[0]
#             idx_neg = np.where(mask & (y_hat == 0))[0]
#             k = min(len(idx_pos), len(idx_neg))
#             counts.append(2*k)
#             if k >= max(min_batch//2, 1):
#                 sel = np.concatenate([
#                     rng.choice(idx_pos, k, replace=False),
#                     rng.choice(idx_neg, k, replace=False)
#                 ])
#                 X_pl, y_pl = X_t[sel], y_hat[sel]
#                 w_pl = conf[sel]

#                 # optional tiny replay (usually 0 is best here)
#                 k_src = int(replay_ratio * len(sel))
#                 if k_src > 0:
#                     idx_s = rng.choice(len(X_source), size=k_src, replace=False)
#                     X_mix  = np.vstack([X_pl, X_source[idx_s]])
#                     y_mix  = np.concatenate([y_pl, Y_source[idx_s]])
#                     w_mix  = np.concatenate([w_pl, np.ones(k_src)])
#                 else:
#                     X_mix, y_mix, w_mix = X_pl, y_pl, w_pl

#                 # a few repeats so the batch has effect
#                 for _ in range(max_iter_steps):
#                     clf.partial_fit(X_mix, y_mix, sample_weight=w_mix)

#             # --- metrics
#             P = clf.predict_proba(X_t)[:, 1]
#             accs.append(accuracy_score(y_true, (P >= 0.5).astype(int)))
#             ces.append(log_loss(y_true, P, labels=[0,1]))

#             P_b = clf_src.predict_proba(X_t)[:, 1]
#             accs_b.append(accuracy_score(y_true, (P_b >= 0.5).astype(int)))
#             ces_b.append(log_loss(y_true, P_b, labels=[0,1]))

#             if step in set(snap_idx):
#                 snapshots.append({
#                     "step": int(step),
#                     # keep fields for plotting labels; μ/λ here are no longer used for data gen
#                     "mu_left": float(mu_path_neg[min(step, len(mu_path_neg)-1)]),
#                     "mu_right": float(mu_path_pos[min(step, len(mu_path_pos)-1)]),
#                     "lam": float(lam_path[min(step, len(lam_path)-1)]) if len(lam_path)>0 else 0.0,
#                     "clf": copy.deepcopy(clf)
#                 })

#         return {
#             "mu_path_pos": mu_path_pos,
#             "mu_path_neg": mu_path_neg,
#             "lam_path":    lam_path,   # placeholder; not used in sampling now
#             "t_arc":       t_arc,
#             "accs":        np.array(accs),
#             "ces":         np.array(ces),
#             "counts":      np.array(counts),
#             "accs_base":   np.array(accs_b),
#             "ces_base":    np.array(ces_b),
#             "snapshots":   snapshots
#         }
        

#     mus_neg, lams_neg = build_grid_mu_lambda(mu_left0, mu_left1, lam0, lam1, grid_n, grid_n)
#     mus_pos, lams_pos = build_grid_mu_lambda(mu_right0, mu_right1, lam0, lam1, grid_n, grid_n)
#     # lam_path, t_arc, _ = a_star_lambda_path(metric, lam0, lam1, grid, neighbor_span)
#     # Create two paths: one for positive class, one for negative class
#     mu_path_pos, lam_path_pos, t_arc_pos, total_pos = a_star_mu_lambda_path(
#     metric=metric,
#     mu0=mu_right0,
#     mu1=mu_right1,
#     lam0=lam0, lam1=lam1,
#     mus=mus_pos, lams=lams_pos,
#     K=1
#     )
#     mu_path_neg, lam_path_neg, t_arc_neg, total_neg = a_star_mu_lambda_path(
#     metric=metric,
#     mu0=mu_left0,
#     mu1=mu_left1,
#     lam0=lam0, lam1=lam1,
#     mus=mus_neg, lams=lams_neg,
#     K=1
#     )
#     # --- Align negative path to positive path's arc-length parameterization
#     from scipy.interpolate import interp1d
#     mu_neg_on_pos  = interp1d(t_arc_neg, mu_path_neg,  kind="linear", bounds_error=False, fill_value="extrapolate")(t_arc_pos)
#     lam_neg_on_pos = interp1d(t_arc_neg, lam_path_neg, kind="linear", bounds_error=False, fill_value="extrapolate")(t_arc_pos)

#     # Use POSITIVE lambda path as the shared covariance path (they should match; this keeps it simple)
#     t_arc    = t_arc_pos
#     lam_path = lam_path_pos


#     # Optional stride AFTER alignment to keep pairs matched
#     # mu_path_pos = mu_path_pos[::stride]
#     # mu_path_neg = mu_path_neg[::stride]
#     # lam_path    = lam_path[::stride]
#     # t_arc       = t_arc[::stride]

#     # breakpoint()
#     # after computing mu_neg_on_pos, lam_neg_on_pos
#     mu_path_pos = mu_path_pos[::stride]
#     mu_path_neg = mu_neg_on_pos[::stride]   # <— use the aligned version
#     lam_path    = lam_path_pos[::stride]    # shared covariance path
#     t_arc       = t_arc_pos[::stride]


#     # choose evenly spaced snapshot indices
#     snap_idx = np.linspace(0, len(lam_path) - 1, snapshot_k, dtype=int)
#     snapshots = []

#     # clf = SGDClassifier(loss="log_loss", penalty='l2',alpha=1e-4, random_state=0)  # slight L2 helps CE
#     clf = SGDClassifier(loss="log_loss", penalty="l2",
#                     alpha=1e-4, learning_rate="constant", eta0=0.05,
#                     random_state=0)
#     clf.partial_fit(X_source, Y_source, classes=np.array([0,1]))

#     accs, ces, counts, accs_b, ces_b = [], [], [], [], []


#     for step, (mu_pos_t, mu_neg_t, lam_t) in enumerate(zip(mu_path_pos, mu_path_neg, lam_path)):

#         # X_t, y_true = sample_dataset_with_mu_lambda(mu_t, lam_t, n=n_per_step, rng=RNG)
#         # X_t, y_true = sample_dataset_at_lambda(mulam_t, n_per_step, rng=RNG)
#         X_t, y_true = sample_dataset_with_means(mu_left=mu_neg_t, mu_right=mu_pos_t,
#                                                 lam_val=lam_t, n=n_per_step, rng=RNG)

#         proba = clf.predict_proba(X_t)[:,1]
#         y_hat = (proba >= 0.5).astype(int)
#         conf  = np.maximum(proba, 1 - proba)
#         mask  = conf >= conf_threshold_low
#         # mask  = mask & (conf <= conf_threshold_high)
#         X_pl, y_pl = X_t[mask], y_hat[mask]
#         counts.append(len(X_pl))
#         # breakpoint()
#         if len(X_pl) >= min_batch:
#             # breakpoint()
#             k_src  = max(1, int(replay_ratio * len(X_pl)))
#             idx_s  = rng.choice(len(X_source), size=k_src, replace=False)
#             X_mix  = np.vstack([X_pl, X_source[idx_s]])
#             y_mix  = np.concatenate([y_pl, Y_source[idx_s]])
            
#             mid = 0.5 * (mu_pos_t + mu_neg_t)
#             err_pl = (y_hat[mask] != y_true[mask]).mean() if mask.any() else np.nan
#             pl_rate = mask.mean()
#             pl_pos_frac = (y_hat[mask] == 1).mean() if mask.any() else np.nan
#             true_pos_frac = (y_true[mask] == 1).mean() if mask.any() else np.nan

#             if step > 0:
#                 for _ in range(max_iter_steps):
#                     clf.partial_fit(X_mix, y_mix)
#                     # clf.fit(X_mix, y_mix)

#             w = float(clf.coef_[0,0]); b = float(clf.intercept_[0])
#             x_star = -b / max(w, 1e-12)
#             P = clf.predict_proba(X_t)[:,1]

#             if step % 50 == 0:
#                 print(f"step {step:03d} | mid={mid: .3f} | x*={x_star: .3f} | "
#                     f"PL used={pl_rate: .2f} | PL err={err_pl: .3f} | "
#                     f"PL +frac={pl_pos_frac: .2f} | true +frac={true_pos_frac: .2f}")

#                 print(f"step {step:02d} | |PL|={len(X_pl):4d} | Acc={accuracy_score(y_true, (P >= 0.5).astype(int)):.3f} | CE={log_loss(y_true, P, labels=[0,1]):.3f}")


#         P = clf.predict_proba(X_t)[:,1]
#         accs.append(accuracy_score(y_true, (P >= 0.5).astype(int)))
#         ces.append(log_loss(y_true, P, labels=[0,1]))

#         P_b = clf_src.predict_proba(X_t)[:,1]
#         accs_b.append(accuracy_score(y_true, (P_b >= 0.5).astype(int)))
#         ces_b.append(log_loss(y_true, P_b, labels=[0,1]))
#         if step in set(snap_idx):
#             snapshots.append({
#                 "step": int(step),
#                 "mu_left": float(mu_neg_t),
#                 "mu_right": float(mu_pos_t),
#                 "lam": float(lam_t),
#                 "clf": copy.deepcopy(clf)   # keep the model at this step
#             })

#     results = {
#         "mu_path_pos": mu_path_pos,
#         "mu_path_neg": mu_path_neg,
#         "lam_path": lam_path,
#         "t_arc": t_arc,
#         "accs": np.array(accs), "ces": np.array(ces), "counts": np.array(counts),
#         "accs_base": np.array(accs_b), "ces_base": np.array(ces_b),
#         "snapshots": snapshots
#     }

#     return results
from typing import Optional
import numpy as np
import copy
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

# ---- Sinkhorn helpers (3.8+ Optional typing)
import ot

def sinkhorn_barycentric_map(X_src: np.ndarray,
                             X_tgt: np.ndarray,
                             reg: float = 0.05,
                             a: Optional[np.ndarray] = None,
                             b: Optional[np.ndarray] = None) -> np.ndarray:
    n, m = len(X_src), len(X_tgt)
    if a is None: a = np.full(n, 1.0/n)
    if b is None: b = np.full(m, 1.0/m)
    M = ot.dist(X_src, X_tgt, metric='euclidean')**2
    P = ot.sinkhorn(a, b, M, reg)
    denom = P.sum(axis=1, keepdims=True) + 1e-12
    return (P @ X_tgt) / denom

def interpolate_classwise_sinkhorn(X0: np.ndarray, y0: np.ndarray,
                                   X1: np.ndarray, y1: np.ndarray,
                                   t: float, reg: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    classes = np.unique(np.concatenate([y0, y1]))
    outX, outy = [], []
    for c in classes:
        X0c = X0[y0 == c]; X1c = X1[y1 == c]
        if len(X0c) == 0 or len(X1c) == 0:  # skip missing
            continue
        Tc = sinkhorn_barycentric_map(X0c, X1c, reg=reg)
        Xtc = (1.0 - t) * X0c + t * Tc
        outX.append(Xtc)
        outy.append(np.full(len(Xtc), c, dtype=y0.dtype))
    if not outX:
        return np.empty((0, X0.shape[1])), np.empty((0,), dtype=y0.dtype)
    return np.vstack(outX), np.concatenate(outy)

# ---- Your function (fixed)

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
                               max_iter_steps=5,
                               # OT options
                               sample_ot: bool = False,
                               classwise_ot: bool = False,
                               sinkhorn_reg: float = 0.05,
                               X_target: Optional[np.ndarray] = None,
                               Y_target: Optional[np.ndarray] = None
                               ) -> Dict[str, np.ndarray]:
    """Self-train along A* arc. If sample_ot=True, generate X_t by Sinkhorn displacement interpolation."""
    # --- Build A* paths to get t_arc and retain plotting info
    mus_neg, lams_neg = build_grid_mu_lambda(mu_left0, mu_left1, lam0, lam1, grid_n, grid_n)
    mus_pos, lams_pos = build_grid_mu_lambda(mu_right0, mu_right1, lam0, lam1, grid_n, grid_n)

    mu_path_pos, lam_path_pos, t_arc_pos, _ = a_star_mu_lambda_path(
        metric=metric, mu0=mu_right0, mu1=mu_right1, lam0=lam0, lam1=lam1,
        mus=mus_pos, lams=lams_pos, K=1
    )
    mu_path_neg, lam_path_neg, t_arc_neg, _ = a_star_mu_lambda_path(
        metric=metric, mu0=mu_left0, mu1=mu_left1, lam0=lam0, lam1=lam1,
        mus=mus_neg, lams=lams_neg, K=1
    )
    mu_neg_on_pos = interp1d(t_arc_neg, mu_path_neg, kind="linear",
                             bounds_error=False, fill_value="extrapolate")(t_arc_pos)

    # Shared schedule after alignment + stride
    mu_path_pos = mu_path_pos[::stride]
    mu_path_neg = mu_neg_on_pos[::stride]
    lam_path    = lam_path_pos[::stride]
    t_arc       = t_arc_pos[::stride]

    # --- Classifier (constant LR)
    clf = SGDClassifier(loss="log_loss", penalty="l2",
                        alpha=1e-4, learning_rate="constant", eta0=0.05,
                        random_state=0)
    clf.partial_fit(X_source, Y_source, classes=np.array([0, 1]))

    # --- Bookkeeping
    snapshots = []
    accs, ces, counts, accs_b, ces_b = [], [], [], [], []
    snap_idx = np.linspace(0, len(t_arc) - 1, snapshot_k, dtype=int)

    # --- If using OT, prepare once
    if sample_ot:
        if X_target is None or Y_target is None:
            # build a fixed target cloud once using endpoint params (or plug your real target)
            X_target, Y_target = sample_dataset_with_means(
                mu_left=mu_left1, mu_right=mu_right1, lam_val=lam1,
                n=len(X_source), rng=rng
            )

        if classwise_ot:
            # Precompute nothing heavy per step; classwise barycentric maps computed inside helper per step.
            # If performance is an issue and class sizes are large, cache per-class T once and reuse to form X_t = (1-t)X0c + tTc.
            pass
        else:
            # Global barycentric map once
            T_global = sinkhorn_barycentric_map(X_source, X_target, reg=sinkhorn_reg)

    # --- Main loop over steps (use A* arc t for consistency)
    for step, (t, mu_pos_t, mu_neg_t, lam_t) in enumerate(zip(t_arc, mu_path_pos, mu_path_neg, lam_path)):

        # --- Build unlabeled batch X_t
        if sample_ot:
            if classwise_ot:
                X_pool, y_pool_dbg = interpolate_classwise_sinkhorn(
                    X0=X_source, y0=Y_source, X1=X_target, y1=Y_target,
                    t=t, reg=sinkhorn_reg
                )
            else:
                X_pool = (1.0 - t) * X_source + t * T_global
                y_pool_dbg = Y_source  # only for evaluation/monitoring
            # subsample to n_per_step
            if len(X_pool) > n_per_step:
                idx = rng.choice(len(X_pool), size=n_per_step, replace=False)
                X_t = X_pool[idx]; y_true = y_pool_dbg[idx]
            else:
                X_t = X_pool; y_true = y_pool_dbg
        else:
            # Parametric Gaussian generator (your original path-based sampling)
            X_t, y_true = sample_dataset_with_means(
                mu_left=mu_neg_t, mu_right=mu_pos_t,
                lam_val=lam_t, n=n_per_step, rng=rng
            )

        # --- Pseudo-labeling
        proba = clf.predict_proba(X_t)[:, 1]
        y_hat = (proba >= 0.5).astype(int)
        conf  = np.maximum(proba, 1 - proba)

        # Annealed confidence gate
        tau_hi, tau_lo = conf_threshold_high, conf_threshold_low
        tau = tau_hi + (tau_lo - tau_hi) * (step / max(1, len(t_arc) - 1))
        mask = conf >= tau

        # Balance by predicted class
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
            w_pl = conf[sel]

            # optional tiny replay
            k_src = int(replay_ratio * len(sel))
            if k_src > 0:
                idx_s = rng.choice(len(X_source), size=k_src, replace=False)
                X_mix  = np.vstack([X_pl, X_source[idx_s]])
                y_mix  = np.concatenate([y_pl, Y_source[idx_s]])
                w_mix  = np.concatenate([w_pl, np.ones(k_src)])
            else:
                X_mix, y_mix, w_mix = X_pl, y_pl, w_pl

            # a few repeats
            for _ in range(max_iter_steps):
                clf.partial_fit(X_mix, y_mix, sample_weight=w_mix)

        # --- Metrics
        P = clf.predict_proba(X_t)[:, 1]
        accs.append(accuracy_score(y_true, (P >= 0.5).astype(int)))
        ces.append(log_loss(y_true, P, labels=[0, 1]))

        P_b = clf_src.predict_proba(X_t)[:, 1]
        accs_b.append(accuracy_score(y_true, (P_b >= 0.5).astype(int)))
        ces_b.append(log_loss(y_true, P_b, labels=[0, 1]))

        # occasional prints (keep your mid/x* debug if parametric)
        if (step % 50 == 0) and (not sample_ot):
            w = float(clf.coef_[0, 0]); b = float(clf.intercept_[0])
            x_star = -b / max(abs(w), 1e-12)
            mid = 0.5 * (mu_pos_t + mu_neg_t)
            pl_rate = mask.mean()
            err_pl = (y_hat[mask] != y_true[mask]).mean() if mask.any() else np.nan
            pl_pos_frac = (y_hat[mask] == 1).mean() if mask.any() else np.nan
            true_pos_frac = (y_true[mask] == 1).mean() if mask.any() else np.nan
            print(f"step {step:03d} | mid={mid: .3f} | x*={x_star: .3f} | acc={accuracy_score(y_true, (P >= 0.5).astype(int)):.3f} | ce={log_loss(y_true, P, labels=[0,1]):.3f} | "
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
        "counts":      np.array(counts),
        "accs_base":   np.array(accs_b),
        "ces_base":    np.array(ces_b),
        "snapshots":   snapshots
    }

# =========================================
# Section 6: Plotting helpers
# =========================================

def decision_boundary_1d(clf):
    # For 1D logistic: w*x + b = 0  ->  x* = -b/w
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

    # Pick a global x-range that covers all snapshots (≈ μ±4σ)
    xs = []
    for s in snapshots:
        sd = np.sqrt(s["lam"])
        xs += [s["mu_left"] - 4*sd, s["mu_left"] + 4*sd,
               s["mu_right"] - 4*sd, s["mu_right"] + 4*sd]
    xmin, xmax = min(xs), max(xs)
    B = np.linspace(xmin, xmax, bins)

    # Compute baseline boundary once (optional)
    xstar_base = decision_boundary_1d(clf_src) if clf_src is not None else None

    # One panel per snapshot
    # W = max(4, int(3.5 + 0.5*len(snapshots)))  # a bit wider for many panels
    fig, axes = plt.subplots(1, len(snapshots), figsize=(16, 3.6), sharey=True)

    if len(snapshots) == 1:
        axes = [axes]

    for ax, s in zip(axes, snapshots):
        X, Y = sample_dataset_with_means(mu_left=s["mu_left"], mu_right=s["mu_right"],
                                         lam_val=s["lam"], n=n_vis, rng=rng)
        ax.hist(X[Y==0].ravel(), bins=B, alpha=0.6, density=True, label="class 0")
        ax.hist(X[Y==1].ravel(), bins=B, alpha=0.6, density=True, label="class 1")

        # Current model boundary
        xstar = decision_boundary_1d(s["clf"])
        if xstar is not None:
            ax.axvline(xstar, color="k", linestyle="--", linewidth=2, label="self-train x*")

        # Baseline boundary (source model), if provided
        if xstar_base is not None:
            ax.axvline(xstar_base, color="k", linestyle=":", linewidth=2, label="source x*")

        ax.set_title(f"step {s['step']}\nμL={s['mu_left']:.2f}, μR={s['mu_right']:.2f}, λ={s['lam']:.1f}")
        ax.set_xlabel("x")
    axes[0].set_ylabel("density")
    axes[0].legend(loc="upper right")
    plt.tight_layout()
    plt.show()

    plt.savefig(f"{title}.png")

    # Optional: probability curves per snapshot (second figure)
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


# =========================================
# Section 7: Example usage / “main”
# =========================================


def plot_expansion_both(runs: Dict[str, Dict[str, np.ndarray]],
                        key: str, ylabel: str, title: str):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,4))

    marker_stride = 10  # only show one marker every 10 steps

    # --- (A) Normalized: t in [0,1]
    ax1 = plt.subplot(1,2,1)
    # for m in METRICS:
    for m, run in runs.items():
        t = runs[m]["t_arc"]
        y_self = runs[m][key]
        y_base = runs[m]["accs_base"] if key == "accs" else runs[m]["ces_base"]

        # ax1.plot(t, y_self, linewidth=2, label=f"{m} — self-train")
        ax1.plot(t[::marker_stride], y_self[::marker_stride],label=f"{m} — self-train")  # markers every 10
        ax1.plot(t[::marker_stride], y_base[::marker_stride], linestyle=':', linewidth=2, label=f"{m} — baseline")

    ax1.set_xlabel("t (normalized A* arc-length)")
    ax1.set_ylabel(ylabel)
    ax1.set_title(f"{title} — normalized")
    ax1.legend()

    # --- (B) Unnormalized: step index
    ax2 = plt.subplot(1,2,2)
    # breakpoint()
    # for m in METRICS:
    for m, run in runs.items():
        y_self = runs[m][key]
        y_base = runs[m]["accs_base"] if key == "accs" else runs[m]["ces_base"]
        steps = np.arange(len(y_self))

        # ax2.plot(steps, y_self, linewidth=2, label=f"{m} — self-train")
        ax2.plot(steps[::marker_stride], y_self[::marker_stride],label=f"{m} — self-train")
        ax2.plot(steps[::marker_stride], y_base[::marker_stride], linestyle=':', linewidth=2, label=f"{m} — baseline")

    ax2.set_xlabel("step index")
    ax2.set_ylabel(ylabel)
    ax2.set_title(f"{title} — unnormalized (steps)")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f"{title}.png")
    plt.show()


def main():
    # ----- Global Config (defaults)
    LAM0, LAM1 = 1.0, 2.0      # endpoints for eigenvalue
    D = 1                        # data dimension
    MU_LEFT0,  MU_RIGHT0  = -2.0, +2.0    # class means at t = 0
    MU_LEFT1,  MU_RIGHT1  = 2, 5.0    # class means at t = 1

    RNG = np.random.default_rng(7)

    # Non-identity on (x,y): s2 != 1 or rho != 0 makes Sigma non-identity
    S2  = 0.5                          # variance on y
    RHO = 0.6                          # correlation between x and y
    MUY_LEFT, MUY_RIGHT = 0.0, 0.0     # keep y-means fixed (can vary if you want)

    X_source, Y_source, clf_src = train_source_model(n_source=5000, lam_source=LAM0, mu_left0=MU_LEFT0, mu_right0=MU_RIGHT0, rng=RNG)
    ts = np.linspace(0, 1, 9); n_per_t = 4000

    runs_expansion = {
        m: self_train_along_expansion(m, n_per_step=4000,
                                        X_source=X_source, Y_source=Y_source, clf_src=clf_src,
                                        mu_left0=MU_LEFT0, mu_left1=MU_LEFT1, mu_right0=MU_RIGHT0, mu_right1=MU_RIGHT1,
                                        lam0=LAM0, lam1=LAM1, grid_n=GRID_N,
                                        neighbor_span=NEIGHBOR_SPAN, stride=1, rng=RNG)
        for m in METRICS
    }
    runs_expansion['OT'] = self_train_along_expansion("OT", n_per_step=4000,
                                    X_source=X_source, Y_source=Y_source, clf_src=clf_src,
                                    mu_left0=MU_LEFT0, mu_left1=MU_LEFT1, mu_right0=MU_RIGHT0, mu_right1=MU_RIGHT1,
                                    lam0=LAM0, lam1=LAM1, grid_n=GRID_N,
                                    neighbor_span=NEIGHBOR_SPAN, stride=1, rng=RNG, sample_ot=True)


    plot_expansion_both(runs_expansion, "accs", "Accuracy", "Accuracy vs t (A* path)")
    plot_expansion_both(runs_expansion, "ces",  "Cross-entropy ↓", "Risk vs t (A* path)")
    plot_evolution_snapshots(runs_expansion["W2"]["snapshots"], clf_src=clf_src, n_vis=3000, bins=80, rng=RNG, title="evolution_snapshots_W2")
    plot_evolution_snapshots(runs_expansion["FR"]["snapshots"], clf_src=clf_src, n_vis=3000, bins=80, rng=RNG, title="evolution_snapshots_FR")
    plot_evolution_snapshots(runs_expansion["eta"]["snapshots"], clf_src=clf_src, n_vis=3000, bins=80, rng=RNG, title="evolution_snapshots_eta")
    plot_evolution_snapshots(runs_expansion["OT"]["snapshots"], clf_src=clf_src, n_vis=3000, bins=80, rng=RNG, title="evolution_snapshots_OT")


if __name__ == "__main__":
    main()