# ============================================================
# Exponential-class-conditionals: Expansion & Self-Training
# Only NEW/CHANGED pieces are defined here.
# Unchanged utilities are imported from unsup_exp.py
# ============================================================

import copy
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

# ---------- IMPORT UNCHANGED UTILS ----------
# Make sure these exist in your unsup_exp.py with the same names
from unsup_gauss import (
    sinkhorn_barycentric_map,        # unchanged OT barycentric map
    self_train_on_ot_interpolation,  # unchanged OT runner with label_mode {'pl','ot'}
    plot_compare_runs,     
    decision_boundary_1d,
    # plot_evolution_snapshot,          # unchanged dual-panel plotting (normalized vs steps)
    # If you already have these helpers, you can import them too:
    # eval_final_on_target, summarize_final, eval_curve_on_target, ...
)

RNG = np.random.default_rng(7)
METRICS = ("FR", "W2", "eta")


# ------------------------------
# Sampling & source training (Exponential)
# ------------------------------
def sample_dataset_exp(lam0: float, lam1: float, n: int, rng=None, pi: float = 0.5,
                       ensure_both: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Mixture of two Exponential dists with rates lam0 (class 0) and lam1 (class 1)."""
    rng = np.random.default_rng() if rng is None else rng
    y = (rng.random(n) < pi).astype(int)
    if ensure_both and n >= 2 and (y.min() == y.max()):
        y[0], y[1] = 0, 1
    x = np.empty(n, dtype=float)
    x[y == 0] = rng.exponential(scale=1.0 / lam0, size=(y == 0).sum())
    x[y == 1] = rng.exponential(scale=1.0 / lam1, size=(y == 1).sum())
    return x.reshape(-1, 1), y


def train_source_model_exp(n_source: int,
                           lam0_src: float,
                           lam1_src: float,
                           pi_src: float = 0.5,
                           rng=RNG) -> Tuple[np.ndarray, np.ndarray, LogisticRegression]:
    Xs, Ys = sample_dataset_exp(lam0_src, lam1_src, n_source, rng=rng, pi=pi_src)
    clf_src = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=2000)
    clf_src.fit(Xs, Ys)
    return Xs, Ys, clf_src


# ------------------------------
# Metric paths for Exponential (per-class rate λ)
# ------------------------------
def lambda_interp(metric: str, lam0: float, lam1: float, t: float) -> float:
    """
    Interpolation of exponential rates per metric:
      FR : λ_t = λ0^(1-t) * λ1^t         (geometric)
      W2 : 1/λ_t = (1-t)/λ0 + t/λ1       (harmonic)
      η  : λ_t = (1-t)λ0 + tλ1           (arithmetic; natural-parameter linear)
    """
    if metric == "FR":
        return (lam0 ** (1.0 - t)) * (lam1 ** t)
    elif metric == "W2":
        return 1.0 / ((1.0 - t) / lam0 + t / lam1)
    elif metric == "eta":
        return (1.0 - t) * lam0 + t * lam1
    else:
        raise ValueError(f"Unknown metric {metric}")


def pairwise_lambda_path(metric: str,
                         lam0_src: float, lam1_src: float,
                         lam0_tgt: float, lam1_tgt: float,
                         t_arc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lam0_t = np.array([lambda_interp(metric, lam0_src, lam0_tgt, t) for t in t_arc], dtype=float)
    lam1_t = np.array([lambda_interp(metric, lam1_src, lam1_tgt, t) for t in t_arc], dtype=float)
    return lam0_t, lam1_t

import numpy as np
import matplotlib.pyplot as plt

def plot_evolution_snapshots_exp(
    snapshots, clf_src=None, n_vis=3000, bins=80, rng=None, title="evolution_exp_snapshots"
):
    """
    Visualize evolution for mixtures of exponentials using snapshot dicts.
    Snapshot keys expected:
      - rates per class: ('lam0','lam1') OR ('rate0','rate1') OR ('lam_left','lam_right')
      - optional mixing weight: 'pi'  (default 0.5 if missing)
      - a trained classifier under 'clf'
      - 'step' for the panel title (optional)

    Requires:
      - decision_boundary_1d(clf)
      - sample_dataset_exp(lam0, lam1, n, rng, pi=0.5)
    """
    rng = np.random.default_rng() if rng is None else rng
    if len(snapshots) == 0:
        print("No snapshots to plot.")
        return

    def _get_rate(s, cands):
        for k in cands:
            if k in s:
                return float(s[k])
        raise KeyError(f"Snapshot missing any of keys {cands}")

    # pick a robust x-range from 99.9% quantiles of exponentials: q = -ln(1-p)/λ
    # we'll compute per-snapshot and then take the global max
    def _q999(lam):
        return 6.907755278982137 / max(lam, 1e-12)  # ln(1000)/λ

    x_max_list = []
    for s in snapshots:
        lam0 = _get_rate(s, ("lam0","rate0","lam_left"))
        lam1 = _get_rate(s, ("lam1","rate1","lam_right"))
        x_max_list.append(max(_q999(lam0), _q999(lam1)))
    xmin, xmax = 0.0, float(max(x_max_list))
    B = np.linspace(xmin, xmax, bins)

    # baseline decision boundary (if provided)
    xstar_base = decision_boundary_1d(clf_src) if clf_src is not None else None

    # ---------- Panel A: histograms ----------
    fig, axes = plt.subplots(1, len(snapshots), figsize=(16, 3.6), sharey=True)
    if len(snapshots) == 1:
        axes = [axes]

    for ax, s in zip(axes, snapshots):
        lam0 = _get_rate(s, ("lam0","rate0","lam_left"))
        lam1 = _get_rate(s, ("lam1","rate1","lam_right"))
        pi   = float(s.get("pi", 0.5))
        step = s.get("step", "?")

        X, Y = sample_dataset_exp(lam0, lam1, n=n_vis, rng=rng, pi=pi)
        ax.hist(X[Y==0].ravel(), bins=B, alpha=0.6, density=True, label="class 0")
        ax.hist(X[Y==1].ravel(), bins=B, alpha=0.6, density=True, label="class 1")

        xstar = decision_boundary_1d(s["clf"])
        if xstar is not None:
            ax.axvline(xstar, color="k", linestyle="--", linewidth=2, label="self-train x*")
        if xstar_base is not None:
            ax.axvline(xstar_base, color="k", linestyle=":", linewidth=2, label="source x*")

        ax.set_title(f"step {step}\nλ0={lam0:.3f}, λ1={lam1:.3f}, π={pi:.2f}")
        ax.set_xlabel("x")

    axes[0].set_ylabel("density")
    axes[0].legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f"{title}.png", dpi=120)
    plt.show()

    # ---------- Panel B: classifier probabilities ----------
    xgrid = np.linspace(xmin, xmax, 400).reshape(-1, 1)
    fig, axes = plt.subplots(1, len(snapshots), figsize=(10, 3.2), sharey=True)
    if len(snapshots) == 1:
        axes = [axes]

    for ax, s in zip(axes, snapshots):
        step = s.get("step", "?")
        p = s["clf"].predict_proba(xgrid)[:, 1]
        ax.plot(xgrid, p, linewidth=2, label="p(y=1|x)")
        ax.axhline(0.5, color="k", linestyle=":", linewidth=1)
        xstar = decision_boundary_1d(s["clf"])
        if xstar is not None:
            ax.axvline(xstar, color="k", linestyle="--", linewidth=1)
        if xstar_base is not None:
            ax.axvline(xstar_base, color="k", linestyle=":", linewidth=1)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"step {step}")
        ax.set_xlabel("x")

    axes[0].set_ylabel("probability")
    axes[0].legend(loc="upper left")
    plt.tight_layout()
    plt.show()


# ------------------------------
# EM for 2-component Exponential mixture
# ------------------------------
def em_two_exponentials(X: np.ndarray,
                        lam_init: Dict[int, float],
                        pi_init: Dict[int, float],
                        max_iter: int = 200,
                        tol: float = 1e-6,
                        verbose: bool = False) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    EM for mixture: π_c * λ_c * exp(-λ_c x), c in {0,1}.
    M-step: π_c = N_c / n, λ_c = N_c / sum_i γ_ic x_i.
    """
    x = X.ravel()
    n = x.size
    lam = {0: float(lam_init[0]), 1: float(lam_init[1])}
    pi = {0: float(pi_init[0]), 1: float(pi_init[1])}

    ll_old = -np.inf
    for it in range(max_iter):
        l0 = np.log(pi[0] + 1e-12) + np.log(lam[0] + 1e-12) - lam[0] * x
        l1 = np.log(pi[1] + 1e-12) + np.log(lam[1] + 1e-12) - lam[1] * x
        m = np.maximum(l0, l1)
        r0 = np.exp(l0 - m); r1 = np.exp(l1 - m)
        s = r0 + r1 + 1e-12
        gamma0 = r0 / s; gamma1 = r1 / s

        N0 = gamma0.sum(); N1 = gamma1.sum()
        pi[0] = N0 / n;    pi[1] = N1 / n
        lam[0] = float(N0 / ((gamma0 * x).sum() + 1e-12))
        lam[1] = float(N1 / ((gamma1 * x).sum() + 1e-12))

        ll = np.sum(np.log(np.exp(l0 - m) + np.exp(l1 - m)) + m)
        if verbose:
            print(f"EM it={it:03d} | N0={N0:.1f} λ0={lam[0]:.4f}  N1={N1:.1f} λ1={lam[1]:.4f}  ll={ll:.2f}")
        if ll - ll_old < tol: break
        ll_old = ll

    # ensure class-0 is "faster" (bigger λ) to keep left/right convention stable
    if lam[0] < lam[1]:
        lam = {0: lam[1], 1: lam[0]}
        pi  = {0: pi[1], 1: pi[0]}
    return lam, pi


# ------------------------------
# Runner: parametric exponential path (supervised or PL)
# ------------------------------
def self_train_along_expansion_exp(
    metric: str,
    n_per_step: int,
    X_source: np.ndarray,
    Y_source: np.ndarray,
    clf_src: LogisticRegression,
    lam0_src: float, lam1_src: float,          # source class rates
    lam0_tgt: float, lam1_tgt: float,          # target class rates
    t_arc: np.ndarray,
    rng=RNG,
    pi_src: float = 0.5, pi_tgt: float = 0.5,
    supervised: bool = False,
    conf_threshold_low: float = 0.6,
    conf_threshold_high: float = 0.9,
    min_batch: int = 100,
    replay_ratio: float = 0.0,
    max_iter_steps: int = 5,
    X_target: Optional[np.ndarray] = None,     # optional held-out
    Y_target: Optional[np.ndarray] = None,
    snapshot_k: int = 6
) -> Dict[str, np.ndarray]:

    t_arc = np.asarray(t_arc); T = len(t_arc)
    lam0_path, lam1_path = pairwise_lambda_path(metric, lam0_src, lam1_src, lam0_tgt, lam1_tgt, t_arc)

    clf = SGDClassifier(loss="log_loss", penalty="l2",
                        alpha=1e-4, learning_rate="constant", eta0=0.05,
                        random_state=0)
    clf.partial_fit(X_source, Y_source, classes=np.array([0, 1]))

    snapshots = []
    accs, ces, accs_b, ces_b, counts = [], [], [], [], []
    accs_target, ces_target = [], []
    snap_idx = np.linspace(0, T - 1, snapshot_k, dtype=int)

    for step, t in enumerate(t_arc):
        pi_t = (1 - t) * pi_src + t * pi_tgt
        lam0_t = float(lam0_path[step])
        lam1_t = float(lam1_path[step])     
        X_t, y_true = sample_dataset_exp(lam0_path[step], lam1_path[step], n_per_step, rng=rng, pi=pi_t)

        if supervised:
            counts.append(len(X_t))
            for _ in range(max_iter_steps):
                clf.partial_fit(X_t, y_true)
        else:
            proba = clf.predict_proba(X_t)[:, 1]
            y_hat = (proba >= 0.5).astype(int)
            conf  = np.maximum(proba, 1 - proba)
            tau = conf_threshold_high + (conf_threshold_low - conf_threshold_high) * (step / max(1, T - 1))
            mask = conf >= tau

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
                # optional replay
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

        # metrics
        P = clf.predict_proba(X_t)[:, 1]
        accs.append(accuracy_score(y_true, (P >= 0.5).astype(int)))
        ces.append(log_loss(y_true, P, labels=[0, 1]))
        Pb = clf_src.predict_proba(X_t)[:, 1]
        accs_b.append(accuracy_score(y_true, (Pb >= 0.5).astype(int)))
        ces_b.append(log_loss(y_true, Pb, labels=[0, 1]))

        if (X_target is not None) and (Y_target is not None):
            Pt = clf.predict_proba(X_target)[:, 1]
            accs_target.append(accuracy_score(Y_target, (Pt >= 0.5).astype(int)))
            ces_target.append(log_loss(Y_target, Pt, labels=[0, 1]))

        if step in set(snap_idx):
            # snapshots.append({"step": int(step), "clf": copy.deepcopy(clf)})
            snapshots.append({
            "step": step,
            "lam0": lam0_t,          # class 0 rate at step t
            "lam1": lam1_t,          # class 1 rate at step t
            "pi":   pi_t,            # optional mixing weight
            "clf":  copy.deepcopy(clf)
        })

    return {
        "t_arc": t_arc,
        "accs": np.array(accs), "ces": np.array(ces),
        "accs_base": np.array(accs_b), "ces_base": np.array(ces_b),
        "counts": np.array(counts), "snapshots": snapshots,
        "accs_target": np.array(accs_target) if accs_target else np.array([]),
        "ces_target":  np.array(ces_target)  if ces_target  else np.array([]),
    }


# ------------------------------
# Runner: ExpFam unsupervised (EM + η-geodesic, supervised synthetic)
# ------------------------------
def self_train_expFam_unsup_exponential(
    n_per_step: int,
    X_source: np.ndarray,
    Y_source: np.ndarray,
    X_target: np.ndarray,
    clf_src: LogisticRegression,
    t_arc: np.ndarray,
    rng=RNG,
    pi_src: float = 0.5,
    snapshot_k: int = 6,
    max_iter_steps: int = 3
) -> Dict[str, np.ndarray]:

    # estimate source rates
    X0s = X_source[Y_source == 0]; X1s = X_source[Y_source == 1]
    lam0_s = 1.0 / (X0s.mean() + 1e-12)
    lam1_s = 1.0 / (X1s.mean() + 1e-12)
    pi_s = float((Y_source == 0).mean())

    # EM on unlabeled target (init by a coarse split)
    q = np.quantile(X_target.ravel(), [0.35, 0.65])
    left = X_target.ravel() <= q[0]
    lam_init = {
        0: 1.0 / max(X_target[left].mean(), 1e-6),
        1: 1.0 / max(X_target[~left].mean(), 1e-6)
    }
    if lam_init[0] < lam_init[1]:  # keep class-0 as faster
        lam_init = {0: lam_init[1], 1: lam_init[0]}
    pi_init = {0: 0.5, 1: 0.5}

    lam_t, pi_t = em_two_exponentials(X_target, lam_init, pi_init, max_iter=200, tol=1e-6, verbose=False)

    # path: η (linear in λ)
    lam0_path = np.array([lambda_interp("eta", lam0_s, lam_t[0], t) for t in t_arc])
    lam1_path = np.array([lambda_interp("eta", lam1_s, lam_t[1], t) for t in t_arc])
    pi_path   = np.array([(1 - t) * pi_s + t * pi_t[0] for t in t_arc])

    clf = SGDClassifier(loss="log_loss", penalty="l2",
                        alpha=1e-4, learning_rate="constant", eta0=0.05,
                        random_state=0)
    clf.partial_fit(X_source, Y_source, classes=np.array([0, 1]))

    snapshots = []
    accs, ces, accs_b, ces_b, counts = [], [], [], [], []
    snap_idx = np.linspace(0, len(t_arc) - 1, snapshot_k, dtype=int)

    for step, t in enumerate(t_arc):
        n0 = int(round(pi_path[step] * n_per_step)); n1 = n_per_step - n0
        X0 = np.random.default_rng().exponential(scale=1.0 / lam0_path[step], size=n0)
        X1 = np.random.default_rng().exponential(scale=1.0 / lam1_path[step], size=n1)
        X_t = np.concatenate([X0, X1]).reshape(-1, 1)
        y_t = np.concatenate([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)])
        counts.append(len(X_t))
        for _ in range(max_iter_steps):
            clf.partial_fit(X_t, y_t)

        P = clf.predict_proba(X_t)[:, 1]
        accs.append(accuracy_score(y_t, (P >= 0.5).astype(int)))
        ces.append(log_loss(y_t, P, labels=[0, 1]))
        Pb = clf_src.predict_proba(X_t)[:, 1]
        accs_b.append(accuracy_score(y_t, (Pb >= 0.5).astype(int)))
        ces_b.append(log_loss(y_t, Pb, labels=[0, 1]))

        if step in set(snap_idx):
            snapshots.append({"step": int(step), "clf": copy.deepcopy(clf)})

    return {
        "t_arc": t_arc,
        "accs": np.array(accs), "ces": np.array(ces),
        "accs_base": np.array(accs_b), "ces_base": np.array(ces_b),
        "counts": np.array(counts), "snapshots": snapshots,
    }


# ------------------------------
# Main: wire everything together
# ------------------------------
def main():
    # rates (means = 1/λ)
    lam0_src, lam1_src = 1.2, 0.35   # source
    lam0_tgt, lam1_tgt = 0.6, 0.18   # target
    pi_src, pi_tgt = 0.5, 0.5

    # Source and target data
    X_source, Y_source, clf_src = train_source_model_exp(
        n_source=6000, lam0_src=lam0_src, lam1_src=lam1_src, pi_src=pi_src, rng=RNG
    )
    X_target_all, Y_target_all = sample_dataset_exp(lam0_tgt, lam1_tgt, n=16000, rng=RNG, pi=pi_tgt)
    X_target_test, Y_target_test = X_target_all[:4000], Y_target_all[:4000]
    t_arc = np.linspace(0, 1, 201)

    # ----- POINT 1: Metric matters (parametric paths, supervised vs PL) -----
    runs_sup, runs_unsup = {}, {}
    for m in METRICS:
        runs_sup[m] = self_train_along_expansion_exp(
            metric=m, n_per_step=4000,
            X_source=X_source, Y_source=Y_source, clf_src=clf_src,
            lam0_src=lam0_src, lam1_src=lam1_src,
            lam0_tgt=lam0_tgt, lam1_tgt=lam1_tgt,
            t_arc=t_arc, rng=RNG, pi_src=pi_src, pi_tgt=pi_tgt,
            supervised=True,
            X_target=X_target_test, Y_target=Y_target_test,
        )
        runs_unsup[m] = self_train_along_expansion_exp(
            metric=m, n_per_step=4000,
            X_source=X_source, Y_source=Y_source, clf_src=clf_src,
            lam0_src=lam0_src, lam1_src=lam1_src,
            lam0_tgt=lam0_tgt, lam1_tgt=lam1_tgt,
            t_arc=t_arc, rng=RNG, pi_src=pi_src, pi_tgt=pi_tgt,
            supervised=False,
            X_target=X_target_test, Y_target=Y_target_test,
            conf_threshold_low=0.6, conf_threshold_high=0.9,
            replay_ratio=0.0, max_iter_steps=5,
        )

    # Also compare OT interpolation (imported runner)
    run_ot_pl = self_train_on_ot_interpolation(
        n_per_step=4000, X_source=X_source, Y_source=Y_source, clf_src=clf_src,
         t_arc=t_arc, label_mode="pl",
        X_target=X_target_test, Y_target=Y_target_test, sinkhorn_reg=0.05
    )
    run_ot_transport = self_train_on_ot_interpolation(
        n_per_step=4000, X_source=X_source, Y_source=Y_source, clf_src=clf_src,
         t_arc=t_arc, label_mode="ot",
        X_target=X_target_test, Y_target=Y_target_test, sinkhorn_reg=0.05
    )

    # Plots (imported helper)
    plot_compare_runs({f"A^* {m}": runs_sup[m] for m in METRICS},
                      key="accs", ylabel="Accuracy",
                      title="Accuracy, Ground Truth Labels - Exponential Distribution")
    plot_compare_runs({**{f"A^* {m}": runs_unsup[m] for m in METRICS},
                       **{"OT (PL)": run_ot_pl, "OT (transport)": run_ot_transport}},
                      key="accs", ylabel="Accuracy",
                      title="Accuracy, Pseudo-labels")

    plot_compare_runs({f"A^* {m}": runs_sup[m] for m in METRICS},
                      key="accs_target", ylabel="Accuracy",
                      title="Target Accuracy, Ground Truth Labels - Exponential Distribution")
    plot_compare_runs({**{f"A^* {m}": runs_unsup[m] for m in METRICS},
                       **{"OT (PL)": run_ot_pl, "OT (transport)": run_ot_transport}},
                      key="accs_target", ylabel="Accuracy",
                      title="Target Accuracy, Pseudo-labels  - Exponential Distribution")
    plot_evolution_snapshots_exp(runs_sup["W2"]["snapshots"], clf_src=clf_src,
                                title="exp_evolution_W2")
    plot_evolution_snapshots_exp(runs_unsup["FR"]["snapshots"], clf_src=clf_src,
                                title="exp_evolution_FR")
    plot_evolution_snapshots_exp(runs_unsup["eta"]["snapshots"], clf_src=clf_src,
                                title="exp_evolution_eta")



    # ----- POINT 2: EM labels > self-training (ExpFam vs PL) -----
    run_expfam = self_train_expFam_unsup_exponential(
        n_per_step=4000,
        X_source=X_source, Y_source=Y_source,
        X_target=X_target_test,
        clf_src=clf_src,
        t_arc=t_arc, rng=RNG, pi_src=pi_src,
        snapshot_k=6, max_iter_steps=3
    )
    plot_compare_runs({
        "ExpFam (EM + η-geodesic, supervised synthetic)": run_expfam,
        "Parametric W2 — Unsupervised (PL)": runs_unsup["W2"]
    }, key="accs", ylabel="Accuracy", title="POINT 2 — EM vs PL (train)")

    # Final snapshot evaluation on target
    def eval_final_on_target_here(run, X_test, Y_test):
        if run.get("snapshots"):
            clf = run["snapshots"][-1]["clf"]
        else:
            raise ValueError("No snapshots.")
        P = clf.predict_proba(X_test)[:, 1]
        return dict(
            acc=accuracy_score(Y_test, (P >= 0.5).astype(int)),
            ce=log_loss(Y_test, P, labels=[0, 1]),
            auc=roc_auc_score(Y_test, P),
        )

    print("\n=== POINT 2: Final target metrics (last snapshot) ===")
    m_em = eval_final_on_target_here(run_expfam, X_target_test, Y_target_test)
    m_pl = eval_final_on_target_here(runs_unsup["W2"], X_target_test, Y_target_test)
    print(f"ExpFam (EM): Acc={m_em['acc']:.3f}  CE={m_em['ce']:.3f}  AUC={m_em['auc']:.3f}")
    print(f"Parametric W2 — PL: Acc={m_pl['acc']:.3f}  CE={m_pl['ce']:.3f}  AUC={m_pl['auc']:.3f}")


if __name__ == "__main__":
    main()
