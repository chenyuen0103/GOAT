import ot
import torch
from util import *
import numpy as np
import time
import torch.nn as nn
from a_star_util import class_stats_diag, class_stats_full
from dataset import DomainDataset
import torch.nn.functional as F
        

def get_transported_labels(plan, ys, logit=False):
    # plan /= np.sum(plan, 0, keepdims=True)
    ysTemp = ot.utils.label_normalization(np.copy(ys))
    classes = np.unique(ysTemp)
    n = len(classes)
    D1 = np.zeros((n, len(ysTemp)))

    # perform label propagation
    transp = plan

    # set nans to 0
    transp[~ np.isfinite(transp)] = 0

    for c in classes:
        D1[int(c), ysTemp == c] = 1

    # compute propagated labels
    transp_ys = np.dot(D1, transp).T

    if logit:
        return transp_ys 
    
    transp_ys = np.argmax(transp_ys, axis=1)

    return transp_ys


def get_conf_idx(logits, confidence_q=0.2):
    confidence = np.amax(logits, axis=1) - np.amin(logits, axis=1)
    alpha = np.quantile(confidence, confidence_q)
    indices = np.argwhere(confidence >= alpha)[:, 0]
    labels = np.argmax(logits, axis=1)
    
    return labels, indices


def get_OT_plan(X_S, X_T, solver='sinkhorn', weights_S=None, weights_T=None, Y_S=None, numItermax=1e7,
                entropy_coef=1, entry_cutoff=0):

    # X_S, X_T = X_S[:50000], X_T[:50000]
    X_S, X_T = X_S, X_T
    n, m = len(X_S), len(X_T)
    a = np.ones(n) / n if weights_S is None else weights_S
    b = np.ones(m) / m if weights_T is None else weights_T
    print(f'{n} source data, {m} target data. ')
    dist_mat = ot.dist(X_S, X_T).detach().numpy()
    t = time.time()
    if solver == 'emd':
        plan = ot.emd(a, b, dist_mat, numItermax=int(numItermax))
    elif solver == 'sinkhorn':
        plan = ot.sinkhorn(a, b, dist_mat, reg=entropy_coef, numItermax=int(numItermax), stopThr=10e-7)
    elif solver == 'lpl1':
        plan = ot.sinkhorn_lpl1_mm(a, b, Y_S, dist_mat, reg=entropy_coef, numItermax=int(numItermax), stopInnerThr=10e-9)

    if entry_cutoff > 0:
        avg_val = 1 / (n * m)
        print(f'Zero out entries with value < {entry_cutoff}*{avg_val}')
        plan[plan < avg_val * entry_cutoff] = 0

    elapsed = round(time.time() - t, 2)
    print(f"Time for OT calculation: {elapsed}s")
    # plan /= np.sum(plan, 0, keepdims=True)
    # plan[~ np.isfinite(plan)] = 0
    plan = plan * n 

    return plan


def pushforward(X_S, X_T, plan, t):
    print(f'Pushforward to t={t}')
    assert 0 <= t <= 1
    nonzero_indices = np.argwhere(plan > 0)
    weights = plan[plan > 0]
    assert len(nonzero_indices) == len(weights)
    x_t= (1-t)*X_S[nonzero_indices[:,0]] + t*X_T[nonzero_indices[:,1]]

    
    return x_t, weights

def pushforward_with_y(X_S, y_s, X_T, y_T, plan, t):
    print(f'Pushforward to t={t}')
    assert 0 <= t <= 1
    nonzero_indices = np.argwhere(plan > 0)
    weights = plan[plan > 0]
    assert len(nonzero_indices) == len(weights)
    x_t= (1-t)*X_S[nonzero_indices[:,0]] + t*X_T[nonzero_indices[:,1]]
    y_t = (1-t)*y_s[nonzero_indices[:,0]] + t*y_T[nonzero_indices[:,1]]
    
    
    return x_t, y_t, weights


# from geomstats.geometry.hypersphere import Hypersphere
# sphere = Hypersphere(dim=2)


def pushforward_geo(X_S, X_T, plan, t):
    print(f'Geodesic pushforward to t={t}')
    indices = np.argwhere(plan > 0)
    weights = plan[plan > 0]
    
    xs = X_S[indices[:, 0]]
    xt = X_T[indices[:, 1]]

    # Ensure shape is [N, d] not [d, N]
    if xs.shape[0] < xs.shape[1]:
        xs = xs.T
    if xt.shape[0] < xt.shape[1]:
        xt = xt.T

    # Convert to torch and normalize onto the sphere
    xs = torch.tensor(xs, dtype=torch.float64).view(len(xs), -1)
    xt = torch.tensor(xt, dtype=torch.float64).view(len(xt), -1)

    xs = xs / xs.norm(dim=1, keepdim=True).clamp(min=1e-8)
    xt = xt / xt.norm(dim=1, keepdim=True).clamp(min=1e-8)

    xs_np = xs.cpu().numpy()
    xt_np = xt.cpu().numpy()

    assert xs_np.shape == xt_np.shape, f"Shape mismatch: xs={xs_np.shape}, xt={xt_np.shape}"

    inner_prods = np.sum(xs_np * xt_np, axis=1)
    inner_prods = np.clip(inner_prods, -1.0, 1.0)
    valid_mask = (np.abs(inner_prods) < 0.999)

    xs_np_valid = xs_np[valid_mask]
    xt_np_valid = xt_np[valid_mask]
    weights_valid = weights[valid_mask]

    if len(xs_np_valid) == 0:
        raise ValueError("No valid log-exp pairs remain after filtering.")

    log_vecs = sphere.metric.log(xt_np_valid, base_point=xs_np_valid)
    x_t = sphere.metric.exp(t * log_vecs, base_point=xs_np_valid)

    return x_t, weights_valid

# def generate_domains(n_inter, dataset_s, dataset_t, plan=None, entry_cutoff=0, conf=0,
#                     *,
#                     cov_type: str = "diag",        # 'diag' | 'full'  (matches FR/natural generators)
#                     reg: float = 1e-6,             # ridge on eigenvalues for full covariances
#                     ddof: int = 0,                 # 0 → population covariance
#                     ):
#     print("------------Generate Intermediate domains----------")

#     xs, xt = dataset_s.data, dataset_t.data
#     ys = dataset_s.targets

#     if plan is None:
#         if len(xs.shape) > 2:
#             xs_flat, xt_flat = nn.Flatten()(xs), nn.Flatten()(xt)
#             plan = get_OT_plan(xs_flat, xt_flat, solver='emd', entry_cutoff=entry_cutoff)
#         else:
#             plan = get_OT_plan(xs, xt, solver='emd', entry_cutoff=entry_cutoff)

#     logits_t = get_transported_labels(plan, ys, logit=True)
#     yt_hat, conf_idx = get_conf_idx(logits_t, confidence_q=conf) # currenly 0, change later
#     # --- make sure we use the right index dtype for tensor indexing
#     if torch.is_tensor(xt):
#         # conf_idx might be a numpy array of ints/bools -> convert to Long for indexing
#         conf_idx_t = torch.as_tensor(conf_idx, dtype=torch.long, device=xt.device)
#         xt_f = xt[conf_idx_t]
#     else:
#         xt_f = xt[conf_idx]

#     plan_f = plan[:, conf_idx]
#     yt_hat = yt_hat[conf_idx]

#     xt_f_t = xt_f if torch.is_tensor(xt_f) else torch.as_tensor(xt_f).float()
#     yt_hat_t = torch.as_tensor(yt_hat, dtype=torch.long)  # <-- FIX: tensor labels
#     W_t      = torch.ones(len(yt_hat_t), dtype=torch.float32)
#     print(f"Remaining data after confidence filter: {len(conf_idx)}")
#     all_domains = []
#     D_t = DomainDataset(xt_f_t, W_t, targets=yt_hat_t, targets_em=yt_hat_t.clone())


#     for i in range(1, n_inter+1):
#         t = i / (n_inter + 1)
#         # x, weights = pushforward(xs, xt, plan, i / (n_inter+1))
#         x_i, y_i, w_i = pushforward_with_y(xs, ys, xt_f, yt_hat, plan_f, t)
#         y_i = torch.tensor(y_i, dtype=torch.long)
#         if isinstance(x_i, np.ndarray):
#             x_i = torch.from_numpy(x_i).float()
#         all_domains.append(DomainDataset(x_i, w_i, targets=y_i, targets_em=y_i))
#     W_t = torch.ones(len(yt_hat), dtype=torch.float32)
#     # D_t = DomainDataset(xt_f if torch.is_tensor(xt_f) else torch.as_tensor(xt_f).float(),
#     #                     W_t, targets=yt_hat, targets_em=yt_hat)
#     all_domains.append(D_t)
    
#     # -------- stats helpers ----------
#     def _to_np(x):
#         return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

#     def _diag_from_full_stack(Sig_kdd: torch.Tensor) -> torch.Tensor:
#         return torch.stack([torch.diag(Sig_kdd[k]) for k in range(Sig_kdd.shape[0])], dim=0)

#     def _eta_diag_from_mu_var_np(mu_kd: np.ndarray, var_kd: np.ndarray, eps: float = 1e-12):
#         var = np.clip(var_kd, eps, None)
#         prec = 1.0 / var
#         eta1 = prec * mu_kd
#         eta2d = -0.5 * prec
#         return eta1, eta2d

#     def _eta_full_from_Sigma_np(mu_kd: np.ndarray, Sig_kdd: np.ndarray, eps: float = 1e-12):
#         K, d = mu_kd.shape
#         e1 = np.full((K, d), np.nan, dtype=np.float64)
#         e2d = np.full((K, d), np.nan, dtype=np.float64)
#         for k in range(K):
#             S = Sig_kdd[k]
#             if not np.isfinite(S).all():
#                 continue
#             S = 0.5 * (S + S.T)
#             w, V = np.linalg.eigh(S)
#             w = np.clip(w, eps, None)
#             Lam = (V * (1.0 / w)) @ V.T
#             e1[k]  = Lam @ mu_kd[k]
#             e2d[k] = -0.5 * np.diag(Lam)
#         return e1, e2d

#     # collect source/target tensors
#     Xs = xs if torch.is_tensor(xs) else torch.as_tensor(xs)
#     Ys = ys if torch.is_tensor(ys) else torch.as_tensor(ys, dtype=torch.long)
#     Xt = D_t.data if torch.is_tensor(D_t.data) else torch.as_tensor(D_t.data)
#     Yt = D_t.targets_em if torch.is_tensor(D_t.targets_em) else torch.as_tensor(D_t.targets_em, dtype=torch.long)

#     device = Xs.device if torch.is_tensor(Xs) else torch.device("cpu")
#     Xs = Xs.to(device); Xt = Xt.to(device)
#     Ys = Ys.to(device, dtype=torch.long); Yt = Yt.to(device, dtype=torch.long)

#     K = int(max(Ys.max(), Yt.max()).item()) + 1
#     d = Xs.shape[1] if Xs.ndim == 2 else int(np.prod(Xs.shape[1:]))

#     # source stats (t=0)
#     if cov_type == "diag":
#         mu_s, var_s, cnt_s = class_stats_diag(Xs, Ys, K)
#         Sig_s = None
#     else:
#         mu_s, Sig_s, cnt_s = class_stats_full(Xs, Ys, K, reg=reg, ddof=ddof)
#         var_s = _diag_from_full_stack(Sig_s)

#     # target stats (t=1) based on filtered target + pseudo labels
#     if cov_type == "diag":
#         mu_t, var_t, cnt_t = class_stats_diag(Xt, Yt, K)
#         Sig_t = None
#     else:
#         mu_t, Sig_t, cnt_t = class_stats_full(Xt, Yt, K, reg=reg, ddof=ddof)
#         var_t = _diag_from_full_stack(Sig_t)

#     cnt_s = torch.bincount(Yt, minlength=K)   # "source counts by target labels" as requested

#     # (b) For the TARGET step t=1: count from Yt as usual
#     cnt_t = torch.bincount(Yt, minlength=K)

#     present_s = (_to_np(cnt_s) > 0)
#     present_t = (_to_np(cnt_t) > 0)

#     # containers
#     steps      = [0.0]
#     mu_list    = [_to_np(mu_s)]
#     var_list   = [_to_np(var_s)]
#     counts_list= [_to_np(cnt_s).astype(np.int64)]
#     pi_list    = [( _to_np(cnt_s) / max(1, int(cnt_s.sum().item() if torch.is_tensor(cnt_s) else np.sum(cnt_s))) )]
#     total_s     = int(cnt_s.sum().item() if torch.is_tensor(cnt_s) else np.sum(cnt_s))
#     Sigma_list = [] if cov_type == "full" else None
#     if cov_type == "full":
#         Sigma_list.append(_to_np(Sig_s))
#     eta1_list, eta2d_list = [], []

#     # source eta
#     if cov_type == "full":
#         e1, e2d = _eta_full_from_Sigma_np(mu_list[-1], Sigma_list[-1])
#     else:
#         e1, e2d = _eta_diag_from_mu_var_np(mu_list[-1], var_list[-1])
#     eta1_list.append(e1)
#     eta2d_list.append(e2d)

#     # intermediates stats from the generated datasets
#     for i, D in enumerate(all_domains[:-1], start=1):  # exclude the appended target here
#         Xi = D.data if torch.is_tensor(D.data) else torch.as_tensor(D.data)
#         Yi = D.targets if torch.is_tensor(D.targets) else torch.as_tensor(D.targets, dtype=torch.long)
#         Xi = Xi.to(device); Yi = Yi.to(device, dtype=torch.long)

#         if cov_type == "diag":
#             mu_i, var_i, cnt_i = class_stats_diag(Xi, Yi, K)
#             mu_np = _to_np(mu_i); var_np = _to_np(var_i)
#             Sig_np = None
#         else:
#             mu_i, Sig_i, cnt_i = class_stats_full(Xi, Yi, K, reg=reg, ddof=ddof)
#             mu_np = _to_np(mu_i); Sig_np = _to_np(Sig_i)
#             var_np = _to_np(_diag_from_full_stack(Sig_i))

#         # counts for this *actual* intermediate dataset
#         cnt_i = torch.bincount(Yi, minlength=K)
#         cnt_np = _to_np(cnt_i).astype(np.int64)
#         total_i = int(cnt_i.sum().item())
#         steps.append(i / (n_inter + 1))
#         mu_list.append(mu_np)
#         var_list.append(var_np)
#         counts_list.append(cnt_np)
#         pi_list.append((cnt_np / max(1, total_i)))
#         if cov_type == "full":
#             Sigma_list.append(Sig_np)

#         # eta at this step
#         if cov_type == "full":
#             e1, e2d = _eta_full_from_Sigma_np(mu_np, Sig_np)
#         else:
#             e1, e2d = _eta_diag_from_mu_var_np(mu_np, var_np)
#         eta1_list.append(e1)
#         eta2d_list.append(e2d)
#     # final target stats (t=1)
#     # --- target step (t=1) ---
#     steps.append(1.0)
#     mu_np_t  = _to_np(mu_t)
#     var_np_t = _to_np(var_t)
#     mu_list.append(mu_np_t)
#     var_list.append(var_np_t)

#     cnt_np_t = _to_np(cnt_t).astype(np.int64)
#     counts_list.append(cnt_np_t)
#     total_t = int(cnt_t.sum().item())
#     pi_list.append(cnt_np_t.astype(np.float64) / max(1.0, float(total_t)))

#     if cov_type == "full":
#         Sig_np_t = _to_np(Sig_t)
#         Sigma_list.append(Sig_np_t)
#         e1, e2d = _eta_full_from_Sigma_np(mu_np_t, Sig_np_t)
#     else:
#         e1, e2d = _eta_diag_from_mu_var_np(mu_np_t, var_np_t)
#     eta1_list.append(e1)
#     eta2d_list.append(e2d)

#     # pack domain_params
#     domain_params = {
#         "K": int(K),
#         "d": int(d),
#         "cov_type": cov_type,
#         "steps": np.asarray(steps, dtype=np.float64),              # (S,)
#         "mu":    np.asarray(mu_list,  dtype=np.float64),           # (S,K,d)
#         "var":   np.asarray(var_list, dtype=np.float64),           # (S,K,d)
#         "counts": np.asarray(counts_list, dtype=np.int64),         # (S,K)
#         "pi":      np.asarray(pi_list,    dtype=np.float64),       # (S,K)
#         "present_source": present_s.astype(np.bool_),
#         "present_target": present_t.astype(np.bool_),
#         "eta1":      np.asarray(eta1_list,  dtype=np.float64),     # (S,K,d)
#         "eta2_diag": np.asarray(eta2d_list, dtype=np.float64),
#     }
#     if cov_type == "full":
#         domain_params["Sigma"] = np.asarray(Sigma_list, dtype=np.float64)  # (S,K,d,d)

#     # print(f"Total data for each intermediate domain: {len(x)}")
#     # breakpoint()
#     return all_domains, yt_hat, domain_params


def _to_tensor(x, dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(x, dtype=dtype)

def _pushforward_row_bary(X_S, X_T, plan, t, eps=1e-12):
    """
    Row-barycentric (source-centered) pushforward.
    Output size = n_s (number of source points).

      For each source i:
        m_i      = sum_j P_{ij}
        mu_tgt_i = (sum_j P_{ij} * X_T[j]) / max(m_i, eps)
        x_i(t)   = (1 - t) * X_S[i] + t * mu_tgt_i
        w_i      = m_i

    If a source row has zero mass (m_i == 0), we set mu_tgt_i = X_S[i],
    yielding x_i(t) = X_S[i] (i.e., no movement), and keep w_i = 0.
    """
    assert 0.0 <= float(t) <= 1.0
    XS = _to_tensor(X_S, dtype=torch.float32)
    XT = _to_tensor(X_T, dtype=torch.float32)
    P  = _to_tensor(plan, dtype=torch.float32)

    assert XS.ndim == 2 and XT.ndim == 2
    assert P.shape == (XS.shape[0], XT.shape[0]), \
        f"plan has shape {tuple(P.shape)} but expected {(XS.shape[0], XT.shape[0])}"

    row_mass = P.sum(dim=1)                             # (n_s,)
    safe_mass = torch.clamp(row_mass, min=eps).unsqueeze(1)  # (n_s,1)
    # row-wise barycenter of target features
    mu_tgt = (P @ XT) / safe_mass                       # (n_s, d)
    # For truly zero-mass rows, fall back to staying at source location
    zero_mask = (row_mass <= eps).unsqueeze(1)          # (n_s,1)
    mu_tgt = torch.where(zero_mask, XS, mu_tgt)         # stay put if no mass

    x_t = (1.0 - t) * XS + t * mu_tgt                   # (n_s, d)
    w   = row_mass                                      # (n_s,)
    return x_t, w


def generate_domains(n_inter, dataset_s, dataset_t, plan=None, entry_cutoff=0, conf=0):
    print("------------Generate Intermediate domains----------")
    all_domains = []
    
    xs, xt = dataset_s.data, dataset_t.data
    ys = dataset_s.targets

    if plan is None:
        if len(xs.shape) > 2:
            xs_flat, xt_flat = nn.Flatten()(xs), nn.Flatten()(xt)
            plan = get_OT_plan(xs_flat, xt_flat, solver='emd', entry_cutoff=entry_cutoff)
        else:
            plan = get_OT_plan(xs, xt, solver='emd', entry_cutoff=entry_cutoff)

    logits_t = get_transported_labels(plan, ys, logit=True)
    yt_hat, conf_idx = get_conf_idx(logits_t, confidence_q=conf)
    xt = xt[conf_idx]
    plan = plan[:, conf_idx]
    yt_hat = yt_hat[conf_idx]

    print(f"Remaining data after confidence filter: {len(conf_idx)}")

    for i in range(1, n_inter+1):
        # x, weights = pushforward(xs, xt, plan, i / (n_inter+1))
        x, weights = _pushforward_row_bary(xs, xt, plan, i / (n_inter+1))
        if isinstance(x, np.ndarray):
            all_domains.append(DomainDataset(torch.from_numpy(x).float(), weights))
        else:
            all_domains.append(DomainDataset(x, weights))
    all_domains.append(dataset_t)

    print(f"Total data for each intermediate domain: {len(x)}")

    return all_domains, 0, 0




def ot_ablation(size, mode):
    ns, nt = size, size
    plan = np.zeros((ns, nt))
    ran = np.arange(ns*nt)
    np.random.shuffle(ran)
    idx = ran[:size]

    for i in idx:
        row = i // nt
        col = i-i//nt * nt
        if mode == "random":
            plan[row, col] = np.random.uniform()
        elif mode == "uniform":
            plan[row, col] = 1
    
    plan /= np.sum(plan, 1, keepdims=True)
    plan[~ np.isfinite(plan)] = 0

    return plan


