import ot
import torch
from util import *
import numpy as np
import time
import torch.nn as nn
        

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

    breakpoint()
    xs_np_valid = xs_np[valid_mask]
    xt_np_valid = xt_np[valid_mask]
    weights_valid = weights[valid_mask]

    if len(xs_np_valid) == 0:
        raise ValueError("No valid log-exp pairs remain after filtering.")

    log_vecs = sphere.metric.log(xt_np_valid, base_point=xs_np_valid)
    x_t = sphere.metric.exp(t * log_vecs, base_point=xs_np_valid)

    return x_t, weights_valid

def generate_domains(n_inter, dataset_s, dataset_t, plan=None, entry_cutoff=0, conf=0):
    print("------------Generate Intermediate domains----------")
    all_domains = []
    # generate_images = []
    
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
    start = 0
    for i in range(1, n_inter+1):
        # x, weights = pushforward(xs, xt, plan, i / (n_inter+1))
        x, y ,weights = pushforward_with_y(xs, ys, xt, yt_hat, plan, i / (n_inter+1))
        y_tensor = torch.tensor(y, dtype=torch.long)
        if isinstance(x, np.ndarray):
            all_domains.append(DomainDataset(torch.from_numpy(x).float(), weights, targets=y_tensor, targets_em=y_tensor))
        else:
            all_domains.append(DomainDataset(x, weights, targets=y_tensor, targets_em=y_tensor))

    
    # generate_images.append(x)
    all_domains.append(dataset_t)

    print(f"Total data for each intermediate domain: {len(x)}")
    # breakpoint()
    return all_domains, yt_hat #, generate_images





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


