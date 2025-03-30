import torch
from util import *
import numpy as np
import time
import torch.nn as nn
from tqdm import tqdm
import pickle
import os
device = "cuda:0" if torch.cuda.is_available() else "cpu"

import ot  # pip install POT
import numpy as np
import torch

def compute_w2_distance_pot(X_src, X_tgt, p=2):
    """
    Compute Wasserstein-p distance between empirical distributions X_src and X_tgt using POT.

    Args:
        X_src (torch.Tensor): shape (n, d)
        X_tgt (torch.Tensor): shape (m, d)
        p (int): Power of the Wasserstein distance (usually 2)

    Returns:
        w2_distance (float)
    """
    # Convert to numpy
    Xs = X_src.detach().cpu().numpy()
    Xt = X_tgt.detach().cpu().numpy()

    n, m = len(Xs), len(Xt)
    a = np.ones((n,)) / n  # uniform weights
    b = np.ones((m,)) / m

    # Cost matrix (pairwise squared Euclidean distances)
    M = ot.dist(Xs, Xt, metric='euclidean') ** p

    # Solve OT problem using Sinkhorn or Earth Mover’s Distance (EMD)
    # EMD is the true W2, but slower
    # T = ot.sinkhorn(a, b, M, reg=1e-3)
    T = ot.emd(a, b, M)

    # Wasserstein-p distance
    W_p = np.sum(T * M) ** (1 / p)
    return W_p


def get_conf_idx(logits, confidence_q=0.2):
    confidence = np.amax(logits, axis=1) - np.amin(logits, axis=1)
    alpha = np.quantile(confidence, confidence_q)
    indices = np.argwhere(confidence >= alpha)[:, 0]
    labels = np.argmax(logits, axis=1)
    
    return labels, indices


def wasserstein_interpolation(n_samples, encoded_src, encoded_tgt, delta=0.5, lr=0.05, max_iters=50, device="cuda"):
    """
    Generates intermediate domains by interpolating between the source and target
    distributions in the latent space using Wasserstein barycenters **per class**.

    Args:
        n_samples (int): Number of samples to generate per class.
        encoded_src (Dataset): Encoded source domain with (z, y) pairs.
        encoded_tgt (Dataset): Encoded target domain with (z, y) pairs.
        delta (float): Step size for moving from source to target.
        lr (float): Learning rate for Wasserstein distance optimization.
        max_iters (int): Maximum iterations for convergence.
        device (str): Device to use ('cuda' or 'cpu').

    Returns:
        List of (z, y) tuples for the interpolated dataset.
    """

    # # Step 1: Get latents & labels from both datasets
    # z_src, y_src = get_latents_and_labels(encoded_src)
    # z_tgt, y_tgt = get_latents_and_labels(encoded_tgt)

    # Step 2: Estimate class-conditional Gaussians
    src_class_params = estimate_class_conditional_gaussian_params(encoded_src)
    tgt_class_params = estimate_class_conditional_gaussian_params(encoded_tgt)

    interpolated_samples = []

    # Step 3: Perform Wasserstein interpolation for each class
    unique_classes = sorted(src_class_params.keys())
    for c in unique_classes:
        if c not in tgt_class_params:
            continue  # Skip if the class is missing in the target

        mu_src, Sigma_src = src_class_params[c]
        mu_tgt, Sigma_tgt = tgt_class_params[c]

        # Perform Wasserstein step from source class distribution to target class distribution
        interpolated_mu, interpolated_Sigma, _, _ = wasserstein_step_full(
            mu_t=mu_src, sigma_t=Sigma_src,
            mu_T=mu_tgt, sigma_T=Sigma_tgt,
            delta=delta, lr=lr, max_iters=max_iters
        )

        # Ensure PSD covariance matrix
        interpolated_Sigma = make_positive_definite(interpolated_Sigma)

        # Sample from the interpolated Gaussian
        z_samples = np.random.multivariate_normal(
            mean=to_cpu_numpy(interpolated_mu),
            cov=to_cpu_numpy(interpolated_Sigma),
            size=n_samples
        )
        y_samples = np.full((n_samples,), c, dtype=int)  # Assign class labels

        # Convert to PyTorch tensors
        z_samples = torch.tensor(z_samples, dtype=torch.float32, device=device)
        y_samples = torch.tensor(y_samples, dtype=torch.long, device=device)

        # Collect samples
        interpolated_samples.extend(zip(z_samples, y_samples))

    return interpolated_samples  # Return class-conditional interpolated dataset



def safe_cholesky(Sigma, eps=1e-4, max_tries=5):
    for i in range(max_tries):
        try:
            return torch.linalg.cholesky(Sigma + eps * torch.eye(Sigma.shape[0], device=Sigma.device))
        except RuntimeError:
            eps *= 10
    raise RuntimeError("Cholesky decomposition failed after multiple attempts")


def generate_qda_domains(
    source_dataset,
    target_dataset,
    n_wsteps=10,
    samples_per_class=1000,
    delta=1.0,
    device="cpu",
    use_diagonal_cov=False,
    cache_prefix="qda"
):
    """
    Generates intermediate domains using QDA-style class-conditional interpolation.
    Assumes Sigma_src stays fixed. Interpolates mu_src towards mu_tgt with W2 constraint.

    Returns:
        all_domains: list of DomainDataset objects.
    """
    print("------------Generate Intermediate domains----------")
    # print("Estimating class-conditional Gaussians for SOURCE...")
    source_class_params = estimate_class_conditional_gaussian_params(
        source_dataset,
        cache_path=f"{cache_prefix}_source.pkl",
        force_recompute=False,
        use_diagonal_cov=use_diagonal_cov
    )

    # print("Estimating class-conditional Gaussians for TARGET...")
    target_class_params = estimate_class_conditional_gaussian_params(
        target_dataset,
        cache_path=f"{cache_prefix}_target.pkl",
        force_recompute=False,
        use_diagonal_cov=use_diagonal_cov
    )

    all_domains = []
    classes = sorted(source_class_params.keys())

    # Initialize each class's current mean as source mean
    current_mus = {c: source_class_params[c][0].clone() for c in classes}

    for step in range(1, n_wsteps + 1):
        z_list, y_list = [], []

        for c in classes:
            mu_src, Sigma_src = source_class_params[c]
            mu_tgt, _ = target_class_params[c]
            mu_cur = current_mus[c]

            # Compute direction vector
            direction = mu_tgt - mu_cur
            dist = torch.norm(direction)
            step_size = min(delta, dist)
            new_mu = mu_cur + step_size * direction / (dist + 1e-8)
            current_mus[c] = new_mu.detach()

            # Sample from N(new_mu, Sigma_src)
            Sigma_eps = 1e-10 * torch.eye(Sigma_src.shape[0], device=device)
            L = torch.linalg.cholesky(Sigma_src + Sigma_eps)
            noise = torch.randn(samples_per_class, new_mu.shape[0], device=device)
            z_samples = new_mu + noise @ L.T
            y_samples = torch.full((samples_per_class,), c, dtype=torch.long, device=device)
            weights = np.ones((samples_per_class,))

            z_list.append(z_samples)
            y_list.append(y_samples)

        z_domain = torch.cat(z_list)
        y_domain = torch.cat(y_list)
        all_domains.append(DomainDataset(z_domain, y_domain, weights))

    return all_domains


def generate_gauss_domains(
    source_dataset,
    target_dataset,
    n_wsteps=20,
    samples_per_class=1000,
    batch_size=512,
    device="cuda",
    step_size=0.05,
    stop_threshold=0.1,
):
    """
    Generates intermediate domains by interpolating from source to target distribution
    using a Wasserstein-based approach. Returns datasets similar to `generate_domains()`.
    """
    print("------------Generate Intermediate domains----------")
    # ---------------------------
    #  (1) Estimate class-conditional Gaussians
    # ---------------------------
    # print("Estimating class-conditional Gaussians for SOURCE...")
    source_class_params = estimate_class_conditional_gaussian_params(
        source_dataset,
        cache_path="class_conditional_source.pkl",
        force_recompute=True
    )

    
    target_class_params = estimate_class_conditional_gaussian_params(
        target_dataset,
        cache_path="class_conditional_target.pkl",
        force_recompute=True
    )

    # ---------------------------
    #  (2) Generate Intermediate Domains
    # ---------------------------
    all_domains = []
    unique_classes = sorted(source_class_params.keys())

    progress_bar = tqdm(range(n_wsteps), desc="Generating Intermediate Domains", unit="step")
    step_idx = 1
    avg_target = float("inf")
    # Initialize progress bar
    # print(f"🔹 Step 3: Beginning interpolation over {len(unique_classes)} classes for {n_wsteps} steps...")
    progress_bar = tqdm(range(1, n_wsteps + 1), desc="Generating Intermediate Domains", unit="step")
    for step_idx in progress_bar:
        # print(f"🔹 Step 4: Processing step {step_idx+1}/{n_wsteps}...")
        z_new_list = []
        y_new_list = []
        weights_list = []
        all_w2_target = []

        for c in unique_classes:
            mu_c, Sigma_c = source_class_params[c]
            mu_c_T, Sigma_c_T = target_class_params[c]
            # updated_mu, _, w2_target = wasserstein_step_mean_only(
            #     mu_t=mu_c, mu_T=mu_c_T,
            #     step_size=step_size,
            #     max_iters=50,
            #     to_cpu=False
            # )
            updated_mu, updated_Sigma_c, w2_target = wasserstein_step_full(
                mu_t=mu_c, sigma_t=Sigma_c,
                mu_T=mu_c_T, sigma_T=Sigma_c_T,
                step_size=step_size,
                lr=0.01,
                max_iters=50
            )

            # Update source-class distribution
            # print(f"   ✅ Wasserstein step complete. W2 distance: {w2_target:.4f}")

            source_class_params[c] = (updated_mu, updated_Sigma_c)
            all_w2_target.append(w2_target)
            updated_mu = updated_mu.reshape(-1)
            eps = 1e-8
            # breakpoint()
            # Sigma_c += eps * torch.eye(Sigma_c.shape[0], device=Sigma_c.device)
            # L = torch.linalg.cholesky(Sigma_c)
            # eps = torch.randn(samples_per_class, updated_mu.shape[0], device=updated_mu.device)
            # z_samples = (updated_mu + eps @ L.T).cpu()
            var_vector = torch.diag(updated_Sigma_c)  # same as above, works for square matrices

            std_vector = torch.sqrt(var_vector + eps)  # Add small epsilon for stability
            std_matrix = torch.diag(std_vector)  # Diagonal matrix for sampling
            # Instead of using diagonal std_vector sampling:
            # L = torch.linalg.cholesky(Sigma_c + eps * torch.eye(Sigma_c.shape[0], device=Sigma_c.device))


            epsilon = torch.randn(samples_per_class, updated_mu.shape[0], device=updated_mu.device)

            z_samples = (updated_mu + epsilon * std_vector).cpu()
            # z_samples = (updated_mu + epsilon @ updated_Sigma_c).cpu()
            # z_samples = updated_mu + epsilon @ L.T



            y_samples = np.full((samples_per_class,), c, dtype=int)  # Assign class labels
            weights = np.ones((samples_per_class,))  # Equal weights for now

            # z_samples = torch.tensor(z_samples, dtype=torch.float32, device=device)
            y_samples = torch.tensor(y_samples, dtype=torch.long,)

            z_new_list.append(z_samples)
            y_new_list.append(y_samples)
            weights_list.append(weights)
        # print(f"✅ Step {step_idx+1} completed. Aggregating data...")

        # Combine generated latents
        z_intermediate = torch.cat(z_new_list, dim=0)
        y_intermediate = torch.cat(y_new_list, dim=0)
        # weights_intermediate = torch.cat(weights_list, dim=0)
        weights_intermediate = np.concatenate(weights_list, axis=0)

        # Create `DomainDataset` for consistency with `generate_domains()`
        domain_dataset = DomainDataset(z_intermediate, weights_intermediate)
        domain_dataset.targets = y_intermediate
        all_domains.append(domain_dataset)  # Store for later use

        # Compute average Wasserstein distance to target
        avg_target = np.mean([x for x in all_w2_target])
        # print(f"   🔹 Average Wasserstein distance to target: {avg_target:.4f}")


        # ✅ Update the progress bar with W2 distance info
        # if avg_target < stop_threshold:
        #     print(f"🛑 Stopping early at step {step_idx+1} due to low W2 distance.")
        #     break

    progress_bar.close()
    # breakpoint()
    # Append final target dataset for consistency with `generate_domains()`
    all_domains.append(target_dataset)
    print(f"Total data for each intermediate domain: {len(z_intermediate)}")
    return avg_target, all_domains  # Returns a list of `DomainDataset` objects


# def estimate_class_conditional_gaussian_params(
#     encoded_dataset,  # Assumes (latents, labels)
#     cache_path=None,
#     force_recompute=False,
# ):
#     """
#     Computes one (mu, sigma) per class from a dataset of encoded latents.
    
#     Inputs:
#         - encoded_dataset: Tuple (latents, labels), where:
#             * latents: Tensor of shape (N, d) containing latent vectors
#             * labels: Tensor of shape (N,) containing class labels
    
#     Output:
#         - class_params: Dict[class_label -> (mu, Sigma)] (Gaussian parameters per class)
#     """

#     if cache_path is None:
#         cache_path = "class_conditional_gaussians.pkl"

#     # ✅ Load cached results if available
#     if os.path.exists(cache_path) and not force_recompute:
#         print(f"🔹 Loading class-conditional Gaussians from {cache_path}")
#         with open(cache_path, "rb") as f:
#             return pickle.load(f)

#     # Unpack encoded dataset
#     # breakpoint()
#     latents, labels = encoded_dataset.data, encoded_dataset.targets  # latents: (N, d), labels: (N,)
#     unique_classes = torch.unique(labels)

#     class_params = {}

#     # Compute class-conditional mean & covariance
#     for c in unique_classes:
#         latents_c = latents[labels == c]  # All latents for class c

#         if latents_c.shape[0] > 1:
#             mu_c = latents_c.mean(dim=0)  # shape: (d,)
#             centered = latents_c - mu_c   # shape: (N, d)
#             centered_flat = centered.view(centered.shape[0], -1)  # Shape [N, D]
#             # breakpoint()
#             sigma_c = (centered_flat.T @ centered_flat) / (centered_flat.shape[0] - 1)
#             # sigma_c = (centered.T @ centered) / (latents_c.shape[0] - 1)  # shape: (d, d)
#         else:
#             mu_c = latents_c.mean(dim=0)
#             sigma_c = torch.eye(latents_c.shape[1], device=latents_c.device, dtype=latents_c.dtype) * 1e-5  # Small regularization
        
#         class_params[c.item()] = (mu_c, sigma_c)

#     # ✅ Cache results
#     with open(cache_path, "wb") as f:
#         pickle.dump(class_params, f)
#     print(f"✅ Cached class-conditional Gaussians at {cache_path}")

#     return class_params


def estimate_class_conditional_gaussian_params(
    encoded_dataset,  # Assumes (latents, labels)
    cache_path=None,
    force_recompute=False,
    use_diagonal_cov=False  # 🔸 Add flag to control diagonal vs full covariance
):
    """
    Computes one (mu, sigma) per class from a dataset of encoded latents.

    If `use_diagonal_cov=True`, uses a diagonal approximation for covariance.
    
    Inputs:
        - encoded_dataset: Tuple (latents, labels), where:
            * latents: Tensor of shape (N, d)
            * labels: Tensor of shape (N,)
    
    Output:
        - class_params: Dict[class_label -> (mu, sigma)]
            * mu: mean vector (d,)
            * sigma: diagonal var vector (d,) if use_diagonal_cov else full matrix (d, d)
    """
    if cache_path is None:
        cache_path = "class_conditional_gaussians_diag.pkl" if use_diagonal_cov else "class_conditional_gaussians.pkl"

    # if os.path.exists(cache_path) and not force_recompute:
    #     print(f"🔹 Loading class-conditional Gaussians from {cache_path}")
    #     with open(cache_path, "rb") as f:
    #         return pickle.load(f)

    latents, labels = encoded_dataset.data, encoded_dataset.targets
    latents = latents.view(latents.size(0), -1) 
    unique_classes = torch.unique(labels)
    class_params = {}
    for c in unique_classes:
        latents_c = latents[labels == c]  # (Nc, d)
        mu_c = latents_c.mean(dim=0)  # (d,)
        if latents_c.shape[0] > 1:
            if use_diagonal_cov:
                var_vec = latents_c.var(dim=0, unbiased=True)
                sigma_c = torch.diag(var_vec)  # Make it (d, d)
            else:
                centered = (latents_c - mu_c)
                sigma_c = (centered.T @ centered) / (latents_c.shape[0] - 1)  # (d, d)
                # sigma_c += 1e-4 * torch.eye(sigma_c.shape[0], device=sigma_c.device)  # 🔧 Ensure PSD
        else:
            if use_diagonal_cov:
                sigma_c = torch.ones_like(mu_c) * 1e-5
            else:
                sigma_c = torch.eye(latents_c.shape[1], device=latents_c.device) * 1e-5

        class_params[c.item()] = (mu_c, sigma_c)

    with open(cache_path, "wb") as f:
        pickle.dump(class_params, f)
    print(f"✅ Cached class-conditional Gaussians at {cache_path}")

    return class_params


def to_cpu_numpy(x):
    """
    Ensures x is a CPU NumPy array.
      - If x is already a NumPy ndarray, return x as-is.
      - If x is a torch.Tensor (CPU or GPU), convert to CPU and then to NumPy.
      - Otherwise, raise a TypeError.
    """
    if isinstance(x, np.ndarray):
        # It's already a NumPy array (on CPU by definition), so do nothing
        return x
    elif isinstance(x, torch.Tensor):
        # Convert Torch tensor => CPU => NumPy
        return x.cpu().numpy()
    else:
        raise TypeError(
            f"Cannot convert type {type(x).__name__} to a CPU NumPy array. "
            "Expected np.ndarray or torch.Tensor."
        )
    

def make_positive_definite(Sigma, min_eig=1e-5):
    """
    Ensures that Sigma is positive semi-definite by clamping eigenvalues.
    Moves Tensor to CPU before using NumPy operations.
    """
    if isinstance(Sigma, torch.Tensor):
        Sigma = Sigma.detach().cpu().numpy()  # Ensure NumPy format
    
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    eigvals = np.clip(eigvals, min_eig, None)  # Clamp negative eigenvalues
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


# def wasserstein_step_full(
#     mu_t, sigma_t,
#     mu_T, sigma_T,
#     delta=0.5,       # Maximum allowed W2 distance from old dist
#     lr=0.01,
#     max_iters=100,
#     lambda_cov=0.1
# ):
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     # Move inputs to GPU
#     mu_t = mu_t.to(dtype=torch.float16, device=device)
#     sigma_t = sigma_t.to(dtype=torch.float16, device=device)
#     mu_T = mu_T.to(dtype=torch.float16, device=device)
#     sigma_T = sigma_T.to(dtype=torch.float16, device=device)

#     # Cloning for gradient optimization
#     v = mu_t.clone().detach().requires_grad_(True)
#     sigma_v = sigma_t.clone().detach().requires_grad_(True)

#     optimizer = torch.optim.Adam([v, sigma_v], lr=lr)
#     eps = 1e-8

#     # Iterative Wasserstein step
#     for step in range(max_iters):
#         optimizer.zero_grad()
#         loss = w2_gaussian_dist(v, sigma_v, mu_T, sigma_T)
#         loss.backward()
#         optimizer.step()

#     # Ensure positive definiteness
#     with torch.no_grad():
#         sigma_v_pd = sigma_v.detach() + eps * torch.eye(sigma_v.shape[0], device=device)
#         sigma_v_pd = make_positive_definite(sigma_v_pd)
#         final_mu = v.detach()
#         final_sigma = sigma_v_pd

#     # Compute Wasserstein distances
#     w2_previous_final = w2_gaussian_dist(mu_t, sigma_t, final_mu, final_sigma)
#     w2_target_final = w2_gaussian_dist(final_mu, final_sigma, mu_T, sigma_T)

#     # Enforce W2 constraint
    # if w2_previous_final.item() > delta:
    #     alpha = min(1.0, delta / w2_previous_final.item())
    #     final_mu_np, final_sigma_np = wasserstein_interpolation(
    #         mu_t.cpu().numpy(), sigma_t.cpu().numpy(),  
    #         final_mu.cpu().numpy(), final_sigma.cpu().numpy(),  
    #         alpha
    #     )

#         final_mu = torch.tensor(final_mu_np, dtype=torch.float32, device=device)
#         final_sigma = torch.tensor(final_sigma_np, dtype=torch.float32, device=device)

#         # Recompute Wasserstein distances after projection
#         w2_previous_final = w2_gaussian_dist(mu_t, sigma_t, final_mu, final_sigma)
#         w2_target_final = w2_gaussian_dist(final_mu, final_sigma, mu_T, sigma_T)

#     

#     # Free GPU memory if tensors are not needed on GPU
#     mu_t, sigma_t, mu_T, sigma_T = mu_t.cpu(), sigma_t.cpu(), mu_T.cpu(), sigma_T.cpu()
#     final_sigma = final_sigma.cpu()

#     return final_mu, final_sigma, w2_previous_final, w2_target_final

import torch

def w2_shorten(muA, SigmaA, muB, SigmaB, alpha, device="cuda"):
    """
    Computes the alpha-Bures interpolation between:
       N(muA, SigmaA) and N(muB, SigmaB).
    
    Uses PyTorch operations **entirely** on GPU for efficiency.

    Returns:
      mu_out, Sigma_out: Interpolated mean and covariance as Torch tensors.
    """

    # Ensure tensors are on correct device
    muA, SigmaA, muB, SigmaB = map(lambda x: x.to(device, dtype=torch.float32), [muA, SigmaA, muB, SigmaB])

    # Compute square root of SigmaA using Cholesky decomposition (√A ≈ L)
    L = torch.linalg.cholesky(SigmaA)  # L @ L.T ≈ SigmaA

    # Solve for inverse sqrt of SigmaA: inv(sqrtA) ≈ torch.linalg.solve()
    invSqrtA = torch.linalg.solve(L, torch.eye(L.shape[0], device=device))

    # Compute M = inv(sqrtA) * SigmaB * inv(sqrtA)
    M = invSqrtA @ SigmaB @ invSqrtA.T

    # Compute sqrtM using SVD (approximation of sqrtm in PyTorch)
    U, S, Vh = torch.linalg.svd(M)
    sqrtM = U @ torch.diag(torch.sqrt(S)) @ Vh

    # Form the convex combination in "root space"
    part = (1 - alpha) * torch.eye(L.shape[0], device=device) + alpha * sqrtM

    # Reconstruct interpolated covariance
    Sigma_out = L @ part @ part @ L.T
    mu_out = (1 - alpha) * muA + alpha * muB  # Linear interpolation of means

    return mu_out, Sigma_out


# def wasserstein_step_full(
#     mu_t, sigma_t, mu_T, sigma_T,
#     delta=0.5, lr=0.01, max_iters=100, lambda_cov=0.1, to_cpu=False
# ):
#     """
#     Optimized Wasserstein step function for interpolating Gaussian distributions.
#     Uses in-place operations, low-rank updates, and efficient matrix computations.
#     Includes a progress bar for tracking optimization progress.
#     """
#     eps = 1e-8  # Numerical stability constant

#     # Move inputs to device efficiently
#     dtype = torch.float32
#     mu_t, sigma_t, mu_T, sigma_T = map(lambda x: x.to(device, dtype=dtype, non_blocking=True),
#                                        [mu_t, sigma_t, mu_T, sigma_T])

#     # Initialize optimization variables
#     v = mu_t.clone().detach().requires_grad_(True)
#     sigma_v = sigma_t.clone().detach().requires_grad_(True)

#     optimizer = torch.optim.Adam([v, sigma_v], lr=lr)

#     # Initialize tqdm progress bar
#     progress_bar = tqdm(range(max_iters), desc="Wasserstein Step", unit="iter")

#     # Iterative Wasserstein step
#     for step in progress_bar:
#         optimizer.zero_grad()
#         loss = w2_gaussian_implicit(v, sigma_v, mu_T, sigma_T)

#         if loss.item() < delta:  # Early stopping if close enough
#             progress_bar.set_description(f"Converged at step {step+1}")
#             break

#         loss.backward()
#         optimizer.step()

#         # Update progress bar with current loss
#         progress_bar.set_postfix(loss=loss.item())

#     progress_bar.close()

#     # Ensure positive definiteness
#     with torch.no_grad():
#         # Add a small identity matrix for numerical stability
#         sigma_v_pd = sigma_v + eps * torch.eye(sigma_v.shape[0], device=device, dtype=dtype)
#         final_mu, final_sigma = v, sigma_v_pd

#     # Compute Wasserstein distances
#     w2_prev = w2_gaussian_implicit(mu_t, sigma_t, final_mu, final_sigma)

#     # Enforce W2 constraint if needed
#     if w2_prev.item() > delta:
#         alpha = min(1.0, delta / w2_prev.item())
#         final_mu, final_sigma = w2_shorten(final_mu, final_sigma, mu_T, sigma_T, alpha)

#     # Move to CPU if required
#     if to_cpu:
#         final_mu, final_sigma = final_mu.cpu(), final_sigma.cpu()

#     return final_mu, final_sigma, w2_prev

import torch
import time

def w2_gaussian_low_rank(mu1, L1, mu2, L2):
    """
    Computes the 2-Wasserstein distance between two Gaussians 
    with low-rank covariance representation.

    Args:
        mu1: (d,) Mean of Gaussian 1
        L1: (d, r) Low-rank factor of Sigma1 (Sigma1 = L1 @ L1.T)
        mu2: (d,) Mean of Gaussian 2
        L2: (d, r) Low-rank factor of Sigma2 (Sigma2 = L2 @ L2.T)

    Returns:
        Wasserstein distance (scalar)
    """
    print("🔹 Step 1: Computing mean squared difference...")
    start_time = time.time()
    mean_diff_sq = torch.norm(mu1 - mu2, p=2) ** 2
    print(f"✅ Mean squared difference computed in {time.time() - start_time:.4f} sec")

    print("🔹 Step 2: Computing trace of Sigma1 and Sigma2...")
    start_time = time.time()
    trace_term_1 = torch.sum(L1**2)  # Equivalent to tr(Sigma1)
    trace_term_2 = torch.sum(L2**2)  # Equivalent to tr(Sigma2)
    print(f"✅ Trace terms computed in {time.time() - start_time:.4f} sec")

    print("🔹 Step 3: Computing cross term M = L1^T @ L2...")
    start_time = time.time()
    M = L1.T @ L2  # (r, r) small matrix
    print(f"✅ Cross term computed in {time.time() - start_time:.4f} sec. Shape: {M.shape}")

    print("🔹 Step 4: Computing SVD of M for sqrtm...")
    start_time = time.time()
    U, S, Vh = torch.linalg.svd(M)  # Compute sqrtm(M @ M^T)
    sqrt_M = U @ torch.diag(S.sqrt()) @ Vh
    print(f"✅ SVD computed in {time.time() - start_time:.4f} sec")

    print("🔹 Step 5: Computing trace of sqrt(Sigma1 Sigma2)...")
    start_time = time.time()
    trace_sqrt = torch.sum(sqrt_M**2)  # Equivalent to tr(sqrt(Sigma1 Sigma2))
    print(f"✅ Trace of sqrt computed in {time.time() - start_time:.4f} sec")

    print("🔹 Step 6: Computing final Wasserstein distance...")
    start_time = time.time()
    w2_sq = mean_diff_sq + trace_term_1 + trace_term_2 - 2 * trace_sqrt
    w2_distance = torch.sqrt(torch.clamp(w2_sq, min=0.0))  # Avoid numerical issues
    print(f"✅ Wasserstein distance computed in {time.time() - start_time:.4f} sec. Final W2: {w2_distance.item():.6f}")

    return w2_distance

def wasserstein_step_mean_only(
    mu_t, mu_T,
    step_size=0.05,
    lr=0.1,
    max_iters=100,
    to_cpu=False
):
    """
    Wasserstein step between Gaussians by updating the mean only,
    constrained to move no more than `step_size` from mu_t.

    Args:
        mu_t: (d,) current mean
        mu_T: (d,) target mean
        step_size: max distance from mu_t in L2 (sqrt of W2 step)
        lr: learning rate for optimization
        max_iters: number of gradient steps
        to_cpu: whether to return result on CPU

    Returns:
        updated_mu: (d,) updated mean vector
        w2_before: squared distance before step
        w2_after: squared distance after step
    """
    mu_t = mu_t.detach()
    mu_T = mu_T.detach()
    
    v = mu_t.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([v], lr=lr)

    def w2_loss(mu1, mu2):
        return torch.norm((mu1 - mu2) ** 2)

    w2_before = w2_loss(mu_t, mu_T).item()

    for _ in range(max_iters):
        optimizer.zero_grad()
        loss = w2_loss(v, mu_T)
        loss.backward()
        optimizer.step()

        # Project back to step_size ball around mu_t
        with torch.no_grad():
            delta = v - mu_t
            norm = torch.norm(delta)
            if norm > step_size:
                v.copy_(mu_t + delta / norm * step_size)

    updated_mu = v.detach()
    w2_after = w2_loss(updated_mu, mu_T).item()

    if to_cpu:
        updated_mu = updated_mu.cpu()

    return updated_mu, w2_before, w2_after

def wasserstein_step_full(
    mu_t, sigma_t, mu_T, sigma_T,
    step_size=0.5, lr=0.01, max_iters=100, lambda_cov=0.1, to_cpu=False,
    diagonal=True
):
    """
    Wasserstein step between Gaussians, supporting full or diagonal covariance.

    Args:
        mu_t: (d,) mean vector of source Gaussian
        sigma_t: (d,) or (d,d) covariance of source
        mu_T: (d,) mean vector of target Gaussian
        sigma_T: (d,) or (d,d) covariance of target
        delta: W2 constraint radius
        lr: learning rate
        max_iters: number of optimization iterations
        lambda_cov: regularization strength
        to_cpu: move outputs to CPU
        diagonal: whether covariances are diagonal

    Returns:
        updated_mu: (d,) updated mean
        updated_sigma: (d,) or (d,d) updated covariance
        w2_before: initial W2 distance
        w2_after: W2 distance after update
    """


    eps = 1e-10
    device = mu_t.device

    # Initialize trainable mean and log-variance
    v = mu_t.clone().detach().requires_grad_(True)
    var_init = torch.diagonal(sigma_t).clamp(min=eps)
    log_var_t = torch.log(var_init).detach().requires_grad_(True)

    var_T = torch.diagonal(sigma_T).clamp(min=eps)

    optimizer = torch.optim.Adam([v, log_var_t], lr=lr)

    
    def w2_mean_only(mu1, mu2):
        return torch.sum((mu1 - mu2)**2)

    def w2_gaussian_diag(mu1, var1, mu2, var2):
        """
        Compute squared Wasserstein-2 distance between Gaussians with diagonal covariances,
        where Sigma1 and Sigma2 are full (d,d) diagonal matrices.
        
        Args:
            mu1, mu2: (d,) mean vectors
            Sigma1, Sigma2: (d,d) diagonal covariance matrices (only diagonal is used)
        
        Returns:
            Scalar tensor: squared W₂ distance
        """
        mean_term = torch.sum((mu1 - mu2) ** 2)
        sqrt_diff = torch.sqrt(var1) - torch.sqrt(var2)
        cov_term = torch.sum(sqrt_diff ** 2)
        # breakpoint()
        return mean_term + cov_term

    
    def w2_full(mu1, sigma1, mu2, sigma2, eps=1e-4):
        """
        Compute squared 2-Wasserstein distance between N(mu1, sigma1) and N(mu2, sigma2),
        using Cholesky and safe fallback.

        Args:
            mu1, mu2: (d,) mean vectors
            sigma1, sigma2: (d, d) covariance matrices
        Returns:
            Scalar torch.Tensor: squared Wasserstein distance
        """
        device = mu1.device
        mean_term = torch.sum((mu1 - mu2) ** 2)

        # Safely compute sqrt of sigma2
        sqrt_sigma2 = safe_cholesky(sigma2 + eps * torch.eye(sigma2.shape[0], device=device), eps=eps)

        # Compute (sqrt_sigma2 * sigma1 * sqrt_sigma2)^1/2
        sigma_prod = sqrt_sigma2 @ sigma1 @ sqrt_sigma2.T
        # Use eigendecomposition for sqrt to avoid repeated cholesky failures
        sigma_prod_sqrt = matrix_sqrt_torch(sigma_prod)

        trace_term = torch.trace(sigma1 + sigma2 - 2 * sigma_prod_sqrt)
        return mean_term + trace_term

    for _ in range(max_iters):
        optimizer.zero_grad()
        s = torch.exp(log_var_t)  # ensure positivity
        kl_reg = lambda_cov * torch.sum(torch.log(s / var_T) + (var_T / s) - 1)

        loss = w2_gaussian_diag(v, s, mu_T, var_T) + kl_reg
        loss.backward()
        optimizer.step()
        # if _ % 10 == 0:
        #     print(f"Step {_}: Loss (W2 to Target) = {loss.item():.4f}")
    # print(f"(W2 to Target) = {loss.item():.4f}")
    updated_mu = v.detach()
    updated_var = torch.exp(log_var_t.detach()).clamp(min=eps)
    updated_sigma = torch.diag(updated_var)
    # make sure updated_sigma is positive definite
    assert updated_mu.shape == mu_t.shape
    assert updated_sigma.shape == sigma_t.shape


    if diagonal:
        w2_before = w2_gaussian_diag(mu_t, var_init, mu_T, var_T)
        w2_after = w2_gaussian_diag(updated_mu, updated_var, mu_T, var_T)
    else:
        w2_before = w2_full(mu_t, sigma_t, mu_T, sigma_T)
        w2_after = w2_full(updated_mu, updated_sigma, mu_T, sigma_T)

    if to_cpu:
        updated_mu = updated_mu.cpu()
        updated_sigma = updated_sigma.cpu()

    # breakpoint()
    return updated_mu, updated_sigma, w2_after


from tqdm import trange
def wasserstein_step_diag(
    mu_t, sigma_t, mu_T, sigma_T,
    delta=0.5, lr=0.01, max_iters=100, to_cpu=False
):
    device = mu_t.device
    eps = 1e-10

    v = mu_t.clone().detach().requires_grad_(True)
    s = sigma_t.clone().detach().requires_grad_(True)  # (d,) diagonal entries

    optimizer = torch.optim.Adam([v, s], lr=lr)

    def w2_diag(mu1, sigma1, mu2, sigma2):
        sigma1_clamped = torch.clamp(sigma1, min=1e-6)
        sigma2_clamped = torch.clamp(sigma2, min=1e-6)
        return torch.sum((mu1 - mu2) ** 2) + torch.sum((torch.sqrt(sigma1_clamped) - torch.sqrt(sigma2_clamped)) ** 2)

    pbar = trange(max_iters, desc="Optimizing W2 (diag)", leave=False)
    for step in pbar:
        optimizer.zero_grad()
        loss = w2_diag(v, s, mu_T, sigma_T)
        loss.backward()
        optimizer.step()
        pbar.set_description(f"Step {step+1}/{max_iters} | Loss: {loss.item():.4f}")



    updated_mu = v.detach()
    updated_sigma = s.detach()

    w2_before = w2_diag(mu_t, sigma_t, mu_T, sigma_T)
    w2_after = w2_diag(updated_mu, updated_sigma, mu_T, sigma_T)

    if to_cpu:
        updated_mu = updated_mu.cpu()
        updated_sigma = updated_sigma.cpu()

    return updated_mu, updated_sigma, w2_before, w2_after


def matrix_sqrt_torch(A):
    """Compute matrix square root using SVD instead of Cholesky for memory efficiency."""
    U, S, Vh = torch.linalg.svd(A)  # SVD decomposition
    return U @ torch.diag(torch.sqrt(S)) @ Vh


def matrix_sqrt_eigh(A, eps=1e-5):
    """Eigen decomposition method (default)."""
    A_sym = (A + A.T) / 2 + eps * torch.eye(A.shape[0], device=A.device)
    eigvals, eigvecs = torch.linalg.eigh(A_sym)
    sqrt_eigvals = torch.sqrt(torch.clamp(eigvals, min=eps))
    return eigvecs @ torch.diag(sqrt_eigvals) @ eigvecs.T

def matrix_sqrt_cholesky(A, eps=1e-5):
    """Cholesky decomposition fallback."""
    A_sym = (A + A.T) / 2 + eps * torch.eye(A.shape[0], device=A.device)
    L = torch.linalg.cholesky(A_sym)
    return L.T

# def w2_gaussian_dist(mu1, Sigma1, mu2, Sigma2):
#     """
#     2-Wasserstein distance between Gaussians N(mu1, Sigma1) and N(mu2, Sigma2),
#     all in PyTorch.
    
#     Args:
#       mu1, mu2: (d,) torch Tensors
#       Sigma1, Sigma2: (d,d) torch Tensors (symmetric PSD)
#     Returns:
#       A torch.Tensor scalar representing the W2 distance (not squared).
#     """
#     # 1) Mean part
#     mean_diff_sq = torch.norm(mu1 - mu2)**2
    
#     # 2) Covariance part
#     sqrt_Sigma1 = matrix_sqrt_torch(Sigma1)
#     inside = sqrt_Sigma1 @ Sigma2 @ sqrt_Sigma1
#     sqrt_inside = matrix_sqrt_torch(inside)  # = (Sigma1^(1/2) Sigma2 Sigma1^(1/2))^(1/2)
#     breakpoint()
#     cov_term = torch.trace(Sigma1 + Sigma2 - 2.0 * sqrt_inside)
    
#     # 3) Combine
#     w2_sq = mean_diff_sq + cov_term
#     # Clamp to avoid tiny negative due to numerical noise
#     w2_sq = torch.clamp(w2_sq, min=0.0)
    
#     return torch.sqrt(w2_sq)  # final distance


# def w2_gaussian_dist(mu1, Sigma1, mu2, Sigma2, batch_size=512):
#     """
#     Compute the 2-Wasserstein distance between Gaussians N(mu1, Sigma1) and N(mu2, Sigma2)
#     using batch processing to reduce memory usage.
    
#     Args:
#       mu1, mu2: (d,) torch Tensors
#       Sigma1, Sigma2: (d,d) torch Tensors
#       batch_size: Number of rows/cols processed per batch (default: 512)
    
#     Returns:
#       A torch.Tensor scalar representing the W2 distance.
#     """
#     d = Sigma1.shape[0]  # Dimension of covariance matrices

#     # 1) Mean difference squared
#     mean_diff_sq = torch.norm(mu1 - mu2) ** 2

#     # 2) Batched computation for covariance term
#     sqrt_Sigma1 = matrix_sqrt_torch(Sigma1)
#     w2_cov_term = 0.0  # Initialize sum
#     breakpoint()
#     for i in range(0, d, batch_size):
#         batch_end = min(i + batch_size, d)
#         Sigma1_batch = sqrt_Sigma1[i:batch_end, :]
#         Sigma2_batch = Sigma2[:, i:batch_end]
    
#         inside_batch = Sigma1_batch @ Sigma2_batch @ Sigma1_batch.T  # (batch_size, batch_size)
#         sqrt_inside_batch = matrix_sqrt_torch(inside_batch)  # Compute sqrt for this batch
#         w2_cov_term += torch.trace(Sigma1_batch + Sigma2_batch - 2.0 * sqrt_inside_batch)

#     # 3) Final Wasserstein distance
#     w2_sq = mean_diff_sq + w2_cov_term
#     w2_sq = torch.clamp(w2_sq, min=0.0)  # Avoid numerical issues

#     return torch.sqrt(w2_sq)


def optimized_matrix_sqrt(A, method="cholesky", eps=1e-5):
    device = A.device
    # A = A + eps * torch.eye(A.shape[0], device=device)  # Stabilize
    A = A.clone()  # Clone only if required (to avoid modifying the input tensor)

    A.diagonal().add_(eps)  # Modifies A in-place
    # breakpoint()
    try:
        if method == "cholesky":
            L = torch.linalg.cholesky(A)
            return L.T
        if method == "eigh":
            eigvals, eigvecs = torch.linalg.eigh(A)
            sqrt_eigvals = torch.clamp(eigvals, min=eps).sqrt()
            return eigvecs @ torch.diag(sqrt_eigvals) @ eigvecs.T
    except RuntimeError:
        # breakpoint()
        print(f"⚠ Cholesky failed, using Eigh.")
        eigvals, eigvecs = torch.linalg.eigh(A)
        sqrt_eigvals = torch.clamp(eigvals, min=eps).sqrt()
        return eigvecs @ torch.diag(sqrt_eigvals) @ eigvecs.T


def w2_gaussian_dist(mu1, Sigma1, mu2, Sigma2, batch_size=512):
    """
    Compute the 2-Wasserstein distance between Gaussians N(mu1, Sigma1) and N(mu2, Sigma2)
    using optimized batch processing.
    """

    d = Sigma1.shape[0]  # Dimension of covariance matrices

    # print("🔹 Step 1: Computing mean squared difference...")
    # start_time = time.time()
    mean_diff_sq = torch.norm(mu1 - mu2, p=2) ** 2
    # print(f"✅ Mean squared difference computed in {time.time() - start_time:.4f} sec")

    print("🔹 Step 2: Computing sqrt(Sigma1)...")
    start_time = time.time()
    sqrt_Sigma1 = optimized_matrix_sqrt(Sigma1)
    print(f"✅ sqrt(Sigma1) computed in {time.time() - start_time:.4f} sec")

    print("🔹 Step 3: Computing inside term: sqrt(Sigma1) @ Sigma2 @ sqrt(Sigma1.T)...")
    start_time = time.time()
    inside = sqrt_Sigma1 @ Sigma2 @ sqrt_Sigma1.T
    print(f"✅ Inside term computed in {time.time() - start_time:.4f} sec")

    print("🔹 Step 4: Computing sqrt(inside)...")
    start_time = time.time()
    sqrt_inside = optimized_matrix_sqrt(inside)
    
    print(f"✅ sqrt(inside) computed in {time.time() - start_time:.4f} sec")

    print("🔹 Step 5: Computing Wasserstein covariance term...")
    start_time = time.time()
    w2_cov_term = torch.trace(Sigma1) + torch.trace(Sigma2) - 2 * torch.trace(sqrt_inside)
    print(f"✅ Wasserstein covariance term computed in {time.time() - start_time:.4f} sec")

    print("🔹 Step 6: Computing final Wasserstein distance...")
    start_time = time.time()
    w2_sq = mean_diff_sq + torch.clamp(w2_cov_term, min=0.0)  # Clamp to prevent negative values
    w2_distance = torch.sqrt(w2_sq)
    print(f"✅ Final Wasserstein distance computed in {time.time() - start_time:.4f} sec")

    return w2_distance


import torch

def w2_gaussian_implicit(mu1, cov1_func, mu2, cov2_func, k=10, eps=1e-5):
    """
    Compute the W2 distance between two Gaussians without explicitly storing covariance matrices.

    Args:
      - mu1, mu2: Mean vectors (d,)
      - cov1_func: Function that computes cov1 @ v (matrix-free)
      - cov2_func: Function that computes cov2 @ v (matrix-free)
      - k: Number of Hutchinson trace samples (default: 10)
      - eps: Numerical stability term
    
    Returns:
      - W2 distance between N(mu1, cov1) and N(mu2, cov2)
    """
    device = mu1.device
    d = mu1.shape[0]

    # Step 1: Mean Difference Squared
    mean_diff_sq = torch.norm(mu1 - mu2, p=2) ** 2

    # Step 2: Trace Terms (Diagonal Approximation)
    trace_cov1 = torch.sum(torch.diag(cov1_func(torch.eye(d, device=device))))
    trace_cov2 = torch.sum(torch.diag(cov2_func(torch.eye(d, device=device))))

    # Step 3: Hutchinson's Estimator for trace(sqrt(Sigma1 @ Sigma2))
    trace_sqrt = 0.0
    for _ in range(k):
        v = torch.randn(d, device=device)  # Random Gaussian vector
        v1 = cov1_func(v)  # Approximate sqrt(Sigma1) @ v
        v2 = cov2_func(v1)  # Approximate Sigma2 @ v1
        v_final = cov1_func(v2)  # Approximate sqrt(Sigma1) @ v2
        trace_sqrt += v @ v_final  # Hutchinson's estimate

    trace_sqrt /= k  # Average over samples

    # Compute final Wasserstein-2 distance
    w2_sq = mean_diff_sq + trace_cov1 + trace_cov2 - 2 * trace_sqrt
    return torch.sqrt(torch.clamp(w2_sq, min=0.0))  # Ensure non-negative

