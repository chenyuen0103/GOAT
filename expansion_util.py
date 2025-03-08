import torch
from util import *
import numpy as np
import time
import torch.nn as nn
from tqdm import tqdm
import pickle
import os
device = "cuda" if torch.cuda.is_available() else "cpu"

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

def generate_gauss_domains(
    source_dataset,
    target_dataset,
    n_wsteps=20,
    samples_per_class=5000,
    batch_size=512,
    device="cuda",
    stop_threshold=1
):
    """
    Generates intermediate domains by interpolating from source to target distribution
    using a Wasserstein-based approach. Returns datasets similar to `generate_domains()`.
    """

    # ---------------------------
    #  (1) Estimate class-conditional Gaussians
    # ---------------------------
    print("Estimating class-conditional Gaussians for SOURCE...")
    source_class_params = estimate_class_conditional_gaussian_params(
        source_dataset,
        cache_path="class_conditional_source.pkl",
        force_recompute=False
    )

    target_class_params = estimate_class_conditional_gaussian_params(
        target_dataset,
        cache_path="class_conditional_target.pkl",
        force_recompute=False
    )

    # ---------------------------
    #  (2) Generate Intermediate Domains
    # ---------------------------
    all_domains = []
    unique_classes = sorted(source_class_params.keys())

    progress_bar = tqdm(range(n_wsteps), desc="Generating Intermediate Domains", unit="step")
    step_idx = 0
    avg_target = float("inf")

    while step_idx < n_wsteps and avg_target > stop_threshold:
        z_new_list = []
        y_new_list = []
        weights_list = []
        all_w2_target = []

        for c in unique_classes:
            mu_c, Sigma_c = source_class_params[c]
            mu_c_T, Sigma_c_T = target_class_params[c]

            # Wasserstein step towards target
            updated_mu, updated_Sigma, _, w2_target = wasserstein_step_full(
                mu_t=mu_c, sigma_t=Sigma_c,
                mu_T=mu_c_T, sigma_T=Sigma_c_T,
                delta=0.5,   
                lr=0.05,
                max_iters=50,
            )

            # Update source-class distribution
            source_class_params[c] = (updated_mu, updated_Sigma)
            all_w2_target.append(w2_target)

            updated_mu_np = to_cpu_numpy(updated_mu)
            updated_Sigma_np = to_cpu_numpy(updated_Sigma)
            updated_Sigma_np = make_positive_definite(updated_Sigma_np)  # Ensure PSD before sampling

            # Sample intermediate latent representations
            z_samples = np.random.multivariate_normal(
                mean=updated_mu_np,
                cov=updated_Sigma_np,
                size=samples_per_class)

            y_samples = np.full((samples_per_class,), c, dtype=int)  # Assign class labels
            weights = np.ones((samples_per_class,))  # Equal weights for now

            z_samples = torch.tensor(z_samples, dtype=torch.float32, device=device)
            y_samples = torch.tensor(y_samples, dtype=torch.long, device=device)
            weights = torch.tensor(weights, dtype=torch.float32, device=device)

            z_new_list.append(z_samples)
            y_new_list.append(y_samples)
            weights_list.append(weights)

        # Combine generated latents
        z_intermediate = torch.cat(z_new_list, dim=0)
        y_intermediate = torch.cat(y_new_list, dim=0)
        weights_intermediate = torch.cat(weights_list, dim=0)

        # Create `DomainDataset` for consistency with `generate_domains()`
        domain_dataset = DomainDataset(z_intermediate, weights_intermediate)
        all_domains.append(domain_dataset)  # Store for later use

        # Compute average Wasserstein distance to target
        avg_target = np.mean([x.item() for x in all_w2_target])

        progress_bar.set_description(
            f"Step {step_idx+1}: W2 (target): {avg_target:.3f}"
        )

        step_idx += 1

    # Append final target dataset for consistency with `generate_domains()`
    all_domains.append(target_dataset)

    return all_domains  # Returns a list of `DomainDataset` objects


def estimate_class_conditional_gaussian_params(
    encoded_dataset,  # Assumes (latents, labels)
    cache_path=None,
    force_recompute=False,
):
    """
    Computes one (mu, sigma) per class from a dataset of encoded latents.
    
    Inputs:
        - encoded_dataset: Tuple (latents, labels), where:
            * latents: Tensor of shape (N, d) containing latent vectors
            * labels: Tensor of shape (N,) containing class labels
    
    Output:
        - class_params: Dict[class_label -> (mu, Sigma)] (Gaussian parameters per class)
    """

    if cache_path is None:
        cache_path = "class_conditional_gaussians.pkl"

    # ✅ Load cached results if available
    if os.path.exists(cache_path) and not force_recompute:
        print(f"🔹 Loading class-conditional Gaussians from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    # Unpack encoded dataset
    # breakpoint()
    latents, labels = encoded_dataset.data, encoded_dataset.targets  # latents: (N, d), labels: (N,)
    unique_classes = torch.unique(labels)

    class_params = {}

    # Compute class-conditional mean & covariance
    for c in unique_classes:
        latents_c = latents[labels == c]  # All latents for class c

        if latents_c.shape[0] > 1:
            mu_c = latents_c.mean(dim=0)  # shape: (d,)
            centered = latents_c - mu_c   # shape: (N, d)
            centered_flat = centered.view(centered.shape[0], -1)  # Shape [N, D]
            # breakpoint()
            sigma_c = (centered_flat.T @ centered_flat) / (centered_flat.shape[0] - 1)
            # sigma_c = (centered.T @ centered) / (latents_c.shape[0] - 1)  # shape: (d, d)
        else:
            mu_c = latents_c.mean(dim=0)
            sigma_c = torch.eye(latents_c.shape[1], device=latents_c.device, dtype=latents_c.dtype) * 1e-5  # Small regularization
        
        class_params[c.item()] = (mu_c, sigma_c)

    # ✅ Cache results
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

def wasserstein_step_full(
    mu_t, sigma_t,
    mu_T, sigma_T,
    delta=0.5,       # Maximum allowed W2 distance from old dist
    lr=0.01,
    max_iters=100,
    lambda_cov=0.1,
    to_cpu=False  # Move final tensors to CPU if needed
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    eps = 1e-8  # Numerical stability constant

    # Move inputs to device efficiently
    def to_device(tensor):
        return tensor.to(device, dtype=torch.float16, non_blocking=True) if tensor.device != device else tensor

    mu_t, sigma_t, mu_T, sigma_T = map(to_device, [mu_t, sigma_t, mu_T, sigma_T])

    # Initialize optimization variables
    v = mu_t.clone().requires_grad_(True)
    sigma_v = sigma_t.clone().requires_grad_(True)

    optimizer = torch.optim.Adam([v, sigma_v], lr=lr)

    # Iterative Wasserstein step
    for step in range(max_iters):
        optimizer.zero_grad()
        loss = w2_gaussian_dist(v, sigma_v, mu_T, sigma_T)
        loss.backward()
        optimizer.step()

    # Ensure positive definiteness
    with torch.no_grad():
        sigma_v_pd = sigma_v + eps * torch.diag_embed(torch.ones_like(sigma_v[0]))  # Efficient PD correction
        sigma_v_pd = make_positive_definite(sigma_v_pd)
        final_mu, final_sigma = v, sigma_v_pd

    # Compute Wasserstein distances **only once**
    w2_prev = w2_gaussian_dist(mu_t, sigma_t, final_mu, final_sigma)
    # w2_target = w2_gaussian_dist(final_mu, final_sigma, mu_T, sigma_T)

    # Enforce W2 constraint
    if w2_prev.item() > delta:
        alpha = min(1.0, delta / w2_prev.item())
        final_mu, final_sigma = w2_shorten(final_mu, final_sigma, mu_T, sigma_T, alpha)

        # Convert to tensor only if needed
        if isinstance(final_mu, np.ndarray):
            final_mu = torch.tensor(final_mu, dtype=torch.float32, device=device)
        if isinstance(final_sigma, np.ndarray):
            final_sigma = torch.tensor(final_sigma, dtype=torch.float32, device=device)
    
        # Recompute Wasserstein distances after projection
    w2_prev = w2_gaussian_dist(mu_t, sigma_t, final_mu, final_sigma)
    w2_target = w2_gaussian_dist(final_mu, final_sigma, mu_T, sigma_T)

    # Move to CPU if required
    if to_cpu:
        final_mu, final_sigma = final_mu.cpu(), final_sigma.cpu()

    return final_mu, final_sigma, w2_prev, w2_target



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


def w2_gaussian_dist(mu1, Sigma1, mu2, Sigma2, batch_size=512):
    """
    Compute the 2-Wasserstein distance between Gaussians N(mu1, Sigma1) and N(mu2, Sigma2)
    using batch processing to reduce memory usage.
    
    Args:
      mu1, mu2: (d,) torch Tensors
      Sigma1, Sigma2: (d,d) torch Tensors
      batch_size: Number of rows/cols processed per batch (default: 512)
    
    Returns:
      A torch.Tensor scalar representing the W2 distance.
    """
    d = Sigma1.shape[0]  # Dimension of covariance matrices

    # 1) Mean difference squared
    mean_diff_sq = torch.norm(mu1 - mu2) ** 2

    # 2) Batched computation for covariance term
    sqrt_Sigma1 = matrix_sqrt_torch(Sigma1)
    w2_cov_term = 0.0  # Initialize sum
    breakpoint()
    for i in range(0, d, batch_size):
        batch_end = min(i + batch_size, d)
        Sigma1_batch = sqrt_Sigma1[i:batch_end, :]
        Sigma2_batch = Sigma2[:, i:batch_end]
    
        inside_batch = Sigma1_batch @ Sigma2_batch @ Sigma1_batch.T  # (batch_size, batch_size)
        sqrt_inside_batch = matrix_sqrt_torch(inside_batch)  # Compute sqrt for this batch
        w2_cov_term += torch.trace(Sigma1_batch + Sigma2_batch - 2.0 * sqrt_inside_batch)

    # 3) Final Wasserstein distance
    w2_sq = mean_diff_sq + w2_cov_term
    w2_sq = torch.clamp(w2_sq, min=0.0)  # Avoid numerical issues

    return torch.sqrt(w2_sq)
