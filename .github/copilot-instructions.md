# GOAT: Gradual Domain Adaptation with Optimal Transport

This codebase implements the GOAT algorithm for gradual domain adaptation using optimal transport theory and synthetic domain generation.

## Core Architecture

**Multi-stage Pipeline**: The system follows a 4-stage workflow:
1. **Source Training**: Train encoder+classifier on labeled source domain (`get_source_model()`)
2. **Domain Encoding**: Transform domains into latent space using trained encoder (`get_encoded_dataset()`)
3. **Synthetic Generation**: Create intermediate domains via optimal transport interpolation (`generate_domains()`, `generate_domains_find_next()`)
4. **Self-Training**: Progressively adapt classifier through generated domains (`self_train()`, `self_train_one_domain()`)

**Key Components**:
- `model.py`: Neural architectures (ENCODER, Classifier, MLP, VAE for different datasets)
- `ot_util.py`: Optimal transport planning and domain interpolation via Sinkhorn/EMD
- `expansion_util.py`: Wasserstein-based domain generation with stopping criteria
- `train_model.py`: Self-training procedures with pseudo-labeling
- `experiments.py`: Main experimental pipeline with caching and visualization

## Critical Patterns

**Domain Representation**: All domains use `EncodeDataset` format with `.data` (features) and `.targets` (labels). Intermediate domains maintain this structure for seamless pipeline integration.

**Caching Strategy**: Encoded domains are cached as `cache{ssl_weight}/target{angle}/encoded_{domain_idx}.pt` to avoid recomputation. Check cache existence before expensive encoding operations.

**Device Management**: Consistent CUDA handling with global `device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")`. All tensors must be moved to device before operations.

**Pseudo-labeling**: Use `get_pseudo_labels()` with confidence thresholding. Source domains use ground-truth labels; intermediate domains use teacher model predictions.

## Dataset-Specific Conventions

- **MNIST**: Rotation-based domain shift (`get_single_rotate()` with angle parameter)
- **Portraits**: Gender-based domain shift with predefined intermediate domain indices
- **CovType**: Tabular data requiring `MLP_Encoder` instead of convolutional encoder
- **ColorMNIST**: Color shift requiring VAE encoder for proper representation

## Experimental Framework

**Algorithm Variants**:
- `run_goat()`: Original GOAT with fixed intermediate domain generation
- `run_main_algo()`: Adaptive generation with W2 distance stopping criteria
- `run_main_algo_oracle()`: Oracle version with ground-truth guidance

**Logging**: Use structured CSV logging with `log_progress()` for tracking domain adaptation metrics across steps.

**Visualization**: PCA projections via `plot_encoded_domains()` show domain alignment quality. Consistent naming: `encoded_domains_{method}.png`

## Development Workflows

**Adding New Datasets**: 
1. Create dataset factory in `dataset.py` returning `EncodeDataset` format
2. Add corresponding encoder architecture in `model.py`
3. Implement experiment function following `run_{dataset}_experiment()` pattern

**Model Caching**: Use `model_path` parameter in `get_source_model()` with `force_recompute=False` for development iteration. Cache keys include SSL weight and target domain.

**Debugging**: Set strategic breakpoints in domain generation loops. Use `plot_encoded_domains()` to visualize domain quality during development.

## Key Dependencies

- **POT**: Optimal transport computations (Sinkhorn, EMD)
- **GeomLoss**: Differentiable Wasserstein losses for domain generation
- **Kornia**: Data augmentation for self-supervised learning
- **sklearn**: K-means++ baselines and PCA projections

When extending algorithms, maintain the encoder→encoding→generation→adaptation pipeline structure for consistency with existing experimental framework.
