# Generative Gradual Domain Adaptation with Optimal Transport (GOAT)

This repository contains the official implementation of **G**radual
D**O**main **A**daptation with Optimal **T**ransport (GOAT) from
[Gradual Domain Adaptation: Theory and Algorithms](https://jmlr.org/papers/v25/23-1180.html)
(JMLR 2024). The algorithm builds on
[Understanding Gradual Domain Adaptation: Improved Analysis, Optimal Path and Beyond](https://proceedings.mlr.press/v162/wang22aj.html)
(ICML 2022).

The repository is undergoing a staged reorganization. The lightweight `goat`
package provides stable interfaces, while the root-level research scripts remain
the source of truth for the complete experiment implementations.

## Repository structure

```text
GOAT/
├── goat/
│   ├── core/          # configurations, seeding, artifact paths, metrics, schemas
│   ├── data/          # encoded-domain helpers
│   ├── models/        # checkpoint helpers
│   ├── adaptation/    # domain-chain utilities
│   ├── experiments/   # package runners and prepared-sweep orchestration
│   └── analysis/      # log readers and validation summaries
├── experiment_refrac.py       # current full GOAT/CGDA experiment entry point
├── experiments.py             # original paper-era experiment entry point
├── experiment_new.py          # shared training and adaptation implementation
├── run_prepared_sweep.py      # cache-safe, isolated multi-configuration sweeps
├── run_rmnist_label_shift.py  # controlled label/feature-shift study
├── dataset.py, model.py       # datasets and model definitions
├── em_utils.py                # EM fitting and cluster-to-class mappings
├── da_algo.py, a_star_util.py # adaptation and path-generation algorithms
├── scripts/                   # Conda setup and thin command-line wrappers
├── docs/                      # artifact and reproducibility notes
├── tests/                     # lightweight and regression tests
└── legacy/                    # notes for compatibility entry points
```

Generated datasets, checkpoints, encoded features, prepared artifacts, logs,
and plots are local experiment artifacts; they are not part of the Python
package.

## Setup

Run commands from the repository root. Python 3.9 or 3.10 is recommended for
the legacy TensorFlow 2.14/PyTorch 2.2 dependency baseline. With Conda already
installed and initialized, the setup script creates or updates an environment
named `goat`:

```bash
git clone https://github.com/chenyuen0103/GOAT.git
cd GOAT

bash scripts/setup_env.sh
conda activate goat
```

The default profile installs `requirements-gpu.txt`, including TensorFlow (used
by the dataset code) and PyTorch. Useful setup variants are:

```bash
# Include test/lint dependencies and verify the repository.
bash scripts/setup_env.sh --dev --test

# Install only the lightweight package/analysis dependencies.
bash scripts/setup_env.sh --minimal

# Inspect all environment-changing commands without executing them.
bash scripts/setup_env.sh --dev --dry-run
```

Use `--name` and `--python` to override the environment name and Python
version. On a GPU machine, ensure that the installed PyTorch build matches the
local CUDA driver. When the default PyPI build is unsuitable, pass the matching
PyTorch wheel index through `--torch-index-url URL` or `TORCH_INDEX_URL`.

Other dependency sets are available for narrower use cases:

- `requirements-minimal.txt`: package imports, analysis, and CPU utilities.
- `requirements-dev.txt`: minimal dependencies plus `pytest` and `ruff`.
- `requirements.txt`: a historical workstation freeze containing Ubuntu system
  packages; it is retained for provenance and is not the recommended portable
  installation path.

Run `bash scripts/setup_env.sh --help` for every setup option. The script is
idempotent: it reuses an existing named Conda environment and refreshes its pip
dependencies instead of deleting it.

### New-machine setup under `/project/yuen_chen`

For the second machine, use `/project/yuen_chen` as the persistent artifact
root. This creates separate data, model, cache, and output directories and
stores their locations in the Conda environment:

```bash
bash scripts/setup_env.sh \
  --project-root /project/yuen_chen \
  --dev \
  --test
conda activate goat
```

The resulting layout is:

```text
/project/yuen_chen/
├── data/
│   ├── mnist/
│   ├── portraits/dataset_32x32/{M,F}/
│   └── covtype/covtype.data
├── models/
├── cache/{goat,keras,prepared}/
└── outputs/{logs,plots}/
```

The setup persists `GOAT_DATA_DIR`, `GOAT_MODEL_DIR`, `GOAT_CACHE_DIR`,
`GOAT_PREPARED_ARTIFACT_ROOT`, `GOAT_OUTPUT_DIR`, `LOG_ROOT`, `PLOT_ROOT`, and
`KERAS_HOME` with
`conda env config vars`; a Codex agent does not need to re-export them in each
shell. After activation, verify the resolved paths and accelerator before
starting a sweep:

```bash
python -c "from goat.core import ArtifactPaths; print(ArtifactPaths.from_env().to_dict())"
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
python run_prepared_sweep.py --dataset mnist --seeds 0 --gt-domains 0 --generated-domains 0 1 --dry-run
```

### Data preparation

The full experiments support `mnist`, `color_mnist`, `portraits`, and
`covtype`.

- **Rotated MNIST:** `torchvision` downloads MNIST automatically into
  `$GOAT_DATA_DIR/mnist`; source checkpoints are stored in
  `$GOAT_MODEL_DIR/mnist`. Without these variables, the historical Euler paths
  remain the backward-compatible defaults.
- **Color-shift MNIST:** Keras downloads MNIST automatically into `$KERAS_HOME`
  (or its normal user cache when that variable is unset).
- **Portraits:** download the
  [aligned portraits archive](https://www.dropbox.com/s/ubjjoo0b2wz4vgz/faces_aligned_small_mirrored_co_aligned_cropped_cleaned.tar.gz?dl=0)
  used by
  [gradual_domain_adaptation](https://github.com/p-lambda/gradual_domain_adaptation),
  extract its `M` and `F` directories into
  `$GOAT_DATA_DIR/portraits/dataset_32x32/`, and run:

  ```bash
  python create_dataset.py
  ```

  This creates `$GOAT_DATA_DIR/portraits/dataset_32x32.mat`. Without
  `GOAT_DATA_DIR`, both input and output retain their repository-root defaults.
- **Covertype:** download the UCI
  [Covertype dataset](https://archive.ics.uci.edu/dataset/31/covertype) and place
  the uncompressed data file at `$GOAT_DATA_DIR/covtype/covtype.data`.

## Running experiments

### Single run

The package runner is the stable entry point for a standard experiment. It
prints and delegates to `experiment_refrac.py`:

```bash
python -m goat.experiments.runner \
  --dataset mnist \
  --rotation-angle 45 \
  --seed 0 \
  --gt-domains 0 \
  --generated-domains 2 \
  --em-match prototypes
```

Use `--dry-run` to inspect the delegated command without training:

```bash
python -m goat.experiments.runner \
  --dataset mnist --rotation-angle 45 \
  --seed 0 --gt-domains 0 --generated-domains 2 \
  --dry-run
```

For the complete set of CGDA, EM, and diagnostic options, call the full entry
point directly:

```bash
python experiment_refrac.py \
  --dataset mnist \
  --rotation-angle 45 \
  --seed 0 \
  --gt-domains 0 \
  --generated-domains 2 \
  --label-source pseudo \
  --em-match prototypes \
  --em-select bic \
  --em-ensemble \
  --em-seeds 0 1 2
```

The main controls are:

- `--gt-domains`: number of observed real intermediate domains (`Gobs`). The
  supported counts depend on the dataset-specific domain construction.
- `--generated-domains`: number of generated intermediate domains per real
  segment (`Gsyn`). With `0`, generated-domain and EM-dependent CGDA methods are
  skipped.
- `--em-match prototypes`: match EM components to source class prototypes.
- `--em-match pseudo`: match EM components by agreement with teacher
  pseudo-labels.
- `--label-source {pseudo,em}`: choose labels consumed by self-training; this is
  separate from the EM component-mapping rule.
- `--goat-gen-methods w2,fr,natural`: select interpolation geometries.
- `--no-plots`: omit PCA and diagnostic plots during batch runs.

Run `python experiment_refrac.py --help` for all options.

### Prepared multi-seed sweeps

For grids over seeds, observed domains, generated domains, and both EM mapping
schemes, use the prepared sweep. It computes immutable encoded features and raw
EM fits once per `(dataset, seed, Gobs)`, then runs each `(Gsyn, mapping)` branch
in a fresh process. This avoids both repeated preparation and mutable/stale
mapped-label state leaking between configurations.

Dry-run the planned processes first:

```bash
python run_prepared_sweep.py \
  --dataset mnist \
  --rotation-angle 45 \
  --seeds 0 1 2 3 4 \
  --gt-domains 0 \
  --generated-domains 0 1 2 3 \
  --em-matches prototypes pseudo \
  --prepared-artifact-root prepared_artifacts/headline_v1 \
  --log-root logs_prepared_v1 \
  --plot-root plots_prepared_v1 \
  --dry-run
```

Remove `--dry-run` to execute it. Batch sweeps skip plots by default; pass
`--with-plots` for selected runs. After a completed preparation phase, resume
only the isolated workers with the identical configuration and
`--skip-prepare`. Missing, corrupt, or mismatched prepared artifacts fail closed
rather than being silently recomputed.

Use new versioned artifact and log roots whenever the protocol, source model,
EM options, or pseudo-label threshold changes. See
[`docs/prepared_sweeps.md`](docs/prepared_sweeps.md) for the cache identity and
isolation guarantees.

### Controlled label/feature-shift study

The RMNIST diagnostic can vary label shift, feature shift, or both:

```bash
python run_rmnist_label_shift.py \
  --seeds 0 1 2 3 4 \
  --conditions label feature combined \
  --target-rotations 45 \
  --skews 0.0 0.25 0.5 0.75 0.9 \
  --generated-domains 2 \
  --output-dir analysis_outputs/rmnist_label_shift
```

Add `--dry-run` to enumerate the cells without training and `--skip-existing`
to resume an incomplete grid.

## Outputs and path configuration

Standard runs write logs under `logs_rerun/` and plots under `plots_rerun/`.
Prepared sweeps use the roots supplied on their command line. Source-model and
encoded-feature caches are reused when their configuration matches.

The package artifact resolver recognizes:

- `GOAT_DATA_DIR`: raw/processed data root (default `data/`).
- `GOAT_MODEL_DIR`: source-checkpoint root (default `models/` for package code;
  legacy experiment defaults are preserved when unset).
- `GOAT_CACHE_DIR`: package cache root (default `cache0.1/`).
- `GOAT_PREPARED_ARTIFACT_ROOT`: immutable prepared-sweep artifact root
  (default `prepared_artifacts/`).
- `GOAT_OUTPUT_DIR`: base output directory (default: repository root).
- `LOG_ROOT`: log root (default `logs_rerun/`).
- `PLOT_ROOT`: plot root (default `plots_rerun/`).

Dataset and source-checkpoint paths used by the current refactored experiments
honor these overrides. Older archival entry points may still contain historical
paths. More detail is available in
[`docs/artifacts.md`](docs/artifacts.md) and
[`docs/reproducible_runs.md`](docs/reproducible_runs.md).

## Citation

```bibtex
@article{JMLR:v25:23-1180,
  author  = {Yifei He and Haoxiang Wang and Bo Li and Han Zhao},
  title   = {Gradual Domain Adaptation: Theory and Algorithms},
  journal = {Journal of Machine Learning Research},
  year    = {2024},
  volume  = {25},
  number  = {361},
  pages   = {1--40},
  url     = {http://jmlr.org/papers/v25/23-1180.html}
}

@inproceedings{wang2022understanding,
  title     = {Understanding Gradual Domain Adaptation: Improved Analysis,
               Optimal Path and Beyond},
  author    = {Wang, Haoxiang and Li, Bo and Zhao, Han},
  booktitle = {International Conference on Machine Learning},
  pages     = {22784--22801},
  year      = {2022},
  publisher = {PMLR}
}
```
