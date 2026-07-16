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
├── scripts/                   # thin command-line wrappers around `goat`
├── docs/                      # artifact and reproducibility notes
├── tests/                     # lightweight and regression tests
└── legacy/                    # notes for compatibility entry points
```

Generated datasets, checkpoints, encoded features, prepared artifacts, logs,
and plots are local experiment artifacts; they are not part of the Python
package.

## Setup

Run commands from the repository root. Python 3.9 or 3.10 is recommended for
the legacy TensorFlow 2.14/PyTorch 2.2 dependency baseline.

```bash
git clone https://github.com/chenyuen0103/GOAT.git
cd GOAT

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-gpu.txt
```

`requirements-gpu.txt` installs the dependencies needed by the full experiment
scripts, including TensorFlow (used by the dataset code) and PyTorch. On a GPU
machine, ensure that the installed PyTorch build is compatible with the local
CUDA driver.

Other dependency sets are available for narrower use cases:

- `requirements-minimal.txt`: package imports, analysis, and CPU utilities.
- `requirements-dev.txt`: minimal dependencies plus `pytest` and `ruff`.
- `requirements.txt`: a historical workstation freeze containing Ubuntu system
  packages; it is retained for provenance and is not the recommended portable
  installation path.

For development, install the experiment and test dependencies and run the test
suite:

```bash
python -m pip install -r requirements-gpu.txt
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

### Data preparation

The full experiments support `mnist`, `color_mnist`, `portraits`, and
`covtype`.

- **Rotated MNIST:** `torchvision` downloads MNIST automatically. The current
  legacy loader stores it under `/data/common/yuenchen`, and the RMNIST runner
  stores source checkpoints under `/data/common/yuenchen/GDA/mnist_models`.
  Ensure those locations exist and are writable, create suitable symlinks, or
  change the historical paths in `util.py` and `experiment_refrac.py` before
  running on another machine.
- **Color-shift MNIST:** Keras downloads MNIST automatically into its normal
  user cache.
- **Portraits:** download the
  [aligned portraits archive](https://www.dropbox.com/s/ubjjoo0b2wz4vgz/faces_aligned_small_mirrored_co_aligned_cropped_cleaned.tar.gz?dl=0)
  used by
  [gradual_domain_adaptation](https://github.com/p-lambda/gradual_domain_adaptation),
  extract its `M` and `F` directories into `dataset_32x32/`, and run:

  ```bash
  python create_dataset.py
  ```

  This creates `dataset_32x32.mat` in the repository root.
- **Covertype:** download the UCI
  [Covertype dataset](https://archive.ics.uci.edu/dataset/31/covertype) and place
  the uncompressed data file at `covtype.data` in the repository root.

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
- `GOAT_CACHE_DIR`: package cache root (default `cache0.1/`).
- `GOAT_OUTPUT_DIR`: base output directory (default: repository root).
- `LOG_ROOT`: log root (default `logs_rerun/`).
- `PLOT_ROOT`: plot root (default `plots_rerun/`).

The root-level legacy scripts still contain some historical repo-relative and
absolute paths, including the RMNIST paths noted above; the environment
variables do not override every legacy path yet. More detail is available in
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
