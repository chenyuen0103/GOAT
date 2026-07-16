# Codex handoff: CGDA prepared sweeps

Last updated: 2026-07-16

This file carries the operational context for continuing the CGDA experiment
reruns on another machine. It intentionally contains no credentials or secret
values. Treat statements about remote experiment completion as user-reported
until the corresponding artifacts and logs have been inspected on that host.

## Objective

Rerun the paper-critical CGDA/GOAT experiments with a clean, efficient, and
auditable protocol. The immediate experiment is Rotated MNIST at 45 degrees,
comparing:

- five outer seeds: `0 1 2 3 4`;
- generated intermediate-domain counts `Gsyn = 0, 1, 2, 3`;
- no real intermediate domains for the first headline sweep (`Gobs = 0`);
- EM-to-class mappings `prototypes` and `pseudo`;
- three diagonal-covariance EM initializations per outer seed;
- BIC-trimmed, BIC-weighted EM posterior ensembling.

The intended follow-up is to run the analogous Portraits sweep and then study
real intermediate domains (`Gobs > 0`) after the generated-only headline grid
is complete and checked.

## Repository and environment

- Repository: `~/GOAT`
- Branch: `main`
- Verified commit at handoff creation: `35252f5` (`polish setup`)
- Persistent experiment root on the second machine: `/project/yuen_chen`
- Conda environment: `goat`
- Paper repository: `~/GDA_JMLR`
- Paper-side bound notes:
  `~/GDA_JMLR/notebooks/bound_instantiation.md`

The Conda setup should persist these variables:

- `GOAT_DATA_DIR`
- `GOAT_MODEL_DIR`
- `GOAT_CACHE_DIR`
- `GOAT_PREPARED_ARTIFACT_ROOT`
- `GOAT_OUTPUT_DIR`
- `LOG_ROOT`
- `PLOT_ROOT`
- `KERAS_HOME`

Check them after activating the environment:

```bash
conda activate goat
codex login status
python -c "from goat.core import ArtifactPaths; print(ArtifactPaths.from_env().to_dict())"
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

The CLI used to create this handoff was authenticated with an API key rather
than a ChatGPT login. Do not copy `~/.codex` between hosts. Use this file and
the repository history to transfer context.

## Implemented experiment infrastructure

The relevant entrypoint is `run_prepared_sweep.py`. It orchestrates:

1. one preparation process per `(dataset, outer seed, Gobs)`;
2. content-addressed, checksum-validated prepared artifacts;
3. isolated worker processes for each `(Gsyn, EM mapping)` configuration;
4. reuse of the source model, encoded real domains, source prototypes, and raw
   EM fits;
5. omission of the redundant second mapping when `Gsyn = 0`;
6. omission of mapping-invariant pooled GOAT work in the second mapping worker;
7. fail-closed loading when workers require missing, mismatched, or corrupt
   prepared artifacts.

Important files:

- `run_prepared_sweep.py`
- `goat/experiments/prepared_sweep.py`
- `goat/core/prepared_artifacts.py`
- `experiment_refrac.py`
- `em_utils.py`
- `docs/prepared_sweeps.md`
- `README.md`

## EM protocol

The headline configuration uses:

```text
K = number of classes
covariance = diagonal
PCA = none
EM seeds = 0, 1, 2 interpreted as offsets from the outer seed
BIC retention threshold = 10
mapping schemes = prototypes, pseudo
```

For EM fit `r`, define

```text
delta_BIC_r = BIC_r - min_s BIC_s.
```

Fits with `delta_BIC_r > 10` are discarded. Retained fits receive normalized
weights proportional to

```text
exp(-0.5 * delta_BIC_r).
```

The mapped class-posterior matrices are averaged with those weights. The saved
metadata records all BIC values, retained indices, weights, configurations,
and the ensemble criterion.

Without `--em-ensemble`, the code still fits the EM grid but selects one fit by
BIC. The headline run should include both `--em-ensemble` and
`--em-bic-delta 10`.

## Verification already completed locally

- Full tests: `24 passed, 1 skipped`.
- The skipped test was optional/TensorFlow-dependent in the lighter test
  environment.
- Python compilation completed successfully.
- `git diff --check` was clean.
- A dry run of the complete five-seed command expanded to five preparation
  jobs and 35 isolated workers.
- The repository was clean and matched `origin/main` when this file was
  created.

## User-reported completed canary

The following command completed on the experiment machine:

```bash
python run_prepared_sweep.py \
  --dataset mnist \
  --rotation-angle 45 \
  --seeds 0 \
  --gt-domains 0 \
  --generated-domains 0 1 \
  --em-matches prototypes pseudo \
  --em-ensemble \
  --em-bic-delta 10 \
  --prepared-artifact-root "$GOAT_PREPARED_ARTIFACT_ROOT/headline_bicensemble_v1" \
  --log-root "$LOG_ROOT/headline_bicensemble_v1" \
  --plot-root "$PLOT_ROOT/headline_bicensemble_v1"
```

Because the sweep runner invokes subprocesses with `check=True`, returning to
the shell without a traceback implies that its preparation and all three
canary workers exited successfully. The logs must still be inspected before
treating their numerical results as validated.

Expected canary curve records under
`$LOG_ROOT/headline_bicensemble_v1/mnist/s0/target45`:

- `Gsyn=0`, `prototypes`;
- `Gsyn=1`, `prototypes`;
- `Gsyn=1`, `pseudo`.

Verify them with:

```bash
find "$LOG_ROOT/headline_bicensemble_v1/mnist/s0/target45" \
  -name '*_curves.jsonl' -size +0c -print
```

There should be exactly three nonempty curve records at this stage. Inspect
their JSON metadata and ensure that it reports `em_ensemble: true`,
`em_bic_delta: 10.0`, and the requested mapping and generated-domain count.

## Next actions

### 0. Pull the output-audit implementation and backfill the canary

The output layer now writes canonical run directories, real CSV tables,
complete EM diagnostics, captured stdout/stderr, status files, and sweep
manifests. After pulling the updated repository, import the three completed
legacy canary records:

```bash
python aggregate_prepared_sweep.py \
  --log-root "$LOG_ROOT/headline_bicensemble_v1" \
  --prepared-artifact-root "$GOAT_PREPARED_ARTIFACT_ROOT/headline_bicensemble_v1" \
  --dataset mnist --target 45 \
  --seeds 0 --gt-domains 0 --generated-domains 0 1 \
  --em-matches prototypes pseudo \
  --em-ensemble --em-bic-delta 10 \
  --backfill-legacy
```

The validation must report `passed: true`, `expected_result_rows: 3`, and
`actual_result_rows: 3`. Legacy stdout cannot be recovered, but BIC retention
and ensemble weights are reconstructed from checksum-validated raw EM
artifacts.

### 1. Complete seed 0 without repeating its canary workers

The seed-0 prepared EM artifacts are independent of `Gsyn`, so run only the
missing `Gsyn=2,3` workers:

```bash
python run_prepared_sweep.py \
  --dataset mnist \
  --rotation-angle 45 \
  --seeds 0 \
  --gt-domains 0 \
  --generated-domains 2 3 \
  --em-matches prototypes pseudo \
  --em-ensemble \
  --em-bic-delta 10 \
  --prepared-artifact-root "$GOAT_PREPARED_ARTIFACT_ROOT/headline_bicensemble_v1" \
  --log-root "$LOG_ROOT/headline_bicensemble_v1" \
  --plot-root "$PLOT_ROOT/headline_bicensemble_v1" \
  --skip-prepare \
  --resume
```

### 2. Run seeds 1 through 4

Do not pass `--skip-prepare` here because these seeds need their own prepared
artifacts:

```bash
python run_prepared_sweep.py \
  --dataset mnist \
  --rotation-angle 45 \
  --seeds 1 2 3 4 \
  --gt-domains 0 \
  --generated-domains 0 1 2 3 \
  --em-matches prototypes pseudo \
  --em-ensemble \
  --em-bic-delta 10 \
  --prepared-artifact-root "$GOAT_PREPARED_ARTIFACT_ROOT/headline_bicensemble_v1" \
  --log-root "$LOG_ROOT/headline_bicensemble_v1" \
  --plot-root "$PLOT_ROOT/headline_bicensemble_v1" \
  --resume
```

Run these commands sequentially on a single GPU unless resource isolation has
been checked explicitly.

### 3. Validate the complete grid

For each seed, expect seven worker configurations:

```text
Gsyn=0: prototypes only
Gsyn=1: prototypes, pseudo
Gsyn=2: prototypes, pseudo
Gsyn=3: prototypes, pseudo
```

The complete five-seed grid therefore contains 35 nonempty curve records.
Check that every record has the expected seed, `Gobs`, `Gsyn`, mapping, EM
ensemble flag, BIC threshold, and method curves. Do not select configurations
using target test accuracy; target labels are for final evaluation only.

Run the strict validator and aggregator:

```bash
python aggregate_prepared_sweep.py \
  --log-root "$LOG_ROOT/headline_bicensemble_v1" \
  --prepared-artifact-root "$GOAT_PREPARED_ARTIFACT_ROOT/headline_bicensemble_v1" \
  --dataset mnist --target 45 \
  --seeds 0 1 2 3 4 --gt-domains 0 --generated-domains 0 1 2 3 \
  --em-matches prototypes pseudo \
  --em-ensemble --em-bic-delta 10
```

It must report 35 expected and 35 actual canonical records with no missing,
unexpected, or duplicate configurations.

### 4. Summarize before launching the next benchmark

Produce seed-level and aggregate tables for at least:

- final target accuracy of each method;
- mean, standard deviation, and preferably a confidence interval across the
  five outer seeds;
- the effect of `Gsyn` within each mapping scheme;
- the difference between `prototypes` and `pseudo` mapping;
- retained EM fits, BIC deltas, and ensemble weights;
- runtime per preparation phase and worker.

Only after checking this table should the same design be applied to Portraits.

Create plots after validation without rerunning adaptation:

```bash
python plot_prepared_sweep.py \
  --log-root "$LOG_ROOT/headline_bicensemble_v1" \
  --plot-root "$PLOT_ROOT/headline_bicensemble_v1" \
  --dataset mnist
```

## Protocol caveats

1. **Source-model seeds.** The Rotated MNIST source checkpoint name is
   seed-invariant:
   `src0_tgt45_ssl<weight>_dim<dimension>.pth`. The five outer seeds therefore
   vary EM fitting and adaptation, but reuse one source model. Report this
   explicitly. Independent end-to-end source-training seeds would require a
   separate protocol/code change.
2. **Target labels.** EM-to-class accuracy printed during experiments is a
   diagnostic using labels available to the benchmark evaluator. It must not
   be used to choose EM fits or tune the headline configuration.
3. **Mapping comparison.** Keep both `prototypes` and `pseudo`; neither should
   be chosen post hoc from target accuracy.
4. **Prepared artifacts.** Use the versioned
   `headline_bicensemble_v1` root. Do not mix older prepared caches into this
   run. Artifact metadata and checksums fail closed on mismatches.
5. **Diagonal covariance.** This is the stable and efficient high-dimensional
   baseline. Full covariance at encoder dimension 2048 would be much more
   expensive and poorly conditioned.
6. **Plots.** Batch plots are disabled by default. Generate publication plots
   from the validated curve records afterward.

## Paper-side context

The numeric Theorem-1 instantiation currently concludes that the conditional
path term is not absolutely scale-identified because encoder-space W1 must be
multiplied by an unidentified IPM-domination constant `C`. The label-marginal
term is already in probability/risk units. Consequently:

- treat conditional-path experiments comparatively unless `C` is controlled;
- treat endpoint-derived `C_endpoint` and `C_max` as sensitivity analyses, not
  certificates;
- retain the distinction between label shift and class-conditional feature
  shift;
- do not oversell full statistical placeholder rows dominated by the
  sequential-complexity placeholder.

The reruns are meant to strengthen empirical comparisons across EM mappings,
generated-domain counts, and eventually real-domain counts. They do not by
themselves identify the absolute constant `C`.

## Suggested continuation prompt

From `~/GOAT` on the experiment machine, start Codex with:

```bash
codex -C ~/GOAT \
  'Read docs/CODEX_HANDOFF.md, README.md, and docs/prepared_sweeps.md. Inspect git status, the prepared-artifact manifests, and the canary logs on this machine. Continue from Next actions without rerunning completed configurations. Report any discrepancy before launching further jobs.'
```
