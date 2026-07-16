# Prepared, isolated CGDA sweeps

`run_prepared_sweep.py` separates expensive immutable preparation from mutable
adaptation runs:

1. One preparation subprocess per `(dataset, seed, Gobs)` writes encoded-feature
   caches and content-addressed raw EM artifacts.
2. Every `(Gsyn, EM mapping)` branch runs in a fresh subprocess.
3. Workers validate and require the prepared raw EM artifact; they never share
   mapped labels, dataset wrappers, generated domains, models, or RNG state.
4. Mapping-invariant pooled GOAT is run only in the first mapping worker; the
   second mapping worker still runs the mapping-dependent CGDA variants.
5. Every worker receives a configuration-derived run ID and writes atomic,
   schema-versioned outputs plus captured stdout/stderr and completion status.

Raw EM artifact identity includes the complete encoded-feature SHA-256 digest,
the domain/seed configuration, and every EM fitting option. Prototype and
agreement mappings are deliberately excluded because they are applied locally
inside each worker.

Example five-seed headline sweep:

```bash
python run_prepared_sweep.py \
  --dataset mnist \
  --rotation-angle 45 \
  --seeds 0 1 2 3 4 \
  --gt-domains 0 \
  --generated-domains 0 1 2 3 \
  --em-matches prototypes pseudo \
  --em-ensemble \
  --em-bic-delta 10 \
  --prepared-artifact-root prepared_artifacts/headline_v1 \
  --log-root logs_prepared_v1 \
  --plot-root plots_prepared_v1
```

Batch sweeps skip plots and full covariance diagnostic files by default. Pass
`--with-plots` only for selected publication runs. To resume after preparation,
pass `--skip-prepare`; missing, mismatched, or corrupt raw EM artifacts then fail
closed instead of being silently recomputed by a worker.

Use `--resume` when repeating a sweep command. It skips exact completed run IDs
and canonical legacy records with the same material configuration. Without
`--resume` or `--force`, an exact completed run is protected from overwrite.
Use `--force` only when intentionally replacing the exact same canonical run.

Use a new log root whenever the protocol changes. The prepared artifacts reduce
computation but do not make results from different pseudo-label thresholds, EM
seed modes, or target-training conventions poolable.

With `--em-ensemble`, mapped EM fits are restricted to
`BIC - BIC_best <= em_bic_delta`, then averaged with weights proportional to
`exp(-0.5 * (BIC - BIC_best))`. The default threshold is 10. Aggregation is
mapping-dependent but reuses the same prepared raw EM fits.

## Canonical outputs and aggregation

In addition to the historical `.txt` and `_curves.jsonl` files, each worker
writes `run.json`, `methods.csv`, `curves.json`, `em_diagnostics.json`, captured
stdout/stderr, and `status.json` under a human-readable configuration hierarchy
ending in a collision-safe run ID. EM diagnostics include every BIC, retained
indices, normalized weights, mappings, anchor configuration, posterior entropy,
and prepared-artifact references.

Import an already completed legacy canary and validate its expected three
configurations with:

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

The command returns nonzero when expected records are missing, duplicated, or
unexpected. Its aggregate directory contains a long-form `runs.csv`, grouped
`summary.csv` with sample standard deviation and a 95% t interval, and
`validation.json`. Pass `--allow-incomplete` only for deliberate partial-grid
diagnostics.

After all five seeds finish, validate the 35-record grid:

```bash
python aggregate_prepared_sweep.py \
  --log-root "$LOG_ROOT/headline_bicensemble_v1" \
  --prepared-artifact-root "$GOAT_PREPARED_ARTIFACT_ROOT/headline_bicensemble_v1" \
  --dataset mnist --target 45 \
  --seeds 0 1 2 3 4 --gt-domains 0 --generated-domains 0 1 2 3 \
  --em-matches prototypes pseudo \
  --em-ensemble --em-bic-delta 10
```

Generate per-run and aggregate plots afterward, without training:

```bash
python plot_prepared_sweep.py \
  --log-root "$LOG_ROOT/headline_bicensemble_v1" \
  --plot-root "$PLOT_ROOT/headline_bicensemble_v1" \
  --dataset mnist
```
