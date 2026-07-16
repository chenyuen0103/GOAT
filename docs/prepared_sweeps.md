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

Use a new log root whenever the protocol changes. The prepared artifacts reduce
computation but do not make results from different pseudo-label thresholds, EM
seed modes, or target-training conventions poolable.

With `--em-ensemble`, mapped EM fits are restricted to
`BIC - BIC_best <= em_bic_delta`, then averaged with weights proportional to
`exp(-0.5 * (BIC - BIC_best))`. The default threshold is 10. Aggregation is
mapping-dependent but reuses the same prepared raw EM fits.
