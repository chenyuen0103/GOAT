# Reproducible GOAT Runs

This repository now has a lightweight `goat` package for stable experiment
interfaces while the legacy research scripts remain available.

## Canonical Commands

- Paper-style experiment runner:
  `python -m goat.experiments.runner --dataset mnist --gt-domains 0 --generated-domains 3 --seed 0`
- Dry-run the delegated legacy command:
  `python -m goat.experiments.runner --dataset mnist --generated-domains 1 --seed 0 --dry-run`
- Missing-seed compatibility wrapper:
  `python -m goat.experiments.missing_seeds --dry-run -- --csv results_all_with_settings.csv`
- RMNIST label-shift compatibility wrapper:
  `python -m goat.experiments.rmnist_label_shift --dry-run -- --output-dir analysis_outputs/rmnist_label_shift_smoke`
- Validation summaries can be read with `goat.analysis.ValidationSummary`.
  CLI form: `python -m goat.analysis.validate analysis_outputs/rmnist_label_shift_spectrum/validation.json`.

## Compatibility Policy

The root scripts remain the source of truth for full experiment semantics during
this staged migration. New package commands delegate to those scripts and keep
the current artifact defaults (`logs_rerun`, `plots_rerun`, cached feature
paths) unless environment overrides are set.
