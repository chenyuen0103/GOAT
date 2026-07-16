# GOAT Artifact Layout

The artifact resolver is `goat.core.ArtifactPaths`.

## Environment Overrides

- `GOAT_DATA_DIR`: local raw/processed data root. Default: `<repo>/data`.
- `GOAT_CACHE_DIR`: encoded-feature cache root. Default: `<repo>/cache0.1`.
- `GOAT_OUTPUT_DIR`: output root for generated experiment artifacts. Default:
  `<repo>`.
- `LOG_ROOT`: legacy log root. Default: `<GOAT_OUTPUT_DIR>/logs_rerun`.
- `PLOT_ROOT`: legacy plot root. Default: `<GOAT_OUTPUT_DIR>/plots_rerun`.

## Artifact Classes

- Reproducible outputs: JSONL logs, validation summaries, derived CSVs, tables,
  and figures generated from explicit commands.
- Local caches: encoded features, checkpoints, intermediate domain stats, and
  raw downloaded datasets.
- Archival outputs: old plots/logs/results kept for comparison but not required
  for package import or smoke tests.

Large datasets, checkpoints, caches, plots, and logs should stay out of code
review unless a specific paper artifact must be versioned.

