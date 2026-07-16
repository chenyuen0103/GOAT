# GOAT Artifact Layout

The artifact resolver is `goat.core.ArtifactPaths`.

## Environment Overrides

- `GOAT_DATA_DIR`: local raw/processed data root. Default: `<repo>/data`.
- `GOAT_MODEL_DIR`: local source-model root. Default: `<repo>/models` for the
  package resolver. Current experiment entry points preserve their historical
  defaults when this variable is unset.
- `GOAT_CACHE_DIR`: encoded-feature cache root. Default: `<repo>/cache0.1`.
- `GOAT_PREPARED_ARTIFACT_ROOT`: prepared-sweep root. Default:
  `<repo>/prepared_artifacts`.
- `GOAT_OUTPUT_DIR`: output root for generated experiment artifacts. Default:
  `<repo>`.
- `LOG_ROOT`: legacy log root. Default: `<GOAT_OUTPUT_DIR>/logs_rerun`.
- `PLOT_ROOT`: legacy plot root. Default: `<GOAT_OUTPUT_DIR>/plots_rerun`.

`bash scripts/setup_env.sh --project-root /project/yuen_chen` creates a portable
layout beneath that directory and persists all of these overrides in the Conda
environment. Dataset-specific overrides (`GOAT_MNIST_ROOT`,
`GOAT_MNIST_MODEL_DIR`, `GOAT_PORTRAITS_RAW_DIR`, `GOAT_PORTRAITS_FILE`, and
`GOAT_COVTYPE_FILE`) take precedence when set.

## Artifact Classes

- Reproducible outputs: JSONL logs, validation summaries, derived CSVs, tables,
  and figures generated from explicit commands.
- Local caches: encoded features, checkpoints, intermediate domain stats, and
  raw downloaded datasets.
- Archival outputs: old plots/logs/results kept for comparison but not required
  for package import or smoke tests.

Large datasets, checkpoints, caches, plots, and logs should stay out of code
review unless a specific paper artifact must be versioned.
