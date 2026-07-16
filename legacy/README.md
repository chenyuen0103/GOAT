# Legacy Entry Points

The large root-level experiment files remain in place during the staged
reorganization so existing commands and imports keep working.

Current compatibility wrappers live under `goat.experiments` and `scripts/`.
After parity tests cover the paper runs, root-level orchestration files can be
moved here or reduced to thin delegating shims.

