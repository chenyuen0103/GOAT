"""Canonical experiment CLI wrappers and run specifications."""

__all__ = ["build_legacy_experiment_command"]


def __getattr__(name):
    if name == "build_legacy_experiment_command":
        from goat.experiments.runner import build_legacy_experiment_command

        return build_legacy_experiment_command
    raise AttributeError(name)

