def test_lightweight_goat_imports():
    import goat
    from goat.core import ArtifactPaths, ExperimentConfig, RunSpec

    assert goat.repo_root().name == "GOAT"
    assert ArtifactPaths.from_env().repo_root.name == "GOAT"
    assert ExperimentConfig(dataset="mnist").dataset == "mnist"
    assert RunSpec(dataset="mnist", seed=0, gt_domains=0, generated_domains=1).seed == 0


def test_subpackage_imports_are_lightweight():
    import goat.analysis
    import goat.experiments

    assert hasattr(goat.analysis, "ValidationSummary")
    assert goat.experiments.build_legacy_experiment_command.__name__ == "build_legacy_experiment_command"
