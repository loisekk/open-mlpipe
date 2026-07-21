"""Tests for config/resolver.py — config loading and resolution."""

from __future__ import annotations

import pytest
import yaml

from open_mlpipe.config.resolver import build_level1_config, load_config, resolve_config
from open_mlpipe.config.schema import DataConfig, PipelineConfig


@pytest.mark.unit
def test_build_level1_config_returns_pipeline_config(temp_csv: str):
    config = build_level1_config(temp_csv)
    assert isinstance(config, PipelineConfig)
    assert config.data.path == temp_csv
    assert config.level == 1
    assert config.task != "auto"


@pytest.mark.unit
def test_build_level1_config_explicit_target(temp_csv: str):
    config = build_level1_config(temp_csv, target="target")
    assert config.data.target == "target"


@pytest.mark.unit
def test_resolve_config_resolves_auto_task(temp_csv: str):
    config = build_level1_config(temp_csv)
    config.task = "auto"
    import pandas as pd

    df = pd.read_csv(temp_csv)
    resolved = resolve_config(config, df=df)
    assert resolved.task != "auto"
    assert resolved.task in ("classification", "regression")


@pytest.mark.unit
def test_load_config_from_yaml(temp_dir, base_config):
    import os

    config_path = os.path.join(str(temp_dir), "config.yaml")
    dumped = base_config.model_dump()
    with open(config_path, "w") as f:
        yaml.safe_dump(dumped, f)

    loaded = load_config(config_path)
    assert isinstance(loaded, PipelineConfig)
    assert loaded.project == base_config.project
    assert loaded.data.path == base_config.data.path


@pytest.mark.unit
def test_load_config_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path/config.yaml")


# ── Phase 3: YAML config loading tests ───────────────────────────────────────

@pytest.mark.unit
def test_load_config_regression_yaml():
    """Load configs/regression-default.yaml and verify structure."""
    from pathlib import Path

    config_path = Path(__file__).parent.parent / "configs" / "regression-default.yaml"
    if not config_path.exists():
        pytest.skip("configs/regression-default.yaml not found")

    config = load_config(str(config_path))
    assert isinstance(config, PipelineConfig)
    assert config.task == "auto"
    assert config.data.target == "charges"
    assert config.data.path == "test_datasets/insurance.csv"
    assert config.level == 2
    assert isinstance(config.model_selection.candidates, list)
    assert "ridge" in config.model_selection.candidates


@pytest.mark.unit
def test_load_config_classification_yaml():
    """Load configs/classification-default.yaml and verify structure."""
    from pathlib import Path

    config_path = Path(__file__).parent.parent / "configs" / "classification-default.yaml"
    if not config_path.exists():
        pytest.skip("configs/classification-default.yaml not found")

    config = load_config(str(config_path))
    assert isinstance(config, PipelineConfig)
    assert config.data.target == "target"
    assert config.data.path == "test_datasets/heart_disease.csv"
    assert config.data.stratify is True
    assert config.model_selection.cross_validation.strategy == "stratified_kfold"


@pytest.mark.unit
def test_load_config_quick_test_yaml():
    """Load configs/quick-test.yaml — minimal config with most fields at defaults."""
    from pathlib import Path

    config_path = Path(__file__).parent.parent / "configs" / "quick-test.yaml"
    if not config_path.exists():
        pytest.skip("configs/quick-test.yaml not found")

    config = load_config(str(config_path))
    assert isinstance(config, PipelineConfig)
    assert config.tuning.enabled is False
    assert config.feature_selection.enabled is False
    assert config.evaluation.explainability is False
    assert config.deployment.enabled is False


@pytest.mark.unit
def test_load_config_production_yaml():
    """Load configs/production.yaml — full config with deployment enabled."""
    from pathlib import Path

    config_path = Path(__file__).parent.parent / "configs" / "production.yaml"
    if not config_path.exists():
        pytest.skip("configs/production.yaml not found")

    config = load_config(str(config_path))
    assert isinstance(config, PipelineConfig)
    assert config.deployment.enabled is True
    assert config.deployment.framework == "fastapi"
    assert config.deployment.port == 8000
    assert config.tuning.n_trials == 50


@pytest.mark.unit
def test_resolve_config_auto_task_from_yaml():
    """resolve_config should detect task from data when task=auto."""
    import os
    import tempfile

    import numpy as np

    # Create a temp CSV file for testing
    import pandas as pd

    df = pd.DataFrame({
        "feature1": np.random.normal(0, 1, 100),
        "feature2": np.random.normal(5, 2, 100),
        "target": np.random.normal(10, 3, 100),
    })

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f, index=False)
        temp_path = f.name

    try:
        config = PipelineConfig(
            project="test",
            data=DataConfig(path=temp_path, target="target"),
        )
        assert config.task == "auto"

        df = pd.read_csv(temp_path)
        resolved = resolve_config(config, df=df)
        assert resolved.task == "regression"
        assert resolved.primary_metric != "auto"
    finally:
        os.unlink(temp_path)


@pytest.mark.unit
def test_partial_yaml_config_fills_defaults():
    """A YAML with only data section should fill all other defaults."""
    import os
    import tempfile

    partial = {
        "project": "partial-test",
        "data": {"path": "test.csv", "target": "y"},
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(partial, f)
        config_path = f.name

    try:
        config = load_config(config_path)
        assert isinstance(config, PipelineConfig)
        assert config.project == "partial-test"
        assert config.data.path == "test.csv"
        # Defaults should be filled
        assert config.tuning.enabled is True
        assert config.deployment.enabled is False
        assert config.cleaning.remove_duplicates is True
    finally:
        os.unlink(config_path)
