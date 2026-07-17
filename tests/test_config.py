"""Tests for config/schema.py — Pydantic model validation."""

from __future__ import annotations

import pytest

from mlpipe.config.schema import (
    ArtifactConfig,
    CleaningConfig,
    DataConfig,
    DeploymentConfig,
    EvaluationConfig,
    FeatureSelectionConfig,
    ModelSelectionConfig,
    PipelineConfig,
    TuningConfig,
)


@pytest.mark.unit
def test_data_config_defaults():
    config = DataConfig(path="data.csv", target="target_col")
    assert config.test_size == 0.2
    assert config.validation_size == 0.0
    assert config.random_state == 42
    assert config.format == "auto"
    assert config.separator == ","
    assert config.encoding == "utf-8"
    assert config.max_rows is None


@pytest.mark.unit
def test_data_config_test_size_must_be_positive():
    config = DataConfig(path="data.csv", test_size=0.33)
    assert config.test_size == 0.33


@pytest.mark.unit
def test_cleaning_config_custom_outlier_method():
    config = CleaningConfig(outlier_method="trim", outlier_threshold=2.5)
    assert config.outlier_method == "trim"
    assert config.outlier_threshold == 2.5


@pytest.mark.unit
def test_pipeline_config_minimal_fields():
    config = PipelineConfig(
        data=DataConfig(path="test.csv", target="target"),
    )
    assert config.project == "mlpipe-run"
    assert config.version == "1.0.0"
    assert config.task == "auto"
    assert config.level == 1


@pytest.mark.unit
def test_pipeline_config_model_dump_contains_sub_configs():
    config = PipelineConfig(
        data=DataConfig(path="test.csv", target="target"),
    )
    dumped = config.model_dump()
    assert "data" in dumped
    assert "profiling" in dumped
    assert "cleaning" in dumped
    assert "feature_engineering" in dumped
    assert "preprocessing" in dumped
    assert "model_selection" in dumped
    assert "tuning" in dumped
    assert "feature_selection" in dumped
    assert "evaluation" in dumped
    assert "deployment" in dumped
    assert "artifacts" in dumped


@pytest.mark.unit
def test_model_selection_config_candidates_as_list():
    config = ModelSelectionConfig(
        candidates=["ridge", "lasso", "elastic_net"],
        scoring=["neg_root_mean_squared_error", "r2"],
        ranking_primary="neg_root_mean_squared_error",
    )
    assert config.candidates == ["ridge", "lasso", "elastic_net"]
    assert "r2" in config.scoring


@pytest.mark.unit
def test_tuning_config_n_trials_auto():
    config = TuningConfig(n_trials="auto")
    assert config.n_trials == "auto"
    assert config.engine == "optuna"
    assert config.sampler == "tpe"
    assert config.pruner == "median"
    assert config.timeout == 3600


@pytest.mark.unit
def test_feature_selection_config_defaults():
    config = FeatureSelectionConfig()
    assert config.enabled is True
    assert config.method == "shap_importance"
    assert config.top_k == "auto"
    assert config.min_importance == 0.01


@pytest.mark.unit
def test_evaluation_config_shap_plots():
    config = EvaluationConfig()
    assert "summary" in config.shap_plots
    assert "dependence" in config.shap_plots
    assert "waterfall" in config.shap_plots
    assert config.explainability is True


@pytest.mark.unit
def test_deployment_config_defaults():
    config = DeploymentConfig()
    assert config.enabled is False
    assert config.framework == "fastapi"
    assert config.host == "0.0.0.0"
    assert config.port == 8000


@pytest.mark.unit
def test_artifact_config_defaults():
    config = ArtifactConfig()
    assert config.output_dir == "artifacts"
    assert config.save_format == "joblib"
    assert config.mlflow_tracking is True
    assert config.compress is True
