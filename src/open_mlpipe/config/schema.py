"""Pydantic models for pipeline configuration."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    path: str
    format: str = "auto"
    target: str | None = None
    test_size: float = 0.2
    validation_size: float = 0.0
    stratify: bool | None = None
    random_state: int = 42
    max_rows: int | None = None
    separator: str = ","
    encoding: str = "utf-8"


class ProfilingConfig(BaseModel):
    enabled: bool = True
    sample_size: int = 10_000
    correlations: bool = True
    missing_analysis: bool = True
    distribution_plots: bool = True
    statistical_tests: bool = True
    output_format: str = "html"
    output_dir: str = "reports"


class CleaningConfig(BaseModel):
    remove_duplicates: bool = True
    fix_dtypes: bool = True
    optimize_memory: bool = True
    clean_column_names: bool = True
    missing_strategy: str = "auto"
    outlier_method: str = "auto"
    outlier_threshold: float = 3.0
    outlier_action: str = "clip"

class FeatureEngConfig(BaseModel):
    auto_interactions: bool = True
    max_interaction_pairs: int = 10
    auto_polynomial: bool = False
    polynomial_degree: int = 2
    auto_log: bool = True
    log_skew_threshold: float = 1.0
    datetime_features: bool = True
    missingness_flags: bool = True
    binning: bool = False
    n_bins: int = 5


class PreprocessingConfig(BaseModel):
    numeric_strategy: str = "auto"
    categorical_strategy: str = "auto"
    high_cardinality_threshold: int = 20
    ordinal_mappings: dict[str, list[str]] = Field(default_factory=dict)


class CVConfig(BaseModel):
    n_splits: int = 5
    strategy: str = "auto"
    shuffle: bool = True
    random_state: int = 42


class ModelSelectionConfig(BaseModel):
    candidates: Any = "auto"
    cross_validation: CVConfig = Field(default_factory=CVConfig)
    scoring: list[str] = Field(default_factory=list)
    ranking_primary: str = "auto"


class TuningConfig(BaseModel):
    enabled: bool = True
    engine: str = "optuna"
    n_trials: Any = "auto"
    timeout: int = 3600
    sampler: str = "tpe"
    pruner: str = "median"


class FeatureSelectionConfig(BaseModel):
    enabled: bool = True
    method: str = "shap_importance"
    top_k: str = "auto"
    min_importance: float = 0.01


class EvaluationConfig(BaseModel):
    metrics: str = "auto"
    calibration: str = "auto"
    explainability: bool = True
    shap_plots: list[str] = Field(
        default_factory=lambda: ["summary", "dependence", "waterfall"]
    )


class DeploymentConfig(BaseModel):
    enabled: bool = False
    framework: str = "fastapi"
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    docker: bool = True
    docker_base_image: str = "python:3.11-slim"


class ArtifactConfig(BaseModel):
    output_dir: str = "artifacts"
    save_format: str = "joblib"
    mlflow_tracking: bool = True
    mlflow_uri: str = "./mlruns"
    mlflow_experiment: str = "auto"
    compress: bool = True


class PipelineConfig(BaseModel):
    """Full pipeline configuration."""

    project: str = "mlpipe-run"
    version: str = "1.0.0"
    task: str = "auto"
    primary_metric: str = "auto"
    level: int = 1

    data: DataConfig
    profiling: ProfilingConfig = Field(default_factory=ProfilingConfig)
    cleaning: CleaningConfig = Field(default_factory=CleaningConfig)
    feature_engineering: FeatureEngConfig = Field(default_factory=FeatureEngConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    model_selection: ModelSelectionConfig = Field(default_factory=ModelSelectionConfig)
    tuning: TuningConfig = Field(default_factory=TuningConfig)
    feature_selection: FeatureSelectionConfig = Field(default_factory=FeatureSelectionConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    deployment: DeploymentConfig = Field(default_factory=DeploymentConfig)
    artifacts: ArtifactConfig = Field(default_factory=ArtifactConfig)
