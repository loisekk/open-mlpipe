"""PipelineContext — shared state bus between all stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd

from open_mlpipe.utils.typing import ColumnType, TaskType


@dataclass
class StageMetadata:
    """Immutable record of what a stage did."""

    stage_name: str
    stage_version: str = "1.0"
    input_rows: int = 0
    input_cols: int = 0
    output_rows: int = 0
    output_cols: int = 0
    parameters: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    duration_seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)


if TYPE_CHECKING:
    from open_mlpipe.config.schema import PipelineConfig  # noqa: F401


@dataclass
class PipelineContext:
    """Shared state passed between stages."""

    # Data
    raw_data: pd.DataFrame | None = None
    clean_data: pd.DataFrame | None = None
    featured_data: pd.DataFrame | None = None
    preprocessed_data: pd.DataFrame | None = None

    # Splits
    X_train: pd.DataFrame | None = None
    X_test: pd.DataFrame | None = None
    y_train: pd.Series | None = None
    y_test: pd.Series | None = None

    # Models
    baseline_models: dict[str, Any] = field(default_factory=dict)
    best_model_name: str | None = None
    best_model: Any | None = None
    tuned_model: Any | None = None
    final_model: Any | None = None

    # Metadata
    config: PipelineConfig | None = None  # type: ignore[valid-type]
    task_type: TaskType | None = None
    target_column: str | None = None
    raw_feature_columns: list[str] = field(default_factory=list)
    column_types: dict[str, ColumnType] = field(default_factory=dict)
    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    datetime_columns: list[str] = field(default_factory=list)
    skewed_columns: list[str] = field(default_factory=list)
    normal_columns: list[str] = field(default_factory=list)
    high_cardinality_columns: list[str] = field(default_factory=list)
    low_cardinality_columns: list[str] = field(default_factory=list)
    columns_to_drop: list[str] = field(default_factory=list)

    # Preprocessor
    preprocessor: Any = None

    # Encoder state (set by SplitStage)
    _label_encoder: Any | None = None

    # Stage tracking
    stage_history: list[StageMetadata] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)  # float | str | dict | list

    # EDA results
    eda_report: dict[str, Any] | None = None
    feature_importance: pd.DataFrame | None = None
    correlation_matrix: pd.DataFrame | None = None
    vif_scores: pd.DataFrame | None = None
    statistical_tests: pd.DataFrame | None = None

    # MLflow
    experiment_id: str | None = None
    run_id: str | None = None

    # Artifacts
    artifact_dir: str | None = None
    reports: dict[str, Any] = field(default_factory=dict)

    def add_stage(self, meta: StageMetadata) -> None:
        """Record a stage execution."""
        self.stage_history.append(meta)

    def summary(self) -> str:
        """Human-readable summary of pipeline state."""
        lines = [
            f"Task: {self.task_type}",
            f"Target: {self.target_column}",
            f"Columns: {len(self.column_types)}",
            f"Numeric: {len(self.numeric_columns)}",
            f"Categorical: {len(self.categorical_columns)}",
            f"Stages run: {len(self.stage_history)}",
        ]
        if self.final_model:
            lines.append(f"Model: {self.best_model_name}")
        return "\n".join(lines)
