"""Tests for core/context.py — PipelineContext and StageMetadata."""

from __future__ import annotations

import pytest

from open_mlpipe.core.context import PipelineContext, StageMetadata
from open_mlpipe.utils.typing import TaskType


@pytest.mark.unit
def test_pipeline_context_default_values():
    ctx = PipelineContext()
    assert ctx.raw_data is None
    assert ctx.clean_data is None
    assert ctx.X_train is None
    assert ctx.X_test is None
    assert ctx.y_train is None
    assert ctx.y_test is None
    assert ctx.best_model is None
    assert ctx.final_model is None
    assert ctx.task_type is None
    assert ctx.target_column is None
    assert ctx.stage_history == []
    assert ctx.metrics == {}
    assert ctx.numeric_columns == []
    assert ctx.categorical_columns == []
    assert ctx.column_types == {}


@pytest.mark.unit
def test_stage_metadata_creation():
    meta = StageMetadata(
        stage_name="clean",
        stage_version="1.0",
        input_rows=500,
        input_cols=10,
        output_rows=480,
        output_cols=9,
        parameters={"strategy": "median"},
        metrics={"missing_fixed": 20},
        artifacts={"cleaned_csv": "clean.csv"},
        duration_seconds=2.3,
        warnings=["column dropped: id_col"],
    )
    assert meta.stage_name == "clean"
    assert meta.stage_version == "1.0"
    assert meta.input_rows == 500
    assert meta.output_rows == 480
    assert meta.parameters["strategy"] == "median"
    assert meta.metrics["missing_fixed"] == 20
    assert meta.artifacts["cleaned_csv"] == "clean.csv"
    assert meta.duration_seconds == 2.3
    assert meta.warnings == ["column dropped: id_col"]


@pytest.mark.unit
def test_context_add_stage_appends_to_history(sample_metadata: StageMetadata):
    ctx = PipelineContext()
    ctx.add_stage(sample_metadata)
    assert len(ctx.stage_history) == 1
    assert ctx.stage_history[0].stage_name == sample_metadata.stage_name

    ctx.add_stage(sample_metadata)
    assert len(ctx.stage_history) == 2


@pytest.mark.unit
def test_context_summary_returns_string():
    ctx = PipelineContext(
        task_type=TaskType.REGRESSION,
        target_column="price",
        numeric_columns=["sqft", "bedrooms"],
        categorical_columns=["zipcode"],
        best_model_name="ridge",
    )
    result = ctx.summary()
    assert isinstance(result, str)
    assert "regression" in result.lower()
    assert "price" in result
    assert len(result) > 0


@pytest.mark.unit
def test_context_with_task_type_set():
    ctx = PipelineContext(task_type=TaskType.CLASSIFICATION)
    assert ctx.task_type == TaskType.CLASSIFICATION
