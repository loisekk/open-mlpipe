"""Tests for SplitStage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from open_mlpipe.config.schema import DataConfig, PipelineConfig
from open_mlpipe.core.context import PipelineContext
from open_mlpipe.stages.split import SplitStage
from open_mlpipe.utils.typing import ColumnType, TaskType


@pytest.mark.unit
def test_split_name_is_split():
    assert SplitStage.name == "split"


@pytest.mark.unit
def test_execute_creates_train_test_splits(clean_context):
    stage = SplitStage()
    result = stage.execute(clean_context)

    assert result.X_train is not None
    assert result.X_test is not None
    assert result.y_train is not None
    assert result.y_test is not None
    assert len(result.X_train) > 0
    assert len(result.X_test) > 0
    assert len(result.y_train) > 0
    assert len(result.y_test) > 0
    assert len(result.X_train) + len(result.X_test) == len(clean_context.clean_data)


@pytest.mark.unit
def test_classification_with_string_target_encodes_y():
    rng = np.random.default_rng(42)
    n = 300
    df = pd.DataFrame({
        "x1": rng.normal(0, 1, n),
        "x2": rng.normal(3, 2, n),
        "target": rng.choice(["yes", "no"], n),
    })
    config = PipelineConfig(data=DataConfig(path="dummy.csv", target="target"))
    ctx = PipelineContext(
        config=config,
        clean_data=df,
        target_column="target",
        task_type=TaskType.CLASSIFICATION,
        column_types={
            "x1": ColumnType.NUMERIC,
            "x2": ColumnType.NUMERIC,
        },
        numeric_columns=["x1", "x2"],
        categorical_columns=[],
        datetime_columns=[],
        skewed_columns=[],
        normal_columns=["x1", "x2"],
        high_cardinality_columns=[],
        low_cardinality_columns=[],
        columns_to_drop=[],
    )
    stage = SplitStage()
    result = stage.execute(ctx)

    assert result.y_train is not None
    assert result.y_test is not None
    assert result.y_train.dtype == int or result.y_train.dtype.name == "int64"
    assert result.y_test.dtype == int or result.y_test.dtype.name == "int64"
    assert set(result.y_train.unique()) == {0, 1}
    assert set(result.y_test.unique()) == {0, 1}


@pytest.mark.unit
def test_test_size_adjusted_for_small_datasets(clean_context):
    stage = SplitStage()
    result = stage.execute(clean_context)

    n = len(clean_context.clean_data)
    expected_test = int(n * 0.1)
    assert len(result.X_test) == expected_test


@pytest.mark.unit
def test_should_skip_returns_false():
    stage = SplitStage()
    ctx = PipelineContext()
    assert stage.should_skip(ctx) is False
