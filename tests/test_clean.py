"""Tests for CleanStage."""

from __future__ import annotations

import numpy as np
import pytest

from open_mlpipe.config.schema import DataConfig, PipelineConfig
from open_mlpipe.core.context import PipelineContext
from open_mlpipe.stages.clean import CleanStage
from open_mlpipe.utils.typing import ColumnType


@pytest.mark.unit
def test_clean_name_is_clean():
    assert CleanStage.name == "clean"


@pytest.mark.unit
def test_execute_removes_duplicates(sample_dataframe):
    config = PipelineConfig(data=DataConfig(path="dummy.csv", target="target"))
    ctx = PipelineContext(
        config=config,
        raw_data=sample_dataframe,
        clean_data=sample_dataframe.copy(),
        target_column="target",
        column_types={
            "age": ColumnType.NUMERIC,
            "height": ColumnType.NUMERIC,
            "salary": ColumnType.NUMERIC,
            "city": ColumnType.CATEGORICAL,
            "gender": ColumnType.CATEGORICAL,
            "employed": ColumnType.NUMERIC,
            "joined": ColumnType.DATETIME,
            "target": ColumnType.NUMERIC,
        },
    )
    stage = CleanStage()
    result = stage.execute(ctx)

    assert len(result.clean_data) < len(sample_dataframe)


@pytest.mark.unit
def test_execute_drops_rows_with_null_target(sample_dataframe_regression):
    df = sample_dataframe_regression.copy()
    df.loc[0, "target"] = np.nan

    config = PipelineConfig(data=DataConfig(path="dummy.csv", target="target"))
    ctx = PipelineContext(
        config=config,
        clean_data=df,
        target_column="target",
        column_types={},
    )
    stage = CleanStage()
    result = stage.execute(ctx)

    assert result.clean_data["target"].isnull().sum() == 0


@pytest.mark.unit
def test_execute_drops_id_like_columns(sample_dataframe_regression):
    df = sample_dataframe_regression.copy()
    df["id"] = range(len(df))

    config = PipelineConfig(data=DataConfig(path="dummy.csv", target="target"))
    ctx = PipelineContext(
        config=config,
        clean_data=df,
        target_column="target",
        column_types={
            "feat_a": ColumnType.NUMERIC,
            "feat_b": ColumnType.NUMERIC,
            "cat_x": ColumnType.CATEGORICAL,
            "target": ColumnType.NUMERIC,
            "id": ColumnType.ID_LIKE,
        },
    )
    stage = CleanStage()
    result = stage.execute(ctx)

    assert "id" not in result.clean_data.columns


@pytest.mark.unit
def test_should_skip_returns_false():
    stage = CleanStage()
    ctx = PipelineContext()
    assert stage.should_skip(ctx) is False
