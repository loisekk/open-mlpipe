"""Tests for DataLoaderStage."""

from __future__ import annotations

import pytest

from mlpipe.config.schema import DataConfig, PipelineConfig
from mlpipe.core.context import PipelineContext
from mlpipe.stages.load import DataLoaderStage
from mlpipe.utils.typing import TaskType


@pytest.mark.unit
def test_data_loader_name_is_load():
    assert DataLoaderStage.name == "load"


@pytest.mark.unit
def test_data_loader_version_is_1_0():
    assert DataLoaderStage.version == "1.0"


@pytest.mark.unit
def test_execute_populates_raw_and_clean_data(temp_csv):
    config = PipelineConfig(data=DataConfig(path=temp_csv, target="target"))
    ctx = PipelineContext(config=config)
    stage = DataLoaderStage()
    result = stage.execute(ctx)

    assert result.raw_data is not None
    assert result.clean_data is not None
    assert len(result.raw_data) > 0
    assert len(result.clean_data) > 0


@pytest.mark.unit
def test_execute_detects_task_type(temp_csv):
    config = PipelineConfig(data=DataConfig(path=temp_csv, target="target"))
    ctx = PipelineContext(config=config)
    stage = DataLoaderStage()
    result = stage.execute(ctx)

    assert result.task_type is not None
    assert result.task_type == TaskType.REGRESSION


@pytest.mark.unit
def test_should_skip_returns_true_when_raw_data_set(sample_dataframe_regression):
    ctx = PipelineContext(raw_data=sample_dataframe_regression)
    stage = DataLoaderStage()
    assert stage.should_skip(ctx) is True
