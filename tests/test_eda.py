"""Tests for EDALoaderStage."""

from __future__ import annotations

import pandas as pd
import pytest

from mlpipe.config.schema import DataConfig, PipelineConfig
from mlpipe.core.context import PipelineContext
from mlpipe.stages.eda import EDALoaderStage


@pytest.mark.unit
def test_eda_name_is_eda():
    assert EDALoaderStage.name == "eda"


@pytest.mark.unit
def test_execute_populates_eda_report(loaded_context):
    stage = EDALoaderStage()
    result = stage.execute(loaded_context)

    assert result.eda_report is not None
    assert isinstance(result.eda_report, dict)


@pytest.mark.unit
def test_eda_report_contains_quality_key(loaded_context):
    stage = EDALoaderStage()
    result = stage.execute(loaded_context)

    assert "quality" in result.eda_report
    assert isinstance(result.eda_report["quality"], dict)


@pytest.mark.unit
def test_execute_populates_correlation_matrix(loaded_context):
    stage = EDALoaderStage()
    result = stage.execute(loaded_context)

    assert result.correlation_matrix is not None


@pytest.mark.unit
def test_execute_handles_column_name_cleaning():
    df = pd.DataFrame({
        "First Name": ["Alice", "Bob", "Charlie"],
        "Last Name ": ["Smith", "Jones", "Brown"],
        "Age#": [25.0, 30.0, 35.0],
        "Salary $": [50000.0, 60000.0, 70000.0],
        "target": [100.0, 200.0, 300.0],
    })
    config = PipelineConfig(data=DataConfig(path="dummy.csv", target="target"))
    ctx = PipelineContext(
        config=config,
        clean_data=df.copy(),
        target_column="target",
        column_types={},
        numeric_columns=["Age#", "Salary $"],
        categorical_columns=["First Name", "Last Name "],
        datetime_columns=[],
        skewed_columns=[],
        normal_columns=["Age#", "Salary $"],
        high_cardinality_columns=[],
        low_cardinality_columns=[],
        columns_to_drop=[],
    )
    stage = EDALoaderStage()
    result = stage.execute(ctx)

    assert "first_name" in result.clean_data.columns
    assert "last_name" in result.clean_data.columns
    assert "age" in result.clean_data.columns
    assert "salary_" in result.clean_data.columns
    assert "target" in result.clean_data.columns
    assert "First Name" not in result.clean_data.columns
    assert "Last Name " not in result.clean_data.columns
    assert "Age#" not in result.clean_data.columns


@pytest.mark.unit
def test_should_skip_returns_false():
    stage = EDALoaderStage()
    ctx = PipelineContext()
    assert stage.should_skip(ctx) is False
