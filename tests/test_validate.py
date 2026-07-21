"""Tests for ValidateStage — pre-split data quality checks."""

from __future__ import annotations

import pandas as pd
import pytest

from open_mlpipe.config.schema import DataConfig, PipelineConfig
from open_mlpipe.core.context import PipelineContext
from open_mlpipe.stages.validate import ValidateStage


@pytest.fixture
def ctx_with_data():
    """PipelineContext with clean data and config."""
    df = pd.DataFrame({
        "age": [25, 31, 35, 42, 47, 23, 33, 38, 44, 49],
        "income": [50, 62, 70, 85, 95, 48, 65, 75, 82, 93],
        "target": [0, 0, 1, 1, 1, 0, 0, 1, 1, 1],
    })
    ctx = PipelineContext()
    ctx.clean_data = df
    ctx.config = PipelineConfig(  # type: ignore[call-arg]
        data=DataConfig(path="dummy.csv", target="target"),
    )
    return ctx


@pytest.mark.unit
def test_validate_name():
    stage = ValidateStage()
    assert stage.name == "validate"


@pytest.mark.unit
def test_validate_clean_data_passes(ctx_with_data):
    stage = ValidateStage()
    result = stage.execute(ctx_with_data)
    assert result.metrics.get("validation_issues", 0) == 0


@pytest.mark.unit
def test_validate_detects_duplicates():
    # 5 duplicate pairs -> 50% dup rate
    rows = [[25, 50], [25, 50], [30, 60], [30, 60], [35, 70],
            [35, 70], [40, 80], [40, 80], [45, 90], [45, 90]]
    df = pd.DataFrame(rows, columns=pd.Index(["age", "income"]))
    df["target"] = [0, 0, 1, 1, 1, 0, 0, 1, 1, 1]

    ctx = PipelineContext()
    ctx.clean_data = df
    ctx.config = PipelineConfig(  # type: ignore[call-arg]
        data=DataConfig(path="dummy.csv", target="target"),
    )

    stage = ValidateStage()
    result = stage.execute(ctx)
    assert result.metrics.get("validation_issues", 0) > 0


@pytest.mark.unit
def test_validate_detects_constant_column():
    df = pd.DataFrame({
        "const_col": [1, 1, 1, 1, 1],
        "good_col": [10, 20, 30, 40, 50],
        "target": [0, 1, 0, 1, 0],
    })

    ctx = PipelineContext()
    ctx.clean_data = df
    ctx.config = PipelineConfig(  # type: ignore[call-arg]
        data=DataConfig(path="dummy.csv", target="target"),
    )

    stage = ValidateStage()
    result = stage.execute(ctx)
    assert result.metrics.get("validation_issues", 0) > 0