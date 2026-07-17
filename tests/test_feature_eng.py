"""Tests for FeatureEngStage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from open_mlpipe.config.schema import DataConfig, PipelineConfig
from open_mlpipe.core.context import PipelineContext
from open_mlpipe.stages.feature_eng import FeatureEngStage


@pytest.mark.unit
def test_feature_eng_name_is_feature_eng():
    assert FeatureEngStage.name == "feature_eng"


@pytest.mark.unit
def test_execute_creates_missingness_flags():
    df = pd.DataFrame({
        "feat_a": [1.0, 2.0, np.nan, 4.0, 5.0],
        "feat_b": [np.nan, 2.0, 3.0, np.nan, 5.0],
        "target": [10, 20, 30, 40, 50],
    })
    config = PipelineConfig(data=DataConfig(path="dummy.csv", target="target"))
    ctx = PipelineContext(config=config, clean_data=df, target_column="target", column_types={})
    stage = FeatureEngStage()
    result = stage.execute(ctx)

    assert "was_missing_feat_a" in result.clean_data.columns
    assert "was_missing_feat_b" in result.clean_data.columns
    assert result.clean_data["was_missing_feat_a"].iloc[2] == 1
    assert result.clean_data["was_missing_feat_a"].iloc[0] == 0


@pytest.mark.unit
def test_execute_creates_datetime_features():
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=10, freq="D"),
        "value": range(10),
        "target": range(10, 20),
    })
    config = PipelineConfig(data=DataConfig(path="dummy.csv", target="target"))
    ctx = PipelineContext(
        config=config,
        clean_data=df,
        target_column="target",
        datetime_columns=["date"],
        column_types={},
    )
    stage = FeatureEngStage()
    result = stage.execute(ctx)

    assert "date_year" in result.clean_data.columns
    assert "date_month" in result.clean_data.columns
    assert "date_day" in result.clean_data.columns
    assert "date_dayofweek" in result.clean_data.columns
    assert "date_is_weekend" in result.clean_data.columns
    assert "date" not in result.clean_data.columns


@pytest.mark.unit
def test_execute_applies_log_transforms_to_skewed_columns():
    rng = np.random.default_rng(42)
    skewed_data = np.exp(rng.normal(0, 1, 200))
    normal_data = rng.normal(0, 1, 200)
    df = pd.DataFrame({
        "feat_a": skewed_data,
        "feat_b": normal_data,
        "target": rng.normal(0, 1, 200),
    })
    config = PipelineConfig(data=DataConfig(path="dummy.csv", target="target"))
    ctx = PipelineContext(
        config=config,
        clean_data=df,
        target_column="target",
        skewed_columns=["feat_a"],
        column_types={},
    )
    stage = FeatureEngStage()
    result = stage.execute(ctx)

    assert "log_feat_a" in result.clean_data.columns


@pytest.mark.unit
def test_should_skip_returns_false():
    stage = FeatureEngStage()
    ctx = PipelineContext()
    assert stage.should_skip(ctx) is False
