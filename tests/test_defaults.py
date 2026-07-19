"""Tests for config/defaults.py — SmartDefaults decision engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from open_mlpipe.config.defaults import SmartDefaults
from open_mlpipe.utils.typing import TaskType


@pytest.mark.unit
def test_detect_task_regression():
    series = pd.Series(np.random.normal(100, 15, 500))
    result = SmartDefaults.detect_task(series)
    assert result == TaskType.REGRESSION


@pytest.mark.unit
def test_detect_task_binary_classification():
    series = pd.Series(np.random.choice([0, 1], 200))
    result = SmartDefaults.detect_task(series)
    assert result == TaskType.CLASSIFICATION


@pytest.mark.unit
def test_detect_target_column_finds_numeric():
    df = pd.DataFrame({
        "x1": [1, 2, 3],
        "x2": [4, 5, 6],
        "target": [0.1, 0.2, 0.3],
    })
    result = SmartDefaults.detect_target_column(df)
    assert result == "target"


@pytest.mark.unit
def test_detect_column_types_returns_dict():
    df = pd.DataFrame({
        "num_col": np.random.normal(0, 1, 100),
        "cat_col": np.random.choice(["A", "B", "C"], 100),
        "target": np.random.choice([0, 1], 100),
    })
    types = SmartDefaults.detect_column_types(df)
    assert isinstance(types, dict)
    assert "num_col" in types
    assert "cat_col" in types
    assert "target" in types


@pytest.mark.unit
def test_select_models_regression():
    models = SmartDefaults.select_models(TaskType.REGRESSION, n_rows=5000, n_features=20)
    assert "ridge" in models
    assert "svm" in models
    assert "random_forest" in models
    assert "lightgbm" in models
    assert "xgboost" in models
    assert "catboost" in models
    assert isinstance(models, list)


@pytest.mark.unit
def test_select_models_ratio_gates_svm_out():
    """300 rows / 18 features = ratio 16.7 — SVM should not be included."""
    models = SmartDefaults.select_models(TaskType.CLASSIFICATION, n_rows=300, n_features=18)
    assert "logistic_regression" in models
    assert "svm" not in models
    assert len(models) == 1


@pytest.mark.unit
def test_select_models_ratio_uses_encoded_features():
    """300 rows / 70 encoded features = ratio 4.3 — only LR."""
    models = SmartDefaults.select_models(
        TaskType.CLASSIFICATION, n_rows=300, n_features=18, n_features_encoded=70
    )
    assert models == ["logistic_regression"]


@pytest.mark.unit
def test_select_models_catboost_gate():
    """CatBoost needs >=5000 rows AND ratio > 30."""
    # 5000/200 = 25 — ratio too low
    models = SmartDefaults.select_models(
        TaskType.CLASSIFICATION, n_rows=5000, n_features=30, n_features_encoded=200
    )
    assert "catboost" not in models

    # 10000/200 = 50 — ratio passes
    models2 = SmartDefaults.select_models(
        TaskType.CLASSIFICATION, n_rows=10000, n_features=30, n_features_encoded=200
    )
    assert "catboost" in models2


@pytest.mark.unit
def test_default_scoring_classification():
    scores = SmartDefaults.default_scoring(TaskType.CLASSIFICATION)
    assert "accuracy" in scores
    assert "f1_macro" in scores
    assert "roc_auc_ovr" in scores
    assert isinstance(scores, list)
