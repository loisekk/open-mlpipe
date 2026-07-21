"""Tests for EvaluateStage."""

from __future__ import annotations

from typing import cast

import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from open_mlpipe.stages.evaluate import EvaluateStage
from open_mlpipe.utils.typing import TaskType


def _build_preprocessor(ctx):
    return ColumnTransformer(
        [
            ("num", Pipeline([("impute", SimpleImputer(strategy="mean")), ("scale", StandardScaler())]), ctx.numeric_columns),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(drop="first", sparse_output=False))]), ctx.categorical_columns),
        ],
        remainder="passthrough",
    )


class TestEvaluateStage:
    @pytest.mark.unit
    def test_name(self):
        assert EvaluateStage.name == "evaluate"

    @pytest.mark.unit
    def test_execute_populates_regression_metrics(self, preprocessed_context):
        ctx = preprocessed_context
        pre = _build_preprocessor(ctx)
        pre.fit(ctx.X_train)
        ctx.preprocessor = pre
        pipe = Pipeline([("preprocessor", pre), ("model", Ridge(alpha=1.0))])
        pipe.fit(ctx.X_train, ctx.y_train)
        ctx.tuned_model = None
        ctx.best_model = pipe
        stage = EvaluateStage()
        ctx = stage.execute(ctx)
        assert "test_r2" in ctx.metrics
        assert "test_rmse" in ctx.metrics
        assert "test_mae" in ctx.metrics

    @pytest.mark.unit
    def test_r2_is_reasonable(self, preprocessed_context):
        ctx = preprocessed_context
        pre = _build_preprocessor(ctx)
        pre.fit(ctx.X_train)
        ctx.preprocessor = pre
        pipe = Pipeline([("preprocessor", pre), ("model", Ridge(alpha=1.0))])
        pipe.fit(ctx.X_train, ctx.y_train)
        ctx.tuned_model = None
        ctx.best_model = pipe
        stage = EvaluateStage()
        ctx = stage.execute(ctx)
        r2 = ctx.metrics["test_r2"]
        assert -1.0 <= r2 <= 1.0

    @pytest.mark.unit
    def test_execute_populates_classification_metrics(
        self, sample_dataframe_classification,
    ):
        from open_mlpipe.config.schema import (
            ArtifactConfig,
            CVConfig,
            DataConfig,
            DeploymentConfig,
            EvaluationConfig,
            FeatureSelectionConfig,
            ModelSelectionConfig,
            PipelineConfig,
            TuningConfig,
        )
        from open_mlpipe.core.context import PipelineContext
        df = sample_dataframe_classification
        ctx = PipelineContext()
        ctx.config = PipelineConfig(
            project="test",
            data=DataConfig(path="test.csv", target="target"),
            model_selection=ModelSelectionConfig(cross_validation=CVConfig(n_splits=3)),
            tuning=TuningConfig(enabled=False),
            feature_selection=FeatureSelectionConfig(enabled=False),
            evaluation=EvaluationConfig(explainability=False),
            deployment=DeploymentConfig(enabled=False),
            artifacts=ArtifactConfig(mlflow_tracking=False),
        )
        X = df.drop(columns=["target"])
        y = df["target"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        ctx.X_train = cast(pd.DataFrame, X_train)
        ctx.X_test = cast(pd.DataFrame, X_test)
        ctx.y_train = cast(pd.Series, y_train)
        ctx.y_test = cast(pd.Series, y_test)
        ctx.task_type = TaskType.CLASSIFICATION
        ctx.numeric_columns = ["x1", "x2"]
        ctx.categorical_columns = ["cat_a", "cat_b"]
        pre = ColumnTransformer(
            [
                ("num", Pipeline([("impute", SimpleImputer(strategy="mean")), ("scale", StandardScaler())]), ctx.numeric_columns),
                ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(drop="first", sparse_output=False))]), ctx.categorical_columns),
            ],
            remainder="passthrough",
        )
        pre.fit(X_train)
        ctx.preprocessor = pre
        pipe = Pipeline([("preprocessor", pre), ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))])
        pipe.fit(X_train, y_train)
        ctx.best_model = pipe
        ctx.best_model_name = "logistic_regression"
        ctx.tuned_model = None
        stage = EvaluateStage()
        ctx = stage.execute(ctx)
        assert "test_accuracy" in ctx.metrics
        assert "test_f1_macro" in ctx.metrics
        assert "confusion_matrix" in ctx.metrics
