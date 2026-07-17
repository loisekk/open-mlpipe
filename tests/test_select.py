"""Tests for SelectStage."""

from __future__ import annotations

import pytest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mlpipe.stages.select import SelectStage


class TestSelectStage:
    @pytest.mark.unit
    def test_name(self):
        assert SelectStage.name == "select"

    @pytest.mark.unit
    def test_should_skip_when_feature_selection_disabled(self, preprocessed_context):
        stage = SelectStage()
        ctx = preprocessed_context
        ctx.config.feature_selection.enabled = False
        assert stage.should_skip(ctx) is True

    @pytest.mark.unit
    def test_execute_populates_feature_importance(self, preprocessed_context):
        ctx = preprocessed_context
        ctx.config.feature_selection.enabled = True
        pre = ColumnTransformer(
            [
                ("num", Pipeline([("impute", SimpleImputer(strategy="mean")), ("scale", StandardScaler())]), ctx.numeric_columns),
                ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(drop="first", sparse_output=False))]), ctx.categorical_columns),
            ],
            remainder="passthrough",
        )
        pre.fit(ctx.X_train)
        ctx.preprocessor = pre
        pipe = Pipeline([("preprocessor", pre), ("model", Ridge(alpha=1.0))])
        pipe.fit(ctx.X_train, ctx.y_train)
        ctx.tuned_model = None
        ctx.best_model = pipe
        ctx.best_model_name = "ridge"
        stage = SelectStage()
        ctx = stage.execute(ctx)
        assert ctx.feature_importance is not None
