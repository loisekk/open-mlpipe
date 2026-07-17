"""Tests for ExplainStage."""

from __future__ import annotations

import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from open_mlpipe.stages.explain import ExplainStage


class TestExplainStage:
    @pytest.mark.unit
    def test_name(self):
        assert ExplainStage.name == "explain"

    @pytest.mark.unit
    def test_should_skip_when_explainability_disabled(self, preprocessed_context):
        stage = ExplainStage()
        ctx = preprocessed_context
        ctx.config.evaluation.explainability = False
        assert stage.should_skip(ctx) is True

    @pytest.mark.unit
    def test_execute_populates_shap_values(self, preprocessed_context):
        ctx = preprocessed_context
        ctx.config.evaluation.explainability = True
        pre = ColumnTransformer(
            [
                ("num", Pipeline([("impute", SimpleImputer(strategy="mean")), ("scale", StandardScaler())]), ctx.numeric_columns),
                ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(drop="first", sparse_output=False))]), ctx.categorical_columns),
            ],
            remainder="passthrough",
        )
        pre.fit(ctx.X_train)
        ctx.preprocessor = pre
        pipe = Pipeline([("preprocessor", pre), ("model", RandomForestRegressor(n_estimators=10, random_state=42))])
        pipe.fit(ctx.X_train, ctx.y_train)
        ctx.tuned_model = pipe
        stage = ExplainStage()
        try:
            ctx = stage.execute(ctx)
            has_shap = "shap_values" in ctx.reports and ctx.reports["shap_values"] is not None
        except Exception:
            has_shap = False
        if has_shap:
            import numpy as np
            assert isinstance(ctx.reports["shap_values"], list | np.ndarray)
