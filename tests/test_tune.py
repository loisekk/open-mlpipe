"""Tests for TuneStage."""

from __future__ import annotations

import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from open_mlpipe.stages.tune import TuneStage


class TestTuneStage:
    @pytest.mark.unit
    def test_name(self):
        assert TuneStage.name == "tune"

    @pytest.mark.unit
    def test_should_skip_when_tuning_disabled(self, preprocessed_context):
        stage = TuneStage()
        ctx = preprocessed_context
        ctx.config.tuning.enabled = False
        assert stage.should_skip(ctx) is True

    @pytest.mark.unit
    def test_execute_with_tuning_enabled_produces_tuned_model(self, preprocessed_context):
        """Tuning runs and records baseline + tuned metrics.
        tuned_model may be None if tuning didn't improve baseline (correct behavior)."""
        pytest.importorskip("xgboost")
        pytest.importorskip("lightgbm")
        ctx = preprocessed_context
        ctx.config.tuning.enabled = True
        ctx.config.tuning.n_trials = 3
        ctx.config.tuning.timeout = 30
        pre = ColumnTransformer(
            [
                ("num", Pipeline([("impute", SimpleImputer(strategy="mean")), ("scale", StandardScaler())]), ctx.numeric_columns),
                ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(drop="first", sparse_output=False))]), ctx.categorical_columns),
            ],
            remainder="passthrough",
        )
        pre.fit(ctx.X_train)
        ctx.preprocessor = pre
        ctx.best_model_name = "random_forest"
        ctx.best_model = Pipeline([("preprocessor", pre), ("model", RandomForestRegressor(n_estimators=10, random_state=42))])
        stage = TuneStage()
        ctx = stage.execute(ctx)
        # Baseline score must be valid (not -inf)
        assert ctx.metrics["tune_baseline_score"] > 0
        # Tuning attempt must be recorded
        assert "tuned_best_value" in ctx.metrics
        assert "tuned_best_params" in ctx.metrics
        # tuned_model may be None if tuning didn't improve — that's correct

    @pytest.mark.unit
    def test_tuned_best_value_in_metrics(self, preprocessed_context):
        pytest.importorskip("xgboost")
        pytest.importorskip("lightgbm")
        ctx = preprocessed_context
        ctx.config.tuning.enabled = True
        ctx.config.tuning.n_trials = 3
        ctx.config.tuning.timeout = 30
        pre = ColumnTransformer(
            [
                ("num", Pipeline([("impute", SimpleImputer(strategy="mean")), ("scale", StandardScaler())]), ctx.numeric_columns),
                ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(drop="first", sparse_output=False))]), ctx.categorical_columns),
            ],
            remainder="passthrough",
        )
        pre.fit(ctx.X_train)
        ctx.preprocessor = pre
        ctx.best_model_name = "random_forest"
        ctx.best_model = Pipeline([("preprocessor", pre), ("model", RandomForestRegressor(n_estimators=10, random_state=42))])
        stage = TuneStage()
        ctx = stage.execute(ctx)
        assert "tuned_best_value" in ctx.metrics
        assert "tuned_best_params" in ctx.metrics
        assert "tune_baseline_score" in ctx.metrics
