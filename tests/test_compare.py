"""Tests for CompareStage."""

from __future__ import annotations

import pytest

from mlpipe.stages.compare import CompareStage


class TestCompareStage:
    @pytest.mark.unit
    def test_name(self):
        assert CompareStage.name == "compare"

    @pytest.mark.unit
    def test_execute_populates_baseline_models(self, preprocessed_context):
        pytest.importorskip("xgboost")
        pytest.importorskip("lightgbm")
        ctx = preprocessed_context
        stage = CompareStage()
        ctx = stage.execute(ctx)
        assert isinstance(ctx.baseline_models, dict)
        assert len(ctx.baseline_models) > 0

    @pytest.mark.unit
    def test_execute_sets_best_model_name(self, preprocessed_context):
        pytest.importorskip("xgboost")
        pytest.importorskip("lightgbm")
        ctx = preprocessed_context
        stage = CompareStage()
        ctx = stage.execute(ctx)
        assert ctx.best_model_name is not None
        assert isinstance(ctx.best_model_name, str)

    @pytest.mark.unit
    def test_should_skip_defaults_to_false(self, preprocessed_context):
        stage = CompareStage()
        assert stage.should_skip(preprocessed_context) is False
