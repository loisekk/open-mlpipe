"""Tests for PreprocessStage."""

from __future__ import annotations

import pytest
from sklearn.compose import ColumnTransformer

from mlpipe.stages.preprocess import PreprocessStage


class TestPreprocessStage:
    @pytest.mark.unit
    def test_name(self):
        assert PreprocessStage.name == "preprocess"

    @pytest.mark.unit
    def test_execute_creates_preprocessor(self, split_context):
        stage = PreprocessStage()
        ctx = stage.execute(split_context)
        assert ctx.preprocessor is not None

    @pytest.mark.unit
    def test_preprocessor_is_column_transformer(self, split_context):
        stage = PreprocessStage()
        ctx = stage.execute(split_context)
        assert isinstance(ctx.preprocessor, ColumnTransformer)
        assert len(ctx.preprocessor.transformers) >= 1

    @pytest.mark.unit
    def test_should_skip_defaults_to_false(self, split_context):
        stage = PreprocessStage()
        assert stage.should_skip(split_context) is False
