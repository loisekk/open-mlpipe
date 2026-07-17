"""Tests for core/stage.py — Stage ABC."""

from __future__ import annotations

import pytest

from open_mlpipe.core.context import PipelineContext
from open_mlpipe.core.stage import Stage


class _TestConcreteStage(Stage):
    """Minimal concrete Stage for testing the abstract interface."""

    name = "test_concrete"
    version = "1.0"

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        ctx.best_model_name = "test_model"
        return ctx

    def should_skip(self, ctx: PipelineContext) -> bool:
        return ctx.raw_data is not None


@pytest.mark.unit
def test_concrete_stage_has_name():
    stage = _TestConcreteStage()
    assert stage.name == "test_concrete"


@pytest.mark.unit
def test_concrete_stage_version():
    stage = _TestConcreteStage()
    assert stage.version == "1.0"


@pytest.mark.unit
def test_should_skip_returns_true_when_raw_data_present():
    stage = _TestConcreteStage()
    ctx = PipelineContext(raw_data=object())  # any non-None value
    assert stage.should_skip(ctx) is True


@pytest.mark.unit
def test_should_skip_returns_false_when_raw_data_none():
    stage = _TestConcreteStage()
    ctx = PipelineContext(raw_data=None)
    assert stage.should_skip(ctx) is False


@pytest.mark.unit
def test_execute_modifies_context():
    stage = _TestConcreteStage()
    ctx = PipelineContext()
    result = stage.execute(ctx)
    assert result.best_model_name == "test_model"
    assert result is ctx
