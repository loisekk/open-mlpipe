"""Tests for core/registry.py — StageRegistry."""

from __future__ import annotations

import pytest

from open_mlpipe.core.registry import StageRegistry
from open_mlpipe.stages.load import DataLoaderStage


@pytest.mark.unit
def test_registry_empty_by_default():
    registry = StageRegistry()
    assert registry.list_stages() == []
    assert registry.has("load") is False


@pytest.mark.unit
def test_registry_register_and_get():
    registry = StageRegistry()
    load_stage = DataLoaderStage()
    registry.register(load_stage)
    assert registry.has("load") is True
    retrieved = registry.get("load")
    assert retrieved is load_stage
    assert retrieved.name == "load"


@pytest.mark.unit
def test_registry_get_missing_raises_keyerror():
    registry = StageRegistry()
    with pytest.raises(KeyError, match="not found"):
        registry.get("nonexistent")


@pytest.mark.unit
def test_default_order_contains_all_expected_stages():
    registry = StageRegistry()
    order = registry.default_order()
    assert "load" in order
    assert "eda" in order
    assert "clean" in order
    assert "feature_eng" in order
    assert "split" in order
    assert "preprocess" in order
    assert "compare" in order
    assert "tune" in order
    assert "select" in order
    assert "evaluate" in order
    assert "explain" in order
    assert "save" in order
    assert "deploy" in order
    assert order.index("load") < order.index("clean")
    assert order.index("clean") < order.index("split")


@pytest.mark.unit
def test_list_stages_returns_names(populated_registry: StageRegistry):
    names = populated_registry.list_stages()
    assert isinstance(names, list)
    assert "eda" in names
    assert "load" in names
    # sorted alphabetically
    assert names == sorted(names)
