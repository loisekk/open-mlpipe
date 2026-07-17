"""Config resolver — merges user config with smart defaults based on level."""

from __future__ import annotations

import yaml

from open_mlpipe.config.defaults import SmartDefaults
from open_mlpipe.config.schema import PipelineConfig
from open_mlpipe.utils.typing import TaskType


def load_config(path: str) -> PipelineConfig:
    """Load YAML config file into PipelineConfig."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    return PipelineConfig(**raw)


def build_level1_config(data_path: str, target: str | None = None) -> PipelineConfig:
    """Build a complete config from just a data path (zero-touch)."""

    from open_mlpipe.utils.io import load_data

    df = load_data(data_path)

    if target is None:
        target = SmartDefaults.detect_target_column(df)
        if target is None:
            # Heuristic: last column is target
            target = df.columns[-1]

    task = SmartDefaults.detect_task(df[target])
    n_rows, n_features = len(df), len(df.columns) - 1

    return PipelineConfig(
        project="mlpipe-run",
        task=task.value,
        primary_metric=SmartDefaults.default_metric(task),
        level=1,
        data={
            "path": data_path,
            "target": target,
            "test_size": 0.2,
            "random_state": 42,
        },
        model_selection={
            "candidates": "auto",
            "scoring": SmartDefaults.default_scoring(task),
        },
        tuning={
            "n_trials": str(SmartDefaults.allocate_tuning_budget(n_rows, n_features)),
        },
    )


def resolve_config(config: PipelineConfig, df=None) -> PipelineConfig:
    """Resolve 'auto' values using smart defaults."""
    if df is not None and config.task == "auto":
        target = config.data.target
        if target and target in df.columns:
            config.task = SmartDefaults.detect_task(df[target]).value

    if config.task == "auto" and df is not None:
        target = config.data.target or df.columns[-1]
        config.task = SmartDefaults.detect_task(df[target]).value

    if config.primary_metric == "auto":
        task = TaskType(config.task)
        config.primary_metric = SmartDefaults.default_metric(task)

    if df is not None:
        task = TaskType(config.task)
        target = config.data.target
        n_rows = len(df)
        n_features = len(df.columns) - (1 if target and target in df.columns else 0)
        config.model_selection.candidates = SmartDefaults.select_models(task, n_rows, n_features)

    if isinstance(config.tuning.n_trials, str) and config.tuning.n_trials == "auto":
        if df is not None:
            n_rows, n_features = len(df), len(df.columns) - 1
            config.tuning.n_trials = SmartDefaults.allocate_tuning_budget(n_rows, n_features)
        else:
            config.tuning.n_trials = 50

    if isinstance(config.model_selection.ranking_primary, str) and config.model_selection.ranking_primary == "auto":
        config.model_selection.ranking_primary = config.primary_metric

    return config
