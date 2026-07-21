"""Shared pytest fixtures for mlpipe tests."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
from typing import cast

import pandas as pd
import pytest

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
from open_mlpipe.core.context import PipelineContext, StageMetadata
from open_mlpipe.core.registry import StageRegistry
from open_mlpipe.stages.eda import EDALoaderStage
from open_mlpipe.stages.load import DataLoaderStage
from open_mlpipe.utils.typing import ColumnType, TaskType

TEST_DIR = Path(__file__).parent
PROJECT_ROOT = TEST_DIR.parent
DATA_FILES = (
    PROJECT_ROOT / "test_datasets"
    if (PROJECT_ROOT / "test_datasets").exists()
    else PROJECT_ROOT / "examples"
)

np.random.seed(42)


# ── Fixture: empty context ────────────────────────────────────────────────

@pytest.fixture
def empty_context() -> PipelineContext:
    return PipelineContext()


# ── Fixtures: config ──────────────────────────────────────────────────────

@pytest.fixture
def base_config() -> PipelineConfig:
    return PipelineConfig(
        project="test-project",
        data=DataConfig(path="test.csv", target="target_col"),
        model_selection=ModelSelectionConfig(
            candidates=["ridge", "random_forest"],
            cross_validation=CVConfig(n_splits=3),
            scoring=["r2"],
            ranking_primary="r2",
        ),
        tuning=TuningConfig(enabled=False),
        feature_selection=FeatureSelectionConfig(enabled=False),
        evaluation=EvaluationConfig(explainability=False),
        deployment=DeploymentConfig(enabled=False),
        artifacts=ArtifactConfig(mlflow_tracking=False),
    )


@pytest.fixture
def base_config_classification() -> PipelineConfig:
    return PipelineConfig(
        project="test-class",
        task="classification",
        primary_metric="f1",
        data=DataConfig(path="test.csv", target="target"),
        model_selection=ModelSelectionConfig(
            candidates=["logistic_regression", "random_forest"],
            cross_validation=CVConfig(n_splits=3, strategy="stratified_kfold"),
            scoring=["f1"],
            ranking_primary="f1",
        ),
        tuning=TuningConfig(enabled=False),
        feature_selection=FeatureSelectionConfig(enabled=False),
        evaluation=EvaluationConfig(explainability=False),
        deployment=DeploymentConfig(enabled=False),
        artifacts=ArtifactConfig(mlflow_tracking=False),
    )


# ── Fixtures: DataFrames ──────────────────────────────────────────────────

@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Mixed-type DataFrame: numeric + categorical + skewed + missing + dupes."""
    n = 200
    df = pd.DataFrame({
        "age":       np.random.normal(45, 14, n).astype(float),
        "height":    np.random.normal(170, 11, n).astype(float),
        "salary":    np.exp(np.random.normal(10, 1, n)).round(2),  # skewed
        "city":      np.random.choice(["NYC", "LA", "Chicago", "Houston", "Miami"], n),
        "gender":    np.random.choice(["M", "F"], n),
        "employed":  np.random.choice([0, 1], n),
        "joined":    pd.date_range("2020-01-01", periods=n, freq="D"),
        "target":    (np.random.random(n) > 0.6).astype(int),
    })
    # Missing values
    df.loc[:8, "age"] = np.nan
    df.loc[:3, "salary"] = np.nan
    df.loc[:4, "city"] = np.nan
    # Duplicates
    df = pd.concat([df, df.head(5)], ignore_index=True)
    return df


@pytest.fixture
def sample_dataframe_regression() -> pd.DataFrame:
    """DataFrame with continuous target for regression tests."""
    n = 200
    X1 = np.random.normal(0, 1, n)
    X2 = np.random.normal(5, 2, n)
    noise = np.random.normal(0, 0.5, n)
    df = pd.DataFrame({
        "feat_a": X1,
        "feat_b": X2,
        "cat_x":  np.random.choice(["Alpha", "Beta", "Gamma"], n),
        "target": X1 * 2.0 + X2 * -1.5 + noise,
    })
    df.loc[:4, "feat_a"] = np.nan
    return df


@pytest.fixture
def sample_dataframe_classification() -> pd.DataFrame:
    """DataFrame with binary target for classification tests."""
    n = 200
    df = pd.DataFrame({
        "x1": np.random.normal(0, 1, n),
        "x2": np.random.normal(3, 2, n),
        "cat_a": np.random.choice(["Yes", "No"], n),
        "cat_b": np.random.choice(["Red", "Blue", "Green"], n),
        "target": np.random.choice([0, 1], n),
    })
    df.loc[:3, "x1"] = np.nan
    return df


@pytest.fixture
def sample_dataframe_all_numeric() -> pd.DataFrame:
    """All-numeric DataFrame, no categoricals."""
    n = 150
    df = pd.DataFrame({f"feat_{i}": np.random.normal(0, 1, n) for i in range(5)})
    df["target"] = df["feat_0"] * 3 + df["feat_1"] * -2 + np.random.normal(0, 0.3, n)
    return df


# ── Fixtures: stage-prefilled contexts ────────────────────────────────────

@pytest.fixture
def loaded_context(
    sample_dataframe_regression: pd.DataFrame, base_config: PipelineConfig
) -> PipelineContext:
    """Context after DataLoaderStage: raw_data, clean_data, column metadata."""
    df = sample_dataframe_regression
    ctx = PipelineContext(raw_data=df, clean_data=df.copy(), config=base_config)
    # ensure config is present for static analyzers / optional config implementations
    if ctx.config is None:
        ctx.config = base_config
    ctx.config.data.target = "target"
    ctx.task_type = TaskType.REGRESSION
    ctx.target_column = "target"
    ctx.column_types = {
        "feat_a": ColumnType.NUMERIC,
        "feat_b": ColumnType.NUMERIC,
        "cat_x": ColumnType.CATEGORICAL,
    }
    ctx.numeric_columns = ["feat_a", "feat_b"]
    ctx.categorical_columns = ["cat_x"]
    ctx.datetime_columns = []
    ctx.skewed_columns = []
    ctx.normal_columns = ["feat_a", "feat_b"]
    ctx.high_cardinality_columns = []
    ctx.low_cardinality_columns = ["cat_x"]
    return ctx


@pytest.fixture
def eda_context(
    loaded_context: PipelineContext, sample_dataframe_regression: pd.DataFrame
) -> PipelineContext:
    """Context after EDALoaderStage: quality report + statistical tests."""
    rng = np.random.default_rng(42)
    ctx = loaded_context
    df = sample_dataframe_regression
    ctx.clean_data = df.copy()
    ctx.eda_report = {
        "quality": {
            "total_rows": len(df),
            "total_cols": len(df.columns),
            "missing": {"feat_a": {"count": 5, "pct": 2.5}},
            "total_missing_rows": 5,
            "total_missing_pct": 0.63,
            "duplicates": 0,
            "constant_columns": [],
            "near_constant_columns": [],
            "outlier_counts": {"feat_a": 0, "feat_b": 0},
            "cardinality": {"cat_x": 3},
        },
        "distributions": {},
        "statistical_tests": pd.DataFrame({
            "feature": ["feat_a", "feat_b", "cat_x"],
            "p_value": [0.01, 0.001, 0.15],
            "significant": [True, True, False],
            "test": ["Pearson", "Pearson", "ANOVA"],
        }),
    }
    # Build a correlation matrix from random data for test use
    corr_rows = rng.normal(0, 1, (len(df.columns), len(df.columns)))
    ctx.correlation_matrix = pd.DataFrame(corr_rows, columns=df.columns, index=df.columns)
    ctx.statistical_tests = pd.DataFrame({
        "feature": ["feat_a", "feat_b", "cat_x"],
        "p_value": [0.01, 0.001, 0.15],
        "significant": [True, True, False],
        "test": ["Pearson", "Pearson", "ANOVA"],
    })
    ctx.columns_to_drop = []
    return ctx


@pytest.fixture
def clean_context(
    loaded_context: PipelineContext, sample_dataframe_regression: pd.DataFrame
) -> PipelineContext:
    """Context after CleanStage: duplicates removed, ID/TEXT/CONSTANT columns dropped."""
    df = sample_dataframe_regression.copy()
    df = df.drop_duplicates().dropna(subset=["target"])
    ctx = loaded_context
    ctx.clean_data = df
    ctx.columns_to_drop = []
    return ctx


@pytest.fixture
def split_context(
    clean_context: PipelineContext, sample_dataframe_regression: pd.DataFrame
) -> PipelineContext:
    """Context after SplitStage: X_train, X_test, y_train, y_test populated."""
    df = clean_context.clean_data
    assert df is not None, "clean_data must be set before split_context"
    X = df.drop(columns=["target"])
    y = df["target"]
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    ctx = clean_context
    ctx.X_train = cast(pd.DataFrame, X_train)
    ctx.X_test = cast(pd.DataFrame, X_test)
    ctx.y_train = cast(pd.Series, y_train)
    ctx.y_test = cast(pd.Series, y_test)
    ctx.numeric_columns = ["feat_a", "feat_b"]
    ctx.categorical_columns = ["cat_x"]
    return ctx


@pytest.fixture
def preprocessed_context(
    split_context: PipelineContext,
) -> PipelineContext:
    """Context after PreprocessStage: has preprocessor fitted on X_train."""
    ctx = split_context
    numeric_pipe = SKPipeline([
        ("impute", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = SKPipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(drop="first", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipe, ctx.numeric_columns),
            ("cat", categorical_pipe, ctx.categorical_columns),
        ],
        remainder="passthrough",
    )
    assert ctx.X_train is not None, "X_train must be set before preprocessor fitting"
    X_train = ctx.X_train
    preprocessor.fit(X_train)
    ctx.preprocessor = preprocessor
    return ctx


@pytest.fixture
def compared_context_regression(
    preprocessed_context: PipelineContext,
) -> PipelineContext:
    """Context after CompareStage (regression): best_model set."""
    ctx = preprocessed_context
    model = SKPipeline([
        ("pre", ctx.preprocessor),
        ("model", Ridge(alpha=1.0)),
    ])
    assert ctx.X_train is not None, "X_train must be set before model fitting"
    X_train = ctx.X_train
    model.fit(X_train, ctx.y_train)
    ctx.baseline_models = {"ridge": model}
    ctx.best_model_name = "ridge"
    ctx.best_model = model
    ctx.metrics["ridge_r2"] = 0.85
    ctx.metrics["ridge_mae"] = 2.3
    ctx.metrics["ridge_rmse"] = 3.1
    return ctx


@pytest.fixture
def tuned_context_regression(
    compared_context_regression: PipelineContext,
) -> PipelineContext:
    """Context after TuneStage: tuned_model set (same as best in tests)."""
    ctx = compared_context_regression
    ctx.tuned_model = ctx.best_model
    ctx.metrics["tuned_best_value"] = 0.87
    ctx.metrics["tuned_best_params_alpha"] = 1.0
    return ctx


# ── Fixture: temp directory ───────────────────────────────────────────────

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    p = Path(mkdtemp())
    yield p
    import shutil
    shutil.rmtree(str(p), ignore_errors=True)


# ── Fixture: StageMetadata ────────────────────────────────────────────────

@pytest.fixture
def sample_metadata() -> StageMetadata:
    return StageMetadata(
        stage_name="test-stage",
        stage_version="1.0",
        input_rows=200,
        input_cols=5,
        output_rows=195,
        output_cols=5,
        parameters={"strategy": "mean"},
        metrics={"r2": 0.88},
        artifacts={"model": "model.joblib"},
        duration_seconds=1.5,
        warnings=["outlier clipped"],
    )


# ── Fixture: StageRegistry ────────────────────────────────────────────────

@pytest.fixture
def populated_registry() -> StageRegistry:
    r = StageRegistry()
    r.register(DataLoaderStage())
    r.register(EDALoaderStage())
    return r


# ── Fixture: existing test dataset paths ──────────────────────────────────

@pytest.fixture
def insurance_csv_path() -> str:
    for candidate in [
        str(PROJECT_ROOT / "test_datasets" / "insurance.csv"),
        str(PROJECT_ROOT / "examples" / "insurance.csv"),
    ]:
        if Path(candidate).exists():
            return candidate
    pytest.skip("insurance.csv not found in test_datasets or examples")


@pytest.fixture
def insurance_dataframe(insurance_csv_path: str) -> pd.DataFrame:
    return pd.read_csv(insurance_csv_path)


# ── Fixture: CSV temp file ────────────────────────────────────────────────

@pytest.fixture
def temp_csv(temp_dir: Path, sample_dataframe_regression: pd.DataFrame) -> str:
    p = temp_dir / "test.csv"
    sample_dataframe_regression.to_csv(p, index=False)
    return str(p)


# ── Fixture: generated test datasets ──────────────────────────────────────

@pytest.fixture
def test_datasets_dir() -> Path:
    d = Path("test_datasets")
    if d.exists():
        return d
    p = PROJECT_ROOT / "test_datasets"
    if p.exists():
        return p
    raise FileNotFoundError("test_datasets/ not found – run generate_test_data.py first")
