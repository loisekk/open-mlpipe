"""Smart defaults engine — makes all pipeline decisions automatically."""

from __future__ import annotations

import warnings

import pandas as pd

from open_mlpipe.utils.typing import (
    ColumnType,
    EncoderType,
    MissingStrategy,
    ScalerType,
    TaskType,
)


class SmartDefaults:
    """Makes pipeline decisions automatically based on data characteristics."""

    # ── Task Detection ──

    @staticmethod
    def detect_task(target: pd.Series) -> TaskType:
        """Auto-detect classification vs regression from target column."""
        dtype_str = str(target.dtype)
        if dtype_str == "object" or dtype_str == "category":
            return TaskType.CLASSIFICATION
        if target.nunique() < 20 and target.nunique() / len(target) < 0.05:
            return TaskType.CLASSIFICATION
        if dtype_str in ("int64", "int32") and target.nunique() < 20:
            return TaskType.CLASSIFICATION
        return TaskType.REGRESSION

    @staticmethod
    def detect_target_column(df: pd.DataFrame) -> str | None:
        """Guess the target column by name conventions."""
        candidates = ["target", "label", "y", "class", "output", "outcome"]
        for col in df.columns:
            if col.lower().strip() in candidates:
                return col
        return None

    # ── Column Type Detection ──

    @staticmethod
    def detect_column_types(df: pd.DataFrame) -> dict[str, ColumnType]:
        """Auto-classify every column."""
        result = {}
        for col in df.columns:
            result[col] = SmartDefaults._classify_column(df, col)
        return result

    @staticmethod
    def _classify_column(df: pd.DataFrame, col: str) -> ColumnType:
        """Classify a single column."""
        series = df[col]
        nunique = series.nunique()
        total = len(series)

        # Constant
        if nunique <= 1:
            return ColumnType.CONSTANT

        # ID-like: high cardinality AND looks like an ID (name pattern or monotonically increasing)
        id_patterns = {"id", "key", "uuid", "index", "row", "record", "pk"}
        name_looks_like_id = any(p in col.lower() for p in id_patterns)

        # Monotonically increasing = likely auto-increment ID
        dtype_str = str(series.dtype)
        if dtype_str in ("float64", "int64"):
            diffs = series.dropna().diff().dropna()
            is_monotonic = len(diffs) > 0 and ((diffs > 0).all() or (diffs < 0).all())
        else:
            is_monotonic = False

        # ID-like if: name matches pattern + high cardinality, OR monotonically increasing + high cardinality
        if nunique > total * 0.8 and (name_looks_like_id or is_monotonic):
            return ColumnType.ID_LIKE

        # Boolean
        if nunique == 2 and set(series.dropna().unique()).issubset({True, False, 0, 1, "yes", "no", "True", "False"}):
            return ColumnType.BOOLEAN

        # Datetime
        if dtype_str == "datetime64[ns]":
            return ColumnType.DATETIME
        if dtype_str == "object":
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    pd.to_datetime(series.head(20), errors="raise", format=None)
                return ColumnType.DATETIME
            except (ValueError, TypeError):
                pass

        # Numeric
        if dtype_str in ("float64", "int64"):
            return ColumnType.NUMERIC

        # Categorical: string with low-to-moderate cardinality
        if dtype_str == "object" or dtype_str == "category":
            if nunique < total * 0.5:
                return ColumnType.CATEGORICAL
            return ColumnType.TEXT

        return ColumnType.CATEGORICAL

    # ── Missing Strategy ──

    @staticmethod
    def select_missing_strategy(
        col: pd.Series, col_type: ColumnType, n_rows: int
    ) -> MissingStrategy:
        """Pick imputation strategy per column."""
        missing_pct = col.isnull().mean() * 100

        if missing_pct == 0:
            return MissingStrategy.MEAN  # no-op

        if missing_pct > 40:
            return MissingStrategy.MEAN  # will be dropped

        if col_type in (ColumnType.NUMERIC, ColumnType.BOOLEAN):
            if missing_pct < 5:
                return MissingStrategy.MEDIAN
            if missing_pct < 30 and n_rows < 10_000:
                return MissingStrategy.KNN
            return MissingStrategy.MEDIAN

        if col_type == ColumnType.CATEGORICAL:
            return MissingStrategy.MODE

        return MissingStrategy.MODE

    # ── Scaler Selection ──

    @staticmethod
    def select_numeric_scaler(
        col: pd.Series, model_family: str = "tree"
    ) -> ScalerType:
        """Pick scaling strategy per numeric column."""
        if model_family in ("tree", "forest", "boosting"):
            return ScalerType.NONE  # trees are scale-invariant

        skew = float(col.skew())  # type: ignore[arg-type]

        if skew > 1.0:
            return ScalerType.POWER
        q1, q3 = col.quantile(0.25), col.quantile(0.75)
        iqr = q3 - q1
        n_outliers = ((col < q1 - 1.5 * iqr) | (col > q3 + 1.5 * iqr)).sum()
        if n_outliers / len(col) > 0.05:
            return ScalerType.ROBUST
        return ScalerType.STANDARD

    # ── Encoder Selection ──

    @staticmethod
    def select_categorical_encoder(
        col: pd.Series, n_categories: int
    ) -> EncoderType:
        """Pick encoding strategy per categorical column."""
        if n_categories == 2:
            return EncoderType.LABEL
        if n_categories <= 15:
            return EncoderType.ONEHOT
        if n_categories <= 50:
            return EncoderType.TARGET
        return EncoderType.FREQUENCY

    # ── Model Selection ──

    @staticmethod
    def select_models(
        task: TaskType, n_rows: int, n_features: int, n_features_encoded: int | None = None
    ) -> list[str]:
        """Pick model candidates based on data characteristics.

        Uses rows-per-feature ratio as the primary signal — pure row-count
        thresholds are blind to dimensionality explosion from OHE.

        Research-backed (Feldman 2020, Kösters 2024, TabArena 2025):
        - Tree models beat distance-based on mixed-type tabular data
        - KNN excluded from defaults (curse of dimensionality with OHE)
        - SVM/gaussian kernel models need adequate sample density (ratio > 20)
        - Complex tree/boosting needs enough rows per feature to avoid memorization
        """
        n_feat = max(n_features_encoded or n_features, 1)
        ratio = n_rows / n_feat
        models: list[str] = []

        if task == TaskType.CLASSIFICATION:
            models.append("logistic_regression")

            # SVM needs dense coverage of input space
            if ratio > 20:
                models.append("svm")

            # Tree ensembles need enough rows per feature
            if n_rows >= 1000 and ratio > 10:
                models.extend(["random_forest", "lightgbm", "xgboost"])

            if n_rows >= 5000 and ratio > 30:
                models.append("catboost")
        else:
            models.append("ridge")

            if ratio > 20:
                models.append("svm")

            if n_rows >= 1000 and ratio > 10:
                models.extend(["random_forest", "lightgbm", "xgboost"])

            if n_rows >= 5000 and ratio > 30:
                models.append("catboost")

        return models

    # ── Tuning Budget ──

    @staticmethod
    def allocate_tuning_budget(n_rows: int, n_features: int) -> int:
        """Pick Optuna trial count based on data complexity."""
        base = min(n_features * 2, 50)
        if n_rows < 1_000:
            return max(20, base)
        if n_rows < 10_000:
            return max(30, base)
        if n_rows < 100_000:
            return max(50, base)
        return max(80, base)

    # ── Metric Selection ──

    @staticmethod
    def default_metric(task: TaskType) -> str:
        if task == TaskType.CLASSIFICATION:
            return "f1_macro"
        return "neg_root_mean_squared_error"

    @staticmethod
    def default_scoring(task: TaskType) -> list[str]:
        if task == TaskType.CLASSIFICATION:
            return ["accuracy", "f1_macro", "roc_auc_ovr"]
        return ["neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"]

    # ── Skew Detection ──

    @staticmethod
    def detect_skewed_columns(df: pd.DataFrame, threshold: float = 0.5) -> tuple[list[str], list[str]]:
        """Return (skewed_cols, normal_cols) based on skewness threshold."""
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        skewed, normal = [], []
        for col in num_cols:
            skew_val = float(abs(df[col].skew()))
            if skew_val > threshold:
                skewed.append(col)
            else:
                normal.append(col)
        return skewed, normal
