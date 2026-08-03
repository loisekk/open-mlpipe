"""DataSplitter stage — stratified or random train/test split."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from open_mlpipe.core.context import PipelineContext
from open_mlpipe.core.stage import Stage
from open_mlpipe.utils.typing import TaskType


class SplitStage(Stage):
    name = "split"
    version = "1.0"

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        df = ctx.clean_data
        config = ctx.config
        target = ctx.target_column
        task_type = ctx.task_type
        if df is None or config is None or target is None or task_type is None:
            return ctx

        X = df.drop(columns=[target])
        y = df[target]

        # Encode target if classification and string/category
        y_dtype = str(y.dtype)
        if task_type == TaskType.CLASSIFICATION and y_dtype in ("object", "category"):
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            encoded = le.fit_transform(y.to_numpy())
            y = pd.Series(encoded, name=target, dtype="int64")  # type: ignore[arg-type]
            ctx._label_encoder = le

        # Determine stratification
        stratify = None
        if task_type == TaskType.CLASSIFICATION:
            stratify = y
        elif task_type == TaskType.REGRESSION:
            # Stratify regression by quantile-binning the target so the
            # train/test split preserves the target distribution. Without
            # this, a skewed target (e.g. price skew=24.79) puts extreme
            # outliers only in train OR test -- the model never sees them
            # during training and predicts garbage on test (test_r2 crashes
            # from 0.63 CV to 0.05). This is the #1 accuracy lever for
            # heavy-tail regression.
            import warnings as _w

            import numpy as np

            def _safe_skew(s):
                """Skew that survives NaN/inf and non-numeric dtypes."""
                with _w.catch_warnings():
                    _w.simplefilter("ignore")
                    if not pd.api.types.is_numeric_dtype(s):
                        return 0.0
                    s_clean = s.dropna()
                    if len(s_clean) < 3:
                        return 0.0
                    return float(s_clean.skew())

            skew_val = abs(_safe_skew(y))
            if skew_val > 1.0:
                n_bins = min(10, max(3, int(np.sqrt(len(y)))))
                with _w.catch_warnings():
                    _w.simplefilter("ignore")
                    stratify = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")
                if stratify is not None and stratify.nunique() >= 2:
                    print(f"    Stratified split on target (skew={skew_val:.2f}, "
                          f"{n_bins} quantile bins)")
                else:
                    stratify = None  # degenerate bins -- fall back to random

        test_size = config.data.test_size
        if len(df) < 500:
            test_size = 0.1

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=config.data.random_state,
            stratify=stratify,
        )

        ctx.X_train = X_train  # type: ignore[assignment]
        ctx.X_test = X_test  # type: ignore[assignment]
        ctx.y_train = y_train  # type: ignore[assignment]
        ctx.y_test = y_test  # type: ignore[assignment]

        # Update column types after split
        ctx.numeric_columns = [c for c in ctx.numeric_columns if c in X_train.columns]  # type: ignore[union-attr]
        ctx.categorical_columns = [c for c in ctx.categorical_columns if c in X_train.columns]  # type: ignore[union-attr]

        return ctx
