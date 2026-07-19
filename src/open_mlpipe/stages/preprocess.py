"""Preprocessor stage — auto-builds ColumnTransformer."""

from __future__ import annotations

import warnings

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler

from open_mlpipe.core.context import PipelineContext
from open_mlpipe.core.stage import Stage


class PreprocessStage(Stage):
    name = "preprocess"
    version = "1.0"

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.X_train is None:
            raise ValueError("ctx.X_train is None — cannot preprocess without training data")
        X_train = ctx.X_train.copy()

        # Build numeric pipelines
        skewed_cols = [c for c in ctx.skewed_columns if c in X_train.columns]
        normal_cols = [c for c in ctx.normal_columns if c in X_train.columns]

        # Catch-all: any numeric column not in skewed or normal
        all_numeric = set(X_train.select_dtypes(include=["number"]).columns)
        accounted_numeric = set(skewed_cols + normal_cols)
        other_numeric = sorted(all_numeric - accounted_numeric)

        transformers = []

        if skewed_cols:
            skewed_pipe = Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("power", PowerTransformer(method="yeo-johnson")),
                ("scale", StandardScaler()),
            ])
            transformers.append(("skewed", skewed_pipe, skewed_cols))

        if normal_cols:
            normal_pipe = Pipeline([
                ("impute", SimpleImputer(strategy="mean")),
                ("scale", StandardScaler()),
            ])
            transformers.append(("normal", normal_pipe, normal_cols))

        if other_numeric:
            other_pipe = Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ])
            transformers.append(("other_num", other_pipe, other_numeric))

        # Build categorical pipelines
        low_cat_cols = [c for c in ctx.low_cardinality_columns if c in X_train.columns]
        high_cat_cols = [c for c in ctx.high_cardinality_columns if c in X_train.columns]

        if low_cat_cols:
            cat_ohe_pipe = Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first")),
            ])
            transformers.append(("cat_ohe", cat_ohe_pipe, low_cat_cols))

        if high_cat_cols:
            try:
                from sklearn.preprocessing import TargetEncoder
                cat_target_pipe = Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("target", TargetEncoder(smooth="auto")),
                ])
                transformers.append(("cat_target", cat_target_pipe, high_cat_cols))
            except ImportError:
                warnings.warn("TargetEncoder not available (sklearn < 1.3), using OHE for high-cardinality")
                cat_ohe_pipe = Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ])
                transformers.append(("cat_ohe_high", cat_ohe_pipe, high_cat_cols))

        if not transformers:
            from sklearn.preprocessing import FunctionTransformer
            transformers.append(("pass", FunctionTransformer(), list(X_train.columns)))

        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="passthrough",
            sparse_threshold=0.3,
        )

        ctx.preprocessor = preprocessor
        return ctx
