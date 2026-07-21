"""FeatureEngineer stage — auto feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd

from open_mlpipe.core.context import PipelineContext
from open_mlpipe.core.stage import Stage
from open_mlpipe.utils.typing import ColumnType


class FeatureEngStage(Stage):
    name = "feature_eng"
    version = "1.0"

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.clean_data is None:
            return ctx
        df = ctx.clean_data.copy()

        # Store raw columns before feature engineering (for inference pipeline)
        ctx.raw_feature_columns = list(df.columns)

        config = ctx.config
        if config is None:
            return ctx

        # Track which columns existed before to detect new ones
        old_cols = set(df.columns)

        # 1. Missingness flags
        if config.feature_engineering.missingness_flags:
            for col in df.columns:
                if df[col].isnull().sum() > 0:
                    df[f"was_missing_{col}"] = df[col].isnull().astype(int)

        # 2. Datetime features
        if config.feature_engineering.datetime_features:
            df = self._extract_datetime_features(df, ctx)

        # 3. Auto interactions (top correlated numeric pairs)
        if config.feature_engineering.auto_interactions:
            df = self._create_interactions(df, ctx)

        # 4. Log transforms for highly skewed features
        if config.feature_engineering.auto_log:
            df = self._log_transforms(df, ctx)

        # Register new engineered columns in ctx column-type tracking
        new_cols = [c for c in df.columns if c not in old_cols]
        for c in new_cols:
            ctx.column_types[c] = ColumnType.NUMERIC
            ctx.numeric_columns.append(c)

        ctx.clean_data = df
        return ctx

    def _extract_datetime_features(self, df, ctx):
        """Extract year, month, day, is_weekend from datetime columns."""
        for col in ctx.datetime_columns:
            if col not in df.columns:
                continue
            try:
                dt = pd.to_datetime(df[col], errors="coerce")
                if dt.notna().sum() > 0:
                    df[f"{col}_year"] = dt.dt.year
                    df[f"{col}_month"] = dt.dt.month
                    df[f"{col}_day"] = dt.dt.day
                    df[f"{col}_dayofweek"] = dt.dt.dayofweek
                    df[f"{col}_is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(int)
                    df = df.drop(columns=[col], errors="ignore")
            except Exception:
                pass
        return df

    def _create_interactions(self, df, ctx):
        """Create interaction features for top correlated numeric pairs."""
        num_cols = [c for c in ctx.numeric_columns if c in df.columns]
        if len(num_cols) < 2:
            return df

        # Use correlation with target to pick top features
        target = ctx.target_column
        if target and target in df.columns and ctx.task_type:
            if ctx.task_type.value == "regression":
                corrs = df[num_cols].corrwith(df[target]).abs().sort_values(ascending=False)
            else:
                # For classification: use mutual information for feature ranking
                try:
                    from sklearn.feature_selection import mutual_info_classif
                    valid_rows = df[num_cols + [target]].dropna()
                    if len(valid_rows) > 10:
                        mi = mutual_info_classif(valid_rows[num_cols], valid_rows[target], random_state=42)
                        corrs = pd.Series(mi, index=num_cols).sort_values(ascending=False)
                    else:
                        corrs = pd.Series(0, index=num_cols)
                except Exception:
                    corrs = pd.Series(0, index=num_cols)
        else:
            corrs = pd.Series(0, index=num_cols)

        top_features = corrs.head(min(5, len(num_cols))).index.tolist()

        pairs_created = 0
        max_pairs = ctx.config.feature_engineering.max_interaction_pairs
        for i in range(len(top_features)):
            for j in range(i + 1, len(top_features)):
                if pairs_created >= max_pairs:
                    break
                a, b = top_features[i], top_features[j]
                if a in df.columns and b in df.columns:
                    df[f"{a}_x_{b}"] = df[a] * df[b]
                    pairs_created += 1

        return df

    def _log_transforms(self, df, ctx):
        """Log-transform highly skewed numeric features."""
        threshold = ctx.config.feature_engineering.log_skew_threshold
        for col in ctx.skewed_columns:
            if col not in df.columns or col == ctx.target_column:
                continue
            if df[col].min() >= 0 and df[col].skew() > threshold:
                df[f"log_{col}"] = np.log1p(df[col])
        return df
