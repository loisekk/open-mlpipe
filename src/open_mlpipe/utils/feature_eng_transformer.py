"""FeatureEngTransformer — sklearn-compatible transformer for inference."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngTransformer(BaseEstimator, TransformerMixin):
    """Apply feature engineering at inference time.

    Remembers which columns had missing values, which interactions to create,
    and which columns to log-transform — all fitted during training.
    """

    def __init__(self):
        self.missing_cols_: list[str] = []
        self.interaction_pairs_: list[tuple[str, str]] = []
        self.log_cols_: list[str] = []
        self.columns_in_: list[str] = []

    def fit(self, X, y=None):
        df = pd.DataFrame(X).copy() if not isinstance(X, pd.DataFrame) else X.copy()
        self.columns_in_ = list(df.columns)

        # Build lowercase→original mapping for case-insensitive matching
        self._col_map_ = {c.lower(): c for c in df.columns}

        # Detect columns with missing values
        self.missing_cols_ = [c for c in df.columns if df[c].isnull().sum() > 0]

        # Detect interaction pairs (top correlated numeric pairs)
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if len(num_cols) >= 2 and y is not None:
            try:
                corrs = {}
                for c in num_cols:
                    valid = df[[c]].dropna()
                    if len(valid) > 10:
                        corrs[c] = abs(np.corrcoef(df[c].fillna(0).values, np.asarray(y).ravel())[0, 1])
                ranked = sorted(corrs, key=corrs.get, reverse=True)[:5]
                pairs = []
                for i in range(len(ranked)):
                    for j in range(i + 1, min(len(ranked), 5)):
                        pairs.append((ranked[i], ranked[j]))
                        if len(pairs) >= 10:
                            break
                    if len(pairs) >= 10:
                        break
                self.interaction_pairs_ = pairs
            except Exception:
                self.interaction_pairs_ = []

        # Detect skewed columns for log transform
        for c in num_cols:
            if c in df.columns:
                try:
                    s = df[c].dropna()
                    if s.min() >= 0 and s.skew() > 1.0:
                        self.log_cols_.append(c)
                except Exception:
                    pass

        return self

    def _resolve_col(self, name, df):
        """Resolve column name case-insensitively."""
        if name in df.columns:
            return name
        lower_map = {c.lower(): c for c in df.columns}
        return lower_map.get(name.lower())

    def transform(self, X):
        df = pd.DataFrame(X).copy() if not isinstance(X, pd.DataFrame) else X.copy()

        # Normalize column names to lowercase (matches training data after EDA)
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('[^a-zA-Z0-9_]', '', regex=True)

        # 1. Missingness flags
        for col in self.missing_cols_:
            real = self._resolve_col(col, df)
            if real is not None:
                df[f"was_missing_{real}"] = df[real].isnull().astype(int)

        # 2. Interaction features
        for a, b in self.interaction_pairs_:
            ra = self._resolve_col(a, df)
            rb = self._resolve_col(b, df)
            if ra is not None and rb is not None:
                df[f"{ra}_x_{rb}"] = df[ra] * df[rb]

        # 3. Log transforms
        for col in self.log_cols_:
            real = self._resolve_col(col, df)
            if real is not None:
                df[f"log_{real}"] = np.log1p(df[real].clip(lower=0))

        return df
