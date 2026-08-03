"""DataCleaner stage — applies cleaning actions based on EDA findings."""

from __future__ import annotations

from open_mlpipe.core.context import PipelineContext
from open_mlpipe.core.stage import Stage


class CleanStage(Stage):
    name = "clean"
    version = "1.0"

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        df = ctx.clean_data
        config = ctx.config
        if df is None or config is None:
            return ctx
        df = df.copy()

        # 1. Remove duplicates
        if config.cleaning.remove_duplicates:
            before = len(df)
            df = df.drop_duplicates().reset_index(drop=True)
            n_duped = before - len(df)
            if n_duped > 0:
                print(f"    Removed {n_duped} duplicate rows")

        # 2. Drop target nulls
        if ctx.target_column and ctx.target_column in df.columns:
            before = len(df)
            df = df.dropna(subset=[ctx.target_column])
            n_nulls = before - len(df)
            if n_nulls > 0:
                print(f"    Dropped {n_nulls} rows with null target")

        # 3. Drop constant columns (from EDA)
        if ctx.columns_to_drop:
            df = df.drop(columns=ctx.columns_to_drop, errors="ignore")
            print(f"    Dropped {len(ctx.columns_to_drop)} constant columns: {ctx.columns_to_drop}")

        # 4. Drop ID-like columns
        id_cols = [col for col, ct in ctx.column_types.items() if ct.value == "id_like" and col in df.columns]
        if id_cols:
            df = df.drop(columns=id_cols, errors="ignore")
            print(f"    Dropped {len(id_cols)} ID-like columns: {id_cols}")

        # 5. Drop TEXT columns (high cardinality strings that can't be encoded)
        text_cols = [col for col, ct in ctx.column_types.items() if ct.value == "text" and col in df.columns]
        if text_cols:
            df = df.drop(columns=text_cols, errors="ignore")
            print(f"    Dropped {len(text_cols)} text columns: {text_cols}")

        # 6. Outlier treatment
        if config.cleaning.outlier_method != "none":
            df = self._handle_outliers(df, config, ctx.target_column)

        ctx.clean_data = df
        return ctx

    def _handle_outliers(self, df, config, target_col=None):
        """Clip outliers based on IQR method using config's threshold.

        For regression targets, also clips suspicious zero/near-zero values
        (which are usually missing data encoded as 0 — e.g. 49 houses with
        price=0 in a real-estate dataset) and extreme outliers. Without this,
        price=0 rows tank MAPE (division by zero) and 26.6M outliers
        dominate MSE, making the model predict garbage.
        """
        method = config.cleaning.outlier_method
        action = config.cleaning.outlier_action
        threshold = config.cleaning.outlier_threshold

        # Handle target outliers for regression (zero-encoded-missing + extreme)
        if target_col and target_col in df.columns:
            import numpy as np
            if df[target_col].dtype.kind in "biufc":
                skew_val = float(np.abs(df[target_col].skew()))
                if skew_val > 1.0:
                    q_lo = float(df[target_col].quantile(0.01))
                    q_hi = float(df[target_col].quantile(0.99))
                    n_zero = int((df[target_col] <= 0).sum())
                    n_outlier = int((df[target_col] > q_hi).sum() + (df[target_col] < q_lo).sum())
                    if n_zero > 0 or n_outlier > 0:
                        before = len(df)
                        mask = (df[target_col] > 0) & (df[target_col] <= q_hi)
                        df = df.loc[mask].copy()
                        n_dropped = before - len(df)
                        if n_dropped > 0:
                            print(f"    Target outlier cleanup: dropped {n_dropped} rows "
                                  f"(zero/targets: {n_zero}, >99th pct: {n_outlier}, "
                                  f"skew was {skew_val:.2f})")

        if method == "auto" or method == "iqr":
            num_cols = df.select_dtypes(include=["number"]).columns
            for col in num_cols:
                if col == target_col:
                    continue
                Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR

                if action == "clip":
                    df.loc[:, col] = df[col].clip(lower=lower, upper=upper)
                elif action == "remove":
                    df = df[(df[col] >= lower) & (df[col] <= upper)].copy()

        return df
