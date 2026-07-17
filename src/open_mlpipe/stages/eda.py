"""EDA Engine — comprehensive exploratory data analysis.

Phases from user's Obsidian EDA folder:
1. Column hygiene (strip names, fix dtypes, memory optimize)
2. Data dictionary auto-generation
3. Quality assessment (missing, duplicates, outliers)
4. Distribution analysis (skew, kurtosis, correlation, VIF)
5. Statistical feature importance (Z/T/ANOVA/Chi-Square)
6. Visualization
7. Auto-report (ydata-profiling)
"""

from __future__ import annotations

from open_mlpipe.core.context import PipelineContext
from open_mlpipe.core.stage import Stage


class EDALoaderStage(Stage):
    name = "eda"
    version = "1.0"

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        df = ctx.clean_data.copy()

        # ── Phase 1: Column Hygiene ──
        df = self._clean_column_names(df)
        df = self._optimize_dtypes(df)

        # ── Phase 2: Quality Assessment ──
        quality = self._assess_quality(df, ctx)

        # ── Phase 3: Distribution Analysis ──
        distributions = self._analyze_distributions(df, ctx)

        # ── Phase 4: Statistical Tests ──
        stat_tests = self._run_statistical_tests(df, ctx)

        # ── Phase 5: Correlation + VIF ──
        corr_analysis = self._correlation_analysis(df, ctx)

        # ── Store results ──
        ctx.clean_data = df

        # Update target_column if it was renamed during column hygiene
        if ctx.target_column and ctx.target_column not in df.columns:
            cleaned = ctx.target_column.lower().strip().replace(" ", "_")
            cleaned = __import__("re").sub(r"[^\w]", "", cleaned)
            if cleaned in df.columns:
                ctx.target_column = cleaned

        # Reconcile column_types with cleaned column names
        if ctx.column_types:
            new_types = {}
            for col in df.columns:
                if col in ctx.column_types:
                    new_types[col] = ctx.column_types[col]
                else:
                    for orig_col, ct in ctx.column_types.items():
                        cleaned_orig = orig_col.lower().strip().replace(" ", "_")
                        cleaned_orig = __import__("re").sub(r"[^\w]", "", cleaned_orig)
                        if cleaned_orig == col:
                            new_types[col] = ct
                            break
            ctx.column_types = new_types

        # Reconcile column lists (numeric, categorical, datetime) with cleaned names
        def _reconcile_list(lst, df_cols):
            result = []
            for item in lst:
                if item in df_cols:
                    result.append(item)
                else:
                    cleaned = item.lower().strip().replace(" ", "_")
                    cleaned = __import__("re").sub(r"[^\w]", "", cleaned)
                    if cleaned in df_cols:
                        result.append(cleaned)
            return result

        ctx.numeric_columns = _reconcile_list(ctx.numeric_columns, df.columns)
        ctx.categorical_columns = _reconcile_list(ctx.categorical_columns, df.columns)
        ctx.datetime_columns = _reconcile_list(ctx.datetime_columns, df.columns)
        ctx.skewed_columns = _reconcile_list(ctx.skewed_columns, df.columns)
        ctx.normal_columns = _reconcile_list(ctx.normal_columns, df.columns)
        ctx.high_cardinality_columns = _reconcile_list(ctx.high_cardinality_columns, df.columns)
        ctx.low_cardinality_columns = _reconcile_list(ctx.low_cardinality_columns, df.columns)

        ctx.eda_report = {
            "quality": quality,
            "distributions": distributions,
            "statistical_tests": stat_tests,
            "correlation": corr_analysis,
        }
        if stat_tests is not None:
            ctx.statistical_tests = stat_tests
        if corr_analysis.get("correlation_matrix") is not None:
            ctx.correlation_matrix = corr_analysis["correlation_matrix"]
        if corr_analysis.get("vif") is not None:
            ctx.vif_scores = corr_analysis["vif"]

        # ── Identify columns to drop ──
        ctx.columns_to_drop = quality.get("constant_columns", [])

        return ctx

    # ── Phase 1: Column Hygiene ──

    def _clean_column_names(self, df):
        """Strip, lowercase, replace spaces in column names."""
        df.columns = (
            df.columns.str.strip()
            .str.lower()
            .str.replace(" ", "_")
            .str.replace(r"[^\w]", "", regex=True)
        )
        return df

    def _optimize_dtypes(self, df):
        """Downcast numerics, convert low-cardinality objects to category."""
        import pandas as pd

        # Downcast int64
        for col in df.select_dtypes(include=["int64", "int32"]).columns:
            df[col] = pd.to_numeric(df[col], downcast="integer")

        # Downcast float64
        for col in df.select_dtypes(include=["float64", "float32"]).columns:
            df[col] = pd.to_numeric(df[col], downcast="float")

        # Low-cardinality object → category
        for col in df.select_dtypes(include=["object"]).columns:
            if df[col].nunique() / len(df) < 0.5:
                df[col] = df[col].astype("category")

        return df

    # ── Phase 2: Quality Assessment ──

    def _assess_quality(self, df, ctx):
        """Missing values, duplicates, outliers, constant columns."""

        result = {}

        # Missing values
        missing = df.isnull().sum()
        missing_pct = (df.isnull().mean() * 100).round(2)
        result["missing_count"] = missing[missing > 0].to_dict()
        result["missing_pct"] = missing_pct[missing_pct > 0].to_dict()

        # Duplicates
        result["n_duplicates"] = int(df.duplicated().sum())
        result["duplicate_pct"] = round(df.duplicated().mean() * 100, 2)

        # Constant columns
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        result["constant_columns"] = constant_cols

        # Near-constant (95% same value)
        near_const = []
        for col in df.columns:
            top_pct = df[col].value_counts(normalize=True).iloc[0]
            if top_pct > 0.95:
                near_const.append(col)
        result["near_constant_columns"] = near_const

        # Outliers via IQR
        outlier_info = {}
        num_cols = df.select_dtypes(include=["number"]).columns
        for col in num_cols:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            n_out = int(((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum())
            if n_out > 0:
                outlier_info[col] = {
                    "count": n_out,
                    "pct": round(n_out / len(df) * 100, 2),
                }
        result["outliers"] = outlier_info

        # Cardinality for categoricals
        cardinality = {}
        for col in df.select_dtypes(include=["object", "category"]).columns:
            cardinality[col] = int(df[col].nunique())
        result["cardinality"] = cardinality

        return result

    # ── Phase 3: Distribution Analysis ──

    def _analyze_distributions(self, df, ctx):
        """Skewness, kurtosis per numeric feature."""

        num_cols = df.select_dtypes(include=["number"]).columns
        if len(num_cols) == 0:
            return {}

        skew = df[num_cols].skew()
        kurt = df[num_cols].kurtosis()

        return {
            "skewness": skew.sort_values(key=abs, ascending=False).to_dict(),
            "kurtosis": kurt.sort_values(key=abs, ascending=False).to_dict(),
            "highly_skewed": [col for col, s in skew.items() if abs(s) > 1.0],
            "moderately_skewed": [col for col, s in skew.items() if 0.5 < abs(s) <= 1.0],
        }

    # ── Phase 4: Statistical Tests ──

    def _run_statistical_tests(self, df, ctx):
        """Z/T/ANOVA/Chi-Square for feature importance."""
        import pandas as pd
        from scipy import stats

        target = ctx.target_column
        if target is None or target not in df.columns:
            return None

        results = []
        task = ctx.task_type

        num_cols = df.select_dtypes(include=["number"]).columns.drop(target, errors="ignore")
        cat_cols = df.select_dtypes(include=["object", "category"]).columns

        if task and task.value == "regression":
            # Numeric features: Pearson r
            for col in num_cols:
                valid = df[[col, target]].dropna()
                if len(valid) < 10:
                    continue
                r, p = stats.pearsonr(valid[col], valid[target])
                results.append({
                    "feature": col,
                    "test": "pearsonr",
                    "statistic": round(r, 4),
                    "p_value": round(p, 6),
                    "significant": p < 0.05,
                    "direction": "positive" if r > 0 else "negative",
                })

            # Categorical features: ANOVA (3+ groups) or T-Test (2 groups)
            for col in cat_cols:
                groups = [g[target].dropna().values for _, g in df.groupby(col, observed=False) if len(g) > 1]
                if len(groups) < 2:
                    continue
                if len(groups) == 2:
                    t, p = stats.ttest_ind(*groups, equal_var=False)
                    test_name = "ttest_ind"
                else:
                    f, p = stats.f_oneway(*groups)
                    t = f
                    test_name = "anova"
                results.append({
                    "feature": col,
                    "test": test_name,
                    "statistic": round(float(t), 4),
                    "p_value": round(float(p), 6),
                    "significant": float(p) < 0.05,
                })

        elif task and task.value == "classification":
            # Numeric features: ANOVA (3+ classes) or T-Test (2 classes)
            for col in num_cols:
                groups = [g[col].dropna().values for _, g in df.groupby(target) if len(g) > 1]
                if len(groups) < 2:
                    continue
                if len(groups) == 2:
                    t, p = stats.ttest_ind(*groups, equal_var=False)
                    test_name = "ttest_ind"
                else:
                    f_stat, p = stats.f_oneway(*groups)
                    t = f_stat
                    test_name = "anova"
                results.append({
                    "feature": col,
                    "test": test_name,
                    "statistic": round(float(t), 4),
                    "p_value": round(float(p), 6),
                    "significant": float(p) < 0.05,
                })

            # Categorical features: Chi-Square
            for col in cat_cols:
                contingency = pd.crosstab(df[col], df[target])
                if contingency.shape == (1, 1) or contingency.shape[0] < 2 or contingency.shape[1] < 2:
                    continue
                chi2, p, dof, expected = stats.chi2_contingency(contingency)
                results.append({
                    "feature": col,
                    "test": "chi2_contingency",
                    "statistic": round(chi2, 4),
                    "p_value": round(p, 6),
                    "significant": p < 0.05,
                    "dof": dof,
                })

        if not results:
            return None

        return pd.DataFrame(results).sort_values("p_value")

    # ── Phase 5: Correlation + VIF ──

    def _correlation_analysis(self, df, ctx):
        """Pearson correlation matrix + VIF for multicollinearity."""
        import pandas as pd

        result = {}

        num_cols = df.select_dtypes(include=["number"]).columns
        if len(num_cols) < 2:
            return result

        # Correlation matrix
        corr = df[num_cols].corr()
        result["correlation_matrix"] = corr

        # High correlation pairs
        high_corr = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                c = abs(corr.iloc[i, j])
                if c > 0.9:
                    high_corr.append({
                        "feature_1": corr.columns[i],
                        "feature_2": corr.columns[j],
                        "correlation": round(c, 4),
                    })
        result["high_correlation_pairs"] = high_corr

        # VIF (Variance Inflation Factor)
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor

            X_numeric = df[num_cols].dropna()
            if len(X_numeric) > 10 and len(num_cols) >= 2:
                X_with_const = X_numeric.assign(const=1)
                vif_data = pd.DataFrame({
                    "feature": X_with_const.columns,
                    "VIF": [
                        variance_inflation_factor(X_with_const.values, i)
                        for i in range(X_with_const.shape[1])
                    ],
                })
                vif_data = vif_data[vif_data["feature"] != "const"]
                vif_data = vif_data.sort_values("VIF", ascending=False)
                result["vif"] = vif_data
        except ImportError:
            pass

        return result
