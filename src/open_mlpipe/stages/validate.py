"""Pre-split validation — schema integrity, leakage detection, quality gates.

Runs after CleanStage, before SplitStage — catches issues that would
poison CV or inflate scores before any model sees the data.
"""

from __future__ import annotations

import warnings

from open_mlpipe.core.context import PipelineContext
from open_mlpipe.core.stage import Stage


class ValidateStage(Stage):
    name = "validate"
    version = "1.0"

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        config = ctx.config
        df = ctx.clean_data
        if config is None or df is None:
            return ctx

        target = config.data.target or ctx.target_column
        issues: list[str] = []

        # ── Near-duplicate check ──
        dup_pct = df.duplicated().mean() * 100
        if dup_pct > 1:
            issues.append(
                f"{dup_pct:.1f}% duplicate rows detected — "
                "near-duplicates inflate CV if split randomly"
            )

        # ── Leakage: features with |correlation| > 0.95 to target ──
        if target and target in df.columns:
            num_df = df.select_dtypes(include="number")
            if target in num_df.columns:
                target_corr = num_df.corr()[target].drop(target, errors="ignore").abs()
                suspects = target_corr[target_corr > 0.95]  # type: ignore[index, operator]
                if not suspects.empty:  # pyright: ignore[reportAttributeAccessIssue]
                    col_list = suspects.index.tolist()  # pyright: ignore[reportAttributeAccessIssue]
                    issues.append(
                        f"Leakage suspects (|corr| > 0.95 with target): {col_list}"
                    )

        # ── Rare-class check ──
        task = ctx.task_type
        if task is not None and task.value == "classification" and target and target in df.columns:
            counts = df[target].value_counts()
            if counts.min() < 10:
                issues.append(
                    f"Rarest class has {counts.min()} rows — "
                    "StratifiedKFold may drop it from some folds entirely"
                )

        # ── Constant / near-constant columns ──
        for col in df.columns:
            if col == target:
                continue
            n_unique = df[col].nunique()
            if n_unique <= 1:
                issues.append(f"Column '{col}' is constant — carries zero signal")
            elif n_unique == 2 and df[col].value_counts(normalize=True).iloc[0] > 0.98:
                issues.append(
                    f"Column '{col}' is near-constant (98%+ one value)"
                )

        # ── Store and surface ──
        ctx.metrics["validation_issues"] = len(issues)
        ctx.metrics["validation_issue_details"] = issues  # type: ignore[dict-item]

        if issues:
            for i in issues:
                warnings.warn(f"[validate] {i}", UserWarning, stacklevel=2)

        return ctx
