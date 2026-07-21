"""Explainer stage — SHAP feature importance."""

from __future__ import annotations

from open_mlpipe.core.context import PipelineContext
from open_mlpipe.core.stage import Stage


class ExplainStage(Stage):
    name = "explain"
    version = "1.0"

    def should_skip(self, ctx: PipelineContext) -> bool:
        config = ctx.config
        if config is None:
            return True
        return not config.evaluation.explainability

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        model = ctx.tuned_model or ctx.best_model
        X_test = ctx.X_test
        if model is None or X_test is None:
            return ctx

        try:
            import shap

            # Get model from pipeline
            if hasattr(model, "named_steps") and "model" in model.named_steps:
                m = model.named_steps["model"]
                pre = model.named_steps.get("preprocessor")
                X_test_trans = pre.transform(X_test) if pre is not None else X_test.values
            else:
                m = model
                X_test_trans = X_test.values

            # Subsample for speed — SHAP is O(n²) on full data
            import numpy as np
            max_shap_rows = 1000
            if hasattr(X_test_trans, "shape") and X_test_trans.shape[0] > max_shap_rows:
                rng = np.random.RandomState(42)
                idx = rng.choice(X_test_trans.shape[0], max_shap_rows, replace=False)
                X_shap = X_test_trans[idx]
            else:
                X_shap = X_test_trans

            # Use TreeExplainer for tree models
            if hasattr(m, "feature_importances_"):
                explainer = shap.TreeExplainer(m)
                shap_values = explainer.shap_values(X_shap)

                # Store for later reporting
                ctx.reports["shap_values"] = shap_values
                ctx.reports["shap_expected_value"] = explainer.expected_value
        except Exception as e:
            print(f"    SHAP analysis skipped: {e}")

        return ctx
