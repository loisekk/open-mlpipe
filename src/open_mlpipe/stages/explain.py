"""Explainer stage — SHAP feature importance."""

from __future__ import annotations

from open_mlpipe.core.context import PipelineContext
from open_mlpipe.core.stage import Stage


class ExplainStage(Stage):
    name = "explain"
    version = "1.0"

    def should_skip(self, ctx: PipelineContext) -> bool:
        return not ctx.config.evaluation.explainability

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        model = ctx.tuned_model or ctx.best_model
        if model is None:
            return ctx

        try:
            import shap

            # Get model from pipeline
            if hasattr(model, "named_steps") and "model" in model.named_steps:
                m = model.named_steps["model"]
                pre = model.named_steps["preprocessor"]
                X_test_trans = pre.transform(ctx.X_test)
            else:
                m = model
                X_test_trans = ctx.X_test.values

            # Use TreeExplainer for tree models
            if hasattr(m, "feature_importances_"):
                explainer = shap.TreeExplainer(m)
                shap_values = explainer.shap_values(X_test_trans)

                # Store for later reporting
                ctx.reports["shap_values"] = shap_values
                ctx.reports["shap_expected_value"] = explainer.expected_value
        except Exception as e:
            print(f"    SHAP analysis skipped: {e}")

        return ctx
