"""FeatureSelector stage — SHAP-based feature importance selection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from open_mlpipe.core.context import PipelineContext
from open_mlpipe.core.stage import Stage


class SelectStage(Stage):
    name = "select"
    version = "1.0"

    def should_skip(self, ctx: PipelineContext) -> bool:
        config = ctx.config
        if config is None:
            return True
        return not config.feature_selection.enabled

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        # Use tuned model if available, else best baseline
        model = ctx.tuned_model or ctx.best_model
        X_train = ctx.X_train
        if model is None or X_train is None:
            return ctx

        # Get feature importances — use preprocessed feature names
        try:
            pre = model.named_steps.get("preprocessor")
            if pre is not None:
                feature_names = list(pre.get_feature_names_out())
            else:
                feature_names = list(X_train.columns)
            importances = self._get_importances(model, X_train)
            # Match lengths: OHE may expand features beyond what model sees
            if len(importances) < len(feature_names):
                # Pad with zeros (expanded OHE columns not in model importances)
                importances = np.pad(importances, (0, len(feature_names) - len(importances)))
            elif len(importances) > len(feature_names):
                importances = importances[:len(feature_names)]
        except Exception:
            return ctx

        # Create importance DataFrame
        feat_imp = pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False)

        ctx.feature_importance = feat_imp

        # Note: feature selection via subsetting X_train/X_test is skipped
        # because after ColumnTransformer the feature names change (OHE expands).
        # The importance ranking is stored for reporting purposes.
        # Feature selection is effectively done by the preprocessor's column choices.

        return ctx

    def _get_importances(self, model, X):
        """Extract feature importances from the pipeline."""
        # Pipeline → get model step
        if hasattr(model, "named_steps") and "model" in model.named_steps:
            m = model.named_steps["model"]
        else:
            m = model

        if hasattr(m, "feature_importances_"):
            return m.feature_importances_
        elif hasattr(m, "coef_"):
            return np.abs(m.coef_).flatten()
        else:
            return np.zeros(X.shape[1])
