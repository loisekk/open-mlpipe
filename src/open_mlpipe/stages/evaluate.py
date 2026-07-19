"""Evaluator stage — final test set evaluation with production metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from open_mlpipe.core.context import PipelineContext
from open_mlpipe.core.stage import Stage
from open_mlpipe.utils.typing import TaskType


class EvaluateStage(Stage):
    name = "evaluate"
    version = "1.0"

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        model = ctx.tuned_model or ctx.best_model
        if model is None:
            return ctx

        X_test = ctx.X_test
        y_test = ctx.y_test
        if X_test is None or y_test is None:
            return ctx

        y_pred = model.predict(X_test)
        task = ctx.task_type
        if task is None:
            return ctx

        if task == TaskType.CLASSIFICATION:
            self._eval_classification(ctx, y_test, y_pred, model, X_test)
        else:
            self._eval_regression(ctx, y_test, y_pred)

        return ctx

    def _eval_classification(self, ctx, y_test, y_pred, model, X_test):
        ctx.metrics["test_accuracy"] = accuracy_score(y_test, y_pred)
        ctx.metrics["test_f1_macro"] = f1_score(y_test, y_pred, average="macro")
        ctx.metrics["test_f1_weighted"] = f1_score(y_test, y_pred, average="weighted")
        ctx.metrics["test_mcc"] = matthews_corrcoef(y_test, y_pred)

        try:
            y_prob = model.predict_proba(X_test)
            if y_prob.shape[1] == 2:
                roc = roc_auc_score(y_test, y_prob[:, 1])
                if not np.isnan(roc):
                    ctx.metrics["test_roc_auc"] = float(roc)
            else:
                roc = roc_auc_score(y_test, y_prob, multi_class="ovr")
                if not np.isnan(roc):
                    ctx.metrics["test_roc_auc"] = float(roc)
        except Exception:
            pass

        ctx.metrics["confusion_matrix"] = confusion_matrix(
            y_test, y_pred, labels=sorted(set(y_test) | set(y_pred))
        ).tolist()

    def _eval_regression(self, ctx, y_test, y_pred):
        ctx.metrics["test_rmse"] = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        ctx.metrics["test_mae"] = float(mean_absolute_error(y_test, y_pred))
        ctx.metrics["test_r2"] = float(r2_score(y_test, y_pred))
        ctx.metrics["test_mape"] = float(mean_absolute_percentage_error(y_test, y_pred) * 100)
