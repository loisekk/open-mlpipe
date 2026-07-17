"""ModelSaver stage — save pipeline to disk + MLflow logging."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from open_mlpipe.core.context import PipelineContext
from open_mlpipe.core.stage import Stage


class SaveStage(Stage):
    name = "save"
    version = "1.0"

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        config = ctx.config
        output_dir = Path(config.artifacts.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        model = ctx.tuned_model or ctx.best_model
        if model is None:
            return ctx

        # Build full inference pipeline: feature_engineering + model
        from sklearn.pipeline import Pipeline as SKPipeline

        from open_mlpipe.utils.feature_eng_transformer import FeatureEngTransformer

        fe_transformer = FeatureEngTransformer()

        # Fit on raw data (before feature engineering) so it learns to create
        # interaction columns, missingness flags, etc. from scratch
        if ctx.raw_feature_columns:
            # Exclude target column (not in X_train)
            raw_cols = [c for c in ctx.raw_feature_columns if c != ctx.target_column and c in ctx.X_train.columns]
            raw_df = ctx.X_train[raw_cols].copy()
        else:
            raw_df = ctx.X_train.copy()
        fe_transformer.fit(raw_df, ctx.y_train)

        full_pipeline = SKPipeline([
            ("feature_eng", fe_transformer),
            ("model", model),
        ])

        # Save full pipeline
        model_path = output_dir / f"model_v1.{config.artifacts.save_format}"
        joblib.dump(full_pipeline, model_path, compress=config.artifacts.compress)
        ctx.reports["model_path"] = str(model_path)

        # Save metadata
        metadata = {
            "project": config.project,
            "task": ctx.task_type.value if ctx.task_type else None,
            "target": ctx.target_column,
            "best_model": ctx.best_model_name,
            "metrics": {k: v for k, v in ctx.metrics.items()
                       if k != "confusion_matrix" and not isinstance(v, dict)
                       and not (isinstance(v, float) and (np.isnan(v) or np.isinf(v)))},
            "confusion_matrix": ctx.metrics.get("confusion_matrix"),
            "n_features": len(ctx.X_train.columns) if ctx.X_train is not None else 0,
            "n_train": len(ctx.X_train) if ctx.X_train is not None else 0,
            "n_test": len(ctx.X_test) if ctx.X_test is not None else 0,
        }
        meta_path = output_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # MLflow logging
        if config.artifacts.mlflow_tracking:
            self._log_mlflow(ctx, model, metadata)

        return ctx

    def _log_mlflow(self, ctx, model, metadata):
        """Log to MLflow."""
        try:
            import mlflow
            import mlflow.sklearn

            experiment_name = ctx.config.artifacts.mlflow_experiment
            if experiment_name == "auto":
                experiment_name = ctx.config.project

            mlflow.set_experiment(experiment_name)

            with mlflow.start_run(run_name=f"{ctx.best_model_name}-v1"):
                mlflow.log_param("model", ctx.best_model_name)
                mlflow.log_param("task", metadata["task"])
                mlflow.log_param("n_features", metadata["n_features"])
                mlflow.log_param("n_train", metadata["n_train"])

                for k, v in metadata["metrics"].items():
                    if isinstance(v, int | float):
                        mlflow.log_metric(k, v)

                mlflow.sklearn.log_model(model, "model")
                ctx.reports["mlflow_run"] = True
        except Exception as e:
            print(f"    MLflow logging skipped: {e}")
