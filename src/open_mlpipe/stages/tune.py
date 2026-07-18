"""HyperparameterTuner stage — Optuna-based tuning."""

from __future__ import annotations

import optuna
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from open_mlpipe.config.defaults import SmartDefaults
from open_mlpipe.core.context import PipelineContext
from open_mlpipe.core.stage import Stage
from open_mlpipe.utils.typing import TaskType


class TuneStage(Stage):
    name = "tune"
    version = "1.0"

    def should_skip(self, ctx: PipelineContext) -> bool:
        return not ctx.config.tuning.enabled

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        config = ctx.config

        if ctx.best_model_name is None or ctx.best_model is None:
            return ctx

        # Get search space for the best model
        search_space = self._get_search_space(ctx.best_model_name, ctx.task_type)

        if not search_space:
            return ctx

        # Optuna objective
        def objective(trial):
            params = {}
            for param_name, param_config in search_space.items():
                if param_config["type"] == "int":
                    params[param_name] = trial.suggest_int(param_name, param_config["low"], param_config["high"])
                elif param_config["type"] == "float_log":
                    params[param_name] = trial.suggest_float(param_name, param_config["low"], param_config["high"], log=True)
                elif param_config["type"] == "float":
                    params[param_name] = trial.suggest_float(param_name, param_config["low"], param_config["high"])

            # Build model with suggested params
            model = self._build_model(ctx.best_model_name, params, ctx.task_type)
            pipe = Pipeline([
                ("preprocessor", ctx.preprocessor),
                ("model", model),
            ])

            task = ctx.task_type
            if task == TaskType.CLASSIFICATION:
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                scoring = "f1_macro"
            else:
                cv = KFold(n_splits=3, shuffle=True, random_state=42)
                scoring = "neg_root_mean_squared_error"

            scores = cross_val_score(pipe, ctx.X_train, ctx.y_train, cv=cv, scoring=scoring, n_jobs=1)
            return scores.mean()

        # Run Optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        n_trials = config.tuning.n_trials
        if isinstance(n_trials, str):
            n_trials = SmartDefaults.allocate_tuning_budget(len(ctx.X_train), len(ctx.X_train.columns))

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(objective, n_trials=int(n_trials), timeout=config.tuning.timeout)

        # Build best model
        best_model = self._build_model(ctx.best_model_name, study.best_params, ctx.task_type)
        pipe = Pipeline([
            ("preprocessor", ctx.preprocessor),
            ("model", best_model),
        ])
        pipe.fit(ctx.X_train, ctx.y_train)

        ctx.tuned_model = pipe
        ctx.metrics["tuned_best_value"] = study.best_value
        ctx.metrics["tuned_best_params"] = study.best_params

        return ctx

    def _get_search_space(self, model_name, task):
        """Return Optuna search space for a model."""
        spaces = {
            "lightgbm": {
                "n_estimators": {"type": "int", "low": 50, "high": 500},
                "learning_rate": {"type": "float_log", "low": 0.01, "high": 0.3},
                "max_depth": {"type": "int", "low": 3, "high": 12},
                "num_leaves": {"type": "int", "low": 20, "high": 100},
                "min_child_samples": {"type": "int", "low": 5, "high": 50},
                "subsample": {"type": "float", "low": 0.5, "high": 1.0},
                "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0},
            },
            "xgboost": {
                "n_estimators": {"type": "int", "low": 50, "high": 500},
                "learning_rate": {"type": "float_log", "low": 0.01, "high": 0.3},
                "max_depth": {"type": "int", "low": 3, "high": 12},
                "min_child_weight": {"type": "int", "low": 1, "high": 10},
                "subsample": {"type": "float", "low": 0.5, "high": 1.0},
                "colsample_bytree": {"type": "float", "low": 0.3, "high": 1.0},
                "reg_alpha": {"type": "float_log", "low": 1e-5, "high": 1.0},
                "reg_lambda": {"type": "float_log", "low": 1e-5, "high": 1.0},
            },
            "random_forest": {
                "n_estimators": {"type": "int", "low": 100, "high": 500},
                "max_depth": {"type": "int", "low": 5, "high": 30},
                "min_samples_split": {"type": "int", "low": 2, "high": 20},
                "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
                "max_features": {"type": "float", "low": 0.3, "high": 1.0},
            },
            "decision_tree": {
                "max_depth": {"type": "int", "low": 3, "high": 20},
                "min_samples_split": {"type": "int", "low": 2, "high": 20},
                "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
                "criterion": {"type": "categorical", "choices": ["gini", "entropy"]},
            },
            "extra_trees": {
                "n_estimators": {"type": "int", "low": 100, "high": 500},
                "max_depth": {"type": "int", "low": 5, "high": 30},
                "min_samples_split": {"type": "int", "low": 2, "high": 20},
                "max_features": {"type": "float", "low": 0.3, "high": 1.0},
            },
            "gradient_boosting": {
                "n_estimators": {"type": "int", "low": 100, "high": 500},
                "learning_rate": {"type": "float_log", "low": 0.01, "high": 0.3},
                "max_depth": {"type": "int", "low": 3, "high": 10},
                "subsample": {"type": "float", "low": 0.6, "high": 1.0},
            },
            "hist_gradient_boosting": {
                "max_iter": {"type": "int", "low": 100, "high": 500},
                "learning_rate": {"type": "float_log", "low": 0.01, "high": 0.3},
                "max_depth": {"type": "int", "low": 3, "high": 12},
                "min_samples_leaf": {"type": "int", "low": 5, "high": 50},
                "l2_regularization": {"type": "float_log", "low": 1e-4, "high": 1.0},
            },
            "catboost": {
                "iterations": {"type": "int", "low": 100, "high": 500},
                "learning_rate": {"type": "float_log", "low": 0.01, "high": 0.3},
                "depth": {"type": "int", "low": 4, "high": 10},
                "l2_leaf_reg": {"type": "float_log", "low": 1e-3, "high": 10.0},
            },
            "adaboost": {
                "n_estimators": {"type": "int", "low": 50, "high": 300},
                "learning_rate": {"type": "float_log", "low": 0.01, "high": 1.0},
            },
            "knn": {
                "n_neighbors": {"type": "int", "low": 1, "high": 20},
                "weights": {"type": "categorical", "choices": ["uniform", "distance"]},
            },
            "svm": {
                "C": {"type": "float_log", "low": 1e-3, "high": 100.0},
                "gamma": {"type": "categorical", "choices": ["scale", "auto"]},
            },
            "logistic_regression": {
                "C": {"type": "float_log", "low": 1e-3, "high": 100.0},
            },
            "ridge": {
                "alpha": {"type": "float_log", "low": 1e-3, "high": 100.0},
            },
            "lasso": {
                "alpha": {"type": "float_log", "low": 1e-4, "high": 10.0},
            },
            "elasticnet": {
                "alpha": {"type": "float_log", "low": 1e-4, "high": 10.0},
                "l1_ratio": {"type": "float", "low": 0.1, "high": 0.9},
            },
        }
        return spaces.get(model_name, {})

    def _build_model(self, model_name, params, task):
        """Build a model instance with given params."""
        from open_mlpipe.stages.compare import CompareStage
        comparator = CompareStage()
        is_cls = task == TaskType.CLASSIFICATION
        model = comparator._build_single(model_name, is_cls)
        if model is not None:
            try:
                # CalibratedClassifierCV wraps SVC — params need estimator__ prefix
                if model.__class__.__name__ == "CalibratedClassifierCV":
                    params = {f"estimator__{k}": v for k, v in params.items()}
                model.set_params(**params)
            except Exception:
                pass  # Some params may not be settable
            return model
        # Fallback
        if is_cls:
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(n_estimators=200, random_state=42)
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=200, random_state=42)
