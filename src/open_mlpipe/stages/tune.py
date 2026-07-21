"""HyperparameterTuner stage — Optuna-based tuning."""

from __future__ import annotations

import optuna
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from open_mlpipe.config.defaults import SmartDefaults
from open_mlpipe.core.context import PipelineContext
from open_mlpipe.core.stage import Stage
from open_mlpipe.utils.typing import TaskType

# ── Scoring metric map: config ranking_primary → sklearn scoring string ────
_SCORING_MAP = {
    # Regression
    "r2": "r2",
    "neg_root_mean_squared_error": "neg_root_mean_squared_error",
    "neg_mean_absolute_error": "neg_mean_absolute_error",
    "neg_mean_squared_error": "neg_mean_squared_error",
    # Classification
    "f1": "f1_macro",
    "accuracy": "accuracy",
    "roc_auc": "roc_auc",
    "precision": "precision_macro",
    "recall": "recall_macro",
}


def _resolve_scoring(ranking_primary: str, task_type: TaskType) -> str:
    """Resolve the config's ranking_primary to a valid sklearn scoring string."""
    if ranking_primary in _SCORING_MAP:
        return _SCORING_MAP[ranking_primary]
    # Pass through directly — it may already be a valid sklearn scorer name
    return ranking_primary


class TuneStage(Stage):
    name = "tune"
    version = "1.1"

    def should_skip(self, ctx: PipelineContext) -> bool:
        config = ctx.config
        if config is None:
            return True
        return not config.tuning.enabled

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        config = ctx.config
        if config is None:
            return ctx
        task_type = ctx.task_type
        X_train = ctx.X_train
        y_train = ctx.y_train
        preprocessor = ctx.preprocessor
        if (
            ctx.best_model_name is None
            or ctx.best_model is None
            or task_type is None
            or X_train is None
            or y_train is None
            or preprocessor is None
        ):
            return ctx

        # Resolve scoring metric from config
        ranking_primary = config.model_selection.ranking_primary
        scoring_metric = _resolve_scoring(ranking_primary, task_type)

        # Resolve CV splits from config
        n_splits = config.model_selection.cross_validation.n_splits
        if task_type == TaskType.CLASSIFICATION:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        else:
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        # ── Baseline score (untuned model, same CV setup) ──
        # ctx.best_model is a fitted Pipeline from CompareStage.
        # Extract the raw model to avoid double-preprocessing.
        try:
            raw_baseline_model = ctx.best_model.named_steps["model"]
        except (AttributeError, KeyError):
            raw_baseline_model = ctx.best_model
        baseline_pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", raw_baseline_model),
        ])
        try:
            baseline_scores = cross_val_score(
                baseline_pipe, X_train, y_train, cv=cv, scoring=scoring_metric, n_jobs=1)
            baseline_mean = float(baseline_scores.mean())
        except Exception:
            baseline_mean = float("-inf")
        ctx.metrics["tune_baseline_score"] = baseline_mean

        # Get search space for the best model
        search_space = self._get_search_space(ctx.best_model_name, task_type)

        if not search_space:
            return ctx

        # Optuna objective
        def objective(trial):
            params = {}
            for param_name, param_config in search_space.items():
                if param_config["type"] == "int":
                    params[param_name] = trial.suggest_int(
                        param_name, param_config["low"], param_config["high"])
                elif param_config["type"] == "float_log":
                    params[param_name] = trial.suggest_float(
                        param_name, param_config["low"], param_config["high"], log=True)
                elif param_config["type"] == "float":
                    params[param_name] = trial.suggest_float(
                        param_name, param_config["low"], param_config["high"])
                elif param_config["type"] == "categorical":
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_config["choices"])

            # Build model with suggested params
            model = self._build_model(ctx.best_model_name, params, task_type)
            pipe = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model),
            ])

            scores = cross_val_score(
                pipe, X_train, y_train, cv=cv, scoring=scoring_metric, n_jobs=1)
            return scores.mean()

        # Run Optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        n_trials = config.tuning.n_trials
        if isinstance(n_trials, str):
            n_trials = SmartDefaults.allocate_tuning_budget(
                len(X_train), len(X_train.columns))

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(
            objective, n_trials=int(n_trials), timeout=config.tuning.timeout)

        # ── Compare tuned vs baseline — only keep if tuning improved ──
        tuned_best_value = study.best_value
        ctx.metrics["tuned_best_value"] = tuned_best_value
        ctx.metrics["tuned_best_params"] = study.best_params

        if tuned_best_value < baseline_mean - 1e-6:
            print(f"    Tuning did not improve {scoring_metric}: "
                  f"baseline={baseline_mean:.6f}, tuned={tuned_best_value:.6f}. "
                  "Keeping untuned model.")
            return ctx

        # Build and fit the tuned model
        best_model = self._build_model(
            ctx.best_model_name, study.best_params, task_type)
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", best_model),
        ])
        pipe.fit(X_train, y_train)

        ctx.tuned_model = pipe
        print(f"    Tuning improved {scoring_metric}: "
              f"baseline={baseline_mean:.6f} -> tuned={tuned_best_value:.6f}")

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
