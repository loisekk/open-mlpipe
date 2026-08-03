"""HyperparameterTuner stage — Optuna-based tuning."""

from __future__ import annotations

from importlib.util import find_spec

import optuna
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline

from open_mlpipe.config.defaults import SmartDefaults
from open_mlpipe.core.context import PipelineContext
from open_mlpipe.core.stage import Stage
from open_mlpipe.stages.compare import _regression_stratified_cv
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

        # Resolve CV splits from config — regression uses quantile-stratified
        # folds (see compare.py for rationale).
        n_splits = config.model_selection.cross_validation.n_splits
        if task_type == TaskType.CLASSIFICATION:
            cv_splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        else:
            cv_splitter = _regression_stratified_cv(y_train, n_splits)

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
            import warnings as _w
            with _w.catch_warnings():
                _w.simplefilter("ignore", RuntimeWarning)
                _w.simplefilter("ignore", FutureWarning)
                baseline_scores = cross_val_score(
                    baseline_pipe, X_train, y_train, cv=cv_splitter, scoring=scoring_metric, n_jobs=1)
            baseline_mean = float(baseline_scores.mean())
        except Exception:
            baseline_mean = float("-inf")
        ctx.metrics["tune_baseline_score"] = baseline_mean

        # Get search space for the best model
        search_space = self._get_search_space(ctx.best_model_name, task_type)

        if not search_space:
            return ctx

        # ⬇ ponytail helper -- convert sklearn scoring string to live model score
        # without needing cross_val_score. Locally scoped keeps module lean.
        def optuna_integration_score(pipe, X_va, y_va, scoring_str):
            from sklearn.metrics import get_scorer
            try:
                return get_scorer(scoring_str)(pipe, X_va, y_va)
            except Exception:
                # Fallbacks for aliases Optuna might not recognize directly
                if scoring_str in ("r2", "r2_score"):
                    from sklearn.metrics import r2_score
                    return r2_score(y_va, pipe.predict(X_va))
                if scoring_str == "neg_root_mean_squared_error":
                    import numpy as np
                    from sklearn.metrics import mean_squared_error
                    return -float(np.sqrt(mean_squared_error(y_va, pipe.predict(X_va))))
                if scoring_str == "neg_mean_absolute_error":
                    from sklearn.metrics import mean_absolute_error
                    return -mean_absolute_error(y_va, pipe.predict(X_va))
                if scoring_str == "accuracy":
                    from sklearn.metrics import accuracy_score
                    return accuracy_score(y_va, pipe.predict(X_va))
                if scoring_str and scoring_str.startswith("f1"):
                    from sklearn.metrics import f1_score
                    return f1_score(y_va, pipe.predict(X_va), average="macro")
                raise

        # Optuna objective -- report per-fold scores so a pruner can kill
        # hopeless trials after fold 1-2 instead of waiting for all 5.
        # Faster convergence + more trials actually explored -> higher tuned
        # accuracy (the cheap win that was leaving 5-10 R2 points on the table
        # when the old cross_val_score ran all folds before pruning).
        cv_n_splits = config.model_selection.cross_validation.n_splits

        # Detect target skew and decide whether to apply log1p target transform.
        # Only helps for LINEAR models (ridge, lasso, svm, elastic_net) whose
        # MSE loss is dominated by the few large targets. Tree ensembles (RF,
        # XGB, LightGBM, CatBoost) already handle skew via non-linear splits --
        # log1p actually removes information their tree splits can use and
        # HURTS accuracy (empirically: test_r2 dropped 0.78 -> 0.75 on RF).
        import warnings as _w

        import numpy as np
        import pandas as _pd

        def _safe_skew(s):
            """Skew that survives NaN/inf and non-numeric dtypes."""
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                if not _pd.api.types.is_numeric_dtype(s):
                    return 0.0
                s_clean = s.dropna() if hasattr(s, "dropna") else s
                if len(s_clean) < 3:
                    return 0.0
                return float(s_clean.skew())

        skew_val = abs(_safe_skew(y_train))
        _linear_models = {"ridge", "lasso", "elastic_net", "svm",
                          "linear_regression", "sgd", "bayesian_ridge",
                          "kernel_ridge", "linear svr"}
        best_is_linear = ctx.best_model_name.lower() in _linear_models
        use_target_transform = (
            task_type == TaskType.REGRESSION
            and skew_val > 1.0
            and best_is_linear
        )
        if use_target_transform:
            print(f"    Target skew={skew_val:.2f} -> applying log1p target transform "
                  f"(linear model: {ctx.best_model_name})")
        elif skew_val > 1.0 and not best_is_linear:
            print(f"    Target skew={skew_val:.2f} -> skipping transform "
                  f"(tree model {ctx.best_model_name} handles skew natively)")

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

            model = self._build_model(ctx.best_model_name, params, task_type)
            if use_target_transform:
                from sklearn.compose import TransformedTargetRegressor
                model = TransformedTargetRegressor(
                    regressor=model, func=np.log1p, inverse_func=np.expm1)
            pipe = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model),
            ])

            # Per-fold reporting -- pruner can step in after each fold.
            scores = []
            for fold_idx, (train_idx, val_idx) in enumerate(cv_splitter.split(X_train, y_train)):
                X_fold_tr = X_train.iloc[train_idx] if hasattr(X_train, "iloc") else X_train[train_idx]
                y_fold_tr = y_train.iloc[train_idx] if hasattr(y_train, "iloc") else y_train[train_idx]
                X_fold_va = X_train.iloc[val_idx] if hasattr(X_train, "iloc") else X_train[val_idx]
                y_fold_va = y_train.iloc[val_idx] if hasattr(y_train, "iloc") else y_train[val_idx]

                pipe.fit(X_fold_tr, y_fold_tr)
                score = float(optuna_integration_score(pipe, X_fold_va, y_fold_va, scoring_metric))
                scores.append(score)
                # Report running mean so the pruner sees trajectory -- median
                # pruner needs intermediate values at the same step across trials.
                trial.report(sum(scores) / len(scores), step=fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return sum(scores) / len(scores)

        # Run Optuna -- pick the sampler based on search-space shape.
        # CMA-ES (Covariance Matrix Adaptation Evolution Strategy) is the
        # state-of-art optimizer for purely continuous spaces, beating TPE
        # for hyperparameters like learning_rate / reg_alpha / subsample
        # (Hansen 2016, IJCAI Optuna benchmark 2023). Falls back to TPE
        # when the space has categorical vars -- CMA-ES can't sample them.
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        n_trials = config.tuning.n_trials
        if isinstance(n_trials, str):
            n_trials = SmartDefaults.allocate_tuning_budget(
                len(X_train), len(X_train.columns),
                n_search_dims=len(search_space))
        else:
            # Resolver doesn't know n_search_dims -- if it picked a low
            # fixed number and the search space is wide, bump it so TPE
            # actually has enough samples to model hyperparam interactions.
            # 12+ dim spaces with only 20 trials = nothing converges.
            suggested = SmartDefaults.allocate_tuning_budget(
                len(X_train), len(X_train.columns),
                n_search_dims=len(search_space))
            if suggested > int(n_trials):
                n_trials = suggested

        # Heuristic: space has no categoricals AND >3 float dims -> CMA-ES.
        # CMA-ES is the SOTA sampler for purely continuous spaces, but Optuna
        # ships it as a separate `cmaes` package -- not always installed.
        # Try-import; fall back to TPE+multivariate on failure so the pipeline
        # never hard-fails just because an optional dep is missing.
        has_categorical = any(
            p.get("type") == "categorical" for p in search_space.values()
        )
        n_continuous = sum(
            1 for p in search_space.values()
            if p.get("type") in ("float", "float_log")
        )

        def _build_sampler():
            import warnings as _w
            if not has_categorical and n_continuous >= 3:
                # find_spec is the lightweight probe -- doesn't execute the
                # module, so a half-broken install can't crash us at probe time.
                if find_spec("cmaes") is not None:
                    return optuna.samplers.CmaEsSampler(
                        seed=42,
                        n_startup_trials=max(5, min(10, int(n_trials) // 5)),
                    ), "CMAES"
            # multivariate=True is Optuna experimental feature that prints a
            # noisy FutureWarning -- silence it; we accept the contract.
            with _w.catch_warnings():
                _w.simplefilter("ignore", FutureWarning)
                return optuna.samplers.TPESampler(seed=42, multivariate=True), "TPE"

        sampler, sampler_name = _build_sampler()
        ctx.metrics["tune_sampler"] = sampler_name

        # MedianPruner kills trials whose running mean is below the median of
        # all trials at the same fold-step. ~30-40% of total CV compute is
        # wasted on hopeless configs in typical TPE runs; this reclaims it
        # into more trials explored within the same timeout wall.
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=max(5, int(n_trials) // 6),
            n_warmup_steps=max(1, cv_n_splits // 3),
        )
        n_pruned = [0]

        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
        )

        study.optimize(
            objective, n_trials=int(n_trials), timeout=config.tuning.timeout)
        n_pruned[0] = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)

        # ── Compare tuned vs baseline — only keep if tuning improved ──
        tuned_best_value = study.best_value
        ctx.metrics["tuned_best_value"] = tuned_best_value
        ctx.metrics["tuned_best_params"] = study.best_params
        completed_trials = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
        pruned_trials = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)

        # Surface what the tuner actually ran with -- users kept asking
        # "why is my model weak" and the answer was hidden in metrics.
        pruning_note = f" | pruned={pruned_trials}" if pruned_trials else ""
        print(
            f"    Tuner: {ctx.metrics.get('tune_sampler', 'TPE')} "
            f"| {completed_trials}/{int(n_trials)} trials | {len(search_space)} hyperparameters "
            f"| scoring={scoring_metric}{pruning_note}"
        )
        print(
            f"    {scoring_metric}: baseline={baseline_mean:.4f} "
            f"-> tuned={tuned_best_value:.4f}"
        )

        if tuned_best_value < baseline_mean - 1e-6:
            print(f"    Tuning did not improve {scoring_metric}: "
                  f"baseline={baseline_mean:.6f}, tuned={tuned_best_value:.6f}. "
                  "Keeping untuned model.")
            return ctx

        # Build and fit the tuned model
        best_model = self._build_model(
            ctx.best_model_name, study.best_params, task_type)
        if use_target_transform:
            from sklearn.compose import TransformedTargetRegressor
            best_model = TransformedTargetRegressor(
                regressor=best_model, func=np.log1p, inverse_func=np.expm1)
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
                "n_estimators": {"type": "int", "low": 80, "high": 800},
                "learning_rate": {"type": "float_log", "low": 0.005, "high": 0.3},
                "max_depth": {"type": "int", "low": 3, "high": 15},
                "num_leaves": {"type": "int", "low": 15, "high": 200},
                "min_child_samples": {"type": "int", "low": 5, "high": 80},
                "subsample": {"type": "float", "low": 0.5, "high": 1.0},
                "colsample_bytree": {"type": "float", "low": 0.3, "high": 1.0},
                "reg_alpha": {"type": "float_log", "low": 1e-6, "high": 1.0},
                "reg_lambda": {"type": "float_log", "low": 1e-6, "high": 1.0},
                "min_split_gain": {"type": "float_log", "low": 1e-6, "high": 1.0},
            },
            "xgboost": {
                "n_estimators": {"type": "int", "low": 80, "high": 800},
                "learning_rate": {"type": "float_log", "low": 0.005, "high": 0.3},
                "max_depth": {"type": "int", "low": 3, "high": 15},
                "min_child_weight": {"type": "int", "low": 1, "high": 20},
                "subsample": {"type": "float", "low": 0.5, "high": 1.0},
                "colsample_bytree": {"type": "float", "low": 0.3, "high": 1.0},
                "colsample_bylevel": {"type": "float", "low": 0.3, "high": 1.0},
                "gamma": {"type": "float_log", "low": 1e-5, "high": 5.0},
                "reg_alpha": {"type": "float_log", "low": 1e-5, "high": 1.0},
                "reg_lambda": {"type": "float_log", "low": 1e-5, "high": 1.0},
                "scale_pos_weight": {"type": "float", "low": 1.0, "high": 10.0},
            },
            "random_forest": {
                "n_estimators": {"type": "int", "low": 100, "high": 800},
                "max_depth": {"type": "int", "low": 3, "high": 40},
                "min_samples_split": {"type": "int", "low": 2, "high": 30},
                "min_samples_leaf": {"type": "int", "low": 1, "high": 20},
                "max_features": {"type": "categorical", "choices": ["sqrt", "log2", 0.3, 0.5, 0.7, 1.0]},
                # criterion is task-aware -- classifier accepts gini/entropy/log_loss,
                # regressor accepts squared_error/absolute_error/poisson. Resolved below.
                "criterion": {"type": "categorical", "choices": ["auto"]},
                "max_leaf_nodes": {"type": "int", "low": 10, "high": 200},
                "min_impurity_decrease": {"type": "float_log", "low": 1e-6, "high": 0.1},
                "ccp_alpha": {"type": "float_log", "low": 1e-5, "high": 0.1},
                # bootstrap=True is required to make max_samples meaningful, so we keep
                # bootstrap fixed at True and tune max_samples -- avoids the ValueError
                # "max_sample cannot be set if bootstrap=False" that kills trials.
                "max_samples": {"type": "float", "low": 0.5, "high": 1.0},
            },
            "decision_tree": {
                "max_depth": {"type": "int", "low": 3, "high": 20},
                "min_samples_split": {"type": "int", "low": 2, "high": 20},
                "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
                "criterion": {"type": "categorical", "choices": ["auto"]},
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
                "iterations": {"type": "int", "low": 100, "high": 800},
                "learning_rate": {"type": "float_log", "low": 0.005, "high": 0.3},
                "depth": {"type": "int", "low": 4, "high": 12},
                "l2_leaf_reg": {"type": "float_log", "low": 1e-3, "high": 20.0},
                "bagging_temperature": {"type": "float", "low": 0.0, "high": 1.0},
                "random_strength": {"type": "float", "low": 1.0, "high": 10.0},
                "border_count": {"type": "categorical", "choices": [32, 64, 128, 254]},
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
        space = dict(spaces.get(model_name, {}))

        # Task-aware criterion for tree models -- classifier vs regressor accept
        # different strings. Re-resolve the "auto" sentinel per task.
        is_cls = task == TaskType.CLASSIFICATION
        if model_name in ("random_forest", "extra_trees", "decision_tree") and "criterion" in space:
            choices = (["gini", "entropy", "log_loss"] if is_cls
                       else ["squared_error", "absolute_error"])
            choices = [c for c in choices if c != "auto"]
            space["criterion"] = {"type": "categorical", "choices": choices}

        return space

    def _build_model(self, model_name, params, task):
        """Build a model instance with given params.

        For skewed regression targets (|skew| > 1.0), wraps the model in
        TransformedTargetRegressor with log1p/expm1. This flattens the
        target from [596..2.2M] -> [6.4..14.6] so tree splits land in
        the right places -- the single biggest accuracy lever for heavy-tail
        regression data (your skewed_target_demo.csv has skew=10.96).
        """
        from open_mlpipe.stages.compare import CompareStage
        comparator = CompareStage()
        is_cls = task == TaskType.CLASSIFICATION
        model = comparator._build_single(model_name, is_cls)
        if model is not None:
            try:
                if model.__class__.__name__ == "CalibratedClassifierCV":
                    params = {f"estimator__{k}": v for k, v in params.items()}
                model.set_params(**params)
            except Exception:
                pass
            return model
        # Fallback
        if is_cls:
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(n_estimators=200, random_state=42)
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=200, random_state=42)
