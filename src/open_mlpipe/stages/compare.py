"""ModelComparator stage — multi-model cross-validation comparison.

Supports 20+ model types covering the full production ML spectrum:
- Linear: LogisticRegression, Ridge, Lasso, ElasticNet, LinearRegression
- Tree: DecisionTree
- Bagging: RandomForest, ExtraTrees
- Boosting: XGBoost, LightGBM, CatBoost, GradientBoosting, HistGradientBoosting, AdaBoost
- Instance: KNN, SVM
- Probabilistic: NaiveBayes
- Ensemble: Stacking, Voting
"""

from __future__ import annotations

import os
import warnings

import numpy as np

# Fix Windows cp1252 UnicodeDecodeError in joblib/loky subprocesses
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

# Suppress known harmless warnings from third-party libs
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", message="Could not find the number of physical cores")

from sklearn.model_selection import KFold, StratifiedKFold, cross_validate  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from open_mlpipe.config.defaults import SmartDefaults  # noqa: E402
from open_mlpipe.core.context import PipelineContext  # noqa: E402
from open_mlpipe.core.stage import Stage  # noqa: E402
from open_mlpipe.utils.typing import TaskType  # noqa: E402

# ── All supported model names ──────────────────────────────────────────────
ALL_MODELS = [
    # Linear
    "logistic_regression", "ridge", "lasso", "elasticnet", "linear_regression",
    # Tree
    "decision_tree",
    # Bagging
    "random_forest", "extra_trees",
    # Boosting
    "xgboost", "lightgbm", "catboost", "gradient_boosting",
    "hist_gradient_boosting", "adaboost",
    # Instance
    "knn", "svm",
    # Probabilistic
    "naive_bayes",
    # Ensemble
    "stacking", "voting",
]

# Models that are classification-only
CLS_ONLY = {"logistic_regression", "naive_bayes", "svm", "stacking", "voting"}
# Models that are regression-only
REG_ONLY = {"linear_regression", "lasso", "elasticnet"}


class CompareStage(Stage):
    name = "compare"
    version = "2.0"

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        X_train = ctx.X_train
        y_train = ctx.y_train
        config = ctx.config
        if X_train is None or y_train is None or config is None:
            return ctx

        task = ctx.task_type
        if task is None:
            return ctx
        scoring = config.model_selection.scoring or SmartDefaults.default_scoring(task)

        # Get model candidates
        candidates = config.model_selection.candidates
        if isinstance(candidates, str) and candidates == "auto":
            candidates = SmartDefaults.select_models(task, len(X_train), len(X_train.columns))

        # Build model instances
        models = self._build_models(candidates, task)

        # CV strategy
        cv_splits = config.model_selection.cross_validation.n_splits
        if task == TaskType.CLASSIFICATION:
            cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
        else:
            cv = KFold(n_splits=cv_splits, shuffle=True, random_state=42)

        # Run comparison
        results = {}
        for name, model in models.items():
            try:
                pipe = Pipeline([
                    ("preprocessor", ctx.preprocessor),
                    ("model", model),
                ])

                cv_results = cross_validate(
                    pipe, X_train, y_train,
                    cv=cv,
                    scoring=scoring,
                    return_train_score=True,
                    n_jobs=1,  # n_jobs=-1 causes UnicodeDecodeError on Windows (cp1252)
                )

                # Determine primary metric for ranking
                primary = config.model_selection.ranking_primary
                test_key = f"test_{primary}"
                if test_key not in cv_results:
                    primary = scoring if isinstance(scoring, str) else (scoring[0] if scoring else "accuracy")
                    test_key = f"test_{primary}"

                if test_key not in cv_results:
                    test_keys = [k for k in cv_results if k.startswith("test_")]
                    if test_keys:
                        primary = test_keys[0].replace("test_", "")
                    else:
                        results[name] = {"error": "No test metrics found"}
                        continue

                val_mean = cv_results[f"test_{primary}"].mean()
                train_mean = cv_results[f"train_{primary}"].mean()

                results[name] = {
                    "val_mean": val_mean,
                    "val_std": cv_results[f"test_{primary}"].std(),
                    "train_mean": train_mean,
                    "gap": train_mean - val_mean,
                    "scoring": primary,
                }

                # Fit on full train for later stages
                pipe.fit(X_train, y_train)
                ctx.baseline_models[name] = pipe

            except Exception as e:
                results[name] = {"error": str(e)}

        # Rank models — filter out NaN and errors
        valid_results = {
            k: v for k, v in results.items()
            if "val_mean" in v and not (isinstance(v["val_mean"], float) and np.isnan(v["val_mean"]))
        }
        if valid_results:
            ranked = sorted(valid_results.items(), key=lambda x: x[1]["val_mean"], reverse=True)
            ctx.best_model_name = ranked[0][0]
            ctx.best_model = ctx.baseline_models.get(ranked[0][0])

            # Overfitting diagnosis
            best_gap = ranked[0][1]["gap"]
            if best_gap > 0.1:
                ctx.metrics["overfitting_warning"] = f"Large gap ({best_gap:.3f}) — model may be overfitting"
            elif best_gap < 0.01 and ranked[0][1]["val_mean"] < 0.5:
                ctx.metrics["underfitting_warning"] = f"Small gap but low score ({ranked[0][1]['val_mean']:.3f}) — model may be underfitting"

        ctx.metrics["model_comparison"] = results
        return ctx

    def _build_models(self, candidates, task):
        """Build model instances from candidate names.

        Supports 20+ model types covering the full production ML spectrum.
        """
        models = {}
        is_cls = task == TaskType.CLASSIFICATION

        for name in candidates:
            model = self._build_single(name, is_cls)
            if model is not None:
                models[name] = model
            else:
                warnings.warn(f"Unknown or incompatible model candidate: {name} (task={task.value})")

        return models

    def _build_single(self, name, is_cls):
        """Build a single model instance. Returns None if incompatible."""
        # ── Linear Models ──────────────────────────────────────────────────
        if name == "logistic_regression":
            if not is_cls:
                return None
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)

        if name == "ridge":
            if is_cls:
                return None
            from sklearn.linear_model import Ridge
            return Ridge(alpha=1.0)

        if name == "lasso":
            if is_cls:
                return None
            from sklearn.linear_model import Lasso
            return Lasso(alpha=0.01, max_iter=5000)

        if name == "elasticnet":
            if is_cls:
                return None
            from sklearn.linear_model import ElasticNet
            return ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000)

        if name == "linear_regression":
            if is_cls:
                return None
            from sklearn.linear_model import LinearRegression
            return LinearRegression()

        # ── Tree Models ────────────────────────────────────────────────────
        if name == "decision_tree":
            if is_cls:
                from sklearn.tree import DecisionTreeClassifier
                return DecisionTreeClassifier(max_depth=10, class_weight="balanced", random_state=42)
            else:
                from sklearn.tree import DecisionTreeRegressor
                return DecisionTreeRegressor(max_depth=10, random_state=42)

        # ── Bagging Models ─────────────────────────────────────────────────
        if name == "random_forest":
            if is_cls:
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier(n_estimators=300, class_weight="balanced", max_features="sqrt", n_jobs=1, random_state=42)  # type: ignore[arg-type]
            else:
                from sklearn.ensemble import RandomForestRegressor
                return RandomForestRegressor(n_estimators=300, max_features="sqrt", n_jobs=1, random_state=42)  # type: ignore[arg-type]

        if name == "extra_trees":
            if is_cls:
                from sklearn.ensemble import ExtraTreesClassifier
                return ExtraTreesClassifier(n_estimators=300, class_weight="balanced", max_features="sqrt", n_jobs=1, random_state=42)  # type: ignore[arg-type]
            else:
                from sklearn.ensemble import ExtraTreesRegressor
                return ExtraTreesRegressor(n_estimators=300, max_features="sqrt", n_jobs=1, random_state=42)  # type: ignore[arg-type]

        # ── Boosting Models ────────────────────────────────────────────────
        if name == "xgboost":
            from xgboost import XGBClassifier, XGBRegressor
            if is_cls:
                return XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                                     subsample=0.8, colsample_bytree=0.8,
                                     eval_metric="logloss", verbosity=0, random_state=42)
            else:
                return XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                                    subsample=0.8, colsample_bytree=0.8,
                                    verbosity=0, random_state=42)

        if name == "lightgbm":
            import lightgbm as lgb
            if is_cls:
                return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                          subsample=0.8, colsample_bytree=0.8,
                                          class_weight="balanced", verbosity=-1, random_state=42)
            else:
                return lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                         subsample=0.8, colsample_bytree=0.8,
                                         verbosity=-1, random_state=42)

        if name == "catboost":
            try:
                from catboost import (  # type: ignore[import-untyped]
                    CatBoostClassifier,
                    CatBoostRegressor,
                )
                if is_cls:
                    return CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6,
                                             verbose=0, random_state=42)
                else:
                    return CatBoostRegressor(iterations=300, learning_rate=0.05, depth=6,
                                            verbose=0, random_state=42)
            except ImportError:
                warnings.warn("catboost not installed, skipping")
                return None

        if name == "gradient_boosting":
            from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
            if is_cls:
                return GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                                  max_depth=3, subsample=0.8, random_state=42)
            else:
                return GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                                 max_depth=3, subsample=0.8, random_state=42)

        if name == "hist_gradient_boosting":
            from sklearn.ensemble import (
                HistGradientBoostingClassifier,
                HistGradientBoostingRegressor,
            )
            if is_cls:
                return HistGradientBoostingClassifier(
                    max_iter=300, learning_rate=0.05,
                    early_stopping=True, random_state=42)  # pyright: ignore
            else:
                return HistGradientBoostingRegressor(
                    max_iter=300, learning_rate=0.05,
                    early_stopping=True, random_state=42)  # pyright: ignore

        if name == "adaboost":
            from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
            from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
            if is_cls:
                return AdaBoostClassifier(
                    estimator=DecisionTreeClassifier(max_depth=1),
                    n_estimators=200, learning_rate=0.1, random_state=42)
            else:
                return AdaBoostRegressor(
                    estimator=DecisionTreeRegressor(max_depth=1),
                    n_estimators=200, learning_rate=0.1, random_state=42)

        # ── Instance-Based Models ──────────────────────────────────────────
        if name == "knn":
            if is_cls:
                from sklearn.neighbors import KNeighborsClassifier
                return KNeighborsClassifier(n_neighbors=5, weights="distance")
            else:
                from sklearn.neighbors import KNeighborsRegressor
                return KNeighborsRegressor(n_neighbors=5, weights="distance")

        if name == "svm":
            if is_cls:
                # SVC(probability=True) deprecated in sklearn 1.9 — use CalibratedClassifierCV wrapper
                from sklearn.calibration import CalibratedClassifierCV
                from sklearn.svm import SVC
                return CalibratedClassifierCV(
                    SVC(class_weight="balanced", kernel="rbf", random_state=42),
                    cv=5,
                )
            else:
                from sklearn.svm import SVR
                return SVR(kernel="rbf", epsilon=0.1)

        # ── Probabilistic Models ───────────────────────────────────────────
        if name == "naive_bayes":
            if not is_cls:
                return None
            from sklearn.naive_bayes import GaussianNB
            return GaussianNB()

        # ── Ensemble Models (require special handling) ─────────────────────
        if name == "stacking":
            return self._build_stacking(is_cls)

        if name == "voting":
            return self._build_voting(is_cls)

        return None

    def _build_stacking(self, is_cls):
        """Build stacking ensemble — base models + meta-learner."""
        if is_cls:
            from sklearn.ensemble import RandomForestClassifier, StackingClassifier
            from sklearn.linear_model import LogisticRegression
            try:
                import lightgbm as lgb
                from xgboost import XGBClassifier
                base_estimators = [
                    ("rf", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)),
                    ("xgb", XGBClassifier(n_estimators=100, learning_rate=0.05, verbosity=0, random_state=42)),
                    ("lgb", lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, verbosity=-1, random_state=42)),
                ]
            except ImportError:
                base_estimators = [
                    ("rf", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)),
                ]
            return StackingClassifier(
                estimators=base_estimators,
                final_estimator=LogisticRegression(max_iter=1000),
                cv=5, stack_method="predict_proba", n_jobs=1)
        else:
            from sklearn.ensemble import RandomForestRegressor, StackingRegressor
            from sklearn.linear_model import Ridge
            try:
                import lightgbm as lgb
                from xgboost import XGBRegressor
                base_estimators = [
                    ("rf", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)),
                    ("xgb", XGBRegressor(n_estimators=100, learning_rate=0.05, verbosity=0, random_state=42)),
                    ("lgb", lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, verbosity=-1, random_state=42)),
                ]
            except ImportError:
                base_estimators = [
                    ("rf", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)),
                ]
            return StackingRegressor(
                estimators=base_estimators,
                final_estimator=Ridge(alpha=1.0),
                cv=5, n_jobs=1)

    def _build_voting(self, is_cls):
        """Build voting ensemble — soft voting across diverse models."""
        if is_cls:
            from sklearn.ensemble import RandomForestClassifier, VotingClassifier
            from sklearn.linear_model import LogisticRegression
            try:
                from xgboost import XGBClassifier
                estimators = [
                    ("lr", LogisticRegression(max_iter=1000, random_state=42)),
                    ("rf", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)),
                    ("xgb", XGBClassifier(n_estimators=100, learning_rate=0.05, verbosity=0, random_state=42)),
                ]
            except ImportError:
                estimators = [
                    ("lr", LogisticRegression(max_iter=1000, random_state=42)),
                    ("rf", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)),
                ]
            return VotingClassifier(estimators=estimators, voting="soft", n_jobs=1)
        else:
            from sklearn.ensemble import RandomForestRegressor, VotingRegressor
            from sklearn.linear_model import Ridge
            try:
                from xgboost import XGBRegressor
                estimators = [
                    ("ridge", Ridge(alpha=1.0)),
                    ("rf", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)),
                    ("xgb", XGBRegressor(n_estimators=100, learning_rate=0.05, verbosity=0, random_state=42)),
                ]
            except ImportError:
                estimators = [
                    ("ridge", Ridge(alpha=1.0)),
                    ("rf", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)),
                ]
            return VotingRegressor(estimators=estimators, n_jobs=1)
