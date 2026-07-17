# Supported Models

## Linear Models

| Model | Task | Description |
|-------|------|-------------|
| `linear_regression` | Regression | Ordinary least squares |
| `ridge` | Regression | L2 regularization |
| `lasso` | Regression | L1 regularization (feature selection) |
| `elasticnet` | Regression | L1 + L2 regularization |
| `logistic_regression` | Classification | Linear classifier |

## Tree Models

| Model | Task | Description |
|-------|------|-------------|
| `decision_tree` | Both | Single decision tree |

## Bagging Models

| Model | Task | Description |
|-------|------|-------------|
| `random_forest` | Both | Bootstrap aggregating |
| `extra_trees` | Both | Extremely randomized trees |

## Boosting Models

| Model | Task | Description |
|-------|------|-------------|
| `xgboost` | Both | Gradient boosting (GPU support) |
| `lightgbm` | Both | Fast gradient boosting |
| `gradient_boosting` | Both | sklearn gradient boosting |
| `hist_gradient_boosting` | Both | Histogram-based (handles NaN) |
| `adaboost` | Both | Adaptive boosting |

## Instance-Based Models

| Model | Task | Description |
|-------|------|-------------|
| `knn` | Both | K-nearest neighbors |
| `svm` | Both | Support vector machine |

## Probabilistic Models

| Model | Task | Description |
|-------|------|-------------|
| `naive_bayes` | Classification | Gaussian naive Bayes |

## Ensemble Models

| Model | Task | Description |
|-------|------|-------------|
| `stacking` | Both | Stacking ensemble |
| `voting` | Both | Voting ensemble |

## Model Selection

open-mlpipe auto-selects models based on:

- Task type (regression/classification)
- Dataset size
- Number of features
- Data characteristics
