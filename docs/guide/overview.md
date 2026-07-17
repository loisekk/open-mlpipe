# Overview

open-mlpipe is a production-level automated ML pipeline that takes your raw dataset and automatically:

1. Loads and profiles your data (EDA)
2. Cleans duplicates, outliers, and missing values
3. Engineers features (interactions, log transforms, missingness flags)
4. Compares **14+ ML models** head-to-head
5. Tunes hyperparameters with Optuna
6. Selects features via SHAP importance
7. Explains predictions with SHAP
8. Saves a **production-ready inference pipeline**

## Architecture

```
Raw Data
    ↓
[1] Load & EDA
    ↓
[2] Clean
    ↓
[3] Feature Engineering
    ↓
[4] Split (train/test)
    ↓
[5] Preprocess (impute, scale, encode)
    ↓
[6] Compare 14+ Models
    ↓
[7] Tune Best Model (Optuna)
    ↓
[8] Select Features (SHAP)
    ↓
[9] Evaluate (test set)
    ↓
[10] Explain (SHAP)
    ↓
[11] Save (full pipeline)
    ↓
PRODUCTION MODEL READY
```

## Key Features

| Feature | Description |
|---------|-------------|
| **Auto EDA** | Column hygiene, quality, distributions, statistical tests, VIF |
| **Feature Engineering** | Missingness flags, interactions, log transforms, datetime |
| **14+ Models** | Linear, tree, boosting, bagging, ensemble |
| **Hyperparameter Tuning** | Optuna with 18 model-specific search spaces |
| **SHAP Explainability** | Global + local feature importance |
| **Production Inference** | Full pipeline saved (feature engineering + model) |
| **Overfitting Detection** | Automatic train/test gap analysis |
| **Production Metrics** | R², RMSE, MAE, MAPE, F1, ROC-AUC, MCC |

## Supported Tasks

- **Regression** — predict continuous values
- **Classification** — predict categories (binary/multiclass)

## Next Steps

- [Installation](../getting-started/installation.md)
- [Quick Start](../getting-started/quickstart.md)
- [Python API](python-api.md)
