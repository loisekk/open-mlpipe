---
title: I Built an AutoML Pipeline That Trains 14 Models in One Line of Python
published: true
description: After months of building ML pipelines from scratch, I built open-mlpipe — an AutoML pipeline that does everything from data loading to production deployment in a single line of Python.
tags: python, machinelearning, automl, opensource
---

# I Built an AutoML Pipeline That Trains 14 Models in One Line of Python

I spent 6 months building the same ML pipeline for every project. Load data. Clean it. Engineer features. Try 10 models. Tune hyperparameters. Pick the best one. Save it.

**Hours wasted. Every. Single. Time.**

So I built **open-mlpipe** — an AutoML pipeline that does it all in one line.

## The Problem

```python
# What I used to do every time
df = pd.read_csv("data.csv")

# Clean
df = df.drop_duplicates()
df = df.dropna()

# Feature engineering
# ... 50 lines ...

# Try models
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
# ... compare 10 models ...

# Tune
# ... hyperparameter optimization ...

# Evaluate
# ... metrics ...

# Save
# ... model saving ...

# 3+ hours later... done.
```

## The Solution

```python
from open_mlpipe import run

# One line. That's it.
ctx = run("dataset.csv", target="price")

# Done. Production model saved.
```

## What It Does (Automatically)

open-mlpipe runs a **12-stage pipeline** behind the scenes:

```
Raw Data → EDA → Clean → Feature Eng → Split → Preprocess
→ Compare 14 Models → Tune → Select → Evaluate → Explain → Save
```

### Stage Breakdown

| Stage | What Happens |
|-------|--------------|
| 📊 **Load & EDA** | Auto-detect task, profile data, statistical tests |
| 🧹 **Clean** | Duplicates, outliers, missing values, ID columns |
| 🔧 **Feature Eng** | Interactions, log transforms, missingness flags, datetime |
| ✂️ **Split** | Stratified train/test split |
| ⚙️ **Preprocess** | Impute, scale, encode (ColumnTransformer) |
| 🏆 **Compare** | 14+ models head-to-head with cross-validation |
| 🎛️ **Tune** | Optuna hyperparameter optimization |
| 📈 **Select** | SHAP-based feature importance |
| ✅ **Evaluate** | R², RMSE, MAE, MAPE, F1, ROC-AUC, MCC |
| 💡 **Explain** | SHAP summary, dependence, waterfall plots |
| 💾 **Save** | Full inference pipeline (feature_eng + model) |

## 14+ Models Supported

### Linear
- Ridge, Lasso, ElasticNet, LinearRegression
- LogisticRegression (classification)

### Tree & Ensemble
- DecisionTree, RandomForest, ExtraTrees

### Boosting
- **XGBoost** — King of tabular benchmarks
- **LightGBM** — Fastest GBM, best for large data
- **GradientBoosting** — sklearn's classic
- **HistGradientBoosting** — sklearn's answer to LightGBM (10-100x faster)
- **AdaBoost** — Sequential boosting

### Instance & Probabilistic
- KNN, SVM, NaiveBayes

### Ensemble
- **Stacking** — Base models → meta-learner
- **Voting** — Soft voting across diverse models

**Auto-selected** based on data characteristics and task type.

## Real Results

I tested it on the **California Housing dataset** (20K rows):

```
=== Pipeline Complete ===
  Best Model:  lightgbm
  Test R²:     0.8474
  Test MAE:    0.2924
  Test MAPE:   16.77%
  Total Time:  4.5 minutes
```

### Model Comparison (Automatic)

| Model | R² | Gap | Status |
|-------|-----|-----|--------|
| LightGBM | 0.839 | 0.072 | ✅ Best |
| XGBoost | 0.835 | 0.093 | ✅ Good |
| RandomForest | 0.771 | 0.198 | ⚠️ Overfit |
| Ridge | 0.709 | 0.001 | ✅ Stable |

## Advanced Usage

### Python API

```python
from open_mlpipe import PipelineConfig, PipelineRunner
from open_mlpipe.config.resolver import build_level1_config

# Build config
config = build_level1_config("data.csv", target="price")
config.project = "my-project"
config.tuning.enabled = True
config.tuning.n_trials = 50

# Run pipeline
runner = PipelineRunner(config)
ctx = runner.run()

# Access everything
print(ctx.best_model_name)  # "lightgbm"
print(ctx.metrics["test_r2"])  # 0.847
print(ctx.metrics["test_mape"])  # 16.77
```

### CLI

```bash
# Zero-touch — auto-detect everything
open-mlpipe run --data dataset.csv

# With target specified
open-mlpipe run --data dataset.csv --target price

# Config-driven
open-mlpipe run --config configs/regression.yaml
```

### YAML Config

```yaml
project: insurance-charges
task: auto
data:
  path: data/insurance.csv
  target: charges
  test_size: 0.2

model_selection:
  candidates: [lightgbm, xgboost, random_forest, ridge]
  scoring: [r2, neg_mean_absolute_error]
  ranking_primary: r2

tuning:
  enabled: true
  engine: optuna
  n_trials: 50
  timeout: 600

feature_selection:
  enabled: true
  method: shap_importance
  min_importance: 0.01

evaluation:
  explainability: true
  shap_plots: [summary, dependence, waterfall]
```

## Installation

```bash
# Core
pip install open-mlpipe

# With CatBoost
pip install open-mlpipe[catboost]

# With MLflow
pip install open-mlpipe[mlflow]

# Everything
pip install open-mlpipe[full]
```

## Features

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

## What's Next

- [ ] AutoGluon integration
- [ ] Time series support
- [ ] Image/text classification
- [ ] Web UI (React + FastAPI)
- [ ] Cloud deployment (AWS/GCP)

## Links

- **PyPI**: https://pypi.org/project/open-mlpipe/
- **GitHub**: https://github.com/loisekk/open-mlpipe

## Contributing

Contributions welcome! Open an issue or PR.

---

**Built with ❤️ by Yash Brahmankar**

*If this saved you time, star the repo and share it with someone who's tired of writing the same ML pipeline code over and over.*
