# I Built an AutoML Pipeline That Trains 14 Models in One Line of Python

## From raw data to production model — no PhD required

![open-mlpipe](https://img.shields.io/badge/open--mlpipe-v1.0.0-blue) ![Python](https://img.shields.io/badge/python-3.10+-green) ![License](https://img.shields.io/badge/license-MIT-yellow)

---

## The Problem

Every time I start a new ML project, I do the same thing:

1. Load data
2. Clean it
3. Engineer features
4. Try 10 different models
5. Tune hyperparameters
6. Pick the best one
7. Save it

**Hours wasted. Every. Single. Time.**

I wanted a tool that does all of this in **one line of Python**.

## The Solution: open-mlpipe

I built **open-mlpipe** — an AutoML pipeline that takes your raw dataset and automatically:

✅ Loads and profiles your data (EDA)  
✅ Cleans duplicates, outliers, missing values  
✅ Engineers features (interactions, log transforms, missingness flags)  
✅ Compares **14+ ML models** head-to-head  
✅ Tunes hyperparameters with Optuna  
✅ Selects features via SHAP importance  
✅ Explains predictions with SHAP  
✅ Saves a **production-ready inference pipeline**

**One command. One line of code. Production model.**

## Quick Start

### Install

```bash
pip install open-mlpipe
```

### Python API (One Line!)

```python
from open_mlpipe import run

# Full pipeline — data to production model
ctx = run("dataset.csv", target="price")

# Access results
print(f"Best model: {ctx.best_model_name}")
print(f"Test R²: {ctx.metrics['test_r2']:.4f}")

# Load saved model for predictions
import joblib
model = joblib.load("artifacts/model_v1.joblib")
predictions = model.predict(new_data)
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

## What Models Does It Support?

**14+ models** — from linear to ensemble:

| Category | Models |
|----------|--------|
| **Linear** | Ridge, Lasso, ElasticNet, LinearRegression |
| **Tree** | DecisionTree |
| **Bagging** | RandomForest, ExtraTrees |
| **Boosting** | XGBoost, LightGBM, GradientBoosting, HistGradientBoosting, AdaBoost |
| **Instance** | KNN, SVM |
| **Ensemble** | Stacking, Voting |

**Auto-selected** based on your data characteristics.

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

## How It Works

```
RAW DATA
    ↓
[1] Load & Profile (EDA)
    ↓
[2] Clean (duplicates, outliers, missing)
    ↓
[3] Feature Engineering (interactions, log, flags)
    ↓
[4] Split (train/test)
    ↓
[5] Preprocess (impute, scale, encode)
    ↓
[6] Compare 14+ Models (CV)
    ↓
[7] Tune Best Model (Optuna)
    ↓
[8] Select Features (SHAP)
    ↓
[9] Evaluate (test set)
    ↓
[10] Explain (SHAP)
    ↓
[11] Save (full inference pipeline)
    ↓
PRODUCTION MODEL READY
```

## YAML Config (For Reproducibility)

```yaml
# configs/regression.yaml
project: insurance-charges
task: auto
data:
  path: data/insurance.csv
  target: charges
model_selection:
  candidates: [lightgbm, xgboost, random_forest, ridge]
tuning:
  enabled: true
  engine: optuna
  n_trials: 50
```

```bash
open-mlpipe run --config configs/regression.yaml
```

## Features

| Feature | Description |
|---------|-------------|
| **Auto EDA** | Column hygiene, quality, distributions, statistical tests, VIF |
| **Feature Engineering** | Missingness flags, interactions, log transforms, datetime |
| **14+ Models** | Linear, tree, boosting, bagging, ensemble |
| **Hyperparameter Tuning** | Optuna with 18 model-specific search spaces |
| **SHAP Explainability** | Global + local feature importance |
| **Production Inference** | Full pipeline saved (feature_eng + model) |
| **Overfitting Detection** | Automatic train/test gap analysis |
| **Production Metrics** | R², RMSE, MAE, MAPE, F1, ROC-AUC, MCC |

## Installation

```bash
# Core
pip install open-mlpipe

# With CatBoost
pip install open-mlpipe[catboost]

# With MLflow tracking
pip install open-mlpipe[mlflow]

# Everything
pip install open-mlpipe[full]
```

## What's Next?

- [ ] AutoGluon integration
- [ ] Time series support
- [ ] Image/text classification
- [ ] Web UI (React + FastAPI)
- [ ] Cloud deployment (AWS/GCP)

## Links

- **PyPI**: https://pypi.org/project/open-mlpipe/
- **GitHub**: https://github.com/yashb/open-mlpipe
- **Docs**: https://github.com/yashb/open-mlpipe#readme

## Contributing

Contributions welcome! Open an issue or PR.

## License

MIT License

---

**Built with ❤️ by Yash Brahmankar**

*If this saved you time, star the repo and share it with someone who's tired of writing the same ML pipeline code over and over.*
