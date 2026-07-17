# open-mlpipe

**Production-level automated ML pipeline — from raw data to deployed model in one line.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://img.shields.io/pypi/v/mlpipe.svg)](https://pypi.org/project/mlpipe/)

## What is open-mlpipe?

open-mlpipe is an **AutoML pipeline** that takes your raw dataset and automatically:

1. Loads and profiles your data (EDA)
2. Cleans duplicates, outliers, and missing values
3. Engineers features (interactions, log transforms, missingness flags)
4. Compares **14+ ML models** head-to-head
5. Tunes hyperparameters with Optuna
6. Selects features via SHAP importance
7. Explains predictions with SHAP
8. Saves a **production-ready inference pipeline**

**One command. One line of code. Production model.**

## Quick Start

### Install

```bash
pip install open-mlpipe
```

### CLI Usage

```bash
# Zero-touch — auto-detect everything
open-mlpipe run --data dataset.csv

# With target specified
open-mlpipe run --data dataset.csv --target price

# Config-driven
open-mlpipe run --config configs/regression.yaml
```

### Python API

```python
from open_mlpipe import run

# One line — full pipeline
ctx = run("dataset.csv", target="price")

# Access results
print(f"Best model: {ctx.best_model_name}")
print(f"Test R²: {ctx.metrics['test_r2']:.4f}")

# Load saved model for predictions
import joblib
model = joblib.load("artifacts/model_v1.joblib")
predictions = model.predict(new_data)
```

### YAML Config

```yaml
# configs/my_pipeline.yaml
project: my-project
task: auto
data:
  path: data.csv
  target: price
model_selection:
  candidates: [lightgbm, xgboost, random_forest, ridge]
tuning:
  enabled: true
  engine: optuna
  n_trials: 50
```

```bash
open-mlpipe run --config configs/my_pipeline.yaml
```

## Features

| Feature | Description |
|---------|-------------|
| **14+ Models** | Ridge, Lasso, ElasticNet, DecisionTree, RandomForest, ExtraTrees, XGBoost, LightGBM, GradientBoosting, HistGradientBoosting, AdaBoost, KNN, SVM, Stacking, Voting |
| **Auto EDA** | Column hygiene, quality assessment, distributions, statistical tests, VIF |
| **Feature Engineering** | Missingness flags, interaction features, log transforms, datetime extraction |
| **Hyperparameter Tuning** | Optuna with TPE sampler, 18 model-specific search spaces |
| **SHAP Explainability** | Global feature importance, local explanations |
| **Production Inference** | Full pipeline saved (feature engineering + model) — raw data → prediction |
| **Overfitting Detection** | Automatic train/test gap analysis |
| **Production Metrics** | R², RMSE, MAE, MAPE, F1, ROC-AUC, MCC |

## Supported Tasks

- **Regression** — predict continuous values
- **Classification** — predict categories (binary/multiclass)

## Installation Options

```bash
# Core (minimal)
pip install open-mlpipe

# With CatBoost
pip install open-mlpipe[catboost]

# With MLflow tracking
pip install open-mlpipe[mlflow]

# With deployment (FastAPI + Docker)
pip install open-mlpipe[deploy]

# Everything
pip install open-mlpipe[full]
```

## Example Output

```
=== mlpipe Pipeline Runner ===
  Project:  california-housing
  Data:     california_housing.csv
  Target:   median_house_value

>> load       OK (0.1s)
>> eda        OK (0.2s)
>> clean      OK (0.0s)
>> feature_eng OK (0.0s)
>> split      OK (0.0s)
>> preprocess OK (0.0s)
>> compare    OK (25.9s)  — tested 14 models
>> tune       OK (20.2s)  — Optuna 20 trials
>> select     OK (0.0s)
>> evaluate   OK (0.0s)
>> explain    OK (3.3s)
>> save       OK (0.1s)

=== Pipeline Complete ===
  Best Model:  lightgbm
  Test R²:     0.8474
  Test MAE:    0.2924
  Test MAPE:   16.77%
```

## Model Comparison

open-mlpipe automatically compares all models and ranks them:

| Model | R² | Gap | Status |
|-------|-----|-----|--------|
| LightGBM | 0.839 | 0.072 | Best |
| XGBoost | 0.835 | 0.093 | Good |
| RandomForest | 0.771 | 0.198 | Overfit |
| Ridge | 0.709 | 0.001 | Stable |

## Project Structure

```
open-mlpipe/
├── src/open_mlpipe/
│   ├── cli.py              # CLI entry point
│   ├── config/             # Pydantic config + YAML resolver
│   ├── core/               # Pipeline runner, context, stages
│   ├── stages/             # 12 pipeline stages
│   ├── utils/              # I/O, feature engineering
│   └── deploy/             # FastAPI + Docker generation
├── tests/                  # 106 unit tests + 3 integration tests
├── configs/                # 5 example YAML configs
└── pyproject.toml          # Package config
```

## Requirements

- Python >= 3.10
- See `pyproject.toml` for full dependency list

## License

MIT License — see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please open an issue or PR.

## Author

Yash Brahmankar
