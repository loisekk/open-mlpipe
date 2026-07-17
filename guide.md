# open-mlpipe — Complete Usage Guide

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Import Methods](#import-methods)
- [Python API](#python-api)
- [CLI Usage](#cli-usage)
- [YAML Config](#yaml-config)
- [Pipeline Stages](#pipeline-stages)
- [Supported Models](#supported-models)
- [Metrics](#metrics)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)

---

## Installation

```bash
# Basic install
pip install open-mlpipe

# With optional dependencies
pip install open-mlpipe[catboost]    # CatBoost support
pip install open-mlpipe[mlflow]      # MLflow tracking
pip install open-mlpipe[deploy]      # FastAPI + Docker
pip install open-mlpipe[full]        # Everything
```

---

## Quick Start

### 3 Ways to Use open-mlpipe

```python
# Way 1: Simplest (one line)
from open_mlpipe import run
ctx = run("dataset.csv", target="price")

# Way 2: Config-driven
from open_mlpipe import run_config
ctx = run_config("configs/regression.yaml")

# Way 3: Full control
from open_mlpipe import PipelineConfig, PipelineRunner
```

---

## Import Methods

### Method 1: Import the `run` function (Recommended for beginners)

```python
from open_mlpipe import run

# Auto-detect everything
ctx = run("data.csv")

# Specify target column
ctx = run("data.csv", target="price")

# Custom project name
ctx = run("data.csv", target="price", project="my-model")
```

### Method 2: Import `run_config` for YAML configs

```python
from open_mlpipe import run_config

# Run from YAML config
ctx = run_config("configs/regression.yaml")

# Override project name
ctx = run_config("configs/regression.yaml", project="override-name")
```

### Method 3: Import classes for full control

```python
from open_mlpipe import PipelineConfig, PipelineRunner
from open_mlpipe.config.resolver import build_level1_config, load_config, resolve_config
from open_mlpipe.utils.io import load_data
```

### Method 4: Import specific components

```python
# Config schema
from open_mlpipe.config.schema import (
    PipelineConfig,
    DataConfig,
    ModelSelectionConfig,
    TuningConfig,
    FeatureSelectionConfig,
    EvaluationConfig,
    DeploymentConfig,
    ArtifactConfig,
    CVConfig,
    CleaningConfig,
    FeatureEngConfig,
    PreprocessingConfig,
    ProfilingConfig,
)

# Core
from open_mlpipe.core.pipeline import PipelineRunner
from open_mlpipe.core.context import PipelineContext, StageMetadata
from open_mlpipe.core.stage import Stage
from open_mlpipe.core.registry import StageRegistry

# Stages
from open_mlpipe.stages.load import DataLoaderStage
from open_mlpipe.stages.eda import EDALoaderStage
from open_mlpipe.stages.clean import CleanStage
from open_mlpipe.stages.feature_eng import FeatureEngStage
from open_mlpipe.stages.split import SplitStage
from open_mlpipe.stages.preprocess import PreprocessStage
from open_mlpipe.stages.compare import CompareStage
from open_mlpipe.stages.tune import TuneStage
from open_mlpipe.stages.select import SelectStage
from open_mlpipe.stages.evaluate import EvaluateStage
from open_mlpipe.stages.explain import ExplainStage
from open_mlpipe.stages.save import SaveStage
from open_mlpipe.stages.deploy import DeployStage

# Utils
from open_mlpipe.utils.io import load_data, save_dataframe
from open_mlpipe.utils.typing import TaskType, ColumnType
from open_mlpipe.utils.feature_eng_transformer import FeatureEngTransformer
```

### Method 5: Import the version

```python
import open_mlpipe
print(open_mlpipe.__version__)  # "1.0.0"
```

---

## Python API

### Level 1: Zero-Touch (Auto-detect everything)

```python
from open_mlpipe import run

# Just pass your data file
ctx = run("dataset.csv")

# The pipeline automatically:
# - Detects task type (regression/classification)
# - Detects target column
# - Selects best models
# - Tunes hyperparameters
# - Saves production model
```

### Level 2: Specify Target

```python
from open_mlpipe import run

ctx = run("dataset.csv", target="price")

# Access results
print(f"Best model: {ctx.best_model_name}")
print(f"Test R²: {ctx.metrics['test_r2']:.4f}")
print(f"Test MAE: {ctx.metrics['test_mae']:.4f}")
```

### Level 3: Custom Configuration

```python
from open_mlpipe import PipelineConfig, PipelineRunner
from open_mlpipe.config.resolver import build_level1_config

# Build config from data
config = build_level1_config("data.csv", target="price")

# Customize
config.project = "my-project"
config.tuning.enabled = True
config.tuning.n_trials = 50
config.tuning.timeout = 600
config.feature_selection.enabled = True
config.evaluation.explainability = True

# Run pipeline
runner = PipelineRunner(config)
ctx = runner.run()

# Access all results
print(f"Best model: {ctx.best_model_name}")
print(f"Metrics: {ctx.metrics}")
print(f"Feature importance: {ctx.feature_importance}")
```

### Level 4: Full Manual Control

```python
from open_mlpipe.config.schema import (
    PipelineConfig, DataConfig, ModelSelectionConfig,
    TuningConfig, FeatureSelectionConfig, EvaluationConfig,
    DeploymentConfig, ArtifactConfig, CleaningConfig,
    FeatureEngConfig, CVConfig,
)
from open_mlpipe.core.pipeline import PipelineRunner

config = PipelineConfig(
    project="custom-project",
    task="auto",  # or "regression" / "classification"
    data=DataConfig(
        path="data.csv",
        target="price",
        test_size=0.2,
        random_state=42,
    ),
    cleaning=CleaningConfig(
        remove_duplicates=True,
        missing_strategy="auto",
        outlier_method="auto",
    ),
    feature_engineering=FeatureEngConfig(
        auto_interactions=True,
        auto_log=True,
        missingness_flags=True,
    ),
    model_selection=ModelSelectionConfig(
        candidates=[
            "linear_regression", "ridge", "lasso", "elasticnet",
            "decision_tree", "random_forest", "extra_trees",
            "xgboost", "lightgbm", "gradient_boosting",
            "hist_gradient_boosting", "adaboost",
            "knn", "svm", "stacking", "voting",
        ],
        scoring=["r2", "neg_mean_absolute_error"],
        ranking_primary="r2",
        cross_validation=CVConfig(n_splits=5),
    ),
    tuning=TuningConfig(
        enabled=True,
        engine="optuna",
        n_trials=30,
        timeout=300,
    ),
    feature_selection=FeatureSelectionConfig(
        enabled=True,
        method="shap_importance",
        min_importance=0.01,
    ),
    evaluation=EvaluationConfig(
        explainability=True,
        shap_plots=["summary", "dependence", "waterfall"],
    ),
    deployment=DeploymentConfig(enabled=False),
    artifacts=ArtifactConfig(mlflow_tracking=False),
)

runner = PipelineRunner(config)
ctx = runner.run()
```

### Level 5: Load and Predict with Saved Model

```python
import joblib
import pandas as pd

# Load the saved model
model = joblib.load("artifacts/model_v1.joblib")

# Prepare new data
new_data = pd.DataFrame({
    "feature1": [1.0, 2.0, 3.0],
    "feature2": ["a", "b", "c"],
    "feature3": [10, 20, 30],
})

# Predict
predictions = model.predict(new_data)
print(predictions)
```

---

## CLI Usage

### Commands

```bash
# Full pipeline (zero-touch)
open-mlpipe run --data dataset.csv

# With target specified
open-mlpipe run --data dataset.csv --target price

# With project name
open-mlpipe run --data dataset.csv --target price --project my-project

# Config-driven
open-mlpipe run --config configs/regression.yaml

# With deployment
open-mlpipe run --data dataset.csv --deploy

# EDA only (no training)
open-mlpipe profile --data dataset.csv

# Version
open-mlpipe --version

# Help
open-mlpipe --help
open-mlpipe run --help
```

### CLI Options

```
--data, -d       Path to data file (CSV, Parquet, Excel)
--target, -t     Target column name (auto-detected if not given)
--level, -l      Automation level (1=zero-touch, 2=config-driven)
--config, -c     Path to YAML config file
--project, -p    Project name
--deploy         Generate deployment artifacts
--version        Show version
--help           Show help
```

---

## YAML Config

### Minimal Config

```yaml
project: my-project
data:
  path: data.csv
  target: price
```

### Full Config

```yaml
project: insurance-charges
task: auto  # or "regression" / "classification"
level: 2

data:
  path: data/insurance.csv
  target: charges
  test_size: 0.2
  random_state: 42

cleaning:
  remove_duplicates: true
  missing_strategy: auto
  outlier_method: auto
  outlier_action: clip

feature_engineering:
  auto_interactions: true
  auto_log: true
  datetime_features: true
  missingness_flags: true

model_selection:
  candidates:
    - lightgbm
    - xgboost
    - random_forest
    - ridge
  scoring:
    - r2
    - neg_mean_absolute_error
  ranking_primary: r2
  cross_validation:
    n_splits: 5

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
  shap_plots:
    - summary
    - dependence
    - waterfall

deployment:
  enabled: false

artifacts:
  output_dir: artifacts
  save_format: joblib
  mlflow_tracking: false
```

---

## Pipeline Stages

The pipeline runs 12 stages in order:

| # | Stage | Description |
|---|-------|-------------|
| 1 | **Load** | Load data, detect task type, detect target |
| 2 | **EDA** | Column hygiene, quality, distributions, statistical tests |
| 3 | **Clean** | Remove duplicates, handle outliers, drop ID columns |
| 4 | **Feature Eng** | Interactions, log transforms, missingness flags |
| 5 | **Split** | Stratified train/test split |
| 6 | **Preprocess** | Impute, scale, encode (ColumnTransformer) |
| 7 | **Compare** | 14+ models head-to-head with cross-validation |
| 8 | **Tune** | Optuna hyperparameter optimization |
| 9 | **Select** | SHAP-based feature importance |
| 10 | **Evaluate** | Test set metrics (R², RMSE, MAE, etc.) |
| 11 | **Explain** | SHAP summary, dependence, waterfall plots |
| 12 | **Save** | Save full inference pipeline |

---

## Supported Models

### Linear Models
| Model | Task | Description |
|-------|------|-------------|
| `linear_regression` | Regression | Ordinary least squares |
| `ridge` | Regression | L2 regularization |
| `lasso` | Regression | L1 regularization (feature selection) |
| `elasticnet` | Regression | L1 + L2 regularization |
| `logistic_regression` | Classification | Linear classifier |

### Tree Models
| Model | Task | Description |
|-------|------|-------------|
| `decision_tree` | Both | Single decision tree |

### Bagging Models
| Model | Task | Description |
|-------|------|-------------|
| `random_forest` | Both | Bootstrap aggregating |
| `extra_trees` | Both | Extremely randomized trees |

### Boosting Models
| Model | Task | Description |
|-------|------|-------------|
| `xgboost` | Both | Gradient boosting (GPU support) |
| `lightgbm` | Both | Fast gradient boosting |
| `gradient_boosting` | Both | sklearn gradient boosting |
| `hist_gradient_boosting` | Both | Histogram-based (handles NaN) |
| `adaboost` | Both | Adaptive boosting |

### Instance-Based Models
| Model | Task | Description |
|-------|------|-------------|
| `knn` | Both | K-nearest neighbors |
| `svm` | Both | Support vector machine |

### Probabilistic Models
| Model | Task | Description |
|-------|------|-------------|
| `naive_bayes` | Classification | Gaussian naive Bayes |

### Ensemble Models
| Model | Task | Description |
|-------|------|-------------|
| `stacking` | Both | Stacking ensemble |
| `voting` | Both | Voting ensemble |

---

## Metrics

### Regression Metrics
| Metric | Description |
|--------|-------------|
| `test_r2` | R-squared (coefficient of determination) |
| `test_rmse` | Root mean squared error |
| `test_mae` | Mean absolute error |
| `test_mape` | Mean absolute percentage error |

### Classification Metrics
| Metric | Description |
|--------|-------------|
| `test_accuracy` | Accuracy score |
| `test_f1_macro` | F1 score (macro average) |
| `test_f1_weighted` | F1 score (weighted average) |
| `test_mcc` | Matthews correlation coefficient |
| `test_roc_auc` | ROC AUC score |

---

## Advanced Usage

### Accessing Pipeline Results

```python
ctx = run("data.csv", target="price")

# Best model name
print(ctx.best_model_name)  # "lightgbm"

# Best model object
model = ctx.best_model

# All metrics
print(ctx.metrics)

# Feature importance
print(ctx.feature_importance)

# Model comparison
print(ctx.metrics["model_comparison"])

# Stage history
for stage in ctx.stage_history:
    print(f"{stage.stage_name}: {stage.duration_seconds:.1f}s")

# Saved model path
print(ctx.reports["model_path"])  # "artifacts/model_v1.joblib"
```

### Custom Model Candidates

```python
from open_mlpipe import PipelineConfig, PipelineRunner

config = PipelineConfig(
    data={"path": "data.csv", "target": "price"},
    model_selection={
        "candidates": ["lightgbm", "xgboost"],  # Only these 2
        "scoring": ["r2"],
        "ranking_primary": "r2",
    },
)

runner = PipelineRunner(config)
ctx = runner.run()
```

### Disable Specific Stages

```python
config = PipelineConfig(
    data={"path": "data.csv", "target": "price"},
    tuning={"enabled": False},           # Skip tuning
    feature_selection={"enabled": False}, # Skip feature selection
    evaluation={"explainability": False}, # Skip SHAP
)
```

### Enable Deployment

```python
config = PipelineConfig(
    data={"path": "data.csv", "target": "price"},
    deployment={
        "enabled": True,
        "framework": "fastapi",
        "port": 8000,
        "docker": True,
    },
)

runner = PipelineRunner(config)
ctx = runner.run()
# Generates: main.py, Dockerfile, requirements.txt
```

---

## Troubleshooting

### Common Issues

**Import Error:**
```python
# Wrong
import mlpipe

# Correct
import open_mlpipe
# or
from open_mlpipe import run
```

**CLI Not Found:**
```bash
# Wrong
mlpipe run --data data.csv

# Correct
open-mlpipe run --data data.csv

# Or use Python module
python -m open_mlpipe.cli run --data data.csv
```

**Module Not Found:**
```bash
# Reinstall
pip install --upgrade open-mlpipe
```

**Version Check:**
```python
import open_mlpipe
print(open_mlpipe.__version__)
```

---

## Quick Reference

```python
# Simplest usage
from open_mlpipe import run
ctx = run("data.csv", target="price")

# One-liner prediction
import joblib
model = joblib.load("artifacts/model_v1.joblib")
predictions = model.predict(new_data)

# CLI
open-mlpipe run --data data.csv --target price
```
