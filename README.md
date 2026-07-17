<div align="center">

# 🔥 open-mlpipe

**Production-level automated ML pipeline — from raw data to deployed model in one line.**

[![PyPI version](https://img.shields.io/pypi/v/open-mlpipe.svg)](https://pypi.org/project/open-mlpipe/)
[![Python](https://img.shields.io/pypi/pyversions/open-mlpipe.svg)](https://pypi.org/project/open-mlpipe/)
[![Downloads](https://img.shields.io/pypi/dm/open-mlpipe.svg)](https://pypi.org/project/open-mlpipe/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/loisekk/open-mlpipe.svg)](https://github.com/loisekk/open-mlpipe)

---

**`pip install open-mlpipe`** · **14+ Models** · **Auto EDA** · **Optuna Tuning** · **SHAP Explainability** · **Production Ready**

---

[Quick Start](#-quick-start) · [Features](#-features) · [Models](#-supported-models) · [API](#-python-api) · [CLI](#-cli-usage) · [Config](#-yaml-config) · [Examples](#-examples)

</div>

---

## ⚡ Quick Start

### Install

```bash
pip install open-mlpipe
```

### One Line. Done.

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

---

## 🎯 What It Does

```
Raw Data → EDA → Clean → Feature Eng → Split → Preprocess
→ Compare 14 Models → Tune → Select → Evaluate → Explain → Save
```

**All automatic. All production-ready. One command.**

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

---

## 🤖 Supported Models

### Linear
| Model | Type |
|-------|------|
| Ridge | Regression |
| Lasso | Regression |
| ElasticNet | Regression |
| LinearRegression | Regression |
| LogisticRegression | Classification |

### Tree & Ensemble
| Model | Type |
|-------|------|
| DecisionTree | Both |
| RandomForest | Both |
| ExtraTrees | Both |

### Boosting
| Model | Type |
|-------|------|
| XGBoost | Both |
| LightGBM | Both |
| GradientBoosting | Both |
| HistGradientBoosting | Both |
| AdaBoost | Both |

### Instance & Probabilistic
| Model | Type |
|-------|------|
| KNN | Both |
| SVM | Both |
| NaiveBayes | Classification |

### Ensemble
| Model | Type |
|-------|------|
| Stacking | Both |
| Voting | Both |

**Auto-selected** based on data characteristics and task type.

---

## 🐍 Python API

### Simple Usage

```python
from open_mlpipe import run

ctx = run("data.csv", target="price")
```

### Advanced Usage

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

### Config-Driven

```python
from open_mlpipe import run_config

ctx = run_config("configs/regression.yaml")
```

---

## 💻 CLI Usage

### Commands

```bash
# Full pipeline
open-mlpipe run --data dataset.csv

# With options
open-mlpipe run --data dataset.csv --target price --project my-project

# Config-driven
open-mlpipe run --config configs/regression.yaml

# EDA only (no training)
open-mlpipe profile --data dataset.csv
```

### Options

```
--data, -d       Path to data file (CSV, Parquet, Excel)
--target, -t     Target column name (auto-detected if not given)
--level, -l      Automation level (1=zero-touch, 2=config-driven)
--config, -c     Path to YAML config file
--project, -p    Project name
--deploy         Generate deployment artifacts
```

---

## 📋 YAML Config

```yaml
# configs/regression.yaml
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

Run it:

```bash
open-mlpipe run --config configs/regression.yaml
```

---

## 📊 Example Output

```
=== open-mlpipe Pipeline Runner ===
  Project:  california-housing
  Data:     california_housing.csv
  Target:   median_house_value

>> load        OK (0.1s)
>> eda         OK (0.2s)
>> clean       OK (0.0s)
>> feature_eng OK (0.0s)
>> split       OK (0.0s)
>> preprocess  OK (0.0s)
>> compare     OK (25.9s)  — tested 14 models
>> tune        OK (20.2s)  — Optuna 20 trials
>> select      OK (0.0s)
>> evaluate    OK (0.0s)
>> explain     OK (3.3s)
>> save        OK (0.1s)

=== Pipeline Complete ===
  Best Model:  lightgbm
  Test R²:     0.8474
  Test MAE:    0.2924
  Test MAPE:   16.77%
```

---

## 🏗️ Architecture

```
open-mlpipe/
├── src/open_mlpipe/
│   ├── cli.py              # CLI entry point
│   ├── config/             # Pydantic config + YAML resolver
│   ├── core/               # Pipeline runner, context, stages
│   ├── stages/             # 12 pipeline stages
│   │   ├── load.py         # Data loading
│   │   ├── eda.py          # Exploratory data analysis
│   │   ├── clean.py        # Data cleaning
│   │   ├── feature_eng.py  # Feature engineering
│   │   ├── split.py        # Train/test split
│   │   ├── preprocess.py   # Preprocessing pipeline
│   │   ├── compare.py      # Model comparison
│   │   ├── tune.py         # Hyperparameter tuning
│   │   ├── select.py       # Feature selection
│   │   ├── evaluate.py     # Model evaluation
│   │   ├── explain.py      # SHAP explainability
│   │   └── save.py         # Model saving
│   ├── utils/              # I/O, feature engineering
│   └── deploy/             # FastAPI + Docker generation
├── tests/                  # 106 unit tests + 3 integration tests
├── configs/                # 5 example YAML configs
└── pyproject.toml          # Package config
```

---

## 📦 Installation Options

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

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run unit tests only
pytest tests/ -m "not slow"

# Run with coverage
pytest tests/ --cov=open_mlpipe
```

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🔗 Links

| Resource | Link |
|----------|------|
| **Website** | [open-mlpipe.onrender.com](https://open-mlpipe.onrender.com/) |
| **Documentation** | [loisekk.github.io/open-mlpipe](https://loisekk.github.io/open-mlpipe/) |
| **PyPI** | [pypi.org/project/open-mlpipe](https://pypi.org/project/open-mlpipe/) |
| **Issues** | [github.com/loisekk/open-mlpipe/issues](https://github.com/loisekk/open-mlpipe/issues) |

---

<div align="center">

**Built with ❤️ by [Yash Brahmankar](https://github.com/loisekk)**

*If this saved you time, ⭐ star the repo and share it!*

</div>
