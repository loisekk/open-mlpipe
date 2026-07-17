# FAQ

## General

### What is open-mlpipe?

open-mlpipe is a production-level automated ML pipeline that takes a raw dataset and automatically handles the entire machine learning workflow: EDA, cleaning, feature engineering, model comparison, hyperparameter tuning, explainability, and saving a production-ready pipeline.

### When should I use open-mlpipe?

Use open-mlpipe when you want to quickly benchmark multiple models on a tabular dataset without writing boilerplate code. It's ideal for:

- Rapid prototyping and baseline establishment
- AutoML for regression and classification tasks
- Getting a production-ready model in minutes instead of hours
- Comparing 14+ models head-to-head

### What tasks does open-mlpipe support?

- **Regression** — predicting continuous values (house prices, revenue, etc.)
- **Classification** — predicting categories, both binary and multiclass (fraud detection, churn prediction, etc.)

### Does it work with my data format?

open-mlpipe supports:

- **CSV** files
- **Parquet** files
- **Excel** files (.xlsx, .xls)

Just pass the file path and open-mlpipe handles the rest.

---

## Installation

### What Python versions are supported?

Python 3.10, 3.11, 3.12, and 3.13.

### How do I install with all extras?

```bash
pip install open-mlpipe[full]
```

This installs CatBoost, MLflow tracking, FastAPI deployment, and visualization extras.

### Do I need GPU support?

No. open-mlpipe works on CPU. Some models (like XGBoost) can optionally use GPU if available, but it's not required.

---

## Models and Performance

### How many models does open-mlpipe compare?

14+ models including:

- **Linear:** Ridge, Lasso, ElasticNet, Linear/Logistic Regression
- **Tree:** Decision Tree
- **Bagging:** Random Forest, Extra Trees
- **Boosting:** XGBoost, LightGBM, Gradient Boosting, Hist Gradient Boosting, AdaBoost
- **Instance-based:** KNN, SVM
- **Probabilistic:** Naive Bayes
- **Ensemble:** Stacking, Voting

### Which model is usually best?

It depends on the dataset, but **LightGBM** and **XGBoost** are typically among the top performers for both regression and classification tasks on tabular data.

### How does hyperparameter tuning work?

open-mlpipe uses **Optuna** for Bayesian hyperparameter optimization with 18 model-specific search spaces. Each model gets tuned with parameters appropriate for its architecture.

### Can I limit which models are tested?

Yes, specify candidates in the configuration:

```python
config = PipelineConfig(
    data={"path": "data.csv", "target": "target"},
    model_selection={
        "candidates": ["lightgbm", "xgboost"],
    },
)
```

---

## Configuration

### Do I need to write configuration?

No. open-mlpipe works with zero configuration:

```python
from open_mlpipe import run

ctx = run("data.csv", target="price")
```

Configuration is only needed when you want to customize behavior.

### How do I configure the pipeline?

Three ways:

1. **Python API** — pass parameters to `PipelineConfig`
2. **YAML files** — write a config YAML and pass it to `run_config()`
3. **CLI flags** — use command-line options with `open-mlpipe run`

### Can I save and reuse configurations?

Yes. Write a YAML config file and reuse it:

```yaml
# my_config.yaml
project: my-project
data:
  path: data.csv
  target: price
tuning:
  enabled: true
  n_trials: 50
```

```bash
open-mlpipe run --config my_config.yaml
```

---

## Output and Deployment

### What does open-mlpipe save?

- **Model file** (`artifacts/model_v1.joblib`) — the full inference pipeline
- **EDA report** — data profiling summary
- **Metrics JSON** — all evaluation metrics
- **SHAP plots** — feature importance visualizations
- **Model comparison** — cross-validation results for all models

### How do I load and use the saved model?

```python
import joblib
import pandas as pd

model = joblib.load("artifacts/model_v1.joblib")
predictions = model.predict(new_data)
```

### Can I deploy the model as an API?

Yes. Install the deploy extras and use the built-in FastAPI support:

```bash
pip install open-mlpipe[deploy]
open-mlpipe run --data dataset.csv --deploy
```

---

## Troubleshooting

### The pipeline is slow. How can I speed it up?

- Reduce the number of Optuna trials: `tuning={"n_trials": 20}`
- Use fewer models: `model_selection={"candidates": ["lightgbm"]}`
- Disable feature selection: `feature_selection={"enabled": False}`
- Use Level 1 (zero-touch) with smart defaults

### I'm getting "target column not found" error

The target column must exist in your dataset. Check:

1. The column name is spelled correctly
2. There are no extra spaces in the name
3. The column exists in the CSV/Parquet file

### How do I report a bug or request a feature?

Open an issue on [GitHub](https://github.com/loisekk/open-mlpipe/issues).

---

[Back to Home :octicons-arrow-right-24:](index.md)
