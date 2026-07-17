# YAML Config

## Minimal Config

```yaml
project: my-project
data:
  path: data.csv
  target: price
```

## Full Config

```yaml
project: insurance-charges
task: auto
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

## Run from Config

```bash
open-mlpipe run --config configs/regression.yaml
```

```python
from open_mlpipe import run_config

ctx = run_config("configs/regression.yaml")
```

## Next Steps

- [Python API](python-api.md)
- [CLI Usage](cli.md)
