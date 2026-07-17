# Quick Start

## 30-Second Start

```python
from open_mlpipe import run

# One line — full pipeline
ctx = run("dataset.csv", target="price")
```

That's it. The pipeline will:

1. Load your data
2. Run EDA
3. Clean the data
4. Engineer features
5. Compare 14+ models
6. Tune the best model
7. Save a production-ready pipeline

## What You Get Back

```python
# Best model name
print(ctx.best_model_name)  # "lightgbm"

# All metrics
print(ctx.metrics)
# {'test_r2': 0.847, 'test_mae': 0.292, 'test_rmse': 0.447, ...}

# Saved model path
print(ctx.reports["model_path"])  # "artifacts/model_v1.joblib"
```

## Load and Use the Saved Model

```python
import joblib
import pandas as pd

# Load the saved model
model = joblib.load("artifacts/model_v1.joblib")

# Prepare new data
new_data = pd.DataFrame({
    "feature1": [1.0, 2.0, 3.0],
    "feature2": ["a", "b", "c"],
})

# Predict
predictions = model.predict(new_data)
```

## CLI Alternative

```bash
open-mlpipe run --data dataset.csv --target price
```

## Next Steps

- [Python API](../guide/python-api.md) — Full API documentation
- [Configuration](configuration.md) — Customize the pipeline
- [YAML Config](../guide/yaml-config.md) — Config-driven pipelines
