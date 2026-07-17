# Configuration

## Default Configuration

open-mlpipe works out of the box with smart defaults. No configuration needed.

```python
from open_mlpipe import run

# Auto-detect everything
ctx = run("dataset.csv")
```

## Custom Configuration

### Python API

```python
from open_mlpipe import PipelineConfig, PipelineRunner

config = PipelineConfig(
    project="my-project",
    task="auto",
    data={"path": "data.csv", "target": "price"},
    model_selection={
        "candidates": ["lightgbm", "xgboost"],
        "scoring": ["r2"],
    },
    tuning={"enabled": True, "n_trials": 50},
)

runner = PipelineRunner(config)
ctx = runner.run()
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
  candidates: [lightgbm, xgboost]
tuning:
  enabled: true
  n_trials: 50
```

```bash
open-mlpipe run --config configs/my_pipeline.yaml
```

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `project` | `"mlpipe-run"` | Project name |
| `task` | `"auto"` | Task type (auto/regression/classification) |
| `data.test_size` | `0.2` | Test set size |
| `tuning.enabled` | `True` | Enable hyperparameter tuning |
| `tuning.n_trials` | `"auto"` | Number of Optuna trials |
| `feature_selection.enabled` | `True` | Enable feature selection |
| `evaluation.explainability` | `True` | Enable SHAP explainability |
| `deployment.enabled` | `False` | Generate deployment artifacts |

## Next Steps

- [Python API](../guide/python-api.md)
- [YAML Config](../guide/yaml-config.md)
