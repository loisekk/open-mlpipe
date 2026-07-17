# Python API

## Simplest Usage

```python
from open_mlpipe import run

# One line — full pipeline
ctx = run("dataset.csv", target="price")
```

## Import Methods

```python
# Method 1: Import run function
from open_mlpipe import run

# Method 2: Import run_config for YAML
from open_mlpipe import run_config

# Method 3: Import classes for full control
from open_mlpipe import PipelineConfig, PipelineRunner

# Method 4: Import specific components
from open_mlpipe.config.schema import PipelineConfig, DataConfig
from open_mlpipe.core.pipeline import PipelineRunner
from open_mlpipe.utils.io import load_data
```

## Level 1: Zero-Touch

```python
from open_mlpipe import run

# Auto-detect everything
ctx = run("data.csv")
```

## Level 2: Specify Target

```python
from open_mlpipe import run

ctx = run("data.csv", target="price")

# Access results
print(f"Best model: {ctx.best_model_name}")
print(f"Test R²: {ctx.metrics['test_r2']:.4f}")
```

## Level 3: Custom Configuration

```python
from open_mlpipe import PipelineConfig, PipelineRunner
from open_mlpipe.config.resolver import build_level1_config

# Build config from data
config = build_level1_config("data.csv", target="price")

# Customize
config.project = "my-project"
config.tuning.enabled = True
config.tuning.n_trials = 50

# Run pipeline
runner = PipelineRunner(config)
ctx = runner.run()
```

## Level 4: Full Manual Control

```python
from open_mlpipe.config.schema import (
    PipelineConfig, DataConfig, ModelSelectionConfig,
    TuningConfig, FeatureSelectionConfig, EvaluationConfig,
)

config = PipelineConfig(
    project="custom-project",
    task="auto",
    data=DataConfig(path="data.csv", target="price"),
    model_selection=ModelSelectionConfig(
        candidates=["lightgbm", "xgboost"],
        scoring=["r2"],
    ),
    tuning=TuningConfig(enabled=True, n_trials=30),
    feature_selection=FeatureSelectionConfig(enabled=True),
    evaluation=EvaluationConfig(explainability=True),
)

runner = PipelineRunner(config)
ctx = runner.run()
```

## Level 5: Load and Predict

```python
import joblib
import pandas as pd

# Load saved model
model = joblib.load("artifacts/model_v1.joblib")

# Predict
new_data = pd.DataFrame({"feature1": [1.0], "feature2": ["a"]})
predictions = model.predict(new_data)
```

## API Reference

::: open_mlpipe.run
::: open_mlpipe.run_config
::: open_mlpipe.PipelineConfig
::: open_mlpipe.PipelineRunner
