# Metrics

## Regression Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| `test_r2` | R-squared (coefficient of determination) | -∞ to 1 (higher is better) |
| `test_rmse` | Root mean squared error | 0 to ∞ (lower is better) |
| `test_mae` | Mean absolute error | 0 to ∞ (lower is better) |
| `test_mape` | Mean absolute percentage error | 0% to ∞% (lower is better) |

## Classification Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| `test_accuracy` | Accuracy score | 0 to 1 (higher is better) |
| `test_f1_macro` | F1 score (macro average) | 0 to 1 (higher is better) |
| `test_f1_weighted` | F1 score (weighted average) | 0 to 1 (higher is better) |
| `test_mcc` | Matthews correlation coefficient | -1 to 1 (higher is better) |
| `test_roc_auc` | ROC AUC score | 0 to 1 (higher is better) |

## Accessing Metrics

```python
ctx = run("data.csv", target="price")

# Access specific metric
print(ctx.metrics["test_r2"])

# Access all metrics
print(ctx.metrics)

# Model comparison
print(ctx.metrics["model_comparison"])
```

## Overfitting Detection

open-mlpipe automatically detects overfitting:

- **Gap > 0.1**: Warning — model may be overfitting
- **Gap < 0.01 and low score**: Warning — model may be underfitting
