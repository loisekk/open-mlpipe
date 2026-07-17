---
title: I Built an AutoML Pipeline That Trains 14 Models in One Line of Python
subtitle: From raw data to production model — no PhD required
slug: open-mlpipe-automl-pipeline
tags: python, machinelearning, automl, data-science, opensource
---

# I Built an AutoML Pipeline That Trains 14 Models in One Line of Python

**From raw data to production model — no PhD required**

Every ML project starts the same way: load data, clean it, try 10 models, pick the best one. I got tired of repeating this. So I built **open-mlpipe** — an AutoML pipeline that does it all in one line.

## The Pain Point

We've all been there. Starting a new ML project means:

1. Loading data
2. Cleaning it
3. Engineering features
4. Trying 10 different models
5. Tuning hyperparameters
6. Picking the best one
7. Saving it for production

**Hours wasted. Every. Single. Time.**

## Enter open-mlpipe

```python
from open_mlpipe import run

# One line. Full pipeline.
ctx = run("dataset.csv", target="price")

# Access results
print(f"Best model: {ctx.best_model_name}")
print(f"Test R²: {ctx.metrics['test_r2']:.4f}")
```

That's it. One line. Production model.

## What Happens Under the Hood

```
Raw Data → EDA → Clean → Feature Eng → Split → Preprocess
→ Compare 14 Models → Tune → Select → Evaluate → Explain → Save
```

All automatic. All production-ready.

## 14+ Models Compared

### Linear
- Ridge, Lasso, ElasticNet, LinearRegression
- LogisticRegression (classification)

### Tree & Ensemble
- DecisionTree, RandomForest, ExtraTrees

### Boosting
- **XGBoost** — King of tabular benchmarks
- **LightGBM** — Fastest GBM, best for large data
- **GradientBoosting** — sklearn's classic
- **HistGradientBoosting** — sklearn's answer to LightGBM
- **AdaBoost** — Sequential boosting

### Instance & Probabilistic
- KNN, SVM, NaiveBayes

### Ensemble
- **Stacking** — Base models → meta-learner
- **Voting** — Soft voting across diverse models

## Real Results

On California Housing (20K rows):

| Metric | Value |
|--------|-------|
| Best Model | LightGBM |
| R² | 0.847 |
| Time | 4.5 min |

### Model Comparison

| Model | R² | Gap | Status |
|-------|-----|-----|--------|
| LightGBM | 0.839 | 0.072 | ✅ Best |
| XGBoost | 0.835 | 0.093 | ✅ Good |
| RandomForest | 0.771 | 0.198 | ⚠️ Overfit |
| Ridge | 0.709 | 0.001 | ✅ Stable |

## Get Started

```bash
pip install open-mlpipe
```

## Links

- [PyPI](https://pypi.org/project/open-mlpipe/)
- [GitHub](https://github.com/loisekk/open-mlpipe)

---

*Star the repo if this helps!*
