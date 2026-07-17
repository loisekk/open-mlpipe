---
hide:
  - navigation
  - toc
---

<div class="hero-banner" markdown>

<div class="hero-badge">v1.0.0 &middot; MIT License</div>

# open-mlpipe

<p class="hero-subtitle">
Production-level automated ML pipeline &mdash; from raw data to deployed model
in one line. 14+ models, Auto EDA, Optuna tuning, SHAP explainability.
</p>

<div class="hero-code" markdown>

```python
pip install open-mlpipe
```

```python
from open_mlpipe import run

ctx = run("dataset.csv", target="price")
print(f"Best model: {ctx.best_model_name}")
```

</div>

<div class="hero-buttons" markdown>

[:material-rocket-launch: Get Started](getting-started/installation.md){ .btn-primary }
[:material-github: GitHub](https://github.com/loisekk/open-mlpipe){ .btn-secondary }

</div>

</div>

## Why open-mlpipe?

<div class="feature-grid" markdown>

<div class="feature-card" markdown>

:material-lightning-bolt:{ .feature-icon }

**One-Line Pipelines**

No boilerplate. Feed in a CSV, get a production model back. The entire
workflow from EDA to deployment runs automatically.

</div>

<div class="feature-card" markdown>

:material-chart-box:{ .feature-icon }

**14+ Models Compared**

Ridge, XGBoost, LightGBM, RandomForest, CatBoost, SVM, and more compete
head-to-head so you always get the best performer.

</div>

<div class="feature-card" markdown>

:material-tune-variant:{ .feature-icon }

**Smart Hyperparameter Tuning**

Optuna-powered optimization with 18 model-specific search spaces. Finds
better parameters in fewer trials.

</div>

<div class="feature-card" markdown>

:material-eye:{ .feature-icon }

**SHAP Explainability**

Global and local feature importance. Understand *why* your model makes
every prediction with publication-ready plots.

</div>

<div class="feature-card" markdown>

:material-cog:{ .feature-icon }

**Full Pipeline Saved**

Not just the model &mdash; the complete inference pipeline including feature
engineering, preprocessing, and encoding.

</div>

<div class="feature-card" markdown>

:material-shield-check:{ .feature-icon }

**Production Ready**

Built-in overfitting detection, automatic metric selection, and deployment
artifacts. Ship with confidence.

</div>

</div>

---

## Quick Start

<div class="step-grid" markdown>

<div class="step-item" markdown>

<div class="step-number">1</div>

<div class="step-content" markdown>

### Install

```bash
pip install open-mlpipe
```

</div>

</div>

<div class="step-item" markdown>

<div class="step-number">2</div>

<div class="step-content" markdown>

### Run

```python
from open_mlpipe import run

ctx = run("dataset.csv", target="price")
```

</div>

</div>

<div class="step-item" markdown>

<div class="step-number">3</div>

<div class="step-content" markdown>

### Results

```python
print(f"Best model: {ctx.best_model_name}")
print(f"Test R²: {ctx.metrics['test_r2']:.4f}")
```

</div>

</div>

<div class="step-item" markdown>

<div class="step-number">4</div>

<div class="step-content" markdown>

### Deploy

```python
import joblib
model = joblib.load(ctx.reports["model_path"])
predictions = model.predict(new_data)
```

</div>

</div>

</div>

---

## What It Does

| Stage | What Happens |
|-------|-------------|
| **Load & EDA** | Profile data, detect types, run statistical tests |
| **Clean** | Remove duplicates, handle outliers, drop ID columns |
| **Feature Engineering** | Interactions, log transforms, missingness flags |
| **Compare** | 14+ models head-to-head with cross-validation |
| **Tune** | Optuna hyperparameter optimization |
| **Select** | SHAP-based feature importance |
| **Evaluate** | R², RMSE, MAE, MAPE, F1, ROC-AUC, MCC |
| **Explain** | SHAP summary, dependence, waterfall plots |
| **Save** | Full inference pipeline saved to disk |

---

## Installation

```bash
pip install open-mlpipe
```

**Optional extras:**

```bash
pip install open-mlpipe[catboost]    # CatBoost support
pip install open-mlpipe[mlflow]      # MLflow tracking
pip install open-mlpipe[deploy]      # FastAPI + Docker
pip install open-mlpipe[full]        # Everything
```

---

## Explore the Docs

<div class="feature-grid" markdown>

<div class="feature-card" markdown>

:material-rocket-launch:{ .feature-icon }

**Getting Started**

Installation, quick start, and configuration guide.

[:octicons-arrow-right-24: Get Started](getting-started/installation.md)

</div>

<div class="feature-card" markdown>

:material-book-open-variant:{ .feature-icon }

**User Guide**

Python API, CLI usage, YAML configuration, and more.

[:octicons-arrow-right-24: Read the Guide](guide/overview.md)

</div>

<div class="feature-card" markdown>

:material-api:{ .feature-icon }

**API Reference**

Complete auto-generated API documentation.

[:octicons-arrow-right-24: Browse API](api/open_mlpipe.md)

</div>

<div class="feature-card" markdown>

:material-test-tube:{ .feature-icon }

**Examples**

Real-world examples for regression, classification, and custom pipelines.

[:octicons-arrow-right-24: See Examples](examples/regression.md)

</div>

</div>

---

<div style="text-align: center; padding: 2rem 0; color: var(--md-typeset-fg-color--light);">

**open-mlpipe** is open source under the MIT License.
Contributions welcome on [GitHub](https://github.com/loisekk/open-mlpipe).

</div>
