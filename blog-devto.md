# DEV.TO VERSION

---

# I Built an AutoML Pipeline That Trains 14 Models in One Line of Python

Every ML project starts the same way: load data, clean it, try 10 models, pick the best one. I got tired of repeating this. So I built **open-mlpipe** — an AutoML pipeline that does it all in one line.

## The Problem

```python
# What I used to do every time
df = pd.read_csv("data.csv")
# ... 50 lines of cleaning ...
# ... 100 lines of model comparison ...
# ... 50 lines of hyperparameter tuning ...
# ... 30 lines of evaluation ...
# Hours wasted. Every. Single. Time.
```

## The Solution

```python
from open_mlpipe import run

# One line. That's it.
ctx = run("dataset.csv", target="price")

# Done. Production model saved.
```

## What It Does (Automatically)

1. **Loads & profiles** your data (EDA)
2. **Cleans** duplicates, outliers, missing values
3. **Engineers features** (interactions, log transforms, missingness flags)
4. **Compares 14+ models** head-to-head
5. **Tunes hyperparameters** with Optuna
6. **Selects features** via SHAP importance
7. **Explains predictions** with SHAP
8. **Saves a production-ready pipeline** (feature engineering + model)

## Supported Models

| Category | Models |
|----------|--------|
| Linear | Ridge, Lasso, ElasticNet |
| Tree | DecisionTree |
| Bagging | RandomForest, ExtraTrees |
| Boosting | XGBoost, LightGBM, GradientBoosting, HistGradientBoosting, AdaBoost |
| Instance | KNN, SVM |
| Ensemble | Stacking, Voting |

## Real Results

Tested on California Housing (20K rows):

```
Best Model:  lightgbm
Test R²:     0.8474
Total Time:  4.5 minutes
```

## Install

```bash
pip install open-mlpipe
```

## Links

- **PyPI**: https://pypi.org/project/open-mlpipe/
- **GitHub**: https://github.com/loisekk/open-mlpipe

---

*If this saved you time, star the repo!*
