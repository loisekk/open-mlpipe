# REDDIT VERSION (r/MachineLearning or r/Python)

---

**[Project] I built open-mlpipe — AutoML pipeline that trains 14 models in one line of Python**

Hey everyone,

I got tired of building the same ML pipeline from scratch for every project: load data → clean → feature engineer → try models → tune → save.

So I built **open-mlpipe** — an AutoML pipeline that does it all in one line:

```python
from open_mlpipe import run
ctx = run("dataset.csv", target="price")
```

**What it does automatically:**
- EDA (column hygiene, quality, distributions, statistical tests)
- Data cleaning (duplicates, outliers, missing values)
- Feature engineering (interactions, log transforms, missingness flags)
- Compares 14+ models: Ridge, Lasso, ElasticNet, DecisionTree, RandomForest, ExtraTrees, XGBoost, LightGBM, GradientBoosting, HistGradientBoosting, AdaBoost, KNN, SVM, Stacking, Voting
- Hyperparameter tuning with Optuna
- SHAP explainability
- Saves a full inference pipeline (feature engineering + model)

**Tested on California Housing (20K rows):**
- Best model: LightGBM (auto-selected)
- R²: 0.847
- Time: 4.5 minutes

**Install:**
```
pip install open-mlpipe
```

**Links:**
- PyPI: https://pypi.org/project/open-mlpipe/
- GitHub: https://github.com/loisekk/open-mlpipe

Would love feedback! What features would you add?
