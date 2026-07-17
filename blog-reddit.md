# REDDIT POST (r/MachineLearning)

---

**[P] I built open-mlpipe — AutoML pipeline that trains 14 models in one line of Python**

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

**Architecture:**
```
open-mlpipe/
├── src/open_mlpipe/
│   ├── cli.py              # CLI entry point
│   ├── config/             # Pydantic config + YAML resolver
│   ├── core/               # Pipeline runner, context, stages
│   ├── stages/             # 12 pipeline stages
│   ├── utils/              # I/O, feature engineering
│   └── deploy/             # FastAPI + Docker generation
├── tests/                  # 106 unit tests + 3 integration tests
├── configs/                # 5 example YAML configs
└── pyproject.toml          # Package config
```

Would love feedback! What features would you add?

---

*Disclaimer: I'm the author. This is open-source under MIT license.*
