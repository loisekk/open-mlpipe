# LINKEDIN VERSION

---

🚀 **I just published my first Python library to PyPI!**

After months of building ML pipelines from scratch, I got tired of repeating the same steps: load data → clean → feature engineer → try 10 models → tune → pick best → save.

So I built **open-mlpipe** — an AutoML pipeline that does it ALL in one line of Python.

**What it does automatically:**
✅ Loads & profiles your data (EDA)
✅ Cleans duplicates, outliers, missing values
✅ Engineers features (interactions, log transforms)
✅ Compares 14+ ML models head-to-head
✅ Tunes hyperparameters with Optuna
✅ Selects features via SHAP importance
✅ Saves a production-ready inference pipeline

**One line of code:**
```python
from open_mlpipe import run
ctx = run("dataset.csv", target="price")
```

**Results on California Housing (20K rows):**
• Best model: LightGBM (auto-selected)
• R²: 0.847
• Time: 4.5 minutes

**Install:**
```bash
pip install open-mlpipe
```

**Links:**
🔗 PyPI: https://pypi.org/project/open-mlpipe/
🔗 GitHub: https://github.com/loisekk/open-mlpipe

#Python #MachineLearning #AutoML #OpenSource #DataScience #MLOps

---

*If you work with ML pipelines, give it a try. Star the repo if you find it useful!*
