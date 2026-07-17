# X / TWITTER VERSION (Thread)

---

**Tweet 1 (Hook):**
🚀 I just published my first Python library to PyPI!

I built open-mlpipe — an AutoML pipeline that trains 14+ models in ONE line of Python.

From raw data → production model in 4.5 minutes.

Here's what it does 🧵👇

---

**Tweet 2 (Problem):**
Every ML project:
1. Load data
2. Clean it
3. Feature engineer
4. Try 10 models
5. Tune hyperparameters
6. Pick best
7. Save

Hours wasted. Every. Single. Time.

I wanted ONE line that does it all.

---

**Tweet 3 (Solution):**
```python
from open_mlpipe import run

# That's it. One line.
ctx = run("dataset.csv", target="price")

# Production model saved.
```

No PhD required. 🎓

---

**Tweet 4 (What it does):**
What open-mlpipe does automatically:

✅ EDA (quality, distributions, tests)
✅ Clean (duplicates, outliers, missing)
✅ Feature engineering (interactions, log, flags)
✅ Compare 14+ models
✅ Tune with Optuna
✅ SHAP explainability
✅ Save production pipeline

---

**Tweet 5 (Models):**
14+ models compared:

Linear: Ridge, Lasso, ElasticNet
Tree: DecisionTree
Bagging: RandomForest, ExtraTrees
Boosting: XGBoost, LightGBM, HistGB, AdaBoost
Instance: KNN, SVM
Ensemble: Stacking, Voting

All auto-selected. 🤖

---

**Tweet 6 (Results):**
Real results on California Housing (20K rows):

Best Model: LightGBM
R²: 0.847
Time: 4.5 min

From raw CSV → production model. Done. ✅

---

**Tweet 7 (Install):**
Install:

```
pip install open-mlpipe
```

Try it now. Star the repo if you like it! ⭐

PyPI: pypi.org/project/open-mlpipe/
GitHub: github.com/loisekk/open-mlpipe

---

**Tweet 8 (CTA):**
If you work with ML pipelines, give open-mlpipe a try.

Feedback welcome! What features should I add next?

#Python #MachineLearning #AutoML #OpenSource #DataScience
