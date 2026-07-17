# mlpipe — Production ML Pipeline Plan

> **Goal**: Raw data → Deployed model in hours, not weeks. + LLM Agent for zero-code ML.
> **Status**: Phase 1 complete (12 stages, 32 files, 39 bugs fixed).
> **Inspired by**: FINAL-ML-Workflow (18 phases), HuggingFace ml-intern (LLM agent architecture).

---

## Master Pipeline Map

```
RAW DATA
    │
    ▼
PHASE 0 → DEFINE PROBLEM
          task type | metric | cost of error | business baseline
    │
    ▼
PHASE 1 → LOAD DATA
          CSV | Parquet | Excel | DB | API | Kaggle | BigQuery
    │
    ▼
PHASE 2 → EDA
          shape | missing | target dist | imbalance check
          distributions | outliers | correlations | categoricals
          statistical tests (Z/T/ANOVA/Chi-Square) | VIF
    │
    ▼
PHASE 3 → DATA CLEANING
          duplicates | type fixes | outlier treatment | column hygiene
    │
    ▼
PHASE 4 → FEATURE ENGINEERING
          skew detection | domain features | interactions | date
          missingness signal | target encoding | polynomial features
    │
    ├──── [AUTO PATH] ──────────────────────────────────────────┐
    │     TabPFN (< 10k rows) or AutoGluon → baseline in mins   │
    │     → use this as ceiling to beat with manual pipeline    │
    │                                                           │
    ▼ [MANUAL PATH]                                             │
PHASE 5 → TRAIN / TEST SPLIT  ← NOTHING FIT BEFORE THIS LINE    │
          stratify=y | time-aware split | 80/20                 │
    │                                                           │
    ▼                                                           │
PHASE 6 → IMBALANCE HANDLING (classification, if needed)        │
          ratio < 4:1 → class_weight='balanced' only            │
          ratio 4-20:1 → SMOTE + class_weight                   │
          ratio > 20:1 → ADASYN / SMOTETomek + class_weight     │
    │                                                           │
    ▼                                                           │
PHASE 7 → PREPROCESSING PIPELINE (ColumnTransformer)            │
          skewed → PowerTransformer → StandardScaler            │
          normal → StandardScaler                               │
          categorical → OHE / TargetEncoder / OrdinalEncoder    │
    │                                                           │
    ▼                                                           │
PHASE 8 → BASELINE MODEL                                        │
          DummyClassifier / DummyRegressor / LogReg / Ridge     │◄──┘
    │
    ▼
PHASE 9 → CV + MODEL COMPARISON
          StratifiedKFold | TimeSeriesSplit | RepeatedCV
          ExtraTrees | HistGB | CatBoost | Stacking | Voting
    │
    ▼
PHASE 10 → HYPERPARAMETER TUNING
           GridSearchCV | Optuna (TPE + HyperBand) | HalvingGridSearch
    │
    ▼
PHASE 11 → FEATURE SELECTION (optional)
           RFECV | SelectKBest | VarianceThreshold | SHAP-guided
    │
    ▼
PHASE 12 → SHAP EXPLAINABILITY
           TreeExplainer | KernelExplainer
           global | local | dependence | SHAP feature selection
    │
    ▼
PHASE 13 → MODEL CALIBRATION (classification only)
           CalibratedClassifierCV | Platt scaling | Isotonic
           + threshold optimization
    │
    ▼
PHASE 14 → FINAL EVALUATION  ← TEST SET, ONCE, HERE
           all metrics | confusion matrix | residuals
           calibration curve | overfitting gap analysis
    │
    ▼
PHASE 15 → SAVE PIPELINE
           joblib.dump | version naming | DVC tracking
           MLflow: log params + metrics + register model
    │
    ▼
PHASE 16 → MODEL CARD  ← write before deploying
    │
    ▼
PHASE 17 → DEPLOY
           FastAPI | Docker | MLflow Serving | AWS ECS
    │
    ▼
PHASE 18 → MONITOR + RETRAIN
           Evidently drift | retrain triggers
           Airflow/Prefect DAG | canary deploy | rollback
```

---

## Current State (Phase 1 — DONE)

| Component | Status | Notes |
|-----------|--------|-------|
| CLI | Done | `click` + `rich`, `run` and `profile` commands |
| Config | Done | Pydantic models, smart defaults, YAML resolver |
| Pipeline Runner | Done | Sequential stage execution with timing |
| 12 Stages | Done | load → eda → clean → feature_eng → split → preprocess → compare → tune → select → evaluate → explain → save |
| EDA Engine | Done | 5 phases: column hygiene, quality, distributions, stat tests, VIF |
| Bug Fixes | Done | 39 bugs found and fixed (4 critical, 5 high, 5 medium) |
| Tests | **EMPTY** | No pytest tests |
| Git | **MISSING** | No `.git` repo |
| Level 2 | **MISSING** | Config-driven YAML |
| Level 3 | **MISSING** | Manual scaffold |
| 18-Phase Gaps | **5 stages missing** | baseline, imbalance, calibrate, model_card, monitor |
| LLM Agent | **MISSING** | No LLM provider integration |
| Frontend | **MISSING** | No web UI |
| Deploy | **STUB** | Generates FastAPI + Dockerfile but no real deployment |
| Monitoring | **MISSING** | No Evidently drift detection |

---

## Workflow Phase → mlpipe Stage Mapping

| Phase | Workflow Name | mlpipe Stage | Status | Gap |
|-------|--------------|--------------|--------|-----|
| 0 | Define Problem | TaskDetector (defaults.py) | Partial | No business baseline, no cost-of-error |
| 1 | Load Data | LoadStage | Done | Need Parquet, SQL, API, Kaggle |
| 2 | EDA | EDAStage | Done | Missing: cardinality routing decisions |
| 3 | Clean | CleanStage | Done | Missing: dtype fix for string-numbers |
| 4 | Feature Eng | FeatureEngStage | Done | Missing: polynomial, target encoding, binning |
| 4B | Auto Path | **NEW: AutoStage** | Missing | TabPFN, AutoGluon baseline |
| 5 | Split | SplitStage | Done | Missing: time-series, small dataset adjust |
| 6 | Imbalance | **NEW: ImbalanceStage** | Missing | SMOTE, ADASYN, SMOTETomek |
| 7 | Preprocess | PreprocessStage | Done | Missing: RobustScaler, IterativeImputer |
| 8 | Baseline | **NEW: BaselineStage** | Missing | Dummy + analytical baselines |
| 9 | CV + Compare | CompareStage | Done | Missing: ExtraTrees, CatBoost, Stacking, Voting |
| 10 | Tune | TuneStage | Done | Missing: GridSearchCV option |
| 11 | Feature Selection | SelectStage | Done | Missing: RFECV, VarianceThreshold |
| 12 | SHAP | ExplainStage | Done | Missing: KernelExplainer, dependence plot |
| 13 | Calibration | **NEW: CalibrateStage** | Missing | CalibratedClassifierCV, threshold tuning |
| 14 | Final Eval | EvaluateStage | Done | Missing: overfitting diagnosis |
| 15 | Save | SaveStage | Done | Missing: DVC, version naming |
| 16 | Model Card | **NEW: ModelCardStage** | Missing | Auto-generated markdown |
| 17 | Deploy | DeployStage | Stub | Need working FastAPI + Docker |
| 18 | Monitor | **NEW: MonitorStage** | Missing | Evidently drift, retrain triggers |

---

## Phase Plan

### Phase 2: Testing & Quality Gate

**Goal**: Make Phase 1 provably correct.

```
tests/
├── conftest.py                  # Shared fixtures (DataFrames, configs, contexts)
├── test_load.py                 # 4 tests
├── test_eda.py                  # 8 tests
├── test_clean.py                # 6 tests
├── test_feature_eng.py          # 5 tests
├── test_split.py                # 5 tests
├── test_preprocess.py           # 6 tests
├── test_compare.py              # 5 tests
├── test_tune.py                 # 4 tests
├── test_select.py               # 3 tests
├── test_evaluate.py             # 5 tests
├── test_explain.py              # 3 tests
├── test_save.py                 # 4 tests
├── test_deploy.py               # 3 tests
├── test_config.py               # 8 tests
├── test_context.py              # 4 tests
├── test_cli.py                  # 4 tests
└── test_pipeline.py             # 5 integration tests

Total: ~85 tests
```

Config additions:
- `[tool.pytest.ini_options]` in pyproject.toml
- `[tool.ruff]` with `select = ["E", "F", "I", "N", "W", "UP"]`
- `[tool.mypy]` with `disallow_untyped_defs = true`
- `.pre-commit-config.yaml` with ruff + mypy hooks

Exit criteria: `pytest tests/ -v` 100% pass, `ruff check .` clean.

---

### Phase 3: Level 2 — Config-Driven YAML

**Goal**: User defines config in YAML, pipeline executes.

```yaml
# configs/regression-default.yaml
project: insurance-charges
task: auto
data:
  path: data/insurance.csv
  target: charges
  test_size: 0.2
cleaning:
  remove_duplicates: true
  missing_strategy: auto
feature_engineering:
  auto_interactions: true
  auto_log: true
model_selection:
  candidates: [lightgbm, xgboost, random_forest, ridge]
tuning:
  enabled: true
  engine: optuna
  n_trials: 50
```

Create 5 example configs: regression, classification, imbalanced, time-series, production.

Exit criteria: `mlpipe run --config pipeline.yaml` works.

---

### Phase 4: Level 3 — Manual Scaffold

**Goal**: `mlpipe init` generates stage stubs for manual pipeline building.

```bash
mlpipe init --task regression --name my-pipeline
```

Exit criteria: Scaffold generates working code, user can override stages.

---

### Phase 5: 18-Phase Expansion

**Goal**: Add 5 new stages, expand existing stages with missing features.

#### New Stages

```python
# stages/baseline.py — Phase 8
class BaselineStage(Stage):
    """DummyClassifier/Regressor + analytical baselines (LogReg/Ridge)."""
    # Returns baseline metrics so user knows the floor

# stages/imbalance.py — Phase 6
class ImbalanceStage(Stage):
    """Auto-detect imbalance ratio → route to SMOTE/ADASYN/class_weight."""
    # ratio < 4:1 → class_weight only
    # ratio 4-20:1 → SMOTE + class_weight
    # ratio > 20:1 → ADASYN + class_weight

# stages/calibrate.py — Phase 13
class CalibrateStage(Stage):
    """CalibratedClassifierCV + threshold optimization."""
    # Platt scaling or Isotonic
    # Find optimal threshold for F1

# stages/model_card.py — Phase 16
class ModelCardStage(Stage):
    """Auto-generate markdown model card."""
    # Fills in: model type, metrics, limitations, fairness

# stages/monitor.py — Phase 18
class MonitorStage(Stage):
    """Evidently drift detection + retrain triggers."""
    # DataDriftPreset, TargetDriftPreset
    # Save drift_report.html
```

#### Expand Existing Stages

| Stage | Add |
|-------|-----|
| LoadStage | Parquet, Excel, SQL (SQLAlchemy), API (requests) |
| CompareStage | ExtraTrees, HistGradientBoosting, CatBoost, StackingClassifier, VotingClassifier |
| TuneStage | GridSearchCV, HalvingGridSearchCV options |
| SelectStage | RFECV, VarianceThreshold, SelectKBest |
| ExplainStage | KernelExplainer, dependence_plot, SHAP feature selection |
| EvaluateStage | Overfitting gap analysis, calibration curve |
| DeployStage | Working FastAPI with /predict + /health, proper Dockerfile |
| SaveStage | DVC tracking, version naming (model_v1, v2...) |

Exit criteria: `mlpipe run --data data.csv` covers all 18 phases.

---

### Phase 6: LLM Agent Pipeline

**Goal**: Users can create ML models using natural language via LLM providers.

**Inspired by**: HuggingFace ml-intern (smolagents, litellm, code agents).

#### Architecture

```
User: "Train a classifier on heart_disease.csv"
  │
  ▼
┌──────────────────────────────────────────────────┐
│  mlpipe Agent (CLI or Web)                       │
│                                                  │
│  ┌────────────────────────────────────────────┐  │
│  │  Provider Layer (litellm)                  │  │
│  │  ├─ Local: Ollama, vLLM, LM Studio         │  │
│  │  ├─ Free: HF Inference, DeepSeek           │  │
│  │  └─ Paid: OpenAI, Anthropic, Gemini        │  │
│  └────────────────────────────────────────────┘  │
│                                                  │
│  ┌────────────────────────────────────────────┐  │
│  │  Tool Router (ML-specific tools)           │  │
│  │  ├─ load_data, preview_data, profile_data  │  │
│  │  ├─ analyze_missing, check_imbalance       │  │
│  │  ├─ train_model, compare_models            │  │
│  │  ├─ tune_hyperparams, evaluate_model       │  │
│  │  ├─ explain_shap, save_model               │  │
│  │  ├─ generate_api, generate_docker          │  │
│  │  └─ search_sklearn_docs, search_papers     │  │
│  └────────────────────────────────────────────┘  │
│                                                  │
│  ┌────────────────────────────────────────────┐  │
│  │  Context Manager                           │  │
│  │  ├─ Conversation history                   │  │
│  │  ├─ Auto-compaction (long sessions)        │  │
│  │  └─ Doom loop detection                    │  │
│  └────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────┘
  │
  ▼
Model saved → API generated → Docker ready → Deployed
```

#### File Structure

```
src/mlpipe/agent/
├── __init__.py
├── provider.py           # litellm wrapper, provider abstraction
├── tools.py              # 15+ ML tools (load, train, eval, deploy)
├── context.py            # conversation history + compaction
├── prompts.py            # system prompts for ML tasks
├── cli_agent.py          # interactive CLI mode
└── headless.py           # single-command mode
```

#### Provider Config

```yaml
# ~/.mlpipe/providers.yaml
providers:
  ollama:
    base_url: http://localhost:11434
    models: [llama3.1:8b, codellama:13b]
  hf:
    api_key: ${HF_TOKEN}
    models: [Qwen/Qwen3-8B, meta-llama/Llama-3.1-8B-Instruct]
  openai:
    api_key: ${OPENAI_API_KEY}
    models: [gpt-4o, gpt-4o-mini]
  anthropic:
    api_key: ${ANTHROPIC_API_KEY}
    models: [claude-sonnet-4-20250514]
```

#### Usage

```bash
# Interactive
mlpipe agent --provider ollama/llama3.1:8b
> Load insurance.csv and train a regression model

# Headless
mlpipe agent --provider openai/gpt-4o "Train classifier on heart_disease.csv and deploy"
```

Exit criteria: Agent can load data, train model, evaluate, and deploy from natural language.

---

### Phase 7: Frontend (React/Vite/Tailwind)

**Goal**: Web UI for uploading data, picking provider, watching pipeline run.

```
frontend/
├── src/
│   ├── App.tsx
│   ├── components/
│   │   ├── DataUpload.tsx         # Drag-drop CSV/Parquet
│   │   ├── ProviderSelect.tsx     # Pick LLM provider
│   │   ├── PipelineView.tsx       # Live stage execution
│   │   ├── ResultsDashboard.tsx   # Metrics, plots, SHAP
│   │   ├── ChatAgent.tsx          # Chat with ML agent
│   │   └── DeployPanel.tsx        # Deploy controls
│   └── lib/
│       └── api.ts                 # FastAPI backend calls
├── package.json
└── vite.config.ts
```

Backend: FastAPI serves both API + static frontend.

Exit criteria: Upload data → pick provider → watch pipeline → view results in browser.

---

### Phase 8: Git & CI

```bash
git init
# .gitignore, GitHub Actions CI (ruff + pytest)
```

Exit criteria: `git push` triggers CI, all checks pass.

---

### Phase 9: Deployment

- Docker Compose (local): `docker-compose up`
- AWS ECS: `mlpipe deploy --aws`
- MLflow serving: `mlpipe serve --model model_v1`

Exit criteria: One-command deploy to Docker or AWS.

---

### Phase 10: Documentation & Release

- README with architecture diagram
- CLI reference (auto-generated)
- Config reference
- PyPI release: `pip install mlpipe`

---

## Timeline Estimate

| Phase | Scope | Est. Time | Priority |
|-------|-------|-----------|----------|
| Phase 2 | Testing + Linting | 3-4 hours | **1st** |
| Phase 3 | Level 2 (YAML config) | 1-2 hours | **2nd** |
| Phase 8 | Git + CI | 30 min | **3rd** |
| Phase 5 | 18-Phase Expansion | 3-4 hours | **4th** |
| Phase 4 | Level 3 (scaffold) | 1 hour | **5th** |
| Phase 6 | LLM Agent Pipeline | 4-6 hours | **6th** |
| Phase 7 | Frontend | 6-8 hours | **7th** |
| Phase 9 | Deployment | 2-3 hours | **8th** |
| Phase 10 | Docs + Release | 1-2 hours | **9th** |
| **Total** | | **22-30 hours** | |

---

## Key Technical Decisions

1. **litellm** for LLM provider abstraction — same as ml-intern, supports 100+ providers
2. **smolagents pattern** — code agents (write Python) are 30% more efficient than JSON tool calling
3. **Tool registration** — ML tools are plain Python functions with docstrings, agent discovers them
4. **Doom loop detection** — catch stuck patterns after 3 repeated tool calls
5. **Auto-compaction** — summarize old turns to stay within context window
6. **Sandbox execution** — agent code runs in isolated environment for safety

---

## Data Leakage Checklist (from FINAL-ML-Workflow)

```
✅ train_test_split BEFORE any fit/transform
✅ Preprocessors INSIDE Pipeline — not fit on full df
✅ SMOTE inside imblearn Pipeline — only on train fold
✅ No target-derived features in X (except TargetEncoder inside Pipeline)
✅ No future data in time-series training
✅ OHE has handle_unknown='ignore'
✅ CV and tuning done on X_train only
✅ Test set touched ONCE — final evaluation only
✅ Same preprocessing at training and serving time
```