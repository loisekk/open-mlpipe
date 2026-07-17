# Pipeline Stages

The pipeline runs 12 stages in order:

| # | Stage | Description |
|---|-------|-------------|
| 1 | **Load** | Load data, detect task type, detect target |
| 2 | **EDA** | Column hygiene, quality, distributions, statistical tests |
| 3 | **Clean** | Remove duplicates, handle outliers, drop ID columns |
| 4 | **Feature Eng** | Interactions, log transforms, missingness flags |
| 5 | **Split** | Stratified train/test split |
| 6 | **Preprocess** | Impute, scale, encode (ColumnTransformer) |
| 7 | **Compare** | 14+ models head-to-head with cross-validation |
| 8 | **Tune** | Optuna hyperparameter optimization |
| 9 | **Select** | SHAP-based feature importance |
| 10 | **Evaluate** | Test set metrics (R², RMSE, MAE, etc.) |
| 11 | **Explain** | SHAP summary, dependence, waterfall plots |
| 12 | **Save** | Save full inference pipeline |

## Stage Details

### 1. Load

- Loads data from CSV, Parquet, or Excel
- Auto-detects task type (regression/classification)
- Auto-detects target column
- Detects column types (numeric, categorical, datetime)

### 2. EDA

- Column hygiene (strip names, fix dtypes)
- Quality assessment (missing, duplicates, outliers)
- Distribution analysis (skew, kurtosis)
- Statistical tests (Z-test, T-test, ANOVA, Chi-Square)
- Correlation analysis and VIF

### 3. Clean

- Remove duplicates
- Handle missing values
- Clip/remove outliers (IQR method)
- Drop ID-like columns
- Drop text columns (high cardinality)

### 4. Feature Engineering

- Missingness flags (was_missing_X)
- Interaction features (X × Y)
- Log transforms for skewed features
- DateTime feature extraction

### 5. Split

- Stratified split for classification
- Random split for regression
- Configurable test size

### 6. Preprocess

- Numeric: Impute → Scale
- Categorical: Impute → OneHotEncode
- Handles missing values automatically

### 7. Compare

- Cross-validation with multiple metrics
- Auto-selects best model
- Detects overfitting/underfitting

### 8. Tune

- Optuna hyperparameter optimization
- 18 model-specific search spaces
- Early stopping

### 9. Select

- SHAP-based feature importance
- Ranks features by importance

### 10. Evaluate

- Final test set evaluation
- Regression: R², RMSE, MAE, MAPE
- Classification: Accuracy, F1, ROC-AUC, MCC

### 11. Explain

- SHAP summary plot
- SHAP dependence plot
- SHAP waterfall plot

### 12. Save

- Saves full inference pipeline
- Includes feature engineering + model
- Ready for production deployment
