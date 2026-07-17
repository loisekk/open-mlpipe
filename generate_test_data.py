"""Generate test datasets for mlpipe testing."""
import pandas as pd
import numpy as np
import os

np.random.seed(42)
out_dir = "C:/Users/yashb/mlpipe/test_datasets"
os.makedirs(out_dir, exist_ok=True)

# === 1. REGRESSION: Insurance (already exists, regenerate cleaner) ===
n = 1338
ages = np.random.randint(18, 65, n)
sex = np.random.choice(['male', 'female'], n)
bmi = np.random.normal(30, 6, n).round(1)
children = np.random.poisson(1, n)
smoker = np.random.choice(['yes', 'no'], n, p=[0.25, 0.75])
region = np.random.choice(['northeast', 'southeast', 'southwest', 'northwest'], n)
charges = (ages * 250 + bmi * 300 + children * 500 + np.where(smoker=='yes', 20000, 0) + np.random.normal(0, 3000, n)).round(2)
df_ins = pd.DataFrame({'age': ages, 'sex': sex, 'bmi': bmi, 'children': children, 'smoker': smoker, 'region': region, 'charges': charges})
# Add realistic messiness
df_ins.loc[np.random.choice(n, 20, replace=False), 'bmi'] = np.nan
df_ins.loc[np.random.choice(n, 5, replace=False), 'sex'] = np.nan
# Add duplicates
df_ins = pd.concat([df_ins, df_ins.head(10)], ignore_index=True)
df_ins.to_csv(f"{out_dir}/insurance.csv", index=False)
print(f"insurance.csv: {df_ins.shape}")

# === 2. CLASSIFICATION: Heart Disease ===
n = 1000
age = np.random.randint(29, 77, n)
sex = np.random.choice([0, 1], n)
cp = np.random.choice([0, 1, 2, 3], n)
trestbps = np.random.normal(132, 18, n).astype(int)
chol = np.random.normal(246, 52, n).astype(int)
fbs = np.random.choice([0, 1], n, p=[0.85, 0.15])
restecg = np.random.choice([0, 1, 2], n)
thalach = np.random.normal(150, 22, n).astype(int)
exang = np.random.choice([0, 1], n, p=[0.7, 0.3])
oldpeak = np.random.exponential(1.0, n).round(1)
slope = np.random.choice([0, 1, 2], n)
ca = np.random.choice([0, 1, 2, 3], n, p=[0.6, 0.25, 0.1, 0.05])
thal = np.random.choice([0, 1, 2], n)
# Target: heart disease probability based on features
prob = 1 / (1 + np.exp(-(0.03*(age-50) + 0.5*sex + 0.8*cp - 0.01*(thalach-150) + 0.3*exang + 0.5*oldpeak - 0.001*(chol-200))))
target = (np.random.random(n) < prob).astype(int)
df_heart = pd.DataFrame({
    'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
    'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
    'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal, 'target': target
})
# Add messiness
df_heart.loc[np.random.choice(n, 30, replace=False), 'chol'] = np.nan
df_heart.loc[np.random.choice(n, 15, replace=False), 'trestbps'] = np.nan
df_heart = pd.concat([df_heart, df_heart.head(5)], ignore_index=True)
df_heart.to_csv(f"{out_dir}/heart_disease.csv", index=False)
print(f"heart_disease.csv: {df_heart.shape}")

# === 3. MESSY DATA: Lots of problems ===
n = 500
df_messy = pd.DataFrame({
    'CustomerID': range(1, n+1),
    '  Name  ': [f'Customer_{i}' for i in range(n)],
    'AGE': np.random.randint(18, 80, n).astype(float),
    ' income ': np.random.lognormal(10, 1, n).round(2),
    'Department': np.random.choice(['Sales', 'Engineering', 'HR', 'Marketing', 'Finance'], n),
    'City': np.random.choice(['New York', 'London', 'Tokyo', 'Paris', 'Berlin', 'Sydney', 'Mumbai', 'Toronto'], n),
    ' join_date ': pd.date_range('2020-01-01', periods=n, freq='D').astype(str),
    'satisfaction_score': np.random.uniform(1, 10, n).round(1),
    'churned': np.random.choice([0, 1], n, p=[0.8, 0.2]),
})
# Add 15% missing values
for col in ['AGE', ' income ', 'satisfaction_score', 'City']:
    mask = np.random.random(n) < 0.15
    df_messy.loc[mask, col] = np.nan
# Add outliers
df_messy.loc[0, ' income '] = 9999999
df_messy.loc[1, 'AGE'] = 999
# Add duplicates
df_messy = pd.concat([df_messy, df_messy.head(20)], ignore_index=True)
# Add wrong dtypes
df_messy['AGE'] = df_messy['AGE'].astype(str)  # numbers as strings
df_messy.to_csv(f"{out_dir}/messy_data.csv", index=False)
print(f"messy_data.csv: {df_messy.shape}")

# === 4. SMALL DATASET: < 200 rows ===
n = 150
df_small = pd.DataFrame({
    'feature_1': np.random.normal(0, 1, n),
    'feature_2': np.random.normal(5, 2, n),
    'feature_3': np.random.uniform(0, 10, n),
    'category': np.random.choice(['A', 'B', 'C'], n),
    'target': np.random.choice([0, 1], n, p=[0.6, 0.4]),
})
df_small.to_csv(f"{out_dir}/small_dataset.csv", index=False)
print(f"small_dataset.csv: {df_small.shape}")

# === 5. ALL NUMERIC: No categoricals ===
n = 800
df_numeric = pd.DataFrame({f'feat_{i}': np.random.normal(0, 1, n) for i in range(10)})
df_numeric['target'] = df_numeric['feat_0'] * 2 + df_numeric['feat_1'] * -1 + np.random.normal(0, 0.5, n)
df_numeric.to_csv(f"{out_dir}/all_numeric.csv", index=False)
print(f"all_numeric.csv: {df_numeric.shape}")

print(f"\nAll datasets created in {out_dir}")
