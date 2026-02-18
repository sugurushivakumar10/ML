# ============================================================
# CLEARING DAYS PREDICTION - NOTEBOOK VERSION
# ============================================================

import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# ============================================================
# 1. READ DATASET
# ============================================================

df = pd.read_csv("your_dataset.csv")  # change if excel
print("Initial Shape:", df.shape)
df.head()


# ============================================================
# 2. CREATE TARGET COLUMN
# ============================================================

df["bline_date"] = pd.to_datetime(df["bline_date"], errors="coerce")
df["clearing_date"] = pd.to_datetime(df["clearing_date"], errors="coerce")

df["predict_clearing_days"] = (
    df["clearing_date"] - df["bline_date"]
).dt.days

# Remove invalid & outliers
df = df[df["predict_clearing_days"].notna()]
df = df[df["predict_clearing_days"] >= 0]
df = df[df["predict_clearing_days"] <= 365]

print("After Cleaning:", df.shape)


# ============================================================
# 3. EDA
# ============================================================

print("\nTarget Distribution:")
print(df["predict_clearing_days"].describe())

plt.figure(figsize=(8,4))
sns.histplot(df["predict_clearing_days"], bins=50)
plt.title("Distribution of Clearing Days")
plt.show()


# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================

df["baseline_month"] = df["bline_date"].dt.month
df["baseline_quarter"] = df["bline_date"].dt.quarter
df["baseline_dayofweek"] = df["bline_date"].dt.dayofweek


# ============================================================
# 5. FEATURE SELECTION (NO LEAKAGE)
# ============================================================

features = [
    "region",
    "area",
    "country",
    "company_code",
    "vendor_number",
    "PO_type",
    "type",
    "new_payment_term",
    "period",
    "amount_doccurr",
    "amount_usd",
    "discount_base_amount",
    "baseline_month",
    "baseline_quarter",
    "baseline_dayofweek"
]

target = "predict_clearing_days"

X = df[features]
y = df[target]


# ============================================================
# 6. HANDLE MISSING VALUES
# ============================================================

for col in X.select_dtypes(include=["float64", "int64"]).columns:
    X[col] = X[col].fillna(X[col].median())

for col in X.select_dtypes(include=["object"]).columns:
    X[col] = X[col].fillna("Unknown")


# ============================================================
# 7. LABEL ENCODING
# ============================================================

label_encoders = {}

for col in X.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le


# ============================================================
# 8. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0
)


# ============================================================
# 9. MODEL PREPARATION
# ============================================================

lgb_model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    random_state=0
)

rf_model = RandomForestRegressor(
    n_estimators=300,
    random_state=0,
    n_jobs=-1
)


# ============================================================
# 10. MODEL FIT
# ============================================================

lgb_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)


# ============================================================
# 11. MODEL PREDICTION
# ============================================================

lgb_pred = lgb_model.predict(X_test)
rf_pred = rf_model.predict(X_test)


# ============================================================
# 12. MODEL EVALUATION
# ============================================================

def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    acc3 = np.mean(np.abs(y_true - y_pred) <= 3) * 100
    acc5 = np.mean(np.abs(y_true - y_pred) <= 5) * 100

    print(f"\n===== {name} =====")
    print("MAE :", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2  :", round(r2, 3))
    print("Accuracy ±3 days:", round(acc3, 2), "%")
    print("Accuracy ±5 days:", round(acc5, 2), "%")


evaluate("LightGBM", y_test, lgb_pred)
evaluate("RandomForest", y_test, rf_pred)


# ============================================================
# 13. FEATURE IMPORTANCE (LIGHTGBM)
# ============================================================

importance_df = pd.DataFrame({
    "feature": features,
    "importance": lgb_model.feature_importances_
}).sort_values("importance", ascending=False)

print("\nTop 10 Important Features:")
print(importance_df.head(10))


# ============================================================
# 14. PREDICT ON OPEN DATA
# ============================================================

# Example:
# open_df["bline_date"] = pd.to_datetime(open_df["bline_date"])
# open_df["baseline_month"] = open_df["bline_date"].dt.month
# open_df["baseline_quarter"] = open_df["bline_date"].dt.quarter
# open_df["baseline_dayofweek"] = open_df["bline_date"].dt.dayofweek
#
# for col in label_encoders:
#     open_df[col] = label_encoders[col].transform(open_df[col].astype(str))
#
# open_df["predicted_clearing_days"] = lgb_model.predict(open_df[features])
#
# open_df["predicted_clearing_date"] = (
#     open_df["bline_date"] +
#     pd.to_timedelta(open_df["predicted_clearing_days"], unit="D")
# )
