# ============================================================
# ENTERPRISE CLEARING DAYS PREDICTION
# Target: predict_clearing_days
# Split: test_size=0.20, random_state=0
# Models: LightGBM + RandomForest
# ============================================================

import pandas as pd
import numpy as np
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor


# ============================================================
# 1. DATE CONVERSION
# ============================================================

df["bline_date"] = pd.to_datetime(df["bline_date"], errors="coerce")
df["clearing_date"] = pd.to_datetime(df["clearing_date"], errors="coerce")


# ============================================================
# 2. CREATE TARGET COLUMN
# ============================================================

df["predict_clearing_days"] = (
    df["clearing_date"] - df["bline_date"]
).dt.days

df = df[df["predict_clearing_days"].notna()]
df = df[df["predict_clearing_days"] >= 0]


# ============================================================
# 3. REMOVE OUTLIERS (>365 DAYS)
# ============================================================

df = df[df["predict_clearing_days"] <= 365]


# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================

df["baseline_month"] = df["bline_date"].dt.month
df["baseline_quarter"] = df["bline_date"].dt.quarter
df["baseline_dayofweek"] = df["bline_date"].dt.dayofweek


# ============================================================
# 5. SELECT SAFE FEATURES
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


# ============================================================
# 6. HANDLE MISSING VALUES
# ============================================================

# Fill numeric columns with median
num_cols = df[features].select_dtypes(include=["float64", "int64"]).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Convert categorical columns
cat_cols = [
    "region",
    "area",
    "country",
    "company_code",
    "vendor_number",
    "PO_type",
    "type",
    "new_payment_term"
]

for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].astype("category")


# ============================================================
# 7. TRAIN TEST SPLIT (YOUR FORMAT)
# ============================================================

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0
)


# ============================================================
# 8. MODEL 1 — LIGHTGBM
# ============================================================

lgb_model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=-1,
    random_state=0
)

lgb_model.fit(X_train, y_train)
lgb_pred = lgb_model.predict(X_test)


# ============================================================
# 9. MODEL 2 — RANDOM FOREST
# ============================================================

rf_model = RandomForestRegressor(
    n_estimators=300,
    random_state=0,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)


# ============================================================
# 10. EVALUATION FUNCTION
# ============================================================

def evaluate_model(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    within_3 = np.mean(np.abs(y_true - y_pred) <= 3) * 100
    within_5 = np.mean(np.abs(y_true - y_pred) <= 5) * 100

    print(f"\n===== {name} =====")
    print("MAE  :", round(mae, 2), "days")
    print("RMSE :", round(rmse, 2), "days")
    print("R2   :", round(r2, 3))
    print("Accuracy ±3 days :", round(within_3, 2), "%")
    print("Accuracy ±5 days :", round(within_5, 2), "%")


# ============================================================
# 11. PRINT RESULTS
# ============================================================

evaluate_model("LightGBM", y_test, lgb_pred)
evaluate_model("Random Forest", y_test, rf_pred)


# ============================================================
# 12. FEATURE IMPORTANCE (LIGHTGBM)
# ============================================================

importance_df = pd.DataFrame({
    "feature": features,
    "importance": lgb_model.feature_importances_
}).sort_values("importance", ascending=False)

print("\nTop 10 Important Features:")
print(importance_df.head(10))


# ============================================================
# 13. PREDICT ON OPEN DATA
# ============================================================

# Example:
#
# open_df["bline_date"] = pd.to_datetime(open_df["bline_date"])
# open_df["baseline_month"] = open_df["bline_date"].dt.month
# open_df["baseline_quarter"] = open_df["bline_date"].dt.quarter
# open_df["baseline_dayofweek"] = open_df["bline_date"].dt.dayofweek
#
# open_df["predicted_clearing_days"] = lgb_model.predict(open_df[features])
#
# open_df["predicted_clearing_date"] = (
#     open_df["bline_date"] +
#     pd.to_timedelta(open_df["predicted_clearing_days"], unit="D")
# )
