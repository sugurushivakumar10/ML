Needs to check

# ============================================================

# 
# ============================================================
# CLEARING DAYS PREDICTION MODEL (UPDATED TO YOUR COLUMNS)
# ============================================================

import pandas as pd
import numpy as np
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. DATE CONVERSION
# ============================================================

date_cols = ["bline_date", "clearing_date"]

for col in date_cols:
    df["bline_date"] = pd.to_datetime(df["bline_date"], errors="coerce")
    df["clearing_date"] = pd.to_datetime(df["clearing_date"], errors="coerce")


# ============================================================
# 2. CREATE TARGET (if not already correct)
# ============================================================

df["clearing_days"] = (
    df["clearing_date"] - df["bline_date"]
).dt.days

df = df[df["clearing_days"].notna()]
df = df[df["clearing_days"] >= 0]


# ============================================================
# 3. DATE FEATURE ENGINEERING
# ============================================================

df["baseline_month"] = df["bline_date"].dt.month
df["baseline_quarter"] = df["bline_date"].dt.quarter
df["baseline_dayofweek"] = df["bline_date"].dt.dayofweek


# ============================================================
# 4. SELECT SAFE FEATURES (NO LEAKAGE)
# ============================================================

features = [
    "region",
    "area",
    "country",
    "company_code",
    "vendor_number",
    "PO_type",
    "type",
    "new_payment_term",      # use this instead of raw payment_term
    "period",
    "amount_doccurr",
    "amount_usd",
    "discount_base_amount",
    "baseline_month",
    "baseline_quarter",
    "baseline_dayofweek"
]

target = "clearing_days"


# ============================================================
# 5. CATEGORICAL COLUMNS
# ============================================================

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
# 6. TRAIN / VALID SPLIT
# ============================================================

X = df[features]
y = df[target]

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ============================================================
# 7. BUILD MODEL
# ============================================================

model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=-1,
    random_state=42
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric="mae",
    verbose=50
)


# ============================================================
# 8. EVALUATE
# ============================================================

y_pred = model.predict(X_valid)

mae = mean_absolute_error(y_valid, y_pred)
rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
r2 = r2_score(y_valid, y_pred)

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")
print("MAE  :", round(mae, 2), "days")
print("RMSE :", round(rmse, 2), "days")
print("R2   :", round(r2, 3))

within_3 = np.mean(np.abs(y_valid - y_pred) <= 3) * 100
within_5 = np.mean(np.abs(y_valid - y_pred) <= 5) * 100

print("Accuracy ±3 days :", round(within_3, 2), "%")
print("Accuracy ±5 days :", round(within_5, 2), "%")


# ============================================================
# 9. FEATURE IMPORTANCE
# ============================================================

importance_df = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print("\nTop Features:")
print(importance_df.head(10))


# ============================================================
# 10. PREDICT ON OPEN DATA
# ============================================================

# Example for open dataset:

# open_df["bline_date"] = pd.to_datetime(open_df["bline_date"])
# open_df["baseline_month"] = open_df["bline_date"].dt.month
# open_df["baseline_quarter"] = open_df["bline_date"].dt.quarter
# open_df["baseline_dayofweek"] = open_df["bline_date"].dt.dayofweek

# open_df["predicted_clearing_days"] = model.predict(open_df[features])

# open_df["predicted_clearing_date"] = (
#     open_df["bline_date"] +
#     pd.to_timedelta(open_df["predicted_clearing_days"], unit="D")
# )
