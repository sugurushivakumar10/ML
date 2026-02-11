# ==========================================================
# 1. IMPORT LIBRARIES
# ==========================================================
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
import matplotlib.pyplot as plt


# ==========================================================
# 2. LOAD DATA (CHANGE FILE NAME)
# ==========================================================
df = pd.read_excel("your_80k_file.xlsx")   # <-- change file name


# ==========================================================
# 3. DATE CONVERSION
# ==========================================================
date_cols = ["baseline_date", "netdue_date", "payment_date", "clearing_date"]

for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors="coerce")


# ==========================================================
# 4. BUSINESS FILTER
# ==========================================================
df = df[df["Discount Payt"] == "Yes"].copy()
df = df[df["clearing_date"].notna()].copy()


# ==========================================================
# 5. TARGET CLEANING
# ==========================================================
df["clearing_days"] = df["clearing_days"].astype(int)

# Clip extreme outliers
upper_limit = df["clearing_days"].quantile(0.99)
df["clearing_days"] = df["clearing_days"].clip(0, upper_limit)


# ==========================================================
# 6. FEATURE ENGINEERING
# ==========================================================
df["days_baseline_to_payment"] = (
    df["payment_date"] - df["baseline_date"]
).dt.days

df["days_baseline_to_payment"] = df["days_baseline_to_payment"].fillna(0)


# ==========================================================
# 7. FEATURE LIST
# ==========================================================
num_features = [
    "received_days",
    "processing_days",
    "approval_days",
    "paid_days",
    "amount_usd",
    "discount_base_amount",
    "discount_amount",
    "days_baseline_to_payment"
]

cat_features = [
    "region",
    "area",
    "country",
    "company_code",
    "vendor_numb",
    "PO_type",
    "payment_term",
    "type"
]

target = "clearing_days"


# ==========================================================
# 8. HANDLE MISSING VALUES
# ==========================================================
df[num_features] = df[num_features].fillna(0)

for col in cat_features:
    df[col] = df[col].fillna("UNKNOWN")


# ==========================================================
# 9. ENCODE CATEGORICAL FEATURES
# ==========================================================
label_encoders = {}

for col in cat_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le


# ==========================================================
# 10. TIME-BASED TRAIN / VALID SPLIT
# ==========================================================
df = df.sort_values("payment_date")

features = num_features + cat_features
X = df[features]
y = df[target]

split_index = int(0.8 * len(df))

X_train = X.iloc[:split_index]
y_train = y.iloc[:split_index]

X_valid = X.iloc[split_index:]
y_valid = y.iloc[split_index:]


# ==========================================================
# 11. MODEL DEFINITION (STRONG SETTINGS FOR 80K)
# ==========================================================
model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=10,
    num_leaves=128,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    random_state=42
)


# ==========================================================
# 12. TRAIN MODEL (LightGBM 4+ Compatible)
# ==========================================================
model.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric="mae",
    callbacks=[
        early_stopping(stopping_rounds=100),
        log_evaluation(100)
    ]
)


# ==========================================================
# 13. EVALUATION
# ==========================================================
valid_preds = model.predict(X_valid)

mae = mean_absolute_error(y_valid, valid_preds)
rmse = np.sqrt(mean_squared_error(y_valid, valid_preds))
r2 = r2_score(y_valid, valid_preds)

# Accuracy ±3 and ±5
valid_preds_rounded = np.round(valid_preds)

accuracy_3 = (np.abs(y_valid - valid_preds_rounded) <= 3).mean() * 100
accuracy_5 = (np.abs(y_valid - valid_preds_rounded) <= 5).mean() * 100

print("====================================")
print(f"MAE  : {mae:.2f} days")
print(f"RMSE : {rmse:.2f} days")
print(f"R2   : {r2:.3f}")
print(f"Accuracy ±3 days : {accuracy_3:.2f}%")
print(f"Accuracy ±5 days : {accuracy_5:.2f}%")
print("====================================")


# ==========================================================
# 14. FEATURE IMPORTANCE
# ==========================================================
lgb.plot_importance(model, max_num_features=20)
plt.show()


# ==========================================================
# 15. CORRELATION CHECK (OPTIONAL)
# ==========================================================
print("Top Correlations with clearing_days:")
print(df.corr(numeric_only=True)["clearing_days"]
      .sort_values(ascending=False)
      .head(10))


# ==========================================================
# 16. SAVE MODEL OUTPUT
# ==========================================================
df["predicted_clearing_days"] = model.predict(X).round().clip(0)
df["predicted_clearing_date"] = (
    df["payment_date"] +
    pd.to_timedelta(df["predicted_clearing_days"], unit="D")
)

df.to_excel("invoice_predictions_output_80k.xlsx", index=False)

print("✅ MODEL TRAINING COMPLETE")
print("✅ Predictions saved to invoice_predictions_output_80k.xlsx")
