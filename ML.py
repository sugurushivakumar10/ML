


# =========================================================
# 🔷 CLEARING DATE PREDICTION MODEL
# 🔷 TRAINING + OPEN DATA PREDICTION
# =========================================================

# ==========================
# 1️⃣ IMPORTS
# ==========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import warnings
warnings.filterwarnings("ignore")

# ==========================
# 2️⃣ READ CLOSED DATA
# ==========================
df = pd.read_excel("Invoice_Discount_onlyDiscount.xlsx")

date_cols = ["bline_date", "clearing_date", "payment_date", "netdue_date"]

for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

# ==========================
# 3️⃣ CREATE TARGET
# ==========================
df["predict_clearing_days"] = (df["clearing_date"] - df["bline_date"]).dt.days

df = df[df["predict_clearing_days"].notna()]
df = df[df["predict_clearing_days"] >= 0]
df = df[df["predict_clearing_days"] <= 365]

print("Target Distribution:")
print(df["predict_clearing_days"].describe())

# ==========================
# 4️⃣ FEATURE ENGINEERING
# ==========================
df["baseline_month"] = df["bline_date"].dt.month
df["baseline_quarter"] = df["bline_date"].dt.quarter
df["baseline_dayofweek"] = df["bline_date"].dt.dayofweek

df["days_to_discount"] = (df["payment_date"] - df["bline_date"]).dt.days
df["days_to_net_due"] = (df["netdue_date"] - df["bline_date"]).dt.days

# ==========================
# 5️⃣ FEATURE SELECTION
# ==========================
features = [
    "region","area","country","company_code","vendor_number",
    "PO_type","type","new_payment_term","period",
    "amount_doccurr","amount_usd","discount_base_amount",
    "baseline_month","baseline_quarter","baseline_dayofweek",
    "days_to_discount","days_to_net_due"
]

target = "predict_clearing_days"

X = df[features].copy()
y = df[target].copy()

# ==========================
# 6️⃣ HANDLE MISSING VALUES
# ==========================
for col in X.select_dtypes(include=["float64","int64"]).columns:
    X[col] = X[col].fillna(X[col].median())

for col in X.select_dtypes(include=["object"]).columns:
    X[col] = X[col].fillna("Unknown")

# ==========================
# 7️⃣ LABEL ENCODING
# ==========================
label_encoders = {}

for col in X.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# ==========================
# 8️⃣ TRAIN TEST SPLIT
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0
)

# ==========================
# 9️⃣ MODEL TRAINING
# ==========================
rf_model = RandomForestRegressor(
    n_estimators=300,
    random_state=0,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# ==========================
# 🔟 MODEL EVALUATION
# ==========================
rf_pred = rf_model.predict(X_test)

mae = mean_absolute_error(y_test, rf_pred)
rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
r2 = r2_score(y_test, rf_pred)
acc3 = np.mean(np.abs(y_test - rf_pred) <= 3) * 100
acc5 = np.mean(np.abs(y_test - rf_pred) <= 5) * 100

print("\n===== MODEL PERFORMANCE =====")
print("MAE :", round(mae,2))
print("RMSE:", round(rmse,2))
print("R2  :", round(r2,3))
print("Accuracy ±3 days:", round(acc3,2), "%")
print("Accuracy ±5 days:", round(acc5,2), "%")

# ==========================
# 1️⃣1️⃣ VALIDATION DEMO
# ==========================
validation_df = X_test.copy()
validation_df["Actual_Clearing_Days"] = y_test
validation_df["Predicted_Clearing_Days"] = rf_pred.round(0)

df_demo = df.merge(
    validation_df[["Actual_Clearing_Days","Predicted_Clearing_Days"]],
    left_index=True,
    right_index=True,
    how="left"
)

print("\nValidation Sample:")
print(df_demo[["predict_clearing_days","Predicted_Clearing_Days"]].dropna().head())

# =========================================================
# 🚀 OPEN DATA PREDICTION
# =========================================================

df_open = pd.read_excel("Open_Invoices.xlsx")

for col in ["bline_date","payment_date","netdue_date"]:
    df_open[col] = pd.to_datetime(df_open[col], errors="coerce")

df_open["baseline_month"] = df_open["bline_date"].dt.month
df_open["baseline_quarter"] = df_open["bline_date"].dt.quarter
df_open["baseline_dayofweek"] = df_open["bline_date"].dt.dayofweek

df_open["days_to_discount"] = (df_open["payment_date"] - df_open["bline_date"]).dt.days
df_open["days_to_net_due"] = (df_open["netdue_date"] - df_open["bline_date"]).dt.days

X_open = df_open[features].copy()

for col in X_open.select_dtypes(include=["float64","int64"]).columns:
    X_open[col] = X_open[col].fillna(X[col].median())

for col in X_open.select_dtypes(include=["object"]).columns:
    X_open[col] = X_open[col].fillna("Unknown")

for col in X_open.select_dtypes(include=["object"]).columns:
    if col in label_encoders:
        le = label_encoders[col]
        X_open[col] = X_open[col].apply(
            lambda s: le.transform([s])[0] if s in le.classes_ else -1
        )

df_open["Predicted_Clearing_Days"] = rf_model.predict(X_open).round(0)

df_open["Predicted_Clearing_Date"] = (
    df_open["bline_date"] +
    pd.to_timedelta(df_open["Predicted_Clearing_Days"], unit="D")
)

# ==========================
# 🔥 PAST DATE ADJUSTMENT
# ==========================
today = pd.to_datetime("today").normalize()

mask = df_open["Predicted_Clearing_Date"] < today

delay_days = (today - df_open.loc[mask,"Predicted_Clearing_Date"]).dt.days

df_open.loc[mask,"Adjusted_Predicted_Date"] = (
    today + pd.to_timedelta(delay_days, unit="D")
)

df_open.loc[~mask,"Adjusted_Predicted_Date"] = (
    df_open.loc[~mask,"Predicted_Clearing_Date"]
)

# ==========================
# 💰 DISCOUNT LOGIC
# ==========================
df_open["Discount_Status"] = np.where(
    df_open["Adjusted_Predicted_Date"] <= df_open["payment_date"],
    "Will Get Discount",
    "No Discount"
)

# ==========================
# 📊 FINAL OUTPUT
# ==========================
print("\nOpen Invoice Predictions:")
print(df_open[[
    "vendor_number",
    "bline_date",
    "payment_date",
    "netdue_date",
    "Predicted_Clearing_Days",
    "Adjusted_Predicted_Date",
    "Discount_Status"
]].head())
