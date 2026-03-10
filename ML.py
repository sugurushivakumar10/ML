

# =====================================================
# 1. Import Libraries
# =====================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from lightgbm import LGBMRegressor


# =====================================================
# 2. Load Closed Invoice Dataset (Training Data)
# =====================================================

df = pd.read_excel("closed_invoices.xlsx")

print("Dataset Shape:", df.shape)
print(df.head())


# =====================================================
# 3. Convert Date Columns
# =====================================================

date_cols = [
    "baseline_date",
    "clearing_date",
    "netdue_date",
    "discount_deadline_date"
]

for col in date_cols:
    df[col] = pd.to_datetime(df[col])


# =====================================================
# 4. Create Target Variable
# =====================================================

df["clearing_days"] = (df["clearing_date"] - df["baseline_date"]).dt.days


# =====================================================
# 5. Date Feature Engineering
# =====================================================

df["baseline_month"] = df["baseline_date"].dt.month
df["baseline_dayofweek"] = df["baseline_date"].dt.dayofweek

df["due_month"] = df["netdue_date"].dt.month
df["due_dayofweek"] = df["netdue_date"].dt.dayofweek


# =====================================================
# 6. Due Date Gap Feature
# =====================================================

df["due_gap_days"] = (df["netdue_date"] - df["baseline_date"]).dt.days


# =====================================================
# 7. Amount Bucket Feature
# =====================================================

df["amount_bucket"] = pd.qcut(
    df["amount_usd"],
    q=5,
    labels=False,
    duplicates="drop"
)


# =====================================================
# 8. Vendor Behaviour Features
# =====================================================

vendor_avg_delay = df.groupby("vendor_num")["clearing_days"].mean()
vendor_std_delay = df.groupby("vendor_num")["clearing_days"].std()
vendor_invoice_count = df.groupby("vendor_num")["clearing_days"].count()

df["vendor_avg_delay"] = df["vendor_num"].map(vendor_avg_delay)
df["vendor_std_delay"] = df["vendor_num"].map(vendor_std_delay)
df["vendor_invoice_count"] = df["vendor_num"].map(vendor_invoice_count)


# =====================================================
# 9. Prepare Training Data
# =====================================================

target = "clearing_days"

drop_cols = [
    "clearing_date",
    "clearing_days"
]

X = df.drop(columns=drop_cols)
y = df[target]


# =====================================================
# 10. Encode Categorical Variables
# =====================================================

categorical_cols = [
    "region",
    "area",
    "country",
    "company_code",
    "vendor_num",
    "PO_type",
    "payment_term",
    "type"
]

X = pd.get_dummies(X, columns=categorical_cols)


# =====================================================
# 11. Train-Test Split
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =====================================================
# 12. Train LightGBM Model
# =====================================================

model = LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    random_state=42
)

model.fit(X_train, y_train)


# =====================================================
# 13. Model Evaluation
# =====================================================

pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, pred))
print("R2 Score:", r2_score(y_test, pred))


# =====================================================
# 14. Feature Importance
# =====================================================

importance = pd.Series(
    model.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

print("\nTop Features:")
print(importance.head(20))


# =====================================================
# 15. Load Open Invoice Dataset
# =====================================================

open_df = pd.read_excel("open_invoices.xlsx")

print("Open Data Shape:", open_df.shape)


# =====================================================
# 16. Convert Date Columns
# =====================================================

date_cols_open = [
    "baseline_date",
    "netdue_date",
    "discount_deadline_date"
]

for col in date_cols_open:
    open_df[col] = pd.to_datetime(open_df[col])


# =====================================================
# 17. Same Feature Engineering for Open Data
# =====================================================

open_df["baseline_month"] = open_df["baseline_date"].dt.month
open_df["baseline_dayofweek"] = open_df["baseline_date"].dt.dayofweek

open_df["due_month"] = open_df["netdue_date"].dt.month
open_df["due_dayofweek"] = open_df["netdue_date"].dt.dayofweek

open_df["due_gap_days"] = (
    open_df["netdue_date"] - open_df["baseline_date"]
).dt.days

open_df["amount_bucket"] = pd.qcut(
    open_df["amount_usd"],
    q=5,
    labels=False,
    duplicates="drop"
)


# =====================================================
# 18. Add Vendor Behaviour Features
# =====================================================

open_df["vendor_avg_delay"] = open_df["vendor_num"].map(vendor_avg_delay)
open_df["vendor_std_delay"] = open_df["vendor_num"].map(vendor_std_delay)
open_df["vendor_invoice_count"] = open_df["vendor_num"].map(vendor_invoice_count)


# =====================================================
# 19. Encode Categorical Variables
# =====================================================

open_X = pd.get_dummies(open_df)

# align columns with training data
open_X = open_X.reindex(columns=X_train.columns, fill_value=0)


# =====================================================
# 20. Predict Clearing Days
# =====================================================

open_df["predicted_clearing_days"] = model.predict(open_X)


# =====================================================
# 21. Predicted Clearing Date
# =====================================================

open_df["predicted_clearing_date"] = open_df["baseline_date"] + pd.to_timedelta(
    open_df["predicted_clearing_days"],
    unit="D"
)


# =====================================================
# 22. Discount Eligibility
# =====================================================

open_df["discount_eligible"] = np.where(
    open_df["predicted_clearing_date"] <= open_df["discount_deadline_date"],
    "YES",
    "NO"
)


# =====================================================
# 23. Final Output
# =====================================================

result = open_df[
    [
        "predicted_clearing_days",
        "predicted_clearing_date",
        "discount_eligible"
    ]
]

print(result.head())
