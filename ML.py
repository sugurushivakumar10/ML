# =========================================
# 1. IMPORT LIBRARIES
# =========================================
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error

import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")


# =========================================
# 2. LOAD DATA
# =========================================
df = pd.read_csv("invoice_data.csv")   # <-- update path

# Convert dates
date_cols = ['payment_date', 'clearing_date', 'netdue_date']
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')


# =========================================
# 3. FILTER CLOSED INVOICES (TRAINING DATA)
# =========================================
df = df[df['clearing_date'].notna()].copy()

# Ensure target is numeric
df['clearing_days'] = df['clearing_days'].astype(int)

# Cap extreme values (stability)
df['clearing_days'] = df['clearing_days'].clip(0, 120)


# =========================================
# 4. QUICK EDA (OPTIONAL BUT RECOMMENDED)
# =========================================
plt.figure(figsize=(8,4))
sns.histplot(df['clearing_days'], bins=50)
plt.title("Clearing Days Distribution")
plt.show()


# =========================================
# 5. FEATURE SELECTION
# =========================================
num_features = [
    'received_days',
    'processing_days',
    'approval_days',
    'paid_days',
    'amount_usd',
    'discount_base_amt'
]

cat_features = [
    'region',
    'area',
    'country',
    'company_code',
    'PO_type',
    'payment_term',
    'type',
    'Discount Payt'
]

target = 'clearing_days'


# =========================================
# 6. MISSING VALUE HANDLING
# =========================================
df[num_features] = df[num_features].fillna(0)

for col in cat_features:
    df[col] = df[col].fillna("UNKNOWN")


# =========================================
# 7. ENCODING (LABEL ENCODING)
# =========================================
label_encoders = {}

for col in cat_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le


# =========================================
# 8. TRAIN / VALIDATION SPLIT (TIME BASED)
# =========================================
df = df.sort_values('payment_date')

features = num_features + cat_features
X = df[features]
y = df[target]

split_index = int(0.8 * len(df))

X_train = X.iloc[:split_index]
y_train = y.iloc[:split_index]

X_valid = X.iloc[split_index:]
y_valid = y.iloc[split_index:]


# =========================================
# 9. MODEL DEFINITION (HIGH ACCURACY)
# =========================================
model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)


# =========================================
# 10. MODEL TRAINING
# =========================================
model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric='mae',
    early_stopping_rounds=50,
    verbose=50
)


# =========================================
# 11. MODEL EVALUATION
# =========================================
valid_preds = model.predict(X_valid)

mae = mean_absolute_error(y_valid, valid_preds)
rmse = mean_squared_error(y_valid, valid_preds, squared=False)

print("=================================")
print(f"MAE  : {mae:.2f} days")
print(f"RMSE : {rmse:.2f} days")
print("=================================")


# =========================================
# 12. FEATURE IMPORTANCE
# =========================================
lgb.plot_importance(model, max_num_features=15, importance_type='gain')
plt.title("Feature Importance")
plt.show()


# =========================================
# 13. LOAD OPEN INVOICES (PREDICTION DATA)
# =========================================
open_df = pd.read_csv("open_invoices.csv")   # <-- update path

# Convert dates
for col in date_cols:
    open_df[col] = pd.to_datetime(open_df[col], errors='coerce')

# Handle missing values
open_df[num_features] = open_df[num_features].fillna(0)

for col in cat_features:
    open_df[col] = open_df[col].fillna("UNKNOWN")
    open_df[col] = label_encoders[col].transform(open_df[col])


# =========================================
# 14. PREDICT CLEARING DAYS
# =========================================
open_df['predicted_clearing_days'] = model.predict(open_df[features])
open_df['predicted_clearing_days'] = (
    open_df['predicted_clearing_days']
    .round()
    .clip(0)
    .astype(int)
)


# =========================================
# 15. PREDICT CLEARING DATE
# =========================================
open_df['predicted_clearing_date'] = (
    open_df['payment_date'] +
    pd.to_timedelta(open_df['predicted_clearing_days'], unit='D')
)


# =========================================
# 16. RISK BUCKET (OPTIONAL)
# =========================================
def risk_bucket(x):
    if x <= 2:
        return "Low"
    elif x <= 7:
        return "Medium"
    else:
        return "High"

open_df['clearing_risk'] = open_df['predicted_clearing_days'].apply(risk_bucket)


# =========================================
# 17. SAVE OUTPUT
# =========================================
open_df.to_csv("open_invoice_predictions.csv", index=False)

print("✅ Prediction completed and saved successfully.")
