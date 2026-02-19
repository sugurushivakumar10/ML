open_df = pd.read_excel("Open_Invoices.xlsx")

open_df["bline_date"] = pd.to_datetime(open_df["bline_date"], errors="coerce")

open_df["baseline_month"] = open_df["bline_date"].dt.month
open_df["baseline_quarter"] = open_df["bline_date"].dt.quarter
open_df["baseline_dayofweek"] = open_df["bline_date"].dt.dayofweek

X_open = open_df[features].copy()



# ============================================================
# 13. APPLY SAME PREPROCESSING
# ============================================================

for col in X_open.select_dtypes(include=["float64", "int64"]).columns:
    X_open[col] = X_open[col].fillna(X_open[col].median())

for col in X_open.select_dtypes(include=["object"]).columns:
    X_open[col] = X_open[col].fillna("Unknown")

for col in X_open.select_dtypes(include=["object"]).columns:
    if col in label_encoders:
        X_open[col] = label_encoders[col].transform(X_open[col])



# ============================================================
# 14. PREDICT OPEN INVOICES
# ============================================================

open_df["predicted_clearing_days"] = rf_model.predict(X_open).round(0)

open_df["predicted_clearing_date"] = (
    open_df["bline_date"]
    + pd.to_timedelta(open_df["predicted_clearing_days"], unit="D")
)



# ============================================================
# 15. ADJUST IF PREDICTED DATE IS IN THE PAST
# ============================================================

today = pd.Timestamp.today().normalize()

mask = open_df["predicted_clearing_date"] < today

open_df.loc[mask, "delay_days"] = (
    today - open_df.loc[mask, "predicted_clearing_date"]
).dt.days

open_df.loc[mask, "adjusted_predicted_date"] = (
    today + pd.to_timedelta(open_df.loc[mask, "delay_days"], unit="D")
)

open_df["adjusted_predicted_date"].fillna(
    open_df["predicted_clearing_date"],
    inplace=True
)



# ============================================================
# 16. FINAL BUSINESS OUTPUT
# ============================================================

final_output = open_df[[
    "bline_date",
    "predicted_clearing_days",
    "predicted_clearing_date",
    "adjusted_predicted_date"
]]

print(final_output.head())
