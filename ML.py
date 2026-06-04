import re
import pandas as pd
 
# ─────────────────────────────────────────────
# STEP 1 — READ BOTH CSV FILES
# ─────────────────────────────────────────────
# Just change the file names below to match yours
 
df1 = pd.read_csv("vendor_Mastertable.csv", dtype=str, encoding="utf-8-sig")
df2 = pd.read_csv("tickets.csv",            dtype=str, encoding="utf-8-sig")
 
print("df1 shape (vendor master) :", df1.shape)
print("df1 columns               :", df1.columns.tolist())
print()
print("df2 shape (tickets)       :", df2.shape)
print("df2 columns               :", df2.columns.tolist())
print()
 
# ─────────────────────────────────────────────
# STEP 2 — SET YOUR COLUMN NAMES
# ─────────────────────────────────────────────
# Change these to match the actual column names printed above
 
VENDOR_NUMBER_COL = "VendorNumber"    # column in df1
TICKET_ID_COL     = "TicketID"        # column in df2
TITLE_COL         = "Title"           # column in df2
DESCRIPTION_COL   = "Description"     # column in df2
 
# ─────────────────────────────────────────────
# STEP 3 — BUILD VENDOR SET FROM df1
# ─────────────────────────────────────────────
 
vendor_set = set(
    df1[VENDOR_NUMBER_COL]
    .dropna()
    .str.strip()
    .str.lstrip("0")
)
print(f"Vendor master loaded : {len(vendor_set):,} unique vendor numbers")
print()
 
# ─────────────────────────────────────────────
# STEP 4 — DEFINE EXTRACTION FUNCTION
# ─────────────────────────────────────────────
 
TICKET_PATTERN    = re.compile(r'\bTKT-\w+-\w+\b', re.IGNORECASE)
CANDIDATE_PATTERN = re.compile(r'\b(0*[1-9]\d+)\b')
 
def extract_vendor_id(text):
    """
    Reads one cell value (Title or Description).
    Extracts all 7-8 digit candidate numbers.
    Returns the first one that exists in the vendor master set.
    Returns empty string if nothing matches.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
 
    # Remove ticket IDs like TKT-xxxx-xxxx so their digits are not matched
    cleaned = TICKET_PATTERN.sub(" ", text)
 
    for raw in CANDIDATE_PATTERN.findall(cleaned):
        core = raw.lstrip("0")
        # Must be 7 or 8 digits after stripping leading zeros
        if len(core) in (7, 8) and core[0] != "0":
            normalised = core
            if normalised in vendor_set:
                return normalised
 
    return ""
 
# ─────────────────────────────────────────────
# STEP 5 — APPLY FUNCTION ON df2 COLUMNS
# ─────────────────────────────────────────────
 
df2[TITLE_COL]       = df2[TITLE_COL].fillna("")
df2[DESCRIPTION_COL] = df2[DESCRIPTION_COL].fillna("")
 
print("Applying on Title column...")
df2["vendor_num_from_title"]       = df2[TITLE_COL].apply(extract_vendor_id)
 
print("Applying on Description column...")
df2["vendor_num_from_description"] = df2[DESCRIPTION_COL].apply(extract_vendor_id)
 
# ─────────────────────────────────────────────
# STEP 6 — DERIVE FINAL VENDOR NUMBER
# ─────────────────────────────────────────────
 
def get_final_vendor(row):
    t = row["vendor_num_from_title"]
    d = row["vendor_num_from_description"]
 
    if t and d:
        return t if t == d else f"{t}, {d}"   # both same → one value / differ → both
    return t or d                              # whichever has a value
 
df2["vendor_num_final"] = df2.apply(get_final_vendor, axis=1)
df2["match_status"]     = df2["vendor_num_final"].apply(
    lambda x: "Matched" if x else "Not Found"
)
 
# ─────────────────────────────────────────────
# STEP 7 — SUMMARY + SAVE
# ─────────────────────────────────────────────
 
total   = len(df2)
matched = (df2["match_status"] == "Matched").sum()
 
print()
print(f"{'='*45}")
print(f"  Total tickets          : {total:>8,}")
print(f"  Matched to vendor list : {matched:>8,}")
print(f"  Not matched            : {total - matched:>8,}")
print(f"{'='*45}")
print()
 
# Save all results
df2.to_csv("vendor_id_results_all.csv", index=False)
print("Saved → vendor_id_results_all.csv")
 
# Save matched only
matched_df = df2[df2["match_status"] == "Matched"]
if len(matched_df):
    matched_df.to_csv("vendor_id_results_matched.csv", index=False)
    print("Saved → vendor_id_results_matched.csv")
 
# Save not found
notfound_df = df2[df2["match_status"] == "Not Found"]
if len(notfound_df):
    notfound_df.to_csv("vendor_id_results_notfound.csv", index=False)
    print("Saved → vendor_id_results_notfound.csv")
