"""
extract_vendor_id.py
--------------------
Matches vendor/supplier IDs found in ticket records
against a master vendor list — both loaded from CSV files.

Input files:
  1. vendor_master.csv   — ~3,80,000 vendor numbers (one column)
  2. tickets.csv         — ~13,00,000 ticket records (Title + Description columns)

Approach (no DB, no API, no hardcoded keywords, fully pandas vectorised):
  Step 1  Load vendor master CSV → convert to a Python set  (O1 lookup)
  Step 2  Load ticket data CSV
  Step 3  Extract candidate numbers from Title column
  Step 4  Extract candidate numbers from Description column
  Step 5  Check each candidate against vendor set  (O1 per row)
  Step 6  Derive vendor_num_final and match status

Output columns added:
  vendor_num_from_title        matched vendor ID found in Title
  vendor_num_from_description  matched vendor ID found in Description
  vendor_num_final             final resolved vendor ID
  match_source                 title / description / both / both(differ)
  match_status                 Matched / Not Found

Requirements:
    pip install pandas openpyxl

Usage:
    python extract_vendor_id.py
"""

import re
import pandas as pd

# ─────────────────────────────────────────────
# 1.  FILE CONFIG  — change these to your paths
# ─────────────────────────────────────────────

# Vendor master CSV  (~3,80,000 rows, one vendor number column)
VENDOR_MASTER_CSV = "vendor_master.csv"
VENDOR_NUMBER_COL = "VendorNumber"          # column header in vendor master CSV

# Ticket data CSV  (~13,00,000 rows)
TICKETS_CSV       = "tickets.csv"
TICKET_ID_COL     = "TicketID"              # ticket ID column  e.g. TKT-xxxx-xxxx
TITLE_COL         = "Title"                 # subject / title column
DESCRIPTION_COL   = "Description"           # body / description column

# ─────────────────────────────────────────────
# 2.  LOAD VENDOR MASTER INTO A SET
# ─────────────────────────────────────────────

def load_vendor_set(csv_path: str, col: str) -> set:
    """
    Read vendor master CSV and return a set of normalised vendor numbers.
    dtype=str keeps leading zeros intact during load.
    Set gives O(1) lookup — no matter how many vendors are in it.
    """
    print(f"Loading vendor master  : '{csv_path}'")
    df = pd.read_csv(csv_path, dtype=str, usecols=[col])

    vendor_set = set(
        df[col]
        .dropna()
        .str.strip()
        .str.lstrip("0")        # normalise leading zeros
        .replace("", pd.NA)
        .dropna()
    )
    print(f"  Unique vendor numbers : {len(vendor_set):,}\n")
    return vendor_set

# ─────────────────────────────────────────────
# 3.  LOAD TICKET DATA FROM CSV
# ─────────────────────────────────────────────

def load_tickets(csv_path: str) -> pd.DataFrame:
    """
    Read ticket data CSV.
    Only loads the three columns we need to keep memory usage low.
    dtype=str avoids any auto type conversion on IDs or numbers.
    """
    print(f"Loading ticket data    : '{csv_path}'")
    df = pd.read_csv(
        csv_path,
        dtype=str,
        usecols=[TICKET_ID_COL, TITLE_COL, DESCRIPTION_COL]
    )
    df[TITLE_COL]       = df[TITLE_COL].fillna("")
    df[DESCRIPTION_COL] = df[DESCRIPTION_COL].fillna("")

    print(f"  Ticket rows loaded    : {len(df):,}\n")
    return df

# ─────────────────────────────────────────────
# 4.  CANDIDATE NUMBER EXTRACTION
# ─────────────────────────────────────────────

TICKET_PATTERN    = re.compile(r'\bTKT-\w+-\w+\b', re.IGNORECASE)
CANDIDATE_PATTERN = re.compile(r'\b(0*[1-9]\d+)\b')

def is_valid_length(raw: str) -> bool:
    """Core number after stripping leading zeros must be 7 or 8 digits."""
    core = raw.lstrip("0")
    return len(core) in (7, 8) and core[0] != "0"

def normalise(raw: str) -> str:
    """Strip leading zeros for consistent comparison with vendor set."""
    return raw.lstrip("0") or raw

def extract_candidates(text: str) -> list:
    """
    Extract all candidate 7-8 digit numbers from a single text value.
    Applied via Series.apply() directly on Title and Description columns.

    - Removes ticket IDs (TKT-xxxx-xxxx) first
    - Finds all digit sequences
    - Validates length (7 or 8 digits after stripping zeros)
    - Returns deduplicated list of normalised numbers
    """
    if not isinstance(text, str) or not text.strip():
        return []

    cleaned    = TICKET_PATTERN.sub(" ", text)
    candidates = [
        normalise(m)
        for m in CANDIDATE_PATTERN.findall(cleaned)
        if is_valid_length(m)
    ]
    return list(dict.fromkeys(candidates))      # deduplicate, preserve order

# ─────────────────────────────────────────────
# 5.  VENDOR SET MATCH  (O1 lookup)
# ─────────────────────────────────────────────

def match_against_vendor_set(candidates: list, vendor_set: set) -> str:
    """
    Given a list of candidate numbers from one text field,
    return the first candidate that exists in the vendor master set.

    Lookup is O(1) per candidate — not a loop through 3,80,000 vendors.
    Returns empty string if no match.
    """
    for candidate in candidates:
        if candidate in vendor_set:
            return candidate
    return ""

# ─────────────────────────────────────────────
# 6.  FINAL RESOLUTION HELPERS
# ─────────────────────────────────────────────

def resolve_final(match_title: str, match_desc: str) -> str:
    if match_title and match_desc:
        if match_title == match_desc:
            return match_title
        return f"{match_title}, {match_desc}"   # both found but differ
    return match_title or match_desc


def resolve_source(match_title: str, match_desc: str) -> str:
    if match_title and match_desc:
        return "both" if match_title == match_desc else "both(differ)"
    if match_title:  return "title"
    if match_desc:   return "description"
    return ""


def resolve_status(vendor_num_final: str) -> str:
    return "Matched" if vendor_num_final else "Not Found"

# ─────────────────────────────────────────────
# 7.  SELF-TESTS
# ─────────────────────────────────────────────

def run_tests():
    test_vendor_set = {"1234567", "9876543", "1111111"}

    tests = [
        # label,  title,  description,  exp_t, exp_d, exp_final, exp_status
        ("vendor in title only",      "issue with 1234567",          "please check",           "1234567", "",        "1234567",          "Matched"  ),
        ("vendor in description only","payment problem",             "supplier 9876543 unpaid", "",       "9876543", "9876543",          "Matched"  ),
        ("same vendor in both",       "ref 1234567",                 "vendor id 1234567",       "1234567","1234567", "1234567",          "Matched"  ),
        ("different vendor each side","ref 1234567",                 "supplier 9876543",        "1234567","9876543", "1234567, 9876543", "Matched"  ),
        ("number not in master",      "ref 9999999",                 "check 8888888",           "",       "",        "",                 "Not Found"),
        ("ticket digits excluded",    "TKT-1234567-2024 complaint",  "no vendor",               "",       "",        "",                 "Not Found"),
        ("leading zeros in text",     "vendor 0001234567",           "",                        "1234567","",        "1234567",          "Matched"  ),
        ("nothing found",             "general complaint",           "please resolve",          "",       "",        "",                 "Not Found"),
    ]

    print("─── Self-tests ──────────────────────────────────────────────────")
    passed = 0
    for label, title, desc, exp_t, exp_d, exp_f, exp_s in tests:
        cands_t = extract_candidates(title)
        cands_d = extract_candidates(desc)
        match_t = match_against_vendor_set(cands_t, test_vendor_set)
        match_d = match_against_vendor_set(cands_d, test_vendor_set)
        final   = resolve_final(match_t, match_d)
        status  = resolve_status(final)

        ok = (match_t == exp_t and match_d == exp_d and final == exp_f and status == exp_s)
        if ok:
            passed += 1
        print(f"  [{'PASS' if ok else 'FAIL'}]  {label}")
        if not ok:
            if match_t != exp_t: print(f"           title   expected='{exp_t}'  got='{match_t}'")
            if match_d != exp_d: print(f"           desc    expected='{exp_d}'  got='{match_d}'")
            if final   != exp_f: print(f"           final   expected='{exp_f}'  got='{final}'")
            if status  != exp_s: print(f"           status  expected='{exp_s}'  got='{status}'")
    print(f"─── {passed}/{len(tests)} passed ───────────────────────────────────────────\n")

# ─────────────────────────────────────────────
# 8.  MAIN PIPELINE
# ─────────────────────────────────────────────

def main():

    # Step 1 — load vendor master into set
    vendor_set = load_vendor_set(VENDOR_MASTER_CSV, VENDOR_NUMBER_COL)

    # Run self-tests before touching real data
    run_tests()

    # Step 2 — load ticket CSV
    df = load_tickets(TICKETS_CSV)

    # Step 3 & 4 — extract candidates from each column
    print("Extracting candidates from Title column…")
    title_candidates = df[TITLE_COL].apply(extract_candidates)

    print("Extracting candidates from Description column…")
    desc_candidates  = df[DESCRIPTION_COL].apply(extract_candidates)

    # Step 5 — match candidates against vendor set (O1 lookup)
    print("Matching against vendor master…")
    df["vendor_num_from_title"]       = title_candidates.apply(
        lambda c: match_against_vendor_set(c, vendor_set)
    )
    df["vendor_num_from_description"] = desc_candidates.apply(
        lambda c: match_against_vendor_set(c, vendor_set)
    )

    # Step 6 — derive final columns
    df["vendor_num_final"] = df.apply(
        lambda row: resolve_final(
            row["vendor_num_from_title"],
            row["vendor_num_from_description"]
        ), axis=1
    )
    df["match_source"] = df.apply(
        lambda row: resolve_source(
            row["vendor_num_from_title"],
            row["vendor_num_from_description"]
        ), axis=1
    )
    df["match_status"] = df["vendor_num_final"].apply(resolve_status)

    # ── Summary ───────────────────────────────────────────────────
    total   = len(df)
    matched = (df["match_status"] == "Matched").sum()
    from_t  = (df["vendor_num_from_title"] != "").sum()
    from_d  = (df["vendor_num_from_description"] != "").sum()
    differ  = (df["match_source"] == "both(differ)").sum()

    print(f"\n{'='*56}")
    print(f"  Total tickets              : {total:>8,}")
    print(f"  Matched to vendor master   : {matched:>8,}")
    print(f"  Not matched                : {total - matched:>8,}")
    print(f"  ── Found in title only     : {from_t:>8,}")
    print(f"  ── Found in description    : {from_d:>8,}")
    print(f"  ── Title & desc differ     : {differ:>8,}")
    print(f"{'='*56}\n")

    # ── Save outputs ──────────────────────────────────────────────
    df.to_csv("vendor_id_results_all.csv", index=False)
    print("Saved → vendor_id_results_all.csv")

    matched_df = df[df["match_status"] == "Matched"]
    if len(matched_df):
        matched_df.to_csv("vendor_id_results_matched.csv", index=False)
        print("Saved → vendor_id_results_matched.csv")

    notfound_df = df[df["match_status"] == "Not Found"]
    if len(notfound_df):
        notfound_df.to_csv("vendor_id_results_notfound.csv", index=False)
        print("Saved → vendor_id_results_notfound.csv")

    differ_df = df[df["match_source"] == "both(differ)"]
    if len(differ_df):
        differ_df.to_csv("vendor_id_needs_review.csv", index=False)
        print("Saved → vendor_id_needs_review.csv")

    return df


if __name__ == "__main__":
    main()
