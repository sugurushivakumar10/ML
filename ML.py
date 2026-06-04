"""
extract_vendor_id.py
--------------------
Extracts Vendor / Supplier IDs from SQL Server email-ticket records.
No API key. No hardcoded keyword list. Pure Python + pandas.

How it works — context-aware extraction:
  Instead of matching only fixed phrases like "vendor id: 1234567",
  we scan a window of words AROUND every candidate number and
  score how likely it is to be a vendor/supplier ID based on:
    - Nearby words (vendor, supplier, ref, code, account, partner ...)
    - Position in sentence (after a label pattern)
    - What the number looks like (length, leading zeros)
    - Negative signals (amount, total, invoice amount, date, phone ...)

  This means ANY phrasing in the email body is handled naturally,
  not just the phrases we thought of in advance.

Vendor ID rules:
  - May have leading zeros (000128456 → 128456)
  - Core number after stripping zeros: 7 or 8 digits, starts 1-9
  - Ticket IDs like TKT-xxxx-xxxx are excluded

Output columns:
  vendor_num_from_title       - IDs found in Title column
  vendor_num_from_description - IDs found in Description column
  vendor_num_final            - Final resolved vendor ID
  needs_review                - Yes if title and description disagree

Requirements:
    pip install pyodbc pandas openpyxl

Usage:
    python extract_vendor_id.py
"""

import re
import pyodbc
import pandas as pd

# ─────────────────────────────────────────────
# 1.  DATABASE CONNECTION
# ─────────────────────────────────────────────
SERVER   = "YOUR_SERVER_NAME"
DATABASE = "YOUR_DATABASE_NAME"
TABLE    = "YOUR_TABLE_NAME"

CONNECTION_STRING = (
    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
    f"SERVER={SERVER};"
    f"DATABASE={DATABASE};"
    f"Trusted_Connection=yes;"
)
# SQL Server login (uncomment if needed):
# USERNAME = "your_username"
# PASSWORD = "your_password"
# CONNECTION_STRING = (
#     f"DRIVER={{ODBC Driver 17 for SQL Server}};"
#     f"SERVER={SERVER};DATABASE={DATABASE};"
#     f"UID={USERNAME};PWD={PASSWORD};"
# )

# ─────────────────────────────────────────────
# 2.  COLUMN NAMES
# ─────────────────────────────────────────────
TICKET_ID_COL   = "TicketID"
TITLE_COL       = "Title"
DESCRIPTION_COL = "Description"

# ─────────────────────────────────────────────
# 3.  CONTEXT SCORING ENGINE
# ─────────────────────────────────────────────
#
# These word lists drive the scoring — they are CONTEXT signals,
# not hardcoded keyword matchers. A number does not need to sit
# right next to one of these words; they just raise or lower the
# probability score within a ±5 word window.
#
# You can freely add more words to any list without changing logic.

# Words that raise the score — number is likely a vendor/supplier ID
POSITIVE_SIGNALS = {
    # vendor / supplier variants
    "vendor", "vendors", "vend",
    "supplier", "suppliers", "supp",
    # reference / ID language
    "id", "code", "number", "num", "no", "ref", "reference",
    "account", "acct", "partner", "contact",
    # action context that implies an ID
    "register", "registered", "onboard", "onboarded",
    "assigned", "mapped", "linked", "associated",
    "check", "verify", "confirm", "lookup", "search",
    "update", "correct", "wrong", "invalid", "missing",
    "portal", "system", "erp", "sap", "oracle",
}

# Words that lower the score — number is probably NOT a vendor ID
NEGATIVE_SIGNALS = {
    # financial amounts
    "amount", "total", "subtotal", "price", "cost",
    "payment", "paid", "pay", "due", "balance",
    "usd", "inr", "eur", "gbp", "rs", "₹", "$", "€",
    # dates / times
    "date", "day", "month", "year", "time",
    "january","february","march","april","may","june",
    "july","august","september","october","november","december",
    "jan","feb","mar","apr","jun","jul","aug","sep","oct","nov","dec",
    # phone / other IDs
    "phone", "mobile", "fax", "zip", "pin", "pincode",
    "invoice", "po", "order", "ticket", "tkt",
    # quantities
    "qty", "quantity", "units", "count", "items",
}

# If any of these label patterns appear DIRECTLY before the number,
# give a large bonus — strong signal regardless of other context.
# These are intentionally broad — colon/hash/dash after any word ending in
# id / no / code / number / ref signals a structured label.
LABEL_BEFORE_PATTERN = re.compile(
    r'(?:id|no|num|number|code|ref|reference|account|acct)\s*[:\-=#]?\s*$',
    re.IGNORECASE
)

# Ticket ID pattern — excluded entirely
TICKET_PATTERN = re.compile(r'\bTKT-\w+-\w+\b', re.IGNORECASE)

# Candidate number finder — any digit sequence
CANDIDATE_PATTERN = re.compile(r'\b(0*[1-9]\d*)\b')

# Score threshold — tune this if you get too many / too few matches
SCORE_THRESHOLD = 2

# ─────────────────────────────────────────────
# 4.  HELPERS
# ─────────────────────────────────────────────

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return TICKET_PATTERN.sub(" TICKETREF ", text)


def is_valid_length(raw: str) -> bool:
    """Core number (leading zeros stripped) must be 7 or 8 digits."""
    core = raw.lstrip("0")
    return len(core) in (7, 8) and core[0] != "0"


def normalise(raw: str) -> str:
    return raw.lstrip("0") or raw


def tokenise(text: str) -> list:
    """Split text into lowercase word tokens, keeping position info."""
    return re.findall(r"[a-zA-Z₹$€]+|\d+|[#:/\-=]", text.lower())


def score_candidate(tokens: list, num_index: int) -> int:
    """
    Score a candidate number based on tokens in a ±5 word window.

    Returns an integer score:
      >= SCORE_THRESHOLD  →  likely a vendor/supplier ID
      <  SCORE_THRESHOLD  →  likely something else (amount, date, etc.)
    """
    window_start = max(0, num_index - 5)
    window_end   = min(len(tokens), num_index + 6)
    window       = tokens[window_start:window_end]
    before       = tokens[window_start:num_index]

    score = 0

    # +3 if a label pattern sits directly before the number
    before_str = " ".join(before)
    if LABEL_BEFORE_PATTERN.search(before_str):
        score += 3

    # +2 for each positive signal word in the window
    for word in window:
        if word in POSITIVE_SIGNALS:
            score += 2

    # -3 for each negative signal word in the window
    for word in window:
        if word in NEGATIVE_SIGNALS:
            score -= 3

    # +1 if the token immediately before is a separator like : = - #
    if num_index > 0 and tokens[num_index - 1] in (":", "=", "-", "#"):
        score += 1

    return score


def extract_from_text(text: str) -> str:
    """
    Applied directly on a pandas Series (Title or Description column).

    Finds every candidate number in the text, scores it using surrounding
    context, and returns the valid vendor IDs as a comma-separated string.
    Returns empty string if nothing found.
    """
    cleaned = clean_text(text)
    tokens  = tokenise(cleaned)
    found   = set()

    for i, token in enumerate(tokens):
        # Must be a digit sequence
        if not token.isdigit():
            continue

        raw = token
        if not is_valid_length(raw):
            continue

        # Score based on surrounding context
        if score_candidate(tokens, i) >= SCORE_THRESHOLD:
            found.add(normalise(raw))

    # Also run a direct label-match pass for cases where label is right before number
    # e.g. "vendor id: 1234567"  or  "ref: 0001234567"
    for match in CANDIDATE_PATTERN.finditer(cleaned):
        raw = match.group(1)
        if not is_valid_length(raw):
            continue
        # Check the 40 characters directly before this number
        prefix = cleaned[:match.start()].strip()
        if LABEL_BEFORE_PATTERN.search(prefix[-40:]):
            found.add(normalise(raw))

    return ", ".join(sorted(found)) if found else ""


# ─────────────────────────────────────────────
# 5.  FINAL RESOLUTION
# ─────────────────────────────────────────────

def resolve_final(from_title: str, from_desc: str) -> str:
    set_t  = {x.strip() for x in from_title.split(",") if x.strip()} if from_title else set()
    set_d  = {x.strip() for x in from_desc.split(",")  if x.strip()} if from_desc  else set()
    merged = set_t | set_d
    return ", ".join(sorted(merged)) if merged else ""


def needs_review_flag(from_title: str, from_desc: str) -> str:
    set_t = {x.strip() for x in from_title.split(",") if x.strip()} if from_title else set()
    set_d = {x.strip() for x in from_desc.split(",")  if x.strip()} if from_desc  else set()
    return "Yes" if set_t and set_d and set_t != set_d else "No"

# ─────────────────────────────────────────────
# 6.  SELF-TESTS
# ─────────────────────────────────────────────

def run_tests():
    tests = [
        # label,  text,  expected
        ("standard vendor id label",        "Vendor ID: 1234567",                          "1234567"),
        ("vendor number phrase",            "vendor number 1234567 not found",             "1234567"),
        ("supplier number",                 "Supplier number 9876543 is pending",          "9876543"),
        ("leading zeros",                   "vendor code 0001234567",                      "1234567"),
        ("ref colon before number",         "ref: 1234567 please check",                   "1234567"),
        ("account id label",                "account id: 9876543",                         "9876543"),
        ("sap system context",              "check in SAP system 1234567 is missing",      "1234567"),
        ("onboarded context",               "partner 1234567 is not onboarded yet",        "1234567"),
        ("ticket digits excluded",          "TKT-1234567-2024 no vendor here",             ""),
        ("amount should be excluded",       "total amount 1234567 is due",                 ""),
        ("invoice amount not vendor",       "invoice amount 9876543 unpaid",               ""),
        ("phone number not vendor",         "call on phone 9876543210",                    ""),   # 10 digits
        ("too short",                       "ref 123456",                                  ""),   # 6 digits
        ("too long",                        "ref 123456789",                               ""),   # 9 digits
        ("no number at all",                "please resolve this at earliest",             ""),
    ]

    print("─── Self-tests ──────────────────────────────────────────────────")
    passed = 0
    for label, text, expected in tests:
        got    = extract_from_text(text)
        ok     = got == expected
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        print(f"  [{status}]  {label}")
        if not ok:
            print(f"           expected : '{expected}'")
            print(f"           got      : '{got}'")
    print(f"─── {passed}/{len(tests)} passed ───────────────────────────────────────────\n")

# ─────────────────────────────────────────────
# 7.  MAIN PIPELINE
# ─────────────────────────────────────────────

def main():
    run_tests()

    print("Connecting to SQL Server…")
    conn = pyodbc.connect(CONNECTION_STRING)
    df   = pd.read_sql(
        f"SELECT [{TICKET_ID_COL}], [{TITLE_COL}], [{DESCRIPTION_COL}] FROM [{TABLE}]",
        conn
    )
    conn.close()
    print(f"Loaded {len(df):,} rows.\n")

    df[TITLE_COL]       = df[TITLE_COL].fillna("")
    df[DESCRIPTION_COL] = df[DESCRIPTION_COL].fillna("")

    # ── Apply extraction directly on each column ──────────────────
    print("Extracting from Title column…")
    df["vendor_num_from_title"]       = df[TITLE_COL].apply(extract_from_text)

    print("Extracting from Description column…")
    df["vendor_num_from_description"] = df[DESCRIPTION_COL].apply(extract_from_text)

    # ── Derive final and review flag ──────────────────────────────
    df["vendor_num_final"] = df.apply(
        lambda row: resolve_final(
            row["vendor_num_from_title"],
            row["vendor_num_from_description"]
        ), axis=1
    )

    df["needs_review"] = df.apply(
        lambda row: needs_review_flag(
            row["vendor_num_from_title"],
            row["vendor_num_from_description"]
        ), axis=1
    )

    # ── Summary ───────────────────────────────────────────────────
    total        = len(df)
    found        = (df["vendor_num_final"] != "").sum()
    not_found    = total - found
    review_count = (df["needs_review"] == "Yes").sum()

    print(f"\n{'='*56}")
    print(f"  Total tickets              : {total:>6,}")
    print(f"  Vendor ID found (final)    : {found:>6,}")
    print(f"  No vendor ID found         : {not_found:>6,}")
    print(f"  Needs review (mismatch)    : {review_count:>6,}")
    print(f"{'='*56}\n")

    # ── Save outputs ──────────────────────────────────────────────
    df.to_csv("vendor_id_results_all.csv", index=False)
    print("Saved → vendor_id_results_all.csv")

    found_df = df[df["vendor_num_final"] != ""]
    if len(found_df):
        found_df.to_csv("vendor_id_results_found.csv", index=False)
        print("Saved → vendor_id_results_found.csv")

    review_df = df[df["needs_review"] == "Yes"]
    if len(review_df):
        review_df.to_csv("vendor_id_needs_review.csv", index=False)
        print("Saved → vendor_id_needs_review.csv")

    # ── Optional: write back to SQL Server ────────────────────────
    # conn2 = pyodbc.connect(CONNECTION_STRING)
    # cursor = conn2.cursor()
    # # Run once to add columns:
    # # cursor.execute(f"""
    # #     ALTER TABLE [{TABLE}]
    # #     ADD vendor_num_from_title       NVARCHAR(200) NULL,
    # #         vendor_num_from_description NVARCHAR(200) NULL,
    # #         vendor_num_final            NVARCHAR(200) NULL
    # # """)
    # # conn2.commit()
    # for _, row in df.iterrows():
    #     cursor.execute(f"""
    #         UPDATE [{TABLE}]
    #         SET vendor_num_from_title       = ?,
    #             vendor_num_from_description = ?,
    #             vendor_num_final            = ?
    #         WHERE [{TICKET_ID_COL}] = ?
    #     """,
    #         row["vendor_num_from_title"],
    #         row["vendor_num_from_description"],
    #         row["vendor_num_final"],
    #         row[TICKET_ID_COL]
    #     )
    # conn2.commit()
    # conn2.close()
    # print("SQL Server table updated.")

    return df


if __name__ == "__main__":
    main()
