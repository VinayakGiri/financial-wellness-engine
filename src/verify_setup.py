"""
verify_setup.py
---------------
Runs a comprehensive setup verification for the Financial Wellness Engine.
Checks packages, data file, CSV integrity, and data quality.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH    = os.path.join(PROJECT_ROOT, "data", "transactions_synthetic.csv")

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"
INFO = "[ >> ]"

def section(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def check(label: str, ok: bool, detail: str = ""):
    status = PASS if ok else FAIL
    msg = f"  {status}  {label}"
    if detail:
        msg += f"\n         {detail}"
    print(msg)
    return ok

# ─────────────────────────────────────────────────────────────────────────────
# 1. Package imports
# ─────────────────────────────────────────────────────────────────────────────
section("1. PACKAGE IMPORT CHECK")

all_imports_ok = True

# Packages: (module_name, optional)
packages = [
    ("pandas",           False),
    ("numpy",            False),
    ("sklearn",          False),
    ("streamlit",        False),
    ("plotly",           False),
    ("joblib",           False),
    ("faker",            False),
    ("google.generativeai", False),
    ("tensorflow",       True),   # optional — not yet compatible with Python 3.14
]

imported = {}
for module, optional in packages:
    display = module.replace("google.generativeai", "google-generativeai")
    try:
        import importlib
        mod = importlib.import_module(module)
        version = getattr(mod, "__version__", "n/a")
        imported[module] = mod
        check(f"{display:<22} v{version}", True)
    except ImportError as e:
        if optional:
            print(f"  {WARN}  {display:<22} not installed (optional — Python 3.14 incompatible)")
        else:
            check(f"{display:<22}", False, str(e))
            all_imports_ok = False

# Venv check
venv_active = sys.prefix != sys.base_prefix
check(f"venv active          ({sys.prefix})", venv_active)
print(f"  {INFO}  Python  {sys.version.split()[0]}  |  Prefix: {sys.prefix}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Data folder / CSV check
# ─────────────────────────────────────────────────────────────────────────────
section("2. DATA FILE CHECK")

csv_exists = os.path.isfile(DATA_PATH)
check("transactions_synthetic.csv exists", csv_exists, DATA_PATH)

if not csv_exists:
    print(f"\n  {FAIL}  Cannot continue — CSV not found. Run src/generate_data.py first.")
    sys.exit(1)

file_size_kb = os.path.getsize(DATA_PATH) / 1024
check(f"File size                          {file_size_kb:,.1f} KB", file_size_kb > 0)

# ─────────────────────────────────────────────────────────────────────────────
# 3. CSV Content Verification
# ─────────────────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np

section("3. CSV CONTENT VERIFICATION")

df = pd.read_csv(DATA_PATH, parse_dates=["date"])

# Shape
rows, cols = df.shape
check(f"Shape                              {rows:,} rows x {cols} columns",
      rows >= 6_000 and cols == 7)

# ── First 10 rows ─────────────────────────────────────────────────────────────
print(f"\n  {INFO}  First 10 rows:")
print()
print(df.head(10).to_string(index=False))

# ── Data types ────────────────────────────────────────────────────────────────
print(f"\n  {INFO}  Data types:")
for col, dtype in df.dtypes.items():
    print(f"         {col:<20}  {dtype}")

# ── Expected columns ─────────────────────────────────────────────────────────
expected_cols = {"user_id", "date", "amount", "category",
                 "merchant_name", "transaction_type", "description"}
actual_cols   = set(df.columns)
check("All required columns present",
      expected_cols == actual_cols,
      f"Found: {sorted(actual_cols)}")

# ── Category distribution ─────────────────────────────────────────────────────
print(f"\n  {INFO}  Category value counts:")
cat_counts = df["category"].value_counts()
for cat, count in cat_counts.items():
    pct = count / len(df) * 100
    bar = "#" * int(pct / 2)
    print(f"         {cat:<15}  {count:>6,}  ({pct:5.1f}%)  {bar}")

check("All 8 categories present",
      len(cat_counts) == 8,
      f"Found {len(cat_counts)}: {list(cat_counts.index)}")

# ── Revenue vs Expenses ───────────────────────────────────────────────────────
print(f"\n  {INFO}  Financial Summary:")
credits = df[df["transaction_type"] == "Credit"]["amount"].sum()
debits  = df[df["transaction_type"] == "Debit"]["amount"].sum()
net     = credits - debits
print(f"         Total Income  (Credit) : Rs {credits:>14,.2f}")
print(f"         Total Expenses (Debit) : Rs {debits:>14,.2f}")
print(f"         Net Balance             : Rs {net:>14,.2f}  {'(Surplus)' if net >= 0 else '(Deficit)'}")

check("Income (Credit) transactions exist",  credits > 0)
check("Expense (Debit) transactions exist",   debits  > 0)

# ── Amount range ──────────────────────────────────────────────────────────────
amt_min, amt_max = df["amount"].min(), df["amount"].max()
check(f"Amount range in bounds (Rs 50 – Rs 80,000)",
      50 <= amt_min and amt_max <= 80_000,
      f"Actual: Rs {amt_min:,.2f} – Rs {amt_max:,.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Data Quality Report
# ─────────────────────────────────────────────────────────────────────────────
section("4. DATA QUALITY REPORT")

# ── Missing values ────────────────────────────────────────────────────────────
print(f"  {INFO}  Missing values per column:")
missing = df.isnull().sum()
any_missing = False
for col, cnt in missing.items():
    flag = FAIL if cnt > 0 else PASS
    print(f"         {flag}  {col:<20}  {cnt} missing")
    if cnt > 0:
        any_missing = True
check("No missing values in dataset", not any_missing)

# ── Duplicate rows ────────────────────────────────────────────────────────────
dup_rows = df.duplicated().sum()
check(f"No fully duplicate rows            ({dup_rows} found)", dup_rows == 0)

# ── Unique users ──────────────────────────────────────────────────────────────
unique_users = df["user_id"].nunique()
check(f"500 unique users present           ({unique_users} found)", unique_users == 500)

# ── Date range — 12 months ────────────────────────────────────────────────────
date_min  = df["date"].min()
date_max  = df["date"].max()
date_span = (date_max - date_min).days
months_covered = date_span / 30.4
check(f"Date span covers ~12 months        ({months_covered:.1f} months  |  "
      f"{date_min.date()} to {date_max.date()})",
      months_covered >= 11.5)

# ── Transaction type distribution ────────────────────────────────────────────
txn_counts = df["transaction_type"].value_counts()
print(f"\n  {INFO}  Transaction type split:")
for t, c in txn_counts.items():
    print(f"         {t:<10}  {c:,}  ({c/len(df)*100:.1f}%)")
check("Both Debit and Credit types present",
      "Debit" in txn_counts and "Credit" in txn_counts)

# ── Merchant diversity ────────────────────────────────────────────────────────
unique_merchants = df["merchant_name"].nunique()
check(f"Multiple merchants present         ({unique_merchants} unique merchants)",
      unique_merchants >= 10)

# ── Per-user stats ────────────────────────────────────────────────────────────
per_user = df.groupby("user_id")["amount"].agg(["count", "sum", "mean"])
print(f"\n  {INFO}  Per-user transaction stats:")
print(f"         Avg transactions/user : {per_user['count'].mean():.1f}")
print(f"         Min transactions/user : {per_user['count'].min()}")
print(f"         Max transactions/user : {per_user['count'].max()}")
print(f"         Avg spend/user        : Rs {per_user['sum'].mean():,.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# Final verdict
# ─────────────────────────────────────────────────────────────────────────────
section("VERIFICATION SUMMARY")

critical_checks = [
    csv_exists,
    rows >= 6_000,
    not any_missing,
    unique_users == 500,
    months_covered >= 11.5,
    expected_cols == actual_cols,
    credits > 0,
    debits > 0,
]
all_passed = all(critical_checks)

if all_passed:
    print("""
  ╔══════════════════════════════════════════════════════════╗
  ║                                                          ║
  ║   Setup Complete -- Ready to Build Financial Wellness    ║
  ║                        Engine                           ║
  ║                                                          ║
  ╚══════════════════════════════════════════════════════════╝
""")
else:
    failed = sum(1 for c in critical_checks if not c)
    print(f"\n  {FAIL}  {failed} critical check(s) failed. Review output above.\n")
    sys.exit(1)
