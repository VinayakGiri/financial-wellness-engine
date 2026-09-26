"""
leakage_detection.py
--------------------
Identifies forgotten / low-usage subscriptions and calculates a
Leakage Score for each user's recurring payments.

Outputs
-------
data/recurring_transactions.csv   – all recurring (user, merchant) pairs
data/user_leakage.csv             – per-user leakage scores & status
outputs/top_leakages.png          – bar chart of top-20 leakages
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime, timedelta

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TXN_PATH           = os.path.join(PROJECT_ROOT, "data",    "transactions_synthetic.csv")
RECURRING_PATH     = os.path.join(PROJECT_ROOT, "data",    "recurring_transactions.csv")
LEAKAGE_PATH       = os.path.join(PROJECT_ROOT, "data",    "user_leakage.csv")
TOP_LEAKAGES_PATH  = os.path.join(PROJECT_ROOT, "outputs", "top_leakages.png")

os.makedirs(os.path.join(PROJECT_ROOT, "outputs"), exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
RECURRENCE_THRESHOLD   = 3    # appear in ≥ 3 of 12 months to count as recurring
ACTIVE_DAYS            = 30   # last txn within 30 days  → Active
DORMANT_DAYS           = 60   # last txn > 60 days ago   → Dormant
DORMANCY_MULTIPLIER    = 1.5
OVERLAP_MULTIPLIER     = 2.0

# Category groups for duplicate detection
DUPLICATE_GROUPS = {
    "food_delivery":    {"Zomato", "Swiggy", "Dunzo", "Blinkit"},
    "streaming_video":  {"Netflix", "Amazon Prime", "Disney+ Hotstar",
                         "Zee5", "SonyLIV", "Airtel Xstream", "Apple TV+"},
    "streaming_music":  {"Spotify", "JioSaavn", "Gaana", "YouTube Premium"},
    "internet":         {"Jio Fiber", "Jio Recharge", "Airtel Postpaid", "BSNL"},
}
MERCHANT_TO_GROUP = {
    merchant: group
    for group, merchants in DUPLICATE_GROUPS.items()
    for merchant in merchants
}

STATUS_COLORS = {
    "Active":    "#4CAF50",   # green
    "Low Usage": "#FFC107",   # amber
    "Dormant":   "#F44336",   # red
    "Duplicate": "#9C27B0",   # purple
}

def section(title: str):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────────────────────────────────────────
section("1. LOADING TRANSACTIONS")

df = pd.read_csv(TXN_PATH, parse_dates=["date"])
df["amount"] = df["amount"].astype(float)

# Use only subscription & food-delivery debits for recurrence detection
sub_cats = {"Subscriptions", "Food", "Utilities", "Transport", "Entertainment"}
sub_df   = df[(df["transaction_type"] == "Debit") &
              (df["category"].isin(sub_cats))].copy()

sub_df["month"]  = sub_df["date"].dt.to_period("M")
sub_df["ym_int"] = sub_df["date"].dt.year * 100 + sub_df["date"].dt.month

REFERENCE_DATE = df["date"].max()
print(f"  Total transactions       : {len(df):,}")
print(f"  Candidate debit rows     : {len(sub_df):,}  (categories: {sorted(sub_cats)})")
print(f"  Reference date (latest)  : {REFERENCE_DATE.date()}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Identify Recurring Transactions
# ─────────────────────────────────────────────────────────────────────────────
section("2. IDENTIFYING RECURRING PAYMENTS")

group_keys = ["user_id", "merchant_name", "category"]

agg = (
    sub_df.groupby(group_keys)
    .agg(
        total_amount   = ("amount", "sum"),
        avg_amount     = ("amount", "mean"),
        txn_count      = ("amount", "count"),
        frequency_months = ("month", "nunique"),
        first_date     = ("date", "min"),
        last_date      = ("date", "max"),
    )
    .reset_index()
)

# Keep only groups that appear in ≥ RECURRENCE_THRESHOLD distinct months
recurring = agg[agg["frequency_months"] >= RECURRENCE_THRESHOLD].copy()
recurring["monthly_amount"] = (
    recurring["avg_amount"].round(2)
)
recurring["last_date"] = pd.to_datetime(recurring["last_date"])
recurring["dormancy_days"] = (
    (REFERENCE_DATE - recurring["last_date"]).dt.days
)

print(f"  Unique (user, merchant) groups   : {len(agg):,}")
print(f"  Recurring (≥{RECURRENCE_THRESHOLD} months)          : {len(recurring):,}")

recurring_out = recurring[
    ["user_id", "merchant_name", "category",
     "monthly_amount", "frequency_months", "last_date"]
].copy()
recurring_out.to_csv(RECURRING_PATH, index=False)
print(f"  Saved -> {RECURRING_PATH}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Classify by Usage Pattern
# ─────────────────────────────────────────────────────────────────────────────
section("3. CLASSIFYING SUBSCRIPTION STATUS")

# cost_per_use = monthly_amount / avg monthly txn count
# txn_count / frequency_months gives avg uses per month
recurring["uses_per_month"] = (
    recurring["txn_count"] / recurring["frequency_months"].clip(lower=1)
)
recurring["cost_per_use"] = (
    recurring["monthly_amount"] / recurring["uses_per_month"].clip(lower=0.1)
).round(2)

# Thresholds for Low Usage
cpu_median = recurring["cost_per_use"].median()
cpu_std    = recurring["cost_per_use"].std()
LOW_USAGE_THRESHOLD = cpu_median + cpu_std

print(f"  Cost-per-use  median : ₹{cpu_median:.2f}")
print(f"  Cost-per-use  std    : ₹{cpu_std:.2f}")
print(f"  Low-usage threshold  : ₹{LOW_USAGE_THRESHOLD:.2f}")

# Duplicate detection: user has ≥2 merchants from the same duplicate group
recurring["dup_group"] = recurring["merchant_name"].map(MERCHANT_TO_GROUP)

user_dup_groups = (
    recurring.dropna(subset=["dup_group"])
    .groupby(["user_id", "dup_group"])["merchant_name"]
    .nunique()
    .reset_index()
    .rename(columns={"merchant_name": "n_merchants"})
)
user_dup_set = set(
    zip(user_dup_groups[user_dup_groups["n_merchants"] >= 2]["user_id"],
        user_dup_groups[user_dup_groups["n_merchants"] >= 2]["dup_group"])
)

def classify_status(row) -> str:
    # Dormant takes priority
    if row["dormancy_days"] > DORMANT_DAYS:
        return "Dormant"
    # Duplicate: user has another merchant in same service group
    if pd.notna(row["dup_group"]):
        if (row["user_id"], row["dup_group"]) in user_dup_set:
            return "Duplicate"
    # Low usage
    if row["cost_per_use"] > LOW_USAGE_THRESHOLD:
        return "Low Usage"
    return "Active"

recurring["status"] = recurring.apply(classify_status, axis=1)

status_counts = recurring["status"].value_counts()
print(f"\n  Status distribution:")
for status, count in status_counts.items():
    bar = "█" * int(count / len(recurring) * 40)
    print(f"    {status:<12}  {count:>5,}  ({count/len(recurring)*100:5.1f}%)  {bar}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Calculate Leakage Score
# ─────────────────────────────────────────────────────────────────────────────
section("4. CALCULATING LEAKAGE SCORES")

# Normalise cost_per_use to [0, 1] for scoring
cpu_min  = recurring["cost_per_use"].min()
cpu_max  = recurring["cost_per_use"].max()
recurring["cpu_norm"] = (
    (recurring["cost_per_use"] - cpu_min) / (cpu_max - cpu_min + 1e-9)
)

# Dormancy factor
recurring["dormancy_factor"] = np.where(
    recurring["dormancy_days"] > ACTIVE_DAYS, DORMANCY_MULTIPLIER, 1.0
)

# Overlap factor
recurring["overlap_factor"] = np.where(
    recurring["status"] == "Duplicate", OVERLAP_MULTIPLIER, 1.0
)

# Raw leakage score (0-100)
recurring["leakage_score_raw"] = (
    recurring["cpu_norm"] *
    recurring["dormancy_factor"] *
    recurring["overlap_factor"] * 100
)

# Clip & round to 0-100
ls_min = recurring["leakage_score_raw"].min()
ls_max = recurring["leakage_score_raw"].max()
recurring["leakage_score"] = (
    (recurring["leakage_score_raw"] - ls_min) /
    (ls_max - ls_min + 1e-9) * 100
).clip(0, 100).round(2)

# Estimated annual saving = monthly_amount × 12 for non-Active
recurring["estimated_annual_saving"] = np.where(
    recurring["status"] != "Active",
    (recurring["monthly_amount"] * 12).round(2),
    0.0,
)

print(f"  Leakage score range : {recurring['leakage_score'].min():.2f} – {recurring['leakage_score'].max():.2f}")
print(f"  Mean leakage score  : {recurring['leakage_score'].mean():.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Save user_leakage.csv
# ─────────────────────────────────────────────────────────────────────────────
section("5. SAVING USER LEAKAGE DATA")

leakage_out = recurring[[
    "user_id", "merchant_name", "category",
    "monthly_amount", "cost_per_use", "dormancy_days",
    "leakage_score", "status", "estimated_annual_saving"
]].sort_values("leakage_score", ascending=False).reset_index(drop=True)

leakage_out.to_csv(LEAKAGE_PATH, index=False)
print(f"  Rows saved           : {len(leakage_out):,}")
print(f"  Saved -> {LEAKAGE_PATH}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Visualization — Top 20 leakages
# ─────────────────────────────────────────────────────────────────────────────
section("6. VISUALIZATION — TOP 20 LEAKAGES")

# Aggregate by merchant (across users) for a portfolio-level chart
agg_leakage = (
    leakage_out.groupby(["merchant_name", "status"])
    .agg(
        avg_leakage_score     = ("leakage_score",          "mean"),
        avg_monthly_cost      = ("monthly_amount",         "mean"),
        total_annual_savings  = ("estimated_annual_saving","sum"),
        user_count            = ("user_id",                "nunique"),
    )
    .reset_index()
    .sort_values("avg_leakage_score", ascending=False)
    .head(20)
)

# Sort for horizontal bar chart (lowest at top)
plot_df = agg_leakage.sort_values("avg_leakage_score", ascending=True)
colors  = [STATUS_COLORS.get(s, "#90A4AE") for s in plot_df["status"]]

fig, axes = plt.subplots(1, 2, figsize=(16, 9),
                         gridspec_kw={"width_ratios": [3, 1]})
fig.patch.set_facecolor("#f8f9fa")

# ── Left: horizontal bar chart ─────────────────────────────────────────────
ax = axes[0]
ax.set_facecolor("#f8f9fa")

bars = ax.barh(
    range(len(plot_df)),
    plot_df["avg_leakage_score"],
    color=colors,
    edgecolor="white",
    linewidth=0.6,
    height=0.7,
)

# Annotate bars with monthly cost
for i, (_, row) in enumerate(plot_df.iterrows()):
    ax.text(
        row["avg_leakage_score"] + 0.5,
        i,
        f" ₹{row['avg_monthly_cost']:,.0f}/mo  ({row['user_count']} users)",
        va="center", ha="left", fontsize=8, color="#333333",
    )

ax.set_yticks(range(len(plot_df)))
ax.set_yticklabels(
    [f"{name}  [{status[:3].upper()}]"
     for name, status in zip(plot_df["merchant_name"], plot_df["status"])],
    fontsize=9,
)
ax.set_xlabel("Average Leakage Score (0–100)", fontsize=11)
ax.set_title(
    "Top 20 Recurring Subscriptions by Leakage Score\n"
    "Financial Wellness Engine — Subscription Leak Detector",
    fontsize=12, fontweight="bold", pad=12,
)
ax.set_xlim(0, plot_df["avg_leakage_score"].max() * 1.35)
ax.grid(axis="x", alpha=0.25, linestyle="--")
ax.spines[["top", "right"]].set_visible(False)

# Legend
legend_patches = [
    mpatches.Patch(color=c, label=s)
    for s, c in STATUS_COLORS.items()
]
ax.legend(handles=legend_patches, title="Status", fontsize=9,
          loc="lower right", framealpha=0.9)

# ── Right: annual savings donut ────────────────────────────────────────────
ax2 = axes[1]
ax2.set_facecolor("#f8f9fa")

status_savings = (
    leakage_out[leakage_out["status"] != "Active"]
    .groupby("status")["estimated_annual_saving"]
    .sum()
)
donut_colors = [STATUS_COLORS[s] for s in status_savings.index]
wedges, texts, autotexts = ax2.pie(
    status_savings.values,
    labels=status_savings.index,
    colors=donut_colors,
    autopct="%1.1f%%",
    startangle=90,
    pctdistance=0.75,
    wedgeprops={"width": 0.5, "edgecolor": "white", "linewidth": 2},
)
for t in texts:
    t.set_fontsize(9)
for at in autotexts:
    at.set_fontsize(8)

total_saving = status_savings.sum()
ax2.text(0, 0,
         f"₹{total_saving/1e6:.1f}M\nTotal",
         ha="center", va="center",
         fontsize=11, fontweight="bold", color="#333333")
ax2.set_title("Annual Leakage by Status", fontsize=11, fontweight="bold", pad=10)

plt.tight_layout(pad=2)
plt.savefig(TOP_LEAKAGES_PATH, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Chart saved -> {TOP_LEAKAGES_PATH}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Print Summary
# ─────────────────────────────────────────────────────────────────────────────
section("7. SUMMARY REPORT")

total_recurring = len(leakage_out)
print(f"\n  Total recurring subscriptions detected : {total_recurring:,}")
print(f"\n  Distribution by status:")
for status, count in leakage_out["status"].value_counts().items():
    pct = count / total_recurring * 100
    bar = "█" * int(pct / 3)
    print(f"    {status:<12}  {count:>5,}  ({pct:5.1f}%)  {bar}")

print(f"\n  Top 10 Leakages (by Leakage Score):")
print(f"\n  {'#':<3}  {'User':<9}  {'Merchant':<22}  {'Status':<12}  "
      f"{'Score':>7}  {'Monthly':>9}  {'Annual Saving':>14}")
print(f"  {'─'*3}  {'─'*9}  {'─'*22}  {'─'*12}  "
      f"{'─'*7}  {'─'*9}  {'─'*14}")

for i, row in leakage_out.head(10).iterrows():
    print(f"  {i+1:<3}  {row['user_id']:<9}  {row['merchant_name']:<22}  "
          f"{row['status']:<12}  {row['leakage_score']:>7.2f}  "
          f"₹{row['monthly_amount']:>7,.0f}  "
          f"₹{row['estimated_annual_saving']:>12,.0f}")

# Per-user leakage summary
user_summary = (
    leakage_out[leakage_out["status"] != "Active"]
    .groupby("user_id")
    .agg(
        subscriptions_leaking = ("merchant_name", "count"),
        total_monthly_leakage = ("monthly_amount", "sum"),
        total_annual_saving   = ("estimated_annual_saving", "sum"),
        max_leakage_score     = ("leakage_score", "max"),
    )
    .reset_index()
    .sort_values("total_annual_saving", ascending=False)
)

total_potential_saving = leakage_out["estimated_annual_saving"].sum()
avg_per_user = total_potential_saving / leakage_out["user_id"].nunique()

print(f"\n  ─── Aggregate Savings Potential ───────────────────────────")
print(f"  Total potential annual savings (all users) : ₹{total_potential_saving:>12,.0f}")
print(f"  Avg potential saving per user              : ₹{avg_per_user:>12,.0f}")
print(f"  Users with ≥1 leaking subscription        : {len(user_summary):>12,}")
print(f"\n  Top 5 users by potential savings:")
print(f"  {'User':<10}  {'Leaking Subs':>14}  {'Monthly Leakage':>16}  "
      f"{'Annual Saving':>14}  {'Max Score':>10}")
print(f"  {'─'*10}  {'─'*14}  {'─'*16}  {'─'*14}  {'─'*10}")
for _, row in user_summary.head(5).iterrows():
    print(f"  {row['user_id']:<10}  {row['subscriptions_leaking']:>14}  "
          f"₹{row['total_monthly_leakage']:>14,.0f}  "
          f"₹{row['total_annual_saving']:>12,.0f}  "
          f"{row['max_leakage_score']:>10.2f}")

# Verify
loaded = pd.read_csv(LEAKAGE_PATH)
print(f"\n  Verified CSV: {loaded.shape[0]:,} rows × {loaded.shape[1]} columns")

print(f"""
  ╔══════════════════════════════════════════════════════════════╗
  ║                                                              ║
  ║   Leakage Detection Complete!                                ║
  ║                                                              ║
  ║   data/recurring_transactions.csv  -> {len(recurring_out):>5,} recurring pairs  ║
  ║   data/user_leakage.csv            -> {len(leakage_out):>5,} leakage records  ║
  ║   outputs/top_leakages.png         -> top-20 bar chart      ║
  ║                                                              ║
  ║   Total Potential Annual Savings: ₹{total_potential_saving:>12,.0f}       ║
  ║                                                              ║
  ╚══════════════════════════════════════════════════════════════╝
""")
