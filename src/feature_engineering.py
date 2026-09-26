"""
feature_engineering.py
-----------------------
Transforms raw transaction data into user-level behavioural features
for clustering. Saves feature matrix to data/user_features.csv and
a correlation heatmap to outputs/correlation_matrix.png.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH      = os.path.join(PROJECT_ROOT, "data", "transactions_synthetic.csv")
FEATURES_PATH  = os.path.join(PROJECT_ROOT, "data", "user_features.csv")
HEATMAP_PATH   = os.path.join(PROJECT_ROOT, "outputs", "correlation_matrix.png")

os.makedirs(os.path.join(PROJECT_ROOT, "outputs"), exist_ok=True)

def section(title: str):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)

# ─────────────────────────────────────────────────────────────────────────────
# Load & preprocess
# ─────────────────────────────────────────────────────────────────────────────
section("LOADING DATA")

df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df["amount"]  = df["amount"].astype(float)
df["weekday"] = df["date"].dt.dayofweek          # 0=Mon … 6=Sun
df["month"]   = df["date"].dt.to_period("M")
df["day"]     = df["date"].dt.day

debits  = df[df["transaction_type"] == "Debit"].copy()
credits = df[df["transaction_type"] == "Credit"].copy()

print(f"  Loaded {len(df):,} transactions for {df['user_id'].nunique()} users")
print(f"  Debits : {len(debits):,}   |   Credits : {len(credits):,}")

users = df["user_id"].unique()

# ─────────────────────────────────────────────────────────────────────────────
# Feature computation helpers
# ─────────────────────────────────────────────────────────────────────────────

def safe_div(a, b, default=0.0):
    """Division that returns default when denominator is 0."""
    return a / b if b != 0 else default


def compute_features(uid: str) -> dict:
    u_all    = df[df["user_id"] == uid]
    u_debit  = debits[debits["user_id"] == uid]
    u_credit = credits[credits["user_id"] == uid]

    total_spending = u_debit["amount"].sum()
    total_income   = u_credit["amount"].sum()

    # ── 1. Spending Features ─────────────────────────────────────────────────
    monthly_spend = (
        u_debit.groupby("month")["amount"].sum()
        .reindex(pd.period_range(u_all["month"].min(),
                                 u_all["month"].max(), freq="M"),
                 fill_value=0.0)
    )
    avg_monthly_spend  = monthly_spend.mean()
    std_monthly_spend  = monthly_spend.std(ddof=0)
    max_monthly_spend  = monthly_spend.max()

    cat_spend = u_debit.groupby("category")["amount"].sum()
    food_ratio          = safe_div(cat_spend.get("Food", 0),          total_spending)
    entertainment_ratio = safe_div(cat_spend.get("Entertainment", 0), total_spending)
    essential_ratio     = safe_div(
        cat_spend.get("Utilities", 0) +
        cat_spend.get("Transport", 0) +
        cat_spend.get("Medical",   0),
        total_spending,
    )
    discretionary_ratio = safe_div(
        cat_spend.get("Shopping", 0) +
        cat_spend.get("Entertainment", 0),
        total_spending,
    )
    savings_ratio = safe_div(total_income - total_spending, total_income)

    # ── 2. Temporal Features ─────────────────────────────────────────────────
    weekend_mask           = u_debit["weekday"] >= 4          # Fri=4, Sat=5, Sun=6
    weekend_spending       = u_debit.loc[weekend_mask,  "amount"].sum()
    weekday_spending       = u_debit.loc[~weekend_mask, "amount"].sum()
    weekend_spending_ratio = safe_div(weekend_spending, total_spending)
    weekday_spending_ratio = safe_div(weekday_spending, total_spending)

    # Avg spend in the 5 days after salary (days 1-5 of month)
    post_salary = u_debit[u_debit["day"].between(1, 5)]
    salary_day_behavior = post_salary["amount"].mean() if len(post_salary) > 0 else 0.0

    # Spending volatility: cv of monthly spend
    spending_volatility = safe_div(std_monthly_spend, avg_monthly_spend)

    # ── 3. Subscription Features ─────────────────────────────────────────────
    sub_txns = u_debit[u_debit["category"] == "Subscriptions"]

    # Recurring = merchant appears in ≥2 different months
    sub_by_merchant = sub_txns.groupby("merchant_name")["month"].nunique()
    recurring_merchants      = sub_by_merchant[sub_by_merchant >= 2]
    recurring_payment_count  = len(recurring_merchants)

    # Monthly subscription total (avg per month across recurring merchants)
    if len(recurring_merchants) > 0:
        rec_mask = sub_txns["merchant_name"].isin(recurring_merchants.index)
        subscription_total_monthly = (
            sub_txns.loc[rec_mask, "amount"].sum() /
            max(monthly_spend.shape[0], 1)
        )
    else:
        subscription_total_monthly = 0.0

    unique_merchants_count      = u_all["merchant_name"].nunique()
    subscription_to_income_ratio = safe_div(sub_txns["amount"].sum(), total_income)

    # ── 4. Impulse Features ───────────────────────────────────────────────────
    # Impulse = single debit > 2 std above user's own mean
    if len(u_debit) > 1:
        mean_amt  = u_debit["amount"].mean()
        std_amt   = u_debit["amount"].std(ddof=0)
        threshold = mean_amt + 2 * std_amt
        impulse_transaction_count = int((u_debit["amount"] > threshold).sum())
    else:
        impulse_transaction_count = 0

    avg_transaction_amount  = u_debit["amount"].mean() if len(u_debit) > 0 else 0.0
    transaction_frequency   = len(u_all) / 365.0

    return {
        # Spending
        "total_spending":              total_spending,
        "avg_monthly_spend":           avg_monthly_spend,
        "std_monthly_spend":           std_monthly_spend,
        "max_monthly_spend":           max_monthly_spend,
        "food_ratio":                  food_ratio,
        "entertainment_ratio":         entertainment_ratio,
        "essential_ratio":             essential_ratio,
        "discretionary_ratio":         discretionary_ratio,
        "savings_ratio":               savings_ratio,
        # Temporal
        "weekend_spending_ratio":      weekend_spending_ratio,
        "weekday_spending_ratio":      weekday_spending_ratio,
        "salary_day_behavior":         salary_day_behavior,
        "spending_volatility":         spending_volatility,
        # Subscription
        "recurring_payment_count":     recurring_payment_count,
        "subscription_total_monthly":  subscription_total_monthly,
        "unique_merchants_count":      unique_merchants_count,
        "subscription_to_income_ratio": subscription_to_income_ratio,
        # Impulse
        "impulse_transaction_count":   impulse_transaction_count,
        "avg_transaction_amount":      avg_transaction_amount,
        "transaction_frequency":       transaction_frequency,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Compute features for all 500 users
# ─────────────────────────────────────────────────────────────────────────────
section("COMPUTING USER FEATURES")
print(f"  Processing {len(users)} users ", end="", flush=True)

records = []
for i, uid in enumerate(users):
    records.append({"user_id": uid, **compute_features(uid)})
    if (i + 1) % 50 == 0:
        print(".", end="", flush=True)

print(f"  done!\n")

feat_df = pd.DataFrame(records).set_index("user_id")
feature_cols = [c for c in feat_df.columns]

print(f"  Feature matrix shape : {feat_df.shape[0]} users x {feat_df.shape[1]} features")

# ─────────────────────────────────────────────────────────────────────────────
# Feature ranges
# ─────────────────────────────────────────────────────────────────────────────
section("FEATURE NAMES & RANGES (RAW)")
print(f"  {'Feature':<32}  {'Min':>12}  {'Max':>12}  {'Mean':>12}  {'Std':>12}")
print(f"  {'-'*32}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}")
for col in feature_cols:
    s = feat_df[col]
    print(f"  {col:<32}  {s.min():>12.3f}  {s.max():>12.3f}  {s.mean():>12.3f}  {s.std():>12.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# Normalize with StandardScaler
# ─────────────────────────────────────────────────────────────────────────────
section("NORMALIZATION (StandardScaler → 0-1 clipped)")

scaler   = StandardScaler()
scaled   = scaler.fit_transform(feat_df[feature_cols])

# Clip to [0,1] after min-max rescaling on top of standard scaling
from sklearn.preprocessing import MinMaxScaler
mm_scaler = MinMaxScaler()
scaled_01 = mm_scaler.fit_transform(scaled)

feat_scaled = pd.DataFrame(scaled_01, index=feat_df.index, columns=feature_cols)
print(f"  All features normalized to [0, 1]")
print(f"  Global min={feat_scaled.min().min():.4f}  max={feat_scaled.max().max():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Save feature matrix
# ─────────────────────────────────────────────────────────────────────────────
feat_scaled.to_csv(FEATURES_PATH)
print(f"\n  Saved feature matrix -> {FEATURES_PATH}")

# ─────────────────────────────────────────────────────────────────────────────
# Summary statistics
# ─────────────────────────────────────────────────────────────────────────────
section("SUMMARY STATISTICS (NORMALIZED)")
print(feat_scaled.describe().round(4).to_string())

# ─────────────────────────────────────────────────────────────────────────────
# Correlation matrix heatmap
# ─────────────────────────────────────────────────────────────────────────────
section("CORRELATION MATRIX HEATMAP")

corr = feat_scaled.corr()

fig, ax = plt.subplots(figsize=(14, 11))
mask = np.triu(np.ones_like(corr, dtype=bool))           # upper-triangle mask

cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(
    corr,
    mask=mask,
    cmap=cmap,
    vmin=-1, vmax=1,
    center=0,
    annot=True,
    fmt=".2f",
    annot_kws={"size": 7},
    linewidths=0.5,
    ax=ax,
    square=True,
    cbar_kws={"shrink": 0.8, "label": "Pearson r"},
)
ax.set_title("User Behavioural Feature — Correlation Matrix\n(Financial Wellness Engine)",
             fontsize=13, fontweight="bold", pad=16)
ax.tick_params(axis="x", rotation=45, labelsize=8)
ax.tick_params(axis="y", rotation=0,  labelsize=8)
plt.tight_layout()
plt.savefig(HEATMAP_PATH, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Heatmap saved -> {HEATMAP_PATH}")

# ─────────────────────────────────────────────────────────────────────────────
# Top correlations
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n  Top 10 absolute correlations (excluding self):")
corr_pairs = (
    corr.where(~mask)
        .stack()
        .reset_index()
        .rename(columns={"level_0": "feature_a", "level_1": "feature_b", 0: "r"})
)
corr_pairs["abs_r"] = corr_pairs["r"].abs()
top10 = corr_pairs.nlargest(10, "abs_r")
for _, row in top10.iterrows():
    bar = "#" * int(abs(row["r"]) * 20)
    print(f"  {row['feature_a']:<30} <-> {row['feature_b']:<30}  r={row['r']:+.3f}  {bar}")

# ─────────────────────────────────────────────────────────────────────────────
# Final confirmation
# ─────────────────────────────────────────────────────────────────────────────
section("FINAL CONFIRMATION")
saved_df = pd.read_csv(FEATURES_PATH, index_col="user_id")
print(f"  Feature matrix loaded back: {saved_df.shape[0]} users x {saved_df.shape[1]} features")
print(f"  Feature columns ({len(feature_cols)}):")
for i, col in enumerate(feature_cols, 1):
    print(f"    {i:>2}. {col}")

print("""
  ╔══════════════════════════════════════════════════════════════╗
  ║                                                              ║
  ║   Feature Engineering Complete!                             ║
  ║   data/user_features.csv  ->  500 users x 20 features       ║
  ║   outputs/correlation_matrix.png  ->  saved                 ║
  ║                                                              ║
  ╚══════════════════════════════════════════════════════════════╝
""")
