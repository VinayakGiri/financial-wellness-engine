"""
persona_insights.py
-------------------
Generates statistical + Gemini-powered textual insights for each of the
5 spending personas. Saves outputs/persona_insights.json and
outputs/personas_summary.csv.
"""

import os
import sys
import json
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLUSTERS_PATH   = os.path.join(PROJECT_ROOT, "data",    "user_clusters.csv")
FEATURES_PATH   = os.path.join(PROJECT_ROOT, "data",    "user_features.csv")
TRANSACTIONS_PATH = os.path.join(PROJECT_ROOT, "data",  "transactions_synthetic.csv")
INSIGHTS_PATH   = os.path.join(PROJECT_ROOT, "outputs", "persona_insights.json")
SUMMARY_PATH    = os.path.join(PROJECT_ROOT, "outputs", "personas_summary.csv")

os.makedirs(os.path.join(PROJECT_ROOT, "outputs"), exist_ok=True)

# ── Persona metadata ──────────────────────────────────────────────────────────
PERSONA_META = {
    0: {"name": "Budget-Conscious",  "emoji": "💰"},
    1: {"name": "Impulsive Spender", "emoji": "🔥"},
    2: {"name": "Heavy Subscriber",  "emoji": "📺"},
    3: {"name": "Luxury Spender",    "emoji": "💎"},
    4: {"name": "Weekend Shopper",   "emoji": "🛍️"},
}

def section(title: str):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)

# ─────────────────────────────────────────────────────────────────────────────
# Load .env & Gemini client
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

USE_GEMINI = bool(GEMINI_API_KEY and GEMINI_API_KEY != "paste_your_key_here")

if USE_GEMINI:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        print("  Gemini API   : Connected (gemini-1.5-flash)")
    except Exception as e:
        print(f"  Gemini API   : Failed to initialise — {e}")
        USE_GEMINI = False
else:
    print("  Gemini API   : No valid API key found in .env — using fallback descriptions")

# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────
section("LOADING DATA")

clusters_df = pd.read_csv(CLUSTERS_PATH)
feat_df     = pd.read_csv(FEATURES_PATH, index_col="user_id")
txn_df      = pd.read_csv(TRANSACTIONS_PATH, parse_dates=["date"])

# Merge cluster labels into features
feat_df = feat_df.join(
    clusters_df.set_index("user_id")[["cluster_label", "persona_name"]]
)
debits = txn_df[txn_df["transaction_type"] == "Debit"].copy()

print(f"  Clusters  : {len(clusters_df):,} user assignments loaded")
print(f"  Features  : {feat_df.shape} matrix loaded")
print(f"  Transactions: {len(txn_df):,} rows  |  {debits['category'].nunique()} debit categories")

# ─────────────────────────────────────────────────────────────────────────────
# Helper: compute per-persona statistics from raw transactions
# ─────────────────────────────────────────────────────────────────────────────
ESSENTIAL_CATS     = {"Utilities", "Transport", "Medical"}
DISCRETIONARY_CATS = {"Shopping", "Entertainment", "Subscriptions"}

def compute_persona_stats(cluster_id: int) -> dict:
    """Compute rich statistics for a single persona cluster."""
    meta = PERSONA_META[cluster_id]

    # Users in this cluster
    user_ids = clusters_df[clusters_df["cluster_label"] == cluster_id]["user_id"].tolist()
    cluster_txns   = debits[debits["user_id"].isin(user_ids)]
    cluster_feats  = feat_df[feat_df["cluster_label"] == cluster_id]

    total_spend  = cluster_txns["amount"].sum()
    n_users      = len(user_ids)

    # Category breakdown
    cat_spend = cluster_txns.groupby("category")["amount"].sum()
    cat_pct   = (cat_spend / total_spend * 100).round(1)
    top3_cats = cat_pct.nlargest(3).index.tolist()

    essential_pct     = cat_pct[cat_pct.index.isin(ESSENTIAL_CATS)].sum()
    discretionary_pct = cat_pct[cat_pct.index.isin(DISCRETIONARY_CATS)].sum()
    food_pct          = cat_pct.get("Food", 0.0)

    # Monthly spend (per user, averaged)
    cluster_txns = cluster_txns.copy()
    cluster_txns["month"] = cluster_txns["date"].dt.to_period("M")
    monthly_per_user = (
        cluster_txns.groupby(["user_id", "month"])["amount"].sum()
        .groupby("user_id").mean()
    )
    avg_monthly_spend = monthly_per_user.mean()
    std_monthly_spend = monthly_per_user.std()

    # Savings rate (from normalized features, scale back to interpretable)
    avg_savings_rate = cluster_feats["savings_ratio"].mean()

    # Recurring / subscription commitment
    sub_txns = cluster_txns[cluster_txns["category"] == "Subscriptions"]
    avg_sub_monthly = sub_txns["amount"].sum() / (n_users * 12) if n_users > 0 else 0

    # Weekend ratio
    cluster_txns["weekday"] = cluster_txns["date"].dt.dayofweek
    weekend_spend = cluster_txns[cluster_txns["weekday"] >= 4]["amount"].sum()
    weekend_ratio = weekend_spend / total_spend * 100 if total_spend > 0 else 0

    # Avg transaction amount
    avg_txn_amount = cluster_txns["amount"].mean()

    # Impulse count (proxy: transactions > mean + 2*std for the cluster)
    mean_a, std_a = cluster_txns["amount"].mean(), cluster_txns["amount"].std()
    impulse_count = int((cluster_txns["amount"] > mean_a + 2 * std_a).sum())

    return {
        "cluster_id":          cluster_id,
        "persona_name":        meta["name"],
        "emoji":               meta["emoji"],
        "cluster_size":        n_users,
        "avg_monthly_spend":   round(avg_monthly_spend, 2),
        "std_monthly_spend":   round(std_monthly_spend, 2),
        "avg_savings_rate":    round(float(avg_savings_rate), 4),
        "essential_pct":       round(float(essential_pct), 1),
        "discretionary_pct":   round(float(discretionary_pct), 1),
        "food_pct":            round(float(food_pct), 1),
        "top3_categories":     top3_cats,
        "avg_sub_monthly":     round(avg_sub_monthly, 2),
        "weekend_spend_pct":   round(float(weekend_ratio), 1),
        "avg_txn_amount":      round(avg_txn_amount, 2),
        "impulse_count":       impulse_count,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Helper: call Gemini API
# ─────────────────────────────────────────────────────────────────────────────

FALLBACK_INSIGHTS = {
    "Budget-Conscious":  {
        "characteristic": "Maintains disciplined spending habits with strong essentials focus and minimal discretionary splurges.",
        "recommendation": "Channel existing savings discipline into systematic investments via SIP or recurring deposits to build long-term wealth.",
    },
    "Impulsive Spender": {
        "characteristic": "Shows high weekend spending spikes and frequent large single transactions driven by impulse rather than planning.",
        "recommendation": "Set up weekend spending caps and real-time SMS alerts for transactions above ₹2,000 to curb impulsive behaviour.",
    },
    "Heavy Subscriber":  {
        "characteristic": "Accumulates multiple streaming and lifestyle subscriptions, often with overlapping services and low utilisation.",
        "recommendation": "Conduct a monthly subscription audit — cancel duplicate or low-use services to free ₹500–₹1,500 per month.",
    },
    "Luxury Spender":    {
        "characteristic": "Consistently high average transaction amounts across premium merchants and discretionary categories.",
        "recommendation": "Redirect 10% of discretionary spend into a liquid fund to build an emergency corpus without lifestyle compromise.",
    },
    "Weekend Shopper":   {
        "characteristic": "Concentrates entertainment and shopping spend on weekends with predictable but elevated impulse patterns.",
        "recommendation": "Pre-plan a weekend leisure budget each Monday and use cashback credit cards to earn rewards on predictable spend.",
    },
}

def call_gemini(stats: dict) -> dict:
    """Return {characteristic, recommendation} either from Gemini or fallback."""
    prompt = (
        "You are a financial advisor. Summarize in 2-3 sentences the key characteristic "
        "and one primary recommendation for this spending persona.\n\n"
        f"Persona: {stats['persona_name']}\n"
        f"Monthly Spend: ₹{stats['avg_monthly_spend']:,.0f} (±₹{stats['std_monthly_spend']:,.0f})\n"
        f"Essential: {stats['essential_pct']}%\n"
        f"Discretionary: {stats['discretionary_pct']}%\n"
        f"Food: {stats['food_pct']}%\n"
        f"Savings Rate: {stats['avg_savings_rate']*100:.1f}%\n"
        f"Top Spending Categories: {', '.join(stats['top3_categories'])}\n"
        f"Avg Subscription Spend/Month: ₹{stats['avg_sub_monthly']:,.0f}\n"
        f"Weekend Spend %: {stats['weekend_spend_pct']}%\n\n"
        "Respond in JSON with keys: 'characteristic' and 'recommendation'. "
        "Keep each value under 60 words."
    )

    if not USE_GEMINI:
        return FALLBACK_INSIGHTS.get(stats["persona_name"],
               {"characteristic": "N/A", "recommendation": "N/A"})

    try:
        response = gemini_model.generate_content(prompt)
        text = response.text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        parsed = json.loads(text.strip())
        return {
            "characteristic": parsed.get("characteristic", ""),
            "recommendation":  parsed.get("recommendation",  ""),
        }
    except json.JSONDecodeError:
        # Gemini returned plain text — extract manually
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        return {
            "characteristic": lines[0] if len(lines) > 0 else text,
            "recommendation":  lines[1] if len(lines) > 1 else "",
        }
    except Exception as e:
        print(f"    [WARN] Gemini call failed ({e}) — using fallback")
        return FALLBACK_INSIGHTS.get(stats["persona_name"],
               {"characteristic": "N/A", "recommendation": "N/A"})


# ─────────────────────────────────────────────────────────────────────────────
# Main: compute stats + call Gemini for each persona
# ─────────────────────────────────────────────────────────────────────────────
section("GENERATING PERSONA INSIGHTS")

all_insights = []
summary_rows = []

for cluster_id in sorted(PERSONA_META.keys()):
    meta = PERSONA_META[cluster_id]
    print(f"\n  {meta['emoji']}  Processing Cluster {cluster_id}: {meta['name']} ...")

    stats   = compute_persona_stats(cluster_id)
    insight = call_gemini(stats)
    if USE_GEMINI:
        time.sleep(1.2)   # stay within free-tier rate limits

    # Merge
    full_record = {**stats, **insight}
    all_insights.append(full_record)

    # Print persona summary
    print(f"\n  {'─'*60}")
    print(f"  {meta['emoji']}  {meta['name']}  |  {stats['cluster_size']} users")
    print(f"  {'─'*60}")
    print(f"  Avg Monthly Spend   : ₹{stats['avg_monthly_spend']:>10,.2f}  (±₹{stats['std_monthly_spend']:,.0f})")
    print(f"  Savings Rate        : {stats['avg_savings_rate']*100:>10.1f}%")
    print(f"  Essential Spend     : {stats['essential_pct']:>10.1f}%")
    print(f"  Discretionary Spend : {stats['discretionary_pct']:>10.1f}%")
    print(f"  Weekend Spend       : {stats['weekend_spend_pct']:>10.1f}%")
    print(f"  Avg Txn Amount      : ₹{stats['avg_txn_amount']:>10,.2f}")
    print(f"  Sub Spend/Month     : ₹{stats['avg_sub_monthly']:>10,.2f}")
    print(f"  Top 3 Categories    : {', '.join(stats['top3_categories'])}")
    print(f"  Impulse Txns        : {stats['impulse_count']:>10,}")
    source = "Gemini" if USE_GEMINI else "Fallback"
    print(f"\n  [{source}] Characteristic:")
    # word-wrap
    words, line = insight["characteristic"].split(), "    "
    for w in words:
        if len(line) + len(w) + 1 > 63:
            print(line); line = "    " + w + " "
        else:
            line += w + " "
    if line.strip(): print(line)

    print(f"\n  [{source}] Recommendation:")
    words, line = insight["recommendation"].split(), "    "
    for w in words:
        if len(line) + len(w) + 1 > 63:
            print(line); line = "    " + w + " "
        else:
            line += w + " "
    if line.strip(): print(line)

    # Summary row
    summary_rows.append({
        "Persona Name":        meta["name"],
        "Emoji":               meta["emoji"],
        "Cluster Size":        stats["cluster_size"],
        "Avg Monthly Spend":   f"₹{stats['avg_monthly_spend']:,.0f}",
        "Savings Rate":        f"{stats['avg_savings_rate']*100:.1f}%",
        "Essential %":         f"{stats['essential_pct']}%",
        "Discretionary %":     f"{stats['discretionary_pct']}%",
        "Top Categories":      " | ".join(stats["top3_categories"]),
        "Key Characteristic":  insight["characteristic"],
        "Recommendation":      insight["recommendation"],
    })

# ─────────────────────────────────────────────────────────────────────────────
# Save outputs
# ─────────────────────────────────────────────────────────────────────────────
section("SAVING OUTPUTS")

# JSON
with open(INSIGHTS_PATH, "w", encoding="utf-8") as f:
    json.dump(all_insights, f, indent=2, ensure_ascii=False)
print(f"  Saved insights JSON  -> {INSIGHTS_PATH}")

# CSV summary
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(SUMMARY_PATH, index=False)
print(f"  Saved summary CSV    -> {SUMMARY_PATH}")

# ─────────────────────────────────────────────────────────────────────────────
# Final table
# ─────────────────────────────────────────────────────────────────────────────
section("PERSONA SUMMARY TABLE")
pd.set_option("display.max_colwidth", 55)
pd.set_option("display.width", 200)
display_cols = ["Emoji", "Persona Name", "Cluster Size",
                "Avg Monthly Spend", "Savings Rate", "Top Categories"]
print(summary_df[display_cols].to_string(index=False))

# Verify CSV
loaded = pd.read_csv(SUMMARY_PATH)
print(f"\n  Verified CSV: {loaded.shape[0]} personas × {loaded.shape[1]} columns")

print(f"""
  ╔══════════════════════════════════════════════════════════════╗
  ║                                                              ║
  ║   Persona Insights Complete!                                 ║
  ║   outputs/persona_insights.json   -> Gemini/AI summaries    ║
  ║   outputs/personas_summary.csv    -> Full summary table     ║
  ║                                                              ║
  ╚══════════════════════════════════════════════════════════════╝
""")
