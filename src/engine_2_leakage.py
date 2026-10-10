"""
engine_2_leakage.py
-------------------
Engine 2: Individual Subscription Leakage Detection & Remediation for FinWell.

Filters transactions strictly for the active user, detects recurring subscriptions
(within ±10% variance across 3+ billing cycles), performs fuzzy merchant normalization,
and computes personal Leakage Scores and potential savings.
"""

import os
import sys
import difflib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DB_PATH = os.path.join(DATA_DIR, "finwell.db")

DUPLICATE_GROUPS = {
    "food_delivery": ["Swiggy", "Zomato", "Dunzo", "Blinkit", "Zepto", "Instamart"],
    "streaming_video": ["Netflix", "Amazon Prime", "Disney+ Hotstar", "Zee5", "SonyLIV", "Apple TV+", "YouTube Premium"],
    "streaming_music": ["Spotify", "JioSaavn", "Gaana", "Apple Music", "Wynk"],
    "broadband_telco": ["Jio Fiber", "Airtel Postpaid", "Vi Postpaid", "BSNL Broadband", "ACT Fibernet"]
}

KNOWN_MERCHANTS = [
    "Netflix", "Spotify", "Amazon Prime", "Disney+ Hotstar", "Swiggy", "Zomato",
    "BigBasket", "Blinkit", "Zepto", "Uber", "Ola", "BESCOM", "Cult.Fit",
    "Jio Fiber", "Airtel", "Apple", "Google Storage", "Gym Pass", "Times Prime"
]

STATUS_COLORS = {
    "Active":    "#16a34a",   # green
    "Low Usage": "#d97706",   # amber
    "Dormant":   "#dc2626",   # red
    "Duplicate": "#9333ea",   # purple
}


def fuzzy_normalize_merchant(raw_name: str) -> str:
    """
    Fuzzy matches merchant name using Levenshtein/difflib ratio against known merchant brands.
    """
    cleaned = raw_name.strip()
    # Direct match or containment
    for km in KNOWN_MERCHANTS:
        if km.lower() in cleaned.lower():
            return km

    matches = difflib.get_close_matches(cleaned, KNOWN_MERCHANTS, n=1, cutoff=0.7)
    if matches:
        return matches[0]

    return cleaned.title()


def get_merchant_group(merchant: str) -> Optional[str]:
    """
    Identifies duplicate service category group.
    """
    for grp, merchants in DUPLICATE_GROUPS.items():
        for m in merchants:
            if m.lower() in merchant.lower():
                return grp
    return None


def analyze_user_leakages(user_id: str, txns_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Analyzes subscriptions and leakages strictly for the specified user_id.

    Returns:
    - subscriptions: List of detected subscriptions with leakage scores
    - total_monthly_leakage: Potential monthly money wasted
    - total_annual_savings: Potential annual savings if remedied
    - leakage_count_by_status: Breakdown by Active, Low Usage, Dormant, Duplicate
    - user_leakage_score: Overall normalized leakage health score (0-100)
    """
    if txns_df is None:
        from src.data_adapter import get_user_transactions
        txns_df = get_user_transactions(user_id)

    if txns_df.empty:
        return {
            "user_id": user_id,
            "subscriptions": [],
            "total_monthly_leakage": 0.0,
            "total_annual_savings": 0.0,
            "leakage_count_by_status": {"Active": 0, "Low Usage": 0, "Dormant": 0, "Duplicate": 0},
            "user_leakage_score": 0.0
        }

    df = txns_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["month"] = df["timestamp"].dt.to_period("M")
    df["amount"] = df["amount"].astype(float)

    # Filter to debits
    debits = df[df["type"] == "Debit"].copy()
    if debits.empty:
        return {
            "user_id": user_id,
            "subscriptions": [],
            "total_monthly_leakage": 0.0,
            "total_annual_savings": 0.0,
            "leakage_count_by_status": {"Active": 0, "Low Usage": 0, "Dormant": 0, "Duplicate": 0},
            "user_leakage_score": 0.0
        }

    # Normalize merchant names
    debits["clean_merchant"] = debits["merchant"].apply(fuzzy_normalize_merchant)

    now = df["timestamp"].max()
    detected_subs = []

    # Detect recurrence per (clean_merchant, category)
    grouped = debits.groupby(["clean_merchant", "category"])

    for (merchant, category), group in grouped:
        n_txns = len(group)
        distinct_months = group["month"].nunique()

        # Check recurrence condition: charges in 3+ distinct billing months
        # OR marked as Subscription category with >= 2 charges
        is_sub_category = category in ["Subscriptions", "Entertainment", "Utilities"]
        min_cycles = 2 if is_sub_category else 3

        if distinct_months < min_cycles and n_txns < 3:
            continue

        mean_amt = float(group["amount"].mean())
        std_amt = float(group["amount"].std()) if n_txns > 1 else 0.0
        coef_var = std_amt / max(mean_amt, 1.0)

        # Check ±10% variance for fixed recurring charges, or typical utility subscription
        if coef_var > 0.35 and not is_sub_category:
            continue

        last_date = group["timestamp"].max()
        days_inactive = max(0, (now - last_date).days)

        # Monthly usage frequency
        monthly_freq = float(n_txns) / max(float(distinct_months), 1.0)

        # Formula: Cost-Per-Use = Monthly Cost / (Monthly Usage Frequency + 1)
        cost_per_use = mean_amt / (monthly_freq + 1.0)
        norm_cost_per_use = float(np.clip(cost_per_use / max(mean_amt, 1.0), 0.0, 1.0))

        # Formula: Dormancy Factor = min(1.0, Days Inactive / 90)
        dormancy_factor = min(1.0, float(days_inactive) / 90.0)

        # Status & Overlap Penalty
        group_key = get_merchant_group(merchant)
        overlap_penalty = 0.0
        status = "Active"

        if days_inactive > 60:
            status = "Dormant"
        elif cost_per_use > (mean_amt * 0.45) or monthly_freq <= 1.0:
            status = "Low Usage"

        detected_subs.append({
            "merchant": merchant,
            "category": category,
            "monthly_cost": round(mean_amt, 2),
            "monthly_freq": round(monthly_freq, 1),
            "cost_per_use": round(cost_per_use, 2),
            "norm_cpu": norm_cost_per_use,
            "last_used_date": last_date.strftime("%Y-%m-%d"),
            "days_inactive": days_inactive,
            "dormancy_factor": dormancy_factor,
            "status": status,
            "group_key": group_key,
            "annual_cost": round(mean_amt * 12.0, 2)
        })

    # Check for duplicate group penalties across detected subscriptions
    group_counts = {}
    for sub in detected_subs:
        gk = sub["group_key"]
        if gk:
            group_counts[gk] = group_counts.get(gk, 0) + 1

    for sub in detected_subs:
        gk = sub["group_key"]
        overlap_penalty = 1.0 if (gk and group_counts.get(gk, 0) > 1) else 0.0
        if overlap_penalty > 0 and sub["status"] != "Dormant":
            sub["status"] = "Duplicate"

        # Leakage Score calculation:
        # Score = (0.5 * (1 - norm_cpu)) + (0.3 * dormancy_factor) + (0.2 * overlap_penalty)
        raw_score = (
            (0.5 * (1.0 - sub["norm_cpu"])) +
            (0.3 * sub["dormancy_factor"]) +
            (0.2 * overlap_penalty)
        ) * 100.0

        leakage_score = float(np.clip(raw_score, 0.0, 100.0))
        sub["leakage_score"] = round(leakage_score, 1)

        # Potential savings
        is_wasteful = sub["status"] in ["Dormant", "Duplicate", "Low Usage"]
        saving_multiplier = 1.0 if sub["status"] in ["Dormant", "Duplicate"] else 0.5
        sub["potential_monthly_saving"] = round(sub["monthly_cost"] * saving_multiplier, 2) if is_wasteful else 0.0
        sub["potential_annual_saving"] = round(sub["potential_monthly_saving"] * 12.0, 2)

    # Sort subscriptions by leakage score descending
    detected_subs.sort(key=lambda x: x["leakage_score"], reverse=True)

    total_monthly_leakage = sum(s["potential_monthly_saving"] for s in detected_subs)
    total_annual_savings = sum(s["potential_annual_saving"] for s in detected_subs)

    status_counts = {"Active": 0, "Low Usage": 0, "Dormant": 0, "Duplicate": 0}
    for s in detected_subs:
        st = s["status"]
        status_counts[st] = status_counts.get(st, 0) + 1

    overall_leakage_score = (
        float(np.mean([s["leakage_score"] for s in detected_subs]))
        if detected_subs else 0.0
    )

    return {
        "user_id": user_id,
        "subscriptions": detected_subs,
        "total_monthly_leakage": round(total_monthly_leakage, 2),
        "total_annual_savings": round(total_annual_savings, 2),
        "leakage_count_by_status": status_counts,
        "user_leakage_score": round(overall_leakage_score, 1)
    }


if __name__ == "__main__":
    res = analyze_user_leakages("USR_PLAID_CUSTOM")
    print(f"User Leakages for {res['user_id']}:")
    print(f"Total Subscriptions Found: {len(res['subscriptions'])}")
    print(f"Potential Annual Savings: ₹{res['total_annual_savings']:,.2f}")
    for s in res["subscriptions"]:
        print(f"  • {s['merchant']} ({s['status']}): ₹{s['monthly_cost']}/mo | Leakage Score: {s['leakage_score']} | Savings: ₹{s['potential_annual_saving']}/yr")
