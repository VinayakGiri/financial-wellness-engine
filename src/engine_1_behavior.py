"""
engine_1_behavior.py
--------------------
Engine 1: Personal Spending Persona & Cohort Benchmarking for FinWell.

Calculates the 20 behavioral features for an individual user, assigns their
K-Means persona (k=5), projects their coordinates onto the 2D PCA space for
a "You Are Here" visualization, and computes benchmark metrics against their cluster.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DB_PATH = os.path.join(DATA_DIR, "finwell.db")
KMEANS_MODEL_PATH = os.path.join(MODELS_DIR, "kmeans_model.pkl")
USER_FEATURES_PATH = os.path.join(DATA_DIR, "user_features.csv")
USER_CLUSTERS_PATH = os.path.join(DATA_DIR, "user_clusters.csv")

FEATURE_COLS = [
    "total_spending", "avg_monthly_spend", "std_monthly_spend", "max_monthly_spend",
    "food_ratio", "entertainment_ratio", "essential_ratio", "discretionary_ratio",
    "savings_ratio", "weekend_spending_ratio", "weekday_spending_ratio",
    "salary_day_behavior", "spending_volatility", "recurring_payment_count",
    "subscription_total_monthly", "unique_merchants_count",
    "subscription_to_income_ratio", "impulse_transaction_count",
    "avg_transaction_amount", "transaction_frequency"
]

PERSONA_META = {
    0: {"name": "Budget-Conscious",  "emoji": "💰", "color": "#2196F3", "desc": "Low spending, high savings ratio, and disciplined budgeting habits."},
    1: {"name": "Impulsive Spender", "emoji": "🔥", "color": "#F44336", "desc": "High spend volatility, frequent weekend splurges, and elevated impulse buys."},
    2: {"name": "Heavy Subscriber",  "emoji": "📺", "color": "#9C27B0", "desc": "High recurring subscription count and digital services commitments."},
    3: {"name": "Luxury Spender",    "emoji": "💎", "color": "#FF9800", "desc": "High average ticket sizes and premium merchant preferences."},
    4: {"name": "Weekend Shopper",   "emoji": "🛍️", "color": "#4CAF50", "desc": "Disproportionate spending concentrated between Friday and Sunday."},
}

# Module level cached cohort data
_cohort_features_df = None
_cohort_clusters_df = None
_kmeans_model = None
_pca_model = None
_pca_3d_model = None
_cohort_pca_df = None
_cohort_pca_3d_df = None
_hierarchical_linkage = None


def load_cohort_models():
    """
    Loads pretrained K-Means model, 2D and 3D PCA models, and Hierarchical Clustering linkage.
    """
    global _cohort_features_df, _cohort_clusters_df, _kmeans_model, _pca_model, _pca_3d_model, _cohort_pca_df, _cohort_pca_3d_df, _hierarchical_linkage

    if _cohort_features_df is None and os.path.exists(USER_FEATURES_PATH):
        _cohort_features_df = pd.read_csv(USER_FEATURES_PATH, index_col="user_id")

    if _cohort_clusters_df is None and os.path.exists(USER_CLUSTERS_PATH):
        _cohort_clusters_df = pd.read_csv(USER_CLUSTERS_PATH, index_col="user_id")

    if _kmeans_model is None and os.path.exists(KMEANS_MODEL_PATH):
        _kmeans_model = joblib.load(KMEANS_MODEL_PATH)

    if _pca_model is None and _cohort_features_df is not None:
        # 2D PCA Model (Section 4.4.3)
        _pca_model = PCA(n_components=2, random_state=42)
        X_pca = _pca_model.fit_transform(_cohort_features_df.values)
        _cohort_pca_df = pd.DataFrame(
            X_pca,
            index=_cohort_features_df.index,
            columns=["PCA1", "PCA2"]
        )
        if _cohort_clusters_df is not None:
            _cohort_pca_df["cluster_label"] = _cohort_clusters_df["cluster_label"]
            _cohort_pca_df["persona_name"] = _cohort_clusters_df["persona_name"]

    if _pca_3d_model is None and _cohort_features_df is not None:
        # 3D PCA Model (Section 4.4.4)
        _pca_3d_model = PCA(n_components=3, random_state=42)
        X_pca_3d = _pca_3d_model.fit_transform(_cohort_features_df.values)
        _cohort_pca_3d_df = pd.DataFrame(
            X_pca_3d,
            index=_cohort_features_df.index,
            columns=["PCA1", "PCA2", "PCA3"]
        )
        if _cohort_clusters_df is not None:
            _cohort_pca_3d_df["cluster_label"] = _cohort_clusters_df["cluster_label"]
            _cohort_pca_3d_df["persona_name"] = _cohort_clusters_df["persona_name"]

    if _hierarchical_linkage is None and _cohort_features_df is not None:
        # Hierarchical Clustering: Ward's Linkage (Section 4.2.2)
        from scipy.cluster.hierarchy import linkage
        _hierarchical_linkage = linkage(_cohort_features_df.values, method="ward")


def compute_user_behavioral_features(user_id: str, txns_df: Optional[pd.DataFrame] = None) -> Dict[str, float]:
    """
    Calculates the 20 behavioral features for a user from transactions,
    or retrieves their pre-calculated vector if present in cohort features.
    """
    load_cohort_models()

    # Check if user already exists in baseline features
    if _cohort_features_df is not None and user_id in _cohort_features_df.index:
        return _cohort_features_df.loc[user_id].to_dict()

    # Otherwise compute dynamically from transactions
    if txns_df is None:
        from src.data_adapter import get_user_transactions
        txns_df = get_user_transactions(user_id)

    if txns_df.empty:
        # Default fallback feature vector
        return {col: 0.5 for col in FEATURE_COLS}

    df = txns_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["amount"] = df["amount"].astype(float)
    df["weekday"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.to_period("M")
    df["day"] = df["timestamp"].dt.day

    debits = df[df["type"] == "Debit"]
    credits = df[df["type"] == "Credit"]

    total_spending = float(debits["amount"].sum()) if not debits.empty else 0.0
    monthly_spends = debits.groupby("month")["amount"].sum() if not debits.empty else pd.Series([0.0])
    avg_monthly_spend = float(monthly_spends.mean())
    std_monthly_spend = float(monthly_spends.std()) if len(monthly_spends) > 1 else 0.0
    max_monthly_spend = float(monthly_spends.max())

    # Ratios
    food_spend = float(debits[debits["category"] == "Food"]["amount"].sum()) if not debits.empty else 0.0
    ent_spend = float(debits[debits["category"] == "Entertainment"]["amount"].sum()) if not debits.empty else 0.0
    ess_spend = float(debits[debits["category"].isin(["Utilities", "Medical", "Transport"])]["amount"].sum()) if not debits.empty else 0.0

    food_ratio = food_spend / max(total_spending, 1.0)
    entertainment_ratio = ent_spend / max(total_spending, 1.0)
    essential_ratio = ess_spend / max(total_spending, 1.0)
    discretionary_ratio = (food_spend + ent_spend) / max(total_spending, 1.0)

    total_income = float(credits["amount"].sum()) if not credits.empty else (total_spending * 1.5)
    savings_ratio = max(0.0, (total_income - total_spending) / max(total_income, 1.0))

    # Temporal
    weekend_debits = debits[debits["weekday"] >= 4]["amount"].sum() if not debits.empty else 0.0
    weekend_spending_ratio = weekend_debits / max(total_spending, 1.0)
    weekday_spending_ratio = 1.0 - weekend_spending_ratio

    salary_day_debits = debits[debits["day"] <= 5]["amount"].sum() if not debits.empty else 0.0
    salary_day_behavior = salary_day_debits / max(total_spending, 1.0)
    spending_volatility = std_monthly_spend / max(avg_monthly_spend, 1.0)

    # Subscriptions
    recurring_counts = debits[debits["category"] == "Subscriptions"]
    recurring_payment_count = float(len(recurring_counts["merchant"].unique()))
    subscription_total_monthly = float(recurring_counts["amount"].sum()) / max(len(monthly_spends), 1.0)
    unique_merchants_count = float(debits["merchant"].nunique()) if not debits.empty else 0.0
    subscription_to_income_ratio = subscription_total_monthly / max(total_income / 12.0, 1.0)

    # Impulse
    large_txns = debits[debits["amount"] > 2000.0] if not debits.empty else pd.DataFrame()
    impulse_transaction_count = float(len(large_txns))
    avg_transaction_amount = float(debits["amount"].mean()) if not debits.empty else 0.0
    transaction_frequency = float(len(debits)) / 12.0

    raw_features = {
        "total_spending": total_spending,
        "avg_monthly_spend": avg_monthly_spend,
        "std_monthly_spend": std_monthly_spend,
        "max_monthly_spend": max_monthly_spend,
        "food_ratio": min(1.0, food_ratio),
        "entertainment_ratio": min(1.0, entertainment_ratio),
        "essential_ratio": min(1.0, essential_ratio),
        "discretionary_ratio": min(1.0, discretionary_ratio),
        "savings_ratio": min(1.0, savings_ratio),
        "weekend_spending_ratio": min(1.0, weekend_spending_ratio),
        "weekday_spending_ratio": min(1.0, weekday_spending_ratio),
        "salary_day_behavior": min(1.0, salary_day_behavior),
        "spending_volatility": min(2.0, spending_volatility),
        "recurring_payment_count": recurring_payment_count,
        "subscription_total_monthly": subscription_total_monthly,
        "unique_merchants_count": unique_merchants_count,
        "subscription_to_income_ratio": min(1.0, subscription_to_income_ratio),
        "impulse_transaction_count": impulse_transaction_count,
        "avg_transaction_amount": avg_transaction_amount,
        "transaction_frequency": transaction_frequency,
    }

    # Normalize against cohort bounds
    if _cohort_features_df is not None:
        norm_features = {}
        for col in FEATURE_COLS:
            # Bounds from cohort or heuristics
            val = raw_features.get(col, 0.5)
            # Clip between [0, 1]
            norm_features[col] = float(np.clip(val if val <= 1.0 else val / max(val * 1.5, 10.0), 0.0, 1.0))
        return norm_features

    return raw_features


def analyze_user_persona(user_id: str, txns_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Returns full persona assignment, 2D PCA projection coordinates ("You Are Here"),
    and cluster benchmark comparisons.
    """
    load_cohort_models()
    user_features = compute_user_behavioral_features(user_id, txns_df=txns_df)
    feat_vector = np.array([user_features[c] for c in FEATURE_COLS]).reshape(1, -1)

    # 1. Assign cluster
    if _kmeans_model is not None:
        cluster_id = int(_kmeans_model.predict(feat_vector)[0])
    else:
        cluster_id = 0

    persona_info = PERSONA_META.get(cluster_id, PERSONA_META[0])

    # 2. PCA Projection (2D and 3D)
    if _pca_model is not None:
        user_pca_coords = _pca_model.transform(feat_vector)[0]
        user_x = float(user_pca_coords[0])
        user_y = float(user_pca_coords[1])
        centroids_pca = _pca_model.transform(_kmeans_model.cluster_centers_)
    else:
        user_x, user_y = 0.0, 0.0
        centroids_pca = np.zeros((5, 2))

    if _pca_3d_model is not None:
        user_pca_3d_coords = _pca_3d_model.transform(feat_vector)[0]
        user_3d_x = float(user_pca_3d_coords[0])
        user_3d_y = float(user_pca_3d_coords[1])
        user_3d_z = float(user_pca_3d_coords[2])
        centroids_pca_3d = _pca_3d_model.transform(_kmeans_model.cluster_centers_)
    else:
        user_3d_x, user_3d_y, user_3d_z = 0.0, 0.0, 0.0
        centroids_pca_3d = np.zeros((5, 3))

    # 3. Cluster Benchmarks
    cluster_means = {}
    if _cohort_features_df is not None and _cohort_clusters_df is not None:
        cluster_user_ids = _cohort_clusters_df[_cohort_clusters_df["cluster_label"] == cluster_id].index
        if len(cluster_user_ids) > 0:
            cluster_subset = _cohort_features_df.loc[_cohort_features_df.index.isin(cluster_user_ids)]
            cluster_means = cluster_subset[FEATURE_COLS].mean().to_dict()

    if not cluster_means:
        cluster_means = {c: 0.5 for c in FEATURE_COLS}

    # Key comparison metrics for radar/bar comparison
    key_metrics = [
        ("Weekend Spend Ratio", "weekend_spending_ratio"),
        ("Discretionary Ratio", "discretionary_ratio"),
        ("Savings Ratio", "savings_ratio"),
        ("Spending Volatility", "spending_volatility"),
        ("Subscription Ratio", "subscription_to_income_ratio"),
    ]

    benchmark_comparison = []
    for label, col in key_metrics:
        u_val = float(user_features.get(col, 0.0))
        c_val = float(cluster_means.get(col, 0.0))
        diff_pct = ((u_val - c_val) / max(c_val, 0.01)) * 100
        benchmark_comparison.append({
            "metric": label,
            "user_value": round(u_val, 3),
            "cluster_mean": round(c_val, 3),
            "delta_pct": round(diff_pct, 1),
        })

    return {
        "user_id": user_id,
        "cluster_id": cluster_id,
        "persona_name": persona_info["name"],
        "persona_emoji": persona_info["emoji"],
        "persona_color": persona_info["color"],
        "persona_desc": persona_info["desc"],
        "user_pca": {"x": user_x, "y": user_y},
        "user_pca_3d": {"x": user_3d_x, "y": user_3d_y, "z": user_3d_z},
        "centroids_pca": [{"cluster_id": i, "x": float(centroids_pca[i][0]), "y": float(centroids_pca[i][1])} for i in range(len(centroids_pca))],
        "centroids_pca_3d": [{"cluster_id": i, "x": float(centroids_pca_3d[i][0]), "y": float(centroids_pca_3d[i][1]), "z": float(centroids_pca_3d[i][2])} for i in range(len(centroids_pca_3d))],
        "user_features": user_features,
        "cluster_means": cluster_means,
        "benchmark_comparison": benchmark_comparison,
        "cohort_pca_df": _cohort_pca_df,
        "cohort_pca_3d_df": _cohort_pca_3d_df,
        "hierarchical_linkage": _hierarchical_linkage
    }


if __name__ == "__main__":
    load_cohort_models()
    sample = analyze_user_persona("USR_PLAID_GOOD")
    print("User Persona Result:")
    print(f"User: {sample['user_id']} -> {sample['persona_emoji']} {sample['persona_name']}")
    print(f"PCA Coordinates: ({sample['user_pca']['x']:.3f}, {sample['user_pca']['y']:.3f})")
    print("Benchmark Comparison:", sample["benchmark_comparison"])
