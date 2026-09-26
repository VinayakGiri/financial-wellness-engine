"""
clustering.py
-------------
Performs K-Means clustering on user behavioural features to segment
500 users into 5 spending personas. Saves model, cluster labels,
elbow curve, and PCA scatter plot.
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
import seaborn as sns
import joblib
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_PATH   = os.path.join(PROJECT_ROOT, "data",    "user_features.csv")
CLUSTERS_PATH   = os.path.join(PROJECT_ROOT, "data",    "user_clusters.csv")
MODEL_PATH      = os.path.join(PROJECT_ROOT, "models",  "kmeans_model.pkl")
ELBOW_PATH      = os.path.join(PROJECT_ROOT, "outputs", "elbow_curve.png")
SCATTER_PATH    = os.path.join(PROJECT_ROOT, "outputs", "personas_scatter.png")

os.makedirs(os.path.join(PROJECT_ROOT, "outputs"), exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, "models"),  exist_ok=True)

# ── Persona definitions (k=5) ─────────────────────────────────────────────────
PERSONA_MAP = {
    0: ("Budget-Conscious",   "#2196F3", "💰"),   # blue
    1: ("Impulsive Spender",  "#F44336", "🔥"),   # red
    2: ("Heavy Subscriber",   "#9C27B0", "📺"),   # purple
    3: ("Luxury Spender",     "#FF9800", "💎"),   # orange
    4: ("Weekend Shopper",    "#4CAF50", "🛍️"),   # green
}

def section(title: str):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load features
# ─────────────────────────────────────────────────────────────────────────────
section("1. LOADING FEATURE MATRIX")

feat_df  = pd.read_csv(FEATURES_PATH, index_col="user_id")
X        = feat_df.values
features = feat_df.columns.tolist()

print(f"  Loaded  : {X.shape[0]} users  x  {X.shape[1]} features")
print(f"  Value range: [{X.min():.4f}, {X.max():.4f}]")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Elbow Method — test k = 2 … 8
# ─────────────────────────────────────────────────────────────────────────────
section("2. ELBOW METHOD  (k = 2 to 8)")

k_range    = range(2, 9)
inertias   = []
sil_scores = []

print(f"  {'k':>3}  {'Inertia':>12}  {'Silhouette':>12}")
print(f"  {'--':>3}  {'-------':>12}  {'----------':>12}")

for k in k_range:
    km  = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    inertia = km.inertia_
    sil     = silhouette_score(X, km.labels_)
    inertias.append(inertia)
    sil_scores.append(sil)
    marker = "  <-- selected" if k == 5 else ""
    print(f"  {k:>3}  {inertia:>12.2f}  {sil:>12.4f}{marker}")

# Elbow plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Inertia
axes[0].plot(list(k_range), inertias, "bo-", linewidth=2, markersize=8)
axes[0].axvline(x=5, color="red", linestyle="--", linewidth=1.5, label="Selected k=5")
axes[0].scatter([5], [inertias[3]], color="red", s=120, zorder=5)
axes[0].set_xlabel("Number of Clusters (k)", fontsize=11)
axes[0].set_ylabel("Inertia (Within-cluster SSE)", fontsize=11)
axes[0].set_title("Elbow Curve — Inertia vs k", fontsize=12, fontweight="bold")
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].set_xticks(list(k_range))

# Silhouette
axes[1].plot(list(k_range), sil_scores, "gs-", linewidth=2, markersize=8)
axes[1].axvline(x=5, color="red", linestyle="--", linewidth=1.5, label="Selected k=5")
axes[1].scatter([5], [sil_scores[3]], color="red", s=120, zorder=5)
axes[1].set_xlabel("Number of Clusters (k)", fontsize=11)
axes[1].set_ylabel("Silhouette Score", fontsize=11)
axes[1].set_title("Silhouette Score vs k", fontsize=12, fontweight="bold")
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)
axes[1].set_xticks(list(k_range))

plt.suptitle("K-Means Optimal Cluster Selection — Financial Wellness Engine",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(ELBOW_PATH, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Elbow curve saved -> {ELBOW_PATH}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Fit K-Means with k=5
# ─────────────────────────────────────────────────────────────────────────────
section("3. FITTING K-MEANS  (k=5, random_state=42)")

kmeans = KMeans(n_clusters=5, random_state=42, n_init=20, max_iter=500)
kmeans.fit(X)

labels       = kmeans.labels_
centers      = pd.DataFrame(kmeans.cluster_centers_, columns=features)
final_inertia = kmeans.inertia_
final_sil     = silhouette_score(X, labels)

print(f"  Inertia        : {final_inertia:.2f}")
print(f"  Silhouette     : {final_sil:.4f}")
print(f"  Iterations     : {kmeans.n_iter_}")
print(f"  Cluster counts : {dict(zip(*np.unique(labels, return_counts=True)))}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Assign clusters & save
# ─────────────────────────────────────────────────────────────────────────────
section("4. ASSIGNING CLUSTERS TO USERS")

cluster_df = pd.DataFrame({
    "user_id":       feat_df.index,
    "cluster_label": labels,
    "persona_name":  [PERSONA_MAP[l][0] for l in labels],
})
cluster_df.to_csv(CLUSTERS_PATH, index=False)
print(f"  Saved -> {CLUSTERS_PATH}")
print(f"\n  Sample assignments:")
print(cluster_df.head(10).to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# 5 & 6. Persona profiles — mean features + interpretation
# ─────────────────────────────────────────────────────────────────────────────
section("5 & 6. PERSONA PROFILES")

# Key features to highlight per persona
PERSONA_HIGHLIGHTS = {
    0: ["savings_ratio", "avg_monthly_spend", "total_spending", "discretionary_ratio"],
    1: ["spending_volatility", "weekend_spending_ratio", "impulse_transaction_count", "avg_transaction_amount"],
    2: ["recurring_payment_count", "subscription_total_monthly", "subscription_to_income_ratio", "unique_merchants_count"],
    3: ["avg_transaction_amount", "total_spending", "max_monthly_spend", "discretionary_ratio"],
    4: ["weekend_spending_ratio", "weekday_spending_ratio", "entertainment_ratio", "impulse_transaction_count"],
}

INTERPRETATIONS = {
    0: "Careful spenders with high savings rates. Low discretionary spend, consistent "
       "monthly budgets. Prefer essentials and rarely splurge. Good candidates for "
       "investment & savings product recommendations.",
    1: "High spending volatility with large single transactions. Disproportionate "
       "weekend spending. Frequent impulse buys. Candidates for spend-alert features "
       "and budgeting nudges.",
    2: "Multiple recurring subscriptions with high subscription-to-income ratio. "
       "Often hold overlapping services (Netflix + Prime + Hotstar). Good targets for "
       "subscription audit and savings alerts.",
    3: "High average transaction amounts and peak monthly spend. Prefer premium "
       "merchants and high-value single purchases. Candidates for premium card "
       "and cashback on luxury categories.",
    4: "Significantly higher spend on Fri-Sun vs weekdays. Strong entertainment and "
       "shopping category presence. Impulse-driven but patterns are predictable. "
       "Good targets for weekend cashback offers.",
}

feat_with_labels = feat_df.copy()
feat_with_labels["cluster_label"] = labels

for cluster_id, (name, color, emoji) in PERSONA_MAP.items():
    mask      = labels == cluster_id
    count     = mask.sum()
    pct       = count / len(labels) * 100
    highlights = PERSONA_HIGHLIGHTS[cluster_id]

    print(f"\n  {'─'*60}")
    print(f"  {emoji}  Cluster {cluster_id}  |  {name}  |  {count} users ({pct:.1f}%)")
    print(f"  {'─'*60}")

    cluster_means = feat_with_labels[mask][features].mean()
    overall_means = feat_df[features].mean()

    print(f"  Key Feature Means (vs. overall avg):")
    for feat in highlights:
        val     = cluster_means[feat]
        overall = overall_means[feat]
        diff    = val - overall
        arrow   = "▲" if diff > 0.05 else ("▼" if diff < -0.05 else "~")
        print(f"    {feat:<35}  {val:.3f}  (overall: {overall:.3f})  {arrow}")

    interp = INTERPRETATIONS[cluster_id]
    print(f"\n  Interpretation:")
    # word-wrap at 58 chars
    words = interp.split()
    line  = "    "
    for w in words:
        if len(line) + len(w) + 1 > 62:
            print(line)
            line = "    " + w + " "
        else:
            line += w + " "
    if line.strip():
        print(line)

# ─────────────────────────────────────────────────────────────────────────────
# 7. PCA Scatter Plot
# ─────────────────────────────────────────────────────────────────────────────
section("7. PCA SCATTER PLOT  (2 components)")

pca       = PCA(n_components=2, random_state=42)
X_pca     = pca.fit_transform(X)
var_ratio = pca.explained_variance_ratio_

print(f"  PCA variance explained:")
print(f"    PC1: {var_ratio[0]*100:.1f}%")
print(f"    PC2: {var_ratio[1]*100:.1f}%")
print(f"    Total: {sum(var_ratio)*100:.1f}%")

fig, ax = plt.subplots(figsize=(11, 8))
ax.set_facecolor("#f8f9fa")
fig.patch.set_facecolor("#f8f9fa")

# Scatter per cluster
for cid, (name, color, emoji) in PERSONA_MAP.items():
    mask = labels == cid
    ax.scatter(
        X_pca[mask, 0], X_pca[mask, 1],
        c=color, label=f"{emoji} {name} (n={mask.sum()})",
        alpha=0.65, s=55, edgecolors="white", linewidths=0.4,
    )

# Annotate cluster centers in PCA space
centers_pca = pca.transform(kmeans.cluster_centers_)
for cid, (name, color, emoji) in PERSONA_MAP.items():
    cx, cy = centers_pca[cid]
    ax.scatter(cx, cy, c=color, s=280, marker="*",
               edgecolors="black", linewidths=0.8, zorder=10)
    ax.annotate(
        f"{emoji} {name}",
        xy=(cx, cy),
        xytext=(cx + 0.08, cy + 0.05),
        fontsize=8.5, fontweight="bold", color=color,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.85),
        arrowprops=dict(arrowstyle="-", color=color, lw=0.8),
    )

ax.set_xlabel(f"PC1  ({var_ratio[0]*100:.1f}% variance)", fontsize=11)
ax.set_ylabel(f"PC2  ({var_ratio[1]*100:.1f}% variance)", fontsize=11)
ax.set_title(
    "User Spending Personas — K-Means Clusters (PCA Projection)\n"
    "Financial Wellness Engine  |  k=5  |  500 Users",
    fontsize=13, fontweight="bold", pad=14,
)
ax.legend(
    title="Spending Persona",
    title_fontsize=10,
    fontsize=9,
    loc="lower right",
    framealpha=0.9,
    edgecolor="#cccccc",
)
ax.grid(True, alpha=0.25, linestyle="--")
plt.tight_layout()
plt.savefig(SCATTER_PATH, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Scatter plot saved -> {SCATTER_PATH}")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Save model
# ─────────────────────────────────────────────────────────────────────────────
section("8. SAVING K-MEANS MODEL")

joblib.dump(kmeans, MODEL_PATH)
print(f"  Model saved -> {MODEL_PATH}")
loaded_test = joblib.load(MODEL_PATH)
print(f"  Model load verification: n_clusters={loaded_test.n_clusters}  ✓")

# ─────────────────────────────────────────────────────────────────────────────
# Final Summary Table
# ─────────────────────────────────────────────────────────────────────────────
section("CLUSTER DISTRIBUTION SUMMARY")

print(f"\n  {'Cluster':<3}  {'Persona':<22}  {'Users':>6}  {'%':>6}  {'Avg Spend':>12}  {'Avg Savings':>12}")
print(f"  {'─'*3}  {'─'*22}  {'─'*6}  {'─'*6}  {'─'*12}  {'─'*12}")
for cid, (name, color, emoji) in PERSONA_MAP.items():
    mask   = labels == cid
    count  = mask.sum()
    pct    = count / len(labels) * 100
    # un-normalized avg monthly spend & savings ratio — use raw feat_df values
    avg_sp  = feat_df.loc[feat_df.index[mask], "avg_monthly_spend"].mean()
    avg_sav = feat_df.loc[feat_df.index[mask], "savings_ratio"].mean()
    print(f"  {cid:<3}  {emoji} {name:<20}  {count:>6}  {pct:>5.1f}%  "
          f"{avg_sp:>12.3f}  {avg_sav:>12.3f}")

print(f"""
  ╔══════════════════════════════════════════════════════════════╗
  ║                                                              ║
  ║   Persona Segmentation Complete — 5 clusters identified      ║
  ║                                                              ║
  ║   data/user_clusters.csv       -> cluster assignments        ║
  ║   models/kmeans_model.pkl      -> trained KMeans model       ║
  ║   outputs/elbow_curve.png      -> elbow + silhouette plot    ║
  ║   outputs/personas_scatter.png -> PCA cluster scatter        ║
  ║                                                              ║
  ╚══════════════════════════════════════════════════════════════╝
""")
