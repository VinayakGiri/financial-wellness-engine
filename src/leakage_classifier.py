"""
leakage_classifier.py
---------------------
Trains a Random Forest classifier to predict which recurring
subscriptions are most likely to be forgotten / wasteful.

Outputs
-------
models/leakage_classifier.pkl     – trained RandomForest model
models/leakage_scaler.pkl         – fitted StandardScaler
outputs/leakage_cm.png            – confusion matrix heatmap
outputs/feature_importance.png    – feature importance bar chart
outputs/leakage_features.json     – top-5 feature importances
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
    roc_auc_score,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECURRING_PATH     = os.path.join(PROJECT_ROOT, "data",    "recurring_transactions.csv")
LEAKAGE_PATH       = os.path.join(PROJECT_ROOT, "data",    "user_leakage.csv")
TRANSACTIONS_PATH  = os.path.join(PROJECT_ROOT, "data",    "transactions_synthetic.csv")
MODEL_PATH         = os.path.join(PROJECT_ROOT, "models",  "leakage_classifier.pkl")
SCALER_PATH        = os.path.join(PROJECT_ROOT, "models",  "leakage_scaler.pkl")
CM_PATH            = os.path.join(PROJECT_ROOT, "outputs", "leakage_cm.png")
FI_PATH            = os.path.join(PROJECT_ROOT, "outputs", "feature_importance.png")
FI_JSON_PATH       = os.path.join(PROJECT_ROOT, "outputs", "leakage_features.json")

os.makedirs(os.path.join(PROJECT_ROOT, "models"),  exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, "outputs"), exist_ok=True)

LEAKAGE_THRESHOLD = 60   # leakage_score > 60 → high leakage (label = 1)
ESSENTIAL_CATS    = {"Utilities", "Medical"}

def section(title: str):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────────────────────────────────────────
section("1. LOADING DATA")

recurring = pd.read_csv(RECURRING_PATH, parse_dates=["last_date"])
leakage   = pd.read_csv(LEAKAGE_PATH)
txn_df    = pd.read_csv(TRANSACTIONS_PATH, parse_dates=["date"])

print(f"  Recurring transactions : {len(recurring):,} rows")
print(f"  Leakage records        : {len(leakage):,} rows")
print(f"  Raw transactions       : {len(txn_df):,} rows")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Build feature matrix
# ─────────────────────────────────────────────────────────────────────────────
section("2. BUILDING FEATURE MATRIX")

# Per-user avg monthly spend from raw transactions
user_monthly = (
    txn_df[txn_df["transaction_type"] == "Debit"]
    .assign(month=lambda d: d["date"].dt.to_period("M"))
    .groupby(["user_id", "month"])["amount"].sum()
    .groupby("user_id").mean()
    .rename("user_avg_monthly_spend")
    .reset_index()
)

# Merge leakage score into recurring on (user_id, merchant_name)
merged = recurring.merge(
    leakage[["user_id", "merchant_name", "leakage_score",
             "cost_per_use", "dormancy_days"]],
    on=["user_id", "merchant_name"],
    how="inner",
)
merged = merged.merge(user_monthly, on="user_id", how="left")

# frequency_per_month = frequency_months / 12 (proportion of months active)
REFERENCE_DATE = txn_df["date"].max()
merged["frequency_per_month"] = (merged["frequency_months"] / 12).clip(0, 1)

# days_since_last_use (same as dormancy_days, but computed fresh if missing)
if "dormancy_days" in merged.columns:
    merged["days_since_last_use"] = merged["dormancy_days"]
else:
    merged["last_date"] = pd.to_datetime(merged["last_date"])
    merged["days_since_last_use"] = (
        pd.Timestamp(REFERENCE_DATE) - merged["last_date"]
    ).dt.days

# category_essential_flag
merged["category_essential_flag"] = (
    merged["category"].isin(ESSENTIAL_CATS).astype(int)
)

# Fill any NaN in user_avg_monthly_spend
merged["user_avg_monthly_spend"].fillna(
    merged["user_avg_monthly_spend"].median(), inplace=True
)

# Feature columns
FEATURE_COLS = [
    "monthly_amount",
    "cost_per_use",
    "frequency_per_month",
    "days_since_last_use",
    "user_avg_monthly_spend",
    "category_essential_flag",
]

# Binary target
merged["leakage_label"] = (merged["leakage_score"] > LEAKAGE_THRESHOLD).astype(int)

df_model = merged[FEATURE_COLS + ["leakage_label"]].dropna()

X = df_model[FEATURE_COLS].values
y = df_model["leakage_label"].values

print(f"  Feature matrix shape : {X.shape}")
print(f"  Class distribution   : 0={int((y==0).sum())}  1={int((y==1).sum())}  "
      f"(ratio={int((y==1).sum())/len(y)*100:.1f}% positive)")
print(f"  Features used        : {FEATURE_COLS}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Target encoding summary
# ─────────────────────────────────────────────────────────────────────────────
section("3. FEATURE SUMMARY (pre-split)")

print(f"\n  {'Feature':<28}  {'Min':>9}  {'Max':>9}  {'Mean':>9}  {'Std':>9}")
print(f"  {'─'*28}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*9}")
for i, col in enumerate(FEATURE_COLS):
    vals = X[:, i]
    print(f"  {col:<28}  {vals.min():>9.2f}  {vals.max():>9.2f}  "
          f"{vals.mean():>9.2f}  {vals.std():>9.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Train / test split (stratified)
# ─────────────────────────────────────────────────────────────────────────────
section("4. TRAIN / TEST SPLIT  (80 / 20, stratified)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Scale features
scaler  = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"  Train size : {X_train_sc.shape[0]}  (pos={int(y_train.sum())})")
print(f"  Test  size : {X_test_sc.shape[0]}   (pos={int(y_test.sum())})")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Train Random Forest
# ─────────────────────────────────────────────────────────────────────────────
section("5. TRAINING RANDOM FOREST  (n=100, depth=10, seed=42)")

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight="balanced",   # handle class imbalance
    n_jobs=-1,
)
rf.fit(X_train_sc, y_train)
print(f"  Training complete — {rf.n_estimators} trees fitted")

# Cross-validation on training set
cv_scores = cross_val_score(rf, X_train_sc, y_train, cv=5, scoring="f1")
print(f"  5-fold CV F1 on train : {cv_scores.mean():.4f}  (±{cv_scores.std():.4f})")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Evaluate on test set
# ─────────────────────────────────────────────────────────────────────────────
section("6. EVALUATION ON TEST SET")

y_pred      = rf.predict(X_test_sc)
y_pred_prob = rf.predict_proba(X_test_sc)[:, 1]

acc  = accuracy_score(y_test,  y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec  = recall_score(y_test,    y_pred, zero_division=0)
f1   = f1_score(y_test,        y_pred, zero_division=0)
try:
    roc = roc_auc_score(y_test, y_pred_prob)
except ValueError:
    roc = float("nan")

print(f"\n  Accuracy   : {acc*100:.2f}%")
print(f"  Precision  : {prec*100:.2f}%")
print(f"  Recall     : {rec*100:.2f}%")
print(f"  F1 Score   : {f1*100:.2f}%")
print(f"  ROC-AUC    : {roc:.4f}")

print(f"\n  Classification Report:\n")
print(classification_report(
    y_test, y_pred,
    target_names=["Normal (0)", "High Leakage (1)"],
    zero_division=0,
))

# Confusion Matrix plot
cm = confusion_matrix(y_test, y_pred)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
fig.patch.set_facecolor("#f8f9fa")

# Raw counts
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Normal", "High Leakage"],
    yticklabels=["Normal", "High Leakage"],
    linewidths=1, linecolor="white",
    ax=axes[0], cbar=False,
    annot_kws={"size": 14, "weight": "bold"},
)
axes[0].set_xlabel("Predicted Label", fontsize=11)
axes[0].set_ylabel("True Label",      fontsize=11)
axes[0].set_title("Confusion Matrix — Counts", fontsize=12, fontweight="bold")
axes[0].set_facecolor("#f8f9fa")

# Normalised (%)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
sns.heatmap(
    cm_norm, annot=True, fmt=".1f", cmap="Greens",
    xticklabels=["Normal", "High Leakage"],
    yticklabels=["Normal", "High Leakage"],
    linewidths=1, linecolor="white",
    ax=axes[1], cbar=False,
    annot_kws={"size": 13},
)
axes[1].set_xlabel("Predicted Label", fontsize=11)
axes[1].set_ylabel("True Label",      fontsize=11)
axes[1].set_title("Confusion Matrix — Row % (Recall)", fontsize=12, fontweight="bold")
axes[1].set_facecolor("#f8f9fa")

plt.suptitle(
    f"Random Forest Leakage Classifier  |  Accuracy={acc*100:.1f}%  "
    f"F1={f1*100:.1f}%  ROC-AUC={roc:.3f}",
    fontsize=12, fontweight="bold", y=1.02,
)
plt.tight_layout()
plt.savefig(CM_PATH, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Confusion matrix saved -> {CM_PATH}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Feature importances
# ─────────────────────────────────────────────────────────────────────────────
section("7. FEATURE IMPORTANCES")

importances = rf.feature_importances_
fi_df = pd.DataFrame({
    "feature":    FEATURE_COLS,
    "importance": importances,
}).sort_values("importance", ascending=False).reset_index(drop=True)

print(f"\n  {'Rank':<5}  {'Feature':<28}  {'Importance':>12}  Bar")
print(f"  {'─'*5}  {'─'*28}  {'─'*12}  {'─'*25}")
for _, row in fi_df.iterrows():
    bar = "█" * int(row["importance"] * 100)
    print(f"  {_+1:<5}  {row['feature']:<28}  {row['importance']:>12.4f}  {bar}")

# Save top-5 to JSON
top5 = fi_df.head(5)
top5_dict = top5.set_index("feature")["importance"].round(6).to_dict()
with open(FI_JSON_PATH, "w") as f:
    json.dump(top5_dict, f, indent=2)
print(f"\n  Top-5 features saved -> {FI_JSON_PATH}")

# Feature importance bar chart
palette = sns.color_palette("viridis", len(fi_df))

fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor("#f8f9fa")
ax.set_facecolor("#f8f9fa")

bars = ax.barh(
    fi_df["feature"][::-1],
    fi_df["importance"][::-1],
    color=palette,
    edgecolor="white",
    height=0.65,
)
for bar, val in zip(bars, fi_df["importance"][::-1]):
    ax.text(
        bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
        f"{val:.4f}", va="center", ha="left", fontsize=9,
    )

ax.set_xlabel("Gini Importance", fontsize=11)
ax.set_title(
    "Feature Importances — Leakage Classifier (Random Forest)\n"
    "Financial Wellness Engine",
    fontsize=12, fontweight="bold", pad=12,
)
ax.set_xlim(0, fi_df["importance"].max() * 1.18)
ax.grid(axis="x", alpha=0.3, linestyle="--")
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig(FI_PATH, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Feature importance chart saved -> {FI_PATH}")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Save models
# ─────────────────────────────────────────────────────────────────────────────
section("8. SAVING MODELS")

joblib.dump(rf,     MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
print(f"  Classifier  saved -> {MODEL_PATH}")
print(f"  Scaler      saved -> {SCALER_PATH}")

# Verify by reloading
rf_loaded  = joblib.load(MODEL_PATH)
sc_loaded  = joblib.load(SCALER_PATH)
y_verify   = rf_loaded.predict(sc_loaded.transform(X_test))
assert (y_verify == y_pred).all(), "Loaded model predictions differ!"
print(f"  Reload verification : PASSED  ✓")

# ─────────────────────────────────────────────────────────────────────────────
# Final
# ─────────────────────────────────────────────────────────────────────────────
section("SUMMARY")

print(f"""
  ┌─────────────────────────────────────────────────────────┐
  │  Model          : Random Forest (n=100, depth=10)        │
  │  Features       : {len(FEATURE_COLS)} engineered features                │
  │  Train samples  : {X_train_sc.shape[0]}                                    │
  │  Test  samples  : {X_test_sc.shape[0]}                                     │
  ├─────────────────────────────────────────────────────────┤
  │  Accuracy       : {acc*100:>6.2f}%                                │
  │  Precision      : {prec*100:>6.2f}%                                │
  │  Recall         : {rec*100:>6.2f}%                                │
  │  F1 Score       : {f1*100:>6.2f}%                                │
  │  ROC-AUC        : {roc:>6.4f}                                │
  ├─────────────────────────────────────────────────────────┤
  │  Top Feature    : {fi_df.iloc[0]['feature']:<38} │
  │  Importance     : {fi_df.iloc[0]['importance']:>6.4f}                                │
  └─────────────────────────────────────────────────────────┘
""")

print(f"  Leakage Classification Model Complete — "
      f"Accuracy: {acc*100:.2f}%")
