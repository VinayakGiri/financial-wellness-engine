"""
app.py — Financial Wellness Intelligence Platform
Streamlit dashboard integrating all 3 engines:
  Engine 1: User Segmentation (K-Means Personas)
  Engine 2: Leakage Detection (Subscription Analysis)
  Engine 3: Wellness Forecast (ARIMA + Prophet)
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.decomposition import PCA

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Financial Wellness Intelligence Platform",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT             = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR         = os.path.join(ROOT, "data")
OUTPUTS_DIR      = os.path.join(ROOT, "outputs")
MODELS_DIR       = os.path.join(ROOT, "models")

# ── Persona metadata ──────────────────────────────────────────────────────────
PERSONA_META = {
    0: {"name": "Budget-Conscious",  "emoji": "💰", "color": "#2196F3"},
    1: {"name": "Impulsive Spender", "emoji": "🔥", "color": "#F44336"},
    2: {"name": "Heavy Subscriber",  "emoji": "📺", "color": "#9C27B0"},
    3: {"name": "Luxury Spender",    "emoji": "💎", "color": "#FF9800"},
    4: {"name": "Weekend Shopper",   "emoji": "🛍️", "color": "#4CAF50"},
}

STATUS_COLORS = {
    "Active":    "#4CAF50",
    "Low Usage": "#FFC107",
    "Dormant":   "#F44336",
    "Duplicate": "#9C27B0",
}

RISK_COLORS = {
    "Low Risk":    "#4CAF50",
    "Medium Risk": "#FFC107",
    "High Risk":   "#F44336",
}

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main-title    { font-size:2.2rem; font-weight:800; color:#1565C0; margin-bottom:0; }
  .sub-title     { font-size:1.05rem; color:#546E7A; margin-top:2px; margin-bottom:1.5rem; }
  .engine-card   { background:#f0f4ff; border-left:4px solid #1976D2;
                   border-radius:8px; padding:14px 18px; margin-bottom:10px; }
  .engine-title  { font-size:1.05rem; font-weight:700; color:#1565C0; }
  .engine-desc   { font-size:0.88rem; color:#37474F; margin-top:4px; }
  .insight-box   { background:#e8f5e9; border-left:4px solid #4CAF50;
                   border-radius:8px; padding:14px 18px; margin-top:12px; }
  .warn-box      { background:#fff3e0; border-left:4px solid #FF9800;
                   border-radius:8px; padding:14px 18px; margin-top:8px; }
  .error-box     { background:#ffebee; border-left:4px solid #F44336;
                   border-radius:8px; padding:14px 18px; }
  div[data-testid="metric-container"] {
    background:#f8f9fa; border-radius:10px;
    border:1px solid #e0e0e0; padding:14px;
  }
</style>
""", unsafe_allow_html=True)


# ── Data loaders (cached) ─────────────────────────────────────────────────────
@st.cache_resource
def load_all_data():
    """Load all project data files. Returns dict with status flags."""
    data = {}

    def safe_csv(key, path, **kwargs):
        try:
            data[key] = pd.read_csv(path, **kwargs)
            data[f"{key}_ok"] = True
        except Exception as e:
            data[key] = pd.DataFrame()
            data[f"{key}_ok"] = False
            data[f"{key}_err"] = str(e)

    def safe_json(key, path):
        try:
            with open(path) as f:
                data[key] = json.load(f)
            data[f"{key}_ok"] = True
        except Exception as e:
            data[key] = []
            data[f"{key}_ok"] = False
            data[f"{key}_err"] = str(e)

    safe_csv("txn",        os.path.join(DATA_DIR,    "transactions_synthetic.csv"), parse_dates=["date"])
    safe_csv("features",   os.path.join(DATA_DIR,    "user_features.csv"), index_col="user_id")
    safe_csv("clusters",   os.path.join(DATA_DIR,    "user_clusters.csv"))
    safe_csv("leakage",    os.path.join(DATA_DIR,    "user_leakage.csv"))
    safe_csv("recurring",  os.path.join(DATA_DIR,    "recurring_transactions.csv"))
    safe_csv("forecast",   os.path.join(OUTPUTS_DIR, "wellness_forecast_baseline.csv"))
    safe_csv("summary",    os.path.join(OUTPUTS_DIR, "personas_summary.csv"))
    safe_json("insights",  os.path.join(OUTPUTS_DIR, "persona_insights.json"))
    safe_json("fc_json",   os.path.join(MODELS_DIR,  "baseline_forecasts.json"))

    return data


def metric_card(col, label, value, delta=None, help_text=None):
    col.metric(label=label, value=value, delta=delta, help=help_text)


def missing_data_warning(name: str):
    st.markdown(
        f'<div class="warn-box">⚠️ <b>{name}</b> data not found. '
        f'Run the corresponding src/ script to generate it.</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────
D = load_all_data()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/money-bag.png", width=64)
    st.markdown("## 💰 FinWell Engine")
    st.markdown("---")

    view = st.selectbox(
        "Select View",
        options=[
            "Overview",
            "User Segmentation (Engine 1)",
            "Leakage Detection (Engine 2)",
            "Wellness Forecast (Engine 3)",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
**Financial Wellness Intelligence Platform**

Three integrated ML engines:
- 🎯 **Engine 1** — K-Means User Segmentation
- 🔍 **Engine 2** — Subscription Leakage Detection
- 📈 **Engine 3** — ARIMA + Prophet Forecasting

Built with Python · Scikit-learn · Streamlit
    """)

    # Data status indicators
    st.markdown("---")
    st.markdown("### Data Status")
    for key, label in [("txn","Transactions"), ("clusters","Clusters"),
                       ("leakage","Leakage"), ("forecast","Forecast")]:
        ok = D.get(f"{key}_ok", False)
        icon = "✅" if ok else "❌"
        rows = len(D[key]) if ok else 0
        st.caption(f"{icon} {label} ({rows:,} rows)")

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">💰 Personal Financial Wellness Platform</p>',
            unsafe_allow_html=True)
st.markdown('<p class="sub-title">Understand. Detect. Predict. Recommend.</p>',
            unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if view == "Overview":
    # ── Metric cards ────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)

    total_users = D["txn"]["user_id"].nunique() if D["txn_ok"] else 500
    n_personas  = 5
    n_subs      = len(D["leakage"]) if D["leakage_ok"] else "—"
    mean_ws     = (f"{D['forecast']['wellness_score'].mean():.1f}"
                   if D["forecast_ok"] and len(D["forecast"]) > 0 else "—")

    metric_card(c1, "👥 Total Users",            f"{total_users:,}")
    metric_card(c2, "🎯 Spending Personas",       str(n_personas))
    metric_card(c3, "📺 Subscriptions Detected",  str(n_subs))
    metric_card(c4, "💚 Mean Wellness Score",      mean_ws, help_text="0–100 scale")

    st.markdown("---")

    # ── Engine cards ────────────────────────────────────────────────────────
    st.subheader("🚀 Platform Engines")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
<div class="engine-card">
  <div class="engine-title">🎯 Engine 1 — User Segmentation</div>
  <div class="engine-desc">
    K-Means clustering segments 500 users into 5 behavioural
    spending personas using 20 engineered features.
    <br><br>
    <b>Models:</b> K-Means (k=5) · PCA · StandardScaler<br>
    <b>Output:</b> Cluster labels + interpretations
  </div>
</div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("""
<div class="engine-card">
  <div class="engine-title">🔍 Engine 2 — Leakage Detection</div>
  <div class="engine-desc">
    Identifies forgotten/dormant subscriptions and calculates a
    Leakage Score using cost-per-use, dormancy and overlap factors.
    <br><br>
    <b>Models:</b> Random Forest Classifier · Rule-based scoring<br>
    <b>Output:</b> Leakage scores + annual saving estimates
  </div>
</div>""", unsafe_allow_html=True)

    with col3:
        st.markdown("""
<div class="engine-card">
  <div class="engine-title">📈 Engine 3 — Wellness Forecast</div>
  <div class="engine-desc">
    ARIMA and Prophet models forecast next month's spend and
    compute a Financial Wellness Score (0-100) per user.
    <br><br>
    <b>Models:</b> auto_arima · Facebook Prophet<br>
    <b>Output:</b> Forecast + risk classification
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Category distribution & persona split ──────────────────────────────
    if D["txn_ok"] and D["clusters_ok"]:
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("📊 Spending by Category")
            cat_data = (D["txn"][D["txn"]["transaction_type"] == "Debit"]
                        .groupby("category")["amount"].sum()
                        .reset_index()
                        .sort_values("amount", ascending=False))
            fig = px.pie(cat_data, values="amount", names="category",
                         hole=0.45, color_discrete_sequence=px.colors.qualitative.Set3)
            fig.update_traces(textposition="inside", textinfo="percent+label")
            fig.update_layout(showlegend=False, margin=dict(t=20,b=20,l=20,r=20),
                              height=320)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.subheader("🎯 Persona Distribution")
            persona_counts = D["clusters"]["persona_name"].value_counts().reset_index()
            persona_counts.columns = ["persona_name", "count"]
            colors = [PERSONA_META[i]["color"] for i in range(5)
                      if PERSONA_META[i]["name"] in persona_counts["persona_name"].values]
            fig2 = px.bar(persona_counts, x="count", y="persona_name",
                          orientation="h", color="persona_name",
                          color_discrete_sequence=[m["color"] for m in PERSONA_META.values()])
            fig2.update_layout(showlegend=False, yaxis_title="", xaxis_title="Users",
                               margin=dict(t=20,b=20,l=20,r=20), height=320)
            st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# ENGINE 1 — USER SEGMENTATION
# ─────────────────────────────────────────────────────────────────────────────
elif view == "User Segmentation (Engine 1)":
    st.subheader("🎯 Engine 1 — User Segmentation & Personas")

    if not D["clusters_ok"] or not D["features_ok"]:
        missing_data_warning("User Clusters / Features")
        st.stop()

    clusters = D["clusters"]
    features = D["features"]

    # ── Top metrics ──────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    metric_card(c1, "👥 Users Clustered",    f"{len(clusters):,}")
    metric_card(c2, "🎯 Personas",           "5")
    metric_card(c3, "📊 Features Used",      f"{features.shape[1]}")
    largest = clusters["persona_name"].value_counts().idxmax()
    metric_card(c4, "🔝 Largest Persona",    largest)

    st.markdown("---")

    col_left, col_right = st.columns([1, 2])

    with col_left:
        # ── Persona selector ──────────────────────────────────────────────
        st.markdown("#### 🔎 Explore a Persona")
        persona_options = ["All Personas"] + [m["name"] for m in PERSONA_META.values()]
        selected_persona = st.selectbox("Select Persona", persona_options)

        # Cluster statistics table
        st.markdown("#### 📋 Cluster Statistics")
        stats_rows = []
        for cid, meta in PERSONA_META.items():
            mask  = clusters["cluster_label"] == cid
            count = mask.sum()
            pct   = count / len(clusters) * 100
            # normalized feature means for this cluster
            cluster_feats = features.loc[
                features.index.isin(clusters[mask]["user_id"])
            ]
            stats_rows.append({
                "Persona":     f"{meta['emoji']} {meta['name']}",
                "Users":       count,
                "Share":       f"{pct:.1f}%",
                "Avg Spend":   f"{cluster_feats['avg_monthly_spend'].mean():.3f}",
                "Savings":     f"{cluster_feats['savings_ratio'].mean():.3f}",
                "Volatility":  f"{cluster_feats['spending_volatility'].mean():.3f}",
            })
        stats_df = pd.DataFrame(stats_rows)

        if selected_persona != "All Personas":
            stats_df = stats_df[stats_df["Persona"].str.contains(selected_persona, regex=False)]

        st.dataframe(stats_df, use_container_width=True, hide_index=True)

        # ── Insights from JSON ────────────────────────────────────────────
        if D["insights_ok"] and D["insights"]:
            st.markdown("#### 💡 AI Insight")
            for rec in D["insights"]:
                if selected_persona == "All Personas" or rec["persona_name"] == selected_persona:
                    meta  = next((m for m in PERSONA_META.values()
                                  if m["name"] == rec["persona_name"]), {})
                    emoji = meta.get("emoji", "")
                    st.markdown(f"**{emoji} {rec['persona_name']}**")
                    st.caption(rec.get("characteristic", ""))
                    st.info(rec.get("recommendation", ""))
                    if selected_persona != "All Personas":
                        break

    with col_right:
        # ── PCA scatter ───────────────────────────────────────────────────
        st.markdown("#### 🗺️ User Cluster Map (PCA Projection)")

        feat_vals   = features.values
        pca         = PCA(n_components=2, random_state=42)
        coords      = pca.fit_transform(feat_vals)
        var_ratio   = pca.explained_variance_ratio_

        pca_df = pd.DataFrame({
            "PC1":         coords[:, 0],
            "PC2":         coords[:, 1],
            "user_id":     features.index,
        })
        pca_df = pca_df.merge(
            clusters[["user_id", "cluster_label", "persona_name"]],
            on="user_id", how="left",
        )
        pca_df["color"]  = pca_df["cluster_label"].map(
            {cid: m["color"] for cid, m in PERSONA_META.items()}
        )
        pca_df["label"]  = pca_df["cluster_label"].map(
            {cid: f"{m['emoji']} {m['name']}" for cid, m in PERSONA_META.items()}
        )

        # Filter if persona selected
        plot_df = pca_df if selected_persona == "All Personas" else \
                  pca_df[pca_df["persona_name"] == selected_persona]

        color_map = {f"{m['emoji']} {m['name']}": m["color"]
                     for m in PERSONA_META.values()}

        fig = px.scatter(
            plot_df, x="PC1", y="PC2",
            color="label", color_discrete_map=color_map,
            hover_data={"user_id": True, "persona_name": True,
                        "PC1": ":.3f", "PC2": ":.3f"},
            opacity=0.72,
            labels={"label": "Persona",
                    "PC1": f"PC1 ({var_ratio[0]*100:.1f}% var)",
                    "PC2": f"PC2 ({var_ratio[1]*100:.1f}% var)"},
            title=f"PCA Cluster Map — {len(plot_df)} users  "
                  f"({var_ratio.sum()*100:.1f}% variance explained)",
        )
        fig.update_traces(marker=dict(size=7, line=dict(width=0.3, color="white")))
        fig.update_layout(height=420, legend_title_text="Persona",
                          margin=dict(t=40,b=20,l=20,r=20))
        st.plotly_chart(fig, use_container_width=True)

    # ── Download button ───────────────────────────────────────────────────────
    if D["summary_ok"]:
        st.markdown("---")
        csv_bytes = D["summary"].to_csv(index=False).encode()
        st.download_button(
            label="⬇️ Download Persona Summary CSV",
            data=csv_bytes,
            file_name="personas_summary.csv",
            mime="text/csv",
        )


# ─────────────────────────────────────────────────────────────────────────────
# ENGINE 2 — LEAKAGE DETECTION
# ─────────────────────────────────────────────────────────────────────────────
elif view == "Leakage Detection (Engine 2)":
    st.subheader("🔍 Engine 2 — Subscription Leakage Detection")

    if not D["leakage_ok"]:
        missing_data_warning("Leakage Data")
        st.stop()

    leak = D["leakage"].copy()
    leak["estimated_annual_saving"] = leak["estimated_annual_saving"].fillna(0)

    # ── Top metrics ──────────────────────────────────────────────────────────
    high_leak = (leak["leakage_score"] > 60).sum()
    total_saving = leak["estimated_annual_saving"].sum()
    dormant_count = (leak["status"] == "Dormant").sum()

    c1, c2, c3, c4 = st.columns(4)
    metric_card(c1, "📺 Recurring Subs",       f"{len(leak):,}")
    metric_card(c2, "🚨 High Leakage (>60)",   f"{high_leak}")
    metric_card(c3, "💤 Dormant Subs",          f"{dormant_count}")
    metric_card(c4, "💸 Potential Annual Save", f"₹{total_saving:,.0f}")

    st.markdown("---")
    col_l, col_r = st.columns([3, 2])

    with col_l:
        # ── Top 15 bar chart ─────────────────────────────────────────────
        st.markdown("#### 📊 Top 15 Subscriptions by Leakage Score")
        top15 = (
            leak.groupby(["merchant_name", "status"])
            .agg(avg_score=("leakage_score", "mean"),
                 avg_monthly=("monthly_amount", "mean"),
                 users=("user_id", "nunique"))
            .reset_index()
            .sort_values("avg_score", ascending=False)
            .head(15)
        )
        top15["label"] = top15["merchant_name"] + "  [" + top15["status"].str[:3].str.upper() + "]"

        fig = px.bar(
            top15.sort_values("avg_score"),
            x="avg_score", y="label",
            color="status",
            color_discrete_map=STATUS_COLORS,
            orientation="h",
            hover_data={"avg_monthly": ":.0f", "users": True, "avg_score": ":.1f"},
            labels={"avg_score": "Avg Leakage Score", "label": "", "status": "Status"},
            title="Average Leakage Score per Merchant (across users)",
        )
        fig.update_layout(height=480, margin=dict(t=40,b=20,l=20,r=20),
                          legend_title_text="Status")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        # ── Status donut ─────────────────────────────────────────────────
        st.markdown("#### 🥧 Status Breakdown")
        status_counts = leak["status"].value_counts().reset_index()
        status_counts.columns = ["status", "count"]
        fig2 = px.pie(status_counts, values="count", names="status",
                      hole=0.5, color="status",
                      color_discrete_map=STATUS_COLORS)
        fig2.update_traces(textposition="inside", textinfo="percent+label")
        fig2.update_layout(showlegend=True, height=280,
                           margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig2, use_container_width=True)

        # ── Annual savings by status ─────────────────────────────────────
        st.markdown("#### 💰 Annual Savings Potential")
        saving_by_status = (
            leak[leak["status"] != "Active"]
            .groupby("status")["estimated_annual_saving"]
            .sum()
            .reset_index()
        )
        fig3 = px.bar(saving_by_status, x="status", y="estimated_annual_saving",
                      color="status", color_discrete_map=STATUS_COLORS,
                      labels={"estimated_annual_saving": "Annual Savings (₹)", "status": ""},
                      text_auto=".2s")
        fig3.update_layout(showlegend=False, height=200,
                           margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")

    # ── Interactive filterable table ─────────────────────────────────────────
    st.markdown("#### 🗃️ Leakage Records — Interactive Table")
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        status_filter = st.multiselect(
            "Filter by Status",
            options=leak["status"].unique().tolist(),
            default=leak["status"].unique().tolist(),
        )
    with col_f2:
        min_score = st.slider("Min Leakage Score", 0.0, 100.0, 0.0, 5.0)
    with col_f3:
        top_n = st.number_input("Show Top N rows", min_value=5, max_value=276, value=20)

    display = (
        leak[leak["status"].isin(status_filter) & (leak["leakage_score"] >= min_score)]
        .sort_values("leakage_score", ascending=False)
        .head(int(top_n))
        [["user_id","merchant_name","status","monthly_amount",
          "cost_per_use","dormancy_days","leakage_score","estimated_annual_saving"]]
        .rename(columns={
            "monthly_amount":         "Monthly Cost (₹)",
            "cost_per_use":           "Cost/Use (₹)",
            "dormancy_days":          "Dormancy (days)",
            "leakage_score":          "Leakage Score",
            "estimated_annual_saving":"Annual Saving (₹)",
        })
    )
    st.dataframe(display, use_container_width=True, hide_index=True)

    # ── Insight box ───────────────────────────────────────────────────────────
    st.markdown(
        f'<div class="insight-box">💡 <b>Insight:</b> Across {leak["user_id"].nunique()} users '
        f'with recurring subscriptions, <b>{dormant_count} dormant</b> and '
        f'<b>{high_leak} high-leakage</b> services were detected. '
        f'Cancelling these could save users a combined '
        f'<b>₹{total_saving:,.0f}/year</b> '
        f'(avg ₹{total_saving/max(leak["user_id"].nunique(),1):,.0f} per user).</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENGINE 3 — WELLNESS FORECAST
# ─────────────────────────────────────────────────────────────────────────────
elif view == "Wellness Forecast (Engine 3)":
    st.subheader("📈 Engine 3 — Financial Wellness Forecast")

    if not D["forecast_ok"] or len(D["forecast"]) == 0:
        missing_data_warning("Wellness Forecast")
        st.stop()

    fc = D["forecast"].copy()

    # ── Top metrics ──────────────────────────────────────────────────────────
    mean_ws   = fc["wellness_score"].mean()
    low_risk  = (fc["risk_level"] == "Low Risk").sum()
    med_risk  = (fc["risk_level"] == "Medium Risk").sum()
    high_risk = (fc["risk_level"] == "High Risk").sum()
    cfr_count = fc["cash_flow_risk"].sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    metric_card(c1, "💚 Mean Wellness",    f"{mean_ws:.1f} / 100")
    metric_card(c2, "🟢 Low Risk",         f"{low_risk}")
    metric_card(c3, "🟡 Medium Risk",      f"{med_risk}")
    metric_card(c4, "🔴 High Risk",         f"{high_risk}")
    metric_card(c5, "⚠️ Cash-flow Risk",   f"{int(cfr_count)}")

    st.markdown("---")

    col_top_l, col_top_r = st.columns([2, 1])

    with col_top_l:
        # ── Wellness histogram ────────────────────────────────────────────
        st.markdown("#### 📊 Wellness Score Distribution")
        fig = px.histogram(
            fc, x="wellness_score", nbins=20,
            color="risk_level",
            color_discrete_map=RISK_COLORS,
            barmode="overlay",
            opacity=0.80,
            labels={"wellness_score": "Wellness Score (0–100)", "risk_level": "Risk Level"},
            title="Financial Wellness Score — 30 Forecasted Users",
        )
        fig.add_vline(x=mean_ws, line_dash="dash", line_color="#333333",
                      annotation_text=f"Mean = {mean_ws:.1f}",
                      annotation_position="top right")
        fig.update_layout(height=320, margin=dict(t=40,b=20,l=20,r=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_top_r:
        # ── Score vs forecast scatter ─────────────────────────────────────
        st.markdown("#### 🔮 Forecast vs. Current Avg")
        fig2 = px.scatter(
            fc, x="current_monthly_avg",
            y="forecasted_month13_arima",
            color="risk_level",
            color_discrete_map=RISK_COLORS,
            size="wellness_score",
            hover_data={"user_id": True, "wellness_score": ":.1f"},
            labels={"current_monthly_avg":       "Current Avg (₹)",
                    "forecasted_month13_arima":  "ARIMA Forecast (₹)"},
        )
        # 45-degree reference line
        ax_max = max(fc["current_monthly_avg"].max(),
                     fc["forecasted_month13_arima"].max()) * 1.05
        fig2.add_shape(type="line", x0=0, y0=0, x1=ax_max, y1=ax_max,
                       line=dict(dash="dot", color="#888888", width=1))
        fig2.update_layout(height=320, margin=dict(t=40,b=20,l=20,r=20),
                           showlegend=True)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # ── Interactive user deep-dive ────────────────────────────────────────────
    st.markdown("#### 🔎 User Deep-Dive — Actual + Forecast")

    selected_uid = st.selectbox(
        "Select User ID",
        options=fc["user_id"].tolist(),
        index=0,
    )

    user_row = fc[fc["user_id"] == selected_uid].iloc[0]

    # Retrieve full series from JSON if available
    user_json = None
    if D["fc_json_ok"]:
        for rec in D["fc_json"]:
            if rec["user_id"] == selected_uid:
                user_json = rec
                break

    col_info, col_gauge, col_chart = st.columns([1, 1, 2])

    with col_info:
        st.markdown("**📋 User Stats**")
        st.metric("Current Avg Spend", f"₹{user_row['current_monthly_avg']:,.0f}")
        st.metric("ARIMA Forecast",    f"₹{user_row['forecasted_month13_arima']:,.0f}")
        st.metric("Prophet Forecast",  f"₹{user_row['forecasted_month13_prophet']:,.0f}")
        risk_color = RISK_COLORS.get(user_row["risk_level"], "#888")
        st.markdown(
            f"<b>Risk Level:</b> "
            f"<span style='color:{risk_color};font-weight:700'>"
            f"{user_row['risk_level']}</span>",
            unsafe_allow_html=True,
        )
        cfr_txt = "⚠️ Yes" if user_row["cash_flow_risk"] else "✅ No"
        st.markdown(f"**Cash-flow Risk:** {cfr_txt}")

    with col_gauge:
        # Wellness gauge (Plotly indicator)
        ws = user_row["wellness_score"]
        gauge_color = (
            "#4CAF50" if ws >= 70 else
            "#FFC107" if ws >= 40 else
            "#F44336"
        )
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=ws,
            title={"text": "Wellness Score", "font": {"size": 14}},
            gauge={
                "axis":  {"range": [0, 100], "tickwidth": 1},
                "bar":   {"color": gauge_color},
                "steps": [
                    {"range": [0,   40], "color": "#FFEBEE"},
                    {"range": [40,  70], "color": "#FFF8E1"},
                    {"range": [70, 100], "color": "#E8F5E9"},
                ],
                "threshold": {
                    "line":  {"color": "#333", "width": 3},
                    "thickness": 0.8,
                    "value": ws,
                },
            },
            number={"suffix": " / 100", "font": {"size": 22}},
        ))
        fig_gauge.update_layout(height=240, margin=dict(t=30,b=10,l=20,r=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_chart:
        # Line chart: actual + ARIMA + Prophet
        if user_json and "_series" in user_json:
            series  = user_json["_series"]
            n       = len(series)
            x_act   = list(range(1, n + 1))
            x_fc    = [n, n + 1]

            fig_line = go.Figure()

            # Actual
            fig_line.add_trace(go.Scatter(
                x=x_act, y=series, mode="lines+markers",
                name="Actual", line=dict(color="#1976D2", width=2),
                marker=dict(size=6),
            ))
            # ARIMA
            arima_val = user_row["forecasted_month13_arima"]
            fig_line.add_trace(go.Scatter(
                x=x_fc, y=[series[-1], arima_val],
                mode="lines+markers", name="ARIMA",
                line=dict(color="#F44336", dash="dash", width=2),
                marker=dict(size=9, symbol="diamond"),
            ))
            # Prophet
            prophet_val = user_row["forecasted_month13_prophet"]
            fig_line.add_trace(go.Scatter(
                x=x_fc, y=[series[-1], prophet_val],
                mode="lines+markers", name="Prophet",
                line=dict(color="#9C27B0", dash="dot", width=2),
                marker=dict(size=9, symbol="triangle-up"),
            ))

            fig_line.add_vrect(x0=n + 0.5, x1=n + 1.5,
                               fillcolor="#FFF9C4", opacity=0.4,
                               annotation_text="Forecast", annotation_position="top left")

            fig_line.update_layout(
                height=250,
                title=f"{selected_uid} — Monthly Expense Timeline",
                xaxis_title="Month", yaxis_title="Expense (₹)",
                legend=dict(orientation="h", y=-0.3),
                margin=dict(t=40,b=10,l=20,r=20),
            )
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("Full time-series not available — re-run src/predictive_wellness.py")

    # ── Risk interpretation ───────────────────────────────────────────────────
    interpretations = {
        "Low Risk": (
            "✅ **Low Risk** — This user's forecasted spending is comfortably within "
            "their income minus the ₹10,000 savings target. Financial position is healthy. "
            "_Recommendation:_ Consider increasing the savings target or starting a SIP."
        ),
        "Medium Risk": (
            "⚠️ **Medium Risk** — Forecasted spending is approaching the available budget. "
            "Small shocks (medical, travel) could tip the balance. "
            "_Recommendation:_ Review discretionary categories and build a buffer fund."
        ),
        "High Risk": (
            "🚨 **High Risk** — Forecasted spending exceeds the available budget after "
            "the savings target. Cash-flow pressure is likely next month. "
            "_Recommendation:_ Immediate action needed — cut lowest-priority subscriptions "
            "and reduce impulse spending."
        ),
    }
    st.markdown("---")
    st.markdown("#### 💬 Risk Interpretation")
    st.markdown(interpretations.get(user_row["risk_level"], "No interpretation available."))
