"""
dashboard/app.py
----------------
FinWell: Personal Financial Wellness Intelligence Platform (PFM).
User-centric financial management dashboard powered by 3 personalized ML engines:
  • Engine 1: Personal Spending Persona & Cohort Benchmarking (K-Means + PCA)
  • Engine 2: Individual Subscription Leakage Detection & "What-If" Remediation
  • Engine 3: 30-Day Forward Cash-Flow Forecast & Solvency Defense (Ensemble)
  • GenAI Layer: Hyper-Personalized Advice & Executive Brief (Gemini 1.5 Flash)
"""

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Ensure project root is in python path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data_adapter import (
    init_db, seed_database_from_legacy_data, get_all_users,
    get_user_transactions, get_user_profile,
    parse_and_ingest_statement_file, DB_PATH
)
from src.engine_1_behavior import analyze_user_persona, PERSONA_META
from src.engine_2_leakage import analyze_user_leakages
from src.engine_3_predictive import analyze_user_cashflow_and_wellness
from src.gemini_insights import get_hyper_personalized_pfm_advice

STATUS_COLORS = {
    "Active":    "#16a34a",
    "Low Usage": "#d97706",
    "Dormant":   "#dc2626",
    "Duplicate": "#9333ea",
}

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinWell — Personal Financial Wellness",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS for High-End Fintech UI ────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
  
  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }
  
  .app-header {
    background: linear-gradient(135deg, #0d1b2a 0%, #1b263b 100%);
    color: #ffffff;
    padding: 24px 30px;
    border-radius: 14px;
    margin-bottom: 24px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
  }
  .app-title {
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin: 0;
    color: #ffffff;
  }
  .app-tagline {
    font-size: 1.0rem;
    color: #94d2bd;
    margin-top: 4px;
    font-weight: 500;
  }
  
  .metric-card-box {
    background: #ffffff;
    border-radius: 12px;
    padding: 20px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    height: 100%;
  }
  .metric-card-title {
    font-size: 0.85rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  .metric-card-val {
    font-size: 1.9rem;
    font-weight: 800;
    color: #0f172a;
    margin-top: 6px;
    margin-bottom: 4px;
  }
  .metric-card-sub {
    font-size: 0.85rem;
    color: #475569;
    font-weight: 500;
  }
  
  .advice-card {
    background: #f8fafc;
    border-left: 5px solid #0284c7;
    border-radius: 10px;
    padding: 20px 24px;
    margin: 18px 0;
    box-shadow: 0 2px 10px rgba(0,0,0,0.03);
  }
  .alert-banner-danger {
    background: #fef2f2;
    border: 1px solid #f87171;
    border-left: 6px solid #dc2626;
    color: #991b1b;
    padding: 16px 20px;
    border-radius: 10px;
    font-weight: 600;
    margin: 16px 0;
  }
  .alert-banner-safe {
    background: #f0fdf4;
    border: 1px solid #86efac;
    border-left: 6px solid #16a34a;
    color: #166534;
    padding: 16px 20px;
    border-radius: 10px;
    font-weight: 600;
    margin: 16px 0;
  }
  
  .badge-active { background: #dcfce7; color: #166534; padding: 4px 10px; border-radius: 20px; font-weight: 600; font-size: 0.78rem; }
  .badge-dormant { background: #fee2e2; color: #991b1b; padding: 4px 10px; border-radius: 20px; font-weight: 600; font-size: 0.78rem; }
  .badge-low-usage { background: #fef3c7; color: #92400e; padding: 4px 10px; border-radius: 20px; font-weight: 600; font-size: 0.78rem; }
  .badge-duplicate { background: #f3e8ff; color: #6b21a8; padding: 4px 10px; border-radius: 20px; font-weight: 600; font-size: 0.78rem; }
</style>
""", unsafe_allow_html=True)


# ── Database & Global State Initialization ────────────────────────────────────
@st.cache_resource
def bootstrap_database():
    init_db(DB_PATH)
    seed_database_from_legacy_data(DB_PATH, force=False)
    return True

bootstrap_database()


# ── Sidebar: Account Switcher & Statement Ingestion ───────────────────────────
st.sidebar.markdown("## 👤 Active Profile")

users_df = get_all_users()
user_options = {}

# Build display labels for selectbox
for _, u_row in users_df.iterrows():
    uid = u_row["user_id"]
    uname = u_row["name"]
    # Label
    user_options[uid] = f"{uid} — {uname}"

# Default to authentic verified user
default_idx = 0
preferred_defaults = ["USR_PLAID_GOOD", "USR_PLAID_CUSTOM", "USR_SETU_AA", "USR0001"]
for pref in preferred_defaults:
    if pref in user_options:
        default_idx = list(user_options.keys()).index(pref)
        break

selected_user_id = st.sidebar.selectbox(
    "Select Account",
    options=list(user_options.keys()),
    format_func=lambda x: user_options[x],
    index=default_idx
)

# Fetch Active User Profile
active_profile = get_user_profile(selected_user_id) or {
    "user_id": selected_user_id,
    "name": "Active User",
    "monthly_income": 60000.0,
    "target_savings": 10000.0
}

# Display User Metadata Cards in Sidebar
st.sidebar.markdown(f"""
<div style="background:#f1f5f9; padding:12px 16px; border-radius:8px; margin: 12px 0;">
  <div style="font-size:0.75rem; color:#64748b; font-weight:700;">MONTHLY INCOME</div>
  <div style="font-size:1.15rem; color:#0f172a; font-weight:800;">₹{active_profile['monthly_income']:,.0f}</div>
  <div style="font-size:0.75rem; color:#64748b; font-weight:700; margin-top:8px;">TARGET SAVINGS</div>
  <div style="font-size:1.0rem; color:#0284c7; font-weight:700;">₹{active_profile['target_savings']:,.0f}/mo</div>
</div>
""", unsafe_allow_html=True)

# Statement Uploader in Sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 📥 Ingest Live Statement")
uploaded_file = st.sidebar.file_uploader(
    "Upload Statement (CSV or PDF)",
    type=["csv", "pdf"],
    help="Supports PhonePe, Google Pay, Paytm, HDFC, SBI, ICICI statements in CSV or PDF formats with Zero-Knowledge PII hashing."
)

if uploaded_file is not None:
    try:
        new_txns = parse_and_ingest_statement_file(
            selected_user_id,
            uploaded_file,
            filename=uploaded_file.name
        )
        if new_txns:
            st.sidebar.success(f"✅ Ingested {len(new_txns)} sanitized transactions from {uploaded_file.name}!")
            st.rerun()
        else:
            st.sidebar.warning("No readable transaction entries found in uploaded file.")
    except Exception as e:
        st.sidebar.error(f"Error parsing statement: {e}")

st.sidebar.markdown("---")
st.sidebar.caption("🔒 **Zero-Knowledge Sanitizer Active**: All account numbers & phone numbers hashed with SHA-256 prior to modeling.")


# ── Run Active User Analytics Across All 3 Engines ────────────────────────────
active_txns = get_user_transactions(selected_user_id)

with st.spinner("Analyzing personal financial wellness telemetry..."):
    # Engine 1: Behavioral Persona
    persona_data = analyze_user_persona(selected_user_id, txns_df=active_txns)
    # Engine 2: Subscriptions & Leakages
    leakage_data = analyze_user_leakages(selected_user_id, txns_df=active_txns)
    # Engine 3: Predictive Cash-Flow
    cashflow_data = analyze_user_cashflow_and_wellness(
        selected_user_id,
        txns_df=active_txns,
        monthly_income=active_profile["monthly_income"],
        target_savings=active_profile["target_savings"]
    )


# ── Top App Banner ────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="app-header">
  <div class="app-title">FinWell Financial Intelligence</div>
  <div class="app-tagline">Personalized Financial Health, Leakage Defense & Cash-Flow Solvency</div>
  <div style="margin-top:12px; font-size:0.92rem; color:#cbd5e1;">
    Logged in as: <strong>{active_profile['name']}</strong> &nbsp;|&nbsp; 
    Persona: <span style="background:{persona_data['persona_color']}; color:white; padding:2px 8px; border-radius:12px; font-weight:600;">{persona_data['persona_emoji']} {persona_data['persona_name']}</span> &nbsp;|&nbsp; 
    Verified Telemetry: <strong>{len(active_txns)} transactions</strong>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Tab Navigation ────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 My Financial Health (Overview)",
    "🎯 Spending Persona (Engine 1)",
    "🔍 Subscription Leakage (Engine 2)",
    "📈 30-Day Cash-Flow (Engine 3)"
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW & EXECUTIVE BRIEF
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    # 4 Top Metric Cards
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        w_score = cashflow_data["wellness_score"]
        score_color = "#16a34a" if w_score >= 75 else ("#d97706" if w_score >= 50 else "#dc2626")
        st.markdown(f"""
        <div class="metric-card-box">
          <div class="metric-card-title">Financial Wellness Score</div>
          <div class="metric-card-val" style="color:{score_color};">{w_score:.1f}<span style="font-size:1.1rem; color:#94a3b8;">/100</span></div>
          <div class="metric-card-sub">{cashflow_data['risk_level']} Status</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        if cashflow_data["is_shortage"]:
            status_text = f"Shortage in {cashflow_data['days_to_shortage']} Days"
            status_color = "#dc2626"
            sub_text = f"Deficit date: {cashflow_data['shortfall_date']}"
        else:
            status_text = "Safe & Solvent"
            status_color = "#16a34a"
            sub_text = "30-day buffer intact"

        st.markdown(f"""
        <div class="metric-card-box">
          <div class="metric-card-title">30-Day Solvency Outlook</div>
          <div class="metric-card-val" style="color:{status_color}; font-size:1.55rem;">{status_text}</div>
          <div class="metric-card-sub">{sub_text}</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        annual_leak = leakage_data["total_annual_savings"]
        leak_color = "#dc2626" if annual_leak > 0 else "#16a34a"
        st.markdown(f"""
        <div class="metric-card-box">
          <div class="metric-card-title">Subscription Leakage</div>
          <div class="metric-card-val" style="color:{leak_color};">₹{leakage_data['total_monthly_leakage']:,.0f}<span style="font-size:0.9rem; color:#94a3b8;">/mo</span></div>
          <div class="metric-card-sub">₹{annual_leak:,.0f}/yr recoverable</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="metric-card-box">
          <div class="metric-card-title">Assigned Persona</div>
          <div class="metric-card-val" style="font-size:1.45rem; color:{persona_data['persona_color']};">
            {persona_data['persona_emoji']} {persona_data['persona_name']}
          </div>
          <div class="metric-card-sub">Cohort Cluster #{persona_data['cluster_id']}</div>
        </div>
        """, unsafe_allow_html=True)

    # Critical Shortfall Alert (if any)
    if cashflow_data["is_shortage"]:
        st.markdown(f"""
        <div class="alert-banner-danger">
          ⚠️ <strong>CRITICAL DEFICIT DETECTED:</strong> Your projected cash balance will breach zero on 
          <strong>{cashflow_data['shortfall_date']}</strong> (in {cashflow_data['days_to_shortage']} days). 
          Lowest projected balance: ₹{cashflow_data['lowest_projected_balance']:,.0f}. 
          Review your <strong>Immediate Action Plan</strong> below to preserve solvency.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="alert-banner-safe">
          ✅ <strong>HEALTHY CASH-FLOW:</strong> No deficit projected over the next 30 days. 
          Current liquid balance: <strong>₹{cashflow_data['current_liquid_balance']:,.0f}</strong>. 
          Surplus safety margin: ₹{max(0.0, cashflow_data['lowest_projected_balance']):,.0f}.
        </div>
        """, unsafe_allow_html=True)

    # Gemini 1.5 Flash Executive Brief
    st.markdown("### 🤖 Personalized Remediation Action Plan")
    st.caption("Generated by FinWell Intelligence Layer (Gemini 1.5 Flash)")

    projected_delta = cashflow_data["lowest_projected_balance"]
    personalized_brief = get_hyper_personalized_pfm_advice(
        user_name=active_profile["name"],
        persona_name=persona_data["persona_name"],
        current_balance=cashflow_data["current_liquid_balance"],
        projected_surplus_or_deficit=projected_delta,
        days_to_shortfall=cashflow_data["days_to_shortage"],
        leaking_subscriptions=leakage_data["subscriptions"],
        monthly_income=active_profile["monthly_income"],
        target_savings=active_profile["target_savings"]
    )

    st.markdown(f"""
    <div class="advice-card">
      <div style="font-weight:700; color:#0369a1; font-size:1.05rem; margin-bottom:10px;">
        📋 Actionable Guidance for {active_profile['name']}
      </div>
      <div style="font-size:0.95rem; line-height:1.6; color:#1e293b;">
        {personalized_brief.replace(chr(10), '<br/>')}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Quick Visual Summary
    col_a, col_b = st.columns(2)
    with col_a:
        # Category Spend Distribution
        if not active_txns.empty:
            debits = active_txns[active_txns["type"] == "Debit"]
            if not debits.empty:
                cat_summary = debits.groupby("category")["amount"].sum().reset_index()
                fig_cat = px.pie(
                    cat_summary,
                    names="category",
                    values="amount",
                    title="<b>Your Spending by Category</b>",
                    hole=0.45,
                    color_discrete_sequence=px.colors.qualitative.Prism
                )
                fig_cat.update_layout(margin=dict(t=40, b=10, l=10, r=10), height=300)
                st.plotly_chart(fig_cat, use_container_width=True)
            else:
                st.info("No debit transactions recorded.")
        else:
            st.info("No transaction history available.")

    with col_b:
        # Financial Wellness Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=w_score,
            title={'text': "<b>Financial Wellness Gauge</b>", 'font': {'size': 18}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
                'bar': {'color': score_color},
                'steps': [
                    {'range': [0, 50], 'color': "#fee2e2"},
                    {'range': [50, 75], 'color': "#fef3c7"},
                    {'range': [75, 100], 'color': "#dcfce7"},
                ],
                'threshold': {
                    'line': {'color': "#0f172a", 'width': 3},
                    'thickness': 0.75,
                    'value': w_score
                }
            }
        ))
        fig_gauge.update_layout(margin=dict(t=40, b=10, l=30, r=30), height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2: ENGINE 1 — SPENDING PERSONA & COHORT BENCHMARKING
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(f"### 🎯 Spending Persona: {persona_data['persona_emoji']} {persona_data['persona_name']}")
    st.markdown(f"*{persona_data['persona_desc']}*")

    pcol1, pcol2 = st.columns([3, 2])

    with pcol1:
        st.markdown("#### 📍 You Are Here: 2D Cohort Landscape (PCA Projection)")
        st.caption("All 500 cohort users projected via Principal Component Analysis. Your position is highlighted by the large pulsating star marker.")

        cohort_pca = persona_data["cohort_pca_df"]
        fig_pca = go.Figure()

        # Cohort background points
        if cohort_pca is not None:
            for c_id, meta in PERSONA_META.items():
                c_mask = cohort_pca["cluster_label"] == c_id
                subset = cohort_pca[c_mask]
                fig_pca.add_trace(go.Scatter(
                    x=subset["PCA1"],
                    y=subset["PCA2"],
                    mode="markers",
                    name=f"{meta['emoji']} {meta['name']}",
                    marker=dict(size=6, color=meta["color"], opacity=0.25),
                    hoverinfo="skip"
                ))

        # Centroids
        for c in persona_data["centroids_pca"]:
            fig_pca.add_trace(go.Scatter(
                x=[c["x"]],
                y=[c["y"]],
                mode="markers+text",
                name=f"Centroid #{c['cluster_id']}",
                text=[f"C{c['cluster_id']}"],
                textposition="top center",
                marker=dict(size=11, symbol="cross", color="#0f172a"),
                showlegend=False
            ))

        # Active User Marker (YOU ARE HERE)
        u_x = persona_data["user_pca"]["x"]
        u_y = persona_data["user_pca"]["y"]
        fig_pca.add_trace(go.Scatter(
            x=[u_x],
            y=[u_y],
            mode="markers+text",
            name="★ YOU ARE HERE",
            text=[f"<b>★ {active_profile['name']}</b>"],
            textposition="top right",
            marker=dict(
                size=22,
                symbol="star",
                color="#eab308",
                line=dict(color="#000000", width=2.5)
            ),
            hoverinfo="text",
            hovertext=f"User: {active_profile['name']}<br>Cluster: {persona_data['persona_name']}<br>PCA: ({u_x:.2f}, {u_y:.2f})"
        ))

        fig_pca.update_layout(
            height=460,
            margin=dict(t=20, b=20, l=20, r=20),
            xaxis=dict(title="Principal Component 1 (Overall Spend Magnitude & Velocity)", showgrid=True, gridcolor="#f1f5f9"),
            yaxis=dict(title="Principal Component 2 (Temporal & Category Volatility)", showgrid=True, gridcolor="#f1f5f9"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor="#ffffff"
        )
        st.plotly_chart(fig_pca, use_container_width=True)

    with pcol2:
        st.markdown(f"#### 📊 You vs. Average {persona_data['persona_name']}")
        st.caption("Direct behavioral comparison against peers in your cluster.")

        benchmarks = persona_data["benchmark_comparison"]
        bench_df = pd.DataFrame(benchmarks)

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            y=bench_df["metric"],
            x=bench_df["user_value"],
            name="Your Metric",
            orientation="h",
            marker=dict(color=persona_data["persona_color"])
        ))
        fig_bar.add_trace(go.Bar(
            y=bench_df["metric"],
            x=bench_df["cluster_mean"],
            name="Cohort Mean",
            orientation="h",
            marker=dict(color="#94a3b8")
        ))

        fig_bar.update_layout(
            barmode="group",
            height=460,
            margin=dict(t=20, b=20, l=10, r=10),
            xaxis=dict(title="Normalized Index (0.0 to 1.0)", range=[0, 1.05]),
            legend=dict(orientation="h", y=-0.15),
            plot_bgcolor="#ffffff"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("#### 📋 Detailed Behavioral Indicators")
    st.dataframe(
        bench_df.rename(columns={
            "metric": "Behavioral Dimension",
            "user_value": "Your Value",
            "cluster_mean": "Cohort Average",
            "delta_pct": "Variance (%)"
        }),
        use_container_width=True,
        hide_index=True
    )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3: ENGINE 2 — SUBSCRIPTION & LEAKAGE MANAGER
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 🔍 Personalized Subscription & Leakage Manager")
    st.markdown("Identifies low-usage, dormant, or duplicate services leaking money from your account.")

    subs = leakage_data["subscriptions"]

    if not subs:
        st.info("No recurring subscriptions or leakages detected for this account.")
    else:
        # Action Center & Interactive Simulation
        st.markdown("#### ⚡ Interactive What-If Remediation Simulator")
        st.caption("Check the subscriptions below that you plan to cancel to simulate revised annual savings and updated wellness score in real time.")

        col_sim1, col_sim2 = st.columns([3, 2])

        # Multi-select or checkboxes
        cancel_choices = {}
        for s in subs:
            is_def_waste = s["status"] in ["Dormant", "Duplicate"]
            # Unique session state key
            chk_key = f"chk_{selected_user_id}_{s['merchant']}"
            cancel_choices[s["merchant"]] = st.sidebar.checkbox(
                f"Simulate Cancel: {s['merchant']}",
                value=False,
                key=chk_key,
                help=f"Save ₹{s['monthly_cost']*12:,.0f}/year"
            ) if False else None  # Place in main view below

        sim_cols = st.columns(min(len(subs), 4))
        cancelled_merchants = []
        simulated_extra_saving = 0.0

        for i, s in enumerate(subs):
            col_target = sim_cols[i % len(sim_cols)]
            with col_target:
                chk = st.checkbox(
                    f"Cancel {s['merchant']}",
                    value=(s["status"] == "Dormant"),
                    key=f"sim_chk_{selected_user_id}_{s['merchant']}",
                    help=f"Status: {s['status']} | ₹{s['monthly_cost']}/mo"
                )
                if chk:
                    cancelled_merchants.append(s["merchant"])
                    simulated_extra_saving += s["annual_cost"]

        # Recalculate dynamic savings
        revised_annual_saving = simulated_extra_saving
        savings_boost_points = min(15.0, (simulated_extra_saving / max(active_profile['monthly_income'], 1000.0)) * 10.0)
        simulated_wellness_score = min(99.9, cashflow_data["wellness_score"] + savings_boost_points)

        st.markdown(f"""
        <div style="background:#f8fafc; border:1px solid #cbd5e1; border-radius:10px; padding:16px 20px; margin: 16px 0;">
          <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
            <div>
              <span style="color:#64748b; font-size:0.85rem; font-weight:700;">CANCELLED SERVICES:</span>
              <span style="font-weight:700; color:#0f172a; margin-left:8px;">{len(cancelled_merchants)} of {len(subs)}</span>
              <div style="font-size:0.85rem; color:#475569; margin-top:2px;">{', '.join(cancelled_merchants) if cancelled_merchants else 'None selected'}</div>
            </div>
            <div>
              <span style="color:#64748b; font-size:0.85rem; font-weight:700;">PROJECTED ANNUAL CASH RECOVERY:</span>
              <span style="font-size:1.4rem; font-weight:800; color:#16a34a; margin-left:8px;">₹{revised_annual_saving:,.0f}</span>
            </div>
            <div>
              <span style="color:#64748b; font-size:0.85rem; font-weight:700;">NEW WELLNESS SCORE:</span>
              <span style="font-size:1.4rem; font-weight:800; color:#0284c7; margin-left:8px;">{simulated_wellness_score:.1f}/100</span>
              <span style="color:#16a34a; font-size:0.85rem; font-weight:700;">(+{savings_boost_points:.1f} pts)</span>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Itemized Subscriptions Table
        st.markdown("#### 📋 Itemized Recurring Subscriptions")
        subs_table = []
        for s in subs:
            status_badge = f"""<span class="badge-{s['status'].lower().replace(' ', '-')}">{s['status']}</span>"""
            subs_table.append({
                "Merchant": s["merchant"],
                "Category": s["category"],
                "Monthly Cost": f"₹{s['monthly_cost']:,.2f}",
                "Annual Cost": f"₹{s['annual_cost']:,.2f}",
                "Cost-Per-Use": f"₹{s['cost_per_use']:,.2f}",
                "Days Inactive": f"{s['days_inactive']} days",
                "Leakage Score": s["leakage_score"],
                "Status": s["status"],
                "Potential Annual Saving": f"₹{s['potential_annual_saving']:,.2f}"
            })

        st.dataframe(pd.DataFrame(subs_table), use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4: ENGINE 3 — 30-DAY CASH-FLOW FORECAST & SOLVENCY
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 📈 30-Day Forward Cash-Flow Forecast & Solvency Defense")
    st.markdown("Historical daily balance trajectory combined with an ensemble forecast (LSTM + ARIMA + Prophet) over the next 30 days.")

    fc_info = cashflow_data["forecast_breakdown"]
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Ensemble Forecast", f"₹{cashflow_data['ensemble_monthly_forecast']:,.0f}", help="Weighted average of LSTM, ARIMA, and Prophet models")
    with m_col2:
        st.metric("LSTM (PyTorch)", f"₹{fc_info['lstm']:,.0f}")
    with m_col3:
        st.metric("Auto-ARIMA", f"₹{fc_info['arima']:,.0f}")
    with m_col4:
        st.metric("Facebook Prophet", f"₹{fc_info['prophet']:,.0f}")

    # Build Interactive Time-Series Plotly Chart
    hist_list = cashflow_data["daily_balance_history"]
    proj_list = cashflow_data["daily_balance_projection"]

    df_hist = pd.DataFrame(hist_list)
    df_proj = pd.DataFrame(proj_list)

    fig_ts = go.Figure()

    # 1. Historical Balance (Solid Line)
    if not df_hist.empty:
        fig_ts.add_trace(go.Scatter(
            x=df_hist["date"],
            y=df_hist["balance"],
            mode="lines",
            name="Historical Balance (Actual)",
            line=dict(color="#0284c7", width=3),
            hoverinfo="x+y"
        ))

    # 2. Upper and Lower Confidence Bands
    if not df_proj.empty:
        fig_ts.add_trace(go.Scatter(
            x=df_proj["date"],
            y=df_proj["upper_bound"],
            mode="lines",
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False
        ))
        fig_ts.add_trace(go.Scatter(
            x=df_proj["date"],
            y=df_proj["lower_bound"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(249, 115, 22, 0.15)",
            name="95% Confidence Band",
            hoverinfo="skip"
        ))

        # 3. Projected Balance (Dashed Line)
        fig_ts.add_trace(go.Scatter(
            x=df_proj["date"],
            y=df_proj["projected_balance"],
            mode="lines+markers",
            name="30-Day Forward Forecast (Ensemble)",
            line=dict(color="#ea580c", width=3, dash="dash"),
            marker=dict(size=4),
            hoverinfo="x+y"
        ))

    # 4. Zero Balance Threshold Line
    all_dates = list(df_hist["date"]) + list(df_proj["date"])
    if all_dates:
        fig_ts.add_trace(go.Scatter(
            x=[all_dates[0], all_dates[-1]],
            y=[0, 0],
            mode="lines",
            name="Zero Cash Floor",
            line=dict(color="#dc2626", width=2, dash="dot"),
            hoverinfo="skip"
        ))

    # Highlight Deficit Point (if any)
    if cashflow_data["is_shortage"] and cashflow_data["shortfall_date"]:
        fig_ts.add_annotation(
            x=cashflow_data["shortfall_date"],
            y=0,
            text=f"⚠️ DEFICIT (Day {cashflow_data['days_to_shortage']})",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#dc2626",
            arrowsize=1.5,
            arrowwidth=2,
            ax=0,
            ay=-40,
            font=dict(color="#ffffff", size=12, family="Inter"),
            bgcolor="#dc2626",
            bordercolor="#b91c1c",
            borderwidth=1,
            borderpad=4
        )

    fig_ts.update_layout(
        title="<b>Daily Cash Balance: 60-Day Historical Actual vs. 30-Day Forward Projection</b>",
        xaxis=dict(title="Calendar Date", showgrid=True, gridcolor="#f1f5f9"),
        yaxis=dict(title="Liquid Balance (₹ INR)", showgrid=True, gridcolor="#f1f5f9"),
        height=500,
        margin=dict(t=50, b=30, l=40, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="#ffffff"
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    # Cash-Flow Remediation Strategy Card
    if cashflow_data["is_shortage"]:
        shortfall_amt = abs(cashflow_data["lowest_projected_balance"])
        daily_reduction_needed = shortfall_amt / max(1, cashflow_data["days_to_shortage"])
        st.error(f"""
        **SOLVENCY REMEDIATION REQUIRED:**
        - Projected Deficit: **₹{shortfall_amt:,.0f}** on **{cashflow_data['shortfall_date']}**
        - Action Required: Reduce daily discretionary expenses by at least **₹{daily_reduction_needed:,.0f}/day** immediately.
        - Alternative: Cancel dormant subscriptions in Tab 3 to recover up to **₹{leakage_data['total_annual_savings']:,.0f}/year**.
        """)
    else:
        st.success(f"""
        **SOLVENCY CONFIRMED:**
        - Your liquid buffer stays strictly above zero throughout the next 30 days.
        - Estimated lowest projected balance: **₹{cashflow_data['lowest_projected_balance']:,.0f}**.
        - Recommended habit: Automate **₹{active_profile['target_savings']:,.0f}** into your savings buffer on salary day.
        """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("💎 **FinWell Engine v2.0** — Personalized Financial Health Intelligence Platform. Built with Scikit-learn, PyTorch, Plotly & Streamlit.")
