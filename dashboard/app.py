"""
dashboard/app.py
----------------
FinWell: Personal Financial Wellness Intelligence Platform (PFM).
Credix-Inspired High-End Fintech & SaaS UI/UX.

Powered by 3 personalized ML engines:
  • Engine 1: Personal Spending Persona & Cohort Benchmarking (K-Means + PCA)
  • Engine 2: Individual Subscription Leakage Detection & "What-If" Remediation
  • Engine 3: 30-Day Forward Cash-Flow Forecast & Solvency Defense (Ensemble)
  • Strategic Advisory: Financial Intelligence Briefing (Gemini 1.5 Flash)
"""

import os
import sys
import json
import re
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

# Status colors for badges and tables
STATUS_COLORS = {
    "Active":    "#10b981",
    "Low Usage": "#f59e0b",
    "Dormant":   "#ef4444",
    "Duplicate": "#8b5cf6",
}

# Credix Design System Color Palette
CREDIX_BLUE = "#146ef5"
CREDIX_DARK = "#0f172a"
CREDIX_MUTED = "#64748b"
CREDIX_LIGHT = "#f8fafc"
CREDIX_BORDER = "#e2e8f0"
CREDIX_EMERALD = "#10b981"
CREDIX_AMBER = "#f59e0b"
CREDIX_ROSE = "#ef4444"
CREDIX_PURPLE = "#8b5cf6"

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinWell — Personal Financial Wellness",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Credix Fintech Design System CSS ──────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

  /* Global typography and layout resets */
  html, body, [class*="css"], [class*="st-"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    letter-spacing: -0.011em;
  }
  
  /* Background Canvas */
  .stApp {
    background-color: #f8fafc;
    color: #0f172a;
  }
  
  /* Streamlit default chrome removal */
  header[data-testid="stHeader"] {
    background: transparent !important;
    height: 0 !important;
    visibility: hidden !important;
  }
  .block-container {
    padding-top: 1.25rem !important;
    padding-bottom: 3.5rem !important;
    max-width: 1400px;
  }
  
  /* Credix Top Navbar Container */
  .credix-nav {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 16px 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 24px;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.04), 0 1px 2px -1px rgba(0, 0, 0, 0.02);
  }
  .credix-logo-group {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .credix-logo-mark {
    background: #146ef5;
    color: #ffffff;
    width: 38px;
    height: 38px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
    font-weight: 800;
    box-shadow: 0 4px 12px rgba(20, 110, 245, 0.25);
  }
  .credix-logo-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #0f172a;
    line-height: 1.2;
    letter-spacing: -0.02em;
  }
  .credix-logo-sub {
    font-size: 0.78rem;
    color: #64748b;
    font-weight: 500;
  }
  .credix-status-group {
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .credix-status-chip {
    background: #f1f5f9;
    color: #475569;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    border: 1px solid #e2e8f0;
  }
  .credix-dot-green {
    width: 8px;
    height: 8px;
    background-color: #10b981;
    border-radius: 50%;
    display: inline-block;
    box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.2);
  }
  .credix-user-badge {
    background: #eff6ff;
    color: #1d4ed8;
    border: 1px solid #bfdbfe;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 600;
  }

  /* Metric Cards */
  .credix-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 22px 24px;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.04), 0 1px 2px -1px rgba(0, 0, 0, 0.02);
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
  }
  .credix-card:hover {
    border-color: #cbd5e1;
    box-shadow: 0 4px 16px -2px rgba(20, 110, 245, 0.08);
  }
  .credix-card-label {
    font-size: 0.76rem;
    font-weight: 600;
    text-transform: uppercase;
    color: #64748b;
    letter-spacing: 0.06em;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .credix-card-val {
    font-size: 2.1rem;
    font-weight: 800;
    color: #0f172a;
    letter-spacing: -0.03em;
    margin: 8px 0 6px 0;
    line-height: 1.15;
  }
  .credix-card-sub {
    font-size: 0.82rem;
    color: #64748b;
    font-weight: 500;
    display: flex;
    align-items: center;
    gap: 6px;
  }
  
  /* Tag Badges */
  .chip {
    padding: 3px 9px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    display: inline-block;
  }
  .chip-emerald { background: #ecfdf5; color: #059669; border: 1px solid #a7f3d0; }
  .chip-rose { background: #fff1f2; color: #e11d48; border: 1px solid #fecdd3; }
  .chip-amber { background: #fffbeb; color: #d97706; border: 1px solid #fde68a; }
  .chip-blue { background: #eff6ff; color: #2563eb; border: 1px solid #bfdbfe; }
  .chip-purple { background: #faf5ff; color: #7c3aed; border: 1px solid #e9d5ff; }

  /* Segmented Navigation Tabs */
  .stTabs [data-baseweb="tab-list"] {
    gap: 8px !important;
    background-color: #f1f5f9 !important;
    padding: 6px !important;
    border-radius: 12px !important;
    border-bottom: none !important;
    margin-bottom: 24px !important;
  }
  .stTabs [data-baseweb="tab"] {
    height: 40px !important;
    border-radius: 8px !important;
    padding: 0 20px !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    color: #64748b !important;
    border: none !important;
    background-color: transparent !important;
    transition: all 0.15s ease-in-out !important;
  }
  .stTabs [data-baseweb="tab"]:hover {
    color: #0f172a !important;
    background-color: rgba(255,255,255,0.5) !important;
  }
  .stTabs [aria-selected="true"] {
    background-color: #ffffff !important;
    color: #0f172a !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08), 0 1px 2px rgba(0, 0, 0, 0.04) !important;
  }
  .stTabs [data-baseweb="tab-highlight"] {
    background-color: transparent !important;
  }

  /* Strategy / Intelligence Cards */
  .strategy-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin: 18px 0;
  }
  .strategy-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    position: relative;
    border-top: 3px solid #146ef5;
  }
  .strategy-card-title {
    font-size: 0.82rem;
    font-weight: 700;
    color: #146ef5;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .strategy-card-body {
    font-size: 0.88rem;
    line-height: 1.55;
    color: #334155;
  }

  /* Banner Alerts */
  .credix-alert-critical {
    background: #fff1f2;
    border: 1px solid #fecdd3;
    color: #9f1239;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 16px 0;
    font-size: 0.9rem;
    line-height: 1.5;
  }
  .credix-alert-safe {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    color: #166534;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 16px 0;
    font-size: 0.9rem;
    line-height: 1.5;
  }

  /* Section Container Cards */
  .section-panel {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.04);
    margin-bottom: 20px;
  }
  .panel-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 16px;
    border-bottom: 1px solid #f1f5f9;
    padding-bottom: 12px;
  }
  .panel-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -0.015em;
  }
  .panel-sub {
    font-size: 0.82rem;
    color: #64748b;
  }

  /* Modern Sidebar Tweaks */
  section[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #e2e8f0;
  }
  section[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem !important;
  }
  .sidebar-profile-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px;
    margin: 14px 0;
  }
  .sidebar-stat-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    color: #64748b;
    letter-spacing: 0.04em;
  }
  .sidebar-stat-val {
    font-size: 1.12rem;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 8px;
  }

  /* Clean Dataframe Styling */
  .stDataFrame {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #e2e8f0;
  }
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
st.sidebar.markdown("""
<div style="display:flex; align-items:center; gap:8px; margin-bottom:16px;">
  <span style="background:#146ef5; color:white; font-weight:800; font-size:0.9rem; padding:4px 8px; border-radius:6px;">FW</span>
  <span style="font-size:1.05rem; font-weight:700; color:#0f172a;">Account Center</span>
</div>
""", unsafe_allow_html=True)

users_df = get_all_users()
user_options = {}

for _, u_row in users_df.iterrows():
    uid = u_row["user_id"]
    uname = u_row["name"]
    user_options[uid] = f"{uname} ({uid})"

# Default to authentic verified user
default_idx = 0
preferred_defaults = ["USR_PLAID_GOOD", "USR_PLAID_CUSTOM", "USR_SETU_AA", "USR0001"]
for pref in preferred_defaults:
    if pref in user_options:
        default_idx = list(user_options.keys()).index(pref)
        break

selected_user_id = st.sidebar.selectbox(
    "Active Profile",
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

# Display User Profile Card in Sidebar
st.sidebar.markdown(f"""
<div class="sidebar-profile-card">
  <div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
    <div style="background:#146ef5; color:white; font-weight:700; width:34px; height:34px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:0.95rem;">
      {active_profile['name'][:1]}
    </div>
    <div>
      <div style="font-weight:700; color:#0f172a; font-size:0.95rem;">{active_profile['name']}</div>
      <div style="font-size:0.75rem; color:#64748b;">ID: {selected_user_id}</div>
    </div>
  </div>
  <div class="sidebar-stat-label">Monthly Inflow</div>
  <div class="sidebar-stat-val">₹{active_profile['monthly_income']:,.0f}</div>
  <div class="sidebar-stat-label">Target Savings Goal</div>
  <div class="sidebar-stat-val" style="color:#146ef5; margin-bottom:0;">₹{active_profile['target_savings']:,.0f}/mo</div>
</div>
""", unsafe_allow_html=True)

# Statement Uploader in Sidebar
st.sidebar.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
st.sidebar.markdown("##### 📥 Ingest Statement")
uploaded_file = st.sidebar.file_uploader(
    "Upload Bank / UPI Statement",
    type=["csv", "pdf"],
    help="Supports PhonePe, GPay, Paytm, HDFC, SBI, ICICI statements (CSV & PDF) with Zero-Knowledge PII hashing."
)

if uploaded_file is not None:
    try:
        new_txns = parse_and_ingest_statement_file(
            selected_user_id,
            uploaded_file,
            filename=uploaded_file.name
        )
        if new_txns:
            st.sidebar.success(f"Ingested {len(new_txns)} transactions from {uploaded_file.name}!")
            st.rerun()
        else:
            st.sidebar.warning("No readable transaction entries found in uploaded file.")
    except Exception as e:
        st.sidebar.error(f"Error parsing statement: {e}")

st.sidebar.markdown("""
<div style="margin-top:20px; font-size:0.75rem; color:#94a3b8; line-height:1.4;">
  🔒 <strong>Zero-Knowledge Protocol</strong><br/>
  PII (Phone/Acc#) hashed with SHA-256 prior to modeling layer.
</div>
""", unsafe_allow_html=True)


# ── Run Active User Analytics Across All 3 Engines ────────────────────────────
active_txns = get_user_transactions(selected_user_id)

with st.spinner("Compiling financial intelligence telemetry..."):
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


# ── Credix Top Navbar / Header ────────────────────────────────────────────────
st.markdown(f"""
<div class="credix-nav">
  <div class="credix-logo-group">
    <div class="credix-logo-mark">💎</div>
    <div>
      <div class="credix-logo-title">FinWell Intelligence</div>
      <div class="credix-logo-sub">Personal Financial Management Platform</div>
    </div>
  </div>
  <div class="credix-status-group">
    <span class="credix-status-chip">
      <span class="credix-dot-green"></span> Live Bank Feed
    </span>
    <span class="credix-status-chip">
      🔒 256-bit SHA Verified
    </span>
    <span class="credix-user-badge">
      {persona_data['persona_emoji']} {persona_data['persona_name']}
    </span>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Segmented Tab Navigation ──────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Executive Overview",
    "Behavioral Persona",
    "Subscription Defense",
    "30-Day Liquidity Forecast"
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1: EXECUTIVE OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    # 4 Top Metric Cards (Credix Style)
    c1, c2, c3, c4 = st.columns(4)

    w_score = cashflow_data["wellness_score"]
    score_badge_class = "chip-emerald" if w_score >= 75 else ("chip-amber" if w_score >= 50 else "chip-rose")
    score_color = "#10b981" if w_score >= 75 else ("#f59e0b" if w_score >= 50 else "#ef4444")

    with c1:
        st.markdown(f"""
        <div class="credix-card">
          <div class="credix-card-label">
            <span>Wellness Score</span>
            <span class="chip {score_badge_class}">Health Index</span>
          </div>
          <div class="credix-card-val" style="color:{score_color};">
            {w_score:.1f}<span style="font-size:1.1rem; color:#94a3b8; font-weight:500;">/100</span>
          </div>
          <div class="credix-card-sub">
            <span style="font-weight:600; color:#0f172a;">{cashflow_data['risk_level']}</span> tier stability
          </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        if cashflow_data["is_shortage"]:
            solvency_title = f"Deficit in {cashflow_data['days_to_shortage']} Days"
            solvency_color = "#ef4444"
            solvency_chip = "chip-rose"
            solvency_sub = f"Breach expected: {cashflow_data['shortfall_date']}"
        else:
            solvency_title = "Solvent"
            solvency_color = "#10b981"
            solvency_chip = "chip-emerald"
            solvency_sub = "Full 30-day cushion intact"

        st.markdown(f"""
        <div class="credix-card">
          <div class="credix-card-label">
            <span>Liquidity Outlook</span>
            <span class="chip {solvency_chip}">Forecast</span>
          </div>
          <div class="credix-card-val" style="color:{solvency_color}; font-size:1.85rem;">
            {solvency_title}
          </div>
          <div class="credix-card-sub">
            {solvency_sub}
          </div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        annual_leak = leakage_data["total_annual_savings"]
        leak_color = "#ef4444" if annual_leak > 0 else "#10b981"
        leak_chip = "chip-amber" if annual_leak > 0 else "chip-emerald"

        st.markdown(f"""
        <div class="credix-card">
          <div class="credix-card-label">
            <span>Recurring Leakage</span>
            <span class="chip {leak_chip}">Subscriptions</span>
          </div>
          <div class="credix-card-val" style="color:{leak_color};">
            ₹{leakage_data['total_monthly_leakage']:,.0f}<span style="font-size:0.9rem; color:#94a3b8; font-weight:500;">/mo</span>
          </div>
          <div class="credix-card-sub">
            ₹{annual_leak:,.0f}/yr recoverable cash
          </div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="credix-card">
          <div class="credix-card-label">
            <span>Spending Persona</span>
            <span class="chip chip-blue">Cohort #{persona_data['cluster_id']}</span>
          </div>
          <div class="credix-card-val" style="font-size:1.45rem; color:#146ef5;">
            {persona_data['persona_name']}
          </div>
          <div class="credix-card-sub">
            {len(active_txns)} transactions benchmarked
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Deficit / Healthy Status Alert Banner
    if cashflow_data["is_shortage"]:
        st.markdown(f"""
        <div class="credix-alert-critical">
          <strong>⚠️ Liquidity Alert:</strong> Projected balance is forecasted to fall below zero on 
          <strong>{cashflow_data['shortfall_date']}</strong> (in {cashflow_data['days_to_shortage']} days). 
          Estimated minimum trough: ₹{cashflow_data['lowest_projected_balance']:,.0f}. 
          Review the immediate action items below to prevent overdraft.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="credix-alert-safe">
          <strong>✓ Healthy Cash Reserve:</strong> No liquidity breach forecasted over the next 30 days. 
          Current liquid buffer stands at <strong>₹{cashflow_data['current_liquid_balance']:,.0f}</strong> 
          with a projected floor of ₹{max(0.0, cashflow_data['lowest_projected_balance']):,.0f}.
        </div>
        """, unsafe_allow_html=True)

    # Strategic Financial Advisory (Credix Structured Brief)
    st.markdown("""
    <div style="margin-top:24px; margin-bottom:12px;">
      <div style="font-size:1.15rem; font-weight:700; color:#0f172a; letter-spacing:-0.02em;">
        Strategic Financial Intelligence Brief
      </div>
      <div style="font-size:0.83rem; color:#64748b;">
        Automated prescriptive guidance tailored to current cash velocity and subscription telemetry.
      </div>
    </div>
    """, unsafe_allow_html=True)

    projected_delta = cashflow_data["lowest_projected_balance"]
    raw_brief = get_hyper_personalized_pfm_advice(
        user_name=active_profile["name"],
        persona_name=persona_data["persona_name"],
        current_balance=cashflow_data["current_liquid_balance"],
        projected_surplus_or_deficit=projected_delta,
        days_to_shortfall=cashflow_data["days_to_shortage"],
        leaking_subscriptions=leakage_data["subscriptions"],
        monthly_income=active_profile["monthly_income"],
        target_savings=active_profile["target_savings"]
    )

    # Parse raw brief into 3 distinct structured cards
    p1_text = "Cancel dormant and low-usage subscriptions immediately to free up monthly cash."
    p2_text = f"Maintain daily discretionary spend within ₹{max(300.0, cashflow_data['current_liquid_balance'] / 45.0):,.0f} to protect runway."
    p3_text = f"Lock ₹{active_profile['target_savings']:,.0f} directly into liquid reserves on salary day."

    # Pattern match points
    match_1 = re.search(r'1\.\s*Immediate Action.*?:(.*?)(?=2\.|$)', raw_brief, re.DOTALL | re.IGNORECASE)
    match_2 = re.search(r'2\.\s*Mid-Month Budget.*?:(.*?)(?=3\.|$)', raw_brief, re.DOTALL | re.IGNORECASE)
    match_3 = re.search(r'3\.\s*End-of-Month Solvency.*?:(.*?)(?=$)', raw_brief, re.DOTALL | re.IGNORECASE)

    if match_1:
        p1_text = match_1.group(1).strip()
    if match_2:
        p2_text = match_2.group(1).strip()
    if match_3:
        p3_text = match_3.group(1).strip()

    st.markdown(f"""
    <div class="strategy-grid">
      <div class="strategy-card" style="border-top-color:#ef4444;">
        <div class="strategy-card-title" style="color:#ef4444;">
          <span>⚡ Priority Remediation (24-48h)</span>
        </div>
        <div class="strategy-card-body">{p1_text}</div>
      </div>
      <div class="strategy-card" style="border-top-color:#146ef5;">
        <div class="strategy-card-title" style="color:#146ef5;">
          <span>🛡️ Spend Guardrails (Mid-Month)</span>
        </div>
        <div class="strategy-card-body">{p2_text}</div>
      </div>
      <div class="strategy-card" style="border-top-color:#10b981;">
        <div class="strategy-card-title" style="color:#10b981;">
          <span>💎 Liquidity & Wealth Velocity</span>
        </div>
        <div class="strategy-card-body">{p3_text}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Visual Analytics Row
    col_a, col_b = st.columns(2)

    with col_a:
        # Category Spend Donut Chart
        if not active_txns.empty:
            debits = active_txns[active_txns["type"] == "Debit"]
            if not debits.empty:
                cat_summary = debits.groupby("category")["amount"].sum().reset_index()
                
                # Credix palette for donut
                donut_colors = [CREDIX_BLUE, "#0ea5e9", "#6366f1", "#10b981", "#f59e0b", "#ec4899", "#8b5cf6"]
                
                fig_cat = go.Figure(data=[go.Pie(
                    labels=cat_summary["category"],
                    values=cat_summary["amount"],
                    hole=0.62,
                    textinfo="label+percent",
                    insidetextorientation="radial",
                    marker=dict(colors=donut_colors, line=dict(color="#ffffff", width=2)),
                    hoverinfo="label+value+percent",
                    hoverlabel=dict(bgcolor="#ffffff", font_size=12, font_family="Inter")
                )])
                
                total_spend = cat_summary["amount"].sum()
                fig_cat.update_layout(
                    title=dict(text="<b>Expense Distribution</b>", font=dict(size=15, color="#0f172a", family="Inter")),
                    annotations=[dict(
                        text=f"<b>₹{total_spend:,.0f}</b><br><span style='font-size:10px; color:#64748b;'>TOTAL DEBITS</span>",
                        x=0.5, y=0.5, font_size=14, showarrow=False, font_family="Inter"
                    )],
                    showlegend=False,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(t=45, b=15, l=15, r=15),
                    height=290
                )
                st.plotly_chart(fig_cat, use_container_width=True)
            else:
                st.info("No debit transactions recorded.")
        else:
            st.info("No transaction history available.")

    with col_b:
        # Credix Sleek Financial Wellness Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=w_score,
            title={'text': "<b>Financial Wellness Gauge</b>", 'font': {'size': 15, 'color': '#0f172a', 'family': 'Inter'}},
            number={'suffix': "/100", 'font': {'size': 26, 'color': '#0f172a', 'family': 'Inter'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8", 'tickfont': {'family': 'Inter', 'size': 10}},
                'bar': {'color': CREDIX_BLUE, 'thickness': 0.28},
                'bgcolor': "#ffffff",
                'borderwidth': 0,
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
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=45, b=15, l=25, r=25),
            height=290
        )
        st.plotly_chart(fig_gauge, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2: BEHAVIORAL PERSONA (ENGINE 1)
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(f"""
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px;">
      <div>
        <div style="font-size:1.25rem; font-weight:700; color:#0f172a; letter-spacing:-0.02em;">
          {persona_data['persona_emoji']} {persona_data['persona_name']}
        </div>
        <div style="font-size:0.85rem; color:#64748b; margin-top:2px;">
          {persona_data['persona_desc']}
        </div>
      </div>
      <div class="chip chip-blue" style="font-size:0.8rem; padding:6px 14px;">
        Cluster #{persona_data['cluster_id']}
      </div>
    </div>
    """, unsafe_allow_html=True)

    pcol1, pcol2 = st.columns([3, 2])

    with pcol1:
        st.markdown("""
        <div style="font-size:0.95rem; font-weight:700; color:#0f172a; margin-bottom:4px;">
          Cohort Landscape Projection (PCA)
        </div>
        <div style="font-size:0.8rem; color:#64748b; margin-bottom:10px;">
          Benchmarking your spending habits against all 500 cohort peers. Your position is highlighted by the Credix blue star.
        </div>
        """, unsafe_allow_html=True)

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
                    name=f"{meta['name']}",
                    marker=dict(size=5.5, color=meta["color"], opacity=0.35),
                    hoverinfo="skip"
                ))

        # Centroids
        for c in persona_data["centroids_pca"]:
            fig_pca.add_trace(go.Scatter(
                x=[c["x"]],
                y=[c["y"]],
                mode="markers+text",
                name=f"Centroid {c['cluster_id']}",
                text=[f"C{c['cluster_id']}"],
                textposition="top center",
                textfont=dict(size=9, color="#64748b", family="Inter"),
                marker=dict(size=9, symbol="diamond", color="#0f172a"),
                showlegend=False
            ))

        # Active User Marker (High-Contrast Credix Blue Star)
        u_x = persona_data["user_pca"]["x"]
        u_y = persona_data["user_pca"]["y"]
        fig_pca.add_trace(go.Scatter(
            x=[u_x],
            y=[u_y],
            mode="markers+text",
            name="★ YOU ARE HERE",
            text=[f"<b>★ {active_profile['name']}</b>"],
            textposition="top right",
            textfont=dict(size=12, color=CREDIX_BLUE, family="Inter"),
            marker=dict(
                size=22,
                symbol="star",
                color=CREDIX_BLUE,
                line=dict(color="#ffffff", width=2.5)
            ),
            hoverinfo="text",
            hovertext=f"User: {active_profile['name']}<br>Cluster: {persona_data['persona_name']}<br>PCA Coordinates: ({u_x:.2f}, {u_y:.2f})"
        ))

        fig_pca.update_layout(
            height=440,
            margin=dict(t=20, b=20, l=20, r=20),
            xaxis=dict(title=dict(text="Spend Magnitude & Velocity (PC1)", font=dict(family="Inter", size=11, color="#64748b")), showgrid=True, gridcolor="#f1f5f9"),
            yaxis=dict(title=dict(text="Category Volatility (PC2)", font=dict(family="Inter", size=11, color="#64748b")), showgrid=True, gridcolor="#f1f5f9"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(family="Inter", size=10)),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff"
        )
        st.plotly_chart(fig_pca, use_container_width=True)

    with pcol2:
        st.markdown(f"""
        <div style="font-size:0.95rem; font-weight:700; color:#0f172a; margin-bottom:4px;">
          Behavioral Variance vs. Cohort Mean
        </div>
        <div style="font-size:0.8rem; color:#64748b; margin-bottom:10px;">
          Normalized index (0.0 to 1.0) compared to your peer group.
        </div>
        """, unsafe_allow_html=True)

        benchmarks = persona_data["benchmark_comparison"]
        bench_df = pd.DataFrame(benchmarks)

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            y=bench_df["metric"],
            x=bench_df["user_value"],
            name="Your Metrics",
            orientation="h",
            marker=dict(color=CREDIX_BLUE, line=dict(color="#ffffff", width=1))
        ))
        fig_bar.add_trace(go.Bar(
            y=bench_df["metric"],
            x=bench_df["cluster_mean"],
            name="Cohort Average",
            orientation="h",
            marker=dict(color="#cbd5e1", line=dict(color="#ffffff", width=1))
        ))

        fig_bar.update_layout(
            barmode="group",
            height=440,
            margin=dict(t=20, b=20, l=10, r=10),
            xaxis=dict(title=dict(text="Normalized Intensity", font=dict(family="Inter", size=11, color="#64748b")), range=[0, 1.05], showgrid=True, gridcolor="#f1f5f9"),
            legend=dict(orientation="h", y=-0.12, font=dict(family="Inter", size=10)),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("""
    <div style="font-size:0.95rem; font-weight:700; color:#0f172a; margin-top:16px; margin-bottom:8px;">
      Behavioral Index Table
    </div>
    """, unsafe_allow_html=True)
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
# TAB 3: SUBSCRIPTION DEFENSE (ENGINE 2)
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div style="margin-bottom:18px;">
      <div style="font-size:1.25rem; font-weight:700; color:#0f172a; letter-spacing:-0.02em;">
        Subscription Defense & Leakage Remediation
      </div>
      <div style="font-size:0.83rem; color:#64748b; margin-top:2px;">
        Identifies recurring charges, dormant accounts, and duplicate SaaS services draining liquid runway.
      </div>
    </div>
    """, unsafe_allow_html=True)

    subs = leakage_data["subscriptions"]

    if not subs:
        st.info("No recurring subscriptions or leakages detected for this account.")
    else:
        st.markdown("""
        <div style="font-size:0.95rem; font-weight:700; color:#0f172a; margin-bottom:6px;">
          Interactive What-If Simulation Studio
        </div>
        <div style="font-size:0.8rem; color:#64748b; margin-bottom:14px;">
          Select services below to test cancellation impact on annual savings and overall wellness score in real time.
        </div>
        """, unsafe_allow_html=True)

        sim_cols = st.columns(min(len(subs), 4))
        cancelled_merchants = []
        simulated_extra_saving = 0.0

        for i, s in enumerate(subs):
            col_target = sim_cols[i % len(sim_cols)]
            with col_target:
                status_color = STATUS_COLORS.get(s["status"], "#64748b")
                chk = st.checkbox(
                    f"{s['merchant']}",
                    value=(s["status"] == "Dormant"),
                    key=f"sim_chk_{selected_user_id}_{s['merchant']}",
                    help=f"{s['status']} • ₹{s['monthly_cost']:,.0f}/mo (₹{s['annual_cost']:,.0f}/yr)"
                )
                st.caption(f"₹{s['monthly_cost']:,.0f}/mo • {s['status']}")
                if chk:
                    cancelled_merchants.append(s["merchant"])
                    simulated_extra_saving += s["annual_cost"]

        # Dynamic simulation calculation
        revised_annual_saving = simulated_extra_saving
        savings_boost_points = min(15.0, (simulated_extra_saving / max(active_profile['monthly_income'], 1000.0)) * 10.0)
        simulated_wellness_score = min(99.9, cashflow_data["wellness_score"] + savings_boost_points)

        st.markdown(f"""
        <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:14px; padding:18px 24px; margin: 18px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.03);">
          <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:16px;">
            <div>
              <div style="color:#64748b; font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.04em;">Selected for Removal</div>
              <div style="font-weight:700; color:#0f172a; font-size:1.05rem; margin-top:2px;">
                {len(cancelled_merchants)} of {len(subs)} Services
              </div>
              <div style="font-size:0.8rem; color:#64748b; margin-top:2px;">
                {', '.join(cancelled_merchants) if cancelled_merchants else 'None currently selected'}
              </div>
            </div>
            <div>
              <div style="color:#64748b; font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.04em;">Simulated Annual Recovery</div>
              <div style="font-size:1.55rem; font-weight:800; color:#10b981; margin-top:2px;">
                +₹{revised_annual_saving:,.0f}
              </div>
            </div>
            <div>
              <div style="color:#64748b; font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.04em;">Adjusted Wellness Score</div>
              <div style="font-size:1.55rem; font-weight:800; color:#146ef5; margin-top:2px;">
                {simulated_wellness_score:.1f}<span style="font-size:0.95rem; color:#94a3b8; font-weight:500;">/100</span>
                <span style="color:#10b981; font-size:0.85rem; font-weight:600; margin-left:4px;">(+{savings_boost_points:.1f} pts)</span>
              </div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="font-size:0.95rem; font-weight:700; color:#0f172a; margin-top:18px; margin-bottom:8px;">
          Itemized Subscription Inventory
        </div>
        """, unsafe_allow_html=True)
        
        subs_table = []
        for s in subs:
            subs_table.append({
                "Merchant": s["merchant"],
                "Category": s["category"],
                "Monthly Cost": f"₹{s['monthly_cost']:,.2f}",
                "Annual Cost": f"₹{s['annual_cost']:,.2f}",
                "Cost-Per-Use": f"₹{s['cost_per_use']:,.2f}",
                "Days Inactive": f"{s['days_inactive']} days",
                "Leakage Score": s["leakage_score"],
                "Status": s["status"],
                "Potential Saving": f"₹{s['potential_annual_saving']:,.2f}"
            })

        st.dataframe(pd.DataFrame(subs_table), use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4: 30-DAY LIQUIDITY FORECAST (ENGINE 3)
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("""
    <div style="margin-bottom:18px;">
      <div style="font-size:1.25rem; font-weight:700; color:#0f172a; letter-spacing:-0.02em;">
        30-Day Forward Cash-Flow Forecast & Solvency Defense
      </div>
      <div style="font-size:0.83rem; color:#64748b; margin-top:2px;">
        Historical liquid balance trajectory coupled with multi-model ensemble projection (LSTM + ARIMA + Prophet).
      </div>
    </div>
    """, unsafe_allow_html=True)

    fc_info = cashflow_data["forecast_breakdown"]
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.markdown(f"""
        <div class="credix-card" style="padding:16px 20px;">
          <div class="credix-card-label">Ensemble Forecast</div>
          <div class="credix-card-val" style="font-size:1.6rem; color:#146ef5;">₹{cashflow_data['ensemble_monthly_forecast']:,.0f}</div>
          <div class="credix-card-sub">Weighted 3-Engine Consensus</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col2:
        st.markdown(f"""
        <div class="credix-card" style="padding:16px 20px;">
          <div class="credix-card-label">PyTorch LSTM</div>
          <div class="credix-card-val" style="font-size:1.6rem; color:#0f172a;">₹{fc_info['lstm']:,.0f}</div>
          <div class="credix-card-sub">Deep Recurrent Network</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col3:
        st.markdown(f"""
        <div class="credix-card" style="padding:16px 20px;">
          <div class="credix-card-label">Auto-ARIMA</div>
          <div class="credix-card-val" style="font-size:1.6rem; color:#0f172a;">₹{fc_info['arima']:,.0f}</div>
          <div class="credix-card-sub">Statistical Autoregression</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col4:
        st.markdown(f"""
        <div class="credix-card" style="padding:16px 20px;">
          <div class="credix-card-label">Prophet</div>
          <div class="credix-card-val" style="font-size:1.6rem; color:#0f172a;">₹{fc_info['prophet']:,.0f}</div>
          <div class="credix-card-sub">Bayesian Seasonality Model</div>
        </div>
        """, unsafe_allow_html=True)

    # Plotly Time-Series Chart (Credix Clean Palette)
    hist_list = cashflow_data["daily_balance_history"]
    proj_list = cashflow_data["daily_balance_projection"]

    df_hist = pd.DataFrame(hist_list)
    df_proj = pd.DataFrame(proj_list)

    fig_ts = go.Figure()

    # 1. Historical Actual Balance (Smooth Credix Blue)
    if not df_hist.empty:
        fig_ts.add_trace(go.Scatter(
            x=df_hist["date"],
            y=df_hist["balance"],
            mode="lines",
            name="Historical Balance (Actual)",
            line=dict(color=CREDIX_BLUE, width=3, shape="spline"),
            fill="tozeroy",
            fillcolor="rgba(20, 110, 245, 0.05)",
            hoverinfo="x+y"
        ))

    # 2. Confidence Bands
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
            fillcolor="rgba(245, 158, 11, 0.12)",
            name="95% Confidence Band",
            hoverinfo="skip"
        ))

        # 3. Projected Forward Balance (Dashed Dark Slate Line)
        fig_ts.add_trace(go.Scatter(
            x=df_proj["date"],
            y=df_proj["projected_balance"],
            mode="lines+markers",
            name="30-Day Forward Forecast",
            line=dict(color="#0f172a", width=2.5, dash="dot"),
            marker=dict(size=4, color="#0f172a"),
            hoverinfo="x+y"
        ))

    # 4. Zero Balance Threshold
    all_dates = list(df_hist["date"]) + list(df_proj["date"])
    if all_dates:
        fig_ts.add_trace(go.Scatter(
            x=[all_dates[0], all_dates[-1]],
            y=[0, 0],
            mode="lines",
            name="Zero Cash Floor",
            line=dict(color="#ef4444", width=1.5, dash="dash"),
            hoverinfo="skip"
        ))

    # Highlight Deficit Point (if any)
    if cashflow_data["is_shortage"] and cashflow_data["shortfall_date"]:
        fig_ts.add_annotation(
            x=cashflow_data["shortfall_date"],
            y=0,
            text=f"⚠️ Deficit in {cashflow_data['days_to_shortage']}d",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#ef4444",
            arrowsize=1.2,
            arrowwidth=2,
            ax=0,
            ay=-36,
            font=dict(color="#ffffff", size=11, family="Inter"),
            bgcolor="#ef4444",
            bordercolor="#dc2626",
            borderwidth=1,
            borderpad=4
        )

    fig_ts.update_layout(
        title=dict(text="<b>Daily Liquid Balance Trajectory (Historical vs. 30-Day Outlook)</b>", font=dict(family="Inter", size=15, color="#0f172a")),
        xaxis=dict(title=dict(text="Timeline", font=dict(family="Inter", size=11, color="#64748b")), showgrid=True, gridcolor="#f1f5f9"),
        yaxis=dict(title=dict(text="Liquid Balance (₹)", font=dict(family="Inter", size=11, color="#64748b")), showgrid=True, gridcolor="#f1f5f9"),
        height=480,
        margin=dict(t=50, b=30, l=40, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(family="Inter", size=10)),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff"
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    # Remediation Action Card
    if cashflow_data["is_shortage"]:
        shortfall_amt = abs(cashflow_data["lowest_projected_balance"])
        daily_reduction_needed = shortfall_amt / max(1, cashflow_data["days_to_shortage"])
        st.markdown(f"""
        <div class="credix-alert-critical">
          <strong>Action Plan for Solvency Preservation:</strong><br/>
          • Projected deficit of <strong>₹{shortfall_amt:,.0f}</strong> occurs on <strong>{cashflow_data['shortfall_date']}</strong>.<br/>
          • Reduce daily discretionary outflows by <strong>₹{daily_reduction_needed:,.0f}/day</strong> to defer or eliminate the shortfall.<br/>
          • Cancelling flagged subscriptions in Tab 3 recovers up to <strong>₹{leakage_data['total_annual_savings']:,.0f}/year</strong>.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="credix-alert-safe">
          <strong>Solvency Confirmed:</strong><br/>
          • Liquid balance stays above zero across the entire 30-day forecast horizon.<br/>
          • Projected lowest point: <strong>₹{cashflow_data['lowest_projected_balance']:,.0f}</strong>.<br/>
          • Automate <strong>₹{active_profile['target_savings']:,.0f}</strong> into emergency reserves on salary day to maintain this surplus.
        </div>
        """, unsafe_allow_html=True)


# ── Credix Footer ─────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:40px; padding-top:20px; border-top:1px solid #e2e8f0; display:flex; justify-content:space-between; align-items:center; font-size:0.8rem; color:#94a3b8;">
  <div>
    <strong>FinWell Intelligence</strong> — Enterprise Personal Financial Management Platform
  </div>
  <div>
    Core ML Stack: Scikit-learn • PyTorch LSTM • Auto-ARIMA • Prophet • Plotly
  </div>
</div>
""", unsafe_allow_html=True)
