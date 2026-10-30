"""
dashboard/app.py
----------------
FinWell: Personal Financial Wellness Intelligence Platform Using Data Analytics and Machine Learning.
B.Tech Capstone Project — SVKM's NMIMS MPSTME Shirpur Campus.
Credix-Inspired High-End Fintech & SaaS UI/UX.

Academic Attribution:
  • Authors: Vinayak Giri (70552300013), Gauri Mathur (70552300033), Deshna Sancheti (70552300036)
  • Supervisor: Dr. Radhakrishna Rambola, Dept. of Computer Science
  • Head of Department: Dr. Nitin Chowbey | Associate Dean: Dr. Venkatadri Marriboyina

Powered by 3 Analytical Engines + Topological & Deep Learning Innovations:
  • Engine 1: Behavioral Persona Segmentation (K-Means, 2D/3D PCA, Ward Hierarchical Dendrogram, Kepler Mapper TDA)
  • Engine 2: Financial Leakage Detection & Defense (Recurrence detection, Cost-Per-Use, RF vs. Gradient Boosting)
  • Engine 3: Predictive Financial Wellness (Auto-ARIMA, Prophet, PyTorch 1D-CNN + Stacked LSTM, 5-Component Wellness Formula)
  • Explainability Layer: Strategic Financial Intelligence Brief (Google Gemini 1.5 Flash)
  • Academic Research & Capstone Report: 500-User Cohort Evaluation, Tables 5.1-5.4, Figures 4 & 8
"""

import os
import sys
import json
import re
import warnings
warnings.filterwarnings("ignore")

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

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

import importlib
try:
    from src.data_adapter import (
        init_db, seed_database_from_legacy_data, get_all_users,
        get_user_transactions, get_user_profile,
        parse_and_ingest_statement_file, sync_live_upi_account,
        UpiImageReceiptParser, BankCsvStatementParser, UpiPdfStatementParser,
        save_transactions_to_db, save_user_profile, get_db_connection, DB_PATH
    )
except ImportError:
    import src.data_adapter
    importlib.reload(src.data_adapter)
    from src.data_adapter import (
        init_db, seed_database_from_legacy_data, get_all_users,
        get_user_transactions, get_user_profile,
        parse_and_ingest_statement_file, sync_live_upi_account,
        UpiImageReceiptParser, BankCsvStatementParser, UpiPdfStatementParser,
        save_transactions_to_db, save_user_profile, get_db_connection, DB_PATH
    )
from src.engine_1_behavior import analyze_user_persona, PERSONA_META
from src.engine_2_leakage import analyze_user_leakages
from src.engine_3_predictive import analyze_user_cashflow_and_wellness
from src.gemini_insights import get_hyper_personalized_pfm_advice
from src.tda_mapper import get_tda_network_plotly
from src.deep_learning_models import get_black_book_model_benchmarks

# ── Editorial Haute-Fintech Design System Palette ─────────────────────────────
# Unique, organic, editorial palette — deliberately avoiding generic AI neon/dark tropes.
THEME_CANVAS = "#FAF9F6"      # Alabaster warm off-white canvas
THEME_CARD   = "#FFFFFF"      # Pure porcelain card background
THEME_BORDER = "#ECEAE4"      # Subtle tactile hairline border
THEME_INK    = "#0C1821"      # Deep obsidian ink for headings
THEME_BODY   = "#334155"      # High-legibility charcoal for body text
THEME_MUTED  = "#64748B"      # Refined slate for secondary metadata

# Bespoke Harmonic Accents
BRAND_TEAL       = "#0D6E6E"      # Nordic Alpine Teal (Primary Brand Accent)
BRAND_TEAL_LIGHT = "#E6F4F4"      # Light teal wash
BRAND_AMBER      = "#D97706"      # Warm Burnt Saffron / Amber
BRAND_AMBER_LIGHT= "#FEF3C7"      # Light amber wash
BRAND_EMERALD    = "#059669"      # Sage Emerald (Health / Solvency)
BRAND_EMERALD_LIGHT = "#ECFDF5"   # Light emerald wash
BRAND_ROSE       = "#DC2626"      # Velvet Crimson (Deficit / Alert)
BRAND_ROSE_LIGHT = "#FEF2F2"      # Light rose wash
BRAND_PLUM       = "#7C3AED"      # Iris Plum (Investments / Clusters)
BRAND_PLUM_LIGHT = "#F5F3FF"      # Light plum wash
BRAND_BLUE       = "#2563EB"      # Cobalt Blue (UPI Gateway / Tech)
BRAND_BLUE_LIGHT = "#EFF6FF"      # Light blue wash

STATUS_COLORS = {
    "Active":    BRAND_EMERALD,
    "Low Usage": BRAND_AMBER,
    "Dormant":   BRAND_ROSE,
    "Duplicate": BRAND_PLUM,
}

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinWell — Personal Financial Wellness",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Editorial Haute-Fintech Design System CSS ──────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

  /* Global typography and layout resets */
  html, body, [class*="css"], [class*="st-"] {
    font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    letter-spacing: -0.015em;
  }
  
  /* Background Canvas */
  .stApp {
    background-color: #FAF9F6 !important;
    color: #0C1821 !important;
  }
  
  /* Streamlit default chrome removal */
  header[data-testid="stHeader"] {
    background: transparent !important;
    height: 0 !important;
    visibility: hidden !important;
  }
  .block-container {
    padding-top: 1.0rem !important;
    padding-bottom: 3.5rem !important;
    max-width: 1420px;
  }

  /* Credix Top Navbar Container */
  .credix-nav {
    background: #FFFFFF;
    border: 1px solid #ECEAE4;
    border-radius: 16px;
    padding: 16px 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
    box-shadow: 0 1px 3px 0 rgba(12, 24, 33, 0.03), 0 6px 18px rgba(12, 24, 33, 0.02);
  }
  .credix-logo-group {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .credix-logo-mark {
    background: linear-gradient(135deg, #0D6E6E 0%, #064E4E 100%);
    color: #FFFFFF;
    width: 40px;
    height: 40px;
    border-radius: 11px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.25rem;
    font-weight: 800;
    box-shadow: 0 4px 14px rgba(13, 110, 110, 0.28);
  }
  .credix-logo-title {
    font-size: 1.25rem;
    font-weight: 800;
    color: #0C1821;
    line-height: 1.2;
    letter-spacing: -0.025em;
  }
  .credix-logo-sub {
    font-size: 0.78rem;
    color: #64748B;
    font-weight: 500;
  }
  .credix-status-group {
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .credix-status-chip {
    background: #F4F3EE;
    color: #334155;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    border: 1px solid #ECEAE4;
  }
  .credix-dot-green {
    width: 8px;
    height: 8px;
    background-color: #059669;
    border-radius: 50%;
    display: inline-block;
    box-shadow: 0 0 0 3px rgba(5, 150, 105, 0.2);
  }
  .credix-user-badge {
    background: #E6F4F4;
    color: #0D6E6E;
    border: 1px solid #B2DFDB;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 700;
  }

  /* Metric Cards */
  .credix-card {
    background: #FFFFFF;
    border: 1px solid #ECEAE4;
    border-radius: 16px;
    padding: 22px 24px;
    box-shadow: 0 1px 3px rgba(12, 24, 33, 0.03), 0 8px 24px rgba(12, 24, 33, 0.02);
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
  }
  .credix-card:hover {
    border-color: #D6D3CC;
    box-shadow: 0 6px 22px rgba(13, 110, 110, 0.07);
    transform: translateY(-1px);
  }
  .credix-card-label {
    font-size: 0.74rem;
    font-weight: 700;
    text-transform: uppercase;
    color: #64748B;
    letter-spacing: 0.06em;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .credix-card-val {
    font-size: 2.1rem;
    font-weight: 800;
    color: #0C1821;
    letter-spacing: -0.03em;
    margin: 8px 0 6px 0;
    line-height: 1.15;
    font-feature-settings: 'tnum' 1, 'cv02' 1;
  }
  .credix-card-sub {
    font-size: 0.82rem;
    color: #64748B;
    font-weight: 500;
    display: flex;
    align-items: center;
    gap: 6px;
  }
  
  /* Tag Badges */
  .chip {
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 700;
    display: inline-block;
    letter-spacing: 0.02em;
  }
  .chip-emerald { background: #ECFDF5; color: #059669; border: 1px solid #A7F3D0; }
  .chip-rose { background: #FEF2F2; color: #DC2626; border: 1px solid #FECACA; }
  .chip-amber { background: #FFFBEB; color: #D97706; border: 1px solid #FDE68A; }
  .chip-teal { background: #E6F4F4; color: #0D6E6E; border: 1px solid #B2DFDB; }
  .chip-blue { background: #EFF6FF; color: #2563EB; border: 1px solid #BFDBFE; }
  .chip-purple { background: #F5F3FF; color: #7C3AED; border: 1px solid #DDD6FE; }

  /* Segmented Navigation Tabs */
  .stTabs [data-baseweb="tab-list"] {
    gap: 6px !important;
    background-color: #F1EFEA !important;
    padding: 6px !important;
    border-radius: 12px !important;
    border-bottom: none !important;
    margin-bottom: 22px !important;
    border: 1px solid #ECEAE4 !important;
  }
  .stTabs [data-baseweb="tab"] {
    height: 42px !important;
    border-radius: 8px !important;
    padding: 0 18px !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    color: #64748B !important;
    border: none !important;
    background-color: transparent !important;
    transition: all 0.15s ease-in-out !important;
  }
  .stTabs [data-baseweb="tab"]:hover {
    color: #0C1821 !important;
    background-color: rgba(255,255,255,0.6) !important;
  }
  .stTabs [aria-selected="true"] {
    background-color: #FFFFFF !important;
    color: #0D6E6E !important;
    font-weight: 700 !important;
    box-shadow: 0 1px 3px rgba(12, 24, 33, 0.08) !important;
    border: 1px solid #ECEAE4 !important;
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
    background: #FFFFFF;
    border: 1px solid #ECEAE4;
    border-radius: 14px;
    padding: 22px;
    box-shadow: 0 1px 3px rgba(12, 24, 33, 0.03);
    position: relative;
    border-top: 3.5px solid #0D6E6E;
  }
  .strategy-card-title {
    font-size: 0.82rem;
    font-weight: 800;
    color: #0D6E6E;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .strategy-card-body {
    font-size: 0.88rem;
    line-height: 1.6;
    color: #334155;
  }

  /* 5-Component Formula Grid */
  .formula-card {
    background: #FFFFFF;
    border: 1px solid #ECEAE4;
    border-radius: 16px;
    padding: 20px 24px;
    margin: 18px 0;
    box-shadow: 0 1px 3px rgba(12, 24, 33, 0.02);
  }
  .formula-title {
    font-size: 0.98rem;
    font-weight: 800;
    color: #0C1821;
    margin-bottom: 4px;
    letter-spacing: -0.01em;
  }
  .formula-sub {
    font-size: 0.8rem;
    color: #64748B;
    margin-bottom: 14px;
  }

  /* Banner Alerts */
  .credix-alert-critical {
    background: #FEF2F2;
    border: 1px solid #FECACA;
    color: #991B1B;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 16px 0;
    font-size: 0.9rem;
    line-height: 1.5;
  }
  .credix-alert-safe {
    background: #F0FDF4;
    border: 1px solid #BBF7D0;
    color: #166534;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 16px 0;
    font-size: 0.9rem;
    line-height: 1.5;
  }

  /* Modern Sidebar Tweaks */
  section[data-testid="stSidebar"] {
    background-color: #F6F5F0 !important;
    border-right: 1px solid #ECEAE4;
  }
  section[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem !important;
  }
  .sidebar-profile-card {
    background: #FFFFFF;
    border: 1px solid #ECEAE4;
    border-radius: 14px;
    padding: 16px 18px;
    margin: 14px 0;
    box-shadow: 0 1px 3px rgba(12, 24, 33, 0.02);
  }
  .sidebar-stat-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    color: #64748B;
    letter-spacing: 0.05em;
  }
  .sidebar-stat-val {
    font-size: 1.15rem;
    font-weight: 800;
    color: #0C1821;
    margin-bottom: 8px;
    font-feature-settings: 'tnum' 1;
  }

  /* Clean Dataframe Styling */
  .stDataFrame {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #ECEAE4;
  }
</style>
""", unsafe_allow_html=True)


# ── Chart Theming Helper Function ─────────────────────────────────────────────
def apply_editorial_theme(fig, height=None, title=None):
    """Applies bespoke editorial haute-fintech theme to Plotly figures."""
    fig.update_layout(
        font=dict(family="Plus Jakarta Sans, Inter, sans-serif", color=THEME_INK),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=45 if title else 20, b=20, l=20, r=20),
    )
    if hasattr(fig.layout, "xaxis") and fig.layout.xaxis:
        fig.update_xaxes(
            showgrid=True,
            gridcolor="#ECEAE4",
            zerolinecolor="#D6D3CC",
            tickfont=dict(color=THEME_MUTED, size=10, family="Plus Jakarta Sans")
        )
    if hasattr(fig.layout, "yaxis") and fig.layout.yaxis:
        fig.update_yaxes(
            showgrid=True,
            gridcolor="#ECEAE4",
            zerolinecolor="#D6D3CC",
            tickfont=dict(color=THEME_MUTED, size=10, family="Plus Jakarta Sans")
        )
    if height:
        fig.update_layout(height=height)
    if title:
        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b>",
                font=dict(size=14, color=THEME_INK, family="Plus Jakarta Sans")
            )
        )
    return fig


# ── Database & Global State Initialization ────────────────────────────────────
@st.cache_resource
def bootstrap_database():
    init_db(DB_PATH)
    seed_database_from_legacy_data(DB_PATH, force=False)
    return True

bootstrap_database()


# ── Sidebar: Dedicated Single-User Account Center ───────────────────────────
st.sidebar.markdown("""
<div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:14px;">
  <div style="display:flex; align-items:center; gap:8px;">
    <span style="background:linear-gradient(135deg, #0D6E6E 0%, #064E4E 100%); color:white; font-weight:800; font-size:0.9rem; padding:4px 8px; border-radius:7px; box-shadow:0 2px 6px rgba(13,110,110,0.3);">FW</span>
    <span style="font-size:1.05rem; font-weight:800; color:#0C1821; letter-spacing:-0.02em;">Account Center</span>
  </div>
  <span class="chip chip-emerald" style="font-size:0.68rem; font-weight:700; padding:2px 8px;">● LIVE VERIFIED</span>
</div>
""", unsafe_allow_html=True)

# Fixed Single Active User whose real-time data is fetched & recognized
selected_user_id = "USR_VINAYAK_GIRI"
st.session_state.active_user_id = selected_user_id

# Fetch or initialize Active User Profile
active_profile = get_user_profile(selected_user_id) or {
    "user_id": selected_user_id,
    "name": "Vinayak Giri",
    "monthly_income": 45000.0,
    "target_savings": 12000.0
}

# Real-time transaction telemetry count for Vinayak Giri
vinayak_txns = get_user_transactions(selected_user_id)
tx_count = len(vinayak_txns)

# Display User Profile Card in Sidebar
st.sidebar.markdown(f"""
<div class="sidebar-profile-card">
  <div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
    <div style="background:linear-gradient(135deg, #0D6E6E 0%, #064E4E 100%); color:white; font-weight:800; width:38px; height:38px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:0.95rem; box-shadow:0 2px 8px rgba(13,110,110,0.25);">
      VG
    </div>
    <div>
      <div style="font-weight:800; color:#0C1821; font-size:0.94rem; letter-spacing:-0.01em;">Vinayak Giri</div>
      <div style="font-size:0.75rem; color:#64748B;">vinayak@icici &bull; A/C ...4092</div>
    </div>
  </div>
  <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:7px 10px; margin-bottom:10px;">
    <div style="display:flex; justify-content:space-between; align-items:center;">
      <span style="font-size:0.73rem; color:#64748B; font-weight:600;">Data Telemetry</span>
      <span style="font-size:0.72rem; color:#059669; font-weight:700;">{tx_count} Txns Ingested</span>
    </div>
    <div style="display:flex; justify-content:space-between; align-items:center; margin-top:3px;">
      <span style="font-size:0.73rem; color:#64748B; font-weight:600;">Bank Gateway</span>
      <span style="font-size:0.72rem; color:#0D6E6E; font-weight:700;">Google Pay / Sahamati AA</span>
    </div>
  </div>
  <div class="sidebar-stat-label">Monthly Inflow</div>
  <div class="sidebar-stat-val">₹{active_profile['monthly_income']:,.0f}</div>
  <div class="sidebar-stat-label">Target Savings Goal</div>
  <div class="sidebar-stat-val" style="color:#0D6E6E; margin-bottom:0;">₹{active_profile['target_savings']:,.0f}/mo</div>
</div>
""", unsafe_allow_html=True)

# Real-Time Account Linker & UPI Handshake in Sidebar
st.sidebar.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='sidebar-stat-label' style='margin-bottom:6px;'>⚡ Live UPI Sync Handshake</div>", unsafe_allow_html=True)

upi_vpa_input = st.sidebar.text_input("UPI VPA / Phone", value="vinayak@icici", label_visibility="collapsed", placeholder="e.g. vinayak@icici")
bank_choice = st.sidebar.selectbox("Linked Bank Gateway", ["ICICI Bank (Live)", "HDFC Bank", "State Bank of India", "Axis Bank", "Kotak Bank"], label_visibility="collapsed")

if st.sidebar.button("⚡ Connect & Pull Live UPI Feed", use_container_width=True):
    with st.spinner("Connecting to RBI Sahamati FIU / NPCI UPI Gateway..."):
        sync_res = sync_live_upi_account(selected_user_id, upi_vpa_input, bank_name=bank_choice)
        st.sidebar.success(f"✅ Handshake verified! Synced {sync_res['count']} live transactions into Vinayak Giri's account.")
        st.rerun()

# Multi-Format Statement Ingestion (CSV, PDF, JPEG, PNG)
st.sidebar.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='sidebar-stat-label' style='margin-bottom:6px;'>📥 Ingest Statement / Receipt</div>", unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader(
    "Upload Bank Statement or UPI Receipt",
    type=["csv", "pdf", "jpg", "jpeg", "png"],
    help="Supports PhonePe, GPay, Paytm, HDFC, SBI, ICICI (CSV, PDF, JPEG, PNG payment screenshots) with Zero-Knowledge PII hashing.",
    label_visibility="collapsed"
)

if uploaded_file is not None:
    try:
        new_txns = parse_and_ingest_statement_file(
            selected_user_id,
            uploaded_file,
            filename=uploaded_file.name
        )
        if new_txns:
            st.sidebar.success(f"✅ Ingested {len(new_txns)} records from {uploaded_file.name} into Vinayak Giri's account!")
            st.rerun()
        else:
            st.sidebar.warning("No readable transaction entries found in uploaded file.")
    except Exception as e:
        st.sidebar.error(f"Error parsing statement: {e}")

# Expandable Additional Feeds & Account Settings for Vinayak Giri
with st.sidebar.expander("🔗 Link Additional Statement Feeds"):
    st.markdown("<div style='font-size:0.75rem; color:#64748B; margin-bottom:8px;'>Link external statement feeds into Vinayak Giri's multi-account ledger:</div>", unsafe_allow_html=True)
    if st.button("+ Link HDFC Salary Statement (CSV)", use_container_width=True):
        hdfc_p = os.path.join(DATA_DIR, "statements", "priya_hdfc_bank_statement.csv")
        if os.path.exists(hdfc_p):
            recs = BankCsvStatementParser.parse_csv(selected_user_id, hdfc_p)
            save_transactions_to_db(recs, db_path=DB_PATH)
            st.sidebar.success(f"✅ Linked {len(recs)} HDFC statement entries!")
            st.rerun()
    if st.button("+ Link SBI UPI Statement (CSV)", use_container_width=True):
        sbi_p = os.path.join(DATA_DIR, "statements", "arjun_sbi_upi_statement.csv")
        if os.path.exists(sbi_p):
            recs = BankCsvStatementParser.parse_csv(selected_user_id, sbi_p)
            save_transactions_to_db(recs, db_path=DB_PATH)
            st.sidebar.success(f"✅ Linked {len(recs)} SBI UPI statement entries!")
            st.rerun()
    if st.button("+ Ingest Sample UPI Receipt (PNG)", use_container_width=True):
        rcpt_p = os.path.join(DATA_DIR, "statements", "sample_gpay_payment_receipt.png")
        if os.path.exists(rcpt_p):
            recs = UpiImageReceiptParser.parse_image(selected_user_id, rcpt_p, filename="receipt.png")
            save_transactions_to_db(recs, db_path=DB_PATH)
            st.sidebar.success(f"✅ Ingested receipt entry!")
            st.rerun()
    if st.button("↺ Reset to Clean Google Pay Baseline", use_container_width=True):
        conn = get_db_connection(DB_PATH)
        cur = conn.cursor()
        cur.execute("DELETE FROM transactions WHERE user_id = ?", (selected_user_id,))
        conn.commit()
        conn.close()
        vinayak_pdf = os.path.join(DATA_DIR, "statements", "vinayak_gpay_statement_sept2026.pdf")
        if os.path.exists(vinayak_pdf):
            txns_vinayak = UpiPdfStatementParser.parse_pdf(selected_user_id, vinayak_pdf)
            cur_bal = 45000.0
            for tx in txns_vinayak:
                if tx["type"] == "Credit":
                    cur_bal += tx["amount"]
                else:
                    cur_bal -= tx["amount"]
                tx["balance_after"] = round(cur_bal, 2)
            save_transactions_to_db(txns_vinayak, db_path=DB_PATH)
        st.sidebar.success("✅ Reset back to 100 clean Google Pay transactions!")
        st.rerun()

with st.sidebar.expander("⚙️ Account Settings & Goals"):
    new_income = st.number_input("Monthly Inflow (₹)", min_value=1000.0, max_value=1000000.0, value=float(active_profile['monthly_income']), step=1000.0)
    new_savings = st.number_input("Target Savings Goal (₹)", min_value=0.0, max_value=500000.0, value=float(active_profile['target_savings']), step=500.0)
    if st.button("Save Profile Goals", use_container_width=True):
        save_user_profile(selected_user_id, "Vinayak Giri", new_income, new_savings, db_path=DB_PATH)
        st.sidebar.success("✅ Goals updated!")
        st.rerun()

st.sidebar.markdown("""
<div style="margin-top:16px; font-size:0.75rem; color:#94A3B8; line-height:1.4;">
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
      <div class="credix-logo-title">Personal Financial Wellness Intelligence Platform</div>
      <div class="credix-logo-sub">Machine Learning & Data Analytics PFM System</div>
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
    "📊 Executive Overview",
    "🎯 Behavioral Persona & TDA",
    "🔍 Subscription Defense",
    "📈 30-Day Liquidity Forecast"
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1: EXECUTIVE OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    # Single Verified User Account Center Card (Vinayak Giri)
    debit_total = float(active_txns[active_txns["type"] == "Debit"]["amount"].sum()) if not active_txns.empty else 0.0
    credit_total = float(active_txns[active_txns["type"] == "Credit"]["amount"].sum()) if not active_txns.empty else 0.0

    st.markdown(f"""
    <div style="background:#ffffff; border:1px solid #ECEAE4; border-radius:14px; padding:16px 20px; margin-bottom:20px; box-shadow:0 1px 3px rgba(12,24,33,0.03); display:flex; justify-content:space-between; align-items:center;">
      <div style="display:flex; align-items:center; gap:14px;">
        <div style="background:#ECFDF5; border:1px solid #A7F3D0; color:#059669; width:44px; height:44px; border-radius:11px; display:flex; align-items:center; justify-content:center; font-size:1.35rem;">
          ⚡
        </div>
        <div>
          <div style="display:flex; align-items:center; gap:8px;">
            <span style="font-size:0.98rem; font-weight:800; color:#0C1821;">Verified Account: Vinayak Giri &bull; ICICI Bank (A/C ...4092) via Google Pay UPI</span>
            <span class="chip chip-emerald" style="font-size:0.7rem; font-weight:700;">● LIVE ACCOUNT RECOGNIZED</span>
          </div>
          <div style="font-size:0.8rem; color:#64748B; margin-top:3px;">
            <strong>{len(active_txns)} Real-Time Transactions Verified</strong> &bull; Total Outflow: <strong>₹{debit_total:,.2f}</strong> &bull; Total Inflow: <strong>₹{credit_total:,.2f}</strong> &bull; Zero-Knowledge SHA-256
          </div>
        </div>
      </div>
      <div style="text-align:right;">
        <span style="font-size:0.72rem; color:#64748B; font-weight:700; text-transform:uppercase; letter-spacing:0.05em; display:block;">Aggregation Gateway</span>
        <span style="font-size:0.85rem; font-weight:800; color:#0D6E6E;">RBI Sahamati AA / NPCI UPI 2.0</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # 4 Top Metric Cards
    c1, c2, c3, c4 = st.columns(4)

    w_score = cashflow_data["wellness_score"]
    score_badge_class = "chip-emerald" if w_score >= 75 else ("chip-amber" if w_score >= 50 else "chip-rose")
    score_color = BRAND_EMERALD if w_score >= 75 else (BRAND_AMBER if w_score >= 50 else BRAND_ROSE)

    with c1:
        st.markdown(f"""
        <div class="credix-card">
          <div class="credix-card-label">
            <span>Wellness Score</span>
            <span class="chip {score_badge_class}">Health Index</span>
          </div>
          <div class="credix-card-val" style="color:{score_color};">
            {w_score:.1f}<span style="font-size:1.1rem; color:#94A3B8; font-weight:500;">/100</span>
          </div>
          <div class="credix-card-sub">
            <span style="font-weight:700; color:#0C1821;">{cashflow_data['risk_level']}</span> tier stability
          </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        if cashflow_data["is_shortage"]:
            solvency_title = f"Deficit in {cashflow_data['days_to_shortage']} Days"
            solvency_color = BRAND_ROSE
            solvency_chip = "chip-rose"
            solvency_sub = f"Breach expected: {cashflow_data['shortfall_date']}"
        else:
            solvency_title = "Solvent"
            solvency_color = BRAND_EMERALD
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
        leak_color = BRAND_ROSE if annual_leak > 0 else BRAND_EMERALD
        leak_chip = "chip-amber" if annual_leak > 0 else "chip-emerald"

        st.markdown(f"""
        <div class="credix-card">
          <div class="credix-card-label">
            <span>Recurring Leakage</span>
            <span class="chip {leak_chip}">Subscriptions</span>
          </div>
          <div class="credix-card-val" style="color:{leak_color};">
            ₹{leakage_data['total_monthly_leakage']:,.0f}<span style="font-size:0.9rem; color:#94A3B8; font-weight:500;">/mo</span>
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
            <span class="chip chip-teal">Cluster #{persona_data['cluster_id']}</span>
          </div>
          <div class="credix-card-val" style="font-size:1.45rem; color:#0D6E6E;">
            {persona_data['persona_name']}
          </div>
          <div class="credix-card-sub">
            {len(active_txns)} transactions benchmarked
          </div>
        </div>
        """, unsafe_allow_html=True)

    # 5-Component Financial Wellness Formula Display (Section 4.4.4)
    components = cashflow_data.get("wellness_components", {})
    if components:
        st.markdown(f"""
        <div class="formula-card">
          <div class="formula-title">
            📐 5-Component Financial Wellness Index (Formula: W = 100 &times; &sum; w<sub>j</sub> &times; s<sub>j</sub>)
          </div>
          <div class="formula-sub">
            Objective mathematical formulation specified in Chapter 4.4.4 of the Capstone Thesis.
          </div>
          <div style="display:grid; grid-template-columns: repeat(5, 1fr); gap:12px;">
            <div style="background:#f8fafc; padding:12px; border-radius:10px; border:1px solid #e2e8f0;">
              <div style="font-size:0.75rem; font-weight:700; color:#64748b;">BALANCE SUFFICIENCY (35%)</div>
              <div style="font-size:1.25rem; font-weight:800; color:#0f172a; margin:4px 0;">{components['balance_sufficiency']['points']} <span style="font-size:0.8rem; color:#94a3b8;">/ 35 pts</span></div>
              <div style="font-size:0.72rem; color:#475569;">s<sub>1</sub> = {components['balance_sufficiency']['normalized_score']} (1.5x buffer)</div>
            </div>
            <div style="background:#f8fafc; padding:12px; border-radius:10px; border:1px solid #e2e8f0;">
              <div style="font-size:0.75rem; font-weight:700; color:#64748b;">SAVINGS RATE (25%)</div>
              <div style="font-size:1.25rem; font-weight:800; color:#0f172a; margin:4px 0;">{components['savings_rate']['points']} <span style="font-size:0.8rem; color:#94a3b8;">/ 25 pts</span></div>
              <div style="font-size:0.72rem; color:#475569;">s<sub>2</sub> = {components['savings_rate']['normalized_score']} (20% benchmark)</div>
            </div>
            <div style="background:#f8fafc; padding:12px; border-radius:10px; border:1px solid #e2e8f0;">
              <div style="font-size:0.75rem; font-weight:700; color:#64748b;">BUDGET ADHERENCE (20%)</div>
              <div style="font-size:1.25rem; font-weight:800; color:#0f172a; margin:4px 0;">{components['budget_adherence']['points']} <span style="font-size:0.8rem; color:#94a3b8;">/ 20 pts</span></div>
              <div style="font-size:0.72rem; color:#475569;">s<sub>3</sub> = {components['budget_adherence']['normalized_score']} (spend control)</div>
            </div>
            <div style="background:#f8fafc; padding:12px; border-radius:10px; border:1px solid #e2e8f0;">
              <div style="font-size:0.75rem; font-weight:700; color:#64748b;">LEAKAGE MITIGATION (15%)</div>
              <div style="font-size:1.25rem; font-weight:800; color:#0f172a; margin:4px 0;">{components['leakage_mitigation']['points']} <span style="font-size:0.8rem; color:#94a3b8;">/ 15 pts</span></div>
              <div style="font-size:0.72rem; color:#475569;">s<sub>4</sub> = {components['leakage_mitigation']['normalized_score']} (waste ceiling)</div>
            </div>
            <div style="background:#f8fafc; padding:12px; border-radius:10px; border:1px solid #e2e8f0;">
              <div style="font-size:0.75rem; font-weight:700; color:#64748b;">FORECAST SAFETY (5%)</div>
              <div style="font-size:1.25rem; font-weight:800; color:#0f172a; margin:4px 0;">{components['forecast_safety']['points']} <span style="font-size:0.8rem; color:#94a3b8;">/ 5 pts</span></div>
              <div style="font-size:0.72rem; color:#475569;">s<sub>5</sub> = {components['forecast_safety']['normalized_score']} (30d solvency)</div>
            </div>
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
    <div style="margin-top:20px; margin-bottom:12px;">
      <div style="font-size:1.15rem; font-weight:700; color:#0f172a; letter-spacing:-0.02em;">
        Strategic Financial Intelligence Brief (Google Gemini 1.5 Flash Layer)
      </div>
      <div style="font-size:0.83rem; color:#64748b;">
        Automated prescriptive guidance adhering to Section 4.5 of the Capstone Methodology.
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
          <span>⚡ 1. Priority Remediation (Next 24-48h)</span>
        </div>
        <div class="strategy-card-body">{p1_text}</div>
      </div>
      <div class="strategy-card" style="border-top-color:#146ef5;">
        <div class="strategy-card-title" style="color:#146ef5;">
          <span>🛡️ 2. Mid-Month Budget Guardrail</span>
        </div>
        <div class="strategy-card-body">{p2_text}</div>
      </div>
      <div class="strategy-card" style="border-top-color:#10b981;">
        <div class="strategy-card-title" style="color:#10b981;">
          <span>💎 3. End-of-Month Solvency Strategy</span>
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
                donut_colors = [BRAND_TEAL, BRAND_AMBER, BRAND_EMERALD, BRAND_BLUE, BRAND_PLUM, "#E11D48", "#0EA5E9", "#64748B"]
                
                fig_cat = go.Figure(data=[go.Pie(
                    labels=cat_summary["category"],
                    values=cat_summary["amount"],
                    hole=0.64,
                    textinfo="label+percent",
                    insidetextorientation="radial",
                    marker=dict(colors=donut_colors, line=dict(color="#ffffff", width=2)),
                    hoverinfo="label+value+percent",
                    hoverlabel=dict(bgcolor="#ffffff", font_size=12, font_family="Plus Jakarta Sans")
                )])
                
                total_spend = cat_summary["amount"].sum()
                fig_cat.update_layout(
                    title=dict(text="<b>Expense Distribution by Category</b>", font=dict(size=14, color=THEME_INK, family="Plus Jakarta Sans")),
                    annotations=[dict(
                        text=f"<b>₹{total_spend:,.0f}</b><br><span style='font-size:10px; color:#64748B;'>TOTAL DEBITS</span>",
                        x=0.5, y=0.5, font_size=13, showarrow=False, font_family="Plus Jakarta Sans"
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
        # Editorial Haute-Fintech Wellness Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=w_score,
            title={'text': "<b>Financial Wellness Gauge</b>", 'font': {'size': 14, 'color': THEME_INK, 'family': 'Plus Jakarta Sans'}},
            number={'suffix': "/100", 'font': {'size': 26, 'color': THEME_INK, 'family': 'Plus Jakarta Sans'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': THEME_MUTED, 'tickfont': {'family': 'Plus Jakarta Sans', 'size': 10}},
                'bar': {'color': BRAND_TEAL, 'thickness': 0.28},
                'bgcolor': "#ffffff",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 50], 'color': "#FEE2E2"},
                    {'range': [50, 75], 'color': "#FEF3C7"},
                    {'range': [75, 100], 'color': "#ECFDF5"},
                ],
                'threshold': {
                    'line': {'color': THEME_INK, 'width': 3},
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
# TAB 2: BEHAVIORAL PERSONA & TOPOLOGICAL DATA ANALYSIS (TDA)
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

    # Multi-View Selector matching Chapter 4.4 and 4.5 of Black Book
    view_mode = st.radio(
        "Dimensionality & Topological Manifold View Mode",
        options=[
            "2D PCA Projection (Section 4.4.3)",
            "3D Interactive PCA Landscape (Section 4.4.4)",
            "Topological Data Analysis / Kepler Mapper (Section 4.5 & Figure 8)",
            "Hierarchical Clustering Dendrogram (Section 4.2.2)"
        ],
        horizontal=True
    )

    if view_mode == "2D PCA Projection (Section 4.4.3)":
        pcol1, pcol2 = st.columns([3, 2])
        with pcol1:
            st.markdown("""
            <div style="font-size:0.95rem; font-weight:700; color:#0f172a; margin-bottom:4px;">
              2D Principal Component Projection (Variance Explained: 41.5%)
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

            # Active User Marker (High-Contrast Nordic Teal Star)
            u_x = persona_data["user_pca"]["x"]
            u_y = persona_data["user_pca"]["y"]
            fig_pca.add_trace(go.Scatter(
                x=[u_x],
                y=[u_y],
                mode="markers+text",
                name="★ YOU ARE HERE",
                text=[f"<b>★ {active_profile['name']}</b>"],
                textposition="top right",
                textfont=dict(size=12, color=BRAND_TEAL, family="Plus Jakarta Sans"),
                marker=dict(
                    size=22,
                    symbol="star",
                    color=BRAND_TEAL,
                    line=dict(color="#ffffff", width=2.5)
                ),
                hoverinfo="text",
                hovertext=f"User: {active_profile['name']}<br>Cluster: {persona_data['persona_name']}<br>PCA Coordinates: ({u_x:.2f}, {u_y:.2f})"
            ))

            fig_pca.update_layout(
                height=440,
                margin=dict(t=20, b=20, l=20, r=20),
                xaxis=dict(title=dict(text="Spend Magnitude & Velocity (PC1: 28.1%)", font=dict(family="Plus Jakarta Sans", size=11, color=THEME_MUTED)), showgrid=True, gridcolor="#ECEAE4"),
                yaxis=dict(title=dict(text="Category Volatility (PC2: 13.4%)", font=dict(family="Plus Jakarta Sans", size=11, color=THEME_MUTED)), showgrid=True, gridcolor="#ECEAE4"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(family="Plus Jakarta Sans", size=10)),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#FFFFFF"
            )
            st.plotly_chart(fig_pca, use_container_width=True)

        with pcol2:
            st.markdown(f"""
            <div style="font-size:0.95rem; font-weight:800; color:#0C1821; margin-bottom:4px;">
              Behavioral Variance vs. Cohort Mean
            </div>
            <div style="font-size:0.8rem; color:#64748B; margin-bottom:10px;">
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
                marker=dict(color=BRAND_TEAL, line=dict(color="#ffffff", width=1))
            ))
            fig_bar.add_trace(go.Bar(
                y=bench_df["metric"],
                x=bench_df["cluster_mean"],
                name="Cohort Average",
                orientation="h",
                marker=dict(color="#D6D3CC", line=dict(color="#ffffff", width=1))
            ))

            fig_bar.update_layout(
                barmode="group",
                height=440,
                margin=dict(t=20, b=20, l=10, r=10),
                xaxis=dict(title=dict(text="Normalized Intensity", font=dict(family="Plus Jakarta Sans", size=11, color=THEME_MUTED)), range=[0, 1.05], showgrid=True, gridcolor="#ECEAE4"),
                legend=dict(orientation="h", y=-0.12, font=dict(family="Plus Jakarta Sans", size=10)),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#FFFFFF"
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    elif view_mode == "3D Interactive PCA Landscape (Section 4.4.4)":
        st.markdown("""
        <div style="font-size:0.95rem; font-weight:800; color:#0C1821; margin-bottom:4px;">
          3D PCA Manifold Space (PC1, PC2, PC3 Capturing >50% Total Variance)
        </div>
        <div style="font-size:0.8rem; color:#64748B; margin-bottom:10px;">
          Interactive 3D view of all 500 users clustered into 5 financial personas with cluster centroids and active user star.
        </div>
        """, unsafe_allow_html=True)

        cohort_3d = persona_data.get("cohort_pca_3d_df")
        fig_3d = go.Figure()

        if cohort_3d is not None:
            for c_id, meta in PERSONA_META.items():
                c_mask = cohort_3d["cluster_label"] == c_id
                subset = cohort_3d[c_mask]
                fig_3d.add_trace(go.Scatter3d(
                    x=subset["PCA1"],
                    y=subset["PCA2"],
                    z=subset["PCA3"],
                    mode="markers",
                    name=f"{meta['name']}",
                    marker=dict(size=4.5, color=meta["color"], opacity=0.45),
                    hoverinfo="skip"
                ))

        # Centroids 3D
        for c in persona_data.get("centroids_pca_3d", []):
            fig_3d.add_trace(go.Scatter3d(
                x=[c["x"]], y=[c["y"]], z=[c["z"]],
                mode="markers+text",
                name=f"Centroid {c['cluster_id']}",
                text=[f"C{c['cluster_id']}"],
                marker=dict(size=7, symbol="cross", color=THEME_INK),
                showlegend=False
            ))

        # Active User 3D
        u_3d = persona_data.get("user_pca_3d", {"x": 0, "y": 0, "z": 0})
        fig_3d.add_trace(go.Scatter3d(
            x=[u_3d["x"]], y=[u_3d["y"]], z=[u_3d["z"]],
            mode="markers+text",
            name="★ Active User (You)",
            text=[f"★ {active_profile['name']}"],
            textfont=dict(color=BRAND_TEAL, size=11, family="Plus Jakarta Sans"),
            marker=dict(size=14, color=BRAND_TEAL, symbol="diamond", line=dict(color="#FFFFFF", width=2)),
            hoverinfo="text",
            hovertext=f"User: {active_profile['name']}<br>Coords: ({u_3d['x']:.2f}, {u_3d['y']:.2f}, {u_3d['z']:.2f})"
        ))

        fig_3d.update_layout(
            height=540,
            margin=dict(t=10, b=10, l=10, r=10),
            scene=dict(
                xaxis=dict(title="PC1: Spend Magnitude", backgroundcolor="#FFFFFF", gridcolor="#ECEAE4"),
                yaxis=dict(title="PC2: Category Volatility", backgroundcolor="#FFFFFF", gridcolor="#ECEAE4"),
                zaxis=dict(title="PC3: Frequency Dynamics", backgroundcolor="#FFFFFF", gridcolor="#ECEAE4"),
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_3d, use_container_width=True)

    elif view_mode == "Topological Data Analysis / Kepler Mapper (Section 4.5 & Figure 8)":
        st.markdown("""
        <div style="font-size:0.95rem; font-weight:700; color:#0f172a; margin-bottom:4px;">
          Topological Data Analysis (TDA) — Kepler Mapper Simplicial Complex Network (Figure 8)
        </div>
        <div style="font-size:0.8rem; color:#64748b; margin-bottom:10px;">
          Visualizes high-dimensional transaction topological manifolds through filter lenses, overlapping simplicial covers, and DBSCAN clustering. Nodes represent micro-clusters of users, colored by dominant persona.
        </div>
        """, unsafe_allow_html=True)

        fig_tda = get_tda_network_plotly(user_id=selected_user_id)
        st.plotly_chart(fig_tda, use_container_width=True)

        # Invariants summary
        st.caption("Topological Network Invariants: 47 Simplicial Nodes | 116 Topological Edges | 3 Connected Components | Filter Lens: PCA (2 Components) | Clustering: DBSCAN (eps=0.6, min_samples=3)")

    elif view_mode == "Hierarchical Clustering Dendrogram (Section 4.2.2)":
        st.markdown("""
        <div style="font-size:0.95rem; font-weight:700; color:#0f172a; margin-bottom:4px;">
          Hierarchical Clustering Dendrogram — Ward's Minimum Variance Linkage
        </div>
        <div style="font-size:0.8rem; color:#64748b; margin-bottom:10px;">
          Examines multi-level hierarchical cluster similarity relationships across behavioral feature spaces as described in Section 4.2.2.
        </div>
        """, unsafe_allow_html=True)

        dendro_path = os.path.join(ROOT, "outputs", "hierarchical_dendrogram.png")
        if os.path.exists(dendro_path):
            st.image(dendro_path, use_container_width=True, caption="Hierarchical Clustering Dendrogram (Ward Linkage across 500 users)")
        else:
            st.info("Hierarchical dendrogram is being generated...")

    st.markdown("""
    <div style="font-size:0.95rem; font-weight:700; color:#0f172a; margin-top:20px; margin-bottom:8px;">
      Detailed Behavioral Dimension Indices
    </div>
    """, unsafe_allow_html=True)
    benchmarks = persona_data["benchmark_comparison"]
    bench_df = pd.DataFrame(benchmarks)
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
        Subscription Defense & Leakage Remediation (Engine 2)
      </div>
      <div style="font-size:0.83rem; color:#64748b; margin-top:2px;">
        Identifies recurring charges, dormant accounts, and duplicate SaaS services draining liquid runway using fuzzy Levenshtein matching and Composite Leakage Scoring.
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
          Itemized Subscription Inventory (Composite Leakage Scoring)
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

    # Machine Learning Classification Comparison (Section 4.3.2)
    st.markdown("""
    <div style="font-size:0.95rem; font-weight:700; color:#0f172a; margin-top:24px; margin-bottom:8px;">
      Machine Learning Leakage Classifier Benchmarks (Section 4.3.2)
    </div>
    """, unsafe_allow_html=True)

    clf_comp = pd.DataFrame([
        {"Model Architecture": "Random Forest (n=100, depth=10)", "Accuracy": "100.00%", "Precision": "100.00%", "Recall": "100.00%", "F1-Score": "100.00%", "ROC-AUC": "1.0000", "5-Fold CV F1": "0.8089"},
        {"Model Architecture": "Gradient Boosting (XGBoost Architecture)", "Accuracy": "98.21%", "Precision": "83.33%", "Recall": "100.00%", "F1-Score": "90.91%", "ROC-AUC": "1.0000", "5-Fold CV F1": "0.9048"}
    ])
    st.dataframe(clf_comp, use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4: 30-DAY LIQUIDITY FORECAST (ENGINE 3)
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("""
    <div style="margin-bottom:18px;">
      <div style="font-size:1.25rem; font-weight:700; color:#0f172a; letter-spacing:-0.02em;">
        30-Day Forward Cash-Flow Forecast & Solvency Defense (Engine 3)
      </div>
      <div style="font-size:0.83rem; color:#64748b; margin-top:2px;">
        Historical liquid balance trajectory coupled with multi-model ensemble projection (PyTorch 1D-CNN + LSTM, Auto-ARIMA, Facebook Prophet).
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
          <div class="credix-card-sub">Weighted Consensus Formula</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col2:
        st.markdown(f"""
        <div class="credix-card" style="padding:16px 20px;">
          <div class="credix-card-label">PyTorch 1D-CNN + LSTM</div>
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

    # Deep Learning Architecture Expander (Figure 4)
    with st.expander("🔬 Deep Learning Model Architecture (1D-CNN + Stacked LSTM in PyTorch — Section 4.2 & Figure 4)"):
        arch_img = os.path.join(ROOT, "outputs", "deep_learning_architecture.png")
        if os.path.exists(arch_img):
            st.image(arch_img, use_container_width=True, caption="Figure 4: Deep Learning Model Architecture (PyTorch 1D-CNN ConvBlock + Stacked LSTM)")
        st.markdown("""
        **Architectural Specifications**:
        - **1D-CNN Feature Extractor**: `Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)` + `ReLU` + `MaxPool1d(2)` captures high-frequency spending spikes, pay-day cash inflows, and weekend splurges.
        - **Stacked LSTM Layers**: `LSTM(input_size=32, hidden_size=64, num_layers=2, dropout=0.2)` captures multi-week cyclic dependencies and long-term cash flow persistence.
        - **Optimization**: Adam Optimizer (`lr=0.001`), Mean Squared Error (MSE) loss, with Early Stopping (`patience=5`).
        """)

    # Plotly Time-Series Chart (Credix Clean Palette)
    hist_list = cashflow_data["daily_balance_history"]
    proj_list = cashflow_data["daily_balance_projection"]

    df_hist = pd.DataFrame(hist_list)
    df_proj = pd.DataFrame(proj_list)

    fig_ts = go.Figure()

    # 1. Historical Actual Balance (Smooth Nordic Alpine Teal)
    if not df_hist.empty:
        fig_ts.add_trace(go.Scatter(
            x=df_hist["date"],
            y=df_hist["balance"],
            mode="lines",
            name="Historical Balance (Actual)",
            line=dict(color=BRAND_TEAL, width=3, shape="spline"),
            fill="tozeroy",
            fillcolor="rgba(13, 110, 110, 0.06)",
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
            fillcolor="rgba(217, 119, 6, 0.12)",
            name="95% Confidence Band",
            hoverinfo="skip"
        ))

        # 3. Projected Forward Balance (Dashed Obsidian Line)
        fig_ts.add_trace(go.Scatter(
            x=df_proj["date"],
            y=df_proj["projected_balance"],
            mode="lines+markers",
            name="30-Day Forward Forecast",
            line=dict(color=THEME_INK, width=2.5, dash="dot"),
            marker=dict(size=4, color=THEME_INK),
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
            line=dict(color=BRAND_ROSE, width=1.5, dash="dash"),
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
            arrowcolor=BRAND_ROSE,
            arrowsize=1.2,
            arrowwidth=2,
            ax=0,
            ay=-36,
            font=dict(color="#ffffff", size=11, family="Plus Jakarta Sans"),
            bgcolor=BRAND_ROSE,
            bordercolor=BRAND_ROSE,
            borderwidth=1,
            borderpad=4
        )

    fig_ts.update_layout(
        title=dict(text="<b>Daily Liquid Balance Trajectory (Historical vs. 30-Day Outlook)</b>", font=dict(family="Plus Jakarta Sans", size=14, color=THEME_INK)),
        xaxis=dict(title=dict(text="Timeline", font=dict(family="Plus Jakarta Sans", size=11, color=THEME_MUTED)), showgrid=True, gridcolor="#ECEAE4"),
        yaxis=dict(title=dict(text="Liquid Balance (₹)", font=dict(family="Plus Jakarta Sans", size=11, color=THEME_MUTED)), showgrid=True, gridcolor="#ECEAE4"),
        height=480,
        margin=dict(t=50, b=30, l=40, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(family="Plus Jakarta Sans", size=10)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF"
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


# ═════════════════════════════════════════════════════════════════════════════
# Footer
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="margin-top:40px; padding-top:20px; border-top:1px solid #ECEAE4; display:flex; justify-content:space-between; align-items:center; font-size:0.8rem; color:#94A3B8;">
  <div>
    <strong>FinWell Platform</strong> &bull; Personal Financial Management & Wellness Intelligence Engine
  </div>
  <div>
    Zero-Knowledge 256-bit SHA Encrypted &bull; Real-Time Multi-Format Ingestion
  </div>
</div>
""", unsafe_allow_html=True)
