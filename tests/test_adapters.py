"""
tests/test_adapters.py
----------------------
Unit and integration test suite for FinWell Data Ingestion & Personalized Engines:
1. Plaid Sandbox Adapter Schema Compliance
2. Setu RBI Account Aggregator (AA) Sahamati Parsing & UPI Regex Engine
3. Zero-Knowledge PII Sanitization (SHA-256 Hashes of Phones, Accounts, PAN)
4. Multi-Bank CSV Statement Parser
5. Engine 1: Behavioral Persona Assignment & PCA Coordinates
6. Engine 2: Individual Subscription Leakage & Savings Calculations
7. Engine 3: 30-Day Forward Cash-Flow Projection & Wellness Scoring
8. Database Schema & Query Integrity
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_adapter import (
    PlaidSandboxAdapter, SetuAccountAggregatorAdapter, BankCsvStatementParser,
    UpiPdfStatementParser, parse_and_ingest_statement_file,
    sanitize_pii_string, normalize_merchant_and_category,
    init_db, seed_database_from_legacy_data, get_all_users,
    get_user_transactions, get_user_profile, DB_PATH
)
from src.engine_1_behavior import compute_user_behavioral_features, analyze_user_persona
from src.engine_2_leakage import analyze_user_leakages, fuzzy_normalize_merchant
from src.engine_3_predictive import analyze_user_cashflow_and_wellness
from src.gemini_insights import get_hyper_personalized_pfm_advice


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture(scope="session", autouse=True)
def setup_database():
    init_db(DB_PATH)
    seed_database_from_legacy_data(DB_PATH, force=False)


# ── Suite 1: Data Ingestion & Adapters ───────────────────────────────────────
def test_plaid_sandbox_parsing():
    """Validates Plaid /transactions/get schema parsing without data loss."""
    payload = PlaidSandboxAdapter.generate_sandbox_payload("user_good")
    parsed = PlaidSandboxAdapter.parse_payload("USR_TEST_PLAID", payload)

    assert len(parsed) > 0, "Plaid adapter should return parsed transactions"
    for t in parsed:
        assert "transaction_id" in t
        assert "user_id" in t and t["user_id"] == "USR_TEST_PLAID"
        assert "timestamp" in t
        assert "amount" in t and t["amount"] >= 0.0
        assert t["type"] in ["Debit", "Credit"]
        assert "category" in t
        assert "mode" in t


def test_setu_aa_sahamati_parsing():
    """Validates Setu Account Aggregator FIU payload & UPI regex extraction."""
    payload = SetuAccountAggregatorAdapter.generate_sandbox_payload("USR_TEST_SETU")
    parsed = SetuAccountAggregatorAdapter.parse_payload("USR_TEST_SETU", payload)

    assert len(parsed) > 0, "Setu adapter should parse transactions"
    merchants = [t["merchant"] for t in parsed]
    assert any("Swiggy" in m or "Netflix" in m or "Salary" in m for m in merchants), "UPI merchants must be recognized"

    # Test regex directly
    norm_m, cat, mode = SetuAccountAggregatorAdapter.parse_upi_narration(
        "UPI/523819283912/Payment to Swiggy/swiggy@icici/UPI"
    )
    assert norm_m == "Swiggy"
    assert cat == "Food"
    assert mode == "UPI"


def test_zero_knowledge_pii_sanitization():
    """Validates that sensitive phone numbers, account numbers, and PANs are hashed with SHA-256."""
    raw_sample = "Transferred from 9876543210 to A/C 123456789012 for PAN ABCDE1234F"
    sanitized = sanitize_pii_string(raw_sample)

    assert "9876543210" not in sanitized, "Phone number must be scrubbed"
    assert "123456789012" not in sanitized, "Account number must be scrubbed"
    assert "ABCDE1234F" not in sanitized, "PAN number must be scrubbed"
    assert "ANON_TEL_" in sanitized, "Sanitized text must contain hashed phone token"
    assert "ANON_ACC_" in sanitized, "Sanitized text must contain hashed account token"
    assert "ANON_PAN_" in sanitized, "Sanitized text must contain hashed PAN token"


def test_bank_csv_parsing():
    """Validates multi-bank CSV parsing and automatic format detection."""
    import io
    csv_data = """Date,Narration,Withdrawal Amt.,Deposit Amt.,Balance
01/10/2026,UPI-SWIGGY-swiggy@icici-9876543210,450.00,,45000.00
02/10/2026,NEFT CR-SALARY-TECH CORP,,65000.00,110000.00
"""
    records = BankCsvStatementParser.parse_csv("USR_CSV_TEST", io.StringIO(csv_data))
    assert len(records) == 2
    assert records[0]["type"] == "Debit"
    assert records[0]["amount"] == 450.00
    assert records[1]["type"] == "Credit"
    assert records[1]["amount"] == 65000.00


def test_upi_pdf_statement_parsing():
    """Validates parsing of Indian UPI transaction statements in PDF format."""
    import io
    pdf_bytes = b"""%PDF-1.4
1 0 obj <</Type /Catalog /Pages 2 0 R>> endobj
2 0 obj <</Type /Pages /Kids [3 0 R] /Count 1>> endobj
3 0 obj <</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj
4 0 obj <</Length 240>> stream
BT
/F1 12 Tf
72 700 Td
(05/10/2026 Paid to Swiggy 450.00 DEBIT) Tj
0 -30 Td
(08/10/2026 Paid to Netflix 649.00 DEBIT) Tj
0 -30 Td
(01/10/2026 Received from Tech Corp Salary 65000.00 CREDIT) Tj
ET
endstream
endobj
5 0 obj <</Type /Font /Subtype /Type1 /BaseFont /Helvetica>> endobj
xref
0 6
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000244 00000 n 
0000000534 00000 n 
trailer <</Size 6 /Root 1 0 R>>
startxref
609
%%EOF"""

    records = UpiPdfStatementParser.parse_pdf("USR_PDF_TEST", io.BytesIO(pdf_bytes))
    assert len(records) == 3, "Must extract 3 transactions from UPI PDF"
    assert records[0]["merchant"] == "Swiggy"
    assert records[0]["amount"] == 450.00
    assert records[0]["type"] == "Debit"
    assert records[1]["merchant"] == "Netflix"
    assert records[1]["amount"] == 649.00
    assert records[2]["type"] == "Credit"
    assert records[2]["amount"] == 65000.00


def test_google_pay_authentic_statement_parsing():
    """Validates 100% accurate parsing of Vinayak Giri's authentic Google Pay statement."""
    pdf_path = os.path.join(PROJECT_ROOT, "data", "statements", "vinayak_gpay_statement_sept2026.pdf")
    if os.path.exists(pdf_path):
        txns = UpiPdfStatementParser.parse_pdf("USR_VINAYAK_GIRI", pdf_path)
        assert len(txns) == 100, f"Expected 100 transactions, got {len(txns)}"
        debits = sum(t["amount"] for t in txns if t["type"] == "Debit")
        credits = sum(t["amount"] for t in txns if t["type"] == "Credit")
        assert abs(debits - 39994.60) < 0.01, f"Expected ₹39,994.60 debits, got {debits}"
        assert abs(credits - 28831.00) < 0.01, f"Expected ₹28,831.00 credits, got {credits}"


def test_parse_and_ingest_dispatcher():
    """Validates unified ingestion dispatcher for both CSV and PDF files."""
    import io
    csv_data = "Date,Narration,Amount,Type\n05/10/2026,UPI-Zomato,299.00,Debit\n"
    res = parse_and_ingest_statement_file("USR_DISPATCH_TEST", io.StringIO(csv_data), filename="export.csv")
    assert len(res) == 1
    assert res[0]["amount"] == 299.00


def test_database_schema_integrity():
    """Validates SQLite table structure and indices."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]
    assert "users" in tables
    assert "transactions" in tables

    cur.execute("SELECT COUNT(*) FROM users")
    assert cur.fetchone()[0] >= 500, "Users table must hold cohort and sandbox users"

    cur.execute("SELECT COUNT(*) FROM transactions")
    assert cur.fetchone()[0] >= 1000, "Transactions table must hold parsed records"
    conn.close()


# ── Suite 2: Personalized Engines ────────────────────────────────────────────
def test_engine_1_persona_and_pca():
    """Validates Engine 1 behavioral feature calculation, persona assignment, and PCA coords."""
    test_users = ["USR_PLAID_GOOD", "USR_PLAID_CUSTOM", "USR0001"]
    for uid in test_users:
        res = analyze_user_persona(uid)
        assert res["user_id"] == uid
        assert res["cluster_id"] in [0, 1, 2, 3, 4]
        assert res["persona_name"] in [
            "Budget-Conscious", "Impulsive Spender", "Heavy Subscriber",
            "Luxury Spender", "Weekend Shopper"
        ]
        assert "x" in res["user_pca"] and "y" in res["user_pca"]
        assert not np.isnan(res["user_pca"]["x"])
        assert len(res["benchmark_comparison"]) >= 5
        for b in res["benchmark_comparison"]:
            assert "metric" in b
            assert 0.0 <= b["user_value"] <= 1.5


def test_engine_2_leakage_detection():
    """Validates Engine 2 recurrence detection, Cost-Per-Use, and Leakage Scoring."""
    res = analyze_user_leakages("USR_PLAID_CUSTOM")
    assert res["user_id"] == "USR_PLAID_CUSTOM"
    assert len(res["subscriptions"]) >= 1, "Must detect recurring subscriptions for custom user"

    for sub in res["subscriptions"]:
        assert sub["monthly_cost"] > 0
        assert sub["cost_per_use"] > 0
        assert 0.0 <= sub["leakage_score"] <= 100.0
        assert sub["status"] in ["Active", "Low Usage", "Dormant", "Duplicate"]
        assert sub["potential_annual_saving"] >= 0.0

    assert res["total_annual_savings"] >= 0.0


def test_engine_3_predictive_cashflow_and_solvency():
    """Validates Engine 3 daily cash-flow trajectory, 30-day projection, and wellness scoring."""
    test_users = ["USR_PLAID_GOOD", "USR0001"]
    for uid in test_users:
        res = analyze_user_cashflow_and_wellness(uid)
        assert res["user_id"] == uid
        assert res["current_liquid_balance"] > 0
        assert res["ensemble_monthly_forecast"] > 0
        assert 0.0 <= res["wellness_score"] <= 100.0
        assert res["risk_level"] in ["Low Risk", "Medium Risk", "High Risk"]
        assert len(res["daily_balance_projection"]) == 30, "Projection must be exactly 30 days"

        for p in res["daily_balance_projection"]:
            assert "date" in p
            assert "projected_balance" in p
            assert "lower_bound" in p
            assert "upper_bound" in p
            assert p["lower_bound"] <= p["upper_bound"]


def test_genai_personalized_advice_format():
    """Validates that Gemini/fallback advice strictly complies with the 3-point PFM format."""
    advice = get_hyper_personalized_pfm_advice(
        user_name="Priya Verma",
        persona_name="Impulsive Spender",
        current_balance=28000.0,
        projected_surplus_or_deficit=-4500.0,
        days_to_shortfall=18,
        leaking_subscriptions=[{"merchant": "Cult.Fit", "monthly_cost": 1999.0, "status": "Dormant"}]
    )

    assert "1. Immediate Action" in advice, "Must include Point 1: Immediate Action"
    assert "2. Mid-Month Budget Guardrail" in advice, "Must include Point 2: Budget Guardrail"
    assert "3. End-of-Month Solvency Strategy" in advice, "Must include Point 3: Solvency Strategy"
    assert "₹" in advice, "Must include rupee-denominated advice"
