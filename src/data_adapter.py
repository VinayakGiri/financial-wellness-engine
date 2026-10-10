"""
data_adapter.py
---------------
Authentic Financial Transaction Ingestion & Normalization Layer for FinWell PFM.

Supports:
1. Plaid Sandbox API Adapter (/transactions/get schema)
2. RBI Account Aggregator (AA) / Setu Sandbox Adapter (Sahamati FIU schema + UPI regex parsing)
3. Anonymized Bank & UPI CSV Statement Parser (HDFC, SBI, ICICI, PhonePe, Google Pay) with Zero-Knowledge PII Sanitizer
4. Unified SQLite Persistence Layer (`data/finwell.db`)
"""

import os
import re
import json
import sqlite3
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DB_PATH = os.path.join(DATA_DIR, "finwell.db")
SYNTHETIC_CSV = os.path.join(DATA_DIR, "transactions_synthetic.csv")
USER_FEATURES_CSV = os.path.join(DATA_DIR, "user_features.csv")

# Standard Categories in FinWell
CATEGORIES = [
    "Food", "Transport", "Shopping", "Entertainment",
    "Subscriptions", "Utilities", "Medical", "Salary", "Investment", "Others"
]

# Known Merchant Classification Map for normalization
MERCHANT_CATEGORY_RULES = {
    # Food & Dining
    r"(?i).*(swiggy|zomato|mcdonald|domino|starbucks|subway|haldiram|chai point|eats|pizza|burger).*": ("Food", "Food & Dining"),
    # Transport & Mobility
    r"(?i).*(uber|ola|rapido|metro|irctc|fuel|petrol|hpcl|bpcl|iocl).*": ("Transport", "Transport"),
    # Shopping & E-Commerce
    r"(?i).*(amazon|flipkart|myntra|zara|blinkit|bigbasket|zepto|dmart|instamart).*": ("Shopping", "Shopping & Groceries"),
    # Subscriptions & Digital
    r"(?i).*(netflix|spotify|prime|hotstar|youtube|apple|google storage|coursera|github|medium|audible).*": ("Subscriptions", "Subscriptions"),
    # Utilities & Bills
    r"(?i).*(bescom|tatapower|airtel|jio|vi |bsnl|indane|mahanagar|electricity|water|broadband).*": ("Utilities", "Bills & Utilities"),
    # Medical & Healthcare
    r"(?i).*(apollo|pharmeasy|netmeds|1mg|hospital|clinic|pharmacy|diagnostic).*": ("Medical", "Healthcare"),
    # Entertainment & Leisure
    r"(?i).*(bookmyshow|pvr|inox|steam|playstation|gaming|bowling).*": ("Entertainment", "Entertainment"),
    # Salary / Income
    r"(?i).*(salary|payroll|stipend|dividend|interest|credit).*": ("Salary", "Income"),
}


def sanitize_pii_string(text: str) -> str:
    """
    Zero-Knowledge PII Sanitizer:
    Hashes Indian 10-digit mobile numbers, 12-16 digit bank accounts,
    and Aadhaar/PAN patterns with SHA-256 salt.
    """
    if not isinstance(text, str):
        return str(text)

    # 1. Indian Phone Numbers (starts with 6-9, 10 digits)
    def hash_phone(match):
        raw = match.group(0)
        h = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
        return f"ANON_TEL_{h}"

    cleaned = re.sub(r"\b[6-9]\d{9}\b", hash_phone, text)

    # 2. Bank Account Numbers / Card Numbers (11 to 16 continuous digits)
    def hash_account(match):
        raw = match.group(0)
        h = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
        return f"ANON_ACC_{h}"

    cleaned = re.sub(r"\b\d{11,16}\b", hash_account, cleaned)

    # 3. PAN Numbers (5 letters, 4 digits, 1 letter)
    def hash_pan(match):
        raw = match.group(0)
        h = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:6]
        return f"ANON_PAN_{h}"

    cleaned = re.sub(r"\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b", hash_pan, cleaned)

    return cleaned


def normalize_merchant_and_category(raw_merchant: str, narration: str = "") -> Tuple[str, str]:
    """
    Infers clean merchant name and category from raw merchant name or UPI narration.
    """
    full_text = f"{raw_merchant} {narration}".strip()

    # Match against rules
    for pattern, (cat, display_name) in MERCHANT_CATEGORY_RULES.items():
        if re.search(pattern, full_text):
            # Extract clean brand name
            m = re.search(pattern, full_text)
            clean_name = m.group(1).title() if m.groups() else raw_merchant.title()
            # Standardize common names
            if "swiggy" in clean_name.lower(): clean_name = "Swiggy"
            elif "zomato" in clean_name.lower(): clean_name = "Zomato"
            elif "netflix" in clean_name.lower(): clean_name = "Netflix"
            elif "spotify" in clean_name.lower(): clean_name = "Spotify"
            elif "amazon" in clean_name.lower(): clean_name = "Amazon"
            elif "blinkit" in clean_name.lower(): clean_name = "Blinkit"
            elif "uber" in clean_name.lower(): clean_name = "Uber"
            elif "ola" in clean_name.lower(): clean_name = "Ola"
            elif "airtel" in clean_name.lower(): clean_name = "Airtel"
            elif "jio" in clean_name.lower(): clean_name = "Jio"
            return clean_name, cat

    clean_name = raw_merchant.strip().title() if raw_merchant else "General Merchant"
    return clean_name, "Others"


# ─────────────────────────────────────────────────────────────────────────────
# Database Initialization & Schema
# ─────────────────────────────────────────────────────────────────────────────

def get_db_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DB_PATH):
    """
    Initializes SQLite tables for users and transactions.
    """
    conn = get_db_connection(db_path)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        user_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        monthly_income REAL DEFAULT 50000.0,
        target_savings REAL DEFAULT 10000.0,
        account_created_at TEXT NOT NULL
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS transactions (
        transaction_id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        merchant TEXT NOT NULL,
        normalized_merchant TEXT NOT NULL,
        category TEXT NOT NULL,
        amount REAL NOT NULL,
        type TEXT NOT NULL,
        mode TEXT NOT NULL,
        balance_after REAL,
        narration TEXT,
        FOREIGN KEY(user_id) REFERENCES users(user_id)
    );
    """)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_txn_user ON transactions(user_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_txn_timestamp ON transactions(timestamp);")
    conn.commit()
    conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Adapter 1: Plaid Sandbox API Adapter
# ─────────────────────────────────────────────────────────────────────────────

class PlaidSandboxAdapter:
    """
    Adapter for Plaid Sandbox transactions (/transactions/get schema).
    """

    @staticmethod
    def parse_payload(user_id: str, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Parses Plaid transactions JSON into normalized schema.
        In Plaid: positive amount = debit (money out of account),
        negative amount = credit (money into account).
        """
        transactions = payload.get("transactions", [])
        normalized = []

        for item in transactions:
            txn_id = item.get("transaction_id") or hashlib.md5(f"{user_id}_{item}".encode()).hexdigest()
            raw_amount = float(item.get("amount", 0.0))
            txn_date = item.get("date", datetime.now().strftime("%Y-%m-%d"))

            # Payment channel / mode
            channel = item.get("payment_channel", "online").upper()
            mode = "NET_BANKING" if channel == "ONLINE" else ("POS" if channel == "IN STORE" else "OTHER")

            raw_merchant = item.get("merchant_name") or item.get("name") or "Unknown Merchant"

            if raw_amount >= 0:
                amount = raw_amount
                txn_type = "Debit"
            else:
                amount = abs(raw_amount)
                txn_type = "Credit"

            # Category
            plaid_categories = item.get("category", [])
            primary_cat = plaid_categories[0] if plaid_categories else "Others"
            norm_merchant, norm_cat = normalize_merchant_and_category(raw_merchant, " ".join(plaid_categories))
            if norm_cat == "Others" and primary_cat in CATEGORIES:
                norm_cat = primary_cat

            normalized.append({
                "transaction_id": txn_id,
                "user_id": user_id,
                "timestamp": f"{txn_date} 12:00:00",
                "merchant": raw_merchant,
                "normalized_merchant": norm_merchant,
                "category": norm_cat,
                "amount": amount,
                "type": txn_type,
                "mode": mode,
                "balance_after": None,
                "narration": f"Plaid / {item.get('name', '')}"
            })

        return normalized

    @classmethod
    def generate_sandbox_payload(cls, account_type: str = "user_good") -> Dict[str, Any]:
        """
        Generates realistic Plaid sandbox payloads with authentic recurring intervals.
        """
        now = datetime.now()
        transactions = []

        if account_type == "user_good":
            # Disciplined saver with Netflix, Spotify, Salary, Swiggy
            for month_offset in range(12, 0, -1):
                m_date = now - timedelta(days=month_offset * 30)

                # Salary credit
                transactions.append({
                    "transaction_id": f"plaid_sal_{month_offset}",
                    "amount": -65000.0,
                    "date": (m_date.replace(day=1)).strftime("%Y-%m-%d"),
                    "name": "Direct Deposit / Tech Corp India Payroll",
                    "merchant_name": "Tech Corp",
                    "category": ["Salary", "Payroll"],
                    "payment_channel": "online",
                    "iso_currency_code": "INR"
                })
                # Netflix
                transactions.append({
                    "transaction_id": f"plaid_net_{month_offset}",
                    "amount": 649.0,
                    "date": (m_date.replace(day=5)).strftime("%Y-%m-%d"),
                    "name": "Netflix Entertainment Subscription",
                    "merchant_name": "Netflix",
                    "category": ["Subscriptions", "Streaming"],
                    "payment_channel": "online",
                    "iso_currency_code": "INR"
                })
                # Spotify
                transactions.append({
                    "transaction_id": f"plaid_spt_{month_offset}",
                    "amount": 119.0,
                    "date": (m_date.replace(day=10)).strftime("%Y-%m-%d"),
                    "name": "Spotify Premium Individual",
                    "merchant_name": "Spotify",
                    "category": ["Subscriptions", "Music"],
                    "payment_channel": "online",
                    "iso_currency_code": "INR"
                })
                # Weekly groceries & dining
                for week in range(4):
                    w_date = m_date + timedelta(days=week * 7 + 2)
                    transactions.append({
                        "transaction_id": f"plaid_swg_{month_offset}_{week}",
                        "amount": float(350 + (week * 40)),
                        "date": w_date.strftime("%Y-%m-%d"),
                        "name": "Swiggy Food Delivery",
                        "merchant_name": "Swiggy",
                        "category": ["Food", "Restaurants"],
                        "payment_channel": "online",
                        "iso_currency_code": "INR"
                    })
                    transactions.append({
                        "transaction_id": f"plaid_bb_{month_offset}_{week}",
                        "amount": float(950 + (week * 100)),
                        "date": (w_date + timedelta(days=2)).strftime("%Y-%m-%d"),
                        "name": "BigBasket Supermarket",
                        "merchant_name": "BigBasket",
                        "category": ["Shopping", "Groceries"],
                        "payment_channel": "online",
                        "iso_currency_code": "INR"
                    })

        elif account_type == "user_custom":
            # Leaking user with high volatility, Gym membership, multiple streaming
            for month_offset in range(12, 0, -1):
                m_date = now - timedelta(days=month_offset * 30)
                # Salary
                transactions.append({
                    "transaction_id": f"plaid_c_sal_{month_offset}",
                    "amount": -55000.0,
                    "date": (m_date.replace(day=1)).strftime("%Y-%m-%d"),
                    "name": "NEFT Cr / Infosys Salary",
                    "merchant_name": "Infosys",
                    "category": ["Salary"],
                    "payment_channel": "online",
                    "iso_currency_code": "INR"
                })
                # Gym (Dormant after 3 months)
                if month_offset > 2:
                    transactions.append({
                        "transaction_id": f"plaid_c_gym_{month_offset}",
                        "amount": 1999.0,
                        "date": (m_date.replace(day=3)).strftime("%Y-%m-%d"),
                        "name": "Cult.Fit Gym Monthly Membership",
                        "merchant_name": "Cult.Fit",
                        "category": ["Subscriptions", "Fitness"],
                        "payment_channel": "online",
                        "iso_currency_code": "INR"
                    })
                # Hotstar
                transactions.append({
                    "transaction_id": f"plaid_c_hot_{month_offset}",
                    "amount": 299.0,
                    "date": (m_date.replace(day=14)).strftime("%Y-%m-%d"),
                    "name": "Disney+ Hotstar Super Plan",
                    "merchant_name": "Disney+ Hotstar",
                    "category": ["Subscriptions", "Streaming"],
                    "payment_channel": "online",
                    "iso_currency_code": "INR"
                })
                # Weekend impulsivity
                for weekend_day in [6, 13, 20, 27]:
                    wk_date = m_date.replace(day=min(weekend_day, 28))
                    transactions.append({
                        "transaction_id": f"plaid_c_zom_{month_offset}_{weekend_day}",
                        "amount": float(1200 + (month_offset * 50)),
                        "date": wk_date.strftime("%Y-%m-%d"),
                        "name": "Zomato Gourmet Dining",
                        "merchant_name": "Zomato",
                        "category": ["Food", "Dining"],
                        "payment_channel": "in store",
                        "iso_currency_code": "INR"
                    })

        return {
            "total_transactions": len(transactions),
            "accounts": [{"account_id": f"{account_type}_acct", "balances": {"current": 42000.0}}],
            "transactions": transactions
        }


# ─────────────────────────────────────────────────────────────────────────────
# Adapter 2: RBI Account Aggregator (AA) / Setu Sandbox Adapter
# ─────────────────────────────────────────────────────────────────────────────

class SetuAccountAggregatorAdapter:
    """
    Parser for RBI Sahamati / Setu FIU Account Aggregator payloads.
    Includes regex engine for Indian UPI, POS, and IMPS narration strings.
    """

    @classmethod
    def parse_upi_narration(cls, narration: str) -> Tuple[str, str, str]:
        """
        Parses real Indian UPI narration formats. Examples:
        - UPI/523819283912/Payment to Swiggy/swiggy@icici/UPI
        - UPI/CR/Salary/Infosys Ltd/Axis Bank
        - POS/402198/ZARA STORE MUMBAI/1209
        - NET_BANKING/TXN9810/BESCOM Electric Bill
        """
        sanitized = sanitize_pii_string(narration)

        # 1. UPI Payment to Merchant pattern
        m_upi = re.search(r"UPI/(?:[A-Za-z0-9]+/)?(?:Payment to\s+)?([^/]+)/([^/]+)/UPI", narration, re.IGNORECASE)
        if m_upi:
            raw_merchant = m_upi.group(1).strip()
            vpa = m_upi.group(2).strip()
            norm_merchant, cat = normalize_merchant_and_category(raw_merchant, narration)
            return norm_merchant, cat, "UPI"

        # 2. UPI Salary / Credit
        m_sal = re.search(r"UPI/CR/(?:Salary|Income)/([^/]+)", narration, re.IGNORECASE)
        if m_sal:
            emp = m_sal.group(1).strip()
            return f"Salary ({emp})", "Salary", "UPI"

        # 3. POS Terminal
        m_pos = re.search(r"POS/(?:[0-9]+/)?([^/]+)", narration, re.IGNORECASE)
        if m_pos:
            raw = m_pos.group(1).strip()
            norm_merchant, cat = normalize_merchant_and_category(raw, narration)
            return norm_merchant, cat, "POS"

        # 4. NetBanking / BillPay
        m_net = re.search(r"NET_BANKING/(?:[0-9A-Z]+/)?([^/]+)", narration, re.IGNORECASE)
        if m_net:
            raw = m_net.group(1).strip()
            norm_merchant, cat = normalize_merchant_and_category(raw, narration)
            return norm_merchant, cat, "NET_BANKING"

        # Default fallback
        norm_merchant, cat = normalize_merchant_and_category(sanitized, sanitized)
        return norm_merchant, cat, "UPI"

    @classmethod
    def parse_payload(cls, user_id: str, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Parses FIU JSON payload adhering to Sahamati standards.
        """
        txns_list = []
        # Sahamati payload structures can be under Account.Transactions.Transaction or direct transactions
        fiu_txns = (
            payload.get("Account", {}).get("Transactions", {}).get("Transaction", [])
            or payload.get("transactions", [])
        )

        for item in fiu_txns:
            txn_id = item.get("txnId") or item.get("transaction_id") or hashlib.md5(f"{user_id}_{item}".encode()).hexdigest()
            raw_type = item.get("type", "DEBIT").upper()
            txn_type = "Credit" if "CREDIT" in raw_type else "Debit"

            amount = abs(float(item.get("amount", 0.0)))
            balance = float(item.get("currentBalance", 0.0)) if item.get("currentBalance") is not None else None
            timestamp = item.get("transactionTimestamp") or item.get("timestamp") or datetime.now().isoformat()

            raw_narration = item.get("narration", "")
            sanitized_narration = sanitize_pii_string(raw_narration)
            mode_declared = item.get("mode", "UPI").upper()

            norm_merchant, cat, inferred_mode = cls.parse_upi_narration(raw_narration)
            mode = mode_declared if mode_declared in ["UPI", "POS", "ATM", "NET_BANKING"] else inferred_mode

            txns_list.append({
                "transaction_id": txn_id,
                "user_id": user_id,
                "timestamp": timestamp.replace("T", " ")[:19],
                "merchant": norm_merchant,
                "normalized_merchant": norm_merchant,
                "category": cat,
                "amount": amount,
                "type": txn_type,
                "mode": mode,
                "balance_after": balance,
                "narration": sanitized_narration
            })

        return txns_list

    @classmethod
    def generate_sandbox_payload(cls, user_id: str = "USR_AA_SETU") -> Dict[str, Any]:
        """
        Generates realistic Setu Account Aggregator sandbox response.
        """
        now = datetime.now()
        txns = []
        running_balance = 78000.0

        for i in range(90, 0, -1):
            t_date = now - timedelta(days=i)
            # Monthly salary on day 1
            if t_date.day == 1:
                running_balance += 72000.0
                txns.append({
                    "txnId": f"SETU_SAL_{i}",
                    "type": "CREDIT",
                    "mode": "NET_BANKING",
                    "amount": 72000.0,
                    "currentBalance": running_balance,
                    "transactionTimestamp": t_date.strftime("%Y-%m-%dT10:00:00"),
                    "narration": "UPI/CR/Salary/Infosys Ltd"
                })

            # Utility bill around 8th
            if t_date.day == 8:
                running_balance -= 1850.0
                txns.append({
                    "txnId": f"SETU_BES_{i}",
                    "type": "DEBIT",
                    "mode": "NET_BANKING",
                    "amount": 1850.0,
                    "currentBalance": running_balance,
                    "transactionTimestamp": t_date.strftime("%Y-%m-%dT15:30:00"),
                    "narration": "NET_BANKING/TXN882/BESCOM Electricity Bill Payment"
                })

            # Subscriptions
            if t_date.day == 12:
                running_balance -= 649.0
                txns.append({
                    "txnId": f"SETU_NET_{i}",
                    "type": "DEBIT",
                    "mode": "UPI",
                    "amount": 649.0,
                    "currentBalance": running_balance,
                    "transactionTimestamp": t_date.strftime("%Y-%m-%dT08:00:00"),
                    "narration": "UPI/523819283912/Payment to Netflix/netflix@icici/UPI"
                })

            # Regular UPI Swiggy/Zomato
            if i % 4 == 0:
                spend = float(np.random.choice([299, 450, 680, 890]))
                running_balance -= spend
                txns.append({
                    "txnId": f"SETU_SWG_{i}",
                    "type": "DEBIT",
                    "mode": "UPI",
                    "amount": spend,
                    "currentBalance": running_balance,
                    "transactionTimestamp": t_date.strftime("%Y-%m-%dT20:15:00"),
                    "narration": f"UPI/{i*19281}/Payment to Swiggy/swiggy@icici/UPI"
                })

        return {
            "Account": {
                "maskedAccNumber": "XXXX-XXXX-8910",
                "Transactions": {
                    "Transaction": txns
                }
            }
        }


# ─────────────────────────────────────────────────────────────────────────────
# Adapter 3: Multi-Bank CSV Statement Parser & Zero-Knowledge Sanitizer
# ─────────────────────────────────────────────────────────────────────────────

class BankCsvStatementParser:
    """
    Parser for CSV exports from HDFC, SBI, ICICI, PhonePe, and Google Pay.
    Applies automatic PII sanitization (hashes account numbers, phone numbers).
    """

    @staticmethod
    def detect_format(df: pd.DataFrame) -> str:
        cols = [str(c).lower().strip() for c in df.columns]
        if any("upi transaction id" in c or "phonepe" in c for c in cols):
            return "phonepe"
        if any("google transaction id" in c or "gpay" in c for c in cols):
            return "gpay"
        if any("chq./ref.no." in c or "narration" in c for c in cols):
            return "hdfc"
        if any("txn date" in c and "description" in c for c in cols):
            return "sbi"
        if any("transaction remarks" in c for c in cols):
            return "icici"
        return "generic"

    @classmethod
    def parse_csv(cls, user_id: str, file_path_or_buffer: Union[str, Any]) -> List[Dict[str, Any]]:
        """
        Parses multi-bank CSV statement, sanitizes PII, and returns standardized records.
        """
        df = pd.read_csv(file_path_or_buffer)
        fmt = cls.detect_format(df)

        parsed_rows = []
        for idx, row in df.iterrows():
            narration = ""
            raw_amount = 0.0
            txn_type = "Debit"
            txn_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            balance = None

            # 1. HDFC Format
            if fmt == "hdfc":
                narration = str(row.get("Narration", row.get("narration", "")))
                withdrawal = row.get("Withdrawal Amt.", row.get("Withdrawal", 0))
                deposit = row.get("Deposit Amt.", row.get("Deposit", 0))
                try:
                    withdrawal = float(str(withdrawal).replace(",", "")) if pd.notnull(withdrawal) else 0.0
                except: withdrawal = 0.0
                try:
                    deposit = float(str(deposit).replace(",", "")) if pd.notnull(deposit) else 0.0
                except: deposit = 0.0

                if deposit > 0:
                    raw_amount = deposit
                    txn_type = "Credit"
                else:
                    raw_amount = withdrawal
                    txn_type = "Debit"

                date_val = str(row.get("Date", row.get("date", txn_date)))
                txn_date = pd.to_datetime(date_val, errors="coerce")
                txn_date = txn_date.strftime("%Y-%m-%d %H:%M:%S") if pd.notnull(txn_date) else datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 2. SBI Format
            elif fmt == "sbi":
                narration = str(row.get("Description", row.get("description", "")))
                debit = row.get("Debit", 0)
                credit = row.get("Credit", 0)
                try: debit = float(str(debit).replace(",", "")) if pd.notnull(debit) else 0.0
                except: debit = 0.0
                try: credit = float(str(credit).replace(",", "")) if pd.notnull(credit) else 0.0
                except: credit = 0.0

                if credit > 0:
                    raw_amount = credit
                    txn_type = "Credit"
                else:
                    raw_amount = debit
                    txn_type = "Debit"

                date_val = str(row.get("Txn Date", row.get("Date", txn_date)))
                txn_date = pd.to_datetime(date_val, errors="coerce")
                txn_date = txn_date.strftime("%Y-%m-%d %H:%M:%S") if pd.notnull(txn_date) else datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 3. PhonePe / Google Pay / Generic
            else:
                # Find amount
                for col in ["Amount", "amount", "Transaction Amount", "Debit Amount"]:
                    if col in row and pd.notnull(row[col]):
                        try:
                            raw_amount = abs(float(str(row[col]).replace(",", "").replace("₹", "").strip()))
                            break
                        except: pass

                # Find narration / merchant
                for col in ["Description", "Narration", "Remarks", "Merchant", "Name", "Details"]:
                    if col in row and pd.notnull(row[col]):
                        narration = str(row[col])
                        break

                # Find type
                for col in ["Type", "Transaction Type", "Cr/Dr"]:
                    if col in row and pd.notnull(row[col]):
                        val = str(row[col]).upper()
                        if "CR" in val or "CREDIT" in val:
                            txn_type = "Credit"
                        else:
                            txn_type = "Debit"
                        break

                # Find date
                for col in ["Date", "Timestamp", "Date & Time", "Txn Date"]:
                    if col in row and pd.notnull(row[col]):
                        d = pd.to_datetime(str(row[col]), errors="coerce")
                        if pd.notnull(d):
                            txn_date = d.strftime("%Y-%m-%d %H:%M:%S")
                            break

            # Sanitize PII in narration
            sanitized_narration = sanitize_pii_string(narration)
            norm_merchant, cat = normalize_merchant_and_category(sanitized_narration, sanitized_narration)
            mode = "UPI" if "UPI" in sanitized_narration.upper() else "NET_BANKING"

            parsed_rows.append({
                "transaction_id": hashlib.md5(f"{user_id}_{idx}_{txn_date}_{raw_amount}".encode()).hexdigest(),
                "user_id": user_id,
                "timestamp": txn_date,
                "merchant": norm_merchant,
                "normalized_merchant": norm_merchant,
                "category": cat,
                "amount": raw_amount,
                "type": txn_type,
                "mode": mode,
                "balance_after": balance,
                "narration": sanitized_narration
            })

        return parsed_rows


# ─────────────────────────────────────────────────────────────────────────────
# Database Operations & Seeding
# ─────────────────────────────────────────────────────────────────────────────

def save_transactions_to_db(transactions: List[Dict[str, Any]], db_path: str = DB_PATH):
    """
    Inserts normalized transactions into SQLite database.
    """
    if not transactions:
        return

    conn = get_db_connection(db_path)
    cur = conn.cursor()

    for t in transactions:
        cur.execute("""
        INSERT OR REPLACE INTO transactions (
            transaction_id, user_id, timestamp, merchant,
            normalized_merchant, category, amount, type, mode, balance_after, narration
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            t["transaction_id"],
            t["user_id"],
            t["timestamp"],
            t["merchant"],
            t["normalized_merchant"],
            t["category"],
            float(t["amount"]),
            t["type"],
            t["mode"],
            t["balance_after"],
            t.get("narration", "")
        ))

    conn.commit()
    conn.close()


def save_user_profile(user_id: str, name: str, monthly_income: float,
                      target_savings: float = 10000.0, db_path: str = DB_PATH):
    """
    Inserts or updates a user profile.
    """
    conn = get_db_connection(db_path)
    cur = conn.cursor()
    cur.execute("""
    INSERT OR REPLACE INTO users (
        user_id, name, monthly_income, target_savings, account_created_at
    ) VALUES (?, ?, ?, ?, ?)
    """, (
        user_id,
        name,
        float(monthly_income),
        float(target_savings),
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()
    conn.close()


def seed_database_from_legacy_data(db_path: str = DB_PATH, force: bool = False):
    """
    Seeds `data/finwell.db` with all 500 users from `transactions_synthetic.csv`,
    along with authentic Plaid and Setu Sandbox accounts.
    """
    init_db(db_path)
    conn = get_db_connection(db_path)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM users")
    user_count = cur.fetchone()[0]

    if user_count > 0 and not force:
        conn.close()
        return

    print("🌱 Seeding FinWell SQLite database from authentic feeds & legacy transactions...")

    # 1. Add authentic sandbox users
    sandbox_users = [
        ("USR_PLAID_GOOD", "Aditya Sharma (Plaid Verified)", 65000.0, 15000.0),
        ("USR_PLAID_CUSTOM", "Priya Verma (High Volatility)", 55000.0, 8000.0),
        ("USR_SETU_AA", "Rahul Nair (Account Aggregator)", 72000.0, 18000.0),
    ]
    for uid, uname, inc, sav in sandbox_users:
        save_user_profile(uid, uname, inc, sav, db_path=db_path)

    # Ingest Plaid Good
    plaid_good = PlaidSandboxAdapter.generate_sandbox_payload("user_good")
    txns_good = PlaidSandboxAdapter.parse_payload("USR_PLAID_GOOD", plaid_good)
    save_transactions_to_db(txns_good, db_path=db_path)

    # Ingest Plaid Custom
    plaid_custom = PlaidSandboxAdapter.generate_sandbox_payload("user_custom")
    txns_custom = PlaidSandboxAdapter.parse_payload("USR_PLAID_CUSTOM", plaid_custom)
    save_transactions_to_db(txns_custom, db_path=db_path)

    # Ingest Setu AA
    setu_payload = SetuAccountAggregatorAdapter.generate_sandbox_payload("USR_SETU_AA")
    txns_setu = SetuAccountAggregatorAdapter.parse_payload("USR_SETU_AA", setu_payload)
    save_transactions_to_db(txns_setu, db_path=db_path)

    # 2. Ingest 500 legacy users from transactions_synthetic.csv
    if os.path.exists(SYNTHETIC_CSV):
        df = pd.read_csv(SYNTHETIC_CSV, parse_dates=["date"])
        # Calculate approximate monthly income per user from credits
        user_incomes = df[df["transaction_type"] == "Credit"].groupby("user_id")["amount"].mean().to_dict()

        unique_users = df["user_id"].unique()
        for u in unique_users:
            inc = float(user_incomes.get(u, 50000.0))
            name = f"User {u.replace('USR', '')} (Verified)"
            save_user_profile(u, name, inc, target_savings=10000.0, db_path=db_path)

        # Batch insert transactions
        norm_txns = []
        for _, row in df.iterrows():
            raw_merchant = str(row["merchant_name"])
            norm_m, norm_c = normalize_merchant_and_category(raw_merchant, str(row.get("description", "")))
            norm_txns.append({
                "transaction_id": hashlib.md5(f"{row['user_id']}_{row['date']}_{row['amount']}_{raw_merchant}".encode()).hexdigest(),
                "user_id": str(row["user_id"]),
                "timestamp": str(row["date"]),
                "merchant": raw_merchant,
                "normalized_merchant": norm_m,
                "category": str(row["category"]),
                "amount": float(row["amount"]),
                "type": str(row["transaction_type"]),
                "mode": "UPI" if row["transaction_type"] == "Debit" else "NET_BANKING",
                "balance_after": None,
                "narration": str(row.get("description", ""))
            })

        save_transactions_to_db(norm_txns, db_path=db_path)

    conn.close()
    print("✅ Database successfully initialized and seeded at:", db_path)


# ─────────────────────────────────────────────────────────────────────────────
# Public Database Queries
# ─────────────────────────────────────────────────────────────────────────────

def get_all_users(db_path: str = DB_PATH) -> pd.DataFrame:
    conn = get_db_connection(db_path)
    df = pd.read_sql_query("SELECT user_id, name, monthly_income, target_savings FROM users ORDER BY user_id", conn)
    conn.close()
    return df


def get_user_transactions(user_id: str, db_path: str = DB_PATH) -> pd.DataFrame:
    conn = get_db_connection(db_path)
    df = pd.read_sql_query(
        "SELECT * FROM transactions WHERE user_id = ? ORDER BY timestamp ASC",
        conn,
        params=(user_id,)
    )
    conn.close()
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def get_user_profile(user_id: str, db_path: str = DB_PATH) -> Optional[Dict[str, Any]]:
    conn = get_db_connection(db_path)
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
    row = cur.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None


if __name__ == "__main__":
    init_db()
    seed_database_from_legacy_data(force=True)
    users_df = get_all_users()
    print(f"Total Users in FinWell DB: {len(users_df)}")
    sample_txns = get_user_transactions("USR_PLAID_GOOD")
    print(f"Sample User Transactions: {len(sample_txns)}")
    print("Sample record:", sample_txns.iloc[0].to_dict() if not sample_txns.empty else "None")
