#!/usr/bin/env python3
"""
scripts/automated_monitor.py
=============================================================================
Automated Financial Wellness Background Monitor & Batch Analyzer
=============================================================================
Fast CLI tool for scheduled background automation (e.g. daily/weekly cron,
GitHub Actions, or webhook worker).

Usage:
    python scripts/automated_monitor.py
    python scripts/automated_monitor.py --statement data/statements/vinayak_gpay_statement_sept2026.pdf
    python scripts/automated_monitor.py --notify
"""

import os
import sys
import glob
import json
import argparse
import urllib.request
import pandas as pd
from datetime import datetime

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_adapter import (
    parse_and_ingest_statement_file,
    get_user_transactions,
    DB_PATH
)
from src.engine_1_behavior import analyze_user_persona
from src.engine_2_leakage import analyze_user_leakages
from src.engine_3_predictive import analyze_user_cashflow_and_wellness


def analyze_user_account(user_id: str = "USR_VINAYAK_GIRI") -> dict:
    """Run full 3-engine analytics directly on the recognized user's database records."""
    df = get_user_transactions(user_id)
    if df.empty:
        return {"user_id": user_id, "status": "EMPTY", "error": "No records found in database"}

    total_txns = len(df)
    type_col = "type" if "type" in df.columns else ("transaction_type" if "transaction_type" in df.columns else None)
    if type_col:
        debit_df = df[df[type_col].astype(str).str.lower() == "debit"]
        credit_df = df[df[type_col].astype(str).str.lower() == "credit"]
    else:
        debit_df = df[df["amount"] < 0]
        credit_df = df[df["amount"] > 0]

    total_debit = float(debit_df["amount"].abs().sum()) if not debit_df.empty else 0.0
    total_credit = float(credit_df["amount"].abs().sum()) if not credit_df.empty else 0.0
    net_flow = total_credit - total_debit

    persona = analyze_user_persona(user_id=user_id, txns_df=df)
    leakage_result = analyze_user_leakages(user_id=user_id, txns_df=df)
    subscriptions = leakage_result.get("subscriptions", [])
    total_leakage_annual = leakage_result.get("total_annual_savings", 0.0)

    wellness_result = analyze_user_cashflow_and_wellness(user_id=user_id, txns_df=df)
    score = wellness_result.get("wellness_score", 65.0)
    health_status = "HEALTHY" if score >= 75 else ("MODERATE" if score >= 55 else "AT_RISK")

    return {
        "file": f"Account Center ({user_id})",
        "user_id": user_id,
        "analyzed_at": datetime.now().isoformat(),
        "transactions_parsed": total_txns,
        "total_debit": round(total_debit, 2),
        "total_credit": round(total_credit, 2),
        "net_flow": round(net_flow, 2),
        "persona": persona.get("persona_name", "Weekend Shopper"),
        "wellness_score": round(score, 1),
        "wellness_grade": wellness_result.get("grade", "A"),
        "health_status": health_status,
        "projected_spend_30d": round(wellness_result.get("ensemble_monthly_forecast", 0.0), 2),
        "is_shortage_expected": wellness_result.get("is_shortage", False),
        "active_recurring_streams": len(subscriptions),
        "annual_leakage_waste": round(total_leakage_annual, 2),
    }


def analyze_statement_file(file_path: str) -> dict:
    """Run unified ingestion and full 3-engine analytics on a statement file."""
    user_id = "AUDIT_" + os.path.splitext(os.path.basename(file_path))[0].replace(" ", "_").upper()
    records = parse_and_ingest_statement_file(user_id=user_id, file_path_or_buffer=file_path, filename=os.path.basename(file_path))

    if not records:
        return {"file": file_path, "status": "EMPTY", "error": "No valid transactions parsed"}

    df = pd.DataFrame(records)
    total_txns = len(df)
    
    type_col = "type" if "type" in df.columns else ("transaction_type" if "transaction_type" in df.columns else None)
    if type_col:
        debit_df = df[df[type_col].astype(str).str.lower() == "debit"]
        credit_df = df[df[type_col].astype(str).str.lower() == "credit"]
    else:
        debit_df = df[df["amount"] < 0]
        credit_df = df[df["amount"] > 0]
    
    total_debit = float(debit_df["amount"].abs().sum()) if not debit_df.empty else 0.0
    total_credit = float(credit_df["amount"].abs().sum()) if not credit_df.empty else 0.0
    net_flow = total_credit - total_debit

    # Engine 1: Behavioral Persona
    persona = analyze_user_persona(user_id=user_id, txns_df=df)

    # Engine 2: Recurring Leakage
    leakage_result = analyze_user_leakages(user_id=user_id, txns_df=df)
    subscriptions = leakage_result.get("subscriptions", [])
    total_leakage_annual = leakage_result.get("total_annual_savings", 0.0)

    # Engine 3: Financial Wellness & 30-Day Solvency
    wellness_result = analyze_user_cashflow_and_wellness(user_id=user_id, txns_df=df)
    score = wellness_result.get("wellness_score", 65.0)
    health_status = "HEALTHY" if score >= 75 else ("MODERATE" if score >= 55 else "AT_RISK")

    return {
        "file": os.path.basename(file_path),
        "user_id": user_id,
        "analyzed_at": datetime.now().isoformat(),
        "transactions_parsed": total_txns,
        "total_debit": round(total_debit, 2),
        "total_credit": round(total_credit, 2),
        "net_flow": round(net_flow, 2),
        "persona": persona.get("persona_name", "Balanced Finizen"),
        "wellness_score": round(score, 1),
        "wellness_grade": wellness_result.get("grade", "B"),
        "health_status": health_status,
        "projected_spend_30d": round(wellness_result.get("ensemble_monthly_forecast", 0.0), 2),
        "is_shortage_expected": wellness_result.get("is_shortage", False),
        "active_recurring_streams": len(subscriptions),
        "annual_leakage_waste": round(total_leakage_annual, 2),
    }


def send_webhook_alert(summary: dict, webhook_url: str):
    """Deliver JSON alert to configured webhook (Slack/Discord/n8n/Zapier)."""
    try:
        payload = json.dumps({
            "text": f"📊 *FinWell Automated Audit*\n"
                    f"• File: `{summary['file']}`\n"
                    f"• Persona: *{summary['persona']}*\n"
                    f"• Wellness Score: *{summary['wellness_score']}/100* ({summary['health_status']})\n"
                    f"• Annual Leakage: *₹{summary['annual_leakage_waste']:,.2f}*\n"
                    f"• Total Outflow: *₹{summary['total_debit']:,.2f}*"
        }).encode("utf-8")
        req = urllib.request.Request(
            webhook_url,
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            print(f"[Webhook] Alert dispatched successfully (Status {resp.status})")
    except Exception as e:
        print(f"[Webhook] Failed to deliver alert: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="FinWell Background Monitor & Statement Batch Analyzer")
    parser.add_argument("--statement", type=str, help="Path to single statement (PDF/CSV/PNG)")
    parser.add_argument("--user", type=str, default=None, help="User ID to audit directly from database (defaults to single recognized user)")
    parser.add_argument("--notify", action="store_true", help="Send alert via ALERT_WEBHOOK_URL env var if defined")
    args = parser.parse_args()

    webhook_url = os.environ.get("ALERT_WEBHOOK_URL")

    # If --user or no statement specified, audit the recognized single user
    if args.user or not args.statement:
        target_uid = args.user or "USR_VINAYAK_GIRI"
        print("=" * 70)
        print(" 🚀 FinWell Single-User Account Center Audit")
        print(f" Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f" Auditing User: {target_uid}")
        print("=" * 70)
        res = analyze_user_account(target_uid)
        print(f"    • User ID           : {res.get('user_id')}")
        print(f"    • Txns Ingested     : {res.get('transactions_parsed')}")
        print(f"    • Total Outflow     : ₹{res.get('total_debit'):,.2f}")
        print(f"    • Total Inflow      : ₹{res.get('total_credit'):,.2f}")
        print(f"    • Net Balance       : ₹{res.get('net_flow'):,.2f}")
        print(f"    • Behavioral Persona: {res.get('persona')}")
        print(f"    • Wellness Score    : {res.get('wellness_score')}/100 [{res.get('health_status')}]")
        print(f"    • Annual Leakage    : ₹{res.get('annual_leakage_waste'):,.2f}")

        if args.notify and webhook_url:
            send_webhook_alert(res, webhook_url)

        if not args.statement:
            print("\n" + "=" * 70)
            print(" ✅ Single-user audit complete.")
            print("=" * 70)
            return

    files = []
    if args.statement:
        files = [args.statement]
    else:
        statements_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "statements"))
        for ext in ("*.pdf", "*.csv", "*.png", "*.jpg"):
            files.extend(glob.glob(os.path.join(statements_dir, ext)))

    if not files:
        print("[!] No statements found to process.")
        return

    print("=" * 70)
    print(" 🚀 FinWell Automated Tool Monitor & Intelligence Pipeline")
    print(f" Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Statements queued: {len(files)}")
    print("=" * 70)

    webhook_url = os.environ.get("ALERT_WEBHOOK_URL")

    for fpath in files:
        print(f"\n[+] Analyzing: {os.path.basename(fpath)}...")
        try:
            res = analyze_statement_file(fpath)
            print(f"    • Txns Ingested     : {res.get('transactions_parsed')}")
            print(f"    • Total Debit       : ₹{res.get('total_debit'):,.2f}")
            print(f"    • Total Credit      : ₹{res.get('total_credit'):,.2f}")
            print(f"    • Net Flow          : ₹{res.get('net_flow'):,.2f}")
            print(f"    • Behavioral Persona: {res.get('persona')}")
            print(f"    • Wellness Score    : {res.get('wellness_score')}/100 [{res.get('health_status')}]")
            print(f"    • Annual Leakage    : ₹{res.get('annual_leakage_waste'):,.2f}")

            if args.notify and webhook_url:
                send_webhook_alert(res, webhook_url)
        except Exception as err:
            print(f"    [!] Error analyzing {fpath}: {err}", file=sys.stderr)

    print("\n" + "=" * 70)
    print(" ✅ Automated audit batch complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
