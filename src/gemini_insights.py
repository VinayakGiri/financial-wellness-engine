"""
gemini_insights.py
------------------
Gemini 1.5 Flash integration for natural-language financial recommendations.

Three public functions:
  1. get_user_wellness_narrative()     — personalized 3-point action plan
  2. get_persona_strategy()            — fintech product strategy per segment
  3. get_leakage_cancellation_script() — push-notification copy per merchant

Fallback templates fire automatically when:
  • GEMINI_API_KEY is unset / placeholder
  • Any API / network error occurs

Outputs:
  outputs/user_narratives/<user_id>.txt
  outputs/persona_strategies/<persona_name>.txt
  outputs/gemini_test_samples.txt
  outputs/gemini_errors.log
  outputs/all_insights_batch.json
"""

import os
import sys
import json
import time
import math
import logging
import textwrap
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR       = os.path.join(PROJECT_ROOT, "data")
OUTPUTS_DIR    = os.path.join(PROJECT_ROOT, "outputs")
MODELS_DIR     = os.path.join(PROJECT_ROOT, "models")

NARRATIVES_DIR  = os.path.join(OUTPUTS_DIR, "user_narratives")
STRATEGIES_DIR  = os.path.join(OUTPUTS_DIR, "persona_strategies")
ERROR_LOG       = os.path.join(OUTPUTS_DIR, "gemini_errors.log")
TEST_SAMPLES    = os.path.join(OUTPUTS_DIR, "gemini_test_samples.txt")
BATCH_JSON      = os.path.join(OUTPUTS_DIR, "all_insights_batch.json")

for d in [NARRATIVES_DIR, STRATEGIES_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Error logger ───────────────────────────────────────────────────────────────
logging.basicConfig(
    filename=ERROR_LOG, filemode="a",
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
err_log = logging.getLogger("gemini")

# ── Load API key ──────────────────────────────────────────────────────────────
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
_PLACEHOLDER   = GEMINI_API_KEY in ("", "paste_your_key_here", "YOUR_KEY_HERE")

# ── Initialise Gemini client (if key available) ───────────────────────────────
_gemini_model = None
if not _PLACEHOLDER:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        _gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        print(f"✅ Gemini 1.5 Flash initialised (key: ...{GEMINI_API_KEY[-4:]})")
    except Exception as e:
        err_log.error(f"Gemini init failed: {e}")
        print(f"⚠️  Gemini init failed: {e}  → fallback mode active")
else:
    print("⚠️  GEMINI_API_KEY not set in .env  → fallback mode active")
    print("   Set your key:  echo 'GEMINI_API_KEY=AIza...' > .env")


# ─────────────────────────────────────────────────────────────────────────────
# Core Gemini call wrapper
# ─────────────────────────────────────────────────────────────────────────────
_api_call_count = 0
_RATE_LIMIT_SLEEP = 1.2   # seconds between calls (free tier: ~60 RPM)

def call_gemini(prompt: str, max_tokens: int = 500) -> str:
    """
    Call Gemini 1.5 Flash.  Returns response text or raises RuntimeError.
    Applies rate-limiting sleep between consecutive calls.
    """
    global _api_call_count
    if _gemini_model is None:
        raise RuntimeError("Gemini model not initialised")

    time.sleep(_RATE_LIMIT_SLEEP)
    _api_call_count += 1
    try:
        response = _gemini_model.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": max_tokens,
                "temperature":       0.7,
                "top_p":             0.9,
            },
        )
        return response.text.strip()
    except Exception as e:
        err_log.error(f"Gemini call #{_api_call_count} failed: {e}")
        raise RuntimeError(str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Fallback templates
# ─────────────────────────────────────────────────────────────────────────────
def _risk_label(score: float) -> str:
    if score < 40:   return "High Risk"
    if score < 70:   return "Medium Risk"
    return "Low Risk"

def _fallback_wellness_narrative(user_id, wellness_score, ensemble_forecast,
                                  user_features, leakage_info) -> str:
    risk   = _risk_label(wellness_score)
    income = user_features.get("avg_monthly_income", 0)
    cat    = ", ".join(user_features.get("top_3_categories", ["Food", "Shopping", "Entertainment"]))
    sav    = user_features.get("savings_rate", 0)
    subs   = leakage_info.get("top_3_leakages", [])
    total  = leakage_info.get("total_annual_savings_if_actioned", 0)

    sub_line = ""
    if subs:
        sub_line = f"Your top leaking subscription is **{subs[0].get('merchant','a service')}** " \
                   f"(₹{subs[0].get('monthly_cost',0):,.0f}/mo). "

    if risk == "Low Risk":
        plan = (
            f"Great news, {user_id}! Your wellness score of {wellness_score:.0f}/100 puts you in a "
            f"strong financial position. Your forecast for next month is ₹{ensemble_forecast:,.0f}.\n\n"
            f"**1. Immediate (this week):** Review your top spending areas — {cat} — "
            f"and set a weekly micro-budget for each.\n"
            f"**2. Short-term (2 weeks):** {sub_line}Cancelling unused services could save "
            f"₹{total:,.0f}/year — redirect that to a recurring deposit.\n"
            f"**3. Long-term habit (next month):** Automate 10% of income (₹{income*0.1:,.0f}) "
            f"into a liquid mutual fund on salary day."
        )
    elif risk == "Medium Risk":
        plan = (
            f"Your wellness score is {wellness_score:.0f}/100 — you're doing okay but there's room "
            f"to strengthen your finances. Forecasted spend: ₹{ensemble_forecast:,.0f}.\n\n"
            f"**1. Immediate (this week):** Pause all non-essential subscriptions for 7 days "
            f"and track where ₹{ensemble_forecast*0.2:,.0f} goes daily.\n"
            f"**2. Short-term (2 weeks):** {sub_line}Action your dormant subscriptions — "
            f"₹{total:,.0f}/year is recoverable.\n"
            f"**3. Long-term habit (next month):** Build a ₹{income*0.5:,.0f} emergency buffer "
            f"(1 month expenses) before increasing lifestyle spend."
        )
    else:
        plan = (
            f"Your wellness score is {wellness_score:.0f}/100 — your forecast (₹{ensemble_forecast:,.0f}) "
            f"is stretching your budget. Immediate action needed.\n\n"
            f"**1. Immediate (this week):** Cancel every dormant subscription today. "
            f"₹{total:,.0f}/year in recoverable leakage is your fastest win.\n"
            f"**2. Short-term (2 weeks):** Set hard limits on {cat.split(',')[0]} spending "
            f"— your highest-impact category.\n"
            f"**3. Long-term habit (next month):** Apply the 50/30/20 rule: 50% needs, "
            f"30% wants, 20% savings. Start tracking every transaction."
        )
    return plan

def _fallback_persona_strategy(persona_name, cluster_stats, top_challenges,
                                leakage_by_persona) -> str:
    size   = cluster_stats.get("size", 0)
    income = cluster_stats.get("avg_income", 0)
    spend  = cluster_stats.get("avg_spend", 0)
    sav    = cluster_stats.get("savings_rate", 0)
    total  = leakage_by_persona.get("total_potential_savings", 0)

    strategies = {
        "Budget-Conscious": (
            "**What makes them unique:** Budget-Conscious users spend mindfully but often "
            "miss higher-yield savings opportunities despite having the discipline for them.\n\n"
            "**Features to build:**\n"
            "  • Auto-invest surplus: sweep end-of-month balance into a liquid fund automatically.\n"
            "  • Savings milestone badges: gamify reaching ₹10K, ₹50K, ₹1L thresholds.\n\n"
            "**Notification strategy:** Weekly 'You saved ₹X more than last week!' push "
            "every Sunday evening — positive reinforcement drives retention for this segment."
        ),
        "Impulsive Spender": (
            "**What makes them unique:** Impulsive Spenders have high income potential but "
            "weekend purchase clusters erode savings before month-end.\n\n"
            "**Features to build:**\n"
            "  • Friday evening 'Weekend Budget' alert showing remaining discretionary balance.\n"
            "  • 'Cool-down timer': 24-hour hold on purchases above ₹2,000 from unknown merchants.\n\n"
            "**Notification strategy:** Thursday afternoon nudge — 'Your weekend budget is ₹X. "
            "Lock in savings before Friday?' with a one-tap FD shortcut."
        ),
        "Heavy Subscriber": (
            "**What makes them unique:** Heavy Subscribers commit to many recurring services "
            "but rarely audit overlap — they're the highest leakage segment.\n\n"
            "**Features to build:**\n"
            "  • Subscription dashboard: visual timeline of all recurring debits with usage scores.\n"
            "  • Duplicate-service detector: flag when 2 merchants serve the same need.\n\n"
            "**Notification strategy:** Monthly 'Subscription Audit Day' on the 1st — "
            "'You're paying for X services. Here are 2 you haven't used in 30 days.'"
        ),
        "Luxury Spender": (
            "**What makes them unique:** Luxury Spenders have the highest avg transaction value "
            "and strong income, but lifestyle inflation creeps up silently.\n\n"
            "**Features to build:**\n"
            "  • Lifestyle inflation tracker: year-on-year comparison of category spend.\n"
            "  • Premium concierge: curated cashback deals at preferred merchants.\n\n"
            "**Notification strategy:** Quarterly 'Net Worth Check-in' — compare asset "
            "growth vs spending growth. Frame as status ('You're in the top 15% of savers')."
        ),
        "Weekend Shopper": (
            "**What makes them unique:** Weekend Shoppers concentrate spending on Fri–Sun, "
            "making month-end budgets unpredictable.\n\n"
            "**Features to build:**\n"
            "  • Weekend spend forecast: 'Based on past 4 weekends, you'll spend ~₹X this weekend.'\n"
            "  • Buy-now-pay-later guardrail: block BNPL activation if monthly spend > 80% of budget.\n\n"
            "**Notification strategy:** Saturday morning check-in — 'Good morning! "
            "Your weekend budget is ₹X. Tap to see last weekend's breakdown.'"
        ),
    }
    return strategies.get(persona_name,
        f"**{persona_name}** ({size} users, avg income ₹{income:,.0f}, savings rate {sav:.1f}%): "
        f"Focus on reducing the ₹{total:,.0f}/year segment-wide leakage through targeted "
        f"subscription audits and automated savings nudges."
    )

def _fallback_cancellation_script(merchant_name, monthly_cost, days_dormant,
                                   cost_per_use, user_count_affected) -> str:
    annual = monthly_cost * 12
    return (
        f"👋 Heads up! You haven't used **{merchant_name}** in {days_dormant} days, "
        f"but it's costing you ₹{monthly_cost:,.0f}/month (₹{annual:,.0f}/year). "
        f"Pause or cancel now to keep that money where it works harder for you."
    )


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC FUNCTION 1 — User Wellness Narrative
# ─────────────────────────────────────────────────────────────────────────────
def get_user_wellness_narrative(
    user_id: str,
    wellness_score: float,
    ensemble_forecast: float,
    user_features: dict,
    leakage_info: dict,
) -> str:
    """
    Returns a 3-point personalized financial action plan (≤150 words).

    Parameters
    ----------
    user_id           : e.g. 'USR0042'
    wellness_score    : float 0-100
    ensemble_forecast : predicted monthly expense for next month (₹)
    user_features     : {avg_monthly_income, savings_rate, spending_volatility,
                          top_3_categories, ...}
    leakage_info      : {top_3_leakages: [{merchant, monthly_cost, status, days_dormant},...],
                          total_annual_savings_if_actioned}
    """
    risk   = _risk_label(wellness_score)
    income = user_features.get("avg_monthly_income", 55_000)
    sav    = user_features.get("savings_rate", 0)
    vol    = user_features.get("spending_volatility", 0)
    cats   = ", ".join(user_features.get("top_3_categories", ["Food", "Shopping", "Transport"]))
    leaks  = leakage_info.get("top_3_leakages", [])
    total_sav = leakage_info.get("total_annual_savings_if_actioned", 0)

    def _leak_line(i):
        if i >= len(leaks): return f"  {i+1}. No leakage detected"
        l = leaks[i]
        return (f"  {i+1}. {l.get('merchant','N/A')}: ₹{l.get('monthly_cost',0):,.0f}/month "
                f"({l.get('status','dormant')} {l.get('days_dormant',0)} days)")

    prompt = textwrap.dedent(f"""
        You are a financial wellness advisor. Generate a personalized action plan for this user:

        Wellness Status: {wellness_score:.0f}/100 ({risk})
        Next Month Forecast: ₹{ensemble_forecast:,.0f}
        Monthly Income: ₹{income:,.0f}
        Savings Rate: {sav:.1f}%
        Spending Volatility: {vol:.2f}
        Top Spending Categories: {cats}

        Highest Leakage Subscriptions:
        {_leak_line(0)}
        {_leak_line(1)}
        {_leak_line(2)}

        Potential Savings: ₹{total_sav:,.0f}/year

        Provide a 3-point action plan:
        1. Immediate action (this week)
        2. Short-term goal (next 2 weeks)
        3. Long-term habit (next month)

        Keep each point to 1-2 sentences, max 150 words total.
        Use clear, actionable language. Be encouraging but realistic. No jargon.
    """).strip()

    try:
        text = call_gemini(prompt, max_tokens=300)
        return text
    except Exception as e:
        err_log.warning(f"[{user_id}] wellness_narrative fallback: {e}")
        return _fallback_wellness_narrative(
            user_id, wellness_score, ensemble_forecast, user_features, leakage_info
        )


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC FUNCTION 2 — Persona Strategy
# ─────────────────────────────────────────────────────────────────────────────
def get_persona_strategy(
    persona_name: str,
    cluster_stats: dict,
    top_challenges: list,
    leakage_by_persona: dict,
) -> str:
    """
    Returns a 3-part fintech product strategy for a user segment (≤120 words).

    Parameters
    ----------
    persona_name       : e.g. 'Impulsive Spender'
    cluster_stats      : {size, avg_income, avg_spend, savings_rate, volatility}
    top_challenges     : [{challenge, severity}, ...]
    leakage_by_persona : {merchants: [...], total_potential_savings}
    """
    size    = cluster_stats.get("size", 0)
    income  = cluster_stats.get("avg_income", 0)
    spend   = cluster_stats.get("avg_spend", 0)
    sav     = cluster_stats.get("savings_rate", 0)
    vol     = cluster_stats.get("volatility", 0)
    total   = leakage_by_persona.get("total_potential_savings", 0)
    merch   = leakage_by_persona.get("merchants", [])[:3]
    chal    = "\n".join(
        f"  - {c.get('challenge','N/A')} (severity: {c.get('severity','medium')})"
        for c in top_challenges[:3]
    ) or "  - High spending volatility (high)\n  - Low savings rate (medium)"

    leaks_str = "\n".join(f"  - {m}" for m in merch) if merch else "  - Various streaming/food services"

    prompt = textwrap.dedent(f"""
        You are a fintech product strategist. Generate a segment strategy for our {persona_name} users.

        Segment Profile:
        - Size: {size} users
        - Avg Monthly Income: ₹{income:,.0f}
        - Avg Monthly Spend: ₹{spend:,.0f}
        - Savings Rate: {sav:.1f}%
        - Spending Volatility: {vol:.2f}

        Top Challenges:
        {chal}

        Common Leakages in This Segment:
        {leaks_str}
        Potential Segment-Wide Savings: ₹{total:,.0f}/year

        Generate a 3-part engagement strategy:
        1. What makes this persona unique (1 sentence)
        2. Top 2 features we should build for them
        3. One personalized notification strategy

        Max 120 words. Be specific and actionable.
    """).strip()

    try:
        text = call_gemini(prompt, max_tokens=250)
        return text
    except Exception as e:
        err_log.warning(f"[{persona_name}] persona_strategy fallback: {e}")
        return _fallback_persona_strategy(
            persona_name, cluster_stats, top_challenges, leakage_by_persona
        )


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC FUNCTION 3 — Leakage Cancellation Script
# ─────────────────────────────────────────────────────────────────────────────
def get_leakage_cancellation_script(
    merchant_name: str,
    monthly_cost: float,
    days_dormant: int,
    cost_per_use: float,
    user_count_affected: int,
) -> str:
    """
    Returns 2-sentence push-notification copy (≤50 words).

    Parameters
    ----------
    merchant_name        : e.g. 'Netflix'
    monthly_cost         : ₹649
    days_dormant         : 45
    cost_per_use         : ₹324
    user_count_affected  : 127
    """
    annual = monthly_cost * 12

    prompt = textwrap.dedent(f"""
        Write a friendly, persuasive 2-sentence notification for users to cancel or reconsider
        their {merchant_name} subscription.

        Context:
        - Monthly Cost: ₹{monthly_cost:,.0f}
        - Last Used: {days_dormant} days ago
        - Cost Per Use: ₹{cost_per_use:,.0f}
        - This affects {user_count_affected} of our users
        - Potential Annual Saving: ₹{annual:,.0f}

        Make it feel like a helpful reminder, not a sales pitch.
        Max 2 sentences, under 50 words. Use ₹ for currency, no asterisks.
    """).strip()

    try:
        text = call_gemini(prompt, max_tokens=100)
        return text
    except Exception as e:
        err_log.warning(f"[{merchant_name}] cancellation_script fallback: {e}")
        return _fallback_cancellation_script(
            merchant_name, monthly_cost, days_dormant, cost_per_use, user_count_affected
        )


# ─────────────────────────────────────────────────────────────────────────────
# Batch runner helpers
# ─────────────────────────────────────────────────────────────────────────────
PERSONA_META = {
    0: {"name": "Budget-Conscious",  "color": "#2196F3"},
    1: {"name": "Impulsive Spender", "color": "#F44336"},
    2: {"name": "Heavy Subscriber",  "color": "#9C27B0"},
    3: {"name": "Luxury Spender",    "color": "#FF9800"},
    4: {"name": "Weekend Shopper",   "color": "#4CAF50"},
}

def _load_project_data():
    """Load all CSVs needed for batch generation."""
    txn      = pd.read_csv(os.path.join(DATA_DIR, "transactions_synthetic.csv"),
                           parse_dates=["date"])
    features = pd.read_csv(os.path.join(DATA_DIR, "user_features.csv"), index_col="user_id")
    clusters = pd.read_csv(os.path.join(DATA_DIR, "user_clusters.csv"))
    leakage  = pd.read_csv(os.path.join(DATA_DIR, "user_leakage.csv"))

    # Try final forecast, fall back to baseline
    fc_path  = os.path.join(OUTPUTS_DIR, "wellness_forecast_final.csv")
    bl_path  = os.path.join(OUTPUTS_DIR, "wellness_forecast_baseline.csv")
    forecast = pd.read_csv(fc_path if os.path.exists(fc_path) else bl_path)

    return txn, features, clusters, leakage, forecast

def _build_user_profile(uid, features, clusters, leakage, forecast, txn):
    """Assemble user_features + leakage_info dicts for one user."""
    fc_row = forecast[forecast["user_id"] == uid]

    # Wellness score
    if len(fc_row):
        ws_col = "wellness_score_ensemble" if "wellness_score_ensemble" in fc_row.columns \
                 else "wellness_score"
        ws = float(fc_row[ws_col].iloc[0])
        ens_col = "ensemble_forecast" if "ensemble_forecast" in fc_row.columns \
                  else "forecasted_month13_arima"
        ef = float(fc_row[ens_col].iloc[0])
    else:
        ws = 92.0
        ef = float(txn[txn["user_id"] == uid][txn["transaction_type"] == "Debit"]["amount"].mean()
                   if uid in txn["user_id"].values else 3500)

    # Income from credits
    credit_txn = txn[(txn["user_id"] == uid) & (txn["transaction_type"] == "Credit")]
    income     = float(credit_txn["amount"].mean()) if len(credit_txn) else 55_000

    # Top 3 debit categories
    debit_txn  = txn[(txn["user_id"] == uid) & (txn["transaction_type"] == "Debit")]
    top_cats   = (debit_txn.groupby("category")["amount"].sum()
                  .nlargest(3).index.tolist())

    # Feature row
    fr = features.loc[uid] if uid in features.index else {}
    sav  = float(fr.get("savings_ratio", 0.3)) * 100
    vol  = float(fr.get("spending_volatility", 0.3))

    user_feat = {
        "avg_monthly_income":  round(income, 2),
        "savings_rate":        round(sav, 1),
        "spending_volatility": round(vol, 3),
        "top_3_categories":    top_cats,
    }

    # Top 3 leakages for this user
    u_leak = (leakage[leakage["user_id"] == uid]
              .sort_values("leakage_score", ascending=False)
              .head(3))
    top3 = []
    for _, row in u_leak.iterrows():
        top3.append({
            "merchant":     row["merchant_name"],
            "monthly_cost": round(float(row["monthly_amount"]), 2),
            "status":       row["status"],
            "days_dormant": int(row.get("dormancy_days", 0)),
        })
    total_sav = float(u_leak["estimated_annual_saving"].sum()) \
                if "estimated_annual_saving" in u_leak.columns else 0.0

    leak_info = {"top_3_leakages": top3, "total_annual_savings_if_actioned": round(total_sav, 2)}
    return ws, ef, user_feat, leak_info

def _build_persona_profile(cid, clusters, features, leakage, txn):
    """Assemble cluster_stats + challenge + leakage dicts for one persona."""
    pname  = PERSONA_META[cid]["name"]
    uids   = clusters[clusters["cluster_label"] == cid]["user_id"].tolist()
    size   = len(uids)

    # Income & spend
    credits = txn[(txn["user_id"].isin(uids)) & (txn["transaction_type"] == "Credit")]
    debits  = txn[(txn["user_id"].isin(uids)) & (txn["transaction_type"] == "Debit")]
    income  = float(credits["amount"].mean()) if len(credits) else 55_000
    spend   = float(debits.groupby("user_id")["amount"].sum().mean()) / 12 \
              if len(debits) else 3500

    # Feature means
    pf      = features[features.index.isin(uids)]
    sav     = float(pf["savings_ratio"].mean()) * 100 if "savings_ratio" in pf else 30.0
    vol     = float(pf["spending_volatility"].mean()) if "spending_volatility" in pf else 0.3

    cluster_stats = {"size": size, "avg_income": round(income,2),
                     "avg_spend": round(spend,2), "savings_rate": round(sav,1),
                     "volatility": round(vol,3)}

    # Challenges derived from feature values
    challenges = []
    if vol > 0.5:
        challenges.append({"challenge": "High spending volatility", "severity": "high"})
    if sav < 30:
        challenges.append({"challenge": "Below-average savings rate", "severity": "high"})
    if "weekend_spending_ratio" in pf.columns:
        wknd = float(pf["weekend_spending_ratio"].mean())
        if wknd > 0.45:
            challenges.append({"challenge": "Weekend spending spikes", "severity": "medium"})
    if "recurring_payment_count" in pf.columns:
        rpc = float(pf["recurring_payment_count"].mean())
        if rpc > 0.5:
            challenges.append({"challenge": "High recurring payment burden", "severity": "medium"})
    if not challenges:
        challenges.append({"challenge": "Opportunity to increase savings rate", "severity": "low"})

    # Persona leakage
    p_leak = leakage[leakage["user_id"].isin(uids)]
    merch  = p_leak.groupby("merchant_name")["estimated_annual_saving"].sum() \
                    .nlargest(3).index.tolist() if len(p_leak) else []
    total  = float(p_leak["estimated_annual_saving"].sum()) if len(p_leak) else 0.0

    return pname, cluster_stats, challenges, {"merchants": merch,
                                               "total_potential_savings": round(total, 2)}


# ─────────────────────────────────────────────────────────────────────────────
# Main batch runner
# ─────────────────────────────────────────────────────────────────────────────
def run_batch(
    max_users: int  = 500,
    sample_users: int = 30,     # call Gemini for first N users; rest use fast fallback
):
    """
    Generate narratives for up to max_users users and all 5 personas.
    When Gemini is available, first `sample_users` get live API calls;
    the rest receive fast fallback templates (identical structure, offline).
    """
    print(f"\n{'='*65}")
    print("  BATCH INSIGHT GENERATION")
    print(f"{'='*65}")

    txn, features, clusters, leakage, forecast = _load_project_data()
    all_users = clusters["user_id"].tolist()[:max_users]

    batch_out = {"user_insights": {}, "persona_strategies": {}, "leakage_messages": {}}
    api_mode  = _gemini_model is not None

    # ── 1. User narratives ────────────────────────────────────────────────────
    print(f"\n  [1/3] Generating user wellness narratives ({len(all_users)} users)…")
    live_calls = 0

    for i, uid in enumerate(all_users):
        try:
            ws, ef, uf, li = _build_user_profile(uid, features, clusters,
                                                   leakage, forecast, txn)
        except Exception as e:
            err_log.error(f"Profile build failed for {uid}: {e}")
            ws, ef, uf, li = (88.0, 3500.0,
                               {"avg_monthly_income": 55000, "savings_rate": 35.0,
                                "spending_volatility": 0.3, "top_3_categories": ["Food","Transport","Shopping"]},
                               {"top_3_leakages": [], "total_annual_savings_if_actioned": 0})

        # Use live Gemini only for first `sample_users`
        if api_mode and live_calls < sample_users:
            narrative = get_user_wellness_narrative(uid, ws, ef, uf, li)
            live_calls += 1
        else:
            narrative = _fallback_wellness_narrative(uid, ws, ef, uf, li)

        # Save individual file
        with open(os.path.join(NARRATIVES_DIR, f"{uid}.txt"), "w") as f:
            f.write(narrative)
        batch_out["user_insights"][uid] = {
            "wellness_score":     round(ws, 1),
            "ensemble_forecast":  round(ef, 2),
            "narrative":          narrative,
            "source":             "gemini" if (api_mode and live_calls <= sample_users) else "fallback",
        }

        if (i + 1) % 50 == 0 or i == 0:
            src = f"({live_calls} via Gemini API)" if api_mode else "(fallback mode)"
            print(f"    {i+1:>3}/{len(all_users)}  {src}")

    print(f"  ✅ {len(all_users)} user narratives generated "
          f"({live_calls} live API, {len(all_users)-live_calls} fallback)")

    # ── 2. Persona strategies ─────────────────────────────────────────────────
    print(f"\n  [2/3] Generating persona strategies (5 personas)…")
    for cid in range(5):
        pname, cs, chals, pl = _build_persona_profile(cid, clusters, features, leakage, txn)
        strategy = get_persona_strategy(pname, cs, chals, pl)

        safe_name = pname.replace(" ", "_")
        with open(os.path.join(STRATEGIES_DIR, f"{safe_name}.txt"), "w") as f:
            f.write(strategy)
        batch_out["persona_strategies"][pname] = {
            "cluster_id":      cid,
            "cluster_stats":   cs,
            "top_challenges":  chals,
            "strategy":        strategy,
        }
        src = "gemini" if api_mode else "fallback"
        print(f"    ✅ {pname}  [{src}]")

    # ── 3. Leakage cancellation scripts (top 10 merchants) ───────────────────
    print(f"\n  [3/3] Generating leakage cancellation scripts (top 10 merchants)…")
    top_merchants = (
        leakage.groupby("merchant_name")
        .agg(avg_cost=("monthly_amount", "mean"),
             avg_dormant=("dormancy_days", "mean"),
             avg_cpu=("cost_per_use", "mean"),
             user_count=("user_id", "nunique"))
        .sort_values("avg_cpu", ascending=False)
        .head(10)
    )

    for merchant, row in top_merchants.iterrows():
        script = get_leakage_cancellation_script(
            merchant_name       = merchant,
            monthly_cost        = float(row["avg_cost"]),
            days_dormant        = int(row["avg_dormant"]),
            cost_per_use        = float(row["avg_cpu"]),
            user_count_affected = int(row["user_count"]),
        )
        batch_out["leakage_messages"][merchant] = {
            "monthly_cost":  round(float(row["avg_cost"]), 2),
            "users_affected":int(row["user_count"]),
            "script":        script,
        }
        print(f"    ✅ {merchant:<22} ({int(row['user_count'])} users)")

    # ── Save batch JSON ───────────────────────────────────────────────────────
    with open(BATCH_JSON, "w", encoding="utf-8") as f:
        json.dump(batch_out, f, indent=2, ensure_ascii=False)

    print(f"\n  Batch JSON saved → {BATCH_JSON}")
    print(f"  API calls made   : {_api_call_count}")
    return batch_out


# ─────────────────────────────────────────────────────────────────────────────
# main — test block
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    def section(t):
        print(f"\n{'='*65}\n  {t}\n{'='*65}")

    section("GEMINI INSIGHT LAYER — FUNCTION TESTS")

    # ── Sample data ───────────────────────────────────────────────────────────
    sample_uid      = "USR0042"
    sample_ws       = 78.5
    sample_ef       = 4_320.0
    sample_features = {
        "avg_monthly_income":  52_000,
        "savings_rate":        38.2,
        "spending_volatility": 0.42,
        "top_3_categories":    ["Food", "Shopping", "Entertainment"],
    }
    sample_leakage  = {
        "top_3_leakages": [
            {"merchant": "Netflix",  "monthly_cost": 649, "status": "Dormant",   "days_dormant": 67},
            {"merchant": "Spotify",  "monthly_cost": 119, "status": "Low Usage", "days_dormant": 12},
            {"merchant": "Gym Pass", "monthly_cost": 999, "status": "Dormant",   "days_dormant": 91},
        ],
        "total_annual_savings_if_actioned": 21_804,
    }

    test_lines = []

    # ── Test 1: User Wellness Narrative ───────────────────────────────────────
    print(f"\n  🧪 Test 1 — get_user_wellness_narrative({sample_uid})")
    t0  = time.time()
    n1  = get_user_wellness_narrative(sample_uid, sample_ws, sample_ef,
                                      sample_features, sample_leakage)
    elapsed = time.time() - t0
    words   = len(n1.split())
    status  = "✅ PASS" if (len(n1) > 30 and words <= 200) else "❌ FAIL"
    print(f"  {status}  words={words}  elapsed={elapsed:.2f}s")
    print(f"\n{n1}\n")
    assert len(n1) > 30, "Narrative too short"
    test_lines.extend([f"=== Test 1: User Wellness Narrative [{sample_uid}] ===", n1, ""])

    # ── Test 2: Persona Strategy ──────────────────────────────────────────────
    print(f"\n  🧪 Test 2 — get_persona_strategy('Impulsive Spender')")
    t0  = time.time()
    n2  = get_persona_strategy(
        persona_name = "Impulsive Spender",
        cluster_stats = {"size": 161, "avg_income": 54_200, "avg_spend": 4_800,
                         "savings_rate": 38.5, "volatility": 0.61},
        top_challenges = [
            {"challenge": "Weekend spending spikes", "severity": "high"},
            {"challenge": "High impulse transaction count", "severity": "high"},
            {"challenge": "Below-average savings rate", "severity": "medium"},
        ],
        leakage_by_persona = {
            "merchants": ["Zomato", "Swiggy", "Amazon"],
            "total_potential_savings": 1_84_000,
        },
    )
    elapsed = time.time() - t0
    words   = len(n2.split())
    status  = "✅ PASS" if (len(n2) > 30 and words <= 160) else "❌ FAIL"
    print(f"  {status}  words={words}  elapsed={elapsed:.2f}s")
    print(f"\n{n2}\n")
    assert len(n2) > 30, "Strategy too short"
    test_lines.extend(["=== Test 2: Persona Strategy [Impulsive Spender] ===", n2, ""])

    # ── Test 3: Cancellation Script ───────────────────────────────────────────
    print(f"\n  🧪 Test 3 — get_leakage_cancellation_script('Netflix')")
    t0  = time.time()
    n3  = get_leakage_cancellation_script(
        merchant_name       = "Netflix",
        monthly_cost        = 649,
        days_dormant        = 67,
        cost_per_use        = 324.5,
        user_count_affected = 127,
    )
    elapsed = time.time() - t0
    words   = len(n3.split())
    status  = "✅ PASS" if (len(n3) > 10 and words <= 70) else "❌ FAIL"
    print(f"  {status}  words={words}  elapsed={elapsed:.2f}s")
    print(f"\n{n3}\n")
    assert len(n3) > 10, "Cancellation script too short"
    test_lines.extend(["=== Test 3: Cancellation Script [Netflix] ===", n3, ""])

    # ── Save test samples ─────────────────────────────────────────────────────
    with open(TEST_SAMPLES, "w", encoding="utf-8") as f:
        f.write("GEMINI INSIGHT LAYER — TEST SAMPLES\n")
        f.write(f"Mode: {'LIVE Gemini API' if _gemini_model else 'Fallback templates'}\n")
        f.write(f"API calls made: {_api_call_count}\n")
        f.write("=" * 65 + "\n\n")
        f.write("\n\n".join(test_lines))
    print(f"\n  ✅ Test samples saved → {TEST_SAMPLES}")

    # ── Run full batch ─────────────────────────────────────────────────────────
    section("BATCH GENERATION — 500 USERS + 5 PERSONAS + 10 MERCHANTS")
    batch = run_batch(max_users=500, sample_users=30)

    n_users    = len(batch["user_insights"])
    n_personas = len(batch["persona_strategies"])
    n_merch    = len(batch["leakage_messages"])

    section("FINAL STATUS")
    print(f"""
  ╔══════════════════════════════════════════════════════════════╗
  ║                                                              ║
  ║   Gemini Insight Layer Complete                              ║
  ║                                                              ║
  ║   Mode          : {'Live Gemini API' if _gemini_model else 'Fallback templates (set GEMINI_API_KEY)':<38}║
  ║   API calls     : {_api_call_count:<38}║
  ║   Users         : {n_users:<38}║
  ║   Personas      : {n_personas:<38}║
  ║   Merchants     : {n_merch:<38}║
  ║                                                              ║
  ║   outputs/user_narratives/      → {n_users} .txt files{'':<14}║
  ║   outputs/persona_strategies/   → {n_personas} .txt files{'':<15}║
  ║   outputs/gemini_test_samples.txt                            ║
  ║   outputs/all_insights_batch.json                            ║
  ║   outputs/gemini_errors.log                                  ║
  ║                                                              ║
  ╚══════════════════════════════════════════════════════════════╝

  Gemini Insight Layer Complete — Generated narratives for
  {n_users} users + {n_personas} personas + {n_merch} merchants
""")
