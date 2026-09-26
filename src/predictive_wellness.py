"""
predictive_wellness.py
----------------------
Forecasts next month's expenses using ARIMA & Prophet, then computes
a Financial Wellness Score (0-100) for a sample of 30 users.

Outputs
-------
outputs/wellness_forecast_baseline.csv  – per-user forecast + wellness metrics
outputs/wellness_distribution.png       – histogram by risk level
outputs/sample_forecasts.png            – 3 sample users: actual + ARIMA + Prophet
models/baseline_forecasts.json          – full forecast records
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
import matplotlib.gridspec as gridspec
import seaborn as sns
from datetime import datetime

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TXN_PATH        = os.path.join(PROJECT_ROOT, "data",    "transactions_synthetic.csv")
FORECAST_CSV    = os.path.join(PROJECT_ROOT, "outputs", "wellness_forecast_baseline.csv")
WELLNESS_HIST   = os.path.join(PROJECT_ROOT, "outputs", "wellness_distribution.png")
SAMPLE_FCST     = os.path.join(PROJECT_ROOT, "outputs", "sample_forecasts.png")
FORECAST_JSON   = os.path.join(PROJECT_ROOT, "models",  "baseline_forecasts.json")

os.makedirs(os.path.join(PROJECT_ROOT, "outputs"), exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, "models"),  exist_ok=True)

SAMPLE_N        = 30
SAVINGS_TARGET  = 10_000   # monthly savings goal (INR)
RANDOM_SEED     = 42

RISK_COLORS = {
    "Low Risk":    "#4CAF50",
    "Medium Risk": "#FFC107",
    "High Risk":   "#F44336",
}

def section(title: str):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)

np.random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load & reshape transactions → monthly series
# ─────────────────────────────────────────────────────────────────────────────
section("1. LOADING & BUILDING MONTHLY TIME SERIES")

df = pd.read_csv(TXN_PATH, parse_dates=["date"])
df["amount"] = df["amount"].astype(float)
df["month"]  = df["date"].dt.to_period("M")

debits  = df[df["transaction_type"] == "Debit"]
credits = df[df["transaction_type"] == "Credit"]

# Monthly expense per user
monthly_expense = (
    debits.groupby(["user_id", "month"])["amount"]
    .sum()
    .reset_index()
    .rename(columns={"amount": "monthly_expense"})
)
# Monthly income per user
monthly_income = (
    credits.groupby(["user_id", "month"])["amount"]
    .sum()
    .reset_index()
    .rename(columns={"amount": "monthly_income"})
)

# Keep users with ≥ 6 months of data (for stable ARIMA)
user_month_counts = monthly_expense.groupby("user_id")["month"].nunique()
eligible_users    = user_month_counts[user_month_counts >= 6].index.tolist()

print(f"  Total transactions    : {len(df):,}")
print(f"  Users with ≥6 months  : {len(eligible_users)}")

# Sample 30 users
np.random.seed(RANDOM_SEED)
sample_users = np.random.choice(eligible_users, size=min(SAMPLE_N, len(eligible_users)),
                                replace=False).tolist()
print(f"  Sampled users         : {len(sample_users)}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Import forecasting libraries
# ─────────────────────────────────────────────────────────────────────────────
section("2. IMPORTING FORECASTING LIBRARIES")

try:
    from pmdarima import auto_arima
    ARIMA_AVAILABLE = True
    print("  pmdarima (auto_arima) : OK")
except ImportError:
    ARIMA_AVAILABLE = False
    print("  pmdarima              : NOT available — using statsmodels ARIMA(1,1,1)")
    from statsmodels.tsa.arima.model import ARIMA as StatsARIMA

try:
    from prophet import Prophet
    import logging
    logging.getLogger("prophet").setLevel(logging.ERROR)
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    PROPHET_AVAILABLE = True
    print("  prophet               : OK")
except ImportError:
    PROPHET_AVAILABLE = False
    print("  prophet               : NOT available — using Holt-Winters fallback")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Forecast helpers
# ─────────────────────────────────────────────────────────────────────────────

def fit_arima(series: np.ndarray):
    """Fit ARIMA and return (forecast, lower_ci, upper_ci)."""
    try:
        if ARIMA_AVAILABLE:
            model = auto_arima(
                series,
                start_p=0, max_p=3,
                start_q=0, max_q=3,
                d=None,
                seasonal=False,
                information_criterion="aic",
                error_action="ignore",
                suppress_warnings=True,
                stepwise=True,
            )
            fc, conf = model.predict(n_periods=1, return_conf_int=True, alpha=0.05)
            return float(fc[0]), float(conf[0][0]), float(conf[0][1])
        else:
            model = StatsARIMA(series, order=(1, 1, 1)).fit(disp=False)
            fc = model.forecast(1)
            se = np.std(series) * 1.96
            return float(fc.iloc[0]), float(fc.iloc[0] - se), float(fc.iloc[0] + se)
    except Exception:
        # Last resort: naive (last value)
        return float(series[-1]), float(series[-1] * 0.9), float(series[-1] * 1.1)


def fit_prophet(series: np.ndarray, month_periods):
    """Fit Prophet and return forecast for month 13."""
    try:
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet not available")

        # Build ds as timestamps from period labels
        ds = pd.PeriodIndex(month_periods).to_timestamp()
        prophet_df = pd.DataFrame({"ds": ds, "y": series})

        m = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode="additive",
            changepoint_prior_scale=0.05,
        )
        m.fit(prophet_df)

        last_ds     = ds[-1]
        future_ds   = pd.date_range(last_ds, periods=2, freq="MS")[1:]
        future_df   = pd.DataFrame({"ds": future_ds})
        forecast    = m.predict(future_df)
        yhat        = float(forecast["yhat"].iloc[0])
        yhat_lower  = float(forecast["yhat_lower"].iloc[0])
        yhat_upper  = float(forecast["yhat_upper"].iloc[0])
        return yhat, yhat_lower, yhat_upper

    except Exception:
        # Fallback: exponential smoothing (simple)
        alpha = 0.3
        smoothed = series[0]
        for val in series[1:]:
            smoothed = alpha * val + (1 - alpha) * smoothed
        se = np.std(series) * 1.96
        return float(smoothed), float(smoothed - se), float(smoothed + se)


def mape(actual: np.ndarray, predicted: float) -> float:
    """Mean Absolute Percentage Error using last known value as actual."""
    if actual[-1] == 0:
        return float("nan")
    return abs(actual[-1] - predicted) / actual[-1] * 100


# ─────────────────────────────────────────────────────────────────────────────
# 4. Run forecasts for sample users
# ─────────────────────────────────────────────────────────────────────────────
section("3. RUNNING FORECASTS  (ARIMA + Prophet, 30 users)")

records     = []
arima_mapes = []
prophet_mapes = []

print(f"\n  {'User':<10}  {'Months':>6}  {'Cur Avg':>9}  "
      f"{'ARIMA':>9}  {'Prophet':>9}  {'Wellness':>9}  {'Risk'}")
print(f"  {'─'*10}  {'─'*6}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*12}")

for uid in sample_users:
    # ── Monthly expense series ────────────────────────────────────────────
    u_exp  = monthly_expense[monthly_expense["user_id"] == uid].sort_values("month")
    u_inc  = monthly_income[monthly_income["user_id"]   == uid].sort_values("month")

    series  = u_exp["monthly_expense"].values.astype(float)
    months  = u_exp["month"].values
    n_months = len(series)

    # ── Income stats ─────────────────────────────────────────────────────
    avg_income  = u_inc["monthly_income"].mean() if len(u_inc) > 0 else 55_000
    current_avg = series.mean()

    # ── ARIMA ────────────────────────────────────────────────────────────
    arima_fc, arima_lo, arima_hi = fit_arima(series)
    arima_fc   = max(0, arima_fc)     # spending can't be negative
    arima_mapes.append(mape(series, arima_fc))

    # ── Prophet ──────────────────────────────────────────────────────────
    prophet_fc, prophet_lo, prophet_hi = fit_prophet(series, months)
    prophet_fc = max(0, prophet_fc)
    prophet_mapes.append(mape(series, prophet_fc))

    # ── Wellness Score  (use ARIMA forecast as primary signal) ───────────
    available_budget = avg_income - SAVINGS_TARGET
    if available_budget <= 0:
        available_budget = avg_income   # edge case: income < savings target

    stress_factor    = (arima_fc / available_budget) * 100
    wellness_score   = float(np.clip(100 - stress_factor, 0, 100))

    # Cash-flow risk: projected spend > (income - savings)
    cash_flow_risk = 1 if arima_fc > available_budget else 0

    # Risk level
    if wellness_score >= 70:
        risk_level = "Low Risk"
    elif wellness_score >= 40:
        risk_level = "Medium Risk"
    else:
        risk_level = "High Risk"

    records.append({
        "user_id":                    uid,
        "n_months_data":              n_months,
        "current_monthly_avg":        round(current_avg, 2),
        "avg_monthly_income":         round(avg_income, 2),
        "forecasted_month13_arima":   round(arima_fc,   2),
        "arima_ci_lower":             round(arima_lo,   2),
        "arima_ci_upper":             round(arima_hi,   2),
        "forecasted_month13_prophet": round(prophet_fc, 2),
        "prophet_ci_lower":           round(prophet_lo, 2),
        "prophet_ci_upper":           round(prophet_hi, 2),
        "wellness_score":             round(wellness_score, 2),
        "cash_flow_risk":             cash_flow_risk,
        "risk_level":                 risk_level,
        # store series for plotting
        "_series":                    series.tolist(),
        "_months":                    [str(m) for m in months],
    })

    print(f"  {uid:<10}  {n_months:>6}  ₹{current_avg:>7,.0f}  "
          f"₹{arima_fc:>7,.0f}  ₹{prophet_fc:>7,.0f}  "
          f"{wellness_score:>8.1f}  {risk_level}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. MAPE summary
# ─────────────────────────────────────────────────────────────────────────────
section("4. MODEL ERROR SUMMARY (MAPE vs. Last Known Month)")

valid_arima   = [x for x in arima_mapes   if not np.isnan(x)]
valid_prophet = [x for x in prophet_mapes if not np.isnan(x)]

print(f"  ARIMA   — mean MAPE : {np.mean(valid_arima):.2f}%  "
      f"(min {np.min(valid_arima):.1f}%  max {np.max(valid_arima):.1f}%)")
print(f"  Prophet — mean MAPE : {np.mean(valid_prophet):.2f}%  "
      f"(min {np.min(valid_prophet):.1f}%  max {np.max(valid_prophet):.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Save forecast CSV
# ─────────────────────────────────────────────────────────────────────────────
section("5. SAVING FORECAST CSV")

output_cols = [
    "user_id", "current_monthly_avg",
    "forecasted_month13_arima", "forecasted_month13_prophet",
    "wellness_score", "cash_flow_risk", "risk_level",
]
result_df = pd.DataFrame(records)[output_cols]
result_df.to_csv(FORECAST_CSV, index=False)
print(f"  Saved -> {FORECAST_CSV}  ({len(result_df)} rows)")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Save JSON (full records minus series arrays)
# ─────────────────────────────────────────────────────────────────────────────
json_records = [{k: v for k, v in r.items() if not k.startswith("_")}
                for r in records]
with open(FORECAST_JSON, "w") as f:
    json.dump(json_records, f, indent=2)
print(f"  Saved -> {FORECAST_JSON}  ({len(json_records)} records)")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Visualization A — Wellness Score Distribution
# ─────────────────────────────────────────────────────────────────────────────
section("6. VISUALIZATIONS")

risk_counts = result_df["risk_level"].value_counts()
ws_values   = result_df["wellness_score"].values
risk_labels = result_df["risk_level"].values

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor("#f8f9fa")

# ── Histogram colored by risk ─────────────────────────────────────────────
ax = axes[0]
ax.set_facecolor("#f8f9fa")

bins = np.linspace(0, 100, 21)
for risk, color in RISK_COLORS.items():
    mask = risk_labels == risk
    if mask.any():
        ax.hist(ws_values[mask], bins=bins, color=color, alpha=0.75,
                label=f"{risk} (n={mask.sum()})", edgecolor="white", linewidth=0.5)

ax.axvline(np.mean(ws_values), color="#333333", linestyle="--", linewidth=1.5,
           label=f"Mean = {np.mean(ws_values):.1f}")
ax.set_xlabel("Financial Wellness Score (0–100)", fontsize=11)
ax.set_ylabel("Number of Users", fontsize=11)
ax.set_title("Wellness Score Distribution\n(30 Sampled Users)", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.spines[["top", "right"]].set_visible(False)

# Add zone bands
ax.axvspan(0,  40, alpha=0.05, color="#F44336")
ax.axvspan(40, 70, alpha=0.05, color="#FFC107")
ax.axvspan(70, 100,alpha=0.05, color="#4CAF50")

# ── Donut: risk distribution ─────────────────────────────────────────────
ax2 = axes[1]
ax2.set_facecolor("#f8f9fa")
ordered_risks = ["Low Risk", "Medium Risk", "High Risk"]
sizes   = [risk_counts.get(r, 0) for r in ordered_risks]
colors  = [RISK_COLORS[r] for r in ordered_risks]
labels  = [f"{r}\n(n={risk_counts.get(r, 0)})" for r in ordered_risks]

non_zero = [(s, c, l) for s, c, l in zip(sizes, colors, labels) if s > 0]
if non_zero:
    sz, co, la = zip(*non_zero)
    wedges, texts, autotexts = ax2.pie(
        sz, labels=la, colors=co,
        autopct="%1.0f%%", startangle=90,
        pctdistance=0.75,
        wedgeprops={"width": 0.5, "edgecolor": "white", "linewidth": 2},
    )
    for t in texts:      t.set_fontsize(9)
    for at in autotexts: at.set_fontsize(9)
ax2.set_title("Risk Level Breakdown", fontsize=12, fontweight="bold")

plt.suptitle("Financial Wellness Engine — Predictive Wellness Scores",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(WELLNESS_HIST, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Wellness distribution saved -> {WELLNESS_HIST}")

# ─────────────────────────────────────────────────────────────────────────────
# 8b. Visualization B — 3 sample user forecasts
# ─────────────────────────────────────────────────────────────────────────────

# Pick 3 users with different risk levels
picks = {}
for rec in records:
    rl = rec["risk_level"]
    if rl not in picks:
        picks[rl] = rec
    if len(picks) == 3:
        break

# Ensure exactly 3 (if some risk levels are missing, fill from records)
while len(picks) < 3:
    for rec in records:
        if rec["user_id"] not in {v["user_id"] for v in picks.values()}:
            picks[f"extra_{len(picks)}"] = rec
            break

sample_recs = list(picks.values())[:3]

fig = plt.figure(figsize=(16, 5))
fig.patch.set_facecolor("#f8f9fa")

for idx, rec in enumerate(sample_recs):
    ax = fig.add_subplot(1, 3, idx + 1)
    ax.set_facecolor("#f8f9fa")

    series   = np.array(rec["_series"])
    n        = len(series)
    x_actual = list(range(1, n + 1))
    x_fc     = n + 1      # month 13

    # Actual line
    ax.plot(x_actual, series, "o-", color="#1976D2", linewidth=2,
            markersize=5, label="Actual", zorder=5)

    # ARIMA forecast
    ax.scatter(x_fc, rec["forecasted_month13_arima"], marker="D",
               color="#F44336", s=80, zorder=6, label="ARIMA forecast")
    ax.plot([n, x_fc],
            [series[-1], rec["forecasted_month13_arima"]],
            "--", color="#F44336", linewidth=1.5, alpha=0.7)
    ax.fill_between([n, x_fc],
                    [series[-1], rec["arima_ci_lower"]],
                    [series[-1], rec["arima_ci_upper"]],
                    color="#F44336", alpha=0.12)

    # Prophet forecast
    ax.scatter(x_fc, rec["forecasted_month13_prophet"], marker="^",
               color="#9C27B0", s=80, zorder=6, label="Prophet forecast")
    ax.plot([n, x_fc],
            [series[-1], rec["forecasted_month13_prophet"]],
            ":", color="#9C27B0", linewidth=1.5, alpha=0.7)

    # Styling
    rl_color = RISK_COLORS.get(rec["risk_level"], "#888888")
    ax.set_title(
        f"{rec['user_id']}  |  WS={rec['wellness_score']:.0f}\n"
        f"{rec['risk_level']}",
        fontsize=10, fontweight="bold", color=rl_color,
    )
    ax.set_xlabel("Month", fontsize=9)
    ax.set_ylabel("Monthly Expense (₹)", fontsize=9)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Annotate forecast values
    ax.annotate(f"₹{rec['forecasted_month13_arima']:,.0f}",
                xy=(x_fc, rec["forecasted_month13_arima"]),
                xytext=(5, 5), textcoords="offset points",
                fontsize=7.5, color="#F44336")

plt.suptitle("Sample User Forecasts — Month 13 (ARIMA vs Prophet)",
             fontsize=13, fontweight="bold", y=1.03)
plt.tight_layout()
plt.savefig(SAMPLE_FCST, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Sample forecasts saved     -> {SAMPLE_FCST}")

# ─────────────────────────────────────────────────────────────────────────────
# 9. Print summary statistics
# ─────────────────────────────────────────────────────────────────────────────
section("7. SUMMARY STATISTICS")

ws = result_df["wellness_score"]
print(f"\n  Wellness Score Statistics ({len(result_df)} users):")
print(f"    Mean   : {ws.mean():.2f}")
print(f"    Median : {ws.median():.2f}")
print(f"    Std    : {ws.std():.2f}")
print(f"    Min    : {ws.min():.2f}")
print(f"    Max    : {ws.max():.2f}")

print(f"\n  Risk Level Distribution:")
for risk, color_key in [("Low Risk","L"), ("Medium Risk","M"), ("High Risk","H")]:
    cnt = (result_df["risk_level"] == risk).sum()
    pct = cnt / len(result_df) * 100
    bar = "█" * int(pct / 4)
    print(f"    {risk:<14}  {cnt:>3}  ({pct:5.1f}%)  {bar}")

print(f"\n  Cash-flow Risk:")
cfr = result_df["cash_flow_risk"].sum()
print(f"    Users at cash-flow risk : {cfr} / {len(result_df)}  "
      f"({cfr/len(result_df)*100:.1f}%)")

print(f"\n  Forecast vs Current Avg:")
diff_arima   = (result_df["forecasted_month13_arima"]   - result_df["current_monthly_avg"]).mean()
diff_prophet = (result_df["forecasted_month13_prophet"] - result_df["current_monthly_avg"]).mean()
print(f"    ARIMA mean delta   : ₹{diff_arima:>+8,.0f} vs current avg")
print(f"    Prophet mean delta : ₹{diff_prophet:>+8,.0f} vs current avg")
print(f"\n  MAPE (vs last known month):")
print(f"    ARIMA   : {np.mean(valid_arima):.2f}%")
print(f"    Prophet : {np.mean(valid_prophet):.2f}%")

print(f"""
  ╔══════════════════════════════════════════════════════════════╗
  ║                                                              ║
  ║   Predictive Wellness Engine — Baseline Forecasting Done     ║
  ║                                                              ║
  ║   outputs/wellness_forecast_baseline.csv  -> {len(result_df)} user forecasts ║
  ║   outputs/wellness_distribution.png       -> score histogram ║
  ║   outputs/sample_forecasts.png            -> 3 user plots    ║
  ║   models/baseline_forecasts.json          -> full JSON store ║
  ║                                                              ║
  ╚══════════════════════════════════════════════════════════════╝
""")
