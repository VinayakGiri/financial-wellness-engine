# 💰 Financial Wellness Intelligence Platform
### Full Project Documentation — Review 2

> **Understand. Detect. Predict. Recommend.**

---

## 📋 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [Quick Start](#3-quick-start)
4. [Project Structure](#4-project-structure)
5. [Engine 1 — User Segmentation](#5-engine-1--user-segmentation)
6. [Engine 2 — Leakage Detection](#6-engine-2--leakage-detection)
7. [Engine 3 — Wellness Forecast](#7-engine-3--wellness-forecast)
8. [Streamlit Dashboard](#8-streamlit-dashboard)
9. [Testing Suite](#9-testing-suite)
10. [EDA Notebook](#10-eda-notebook)
11. [Data Dictionary](#11-data-dictionary)
12. [Model Artifacts](#12-model-artifacts)
13. [Output Files](#13-output-files)
14. [Key Metrics & Results](#14-key-metrics--results)
15. [Known Limitations & Next Steps](#15-known-limitations--next-steps)
16. [Dependencies](#16-dependencies)

---

## 1. Project Overview

The **Financial Wellness Intelligence Platform** is an end-to-end personal finance analytics system that ingests synthetic transaction data for 500 users across 12 months and drives three tightly integrated ML engines:

| Engine | Problem | Approach | Output |
|--------|---------|----------|--------|
| 🎯 **Engine 1** | Who are my users? | K-Means Clustering | 5 Spending Personas |
| 🔍 **Engine 2** | What are they wasting? | Rule-based scoring + Random Forest | Leakage Scores + Annual Savings |
| 📈 **Engine 3** | What will they spend next month? | ARIMA + Facebook Prophet | Wellness Score (0–100) |

All three engines feed a **Streamlit dashboard** with Plotly interactive charts, and are validated by a **62-test automated suite**.

---

## 2. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                  RAW DATA LAYER                                 │
│  transactions_synthetic.csv  (10,000 rows · 500 users · 12mo)  │
└────────────────────────────┬────────────────────────────────────┘
                             │
            ┌────────────────┼────────────────┐
            ▼                ▼                ▼
   ┌────────────────┐ ┌──────────────┐ ┌──────────────┐
   │  feature_      │ │  leakage_    │ │  predictive_ │
   │  engineering   │ │  detection   │ │  wellness    │
   │  .py           │ │  .py         │ │  .py         │
   └───────┬────────┘ └──────┬───────┘ └──────┬───────┘
           │                 │                │
           ▼                 ▼                ▼
   user_features.csv  recurring_txns.csv  forecast_baseline.csv
           │           user_leakage.csv         │
           │                 │                  │
           ▼                 ▼                  │
   ┌────────────────┐ ┌──────────────┐          │
   │  clustering.py │ │  leakage_    │          │
   │  (K-Means k=5) │ │  classifier  │          │
   └───────┬────────┘ │  .py (RF)    │          │
           │          └──────┬───────┘          │
           ▼                 ▼                  ▼
   user_clusters.csv  leakage_clf.pkl    wellness_score
   kmeans_model.pkl   leakage_scaler.pkl      (0–100)
           │                 │                  │
           └────────────┬────┘                  │
                        ▼                       ▼
              ┌─────────────────────────────────────┐
              │       dashboard/app.py               │
              │       Streamlit + Plotly              │
              │       http://localhost:8501           │
              └─────────────────────────────────────┘
                        │
              ┌─────────┴──────────┐
              ▼                    ▼
     tests/test_models.py   notebooks/01_eda_and_results.ipynb
     (62 tests — all PASS)  (6-cell EDA + presentation figures)
```

---

## 3. Quick Start

### 🌐 Live Production Cloud App
**Direct URL:** [https://financial-wellness-engine-oqbxbgpzsgahzkvg5x8sm2.streamlit.app](https://financial-wellness-engine-oqbxbgpzsgahzkvg5x8sm2.streamlit.app)

### Local Environment
```bash
# 1. Navigate to project root
cd /Users/vinayakgiri/.gemini/antigravity/scratch/financial-wellness-engine

# 2. Activate virtual environment
source venv/bin/activate

# 3. Launch the dashboard
streamlit run dashboard/app.py

# 4. Run the full test suite
python tests/test_models.py

# 5. Re-run any engine from scratch
python src/generate_data.py          # Prompt 2 — Synthetic data
python src/feature_engineering.py   # Prompt 4 — Feature matrix
python src/clustering.py            # Prompt 5 — Personas
python src/persona_insights.py      # Prompt 6 — AI insights
python src/leakage_detection.py     # Prompt 7 — Leakage scores
python src/leakage_classifier.py    # Prompt 8 — RF classifier
python src/predictive_wellness.py   # Prompt 9 — Forecasts

# 6. Execute EDA notebook
jupyter nbconvert --to notebook --execute --inplace \
    notebooks/01_eda_and_results.ipynb
```

> **Python version:** 3.14.2 (arm64 macOS)  
> **TensorFlow:** Incompatible with Python 3.14 — treated as optional, all other 13 packages installed.

---

## 4. Project Structure

```
financial-wellness-engine/
│
├── data/
│   ├── raw/                         # gitignored raw inputs
│   ├── transactions_synthetic.csv   # 10,000 rows — source of truth
│   ├── user_features.csv            # 500 × 20 normalized features
│   ├── user_clusters.csv            # 500 users → cluster 0-4 + persona name
│   ├── recurring_transactions.csv   # 276 recurring (user, merchant) pairs
│   └── user_leakage.csv             # 276 rows — leakage scores & status
│
├── src/
│   ├── generate_data.py             # Synthetic dataset generator (Faker)
│   ├── verify_setup.py              # Environment & data validation
│   ├── feature_engineering.py       # 20 behavioral features → user_features.csv
│   ├── clustering.py                # K-Means k=5 → personas + PCA scatter
│   ├── persona_insights.py          # Statistical + Gemini-powered descriptions
│   ├── leakage_detection.py         # Recurring payment detection + scoring
│   ├── leakage_classifier.py        # Random Forest leakage predictor
│   └── predictive_wellness.py       # ARIMA + Prophet → wellness scores
│
├── dashboard/
│   └── app.py                       # Streamlit multi-view app (4 views)
│
├── notebooks/
│   └── 01_eda_and_results.ipynb     # 6-cell EDA + presentation figures
│
├── models/
│   ├── kmeans_model.pkl             # KMeans(n_clusters=5, random_state=42)
│   ├── leakage_classifier.pkl       # RandomForest(n=100, depth=10)
│   ├── leakage_scaler.pkl           # StandardScaler fitted on leakage features
│   └── baseline_forecasts.json      # Full ARIMA + Prophet records (30 users)
│
├── outputs/
│   ├── correlation_matrix.png       # Feature correlation heatmap
│   ├── elbow_curve.png              # Elbow + silhouette (k=2–8)
│   ├── personas_scatter.png         # PCA 2D cluster scatter
│   ├── top_leakages.png             # Top-20 leakage bar + donut
│   ├── leakage_cm.png               # Confusion matrix (counts + %)
│   ├── feature_importance.png       # RF feature importance bars
│   ├── wellness_distribution.png    # Wellness histogram + risk donut
│   ├── sample_forecasts.png         # 3 user actual + ARIMA + Prophet
│   ├── persona_insights.json        # AI summaries per persona
│   ├── leakage_features.json        # Top-5 RF feature importances
│   ├── personas_summary.csv         # 5 rows — persona stats + AI text
│   ├── wellness_forecast_baseline.csv  # 30 rows — forecasts + wellness
│   ├── test_report.txt              # Full 62-test pass/fail report
│   ├── eda_spending_analysis.png    # Cell 2 output
│   ├── eda_persona_breakdown.png    # Cell 3 output
│   ├── eda_leakage_analysis.png     # Cell 4 output
│   ├── eda_wellness_distribution.png # Cell 5 output
│   └── eda_key_findings.png         # Cell 6 — dark summary card
│
├── tests/
│   └── test_models.py               # 7 suites, 62 tests, all PASS
│
├── .env                             # GEMINI_API_KEY (gitignored)
├── .gitignore
├── requirements.txt
└── venv/                            # Python 3.14.2 virtual environment
```

---

## 5. Engine 1 — User Segmentation

**Script:** [`src/clustering.py`](../src/clustering.py)  
**Input:** `data/user_features.csv`  
**Outputs:** `data/user_clusters.csv` · `models/kmeans_model.pkl` · `outputs/elbow_curve.png` · `outputs/personas_scatter.png`

### Feature Engineering (src/feature_engineering.py)

20 behavioral features computed per user across 4 groups:

| Group | Features |
|-------|---------|
| **Spending** | `total_spending`, `avg_monthly_spend`, `std_monthly_spend`, `max_monthly_spend` |
| **Ratios** | `food_ratio`, `entertainment_ratio`, `essential_ratio`, `discretionary_ratio`, `savings_ratio` |
| **Temporal** | `weekend_spending_ratio`, `weekday_spending_ratio`, `salary_day_behavior`, `spending_volatility` |
| **Subscription** | `recurring_payment_count`, `subscription_total_monthly`, `unique_merchants_count`, `subscription_to_income_ratio` |
| **Impulse** | `impulse_transaction_count`, `avg_transaction_amount`, `transaction_frequency` |

> Normalization pipeline: **StandardScaler → MinMaxScaler** → all features ∈ [0, 1]

### Clustering Results

| Cluster | Persona | Users | Share | Avg Monthly Spend | Avg Savings |
|---------|---------|-------|-------|-------------------|-------------|
| 0 | 💰 Budget-Conscious | 100 | 20.0% | 0.621 | 0.378 |
| 1 | 🔥 Impulsive Spender | 161 | **32.2%** | 0.356 | 0.531 |
| 2 | 📺 Heavy Subscriber | 85 | 17.0% | 0.209 | **0.708** |
| 3 | 💎 Luxury Spender | 81 | 16.2% | 0.406 | 0.490 |
| 4 | 🛍️ Weekend Shopper | 73 | 14.6% | 0.281 | 0.399 |

**Model parameters:** `KMeans(n_clusters=5, n_init=20, random_state=42)`  
**Silhouette score:** 0.1166 (moderate — expected for synthetic data with 20 features)  
**PCA variance explained:** PC1=28.1%, PC2=13.4% → 41.5% total

### Persona Descriptions

| Persona | Key Trait | Primary Recommendation |
|---------|-----------|----------------------|
| 💰 Budget-Conscious | High discretionary spend despite appearing frugal; strong savings potential | Channel savings into SIP/RD for long-term wealth |
| 🔥 Impulsive Spender | Elevated weekend spending; frequent large single transactions | Set weekend spend caps + ₹2,000+ transaction alerts |
| 📺 Heavy Subscriber | Lowest spend group; overlapping streaming/lifestyle services | Monthly subscription audit — save ₹500–₹1,500/mo |
| 💎 Luxury Spender | Highest avg transaction amounts; premium merchant preference | Redirect 10% discretionary to liquid emergency fund |
| 🛍️ Weekend Shopper | Predictable Fri–Sun impulse pattern; entertainment + shopping heavy | Pre-plan weekly budget + cashback credit card |

---

## 6. Engine 2 — Leakage Detection

**Scripts:** [`src/leakage_detection.py`](../src/leakage_detection.py) · [`src/leakage_classifier.py`](../src/leakage_classifier.py)  
**Input:** `data/transactions_synthetic.csv`  
**Outputs:** `data/recurring_transactions.csv` · `data/user_leakage.csv` · `outputs/top_leakages.png` · `models/leakage_classifier.pkl`

### Recurring Payment Detection

A (user, merchant) pair is classified as **recurring** if the merchant appears in **≥ 3 distinct calendar months**.

- **6,390** unique (user, merchant) groups scanned
- **276** recurring pairs identified (≥3 months)

### Status Classification

| Status | Rule | Count | Share |
|--------|------|-------|-------|
| 🔴 **Dormant** | Last transaction > 60 days ago | 141 | 51.1% |
| 🟢 **Active** | Last transaction ≤ 30 days ago | 94 | 34.1% |
| 🟡 **Low Usage** | `cost_per_use > median + 1σ` (≥ ₹1,717) | 38 | 13.8% |
| 🟣 **Duplicate** | ≥2 merchants from same service group | 3 | 1.1% |

**Duplicate groups checked:** food delivery · streaming video · streaming music · internet providers

### Leakage Score Formula

```
Leakage Score = normalize(cpu_norm × dormancy_factor × overlap_factor × 100, [0, 100])

  cpu_norm         = (cost_per_use − min) / (max − min)    ∈ [0, 1]
  dormancy_factor  = 1.5  if dormancy_days > 30, else 1.0
  overlap_factor   = 2.0  if status == "Duplicate", else 1.0
```

### Leakage Results

| Metric | Value |
|--------|-------|
| Total recurring subscriptions | 276 |
| High leakage (score > 60) | 26 |
| Top leakage score | 100.00 (Swiggy — USR0194) |
| **Total potential annual savings** | **₹30,34,883 (~₹30.3 Lakh)** |
| Avg saving per user | ₹14,182/year |
| Users with ≥1 leaking subscription | 155 / 500 |

### Random Forest Classifier

**Training data:** 276 samples · 6 features · stratified 80/20 split

| Feature | Importance |
|---------|-----------|
| `cost_per_use` | **45.30%** |
| `monthly_amount` | **39.80%** |
| `user_avg_monthly_spend` | 9.06% |
| `days_since_last_use` | 5.22% |
| `frequency_per_month` | 0.61% |
| `category_essential_flag` | 0.00% |

| Metric | Score |
|--------|-------|
| Accuracy | 100.00% |
| Precision | 100.00% |
| Recall | 100.00% |
| F1 Score | 100.00% |
| ROC-AUC | 1.0000 |
| 5-fold CV F1 (train) | 80.89% ± 15.05% |

> **Note:** Perfect test accuracy is expected since the leakage label is derived directly from the leakage score, and `cost_per_use` + `monthly_amount` encode most of that score. CV F1 of **80.89%** is the realistic generalisation estimate.

---

## 7. Engine 3 — Wellness Forecast

**Script:** [`src/predictive_wellness.py`](../src/predictive_wellness.py)  
**Libraries:** `pmdarima 2.1.1` · `prophet 1.4.0` (Stan/cmdstanpy backend)  
**Input:** `data/transactions_synthetic.csv`  
**Outputs:** `outputs/wellness_forecast_baseline.csv` · `models/baseline_forecasts.json` · `outputs/wellness_distribution.png` · `outputs/sample_forecasts.png`

### Methodology

```
For each of 30 sampled users:
  1. Build monthly expense time series (12 data points)
  2. Fit auto_arima  → forecast month 13 + 95% CI
  3. Fit Prophet     → forecast month 13 + uncertainty interval
  4. Compute Wellness Score:

     available_budget  = avg_monthly_income − ₹10,000 (savings target)
     stress_factor     = (ARIMA_forecast / available_budget) × 100
     wellness_score    = clip(100 − stress_factor, 0, 100)

  5. cash_flow_risk  = 1  if ARIMA_forecast > available_budget
  6. risk_level      = "Low Risk"    if wellness_score ≥ 70
                       "Medium Risk" if 40 ≤ wellness_score < 70
                       "High Risk"   if wellness_score < 40
```

### Forecast Results (30 Users)

| Metric | Value |
|--------|-------|
| Mean Wellness Score | **92.84 / 100** |
| Median Wellness Score | 92.15 |
| Score Std Dev | 3.16 |
| Score Range | 87.41 → 100.00 |
| Low Risk users | 30 / 30 (100%) |
| Cash-flow Risk users | 0 / 30 |
| ARIMA mean MAPE (vs last month) | 174.65% |
| Prophet mean MAPE (vs last month) | 171.46% |

> **High MAPE explanation:** MAPE is computed against the single last known monthly value, which fluctuates significantly month-to-month. The models track the *average trend* correctly. With 3+ years of history, MAPE would reduce substantially. All users score above 87 because synthetic spending (avg ₹3,500–₹6,500/month) is well below income (₹30,000–₹80,000/month).

---

## 8. Streamlit Dashboard

**File:** [`dashboard/app.py`](../dashboard/app.py)  
**Launch:** `streamlit run dashboard/app.py`  
**URL:** http://localhost:8501

### Views

| View | Contents |
|------|----------|
| **Overview** | 4 KPI cards · 3 engine description cards · Category spend donut · Persona distribution bar |
| **Engine 1 — Segmentation** | Cluster stats table · AI insight panel · Interactive PCA scatter (Plotly, hover) · Persona filter · CSV download |
| **Engine 2 — Leakage** | KPI cards · Top-15 leakage bar chart · Status donut · Annual savings bar · Filterable dataframe (status, score slider, top-N) · Insight box |
| **Engine 3 — Forecast** | 5 KPI cards · Wellness histogram · Current vs forecast scatter · User selector · Wellness gauge · Actual + ARIMA + Prophet line chart · Risk text |

### Technical Highlights

- `@st.cache_resource` — all data loaded once at startup
- Graceful degradation — every file load wrapped in `try/except` with amber warning boxes
- All charts rendered with **Plotly** (interactive hover, zoom, pan)
- Download button for `personas_summary.csv`
- Custom CSS for metric cards, engine cards, insight boxes

---

## 9. Testing Suite

**File:** [`tests/test_models.py`](../tests/test_models.py)  
**Run:** `python tests/test_models.py`  
**Report:** `outputs/test_report.txt`

### Results: **62 / 62 PASS** in 0.98 seconds

| Suite | Tests | Result |
|-------|-------|--------|
| 🗃️ Data Quality | 9 | ✅ All Pass |
| 🔢 Feature Engineering | 6 | ✅ All Pass |
| 🎯 Clustering | 8 | ✅ All Pass |
| 🤖 Classification | 6 | ✅ All Pass |
| 📈 Forecasting | 7 | ✅ All Pass |
| 💚 Wellness Scores | 5 | ✅ All Pass |
| 📁 Output Files | 21 | ✅ All Pass |
| **Total** | **62** | 🏆 **PERFECT** |

### Test Coverage

- **Data:** row count, date span, unique users, missing values, amount ranges, categories, transaction types, duplicate rows
- **Features:** shape, normalization bounds [0,1], NaN check, variance, required columns
- **Clustering:** 5 clusters, all 500 users assigned once, labels ∈ {0-4}, all persona names, min cluster size, silhouette score, model reload
- **Classification:** model loads, accuracy/precision/recall/F1 thresholds, leakage CSV columns, score range, status values, feature importance JSON
- **Forecasting:** CSV exists, required columns, non-negative values, 5× sanity bound, JSON format, risk level validity, binary cash-flow flag
- **Wellness:** [0,100] range, mean in [25,100], std ≥ 1, no NaNs, IQR ≥ 0
- **Output Files:** 7 CSVs, 3 JSONs, 8 PNGs, 3 PKL models — all size-checked and content-validated

---

## 10. EDA Notebook

**File:** [`notebooks/01_eda_and_results.ipynb`](../notebooks/01_eda_and_results.ipynb)  
**Execute:** `jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda_and_results.ipynb`

| Cell | Title | Charts |
|------|-------|--------|
| 1 | Imports & Data Load | Summary table (shapes + financial totals) |
| 2 | Spending Analysis | Transaction histogram · Category pie · Monthly trend · Top-15 merchants |
| 3 | Persona Breakdown | Cluster size bars · Savings ratio boxplots · Weekend spend boxplots · Feature stats table |
| 4 | Leakage Analysis | Score histogram by status · Status pie · Cost vs score scatter · Top-10 merchants · Annual savings |
| 5 | Wellness Distribution | Score histogram · Risk count bars · ARIMA vs Prophet scatter · Wellness vs current avg · Model comparison |
| 6 | Key Findings Summary | Dark-themed summary card: all 3 engines + 10 dataset KPIs |

All plots saved to `outputs/eda_*.png` at 150 dpi.

---

## 11. Data Dictionary

### transactions_synthetic.csv (10,000 rows)

| Column | Type | Description |
|--------|------|-------------|
| `user_id` | str | `USR0001` – `USR0500` |
| `date` | datetime | 2025-09-21 → 2026-09-21 |
| `amount` | float | ₹50.25 – ₹79,978.00 |
| `category` | str | Food · Transport · Entertainment · Subscriptions · Utilities · Shopping · Medical · Salary |
| `merchant_name` | str | 65 unique merchants (Zomato, Uber, Netflix, …) |
| `transaction_type` | str | Debit · Credit |
| `description` | str | `{merchant} - {category}` |

### user_features.csv (500 × 20, normalized [0,1])

All 20 features are MinMax-scaled to [0, 1] after StandardScaler normalization.

### user_clusters.csv (500 rows)

| Column | Description |
|--------|-------------|
| `user_id` | User identifier |
| `cluster_label` | Integer 0–4 |
| `persona_name` | Budget-Conscious · Impulsive Spender · Heavy Subscriber · Luxury Spender · Weekend Shopper |

### user_leakage.csv (276 rows)

| Column | Description |
|--------|-------------|
| `user_id` | User identifier |
| `merchant_name` | Merchant with recurring payments |
| `category` | Transaction category |
| `monthly_amount` | Average monthly cost (₹) |
| `cost_per_use` | Monthly cost / uses per month (₹) |
| `dormancy_days` | Days since last transaction |
| `leakage_score` | 0–100 normalized score |
| `status` | Active · Low Usage · Dormant · Duplicate |
| `estimated_annual_saving` | Potential annual saving if cancelled (₹) |

### wellness_forecast_baseline.csv (30 rows)

| Column | Description |
|--------|-------------|
| `user_id` | User identifier |
| `current_monthly_avg` | Average monthly expense over 12 months (₹) |
| `forecasted_month13_arima` | ARIMA month-13 forecast (₹) |
| `forecasted_month13_prophet` | Prophet month-13 forecast (₹) |
| `wellness_score` | Financial Wellness Score (0–100) |
| `cash_flow_risk` | 1 if forecast > (income − ₹10,000), else 0 |
| `risk_level` | Low Risk · Medium Risk · High Risk |

---

## 12. Model Artifacts

| File | Algorithm | Parameters | Size |
|------|-----------|------------|------|
| `models/kmeans_model.pkl` | K-Means | n_clusters=5, n_init=20, random_state=42 | 3.4 KB |
| `models/leakage_classifier.pkl` | Random Forest | n_estimators=100, max_depth=10, class_weight=balanced, random_state=42 | 181 KB |
| `models/leakage_scaler.pkl` | StandardScaler | Fitted on 6 leakage features | 0.7 KB |
| `models/baseline_forecasts.json` | ARIMA + Prophet | auto_arima (stepwise, AIC), Prophet (no seasonality) | 12.5 KB |

---

## 13. Output Files

| File | Size | Description |
|------|------|-------------|
| `correlation_matrix.png` | 261 KB | 20×20 feature correlation heatmap |
| `elbow_curve.png` | 88 KB | Inertia + silhouette vs k (2–8) |
| `personas_scatter.png` | 235 KB | PCA 2D scatter — 5 colored clusters |
| `top_leakages.png` | 230 KB | Top-20 leakage bar + savings donut |
| `leakage_cm.png` | 64 KB | Confusion matrix (counts + row %) |
| `feature_importance.png` | 58 KB | RF feature importance bars |
| `wellness_distribution.png` | 94 KB | Wellness histogram + risk donut |
| `sample_forecasts.png` | 174 KB | 3 user actual + ARIMA + Prophet |
| `eda_spending_analysis.png` | 250 KB | Cell 2 — 4-panel spending analysis |
| `eda_persona_breakdown.png` | 191 KB | Cell 3 — cluster stats + boxplots + table |
| `eda_leakage_analysis.png` | 266 KB | Cell 4 — leakage 5-panel analysis |
| `eda_wellness_distribution.png` | 226 KB | Cell 5 — forecast 5-panel analysis |
| `eda_key_findings.png` | 214 KB | Cell 6 — dark-themed summary card |
| `persona_insights.json` | 3.8 KB | AI summaries per persona (5 records) |
| `leakage_features.json` | 0.2 KB | Top-5 RF feature importances |
| `personas_summary.csv` | 1.6 KB | 5 personas × 10 columns |
| `wellness_forecast_baseline.csv` | 1.5 KB | 30 users × 7 columns |
| `test_report.txt` | — | Full 62-test pass/fail detail |

---

## 14. Key Metrics & Results

```
╔═══════════════════════════════════════════════════════════════╗
║          FINANCIAL WELLNESS INTELLIGENCE PLATFORM             ║
║                   Final Results Summary                       ║
╠═══════════════════════════════════════════════════════════════╣
║  DATASET                                                      ║
║    Total transactions        :  10,000                        ║
║    Unique users              :     500                        ║
║    Date range                :  12 months (365 days)          ║
║    Unique merchants          :      65                        ║
║    Total income (synthetic)  :  ₹2.57 Crore                  ║
║    Total expenses            :  ₹1.70 Crore                  ║
║    Net surplus               :  ₹0.87 Crore                  ║
╠═══════════════════════════════════════════════════════════════╣
║  ENGINE 1 — USER SEGMENTATION                                 ║
║    Algorithm                 :  K-Means (k=5)                 ║
║    Features engineered       :  20 (normalized [0,1])         ║
║    Silhouette score          :  0.1166                        ║
║    Largest persona           :  🔥 Impulsive Spender (32.2%) ║
║    PCA variance explained    :  41.5%                         ║
╠═══════════════════════════════════════════════════════════════╣
║  ENGINE 2 — LEAKAGE DETECTION                                 ║
║    Recurring subs detected   :     276                        ║
║    Dormant (>60 days)        :     141 (51.1%)                ║
║    High leakage (score >60)  :      26                        ║
║    Total potential savings   :  ₹30.3 Lakh / year            ║
║    RF Classifier accuracy    :  100%  (CV F1: 80.89%)         ║
║    Top predictive feature    :  cost_per_use (45.3%)          ║
╠═══════════════════════════════════════════════════════════════╣
║  ENGINE 3 — WELLNESS FORECAST                                 ║
║    Users forecasted          :       30 (sample)              ║
║    Mean wellness score       :   92.84 / 100                  ║
║    Low risk users            :  30/30 (100%)                  ║
║    Cash-flow risk users      :       0                        ║
║    ARIMA mean MAPE           :  174.65%  (sparse data)        ║
║    Prophet mean MAPE         :  171.46%  (sparse data)        ║
╠═══════════════════════════════════════════════════════════════╣
║  TESTING                                                      ║
║    Total automated tests     :      62                        ║
║    Tests passed              :  62 / 62  (100%)               ║
║    Execution time            :   0.98 seconds                 ║
╚═══════════════════════════════════════════════════════════════╝
```

---

## 15. Known Limitations & Next Steps

### Current Limitations

| # | Limitation | Impact |
|---|-----------|--------|
| 1 | **Synthetic data** — generated with Faker; patterns are idealized | Models will behave differently on real bank data |
| 2 | **Sparse time series** — only 12 monthly points per user | ARIMA/Prophet MAPE is high; models need 24–36 months for accuracy |
| 3 | **All users "Low Risk"** — synthetic income (₹30K–₹80K) >> synthetic spend (₹2K–₹6K) | Wellness scoring needs income calibration for real users |
| 4 | **TensorFlow excluded** — incompatible with Python 3.14 | Deep learning models (LSTM) not implemented |
| 5 | **Gemini API fallback** — API key not set; using rule-based descriptions | Live AI insights require a valid `GEMINI_API_KEY` in `.env` |
| 6 | **RF leakage label leakage** — classifier trained on features that directly encode the target | Re-train with user-provided ground-truth labels for real evaluation |
| 7 | **30-user forecast sample** — Prophet compilation slow; full 500 would take ~15 min | Batch-process or cache Stan compilation for production |

### Recommended Next Steps

1. **Real data integration** — Connect to bank APIs (Plaid, Razorpay, Setu) for live transactions
2. **LSTM/Transformer forecasting** — Replace ARIMA/Prophet with sequence models using Python 3.12 + TensorFlow
3. **Gemini AI insights** — Add your `GEMINI_API_KEY` to `.env` for live persona descriptions
4. **User authentication** — Add Streamlit Auth or Supabase for multi-user dashboard access
5. **Alert system** — Push notifications when leakage score > 70 or wellness drops below 50
6. **A/B testing framework** — Test recommendation effectiveness (did the user cancel the dormant sub?)
7. **MLflow integration** — Track model versions, hyperparameters, and metrics across training runs
8. **Scheduled retraining** — Cron job to re-cluster and re-score monthly as new transactions arrive

---

## 16. Dependencies

```
# requirements.txt (installed in venv)
pandas==2.3.0
numpy==2.2.5
scikit-learn==1.7.0
matplotlib==3.10.3
seaborn==0.13.2
streamlit==1.45.1
plotly==6.1.2
joblib==1.5.0
jupyter==1.1.1
python-dotenv==1.1.0
faker==37.3.0
statsmodels==0.14.5
google-generativeai==0.8.6   # deprecated → migrate to google-genai
pmdarima==2.1.1              # installed in Prompt 9
prophet==1.4.0               # installed in Prompt 9

# Not installed (incompatible with Python 3.14)
# tensorflow — requires Python ≤ 3.12
```

### Python Environment

```bash
Python   : 3.14.2  (arm64 macOS)
venv     : financial-wellness-engine/venv/
pip      : 25.3
Platform : macOS arm64 (Apple Silicon)
```

---

*Generated by Antigravity — Financial Wellness Intelligence Platform · Review 2 Documentation*
