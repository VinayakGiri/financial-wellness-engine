# 💎 Personal Financial Wellness Intelligence Platform Using Data Analytics and Machine Learning

> **Understand. Detect. Predict. Prescribe.**
> *B.Tech Capstone Project — SVKM's NMIMS MPSTME, Shirpur Campus (Academic Session: 2026-27)*

---

## 🏛️ Academic Capstone Attribution

- **Institution**: SVKM’s NMIMS, Mukesh Patel School of Technology Management & Engineering (MPSTME), Shirpur Campus (M.H.), India.
- **Department**: Department of Computer Science
- **Degree**: Bachelor of Technology (B.Tech) in Computer Science
- **Student Investigators**:
  - **Vinayak Giri** — `70552300013`
  - **Gauri Mathur** — `70552300033`
  - **Deshna Sancheti** — `70552300036`
- **Project Supervisor**: **Dr. Radhakrishna Rambola**, Assistant Professor, Dept. of Computer Science
- **Department Leadership**: **Dr. Nitin Chowbey** (H.O.D.) &bull; **Dr. Venkatadri Marriboyina** (Associate Dean)

---

## 📋 Executive Summary & Abstract

The increasing proliferation of digital payment systems, UPI, and online subscriptions has made personal financial management significantly more complex. Most traditional finance applications center around manual categorization and static budgeting, providing limited forward visibility into behavioral patterns, unnecessary recurring leakages, or impending cash shortages.

This platform delivers a unified intelligent framework combining three machine learning engines:
1. **Behavioral Segmentation Engine**: K-Means clustering, 2D/3D Principal Component Analysis (PCA), Ward's Hierarchical Dendrogram, and Topological Data Analysis (TDA) using Kepler Mapper.
2. **Financial Leakage Detection Engine**: Recurrence detection (3+ payments within 90 days within ±10% variance), fuzzy merchant normalization (Levenshtein distance), and Composite Leakage Scoring powered by Random Forest and Gradient Boosting (XGBoost architecture).
3. **Predictive Financial Wellness Engine**: Multi-model cash-flow forecasting fusing Auto-ARIMA, Facebook Prophet, and a PyTorch 1D-CNN + Stacked LSTM deep learning network with a rigorous **5-Component Financial Wellness Index**.
4. **Natural Language Advisory Layer**: Google Gemini 1.5 Flash grounded strictly on verified engine calculations.

Evaluated on 500 users across 12 months, the platform discovered **5 spending personas**, classified **276 recurring payment streams**, and identified **₹30.35 lakh** in recoverable annual waste (average ₹6,070 per user/year).

---

## 🚀 Quick Start

```bash
# 1. Clone repository
git clone https://github.com/VinayakGiri/financial-wellness-engine.git
cd financial-wellness-engine

# 2. Activate virtual environment
source venv/bin/activate

# 3. Launch Credix Fintech UI Dashboard
streamlit run dashboard/app.py --server.port 8501

# 4. Run automated evaluation suite (76 tests)
python tests/test_models.py
pytest tests/test_adapters.py

# 5. Extract all project figures and statistical reports
unzip finwell_graphs_and_statistics.zip
```

---

## 📦 Download Complete Graphs & Statistics Archive

All publication-ready charts (17 figures), model comparison metrics, test logs, and analytical summaries are packaged into a single standalone ZIP archive in the repository root:
- **Archive Path**: [`finwell_graphs_and_statistics.zip`](finwell_graphs_and_statistics.zip)
- **Archive Contents**:
  - **17 High-Resolution Visualizations**: `deep_learning_architecture.png`, `topological_network.png`, `hierarchical_dendrogram.png`, `elbow_curve.png`, `personas_scatter.png`, `leakage_cm.png`, `feature_importance.png`, `model_comparison_mape.png`, `sample_forecasts.png`, `top_leakages.png`, `wellness_distribution.png`, `correlation_matrix.png`, and comprehensive EDA breakdown plots.
  - **Comprehensive Statistical Summaries**: `personas_summary.csv`, `wellness_forecast_final.csv`, `wellness_forecast_baseline.csv`, `leakage_features.json`, `leakage_model_comparison.json`, `persona_insights.json`, `tda_graph.json`.
  - **Test Execution Audit**: `test_report.txt` (100% pass rate across all 76 unit and integration test assertions).

---

## 🧠 Three Integrated Analytical Engines

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                INGESTION & DATA ADAPTER LAYER                           │
│  Plaid Sandbox API  •  RBI Sahamati Account Aggregator (Setu)  •  Bank Statements (CSV/PDF) │
│  🔒 Zero-Knowledge PII Sanitizer: All account numbers & phone numbers hashed with SHA-256 │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
         ┌─────────────────────────────────┼─────────────────────────────────┐
         ▼                                 ▼                                 ▼
┌─────────────────────────┐   ┌─────────────────────────┐   ┌─────────────────────────┐
│ ENGINE 1: BEHAVIORAL    │   │ ENGINE 2: LEAKAGE       │   │ ENGINE 3: PREDICTIVE    │
│ SEGMENTATION            │   │ DETECTION & DEFENSE     │   │ FINANCIAL WELLNESS      │
├─────────────────────────┤   ├─────────────────────────┤   ├─────────────────────────┤
│ • 20 Engineered Features│   │ • 90-Day Recurrence     │   │ • PyTorch 1D-CNN + LSTM │
│ • K-Means (k=5)         │   │ • Fuzzy Levenshtein     │   │ • Auto-ARIMA (p,d,q)    │
│ • 2D & 3D PCA Projection│   │ • Composite Leakage (0-100)│ • Facebook Prophet      │
│ • Ward Dendrogram       │   │ • RF vs. XGBoost        │   │ • 30-Day Solvency Bands │
│ • Kepler Mapper (TDA)   │   │ • "What-If" Simulator   │   │ • 5-Component Formula   │
└────────────┬────────────┘   └────────────┬────────────┘   └────────────┬────────────┘
             │                             │                             │
             └──────────────────────┬──────┴─────────────────────────────┘
                                    ▼
       ┌───────────────────────────────────────────────────────────┐
       │   EXPLAINABILITY LAYER: GOOGLE GEMINI 1.5 FLASH           │
       │   Transforms verified telemetry into 3 strategic pillars: │
       │   1. Priority Remediation (24-48h)                        │
       │   2. Mid-Month Budget Guardrails                          │
       │   3. End-of-Month Solvency Strategy                       │
       └────────────────────────────┬──────────────────────────────┘
                                    ▼
       ┌───────────────────────────────────────────────────────────┐
       │   CREDIX FINTECH & SAAS STREAMLIT DASHBOARD (5 TABS)      │
       │   Overview • Personas & TDA • Leakage • Forecast • Thesis │
       └───────────────────────────────────────────────────────────┘
```

---

## 📐 Mathematical Formulations (As Specified in Thesis)

### 1. Composite Leakage Score (Chapter 4.3.3)
$$\text{Cost-Per-Use} = \frac{\text{Monthly Cost}}{\text{Monthly Usage Frequency} + 1}$$
$$\text{Dormancy Factor} = \min\left(1.0, \frac{\text{Days Inactive}}{90}\right)$$
$$\text{Leakage Score} = 100 \times \left[(0.5 \times (1 - \text{Normalized CPU})) + (0.3 \times \text{Dormancy Factor}) + (0.2 \times \text{Overlap Penalty})\right]$$

### 2. Multi-Model Ensemble Cash-Flow Forecast (Chapter 4.4.3)
$$F_{\text{ensemble}} = \sum_{k=1}^3 \omega_k F_k \quad \text{where} \quad \sum_{k=1}^3 \omega_k = 1$$
Combining PyTorch 1D-CNN + LSTM, Auto-ARIMA, and Facebook Prophet.

### 3. Objective 5-Component Financial Wellness Score (Chapter 4.4.4)
$$W = 100 \sum_{j=1}^5 w_j s_j$$

| Component ($j$) | Meaning & Objective | Assigned Weight ($w_j$) | Normalized Score ($s_j \in [0, 1]$) |
|---|---|---|---|
| **Balance Sufficiency** | Liquid balance vs. 1.5x monthly burn | **35% ($w_1 = 0.35$)** | $s_1 = \text{clip}(\frac{\text{Balance}}{1.5 \times \text{Monthly Spend}}, 0, 1)$ |
| **Savings Rate** | Monthly savings vs. 20% income standard | **25% ($w_2 = 0.25$)** | $s_2 = \text{clip}(\frac{\text{Savings Ratio}}{0.20}, 0, 1)$ |
| **Budget Adherence** | Expenditure contained within available income | **20% ($w_3 = 0.20$)** | $s_3 = \text{clip}(1 - \frac{\text{Overspend}}{\text{Budget}}, 0, 1)$ |
| **Leakage Mitigation** | Absence of wasteful subscription drain | **15% ($w_4 = 0.15$)** | $s_4 = \text{clip}(1 - \frac{\text{Leakage}}{\text{Ceiling}}, 0, 1)$ |
| **Forecast Safety** | 30-day forward runway without zero-cash breach | **5% ($w_5 = 0.05$)** | $s_5 = 1.0 \text{ if solvent, else } \frac{\text{Days}}{30}$ |

---

## 📊 Prototype Experimental Evaluation (Chapter 5)

### Table 5.1: Distribution of Financial Personas (500 Users)
| Persona | User Count | Percentage | Core Spending Behavior |
|:---|:---:|:---:|:---|
| 🔥 **Impulsive Spender** | 161 | 32.2% | High volatility, large impulse buys, weekend splurges |
| 💰 **Budget-Conscious** | 100 | 20.0% | Strict budget adherence, high savings ratio, minimal luxuries |
| 📺 **Heavy Subscriber** | 85 | 17.0% | Multiple recurring subscriptions, high digital service ratio |
| 💎 **Luxury Spender** | 81 | 16.2% | High average ticket size, premium category preference |
| 🛍️ **Weekend Shopper** | 73 | 14.6% | Disproportionate expenditure concentrated on Fri–Sun |

### Table 5.2: Recurring Payment Classification (276 Streams)
| Status | Stream Count | Percentage | Recoverable Annual Amount |
|:---|:---:|:---:|:---|
| **Dormant** | 141 | 51.1% | ₹18,84,332 (~₹18.84 Lakh) |
| **Active** | 94 | 34.1% | ₹0 (Seemingly utilized periodically) |
| **Low Usage** | 38 | 13.8% | ₹10,99,231 (~₹10.99 Lakh) |
| **Duplicate** | 3 | 1.0% | ₹51,320 (~₹0.51 Lakh) |
| **TOTAL** | **276** | **100.0%** | **₹30,34,883 (~₹30.35 Lakh)** |

*Average recoverable savings opportunity: **₹6,070 / user / year**.*

### Table 5.4: Cash-Flow Forecasting Benchmark (Demonstration User)
| Forecasting Model | Projected Net Drawdown | Model Type |
|:---|:---:|:---|
| **PyTorch LSTM** | ₹3,581 | Deep Recurrent Neural Network |
| **Auto-ARIMA** | ₹3,617 | Statistical Autoregressive Integrated Moving Average |
| **Facebook Prophet** | ₹3,475 | Bayesian Seasonality Generalized Additive Model |
| **Weighted Ensemble Consensus** | **₹3,558** | Multi-Model Combined Consensus |
| *Starting Liquid Balance* | *₹53,304* | Verified Sandbox Account Buffer |
| *Projected Ending Balance* | *₹49,746* | Safe & Solvent (Financial Wellness Score: **99.5 / 100**) |

---

## 🔬 Key Methodological Innovations

1. **Topological Data Analysis (TDA) with Kepler Mapper (Section 4.5 & Figure 8)**:
   - Projects 20-dimensional behavioral features through a PCA lens.
   - Generates a simplicial complex network with **47 simplicial nodes**, **116 topological edges**, and **3 connected components**, uncovering hidden transaction manifolds.
2. **Deep Learning Sequence Architecture (Section 4.2 & Figure 4)**:
   - Integrates a 1D-CNN (`Conv1d`, 32 filters, `ReLU`, `MaxPool1d`) to capture localized pay-day spikes and weekend bursts.
   - Feeds into stacked 2-layer LSTM units (`hidden_size=64`, `dropout=0.2`) to track long-term monthly seasonality.
3. **Machine Learning Classifier Benchmarking (Section 4.3.2)**:
   - Compares **Random Forest** (100.0% Accuracy, 100.0% F1, 1.000 ROC-AUC) against **Gradient Boosting / XGBoost** (98.21% Accuracy, 90.91% F1, 1.000 ROC-AUC).
4. **Zero-Knowledge Regulatory Compliance**:
   - Compliant with **India's Digital Personal Data Protection Act, 2023 (Act No. 22 of 2023)** and RBI Account Aggregator guidelines.

---

## 🗂️ Project Repository Structure

```
financial-wellness-engine/
├── dashboard/
│   └── app.py                     # Credix 5-tab PFM dashboard
├── src/
│   ├── data_adapter.py            # Plaid, Setu AA, Bank CSV/PDF parser with SHA-256 PII hashing
│   ├── engine_1_behavior.py       # Engine 1: Behavioral Persona & 2D/3D PCA & Ward Linkage
│   ├── engine_2_leakage.py        # Engine 2: Subscriptions, Cost-per-use, What-If simulator
│   ├── engine_3_predictive.py     # Engine 3: 30-Day Cash-Flow, 5-Component Wellness Score
│   ├── tda_mapper.py              # Kepler Mapper Topological Data Analysis (Figure 8)
│   ├── deep_learning_models.py    # PyTorch 1D-CNN + Stacked LSTM Architecture (Figure 4)
│   ├── leakage_classifier.py      # Random Forest vs. Gradient Boosting / XGBoost comparison
│   ├── lstm_forecasting.py        # Deep learning forecasting pipeline
│   ├── predictive_wellness.py     # Baseline ARIMA + Prophet models
│   ├── clustering.py              # K-Means (k=5) + Hierarchical Dendrogram
│   ├── feature_engineering.py     # 20 behavioral features extractor
│   ├── gemini_insights.py         # Google Gemini 1.5 Flash natural language advisor
│   └── generate_data.py           # Synthetic cohort generator (500 users, 10K txns)
├── tests/
│   ├── test_adapters.py           # 11 tests for authentic data ingestion & PII sanitization
│   └── test_models.py             # 62 tests across data, ML models, and output integrity
├── data/                          # SQLite DB + CSV files
├── models/                        # Serialized .pkl and PyTorch .pt model checkpoints
└── outputs/                       # Figures (Fig 4, Fig 8, Dendrogram, Elbow, CM, etc.)
```

---

## ⚖️ License & Intellectual Property

Submitted in partial fulfillment of the requirement for the award of the Degree of **Bachelor of Technology in Computer Science** at **SVKM's NMIMS, MPSTME Shirpur Campus**. All rights reserved &copy; 2026-27.
