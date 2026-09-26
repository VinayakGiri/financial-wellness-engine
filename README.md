# 💰 Financial Wellness Intelligence Platform

> **Understand. Detect. Predict. Recommend.**

An end-to-end personal finance analytics system powered by Machine Learning and Deep Learning — built in Python with Streamlit, Scikit-learn, PyTorch, and ARIMA/Prophet.

---

## 🚀 Live Demo

```bash
source venv/bin/activate
streamlit run dashboard/app.py
# Open http://localhost:8501
```

---

## 🧠 Three Integrated ML Engines

| Engine | Problem | Approach | Output |
|--------|---------|----------|--------|
| 🎯 **Engine 1** | Who are my users? | K-Means Clustering (k=5) | 5 Spending Personas |
| 🔍 **Engine 2** | What are they wasting? | Rule-based scoring + Random Forest | Leakage Scores + ₹30.3L savings |
| 📈 **Engine 3** | What will they spend next month? | ARIMA + Prophet + LSTM (PyTorch) | Wellness Score (0–100) |

---

## 📊 Key Results

```
Dataset  : 10,000 transactions · 500 users · 12 months · Indian market (INR)
Engine 1 : 5 personas · Silhouette = 0.1166 · 20 engineered features
Engine 2 : 276 recurring subscriptions · ₹30.3 Lakh potential annual savings
Engine 3 : Mean Wellness Score = 92.84/100 · 30-user ARIMA+Prophet forecast
LSTM     : 30,241 params · PyTorch 2.14 · Early stopped at epoch 14
Tests    : 62/62 PASS in 0.98 seconds
```

---

## 🗂️ Project Structure

```
financial-wellness-engine/
│
├── src/                          # All ML pipeline scripts
│   ├── generate_data.py          # Synthetic data (Faker, 10K txns)
│   ├── verify_setup.py           # Environment validation
│   ├── feature_engineering.py   # 20 behavioral features
│   ├── clustering.py             # K-Means k=5 personas
│   ├── persona_insights.py       # Statistical + Gemini insights
│   ├── leakage_detection.py      # Subscription leakage scoring
│   ├── leakage_classifier.py     # Random Forest classifier
│   ├── predictive_wellness.py    # ARIMA + Prophet forecasting
│   ├── lstm_forecasting.py       # LSTM (PyTorch) deep learning
│   └── gemini_insights.py        # Gemini 1.5 Flash narratives
│
├── dashboard/
│   └── app.py                    # Streamlit 4-view dashboard
│
├── notebooks/
│   └── 01_eda_and_results.ipynb  # 6-cell EDA + presentation
│
├── tests/
│   └── test_models.py            # 62 automated tests
│
├── data/                         # Generated CSVs (gittracked)
│   ├── transactions_synthetic.csv
│   ├── user_features.csv
│   ├── user_clusters.csv
│   ├── recurring_transactions.csv
│   └── user_leakage.csv
│
├── models/                       # JSON artifacts
│   ├── baseline_forecasts.json
│   ├── lstm_training_history.json
│   └── lstm_predictions_all_users.json
│
├── outputs/                      # All charts, reports, insights
│   ├── *.png                     # 13 visualization charts
│   ├── *.csv                     # Result CSVs
│   ├── *.json                    # Insight JSONs
│   ├── test_report.txt           # 62-test report
│   ├── user_narratives/          # 500 personalized .txt plans
│   ├── persona_strategies/       # 5 persona strategy .txt files
│   └── README_REVIEW2.md         # Full technical documentation
│
├── requirements.txt
└── .gitignore
```

---

## 🎯 5 User Personas (Engine 1)

| # | Persona | Users | Key Trait |
|---|---------|-------|-----------|
| 💰 | Budget-Conscious | 100 (20%) | Mindful spender, untapped savings potential |
| 🔥 | Impulsive Spender | 161 (32%) | Weekend purchase clusters, high volatility |
| 📺 | Heavy Subscriber | 85 (17%) | Highest savings rate, overlapping services |
| 💎 | Luxury Spender | 81 (16%) | Highest avg transaction, premium merchants |
| 🛍️ | Weekend Shopper | 73 (15%) | Predictable Fri–Sun spending spikes |

---

## 🔍 Leakage Detection (Engine 2)

```
Dormant subscriptions (>60 days)  : 141  (51.1%)
High leakage score (>60/100)      :  26
Total potential annual savings     : ₹30,34,883
RF Classifier accuracy             : 100%  (CV F1: 80.89%)
Top predictive feature             : cost_per_use (45.3%)
```

---

## 📈 Output Charts

| Chart | Description |
|-------|-------------|
| `outputs/correlation_matrix.png` | 20×20 feature correlation heatmap |
| `outputs/elbow_curve.png` | K-Means elbow + silhouette scores |
| `outputs/personas_scatter.png` | PCA 2D cluster scatter plot |
| `outputs/top_leakages.png` | Top-20 leaking subscriptions |
| `outputs/leakage_cm.png` | RF classifier confusion matrix |
| `outputs/feature_importance.png` | RF feature importances |
| `outputs/wellness_distribution.png` | Wellness score histogram |
| `outputs/sample_forecasts.png` | ARIMA + Prophet forecast plots |
| `outputs/model_comparison_mape.png` | LSTM vs ARIMA vs Prophet |
| `outputs/eda_spending_analysis.png` | Full spending EDA (Cell 2) |
| `outputs/eda_persona_breakdown.png` | Persona analysis (Cell 3) |
| `outputs/eda_leakage_analysis.png` | Leakage analysis (Cell 4) |
| `outputs/eda_wellness_distribution.png` | Forecast analysis (Cell 5) |
| `outputs/eda_key_findings.png` | Dark-themed results summary |

---

## ⚙️ Quick Start

```bash
# 1. Clone
git clone https://github.com/VinayakGiri/financial-wellness-engine
cd financial-wellness-engine

# 2. Create venv & install
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install pmdarima prophet torch google-genai

# 3. Run pipeline (in order)
python src/generate_data.py
python src/feature_engineering.py
python src/clustering.py
python src/persona_insights.py
python src/leakage_detection.py
python src/leakage_classifier.py
python src/predictive_wellness.py
python src/lstm_forecasting.py

# 4. Launch dashboard
streamlit run dashboard/app.py

# 5. Run tests
python tests/test_models.py

# 6. (Optional) Gemini AI insights — add key to .env first
echo "GEMINI_API_KEY=your_key_here" > .env
python src/gemini_insights.py
```

---

## 🧪 Test Suite — 62/62 PASS

```
Suite                        Tests    Result
─────────────────────────────────────────────
Data Quality                   9/9    ✅ All Pass
Feature Engineering            6/6    ✅ All Pass
Clustering                     8/8    ✅ All Pass
Classification                 6/6    ✅ All Pass
Forecasting                    7/7    ✅ All Pass
Wellness Scores                5/5    ✅ All Pass
Output Files                  21/21   ✅ All Pass
─────────────────────────────────────────────
TOTAL                         62/62   🏆 PERFECT
```

---

## 📦 Tech Stack

| Category | Libraries |
|----------|-----------|
| **Data** | pandas, numpy, faker |
| **ML** | scikit-learn, statsmodels |
| **Deep Learning** | PyTorch 2.14, pmdarima, prophet |
| **Visualization** | matplotlib, seaborn, plotly |
| **Dashboard** | Streamlit |
| **AI Insights** | google-genai (Gemini 1.5 Flash) |
| **Testing** | Custom 62-test suite |

**Python:** 3.14.2 · **Platform:** macOS arm64 (Apple Silicon)

---

## 📄 Full Documentation

See [`outputs/README_REVIEW2.md`](outputs/README_REVIEW2.md) for the complete technical documentation including data dictionary, model artifacts, formulas, and known limitations.

---

*Built with ❤️ · Financial Wellness Intelligence Platform*
