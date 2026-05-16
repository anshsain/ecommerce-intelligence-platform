# 🛒 E-Commerce Customer Intelligence Platform

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat&logo=python)
![SQL](https://img.shields.io/badge/SQL-MSSQL-red?style=flat&logo=microsoftsqlserver)
![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange?style=flat)
![HuggingFace](https://img.shields.io/badge/NLP-HuggingFace-yellow?style=flat&logo=huggingface)
![Docker](https://img.shields.io/badge/Docker-Containerised-blue?style=flat&logo=docker)
![Streamlit](https://img.shields.io/badge/App-Streamlit-ff4b4b?style=flat&logo=streamlit)
![Status](https://img.shields.io/badge/Status-Live-brightgreen?style=flat)

An end-to-end data science project built on the **Online Retail II (UCI) dataset** — 824,293 real-world transactions from a UK-based wholesale retailer spanning December 2009 to December 2011.

### 🔗 [Live Demo →](https://ecommerce-intelligence-platform-ar3njx2ozilgoka4qdliqw.streamlit.app/)

---

## Business Problem

> *"Can we identify which customers are about to churn, understand what drives their purchasing behaviour, and recommend products — so a business can act before it loses revenue?"*

This is a real problem every e-commerce company, from Flipkart to a D2C startup, actively solves. This project replicates that analytics pipeline end-to-end.

---

## Project Architecture
Raw Excel (1M+ rows)

▼

Data Cleaning & Engineering (Python · Pandas · Google Colab)
Handled nulls, duplicates, cancellations, and type errors

▼

SQL Analytical Layer (MSSQL · SSMS)
6 Production Views — Revenue Trends, CLV, RFM, Cohorts, Leakage, Products

▼

Machine Learning (Scikit-learn · XGBoost · SHAP)
Churn Prediction · RFM Segmentation · Recommendation Engine

▼

Containerised with Docker

▼

NLP — HuggingFace RoBERTa Transformer
Product Sentiment Analysis · Cancellation Keyword Analysis

▼

Live Streamlit App (Deployed on Streamlit Cloud)
5 Interactive Tabs — Overview · Churn · Segments · Recommendations · Insights

---

## Layer 1 — Data Engineering

- Combined two Excel sheets (2009–2010 and 2010–2011) into a single 1,067,371 row dataframe
- Handled real-world mess: duplicate entries, negative quantities, zero prices, missing CustomerIDs (22% null), inconsistent formats
- Flagged 18,744 cancelled orders for downstream revenue leakage analysis rather than dropping them
- Engineered `TotalAmount`, `IsCancelled` columns
- Final clean dataset: **824,293 rows, zero nulls**
- Exported to MSSQL via CSV import — all downstream analytics run on the database

---

## Layer 2 — SQL Analytical Layer

Built 6 production-grade views in MSSQL:

| View | Business Question |
|---|---|
| `vw_monthly_revenue` | Which months drove the most revenue? |
| `vw_customer_ltv` | Who are our highest value customers? |
| `vw_rfm_scores` | How recent, frequent, and valuable is each customer? |
| `vw_revenue_leakage` | How much revenue is lost to cancellations by country? |
| `vw_product_performance` | Which products drive revenue vs volume? |
| `vw_cohort_retention` | Of customers who bought in month X, how many returned? |

All views use window functions, CTEs, aggregations, and date functions — not ad hoc queries.

---

## Layer 3 — Machine Learning

### Model A — Customer Churn Prediction
- **Definition:** Customer inactive for 90+ days = churned
- **Features:** Frequency, Monetary, F_Score, M_Score, TotalOrders, TotalItemsBought, AvgOrderValue, CustomerLifespanDays
- **Class balance:** Dataset was naturally balanced (50.9% churned vs 49.1% not churned) — no SMOTE required. Pipeline includes SMOTE-ready structure for imbalanced datasets
- **Pipeline:** Logistic Regression → Random Forest → XGBoost
- **Caught and fixed data leakage** — Recency directly encodes churn label, removed from features
- **Result:** XGBoost AUC-ROC **0.863**, Churn Recall **0.81**
- **SHAP explainability** — F_Score identified as 5x more predictive than monetary value

| Model | AUC-ROC | Accuracy | Precision | Churn Recall |
|---|---|---|---|---|
| Logistic Regression | 0.786 | 72% | 0.713 | 74% |
| Random Forest | 0.864 | 76% | 0.761 | 77% |
| **XGBoost ✓** | **0.863** | **76%** | **0.748** | **81%** |

### Model B — RFM Customer Segmentation
- K-Means clustering with elbow method → optimal **K=4**
- **Silhouette Score (K=4): 0.5909** — confirms strong, well-separated clusters (>0.5 threshold)
- Segments discovered from data (not assumed):

| Segment | Customers | Avg Recency | Avg Spend |
|---|---|---|---|
| VIP Whales | 4 | 3 days | £436,835 |
| Champions | 38 | 24 days | £79,380 |
| Loyal Regulars | 3,861 | 67 days | £2,976 |
| Hibernating | 2,012 | 462 days | £768 |

### Model C — Product Recommendation Engine
- Item-item collaborative filtering using cosine similarity
- **Why cosine similarity:** Purchase matrix is sparse and volume-heavy — cosine captures preference direction regardless of order magnitude, unlike Euclidean distance
- Purchase matrix: **4,761 customers × 500 top products**
- Recommends products a customer hasn't bought based on similar buyers' behaviour
- **Cold start handling:** New customers with no purchase history are handled by falling back to top revenue-generating products within the closest RFM segment

---

## 💬 Layer 4 — NLP with HuggingFace Transformers

- Replaced rule-based TextBlob with **fine-tuned RoBERTa** (`cardiffnlp/twitter-roberta-base-sentiment`)
- Ran inference on **5,331 unique product descriptions** in batches
- Model correctly detects informal language — "W/SUCK", "I AM SO POORLY", "YOU'RE CONFUSING ME" all flagged negative with 90%+ confidence — something TextBlob missed entirely
- **Cancellation keyword analysis** — "Retrospot" (1,400), "Cake" (1,227), "Glass" (1,074) dominate cancelled orders

| Sentiment | Products | Example |
|---|---|---|
| Positive | 140 | I LOVE LONDON MINI BACKPACK (0.988) |
| Neutral | 5,133 | REGENCY CAKESTAND 3 TIER (0.826) |
| Negative | 58 | YOU'RE CONFUSING ME METAL SIGN (0.930) |

---

## 🐳 Docker

Application is fully containerised for platform-independent deployment:

```bash
# Build
docker build -t ecommerce-intelligence.

# Run
docker run -p 8501:8501 ecommerce-intelligence
```

Then open `http://localhost:8501` in your browser.

---

## Layer 5 — Streamlit App

5 interactive tabs deployed on Streamlit Cloud:

| Tab | Features |
|---|---|
| 📊 Overview | KPI cards, monthly revenue trend, top countries, top products |
| 🔮 Churn Predictor | Input customer params → gauge chart → churn probability + recommended actions |
| 👥 Customer Segments | Segment KPIs, pie chart, RFM scatter, segment explorer with drill-down |
| 🎯 Recommendations | Select customer → see purchase history + top N recommended products |
| 💡 Key Insights | HuggingFace sentiment analysis + 8 data-driven business findings |

---

## Key Business Insights
1. **4 VIP Whale accounts** average £436K spend each — losing one = £436K revenue impact
2. **November seasonal spike** — £1.17M in Nov 2010, nearly double any other month. Q4 drives ~35% of annual revenue
3. **£916K revenue leakage** from UK cancellations across 6,959 orders
4. **2,012 hibernating customers** inactive 462 days on average — ~£1.5M in recoverable dormant revenue
5. **F_Score is the dominant churn signal** — 5x more predictive than monetary value per SHAP analysis
6. **Netherlands customer ranked #2 globally** at £528K — international accounts punch above their weight
7. **REGENCY CAKESTAND 3 TIER** is the #1 product — £286K revenue across 1,314 unique customers
8. **Dec 2009 cohort retained 35%** in month 1 but stabilised at 300–400 — classic wholesale irregular buying pattern

---

## Tech Stack
| Category | Tools |
|---|---|
| Data Processing | Python, Pandas, NumPy, Google Colab |
| Database | Microsoft SQL Server (MSSQL), SSMS |
| Machine Learning | Scikit-learn, XGBoost, SHAP, K-Means, Cosine Similarity |
| NLP / AI | HuggingFace Transformers, RoBERTa |
| Visualisation | Plotly, Matplotlib, Seaborn |
| Deployment | Docker, Streamlit, Streamlit Cloud, GitHub |

---

## 📁 Repository Structure
ecommerce-intelligence-platform/

│

├── app.py                    ← Streamlit application (5 tabs)

├── requirements.txt          ← Python dependencies

├── Dockerfile                ← Container configuration

├── .dockerignore             ← Docker build exclusions

├── retail_clean.csv.gz       ← Cleaned transaction data (compressed)

├── rfm_segmented.csv         ← RFM scores + cluster labels

├── purchase_matrix.csv       ← Customer-product matrix (top 500 products)

├── xgb_model.pkl             ← Trained XGBoost churn model

├── scaler.pkl                ← StandardScaler for feature preprocessing

│

└── .streamlit/

  └── config.toml           ← Dark theme configuration

---

## 🔗 Links
- **Live App:** https://ecommerce-intelligence-platform-ar3njx2ozilgoka4qdliqw.streamlit.app/
- **Dataset:** [Online Retail II — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
