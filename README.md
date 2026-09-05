# 📊 FRED Economic Intelligence Agent

An AI-powered Economic Intelligence Agent that analyzes U.S. economic indicators, uses Machine Learning to detect recessionary conditions, and uses Generative AI (Google Gemini) to explain findings in plain business language.

**Live App:** https://fred-economic-intelligence-agent-goon.streamlit.app

---

## 1. Problem Statement

**Business Problem:**
Economic indicators are published across multiple datasets and frequencies, making it difficult for decision-makers to quickly understand the overall economic situation.

**Objective:**
Develop an AI-powered Economic Intelligence Agent that integrates economic indicators, applies Machine Learning to identify recessionary signals, and uses Generative AI to explain the results in a concise, business-oriented manner.

**Target Users:** Economists, financial analysts, business leaders, policy analysts, and decision-makers.

**Project Scope:** Level 1 — Minimum (core pipeline). Level 2 (forecasting, anomaly detection, agent/tool-calling) and Level 3 (advanced features) are planned as future work.

---

## 2. Dataset Sources

Data sourced from the **Federal Reserve Economic Data (FRED)** API via the `fredapi` Python library.

| # | Indicator | FRED Series | Frequency |
|---|---|---|---|
| 1 | Unemployment Rate | `UNRATE` | Monthly |
| 2 | CPI (All Items) | `CPIAUCSL` | Monthly |
| 3 | Industrial Production Index | `INDPRO` | Monthly |
| 4 | Federal Funds Rate | `FEDFUNDS` | Monthly |
| 5 | 10-Year Treasury Yield | `DGS10` | Daily |
| 6 | 2-Year Treasury Yield | `DGS2` | Daily |
| 7 | Real GDP | `GDPC1` | Quarterly |
| 8 | NBER Recession Indicator (target) | `USREC` | Monthly |

Full details in [`data/indicator_dictionary.xlsx`](data/indicator_dictionary.xlsx).

---

## 3. Data Collection & Cleaning

**Notebook:** [`notebooks/01_data_collection.ipynb`](notebooks/01_data_collection.ipynb)

**Steps performed:**
1. Pulled all 8 series via FRED API (key secured via Colab Secrets, never hardcoded).
2. Resampled to a unified **monthly** frequency:
   - Daily yields (`DGS10`, `DGS2`) → monthly average
   - Quarterly GDP (`GDPC1`) → forward-filled to monthly
   - Monthly series aligned to month-start
3. Created **4 derived indicators**:
   - `Yield Curve Spread` = 10Y − 2Y Treasury Yield
   - `Inflation Rate` = YoY % change in CPI
   - `Unemployment Change` = month-over-month change in unemployment rate
   - `Industrial Production Growth` = YoY % change in Industrial Production Index
4. **Publication-lag correction (key data-quality fix):** Some monthly series (CPI, Industrial Production) publish with a reporting lag. Naively forward-filling created artificial flat/repeated values in the most recent months. Fixed by computing the true last-reported date per series and trimming the dataset to the earliest confirmed date across all core monthly series.

**Final dataset:** 590 rows × 13 columns, spanning **1977-06-01 to 2026-07-01**, zero duplicate rows, ~9.8% positive recession class (58 recession-months out of 592 pre-trim).

**Artifacts:**
- [`data/processed/economic_dataset.csv`](data/processed/economic_dataset.csv)
- [`data/indicator_dictionary.xlsx`](data/indicator_dictionary.xlsx)

---

## 4. Exploratory Data Analysis (EDA)

**Notebook:** [`notebooks/02_eda.ipynb`](notebooks/02_eda.ipynb)

**Visualizations produced:**
1. Unemployment, Inflation, and Industrial Production Growth trends — each with recession periods shaded for visual correlation
2. Correlation heatmap across 7 core indicators
3. Yield curve visualization (10Y, 2Y yields + spread, with inversion threshold marked)
4. Distribution analysis (mean, median, std dev, quartiles, IQR-based outlier counts) for 5 core indicators

**Key EDA findings:**
- **Industrial Production Growth** showed the strongest contemporaneous correlation with recession status (**r = −0.47**).
- **Yield Curve Spread vs. Recession (same month): r = 0.01** — essentially no relationship.
- **Yield Curve Spread vs. Recession within the next 12 months: r = −0.33** — confirming the yield curve is a genuine **leading indicator**, not a same-month signal. This finding directly shaped feature engineering in the ML phase (see Section 5).
- Inflation Rate showed 57 statistical outliers (IQR method) — these reflect real historical regime shifts (1980–82 and 2021–22 inflation spikes), not data errors.

---

## 5. Machine Learning — Recession Classification

**Notebook:** [`notebooks/03_recession_model.ipynb`](notebooks/03_recession_model.ipynb)

### 5.1 Target & Approach
- **Target:** `Recession` (binary: 1 = NBER-dated recession month, 0 = otherwise)
- **Validation strategy:** `TimeSeriesSplit` (5 folds), **not** random shuffling — recessions are rare and clustered, so a random split (or even a single fixed 80/20 split) risks leaving some folds with near-zero positive examples. Confirmed empirically: one fixed 80/20 split produced a test set with only 1.7% recession rate (2 recession-months), too few to trust.

### 5.2 Avoiding Data Leakage
Following the explicit requirement to never use future information to predict the past, all features were **lagged**:
- Unemployment Change, Inflation Rate, Industrial Production Growth → lag-1 month
- **Yield Curve Spread → lag-12 months** (not lag-1)

The 12-month lag for Yield Curve Spread was a deliberate correction, not an initial choice — see Section 5.4 below.

### 5.3 Model Comparison
Four models were evaluated via out-of-fold cross-validated predictions:

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **Logistic Regression** | 0.947 | 0.727 | 0.444 | 0.552 |
| Gradient Boosting | 0.888 | 0.148 | 0.111 | 0.127 |
| Decision Tree | 0.890 | 0.125 | 0.083 | 0.100 |
| Random Forest | 0.912 | 0.111 | 0.028 | 0.044 |

**Selected model: Logistic Regression.** Per project guidance, **recall was prioritized over raw accuracy** — missing an actual recession is more costly to a decision-maker than an occasional false alarm. Random Forest had the highest accuracy (91.2%) but the worst recall (0.028), making it practically useless for this task — a clear illustration of why accuracy alone is a misleading metric on imbalanced data.

### 5.4 Debugging Coefficient Signs (Multicollinearity)

After initial training, several coefficients had **economically backwards signs** (e.g., Yield Curve Spread positively associated with recession risk, contradicting known economic theory and our own EDA finding of r = −0.33). This was diagnosed and fixed in three iterations:

| Iteration | Change | Yield Curve Spread coefficient |
|---|---|---|
| 1 | 6 raw features, all lag-1 | +2.005 (backwards) |
| 2 | Dropped Federal Funds Rate (correlated with Inflation at r=0.70 and Yield Curve at r=−0.64) | +0.712 (still backwards) |
| 3 | Dropped Unemployment Rate (still correlated with Yield Curve at r=0.45) | +0.513 (still backwards) |
| 4 | **Root cause fix:** changed Yield Curve Spread from lag-1 to **lag-12**, matching its true leading-indicator behavior proven in EDA | **−1.249 (correct)** |

The root cause was not multicollinearity alone — it was a **lag mismatch**. A 1-month lag made the yield curve spread look contemporaneous with recession (near-zero true relationship), preventing the model from learning its real, longer-horizon predictive signal.

**Final feature set:** `Unemployment Change (lag-1)`, `Inflation Rate (lag-1)`, `Industrial Production Growth (lag-1)`, `Yield Curve Spread (lag-12)`

**Final coefficients (all economically consistent):**

| Feature | Coefficient | Direction |
|---|---|---|
| Industrial Production Growth (lag-1) | −1.370 | Lower growth → higher recession risk ✓ |
| Yield Curve Spread (lag-12) | −1.249 | Inversion 12mo ago → higher risk now ✓ |
| Unemployment Change (lag-1) | +1.142 | Rising unemployment → higher risk ✓ |
| Inflation Rate (lag-1) | +0.830 | Higher inflation → higher risk ✓ |

**Final model performance (cross-validated):**

| Metric | Score |
|---|---|
| Accuracy | 0.904 |
| Precision | 0.417 |
| **Recall** | **0.694** |
| F1 | 0.521 |

Recall improved from 0.444 → 0.694 (a ~56% relative improvement) as a direct result of fixing the yield curve lag — the coefficient-sign debugging process was not just cosmetic, it materially improved the model's ability to catch real recessions.

**Artifacts:** [`models/recession_model.pkl`](models/recession_model.pkl), [`models/recession_model_features.pkl`](models/recession_model_features.pkl)

---

## 6. Economic Intelligence Snapshot

A reusable function ([`src/snapshot.py`](src/snapshot.py)) converts the latest data + model output into a structured summary consumed by the GenAI layer:

```json
{
  "date": "2026-07-01",
  "recession_risk": "LOW",
  "recession_probability": 0.183,
  "inflation_trend": "RISING",
  "inflation_rate": 3.3,
  "unemployment_trend": "FALLING",
  "unemployment_rate": 4.1,
  "industrial_production_trend": "RISING",
  "yield_curve_status": "NORMAL",
  "yield_curve_spread": 0.38
}
```

This is the **only** data ever passed to the GenAI layer — a deliberate design choice to prevent the LLM from inventing or hallucinating statistics.

---

## 7. GenAI Layer

**Notebook:** [`notebooks/04_genai_explanation.ipynb`](notebooks/04_genai_explanation.ipynb)
**Provider:** Google Gemini (`google-genai` SDK)

Two functions built, both strictly grounded in the snapshot JSON above — verified line-by-line to confirm no invented numbers appear in outputs:

1. **`explain_economic_snapshot()`** ([`src/genai_explain.py`](src/genai_explain.py)) — converts the snapshot into a 4–6 sentence plain-English business explanation.
2. **`generate_executive_report()`** ([`src/genai_report.py`](src/genai_report.py)) — generates a full Monthly Economic Intelligence Report with 8 structured sections (Executive Summary, Employment, Inflation, Interest Rates & Yield Curve, Industrial Activity, ML Assessment, Key Risks, Key Indicators to Monitor).

**Resilience:** Both functions include automatic retry logic with model fallback (`gemini-3.6-flash` → `gemini-2.5-flash` → `gemini-2.0-flash`), since Gemini's free tier can return transient `503 UNAVAILABLE` errors under high demand. On complete failure, a user-friendly message is shown instead of a crash.

---

## 8. Streamlit Application

**File:** [`app/streamlit_app.py`](app/streamlit_app.py)
**Deployed on:** Streamlit Community Cloud → https://fred-economic-intelligence-agent-goon.streamlit.app

**Current features (Level 1):**
- Live metrics dashboard: Recession Risk, Inflation, Unemployment, Yield Curve status
- Interactive trend chart (selectable indicator)
- On-demand GenAI explanation of current recession risk
- On-demand Monthly Economic Intelligence Report generation

**Configuration:** Gemini API key stored via Streamlit Cloud Secrets (`GEMINI_API_KEY`), never committed to the repository.

---

## 9. Repository Structure

fred-economic-intelligence-agent/
├── app/
│ └── streamlit_app.py
├── data/
│ ├── indicator_dictionary.xlsx
│ └── processed/
│ └── economic_dataset.csv
├── models/
│ ├── recession_model.pkl
│ └── recession_model_features.pkl
├── notebooks/
│ ├── 01_data_collection.ipynb
│ ├── 02_eda.ipynb
│ ├── 03_recession_model.ipynb
│ ├── 04_genai_explanation.ipynb
│ └── 05_streamlit_app.ipynb
├── src/
│ ├── snapshot.py
│ ├── genai_explain.py
│ └── genai_report.py
├── requirements.txt
├── LICENSE
└── README.md


---

## 10. Limitations

1. **GDP forward-fill artifact:** Real GDP (`GDPC1`) is quarterly; monthly rows between releases repeat the last known value. This is visually flat but expected and clearly attributable to reporting frequency, not a data error.
2. **Coefficient interpretation caution:** Even after multicollinearity fixes, Logistic Regression coefficients on correlated economic features should be interpreted as directionally indicative, not as precise, independent effect sizes.
3. **Small positive-class sample:** Only 58 historical recession-months exist in the training data — real-world rarity of recessions limits the statistical power of any model trained on this data, regardless of algorithm choice.
4. **Free-tier GenAI availability:** Gemini's free API tier can return `503 UNAVAILABLE` under high demand; retry/fallback logic mitigates but does not eliminate this.
5. **Single-page app:** The current Streamlit app is a simplified single-page version. The full 5-page structure (Dashboard, Trends, ML Predictions, Agent Chat, Report Generator) is planned for a later phase.

---

## 11. Future Enhancements (Level 2 / Level 3 — Planned)

**Level 2 — Expected:**
- Inflation/unemployment forecasting model (lag-feature regression or ARIMA)
- Anomaly detection (Z-score or Isolation Forest)
- Agent / tool-calling layer (function-routing for user queries)

**Level 3 — Advanced:**
- Automated FRED data refresh (scheduled pipeline)
- Historical recession similarity / clustering
- Multi-agent architecture
- Automated monthly report generation + alerting
- Model monitoring and confidence/explanation layer

---

## 12. Tech Stack

| Component | Tool |
|---|---|
| Data Source | FRED API (`fredapi`) |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| ML | Scikit-learn (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting) |
| GenAI | Google Gemini (`google-genai`) |
| UI | Streamlit |
| Deployment | Streamlit Community Cloud |
| Development | Google Colab |
| Version Control | Git + GitHub |

---

## 13. Author

**Tamojeet** — Economic Intelligence Agent, built as a structured course project.