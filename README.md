# B2B Payment Risk — ML Dashboard

End-to-end machine learning project for predicting B2B payment values and classifying payment risk. Built to support sales teams in prioritizing high-risk accounts before payment deadlines.

## Dashboard

A Streamlit app with four tabs, each targeting a different role:

| Tab | Audience | Content |
|-----|----------|---------|
| 🎯 Działania | Sales rep (daily) | Priority-scored action list — who to call and why |
| 📊 Analiza | Manager | Risk breakdown by industry and salesperson |
| 📅 Cash flow | Finance | Upcoming payments at risk, 30-day timeline |
| 🔬 Model | Data science | SHAP explainability, metrics, raw data |

### 🎯 Działania — priority action list

![Działania tab](imgs/1.png)

Orders are scored by **Priority Score** (urgency 40% + payment value 40% + client repeat risk 20%) and grouped into three tiers:

- 🔴 **Krytyczne** — overdue or high value, act immediately
- 🟠 **Pilne** — payment due within 7 days
- 🟡 **Ważne** — payment due within 30 days

Each card shows the client, industry, responsible salesperson, amount at risk, days to payment deadline, and an auto-generated contact reason.

### 📊 Analiza — risk analysis

![Analiza tab](imgs/2.png)

### 📅 Cash flow — payment timeline

![Cash flow tab](imgs/3.png)

### 🔬 Model — explainability

![Model tab](imgs/4.png)

## Problem Statement

In B2B sales, late or missing payments directly impact cash flow. This project predicts the expected payment value per order and classifies orders as **high risk** (predicted value below a configurable PLN threshold) — enabling proactive intervention before payment deadlines.

## ML Approach

1. **EDA** — distribution analysis, missing data audit, correlation heatmap, target variable inspection
2. **A/B Testing** — statistical group comparison (Mann-Whitney U, Cohen's d, Shapiro-Wilk)
3. **Feature Engineering** — temporal features from order dates (month, quarter, day of week)
4. **sklearn Pipeline** — `ColumnTransformer` for parallel preprocessing; prevents data leakage during cross-validation
5. **Model Selection** — Ridge (baseline), Random Forest, Gradient Boosting via 5-fold CV on MAE and R²
6. **Evaluation** — hold-out test set, residuals plot, actual vs predicted scatter
7. **SHAP** — beeswarm and waterfall plots for global and per-prediction explainability
8. **Experiment Tracking** — MLflow logs parameters, metrics and artifacts

## Results

| Metric | Value |
|--------|-------|
| Best model | Gradient Boosting |
| CV MAE (5-fold) | 1 157.87 PLN |
| Test MAE | 1 301.65 PLN |
| Test RMSE | 1 963.11 PLN |
| Test R² | 0.910 |

## Project Structure

```
├── predict.ipynb          # ML pipeline: EDA, A/B testing, training, SHAP, MLflow
├── app.py                 # Streamlit dashboard — priority scoring, SHAP, cash flow
├── generate_test_data.py  # Generates synthetic dataset for demo purposes
├── zamowienia_testowe.csv # Sample dataset (150 orders, 9 features)
├── imgs/                  # Dashboard screenshots
│   ├── 1.png              # Tab: Działania
│   ├── 2.png              # Tab: Analiza
│   ├── 3.png              # Tab: Cash flow
│   └── 4.png              # Tab: Model
├── .streamlit/config.toml # Light corporate theme
├── requirements.txt
└── README.md
```

## Quickstart

```bash
pip install -r requirements.txt

# Run dashboard (uses built-in demo data)
streamlit run app.py

# Or train + explore in notebook
jupyter notebook predict.ipynb

# View MLflow experiment history
mlflow ui
```

## Tech Stack

| Layer | Tools |
|-------|-------|
| Data processing | pandas, numpy |
| Statistics | scipy (Mann-Whitney U, Shapiro-Wilk, Cohen's d) |
| ML | scikit-learn (Pipeline, ColumnTransformer, GradientBoosting) |
| Explainability | SHAP (TreeExplainer, beeswarm, waterfall) |
| Experiment tracking | MLflow |
| Visualization | Plotly, matplotlib |
| Dashboard | Streamlit |
| Model persistence | joblib |
