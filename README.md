# Customer Payment Prediction – ML Pipeline

End-to-end machine learning project for predicting B2B payment values and classifying payment risk. Built to support sales teams in prioritizing high-risk accounts.

## Problem Statement

In B2B sales, late or missing payments directly impact cash flow. This project predicts the expected payment value per order and classifies orders as high/low risk – enabling proactive intervention before payment deadlines.

## Approach

1. **EDA** – distribution analysis, missing data audit, target variable inspection
2. **Feature Engineering** – temporal features from order dates (month, quarter, day of week), domain-informed numeric features
3. **sklearn Pipeline** – `ColumnTransformer` for parallel preprocessing of numeric and categorical features; full pipeline prevents data leakage during cross-validation
4. **Model Selection** – Ridge (baseline), Random Forest, Gradient Boosting evaluated via 5-fold cross-validation on MAE and R²
5. **Evaluation** – hold-out test set, residuals plot, actual vs predicted scatter
6. **Feature Importance** – tree-based feature importances to interpret model decisions
7. **Deployment** – pipeline serialized with `joblib`; Streamlit app supports both batch CSV upload and live inference

## Results

| Metric | Value |
|--------|-------|
| Best model | Gradient Boosting / Random Forest (CV-selected) |
| Evaluation | 5-fold cross-validation + hold-out test set |
| Output | MAE, RMSE, R² reported per run |

> Exact metrics depend on input data. Run `predict.ipynb` to reproduce.

## Project Structure

```
├── predict.ipynb          # ML pipeline: EDA, feature engineering, training, evaluation
├── app.py                 # Streamlit dashboard – batch predictions or live inference
├── generate_test_data.py  # Generates synthetic dataset for demo purposes
├── zamowienia_testowe.csv # Sample dataset (150 orders, 9 features)
└── README.md
```

## Quickstart

```bash
pip install -r requirements.txt

# 1. Generate sample data (or use your own zamowienia.csv)
python generate_test_data.py

# 2. Run notebook to train model and export predictions
jupyter notebook predict.ipynb

# 3. Launch dashboard
streamlit run app.py
```

## Tech Stack

| Layer | Tools |
|-------|-------|
| Data processing | pandas, numpy |
| ML | scikit-learn (Pipeline, ColumnTransformer, GradientBoosting, RandomForest, Ridge) |
| Evaluation | cross_val_score, MAE, RMSE, R² |
| Visualization | matplotlib, seaborn |
| Dashboard | Streamlit |
| Model persistence | joblib |
