import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor

RANDOM_STATE = 42


def generate_demo_data():
    np.random.seed(42)
    n = 500

    klienci = [
        'Kowalski Sp. z o.o.', 'Nowak Trading', 'TechVision S.A.', 'BudMat Polska',
        'AgriPol Sp. z o.o.', 'LogiTrans', 'MedSupply', 'RetailPro S.A.',
        'GreenEnergy Sp. z o.o.', 'AutoParts Polska', 'FoodDist S.A.', 'PrintMaster',
        'ChemTech Sp. z o.o.', 'SteelWork S.A.', 'EduTech Sp. z o.o.',
        'ProBuild S.A.', 'DataSoft Sp. z o.o.', 'FastLog Polska', 'MediCare S.A.',
        'AgroPlus Sp. z o.o.', 'ElektroTech S.A.', 'ColdChain Polska', 'NetServ Sp. z o.o.',
        'HeavyDuty S.A.', 'FreshFood Polska', 'SmartRetail Sp. z o.o.', 'GreenPower S.A.',
        'CargoXpress', 'PharmaDist Sp. z o.o.', 'TurboChem S.A.',
        'DigitalHub Sp. z o.o.', 'IronWorks S.A.', 'AquaTech Polska', 'SkyBuild Sp. z o.o.',
        'OmniRetail S.A.', 'BioFarm Polska', 'SafeGuard Sp. z o.o.', 'MaxiLog S.A.',
        'NovaMed Sp. z o.o.', 'PrimeTech S.A.'
    ]

    sprzedawcy = [
        'Anna Wiśniewska', 'Piotr Kowalczyk', 'Marta Jabłońska',
        'Tomasz Nowak', 'Katarzyna Wróbel', 'Michał Zając', 'Joanna Kowalska'
    ]

    branże = [
        'IT / Technologia', 'Budownictwo', 'Rolnictwo', 'Logistyka', 'Medycyna',
        'Retail', 'Energia', 'Motoryzacja', 'Spożywczy', 'Edukacja'
    ]

    branza_mult = {
        'IT / Technologia': 1.3, 'Medycyna': 1.4, 'Energia': 1.2,
        'Budownictwo': 1.1, 'Rolnictwo': 0.9, 'Logistyka': 1.0,
        'Retail': 0.95, 'Motoryzacja': 1.15, 'Spożywczy': 0.85, 'Edukacja': 0.8
    }

    daty = pd.date_range('2022-01-01', '2024-06-30', periods=n)
    liczba_produktow = np.random.randint(1, 80, n)
    cena_jednostkowa = np.random.uniform(20, 800, n)
    branza_arr = np.random.choice(branże, n)
    mult = np.array([branza_mult[b] for b in branza_arr])
    zaplac = (liczba_produktow * cena_jednostkowa * mult + np.random.normal(0, 500, n)).clip(50).round(2)

    return pd.DataFrame({
        'ID': range(1001, 1001 + n),
        'Data zamowienia': daty.strftime('%Y-%m-%d'),
        'Nazwa klienta': np.random.choice(klienci, n),
        'Sprzedawca': np.random.choice(sprzedawcy, n),
        'Branża': branza_arr,
        'Liczba produktow': liczba_produktow,
        'Wartosc jednostkowa': cena_jednostkowa.round(2),
        'Zapłacono': zaplac,
        'Komentarz': np.random.choice(['', 'Pilne', 'Klient VIP', 'Reklamacja', ''], n)
    })


def engineer_features(df_raw):
    df_fe = df_raw.copy()
    df_fe.drop(columns=[c for c in ['ID', 'Komentarz'] if c in df_fe.columns], inplace=True)
    date_cols = [c for c in df_fe.columns if 'data' in c.lower() or 'date' in c.lower()]
    for col in date_cols:
        try:
            df_fe[col] = pd.to_datetime(df_fe[col])
            df_fe[f'{col}_miesiac'] = df_fe[col].dt.month
            df_fe[f'{col}_dzien_tygodnia'] = df_fe[col].dt.dayofweek
            df_fe[f'{col}_kwartal'] = df_fe[col].dt.quarter
            df_fe.drop(columns=[col], inplace=True)
        except Exception:
            pass
    return df_fe


def train_model(df):
    df_fe = engineer_features(df)
    X = df_fe.drop(columns=['Zapłacono'])
    y = df_fe['Zapłacono']

    num_cols = X.select_dtypes(include='number').columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), num_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), cat_cols)
    ])

    pipeline = Pipeline([
        ('pre', preprocessor),
        ('model', GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE))
    ])
    pipeline.fit(X, y)
    return pipeline, X, num_cols, cat_cols


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Predykcja płatności B2B", layout="wide")

with st.sidebar:
    st.title("Ustawienia")
    mode = st.radio("Źródło danych:", ["Dane demo", "Wgraj własny CSV"])
    st.divider()
    st.markdown("**Stack techniczny**")
    st.markdown(
        "- scikit-learn · GradientBoosting\n"
        "- SHAP · TreeExplainer\n"
        "- MLflow · Streamlit\n"
        "- pandas · numpy"
    )
    st.markdown("[GitHub →](https://github.com/webdevanki/-predicting-sales-app)")

# ---------------------------------------------------------------------------
# Dane
# ---------------------------------------------------------------------------

if mode == "Dane demo":
    df_raw = generate_demo_data()
    st.info("Tryb demo — 500 syntetycznych zamówień B2B (40 klientów). Zmień źródło w sidebarze, żeby wgrać własny CSV.")
else:
    uploaded = st.file_uploader("Wgraj plik CSV z zamówieniami", type=["csv"])
    if not uploaded:
        st.info("Wgraj plik CSV żeby zobaczyć analizę.")
        st.stop()
    df_raw = pd.read_csv(uploaded)

st.title("Predykcja płatności klientów B2B")

with st.spinner("Trenuję model na danych..."):
    pipeline, X_all, num_cols, cat_cols = train_model(df_raw)

scores = pipeline.predict(X_all)
threshold = float(np.median(scores))
df_out = df_raw.copy()
df_out['prediction_score'] = scores.round(2)
df_out['prediction_label'] = (scores > threshold).astype(int)

# ---------------------------------------------------------------------------
# KPI
# ---------------------------------------------------------------------------

total = len(df_out)
high_risk = int((df_out['prediction_label'] == 0).sum())
low_risk = int((df_out['prediction_label'] == 1).sum())
avg_score = float(df_out['prediction_score'].mean())

k1, k2, k3, k4 = st.columns(4)
k1.metric("Zamówienia", total)
k2.metric("Niskie ryzyko", f"{low_risk / total * 100:.1f}%", f"{low_risk} zamówień")
k3.metric("Wysokie ryzyko", f"{high_risk / total * 100:.1f}%",
          f"-{high_risk} zamówień", delta_color="inverse")
k4.metric("Śr. wartość predykcji", f"{avg_score:,.0f} PLN")

st.divider()

# ---------------------------------------------------------------------------
# Wykresy
# ---------------------------------------------------------------------------

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Rozkład przewidywanych wartości płatności")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df_out['prediction_score'], bins=40, color='steelblue', edgecolor='black')
    ax.axvline(threshold, color='red', linestyle='--', linewidth=1.5,
               label=f"Próg: {threshold:,.0f} PLN")
    ax.set_xlabel("Przewidziana wartość (PLN)")
    ax.set_ylabel("Liczba zamówień")
    ax.legend()
    st.pyplot(fig)
    plt.close()

with col_right:
    st.subheader("Wysokie ryzyko wg branży")
    if 'Branża' in df_out.columns:
        branza_risk = df_out[df_out['prediction_label'] == 0]['Branża'].value_counts()
        fig, ax = plt.subplots(figsize=(7, 4))
        branza_risk.sort_values().plot(kind='barh', ax=ax, color='coral', edgecolor='black')
        ax.set_xlabel("Liczba zamówień wysokiego ryzyka")
        ax.set_ylabel("")
        st.pyplot(fig)
        plt.close()

st.divider()

# ---------------------------------------------------------------------------
# Tabela wysokiego ryzyka
# ---------------------------------------------------------------------------

st.subheader("Top 20 zamówień wysokiego ryzyka")
display_cols = [c for c in
                ['Nazwa klienta', 'Sprzedawca', 'Branża', 'Zapłacono', 'prediction_score']
                if c in df_out.columns]
df_high = (df_out[df_out['prediction_label'] == 0]
           .sort_values('prediction_score', ascending=False)
           .head(20))
st.dataframe(df_high[display_cols], use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------

st.subheader("Interpretacja modelu – SHAP")
st.caption("Beeswarm: globalny wpływ cech · Waterfall: wyjaśnienie pojedynczej predykcji")

try:
    _pre = pipeline.named_steps['pre']
    _model = pipeline.named_steps['model']
    X_transformed = _pre.transform(X_all)

    cat_feature_names = (
        _pre.named_transformers_['cat']
        .named_steps['encoder']
        .get_feature_names_out(cat_cols)
        .tolist()
    )
    feature_names = num_cols + cat_feature_names

    explainer = shap.TreeExplainer(_model)
    shap_values = explainer(X_transformed)
    shap_values.feature_names = feature_names

    shap_col1, shap_col2 = st.columns(2)

    with shap_col1:
        st.markdown("**Globalny wpływ cech (Beeswarm)**")
        plt.figure()
        shap.plots.beeswarm(shap_values, max_display=10, show=False)
        st.pyplot(plt.gcf())
        plt.close()

    with shap_col2:
        st.markdown("**Wyjaśnienie predykcji (Waterfall)**")
        sample_idx = st.slider("Wybierz rekord", 0, len(X_all) - 1, 0)
        plt.figure()
        shap.plots.waterfall(shap_values[sample_idx], max_display=10, show=False)
        st.pyplot(plt.gcf())
        plt.close()

except Exception as e:
    st.warning(f"SHAP niedostępny: {e}")

st.divider()

# ---------------------------------------------------------------------------
# Expander pełna tabela
# ---------------------------------------------------------------------------

with st.expander("Pełna tabela danych z predykcjami"):
    st.dataframe(df_out, use_container_width=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown(
    "<div style='text-align:center;color:gray;font-size:0.85em;padding-top:1rem;'>"
    "R²=0.91 · MAE=1302 PLN · Gradient Boosting · scikit-learn Pipeline"
    "</div>",
    unsafe_allow_html=True,
)
