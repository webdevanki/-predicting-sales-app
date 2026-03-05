import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import shap


def engineer_features(df_raw):
    """Odtwarza feature engineering z notebooka – musi być identyczny."""
    df_fe = df_raw.copy()
    drop_cols = ['ID', 'Komentarz']
    df_fe.drop(columns=[c for c in drop_cols if c in df_fe.columns], inplace=True)

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

st.set_page_config(page_title="Predykcja płatności klientów", layout="wide")
st.title("Predykcja płatności klientów")

# --- Tryb działania ---
MODEL_PATH = "model_pipeline.pkl"
model_loaded = os.path.exists(MODEL_PATH)

if model_loaded:
    st.success(f"Model załadowany: `{MODEL_PATH}`")
    mode = st.radio(
        "Tryb działania:",
        ["Wgraj gotowe predykcje (CSV)", "Wgraj surowe dane i przewiduj na żywo"],
        horizontal=True,
    )
else:
    st.warning("Nie znaleziono `model_pipeline.pkl`. Dostępny tylko tryb z gotowym CSV. Uruchom najpierw notebook `predict.ipynb`.")
    mode = "Wgraj gotowe predykcje (CSV)"

st.divider()

# --- Upload pliku ---
if mode == "Wgraj surowe dane i przewiduj na żywo":
    uploaded_file = st.file_uploader("Wgraj surowe dane zamówień (CSV)", type=["csv"])
else:
    uploaded_file = st.file_uploader("Wgraj plik predykcji (np. predykcje.csv)", type=["csv"])

if not uploaded_file:
    st.info("Wgraj plik CSV, aby zobaczyć analizę.")
    st.stop()

df = pd.read_csv(uploaded_file)

# --- Predykcja na żywo ---
if mode == "Wgraj surowe dane i przewiduj na żywo" and model_loaded:
    pipeline = joblib.load(MODEL_PATH)

    try:
        df['Zapłacono'] = df['Zapłacono'].astype(str).str.replace(',', '.').astype(float)
    except Exception:
        pass

    df_fe = engineer_features(df)
    X = df_fe.drop(columns=[c for c in ['Zapłacono'] if c in df_fe.columns])

    with st.spinner("Uruchamiam model..."):
        scores = pipeline.predict(X)

    df['prediction_score'] = scores.round(2)
    threshold = np.median(scores)
    df['prediction_label'] = (scores > threshold).astype(int)

    st.caption(f"Próg klasyfikacji (mediana predykcji): **{threshold:.2f} PLN**")

    # Zachowaj X i pipeline do sekcji SHAP
    st.session_state['X'] = X
    st.session_state['pipeline'] = pipeline

# --- Walidacja kolumn ---
if 'prediction_label' not in df.columns:
    st.error("Brak kolumny `prediction_label`. Upewnij się, że dane pochodzą z modelu lub notebooka.")
    st.stop()

# --- Podgląd danych ---
st.subheader("Dane z predykcjami")
st.dataframe(df, use_container_width=True)

# --- Metryki ogólne ---
st.subheader("Podsumowanie predykcji")
total = len(df)
count_paid = int((df['prediction_label'] == 1).sum())
count_unpaid = int((df['prediction_label'] == 0).sum())

col1, col2, col3 = st.columns(3)
col1.metric("Wszystkie rekordy", total)
col2.metric("Przewidziane jako zapłacone", count_paid, f"{count_paid / total * 100:.1f}%")
col3.metric("Przewidziane jako niezapłacone", count_unpaid, f"-{count_unpaid / total * 100:.1f}%", delta_color="inverse")

# --- Rozkład prediction_score (jeśli dostępny) ---
if 'prediction_score' in df.columns:
    st.subheader("Rozkład przewidywanych wartości płatności")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(df['prediction_score'], bins=40, color='steelblue', edgecolor='black')
    ax.axvline(df['prediction_score'].median(), color='red', linestyle='--', label=f"Mediana: {df['prediction_score'].median():.2f} PLN")
    ax.set_xlabel("Przewidziana wartość (PLN)")
    ax.set_ylabel("Liczba zamówień")
    ax.legend()
    st.pyplot(fig)

# --- Boxplot wartości wg klasy ---
if 'Zapłacono' in df.columns:
    st.subheader("Rozkład wartości sprzedaży wg predykcji")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(x='prediction_label', y='Zapłacono', data=df, ax=ax, palette=['coral', 'steelblue'])
    ax.set_xticklabels(['Niezapłacone (0)', 'Zapłacone (1)'])
    ax.set_ylabel("Zapłacono (PLN)")
    st.pyplot(fig)

# --- Top sprzedawcy z niezapłaconymi ---
if 'Sprzedawca' in df.columns:
    st.subheader("Top 10 sprzedawców z największą liczbą niezapłaconych zamówień")
    sellers = df[df['prediction_label'] == 0]['Sprzedawca'].value_counts().head(10)
    st.bar_chart(sellers)

# --- Branże z ryzykiem ---
if 'Branża' in df.columns:
    st.subheader("Branże z największą liczbą przewidywanych niezapłaconych")
    branze = df[df['prediction_label'] == 0]['Branża'].value_counts().head(10)
    st.bar_chart(branze)

# --- Tabela: najdroższe niezapłacone ---
if 'Zapłacono' in df.columns:
    st.subheader("Najbardziej wartościowe zamówienia przewidywane jako niezapłacone")
    display_cols = [c for c in ['Nazwa klienta', 'Zapłacono', 'prediction_score', 'Sprzedawca'] if c in df.columns]
    df_unpaid = df[df['prediction_label'] == 0].sort_values('Zapłacono', ascending=False).head(20)
    st.dataframe(df_unpaid[display_cols], use_container_width=True)

# --- SHAP – interpretacja modelu ---
# Działa jeśli model_pipeline.pkl istnieje – niezależnie od trybu
_shap_pipeline = st.session_state.get('pipeline') or (joblib.load(MODEL_PATH) if model_loaded else None)
_shap_X = st.session_state.get('X')

# W trybie CSV odtwórz X z wgranego df jeśli pipeline jest dostępny
if _shap_pipeline is not None and _shap_X is None:
    try:
        df_fe = engineer_features(df)
        drop_target = [c for c in ['Zapłacono', 'prediction_score', 'prediction_label'] if c in df_fe.columns]
        _shap_X = df_fe.drop(columns=drop_target, errors='ignore')
    except Exception:
        _shap_X = None

if _shap_pipeline is not None and _shap_X is not None:
    st.divider()
    st.subheader("Interpretacja modelu – SHAP")
    st.caption("SHAP pokazuje które cechy i jak wpływają na każdą predykcję. Czerwony = wysoka wartość cechy, niebieski = niska.")

    try:
        with st.spinner("Obliczam SHAP values..."):
            _preprocessor = _shap_pipeline.named_steps['preprocessor']
            _model = _shap_pipeline.named_steps['model']
            _X_transformed = _preprocessor.transform(_shap_X)

            num_cols_shap = list(_preprocessor.transformers_[0][2])
            try:
                cat_cols_shap = _preprocessor.transformers_[1][2]
                cat_feature_names = _preprocessor.named_transformers_['cat'] \
                    .named_steps['encoder'].get_feature_names_out(cat_cols_shap).tolist()
            except Exception:
                cat_feature_names = []

            feature_names = num_cols_shap + cat_feature_names
            explainer = shap.TreeExplainer(_model)
            shap_values = explainer(_X_transformed)
            shap_values.feature_names = feature_names

        col_shap1, col_shap2 = st.columns(2)

        with col_shap1:
            st.markdown("**Globalny wpływ cech (Beeswarm)**")
            plt.figure()
            shap.plots.beeswarm(shap_values, max_display=10, show=False)
            st.pyplot(plt.gcf())
            plt.close()

        with col_shap2:
            st.markdown("**Wyjaśnienie pojedynczej predykcji (Waterfall)**")
            sample_idx = st.slider("Wybierz rekord", 0, len(_shap_X) - 1, 0)
            plt.figure()
            shap.plots.waterfall(shap_values[sample_idx], max_display=10, show=False)
            st.pyplot(plt.gcf())
            plt.close()

    except Exception as e:
        st.warning(f"Nie udało się wygenerować SHAP: {e}")
