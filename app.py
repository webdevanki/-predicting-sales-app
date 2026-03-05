import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

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

    drop_cols = [c for c in ['ID', 'Komentarz'] if c in df.columns]
    X = df.drop(columns=drop_cols + (['Zapłacono'] if 'Zapłacono' in df.columns else []))

    with st.spinner("Uruchamiam model..."):
        scores = pipeline.predict(X)

    df['prediction_score'] = scores.round(2)
    threshold = np.median(scores)
    df['prediction_label'] = (scores > threshold).astype(int)

    st.caption(f"Próg klasyfikacji (mediana predykcji): **{threshold:.2f} PLN**")

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
