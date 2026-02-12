import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Predykcja płatności klientów", layout="wide")
st.title("📊 Predykcja płatności klientów")

uploaded_file = st.file_uploader("Wgraj plik predykcyjny (np. predykcje.csv)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("🔍 Dane z predykcjami")
    st.dataframe(df, use_container_width=True)

    if 'prediction_label' not in df.columns:
        st.warning("Brakuje kolumny 'prediction_label'. Upewnij się, że dane pochodzą z predict_model().")
        st.stop()

    # Podsumowanie klas
    st.subheader("📈 Podsumowanie klas predykcji")
    total = len(df)
    count_paid = sum(df['prediction_label'] == 1)
    count_unpaid = sum(df['prediction_label'] == 0)
    st.metric("Wszystkie rekordy", total)
    st.metric("Przewidziane jako zapłacone", count_paid)
    st.metric("Przewidziane jako niezapłacone", count_unpaid)
    st.write(f"Procent zapłaconych: {count_paid / total * 100:.2f}%")
    st.write(f"Procent niezapłaconych: {count_unpaid / total * 100:.2f}%")

    # Wartość sprzedaży wg predykcji
    st.subheader("💰 Rozkład wartości sprzedaży wg predykcji")
    if 'Zapłacono' in df.columns:
        fig, ax = plt.subplots()
        sns.boxplot(x='prediction_label', y='Zapłacono', data=df, ax=ax)
        ax.set_xticklabels(['Niezapłacone (0)', 'Zapłacone (1)'])
        ax.set_ylabel("Zapłacone")
        st.pyplot(fig)
    else:
        st.info("Brak kolumny 'Zapłacono' – nie można pokazać rozkładu wartości.")

    # Sprzedawcy z największą liczbą przewidywanych niezapłaconych
    st.subheader("📍 Top 10 sprzedawców z największą liczbą niezapłaconych klientów")
    if 'Sprzedawca' in df.columns:
        sellers = df[df['prediction_label'] == 0]['Sprzedawca'].value_counts().head(10)
        st.bar_chart(sellers)
    else:
        st.info("Brak kolumny 'Sprzedawca' – nie można wygenerować wykresu.")

    # Branże z największym ryzykiem
    st.subheader("🏭 Branże z największą liczbą przewidywanych niezapłaconych")
    if 'Branża' in df.columns:
        branże = df[df['prediction_label'] == 0]['Branża'].value_counts().head(10)
        st.bar_chart(branże)
    else:
        st.info("Brak kolumny 'Branża' – nie można wygenerować wykresu.")

    # Najbardziej wartościowe zamówienia niezapłacone
    st.subheader("⚠️ Najbardziej wartościowe zamówienia przewidywane jako niezapłacone")
    if 'Zapłacono' in df.columns:
        df_unpaid = df[df['prediction_label'] == 0]
        df_unpaid_sorted = df_unpaid.sort_values(by='Zapłacono', ascending=False).head(20)
        st.dataframe(df_unpaid_sorted[['Nazwa klienta', 'Zapłacono', 'Sprzedawca']])
    else:
        st.info("Brak kolumny 'Zapłacono' – nie można pokazać tabeli.")

else:
    st.info("⬆️ Wgraj plik `predykcje.csv`, aby zobaczyć analizę.")
