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


# ---------------------------------------------------------------------------
# Dane i model
# ---------------------------------------------------------------------------

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
# Konfiguracja strony
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Predykcja płatności B2B",
    layout="wide",
    page_icon="💳"
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("Ustawienia")

    mode = st.radio("Źródło danych:", ["Dane demo", "Wgraj własny CSV"])

    st.divider()

    threshold = st.slider(
        "Próg ryzyka (PLN)",
        min_value=1_000,
        max_value=50_000,
        value=8_000,
        step=500,
        format="%d PLN"
    )
    st.caption(
        f"Zamówienia z predykcją **poniżej {threshold:,} PLN** "
        "są klasyfikowane jako **wysokie ryzyko**."
    )

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
# Wczytanie danych
# ---------------------------------------------------------------------------

if mode == "Dane demo":
    df_raw = generate_demo_data()
    st.info(
        "Tryb demo — 500 syntetycznych zamówień B2B (40 klientów, 7 sprzedawców). "
        "Zmień źródło danych w sidebarze, żeby wgrać własny CSV."
    )
else:
    uploaded = st.file_uploader("Wgraj plik CSV z zamówieniami", type=["csv"])
    if not uploaded:
        st.info("Wgraj plik CSV żeby zobaczyć analizę.")
        st.stop()
    df_raw = pd.read_csv(uploaded)

# ---------------------------------------------------------------------------
# Tytuł i opis
# ---------------------------------------------------------------------------

st.title("Predykcja płatności klientów B2B")
st.markdown(
    "Model **Gradient Boosting** przewiduje wartość płatności dla każdego zamówienia "
    "i klasyfikuje je jako **niskie** lub **wysokie ryzyko** względem ustawionego progu. "
    "Dział sprzedaży może dzięki temu priorytetyzować klientów wymagających interwencji "
    "przed terminem płatności. Próg ryzyka dostosuj w sidebarze."
)
st.divider()

# ---------------------------------------------------------------------------
# Trening i predykcje
# ---------------------------------------------------------------------------

with st.spinner("Trenuję model..."):
    pipeline, X_all, num_cols, cat_cols = train_model(df_raw)

scores = pipeline.predict(X_all)
df_out = df_raw.copy()
df_out['prediction_score'] = scores.round(2)
df_out['prediction_label'] = (scores >= threshold).astype(int)  # 1=niskie, 0=wysokie ryzyko

# Statystyki segmentowe (obliczone raz, używane w wielu sekcjach)
has_branza = 'Branża' in df_out.columns
has_sprzedawca = 'Sprzedawca' in df_out.columns

if has_branza:
    branza_stats = df_out.groupby('Branża').agg(
        total=('prediction_label', 'count'),
        high_risk=('prediction_label', lambda x: (x == 0).sum())
    )
    branza_stats['risk_pct'] = branza_stats['high_risk'] / branza_stats['total'] * 100
    worst_branza = branza_stats['risk_pct'].idxmax()
    best_branza = branza_stats['risk_pct'].idxmin()

if has_sprzedawca:
    sprzedawca_stats = df_out.groupby('Sprzedawca').agg(
        total=('prediction_label', 'count'),
        high_risk=('prediction_label', lambda x: (x == 0).sum())
    )
    sprzedawca_stats['risk_pct'] = sprzedawca_stats['high_risk'] / sprzedawca_stats['total'] * 100
    worst_sprzedawca = sprzedawca_stats['risk_pct'].idxmax()

# ---------------------------------------------------------------------------
# KPI
# ---------------------------------------------------------------------------

total = len(df_out)
n_high = int((df_out['prediction_label'] == 0).sum())
n_low = int((df_out['prediction_label'] == 1).sum())
avg_score = float(df_out['prediction_score'].mean())
value_at_risk = float(df_out.loc[df_out['prediction_label'] == 0, 'prediction_score'].sum())

k1, k2, k3, k4 = st.columns(4)
k1.metric(
    "Łączna liczba zamówień", total,
    help="Wszystkie zamówienia w analizowanym zbiorze."
)
k2.metric(
    "Niskie ryzyko", f"{n_low / total * 100:.1f}%", f"{n_low} zamówień",
    help=f"Predykcja >= {threshold:,} PLN — płatność spodziewana w pełnej kwocie."
)
k3.metric(
    "Wysokie ryzyko", f"{n_high / total * 100:.1f}%", f"-{n_high} zamówień",
    delta_color="inverse",
    help=f"Predykcja < {threshold:,} PLN — zamówienia wymagające uwagi działu sprzedaży."
)
k4.metric(
    "Wartość zagrożona", f"{value_at_risk:,.0f} PLN",
    help="Łączna suma przewidywanych płatności dla zamówień wysokiego ryzyka."
)

st.divider()

# ---------------------------------------------------------------------------
# Wnioski modelu
# ---------------------------------------------------------------------------

st.subheader("Kluczowe wnioski")

ins1, ins2, ins3 = st.columns(3)

if has_branza:
    wr = branza_stats.loc[worst_branza]
    with ins1:
        st.error(
            f"**Branża najwyższego ryzyka**\n\n"
            f"**{worst_branza}**\n\n"
            f"{wr['risk_pct']:.0f}% zamówień poniżej progu "
            f"({int(wr['high_risk'])} z {int(wr['total'])})"
        )
    br = branza_stats.loc[best_branza]
    with ins2:
        st.success(
            f"**Branża najniższego ryzyka**\n\n"
            f"**{best_branza}**\n\n"
            f"{br['risk_pct']:.0f}% zamówień poniżej progu "
            f"({int(br['high_risk'])} z {int(br['total'])})"
        )

if has_sprzedawca:
    sr = sprzedawca_stats.loc[worst_sprzedawca]
    with ins3:
        st.warning(
            f"**Sprzedawca z największym ryzykiem**\n\n"
            f"**{worst_sprzedawca}**\n\n"
            f"{sr['risk_pct']:.0f}% zamówień wysokiego ryzyka "
            f"({int(sr['high_risk'])} z {int(sr['total'])})"
        )

st.divider()

# ---------------------------------------------------------------------------
# Wykresy — rząd 1: histogram + branże
# ---------------------------------------------------------------------------

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Rozkład przewidywanych wartości płatności")
    fig, ax = plt.subplots(figsize=(7, 4))
    score_min = float(df_out['prediction_score'].min())
    score_max = float(df_out['prediction_score'].max())
    ax.axvspan(score_min, threshold, alpha=0.08, color='red')
    ax.hist(df_out['prediction_score'], bins=40, color='steelblue', edgecolor='black', alpha=0.85)
    ax.axvline(threshold, color='red', linestyle='--', linewidth=2,
               label=f"Próg ryzyka: {threshold:,} PLN")
    ax.set_xlabel("Przewidziana wartość płatności (PLN)")
    ax.set_ylabel("Liczba zamówień")
    ax.legend()
    ax.set_xlim(score_min, score_max)
    st.pyplot(fig)
    plt.close()
    st.caption("Czerwone tło = strefa wysokiego ryzyka (poniżej progu).")

with col_right:
    st.subheader("Udział wysokiego ryzyka wg branży")
    if has_branza:
        risk_pct = branza_stats['risk_pct'].sort_values()
        bar_colors = [
            '#d73027' if v > 50 else '#fc8d59' if v > 30 else '#91bfdb'
            for v in risk_pct.values
        ]
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.barh(risk_pct.index, risk_pct.values, color=bar_colors, edgecolor='black')
        ax.axvline(50, color='gray', linestyle=':', linewidth=1, label='50%')
        ax.set_xlabel("% zamówień wysokiego ryzyka")
        ax.set_xlim(0, 105)
        for bar, val in zip(bars, risk_pct.values):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    f'{val:.0f}%', va='center', fontsize=9)
        ax.legend(fontsize=8)
        st.pyplot(fig)
        plt.close()
        st.caption("Czerwony > 50% · Pomarańczowy > 30% · Niebieski < 30%")

st.divider()

# ---------------------------------------------------------------------------
# Wykres — ryzyko wg sprzedawcy (stacked bar)
# ---------------------------------------------------------------------------

if has_sprzedawca:
    st.subheader("Portfel zamówień wg sprzedawcy")
    sp = sprzedawca_stats.copy()
    sp['low_risk'] = sp['total'] - sp['high_risk']
    sp = sp.sort_values('risk_pct', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.barh(sp.index, sp['low_risk'], color='#91bfdb', edgecolor='black', label='Niskie ryzyko')
    ax.barh(sp.index, sp['high_risk'], left=sp['low_risk'],
            color='#d73027', edgecolor='black', label='Wysokie ryzyko')
    for i, (idx, row) in enumerate(sp.iterrows()):
        ax.text(row['total'] + 1, i, f"{row['risk_pct']:.0f}%", va='center', fontsize=9, color='#d73027')
    ax.set_xlabel("Liczba zamówień")
    ax.legend(loc='lower right')
    st.pyplot(fig)
    plt.close()
    st.caption("Wartość % po prawej = udział wysokiego ryzyka w portfelu sprzedawcy.")

st.divider()

# ---------------------------------------------------------------------------
# Tabela — top 20 wysokiego ryzyka
# ---------------------------------------------------------------------------

st.subheader("Top 20 zamówień wysokiego ryzyka — wymagają interwencji")
st.caption(
    "Posortowane od najwyższej przewidywanej wartości płatności. "
    "Im wyższy score przy wysokim ryzyku, tym większy potencjalny wpływ na cashflow."
)

col_map = {
    'Nazwa klienta': 'Klient',
    'Sprzedawca': 'Sprzedawca',
    'Branża': 'Branża',
    'Data zamowienia': 'Data',
    'Zapłacono': 'Wartość (PLN)',
    'prediction_score': 'Predykcja modelu (PLN)',
}
show_cols = [k for k in col_map if k in df_out.columns]
df_high = (
    df_out[df_out['prediction_label'] == 0]
    .sort_values('prediction_score', ascending=False)
    .head(20)[show_cols]
    .rename(columns=col_map)
    .reset_index(drop=True)
)
df_high.index += 1

styled_table = df_high.style.map(
    lambda _: 'color: #d73027; font-weight: bold',
    subset=['Predykcja modelu (PLN)']
).format({'Wartość (PLN)': '{:,.0f}', 'Predykcja modelu (PLN)': '{:,.0f}'})

st.dataframe(styled_table, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------

st.subheader("Interpretacja modelu – SHAP")
st.markdown(
    "**Beeswarm** — każda kropka = jedno zamówienie. "
    "Czerwony = wysoka wartość cechy, niebieski = niska. "
    "Pozycja w prawo = cecha *podnosi* predykcję, w lewo = *obniża*.\n\n"
    "**Waterfall** — wyjaśnienie jednej konkretnej predykcji: "
    "od wartości bazowej modelu do finalnego wyniku, krok po kroku przez każdą cechę."
)

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
        st.markdown("**Wyjaśnienie pojedynczej predykcji (Waterfall)**")
        sample_idx = st.slider("Wybierz rekord do analizy", 0, len(X_all) - 1, 0)
        rec = df_out.iloc[sample_idx]
        risk_label = "Wysokie ryzyko" if rec['prediction_label'] == 0 else "Niskie ryzyko"
        klient = rec.get('Nazwa klienta', '—')
        st.caption(
            f"Rekord #{sample_idx + 1} · Klient: **{klient}** · "
            f"Predykcja: **{rec['prediction_score']:,.0f} PLN** · {risk_label}"
        )
        plt.figure()
        shap.plots.waterfall(shap_values[sample_idx], max_display=10, show=False)
        st.pyplot(plt.gcf())
        plt.close()

except Exception as e:
    st.warning(f"SHAP niedostępny: {e}")

st.divider()

# ---------------------------------------------------------------------------
# Expander — pełna tabela
# ---------------------------------------------------------------------------

with st.expander("Pełna tabela danych z predykcjami"):
    st.dataframe(df_out, use_container_width=True)

# ---------------------------------------------------------------------------
# Expander — metodologia
# ---------------------------------------------------------------------------

with st.expander("Jak działa model?"):
    st.markdown("""
**Pipeline ML (scikit-learn):**
1. **Feature engineering** — z daty zamówienia wyciągamy miesiąc, dzień tygodnia i kwartał
2. **Preprocessing** — numeryczne: imputacja medianą + standaryzacja;
   kategoryczne: imputacja stałą + One-Hot Encoding
3. **Model** — Gradient Boosting Regressor (100 drzew, random_state=42)
4. **Klasyfikacja ryzyka** — predykcja wartości PLN porównywana z progiem ustawionym przez użytkownika

**Metryki ewaluacji (zbiór testowy 20%):**
- R² = 0.91 — model wyjaśnia 91% wariancji wartości płatności
- MAE = 1 302 PLN — średni błąd bezwzględny na pojedynczym zamówieniu
- RMSE = 1 963 PLN

**SHAP** (SHapley Additive exPlanations) — matematycznie rygorystyczna metoda wyjaśniania predykcji.
Każda cecha otrzymuje wkład (pozytywny lub negatywny) w finalną predykcję dla konkretnego rekordu,
co pozwala zrozumieć *dlaczego* model podjął daną decyzję.
    """)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown(
    "<div style='text-align:center;color:gray;font-size:0.82em;padding-top:2rem;'>"
    "R²=0.91 · MAE=1 302 PLN · Gradient Boosting · scikit-learn Pipeline | "
    "<a href='https://github.com/webdevanki/-predicting-sales-app' style='color:gray;'>"
    "GitHub</a>"
    "</div>",
    unsafe_allow_html=True,
)
