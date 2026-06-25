import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
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


@st.cache_resource
def train_model(df_hash, _df):
    df_fe = engineer_features(_df)
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

st.markdown("""
<style>
.stApp { background-color: #0f1117; }

[data-testid="metric-container"] {
    background: #1e2130;
    border: 1px solid #2d3250;
    border-radius: 12px;
    padding: 16px;
}

h2, h3 { color: #e8eaf0 !important; }

[data-testid="stSidebar"] { background: #13161f; }

[data-testid="stDataFrame"] { border-radius: 8px; }

[data-testid="stExpander"] {
    background: #1e2130;
    border: 1px solid #2d3250;
    border-radius: 8px;
}

hr { border-color: #2d3250; }

[data-testid="stAlert"] { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("""
<div style='text-align:center; padding:16px 0 8px;'>
    <div style='font-size:1.4rem; font-weight:700; color:#4F8EF7;'>
        💳 B2B Risk
    </div>
    <div style='font-size:0.7rem; color:#888; letter-spacing:0.1em;'>
        PAYMENT PREDICTION
    </div>
</div>
""", unsafe_allow_html=True)

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
        "- Plotly · Streamlit\n"
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
    df_hash = int(pd.util.hash_pandas_object(df_raw, index=True).sum())
    pipeline, X_all, num_cols, cat_cols = train_model(df_hash, df_raw)

scores = pipeline.predict(X_all)
df_out = df_raw.copy()
df_out['prediction_score'] = scores.round(2)
df_out['prediction_label'] = (scores >= threshold).astype(int)  # 1=niskie, 0=wysokie ryzyko

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
# KPI — kolorowe karty HTML
# ---------------------------------------------------------------------------

total = len(df_out)
n_high = int((df_out['prediction_label'] == 0).sum())
n_low = int((df_out['prediction_label'] == 1).sum())
value_at_risk = float(df_out.loc[df_out['prediction_label'] == 0, 'prediction_score'].sum())

k1, k2, k3, k4 = st.columns(4)

k1.markdown(f"""
<div style='background:#1e2130;border:1px solid #4F8EF7;
            border-radius:12px;padding:20px 16px;text-align:center;'>
    <div style='color:#4F8EF7;font-size:0.8rem;font-weight:600;
                letter-spacing:0.05em;margin-bottom:8px;'>
        ŁĄCZNA LICZBA ZAMÓWIEŃ
    </div>
    <div style='color:#4F8EF7;font-size:2.2rem;font-weight:700;line-height:1;'>
        {total}
    </div>
    <div style='color:#888;font-size:0.78rem;margin-top:6px;'>
        wszystkie rekordy
    </div>
</div>
""", unsafe_allow_html=True)

k2.markdown(f"""
<div style='background:#1e2130;border:1px solid #4CAF50;
            border-radius:12px;padding:20px 16px;text-align:center;'>
    <div style='color:#4CAF50;font-size:0.8rem;font-weight:600;
                letter-spacing:0.05em;margin-bottom:8px;'>
        🟢 NISKIE RYZYKO
    </div>
    <div style='color:#4CAF50;font-size:2.2rem;font-weight:700;line-height:1;'>
        {n_low / total * 100:.1f}%
    </div>
    <div style='color:#888;font-size:0.78rem;margin-top:6px;'>
        {n_low} zamówień ≥ {threshold:,} PLN
    </div>
</div>
""", unsafe_allow_html=True)

k3.markdown(f"""
<div style='background:#1e2130;border:1px solid #E05A5A;
            border-radius:12px;padding:20px 16px;text-align:center;'>
    <div style='color:#E05A5A;font-size:0.8rem;font-weight:600;
                letter-spacing:0.05em;margin-bottom:8px;'>
        🔴 WYSOKIE RYZYKO
    </div>
    <div style='color:#E05A5A;font-size:2.2rem;font-weight:700;line-height:1;'>
        {n_high / total * 100:.1f}%
    </div>
    <div style='color:#888;font-size:0.78rem;margin-top:6px;'>
        {n_high} zamówień &lt; {threshold:,} PLN
    </div>
</div>
""", unsafe_allow_html=True)

k4.markdown(f"""
<div style='background:#1e2130;border:1px solid #F59E0B;
            border-radius:12px;padding:20px 16px;text-align:center;'>
    <div style='color:#F59E0B;font-size:0.8rem;font-weight:600;
                letter-spacing:0.05em;margin-bottom:8px;'>
        ⚠️ WARTOŚĆ ZAGROŻONA
    </div>
    <div style='color:#F59E0B;font-size:2.2rem;font-weight:700;line-height:1;'>
        {value_at_risk:,.0f}
    </div>
    <div style='color:#888;font-size:0.78rem;margin-top:6px;'>
        PLN łącznie
    </div>
</div>
""", unsafe_allow_html=True)

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
# Wykresy — histogram + branże (Plotly)
# ---------------------------------------------------------------------------

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Rozkład przewidywanych wartości płatności")
    score_min = float(df_out['prediction_score'].min())
    score_max = float(df_out['prediction_score'].max())

    fig = px.histogram(
        df_out, x='prediction_score',
        nbins=40,
        template='plotly_dark',
        color_discrete_sequence=['#4F8EF7'],
        labels={'prediction_score': 'Przewidziana wartość (PLN)', 'count': 'Liczba zamówień'}
    )
    fig.add_vrect(
        x0=score_min, x1=threshold,
        fillcolor='#E05A5A', opacity=0.08,
        layer='below', line_width=0
    )
    fig.add_vline(
        x=threshold, line_dash='dash',
        line_color='#E05A5A', line_width=2,
        annotation_text=f'Próg: {threshold:,} PLN',
        annotation_font_color='#E05A5A',
        annotation_position='top right'
    )
    fig.update_layout(
        plot_bgcolor='#1e2130',
        paper_bgcolor='#1e2130',
        font_color='#e8eaf0',
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Czerwone tło = strefa wysokiego ryzyka (poniżej progu).")

with col_right:
    st.subheader("Udział wysokiego ryzyka wg branży")
    if has_branza:
        risk_pct_sorted = branza_stats['risk_pct'].sort_values()
        bar_colors = [
            '#E05A5A' if v > 50 else '#F59E0B' if v > 30 else '#4F8EF7'
            for v in risk_pct_sorted.values
        ]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=risk_pct_sorted.index.tolist(),
            x=risk_pct_sorted.values.tolist(),
            orientation='h',
            marker_color=bar_colors,
            marker_line_color='rgba(0,0,0,0.3)',
            marker_line_width=1,
            text=[f'{v:.0f}%' for v in risk_pct_sorted.values],
            textposition='outside',
            textfont_color='#e8eaf0',
        ))
        fig.add_vline(x=50, line_dash='dot', line_color='#888', line_width=1)
        fig.update_layout(
            template='plotly_dark',
            plot_bgcolor='#1e2130',
            paper_bgcolor='#1e2130',
            font_color='#e8eaf0',
            margin=dict(l=20, r=60, t=20, b=20),
            xaxis_title='% zamówień wysokiego ryzyka',
            xaxis_range=[0, 115],
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Czerwony > 50% · Pomarańczowy > 30% · Niebieski < 30%")

st.divider()

# ---------------------------------------------------------------------------
# Wykres — portfel sprzedawców (stacked bar, Plotly)
# ---------------------------------------------------------------------------

if has_sprzedawca:
    st.subheader("Portfel zamówień wg sprzedawcy")
    sp = sprzedawca_stats.copy()
    sp['low_risk'] = sp['total'] - sp['high_risk']
    sp = sp.sort_values('risk_pct', ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=sp.index.tolist(),
        x=sp['low_risk'].tolist(),
        name='Niskie ryzyko',
        orientation='h',
        marker_color='#4F8EF7',
        marker_line_color='rgba(0,0,0,0.3)',
        marker_line_width=1,
    ))
    fig.add_trace(go.Bar(
        y=sp.index.tolist(),
        x=sp['high_risk'].tolist(),
        name='Wysokie ryzyko',
        orientation='h',
        marker_color='#E05A5A',
        marker_line_color='rgba(0,0,0,0.3)',
        marker_line_width=1,
        text=[f"{v:.0f}%" for v in sp['risk_pct'].values],
        textposition='outside',
        textfont_color='#E05A5A',
    ))
    fig.update_layout(
        barmode='stack',
        template='plotly_dark',
        plot_bgcolor='#1e2130',
        paper_bgcolor='#1e2130',
        font_color='#e8eaf0',
        margin=dict(l=20, r=60, t=20, b=20),
        xaxis_title='Liczba zamówień',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Wartość % po prawej = udział wysokiego ryzyka w portfelu sprzedawcy.")

st.divider()

# ---------------------------------------------------------------------------
# Tabela — top 20 wysokiego ryzyka
# ---------------------------------------------------------------------------

st.subheader("Top 20 zamówień wysokiego ryzyka — wymagają interwencji")
st.caption(
    "Posortowane od najwyższej przewidywanej wartości. "
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
df_high.insert(0, 'Ryzyko', '🔴 Wysokie')

def color_rows(row):
    return ['background-color: #2a1a1a; color: #e8eaf0'] * len(row)

fmt = {}
if 'Wartość (PLN)' in df_high.columns:
    fmt['Wartość (PLN)'] = '{:,.0f}'
if 'Predykcja modelu (PLN)' in df_high.columns:
    fmt['Predykcja modelu (PLN)'] = '{:,.0f}'

styled_table = df_high.style.apply(color_rows, axis=1).format(fmt)
st.dataframe(styled_table, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------

st.markdown("""
<div style='background:#1e2130;border-left:4px solid #4F8EF7;
            border-radius:0 8px 8px 0;padding:12px 16px;margin-bottom:16px;'>
    <strong style='color:#4F8EF7;'>Model Explainability — SHAP</strong><br>
    <span style='color:#888;font-size:0.85rem;'>
    Gradient Boosting · 100 estimators · R²=0.91 · MAE=1 302 PLN
    </span>
</div>
""", unsafe_allow_html=True)

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
        risk_label = "🔴 Wysokie ryzyko" if rec['prediction_label'] == 0 else "🟢 Niskie ryzyko"
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
    "<div style='text-align:center;color:#555;font-size:0.82em;padding-top:2rem;'>"
    "R²=0.91 · MAE=1 302 PLN · Gradient Boosting · scikit-learn Pipeline | "
    "<a href='https://github.com/webdevanki/-predicting-sales-app' style='color:#4F8EF7;'>"
    "GitHub</a>"
    "</div>",
    unsafe_allow_html=True,
)
