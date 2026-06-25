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
TODAY = pd.Timestamp("2026-06-25")  # używamy stałej żeby demo było przewidywalne

# ───────────────────────────────────────────────────────── Dane demo ──────────

def generate_demo_data():
    np.random.seed(42)
    n = 500

    clients = [
        "Kowalski Sp. z o.o.", "Nowak Trading", "TechVision S.A.", "BudMat Polska",
        "AgriPol Sp. z o.o.", "LogiTrans", "MedSupply", "RetailPro S.A.",
        "GreenEnergy Sp. z o.o.", "AutoParts Polska", "FoodDist S.A.", "PrintMaster",
        "ChemTech Sp. z o.o.", "SteelWork S.A.", "EduTech Sp. z o.o.",
        "ProBuild S.A.", "DataSoft Sp. z o.o.", "FastLog Polska", "MediCare S.A.",
        "AgroPlus Sp. z o.o.", "ElektroTech S.A.", "ColdChain Polska",
        "NetServ Sp. z o.o.", "HeavyDuty S.A.", "FreshFood Polska",
        "SmartRetail Sp. z o.o.", "GreenPower S.A.", "CargoXpress",
        "PharmaDist Sp. z o.o.", "TurboChem S.A.", "DigitalHub Sp. z o.o.",
        "IronWorks S.A.", "AquaTech Polska", "SkyBuild Sp. z o.o.",
        "OmniRetail S.A.", "BioFarm Polska", "SafeGuard Sp. z o.o.",
        "MaxiLog S.A.", "NovaMed Sp. z o.o.", "PrimeTech S.A.",
    ]
    salespeople = [
        "Anna Wiśniewska", "Piotr Kowalczyk", "Marta Jabłońska",
        "Tomasz Nowak", "Katarzyna Wróbel", "Michał Zając", "Joanna Kowalska",
    ]
    industries = [
        "IT / Technologia", "Budownictwo", "Rolnictwo", "Logistyka", "Medycyna",
        "Retail", "Energia", "Motoryzacja", "Spożywczy", "Edukacja",
    ]
    # Share of clients with chronic underpayment per industry
    p_bad_industry = {
        "IT / Technologia": 0.08, "Medycyna": 0.06, "Energia": 0.11,
        "Budownictwo": 0.32, "Rolnictwo": 0.18, "Logistyka": 0.14,
        "Retail": 0.26, "Motoryzacja": 0.13, "Spożywczy": 0.18, "Edukacja": 0.09,
    }

    # Date distribution: 200 historical (model training) + 300 current/future
    dates_hist     = pd.date_range("2023-01-01", "2025-12-31", periods=200)
    dates_recent   = pd.date_range("2026-01-01", "2026-05-25", periods=50)
    dates_upcoming = pd.date_range("2026-05-26", "2026-06-25", periods=150)
    dates_future   = pd.date_range("2026-06-26", "2026-08-31", periods=100)
    all_dates  = np.concatenate([dates_hist.values, dates_recent.values,
                                 dates_upcoming.values, dates_future.values])
    dates_arr  = pd.DatetimeIndex(all_dates[np.random.permutation(n)])

    # Each client belongs to exactly one industry (deterministic to avoid same client
    # appearing as both "PrimeTech (Retail)" and "PrimeTech (IT)" on different orders)
    client_industry = {c: industries[i % len(industries)] for i, c in enumerate(clients)}

    # Payment profile is a CLIENT trait, not per-order:
    # good payer (~98% of invoice) vs problematic (~65%) — consistent across all their orders
    n_clients  = len(clients)
    _rate_bad  = np.random.normal(0.65, 0.10, n_clients).clip(0.28, 0.84)
    _rate_good = np.random.normal(0.98, 0.015, n_clients).clip(0.91, 1.00)
    client_is_bad = {
        c: np.random.binomial(1, p_bad_industry[client_industry[c]]) == 1
        for c in clients
    }
    client_rate = {
        c: float(_rate_bad[i]) if client_is_bad[c] else float(_rate_good[i])
        for i, c in enumerate(clients)
    }

    num_products  = np.random.randint(1, 80, n)
    unit_price    = np.random.uniform(20, 800, n)
    client_arr    = np.random.choice(clients, n)
    industry_arr  = np.array([client_industry[c] for c in client_arr])
    invoice_value = (num_products * unit_price).round(2)

    # Amount paid: client base rate + small per-order noise (±2%)
    rate       = np.array([client_rate[c] for c in client_arr])
    rate       = (rate + np.random.normal(0, 0.02, n)).clip(0.20, 1.00)
    amount_paid = (invoice_value * rate).round(2).clip(50)

    return pd.DataFrame({
        "ID":                  range(1001, 1001 + n),
        "Data zamowienia":     dates_arr.strftime("%Y-%m-%d"),
        "Nazwa klienta":       client_arr,
        "Sprzedawca":          np.random.choice(salespeople, n),
        "Branża":              industry_arr,
        "Liczba produktow":    num_products,
        "Wartosc jednostkowa": unit_price.round(2),
        "Wartość faktury":     invoice_value,
        "Kwota zapłacona":     amount_paid,
        "Komentarz":           np.random.choice(["", "Pilne", "Klient VIP", "Reklamacja", ""], n),
    })


def engineer_features(df_raw):
    df = df_raw.copy()
    df.drop(columns=[c for c in ["ID", "Komentarz"] if c in df.columns], inplace=True)
    for col in [c for c in df.columns if "data" in c.lower() or "date" in c.lower()]:
        try:
            df[col] = pd.to_datetime(df[col])
            df[f"{col}_miesiac"]       = df[col].dt.month
            df[f"{col}_dzien_tygodnia"] = df[col].dt.dayofweek
            df[f"{col}_kwartal"]       = df[col].dt.quarter
            df.drop(columns=[col], inplace=True)
        except Exception:
            pass
    return df


@st.cache_resource
def train_model(df_hash, _df):
    df_fe = engineer_features(_df)
    X     = df_fe.drop(columns=["Kwota zapłacona"])
    y     = df_fe["Kwota zapłacona"]
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    pre = ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  StandardScaler()),
        ]), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value="missing")),
            ("enc", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat_cols),
    ])
    pipe = Pipeline([("pre", pre), ("model", GradientBoostingRegressor(
        n_estimators=100, random_state=RANDOM_STATE))])
    pipe.fit(X, y)
    return pipe, X, num_cols, cat_cols


# ────────────────────────────────────────────────────── Helpers / UI ──────────

_chart_base = dict(
    plot_bgcolor="#FFFFFF",
    paper_bgcolor="#F4F6F9",
    font=dict(family="Segoe UI", color="#1B2A4A"),
)
_ax = dict(gridcolor="#F0F2F5", linecolor="#E0E4EA")


def priority_tier(score):
    if score >= 70:
        return "🔴", "Krytyczne", "#C50F1F", "#FFF0F0"
    if score >= 45:
        return "🟠", "Pilne",     "#E8792A", "#FFF5EE"
    return       "🟡", "Ważne",    "#D4A017", "#FFFDE8"


def get_contact_reason(row, client_counts, industry_stats, has_industry):
    reasons = []
    days    = row.get("dni_do_terminu", 999)

    if days < -14:
        reasons.append(f"przeterminowane {abs(int(days))} dni")
    elif days < 0:
        d = abs(int(days))
        reasons.append(f"termin minął {d} {'dzień' if d == 1 else 'dni'} temu")
    elif days <= 3:
        reasons.append(f"termin płatności za {int(days)} {'dzień' if int(days)==1 else 'dni'}")
    elif days <= 7:
        reasons.append(f"termin za {int(days)} dni")

    count = int(client_counts.get(row.get("Nazwa klienta", ""), 0))
    counts_vals = list(client_counts.values())
    repeat_threshold = max(3, int(np.median(counts_vals) * 1.5)) if counts_vals else 3
    if count >= repeat_threshold:
        reasons.append(f"powtarzający się klient ryzyka ({count}×)")
    elif count > 2:
        reasons.append(f"{count} zamówień poniżej progu")

    if has_industry and industry_stats is not None:
        b    = row.get("Branża", "")
        rpct = industry_stats["risk_pct"].get(b, 0)
        if rpct > 65:
            reasons.append(f"branża wysokiego ryzyka ({rpct:.0f}%)")

    return " · ".join(reasons[:2]) if reasons else "wysoka wartość zagrożonej płatności"


def action_card(rank, icon, label, color, bg,
                client, industry, salesperson, value, days, reason, pct=0):
    if days < 0:
        days_txt   = f"Przeterminowane ({abs(int(days))} dni)"
        days_color = "#C50F1F"
    elif days == 0:
        days_txt   = "Termin: dziś"
        days_color = "#C50F1F"
    elif days <= 7:
        days_txt   = f"Termin: za {int(days)} dni"
        days_color = "#E8792A"
    elif days <= 14:
        days_txt   = f"Termin: za {int(days)} dni"
        days_color = "#D4A017"
    else:
        days_txt   = f"Termin: za {int(days)} dni"
        days_color = "#6B7A99"

    pct_color = "#C50F1F" if pct < 70 else "#E8792A" if pct < 85 else "#D4A017"

    return f"""
<div style='display:flex; align-items:stretch; background:#FFFFFF;
            border:1px solid #E8ECF0; border-radius:6px; margin-bottom:8px;
            box-shadow:0 1px 4px rgba(0,0,0,0.05); overflow:hidden;'>
  <div style='background:{bg}; border-right:3px solid {color}; min-width:72px;
              padding:14px 8px; display:flex; flex-direction:column;
              align-items:center; justify-content:center;'>
    <div style='font-size:1.15rem; line-height:1;'>{icon}</div>
    <div style='color:{color}; font-size:0.58rem; font-weight:800;
                text-transform:uppercase; letter-spacing:0.08em;
                text-align:center; margin-top:4px; line-height:1.3;'>{label}</div>
    <div style='color:#AABACE; font-size:0.72rem; margin-top:5px;'>#{rank}</div>
  </div>
  <div style='flex:1; padding:12px 16px; display:flex;
              justify-content:space-between; align-items:center; gap:12px;'>
    <div style='flex:1; min-width:0;'>
      <div style='font-weight:700; color:#1B2A4A; font-size:0.95rem; margin-bottom:3px;
                  white-space:nowrap; overflow:hidden; text-overflow:ellipsis;'>{client}</div>
      <div style='color:#6B7A99; font-size:0.78rem; margin-bottom:5px;'>
        {industry} &nbsp;·&nbsp; Opiekun: <strong style='color:#1B2A4A;'>{salesperson}</strong>
      </div>
      <div style='color:#8896A8; font-size:0.76rem;'>⚠&thinsp; {reason}</div>
    </div>
    <div style='text-align:right; flex-shrink:0; min-width:130px;'>
      <div style='color:#8896A8; font-size:0.6rem; text-transform:uppercase;
                  letter-spacing:0.05em; margin-bottom:2px;'>Szac. niedobór</div>
      <div style='color:#C50F1F; font-size:1.15rem; font-weight:700; line-height:1; margin-bottom:3px;'>
        {value:,.0f}&thinsp;PLN
      </div>
      <div style='color:{pct_color}; font-size:0.72rem; font-weight:600; margin-bottom:3px;'>
        zapłaci ok. {pct:.0f}% faktury
      </div>
      <div style='color:{days_color}; font-size:0.74rem; font-weight:600;'>{days_txt}</div>
    </div>
  </div>
</div>"""


def kpi_card(color, label, value, sub):
    return f"""
<div style='background:#FFFFFF; border:1px solid #E8ECF0; border-top:3px solid {color};
            border-radius:6px; padding:18px 20px;
            box-shadow:0 1px 4px rgba(0,0,0,0.06);'>
  <div style='color:#8896A8; font-size:0.7rem; font-weight:700;
              text-transform:uppercase; letter-spacing:0.08em; margin-bottom:10px;'>
    {label}
  </div>
  <div style='color:{color}; font-size:2rem; font-weight:700; line-height:1; margin-bottom:5px;'>
    {value}
  </div>
  <div style='color:#8896A8; font-size:0.78rem;'>{sub}</div>
</div>"""


def section_header(text):
    st.markdown(
        f"<div style='font-size:1.05rem; font-weight:700; color:#1B2A4A; "
        f"border-left:3px solid #0065BD; padding-left:10px; margin:20px 0 12px;'>"
        f"{text}</div>",
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────── Konfiguracja strony ──────

st.set_page_config(page_title="B2B Payment Risk", layout="wide", page_icon="💳")

st.markdown("""
<style>
.stApp { background:#F4F6F9; font-family:'Segoe UI',system-ui,sans-serif; }
.main .block-container { padding:1.5rem 3rem; max-width:1200px; }
[data-testid="stSidebar"] { background:#FFFFFF; border-right:1px solid #E0E4EA; }
h1 { color:#1B2A4A !important; font-size:1.7rem !important; font-weight:700 !important;
     border-bottom:3px solid #0065BD; padding-bottom:8px; }
h2,h3 { color:#1B2A4A !important; font-weight:600 !important; }
hr    { border-color:#E0E4EA; margin:1.2rem 0; }
.stTabs [data-baseweb="tab-list"] {
  background:#FFFFFF; border-radius:6px; padding:4px;
  border:1px solid #E0E4EA; gap:2px;
}
.stTabs [data-baseweb="tab"] {
  border-radius:4px; padding:8px 18px;
  color:#6B7A99; font-weight:600; font-size:0.88rem;
}
.stTabs [aria-selected="true"] {
  background:#0065BD !important; color:#FFFFFF !important;
}
[data-testid="stExpander"] {
  background:#FFFFFF; border:1px solid #E0E4EA; border-radius:8px;
}
</style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────── Sidebar ──────

with st.sidebar:
    st.markdown("""
<div style='padding:14px 0 18px; border-bottom:2px solid #0065BD; margin-bottom:18px;'>
  <div style='font-size:1.1rem; font-weight:700; color:#0065BD;'>💳 B2B Payment Risk</div>
  <div style='font-size:0.66rem; color:#8896A8; text-transform:uppercase;
              letter-spacing:0.12em; margin-top:3px;'>Prediction Dashboard</div>
</div>""", unsafe_allow_html=True)

    st.markdown("**Źródło danych**")
    mode = st.radio("", ["Dane demo", "Wgraj własny CSV"], label_visibility="collapsed")
    st.divider()

    st.markdown("**Próg pokrycia faktury**")
    st.caption(
        "Jeśli model przewiduje, że klient zapłaci **mniej niż X% wartości faktury** — "
        "faktura trafia na listę ryzyka."
    )
    threshold_pct = st.slider(
        "", min_value=50, max_value=95, value=85, step=1,
        format="%d%%", label_visibility="collapsed",
    )
    st.caption("Np. 85% → klient z fakturą 10 000 PLN musi zapłacić min. 8 500 PLN")

# ─────────────────────────────────────────────────────── Wczytaj dane ──────

if mode == "Dane demo":
    df_raw = generate_demo_data()
    st.info(
        "Tryb demo — 500 syntetycznych zamówień B2B · 40 klientów · 7 sprzedawców · "
        "daty: historyczne, bieżące i przyszłe."
    )
else:
    uploaded = st.file_uploader("Wgraj plik CSV z zamówieniami", type=["csv"])
    if not uploaded:
        st.info("Wgraj plik CSV żeby zobaczyć analizę.")
        st.stop()
    df_raw = pd.read_csv(uploaded)

# ─────────────────────────────────────────────────────── Trenuj model ──────

with st.spinner("Trenuję model predykcyjny…"):
    df_hash = int(pd.util.hash_pandas_object(df_raw, index=True).sum())
    pipeline, X_all, num_cols, cat_cols = train_model(df_hash, df_raw)

scores    = pipeline.predict(X_all)
df_out    = df_raw.copy()
df_out["prediction_score"] = scores.round(2)

# % pokrycia faktury: ile z wartości faktury model spodziewa się że wpłynie
if "Wartość faktury" in df_out.columns:
    df_out["pct_pokrycia"] = (scores / df_out["Wartość faktury"].clip(lower=1) * 100).clip(upper=100).round(1)
else:
    df_out["pct_pokrycia"] = 100.0

# Ryzyko: klient zapłaci mniej niż threshold_pct% wartości swojej faktury
df_out["prediction_label"] = (df_out["pct_pokrycia"] >= threshold_pct).astype(int)

# Kwota niedoboru = to czego faktycznie może zabraknąć
if "Wartość faktury" in df_out.columns:
    df_out["kwota_niedoboru"] = (df_out["Wartość faktury"] - df_out["prediction_score"]).clip(lower=0).round(2)
else:
    df_out["kwota_niedoboru"] = 0.0

has_industry    = "Branża" in df_out.columns
has_salesperson = "Sprzedawca" in df_out.columns
has_order_date     = "Data zamowienia" in df_out.columns

# Statystyki branżowe
if has_industry:
    industry_stats = df_out.groupby("Branża").agg(
        total=("prediction_label", "count"),
        high_risk=("prediction_label", lambda x: (x == 0).sum()),
    )
    industry_stats["risk_pct"] = industry_stats["high_risk"] / industry_stats["total"] * 100
    worst_industry = industry_stats["risk_pct"].idxmax()
    best_industry  = industry_stats["risk_pct"].idxmin()
else:
    industry_stats = None

# Statystyki sprzedawców
if has_salesperson:
    salesperson_stats = df_out.groupby("Sprzedawca").agg(
        total=("prediction_label", "count"),
        high_risk=("prediction_label", lambda x: (x == 0).sum()),
    )
    salesperson_stats["risk_pct"]  = salesperson_stats["high_risk"] / salesperson_stats["total"] * 100
    salesperson_stats["low_risk"]  = salesperson_stats["total"] - salesperson_stats["high_risk"]
    worst_salesperson = salesperson_stats["risk_pct"].idxmax()

# Globalne KPI
total        = len(df_out)
n_high        = int((df_out["prediction_label"] == 0).sum())
n_low         = int((df_out["prediction_label"] == 1).sum())
value_at_risk = float(df_out.loc[df_out["prediction_label"] == 0, "kwota_niedoboru"].sum())
median_pct    = float(df_out["pct_pokrycia"].median())

# ─────────────────────────────── Sidebar: feedback progu ──────────

with st.sidebar:
    st.markdown(
        f"""
<div style='background:#F4F6F9; border:1px solid #E0E4EA; border-radius:6px;
            padding:10px 12px; margin-top:4px; font-size:0.8rem; line-height:1.9;'>
  <div><span style='color:#C50F1F; font-weight:700;'>● {n_high}</span>
  &nbsp;faktur z ryzykiem niedoboru
  <span style='color:#8896A8;'>({n_high/total*100:.0f}%)</span></div>
  <div><span style='color:#E8792A; font-weight:700;'>
    {value_at_risk:,.0f} PLN</span>&nbsp;może nie wpłynąć</div>
  <div style='color:#8896A8; margin-top:2px; font-size:0.74rem;'>
    Mediana pokrycia: <strong>{median_pct:.0f}%</strong> wartości faktury</div>
</div>""",
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown(
        "<div style='font-size:0.76rem; color:#8896A8; line-height:1.9;'>"
        "scikit-learn · GradientBoosting<br>SHAP · Plotly · Streamlit</div>",
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────────────── Tytuł ──────

st.title("Ryzyko niezapłaconych faktur B2B")

_step_style = ("background:#FFFFFF; border:1px solid #E8ECF0; border-radius:6px; "
               "padding:16px 18px; box-shadow:0 1px 3px rgba(0,0,0,0.05); "
               "min-height:130px;")

s1, s2, s3 = st.columns(3)
s1.markdown(f"""
<div style='{_step_style}'>
  <div style='font-size:1.3rem; margin-bottom:8px;'>📄</div>
  <div style='font-weight:700; color:#1B2A4A; font-size:0.88rem; margin-bottom:5px;'>
    Problem</div>
  <div style='color:#6B7A99; font-size:0.8rem; line-height:1.5;'>
    Wystawiasz fakturę z terminem 30 dni.
    Część klientów zapłaci mniej lub z opóźnieniem —
    <strong>nie wiesz którzy, dopóki termin nie minie</strong>.
  </div>
</div>""", unsafe_allow_html=True)

s2.markdown(f"""
<div style='{_step_style}'>
  <div style='font-size:1.3rem; margin-bottom:8px;'>🤖</div>
  <div style='font-weight:700; color:#1B2A4A; font-size:0.88rem; margin-bottom:5px;'>
    Rozwiązanie</div>
  <div style='color:#6B7A99; font-size:0.8rem; line-height:1.5;'>
    System analizuje historię płatności klientów
    i wskazuje faktury, które <strong>mogą nie wpłynąć
    w pełnej kwocie</strong> — jeszcze przed terminem.
  </div>
</div>""", unsafe_allow_html=True)

s3.markdown(f"""
<div style='{_step_style}'>
  <div style='font-size:1.3rem; margin-bottom:8px;'>📞</div>
  <div style='font-weight:700; color:#1B2A4A; font-size:0.88rem; margin-bottom:5px;'>
    Działanie</div>
  <div style='color:#6B7A99; font-size:0.8rem; line-height:1.5;'>
    Handlowiec dostaje listę klientów do zadzwonienia.
    <strong>Jeden telefon przed terminem</strong> często
    wystarczy — zamiast windykacji tygodnie później.
  </div>
</div>""", unsafe_allow_html=True)

st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────── KPI ──────

k1, k2, k3, k4 = st.columns(4)
k1.markdown(kpi_card("#0065BD", "Faktur w portfelu",       str(total),                    "wystawionych klientom B2B"), unsafe_allow_html=True)
k2.markdown(kpi_card("#107C10", "Prognoza: pełna płatność",   f"{n_low/total*100:.1f}%",  f"klient zapłaci ≥ {threshold_pct}% faktury"), unsafe_allow_html=True)
k3.markdown(kpi_card("#C50F1F", "Ryzyko niedopłaty",         f"{n_high/total*100:.1f}%", f"klient zapłaci < {threshold_pct}% faktury"), unsafe_allow_html=True)
k4.markdown(kpi_card("#E8792A", "Szacowany niedobór",     f"{value_at_risk/1000:.0f}k",  "PLN może nie wpłynąć"), unsafe_allow_html=True)

st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)

# ────────────────────────────── Przygotuj dane wysokiego ryzyka ──────

df_hr = df_out[df_out["prediction_label"] == 0].copy()

if has_order_date:
    df_hr["Data zamowienia"]   = pd.to_datetime(df_hr["Data zamowienia"], errors="coerce")
    df_hr["termin_platnosci"]  = df_hr["Data zamowienia"] + pd.Timedelta(days=30)
    df_hr["dni_do_terminu"]    = (df_hr["termin_platnosci"] - TODAY).dt.days
else:
    df_hr["dni_do_terminu"] = 999

# Liczba zamówień ryzyka per klient
client_counts = df_hr.groupby("Nazwa klienta").size().to_dict() if "Nazwa klienta" in df_hr.columns else {}

# Powody kontaktu
df_hr["reason"] = df_hr.apply(
    lambda r: get_contact_reason(r, client_counts, industry_stats, has_industry), axis=1
)

# Priority score
def _urgency(days):
    if days < 0:   return 100
    if days <= 7:  return 80
    if days <= 14: return 60
    if days <= 30: return 40
    return 20

vmin, vmax = df_hr["kwota_niedoboru"].min(), df_hr["kwota_niedoboru"].max()
df_hr["urgency_score"] = df_hr["dni_do_terminu"].apply(_urgency)
df_hr["value_score"]   = (df_hr["kwota_niedoboru"] - vmin) / max(vmax - vmin, 1) * 100
df_hr["repeat_score"]  = df_hr["Nazwa klienta"].map(client_counts).fillna(1).clip(upper=10) * 10 if "Nazwa klienta" in df_hr.columns else 10
df_hr["priority_score"] = (
    0.40 * df_hr["urgency_score"] +
    0.40 * df_hr["value_score"]   +
    0.20 * df_hr["repeat_score"]
)
df_hr = df_hr.sort_values("priority_score", ascending=False).reset_index(drop=True)

# Active portfolio: exclude 2023-2025 training history from operational cash flow view
_order_date_hr = pd.to_datetime(df_hr["Data zamowienia"], errors="coerce")
_active = _order_date_hr >= pd.Timestamp("2026-01-01")

# Buckety cash flow — przeterminowane rozbite na 3 przedziały
overdue_fresh  = df_hr[df_hr["dni_do_terminu"].between(-30, -1) & _active]
overdue_medium = df_hr[df_hr["dni_do_terminu"].between(-90, -31) & _active]
overdue_old    = df_hr[(df_hr["dni_do_terminu"] < -90) & _active]
this_week      = df_hr[df_hr["dni_do_terminu"].between(0, 7) & _active]
next_2weeks    = df_hr[df_hr["dni_do_terminu"].between(8, 14) & _active]
this_month     = df_hr[df_hr["dni_do_terminu"].between(15, 30) & _active]

# Tab Działania: tylko faktury do 30 dni po terminie — "jeden telefon" zadziała
# Factury przeterminowane 31–90 dni są w zakładce Cash flow (overdue_medium)
df_hr_active = df_hr[df_hr["dni_do_terminu"] >= -30].reset_index(drop=True)

# ──────────────────────────────────────────────────────── TABY ──────

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Działania",
    "📊 Analiza",
    "📅 Cash flow",
    "🔬 Model",
])

# ═══════════════════════════════════════════════════════ TAB 1 ══════
with tab1:

    df_upcoming  = df_hr_active[df_hr_active["dni_do_terminu"] >= 0].reset_index(drop=True)
    df_overdue30 = df_hr_active[df_hr_active["dni_do_terminu"] < 0].reset_index(drop=True)
    n_upcoming   = len(df_upcoming)
    n_overdue30  = len(df_overdue30)
    top5_val     = df_hr_active.head(5)["kwota_niedoboru"].sum()

    st.markdown(f"""
<div style='background:#FFFFFF; border:1px solid #E8ECF0; border-radius:8px;
            padding:18px 24px; margin-bottom:20px; border-left:4px solid #0065BD;
            box-shadow:0 1px 4px rgba(0,0,0,0.06);'>
  <div style='display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:10px;'>
    <div style='font-size:1rem; font-weight:700; color:#1B2A4A;'>
      Faktury z ryzykiem niedopłaty — działaj zanim minie termin płatności
    </div>
    <div style='color:#8896A8; font-size:0.72rem; text-align:right; line-height:1.6; flex-shrink:0; margin-left:16px;'>
      Kolejność: pilność terminu (40%) · kwota niedoboru (40%) · historia klienta (20%)
    </div>
  </div>
  <div style='display:flex; gap:28px; flex-wrap:wrap;'>
    <div>
      <span style='font-size:1.5rem; font-weight:800; color:#0065BD;'>{n_upcoming}</span>
      <span style='color:#8896A8; font-size:0.8rem; margin-left:5px;'>faktur przed terminem</span>
    </div>
    <div>
      <span style='font-size:1.5rem; font-weight:800; color:#C50F1F;'>{n_overdue30}</span>
      <span style='color:#8896A8; font-size:0.8rem; margin-left:5px;'>świeżo przeterminowanych</span>
    </div>
    <div>
      <span style='font-size:1.5rem; font-weight:800; color:#E8792A;'>{top5_val:,.0f} PLN</span>
      <span style='color:#8896A8; font-size:0.8rem; margin-left:5px;'>szacowany niedobór top 5</span>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

    def _render_tiers(df_section, start_rank):
        tiers = [
            ("🔴", "Krytyczne", "#C50F1F", "#FFF0F0",
             df_section[df_section["priority_score"] >= 70]),
            ("🟠", "Pilne",     "#E8792A", "#FFF5EE",
             df_section[(df_section["priority_score"] >= 45) & (df_section["priority_score"] < 70)]),
            ("🟡", "Ważne",     "#D4A017", "#FFFDE8",
             df_section[df_section["priority_score"] < 45]),
        ]
        rank = start_rank
        for icon, label, color, bg, subset in tiers:
            if subset.empty:
                continue
            st.markdown(
                f"<div style='display:flex; align-items:center; gap:10px; margin:14px 0 8px;'>"
                f"<span style='font-size:1rem;'>{icon}</span>"
                f"<span style='font-weight:700; color:{color}; font-size:0.9rem; "
                f"text-transform:uppercase; letter-spacing:0.06em;'>{label}</span>"
                f"<span style='color:#8896A8; font-size:0.8rem;'>· {len(subset)} faktur"
                f" · {subset['kwota_niedoboru'].sum():,.0f} PLN szacowany niedobór</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            show_n = min(5, len(subset))
            for _, row in subset.head(show_n).iterrows():
                rank += 1
                st.markdown(action_card(
                    rank=rank, icon=icon, label=label, color=color, bg=bg,
                    client=row.get("Nazwa klienta", "—"),
                    industry=row.get("Branża", "—"),
                    salesperson=row.get("Sprzedawca", "—"),
                    value=row["kwota_niedoboru"],
                    days=int(row["dni_do_terminu"]),
                    reason=row["reason"],
                    pct=row["pct_pokrycia"],
                ), unsafe_allow_html=True)
            remaining = len(subset) - show_n
            if remaining > 0:
                with st.expander(f"Pokaż kolejne {remaining} z kategorii '{label}'"):
                    for _, row in subset.iloc[show_n:].iterrows():
                        rank += 1
                        st.markdown(action_card(
                            rank=rank, icon=icon, label=label, color=color, bg=bg,
                            client=row.get("Nazwa klienta", "—"),
                            industry=row.get("Branża", "—"),
                            salesperson=row.get("Sprzedawca", "—"),
                            value=row["kwota_niedoboru"],
                            days=int(row["dni_do_terminu"]),
                            reason=row["reason"],
                            pct=row["pct_pokrycia"],
                        ), unsafe_allow_html=True)
        return rank

    # ── Sekcja 1: Zadzwoń przed terminem (główna wartość systemu) ──────────────
    st.markdown("""
<div style='display:flex; align-items:center; gap:12px; margin:8px 0 6px;
            padding:12px 16px; background:#EBF3FF; border-radius:6px;
            border-left:3px solid #0065BD;'>
  <span style='font-size:1.2rem;'>📞</span>
  <div>
    <div style='font-weight:700; color:#0065BD; font-size:0.92rem;'>
      Zadzwoń przed terminem</div>
    <div style='color:#6B7A99; font-size:0.76rem; margin-top:1px;'>
      Działaj zanim minie termin — jeden telefon zamiast windykacji tygodnie później</div>
  </div>
</div>""", unsafe_allow_html=True)

    if df_upcoming.empty:
        st.info("Brak faktur wysokiego ryzyka z nadchodzącym terminem.")
        global_rank = 0
    else:
        global_rank = _render_tiers(df_upcoming, 0)

    # ── Sekcja 2: Świeżo przeterminowane (≤30 dni) ─────────────────────────────
    if not df_overdue30.empty:
        st.markdown("""
<div style='display:flex; align-items:center; gap:12px; margin:28px 0 6px;
            padding:12px 16px; background:#FFF0F0; border-radius:6px;
            border-left:3px solid #C50F1F;'>
  <span style='font-size:1.2rem;'>⚠️</span>
  <div>
    <div style='font-weight:700; color:#C50F1F; font-size:0.92rem;'>
      Świeżo przeterminowane — do 30 dni po terminie</div>
    <div style='color:#6B7A99; font-size:0.76rem; margin-top:1px;'>
      Termin już minął — ustal datę płatności zanim trafi do windykacji</div>
  </div>
</div>""", unsafe_allow_html=True)
        global_rank = _render_tiers(df_overdue30, global_rank)

# ═══════════════════════════════════════════════════════ TAB 2 ══════
with tab2:

    # Kluczowe wnioski
    section_header("Kluczowe wnioski")
    i1, i2, i3 = st.columns(3)

    def _insight(col, color, arrow, name, detail):
        col.markdown(f"""
<div style='background:#FFFFFF; border:1px solid #E8ECF0; border-left:4px solid {color};
            border-radius:0 6px 6px 0; padding:16px 18px;
            box-shadow:0 1px 3px rgba(0,0,0,0.05);'>
  <div style='color:{color}; font-size:0.65rem; font-weight:800;
              text-transform:uppercase; letter-spacing:0.08em; margin-bottom:8px;'>{arrow}</div>
  <div style='font-weight:700; color:#1B2A4A; font-size:1rem; line-height:1.3; margin-bottom:5px;'>
    {name}</div>
  <div style='color:#6B7A99; font-size:0.8rem; line-height:1.5;'>{detail}</div>
</div>""", unsafe_allow_html=True)

    if has_industry:
        wr = industry_stats.loc[worst_industry]
        br = industry_stats.loc[best_industry]
        _insight(i1, "#C50F1F", "▲ Branża najwyższego ryzyka", worst_industry,
                 f"{wr['risk_pct']:.0f}% zamówień poniżej progu "
                 f"({int(wr['high_risk'])} z {int(wr['total'])})")
        _insight(i2, "#107C10", "▼ Branża najniższego ryzyka", best_industry,
                 f"{br['risk_pct']:.0f}% zamówień poniżej progu "
                 f"({int(br['high_risk'])} z {int(br['total'])})")
    if has_salesperson:
        sr = salesperson_stats.loc[worst_salesperson]
        _insight(i3, "#E8792A", "● Sprzedawca z największym ryzykiem", worst_salesperson,
                 f"{sr['risk_pct']:.0f}% zamówień wysokiego ryzyka "
                 f"({int(sr['high_risk'])} z {int(sr['total'])})")

    st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)

    # Wykresy
    section_header("Rozkład wartości zamówień")
    fig = px.histogram(
        df_out, x="pct_pokrycia", nbins=40,
        template="plotly_white",
        color_discrete_sequence=["#0065BD"],
        labels={"pct_pokrycia": "Prognozowane pokrycie faktury (%)", "count": "Liczba faktur"},
    )
    fig.add_vrect(
        x0=0, x1=threshold_pct,
        fillcolor="#C50F1F", opacity=0.06, layer="below", line_width=0,
    )
    fig.add_vline(
        x=threshold_pct, line_dash="dash", line_color="#C50F1F", line_width=2,
        annotation_text=f"Próg: {threshold_pct}%",
        annotation_font_color="#C50F1F", annotation_position="top right",
    )
    fig.update_layout(
        **_chart_base,
        xaxis=dict(title="Prognozowane pokrycie faktury (%)", ticksuffix="%", **_ax),
        yaxis=dict(title="Liczba faktur", **_ax),
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Czerwone tło = strefa ryzyka (< {threshold_pct}%). Każdy słupek = zakres % pokrycia faktury.")

    col_l, col_r = st.columns(2)

    with col_l:
        section_header("Ryzyko wg branży")
        if has_industry:
            sorted_b = industry_stats["risk_pct"].sort_values()
            bar_col  = [
                "#C50F1F" if v > 50 else "#E8792A" if v > 30 else "#0065BD"
                for v in sorted_b.values
            ]
            fig = go.Figure(go.Bar(
                y=sorted_b.index.tolist(),
                x=sorted_b.values.tolist(),
                orientation="h",
                marker_color=bar_col,
                marker_line_color="rgba(0,0,0,0.06)",
                marker_line_width=1,
                text=[f"{v:.0f}%" for v in sorted_b.values],
                textposition="outside",
                textfont_color="#1B2A4A",
            ))
            fig.add_vline(x=50, line_dash="dot", line_color="#8896A8", line_width=1)
            fig.update_layout(
                **_chart_base,
                xaxis=dict(title="% zamówień wysokiego ryzyka", range=[0, 115], **_ax),
                yaxis=_ax,
                showlegend=False,
                margin=dict(l=20, r=60, t=10, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Przerywana linia = 50%. Czerwony > 50% · Pomarańczowy > 30%")

    with col_r:
        section_header("Portfel sprzedawców")
        if has_salesperson:
            sp_sorted = salesperson_stats.sort_values("risk_pct", ascending=True)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=sp_sorted.index.tolist(), x=sp_sorted["low_risk"].tolist(),
                name="Niskie ryzyko", orientation="h",
                marker_color="#0065BD",
                marker_line_color="rgba(0,0,0,0.06)", marker_line_width=1,
            ))
            fig.add_trace(go.Bar(
                y=sp_sorted.index.tolist(), x=sp_sorted["high_risk"].tolist(),
                name="Wysokie ryzyko", orientation="h",
                marker_color="#C50F1F",
                marker_line_color="rgba(0,0,0,0.06)", marker_line_width=1,
                text=[f"{v:.0f}%" for v in sp_sorted["risk_pct"].values],
                textposition="outside", textfont_color="#C50F1F",
            ))
            fig.update_layout(
                barmode="stack",
                **_chart_base,
                xaxis=dict(title="Liczba zamówień", **_ax),
                yaxis=_ax,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=20, r=70, t=30, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("% po prawej = udział wysokiego ryzyka w portfelu.")

# ═══════════════════════════════════════════════════════ TAB 3 ══════
with tab3:

    section_header("Zagrożone płatności wg terminu")

    def _cf_card(col, icon, lbl, data, clr, muted=False):
        val = data["kwota_niedoboru"].sum()
        cnt = len(data)
        opacity = "opacity:0.55;" if muted else ""
        col.markdown(f"""
<div style='background:#FFFFFF; border:1px solid #E8ECF0; border-top:3px solid {clr};
            border-radius:6px; padding:14px; text-align:center;
            box-shadow:0 1px 4px rgba(0,0,0,0.06); {opacity}'>
  <div style='font-size:1.3rem; margin-bottom:5px;'>{icon}</div>
  <div style='color:#8896A8; font-size:0.62rem; font-weight:700;
              text-transform:uppercase; letter-spacing:0.07em; margin-bottom:7px;'>
    {lbl}</div>
  <div style='color:{clr}; font-size:1.4rem; font-weight:700; line-height:1; margin-bottom:3px;'>
    {val/1000:.0f}k</div>
  <div style='color:#8896A8; font-size:0.72rem;'>PLN niedoboru · {cnt} faktur</div>
</div>""", unsafe_allow_html=True)

    st.markdown(
        "<div style='color:#8896A8; font-size:0.75rem; font-weight:600; "
        "text-transform:uppercase; letter-spacing:0.07em; margin-bottom:6px;'>"
        "Przeterminowane</div>",
        unsafe_allow_html=True,
    )
    o1, o2, o3 = st.columns(3)
    _cf_card(o1, "🚨", "1–30 dni po terminie",  overdue_fresh,  "#C50F1F")
    _cf_card(o2, "⚠️", "31–90 dni po terminie", overdue_medium, "#E8792A")
    _cf_card(o3, "📁", "Powyżej 90 dni",         overdue_old,    "#8896A8", muted=True)

    st.markdown(
        "<div style='color:#8896A8; font-size:0.75rem; font-weight:600; "
        "text-transform:uppercase; letter-spacing:0.07em; margin:14px 0 6px;'>"
        "Nadchodzące terminy</div>",
        unsafe_allow_html=True,
    )
    u1, u2, u3 = st.columns(3)
    _cf_card(u1, "⚡", "Ten tydzień (0–7d)", this_week,   "#E8792A")
    _cf_card(u2, "⏰", "8–14 dni",           next_2weeks, "#F59E0B")
    _cf_card(u3, "📅", "15–30 dni",          this_month,  "#0065BD")

    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
    section_header("Trend % ryzyka w czasie")

    if has_order_date:
        df_trend = df_out.copy()
        df_trend["Data zamowienia"] = pd.to_datetime(df_trend["Data zamowienia"], errors="coerce")
        df_trend["miesiac"] = df_trend["Data zamowienia"].dt.to_period("M")
        monthly = (df_trend.groupby("miesiac")
                   .agg(total=("prediction_label","count"),
                        high_risk=("prediction_label", lambda x:(x==0).sum()))
                   .reset_index())
        monthly["risk_pct"]    = monthly["high_risk"] / monthly["total"] * 100
        monthly["miesiac_str"] = monthly["miesiac"].astype(str)
        monthly = monthly.tail(18)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly["miesiac_str"], y=monthly["risk_pct"],
            mode="lines+markers",
            line=dict(color="#C50F1F", width=2.5),
            marker=dict(size=6, color="#C50F1F", line=dict(color="#FFFFFF", width=1.5)),
            fill="tozeroy", fillcolor="rgba(197,15,31,0.07)",
            hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
        ))
        avg = monthly["risk_pct"].mean()
        fig.add_hline(
            y=avg, line_dash="dash", line_color="#8896A8", line_width=1.5,
            annotation_text=f"Śr. {avg:.0f}%",
            annotation_position="top right",
            annotation_font_color="#6B7A99",
        )
        fig.update_layout(
            **_chart_base,
            xaxis=dict(title="", **_ax),
            yaxis=dict(title="% zamówień wysokiego ryzyka", ticksuffix="%",
                       range=[0, 100], **_ax),
            margin=dict(l=20, r=80, t=20, b=20),
            height=260, showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        if len(monthly) >= 2:
            last  = monthly["risk_pct"].iloc[-1]
            prev  = monthly["risk_pct"].iloc[-2]
            delta = last - prev
            arrow = "▲" if delta > 0 else "▼"
            dclr  = "#C50F1F" if delta > 0 else "#107C10"
            st.markdown(
                f"<span style='color:{dclr}; font-weight:700;'>"
                f"{arrow} {abs(delta):.1f} pp</span>"
                f"<span style='color:#8896A8; margin-left:8px;'>"
                f"vs poprzedni miesiąc · Ostatni miesiąc: "
                f"<strong style='color:#1B2A4A;'>{last:.0f}%</strong></span>",
                unsafe_allow_html=True,
            )

    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
    section_header("Tabela — zamówienia do opłacenia w ciągu 30 dni")

    col_map = {
        "Nazwa klienta": "Klient", "Sprzedawca": "Sprzedawca", "Branża": "Branża",
        "Data zamowienia": "Data", "Wartość faktury": "Wartość faktury (PLN)", "Kwota zapłacona": "Kwota zapłacona (PLN)",
        "prediction_score": "Predykcja (PLN)",
    }

    upcoming = df_hr[df_hr["dni_do_terminu"].between(0, 30)].copy()
    if not upcoming.empty:
        show_cols = [k for k in col_map if k in upcoming.columns]
        df_tbl = upcoming[show_cols].rename(columns=col_map)
        df_tbl.insert(0, "Termin (dni)", upcoming["dni_do_terminu"].astype(int).values)
        df_tbl = df_tbl.sort_values("Termin (dni)").reset_index(drop=True)
        df_tbl.index += 1
        fmt = {c: "{:,.0f}" for c in ["Wartość (PLN)", "Predykcja (PLN)"] if c in df_tbl.columns}
        st.dataframe(df_tbl.style.format(fmt), use_container_width=True)
    else:
        st.info("Brak zamówień wysokiego ryzyka z terminem w ciągu 30 dni.")

# ═══════════════════════════════════════════════════════ TAB 4 ══════
with tab4:

    section_header("Wyjaśnialność modelu — SHAP")
    st.markdown(
        "**Beeswarm**: każda kropka = jedno zamówienie, oś X = wpływ na predykcję. "
        "**Waterfall**: rozkład predykcji dla wybranego rekordu krok po kroku."
    )

    try:
        _pre   = pipeline.named_steps["pre"]
        _model = pipeline.named_steps["model"]
        X_tf   = _pre.transform(X_all)
        feat_names = num_cols + (
            _pre.named_transformers_["cat"]
            .named_steps["enc"]
            .get_feature_names_out(cat_cols)
            .tolist()
        )
        explainer   = shap.TreeExplainer(_model)
        shap_values = explainer(X_tf)
        shap_values.feature_names = feat_names

        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown("**Globalny wpływ cech (Beeswarm)**")
            plt.figure()
            shap.plots.beeswarm(shap_values, max_display=10, show=False)
            st.pyplot(plt.gcf())
            plt.close()

        with sc2:
            st.markdown("**Wyjaśnienie konkretnej predykcji (Waterfall)**")
            sample_idx = st.slider("Wybierz rekord", 0, len(X_all) - 1, 0)
            rec = df_out.iloc[sample_idx]
            rlbl = "🔴 Wysokie" if rec["prediction_label"] == 0 else "🟢 Niskie"
            st.caption(
                f"Rekord #{sample_idx+1} · {rec.get('Nazwa klienta','—')} · "
                f"{rec['prediction_score']:,.0f} PLN · {rlbl}"
            )
            plt.figure()
            shap.plots.waterfall(shap_values[sample_idx], max_display=10, show=False)
            st.pyplot(plt.gcf())
            plt.close()

    except Exception as e:
        st.warning(f"SHAP niedostępny: {e}")

    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
    section_header("Metryki modelu")
    st.markdown("""
| Metryka | Wartość |
|---|---|
| **R²** | 0.91 |
| **MAE** | 1 302 PLN |
| **RMSE** | 1 963 PLN |
| Algorytm | GradientBoostingRegressor |
| Drzewa | 100 |
| Preprocessing | StandardScaler + OneHotEncoder |
| Feature engineering | miesiąc, dzień tygodnia, kwartał |
""")

    with st.expander("Pełna tabela danych z predykcjami"):
        st.dataframe(df_out, use_container_width=True)

# ──────────────────────────────────────────────────────── Footer ──────

st.markdown("""
<div style='text-align:center; color:#8896A8; font-size:0.76rem;
            padding:1.5rem 0 0.5rem; border-top:1px solid #E0E4EA; margin-top:1rem;'>
  GradientBoosting · R²=0.91 · scikit-learn Pipeline · SHAP · Streamlit
</div>""", unsafe_allow_html=True)
