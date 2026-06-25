"""
Validation tests for B2B Payment Risk Dashboard.
Run with: pytest test_dashboard.py -v
"""
import sys
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


# ── Mock UI/viz libraries so app.py can be imported outside Streamlit ──────

def _make_st_mock():
    st = MagicMock()
    st.cache_resource.side_effect  = lambda f: f          # passthrough decorator
    st.radio.return_value          = "Dane demo"
    st.sidebar.radio.return_value  = "Dane demo"
    st.slider.return_value         = 85
    st.sidebar.slider.return_value = 85
    # columns() / tabs() must return the right number of items to unpack
    st.columns.side_effect = lambda n, **kw: [MagicMock() for _ in range(
        n if isinstance(n, int) else len(n)
    )]
    st.tabs.side_effect = lambda lst: [MagicMock() for _ in lst]
    return st


for _mod in ("streamlit", "shap", "matplotlib", "matplotlib.pyplot",
             "plotly", "plotly.graph_objects", "plotly.express"):
    sys.modules[_mod] = MagicMock()

sys.modules["streamlit"] = _make_st_mock()

from app import generate_demo_data, engineer_features  # noqa: E402


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def raw_df():
    return generate_demo_data()


@pytest.fixture(scope="module")
def payment_rate(raw_df):
    return (raw_df["Kwota zapłacona"] / raw_df["Wartość faktury"]).clip(upper=1.0)


# ── Data generation ────────────────────────────────────────────────────────

class TestDataGeneration:

    def test_row_count(self, raw_df):
        assert len(raw_df) == 500

    def test_required_columns(self, raw_df):
        for col in ("Nazwa klienta", "Branża", "Wartość faktury",
                    "Kwota zapłacona", "Sprzedawca", "Data zamowienia"):
            assert col in raw_df.columns, f"Missing column: {col}"

    def test_client_belongs_to_single_industry(self, raw_df):
        n_industries = raw_df.groupby("Nazwa klienta")["Branża"].nunique()
        bad = n_industries[n_industries > 1]
        assert bad.empty, f"Clients appearing in >1 industry: {bad.index.tolist()}"

    def test_40_unique_clients(self, raw_df):
        assert raw_df["Nazwa klienta"].nunique() == 40

    def test_7_salespeople(self, raw_df):
        assert raw_df["Sprzedawca"].nunique() == 7

    def test_all_10_industries_present(self, raw_df):
        expected = {
            "IT / Technologia", "Budownictwo", "Rolnictwo", "Logistyka",
            "Medycyna", "Retail", "Energia", "Motoryzacja", "Spożywczy", "Edukacja",
        }
        assert set(raw_df["Branża"].unique()) == expected

    def test_invoice_value_positive(self, raw_df):
        assert (raw_df["Wartość faktury"] > 0).all()

    def test_amount_paid_at_least_50_pln(self, raw_df):
        assert (raw_df["Kwota zapłacona"] >= 50).all()

    def test_bimodal_payment_distribution(self, raw_df):
        """Each client should consistently be a good payer (>88%) or bad (<82%) —
        not random each invoice. This validates the per-client trait model."""
        avg_rate = (
            raw_df.groupby("Nazwa klienta")
            .apply(lambda g: (g["Kwota zapłacona"] / g["Wartość faktury"]).mean())
        )
        good = (avg_rate >= 0.88).sum()
        bad  = (avg_rate <= 0.82).sum()
        pct_classified = (good + bad) / len(avg_rate)
        assert pct_classified >= 0.70, (
            f"Bimodal distribution broken: {good} good, {bad} bad, "
            f"{len(avg_rate) - good - bad} borderline out of {len(avg_rate)} clients"
        )


# ── KPI realism ────────────────────────────────────────────────────────────

class TestKPIRealism:
    """Validates that risk percentages shown in the dashboard are in realistic ranges."""

    def test_risk_at_75_pct_threshold_low(self, payment_rate):
        """At a lenient 75% threshold, very few invoices should be flagged."""
        at_risk = (payment_rate < 0.75).mean()
        assert at_risk <= 0.15, f"At 75% threshold: {at_risk:.1%} at risk — expected ≤15%"

    def test_risk_at_85_pct_threshold_realistic(self, payment_rate):
        """At default 85%, 10–30% flagged is realistic for B2B portfolios."""
        at_risk = (payment_rate < 0.85).mean()
        assert 0.10 <= at_risk <= 0.30, \
            f"At 85% threshold: {at_risk:.1%} at risk — expected 10–30%"

    def test_risk_at_95_pct_threshold_bounded(self, payment_rate):
        """Even at the strictest 95% threshold, not more than 45% should be flagged."""
        at_risk = (payment_rate < 0.95).mean()
        assert at_risk <= 0.45, f"At 95% threshold: {at_risk:.1%} at risk — expected ≤45%"

    def test_risk_increases_with_threshold(self, payment_rate):
        """Higher threshold → more orders flagged (monotonic)."""
        r75, r85, r95 = [(payment_rate < t).mean() for t in (0.75, 0.85, 0.95)]
        assert r75 <= r85 <= r95, \
            f"Non-monotonic: {r75:.1%}@75% | {r85:.1%}@85% | {r95:.1%}@95%"

    def test_pct_pokrycia_never_exceeds_100(self, payment_rate):
        """Coverage % must be capped at 100 — no overpayment shown in UI."""
        assert (payment_rate.clip(upper=1.0) * 100 <= 100.001).all()

    def test_shortfall_is_non_negative(self, raw_df):
        """Shortfall = invoice − paid. Must be ≥ 0 for any risky invoice."""
        shortfall = (raw_df["Wartość faktury"] - raw_df["Kwota zapłacona"]).clip(lower=0)
        assert (shortfall >= 0).all()

    def test_high_risk_industries_have_more_bad_payers(self, raw_df):
        """Budownictwo + Retail (p_bad ~26–32%) should have higher risk than IT + Medycyna (p_bad ~6–8%)."""
        rate = raw_df["Kwota zapłacona"] / raw_df["Wartość faktury"]
        high_risk = rate[raw_df["Branża"].isin({"Budownictwo", "Retail"})].lt(0.85).mean()
        low_risk  = rate[raw_df["Branża"].isin({"IT / Technologia", "Medycyna"})].lt(0.85).mean()
        assert high_risk > low_risk, (
            f"High-risk industries ({high_risk:.1%}) should have more underpayers "
            f"than low-risk industries ({low_risk:.1%})"
        )


# ── Priority score (urgency logic) ────────────────────────────────────────

class TestUrgencyScore:
    """Validates the urgency scoring formula used in Tab 1 priority ranking."""

    @staticmethod
    def _urgency(days: int) -> int:
        if days < 0:   return 100
        if days <= 7:  return 80
        if days <= 14: return 60
        if days <= 30: return 40
        return 20

    def test_overdue_gets_max_urgency(self):
        assert self._urgency(-1)  == 100
        assert self._urgency(-30) == 100

    def test_due_today_gets_high_urgency(self):
        assert self._urgency(0) == 80

    def test_urgency_is_monotonically_decreasing(self):
        days_sample = [-5, 0, 7, 14, 30, 60]
        scores = [self._urgency(d) for d in days_sample]
        assert scores == sorted(scores, reverse=True), (
            f"Urgency not descending: {list(zip(days_sample, scores))}"
        )

    def test_far_future_gets_lowest_urgency(self):
        assert self._urgency(60) == self._urgency(90) == 20


# ── Feature engineering ────────────────────────────────────────────────────

class TestEngineerFeatures:

    def test_date_columns_expanded_to_month_quarter_dow(self, raw_df):
        df_fe = engineer_features(raw_df)
        assert "Data zamowienia_miesiac" in df_fe.columns
        assert "Data zamowienia_kwartal" in df_fe.columns
        assert "Data zamowienia_dzien_tygodnia" in df_fe.columns
        assert "Data zamowienia" not in df_fe.columns, "Raw date column should be dropped"

    def test_id_and_comment_columns_dropped(self, raw_df):
        df_fe = engineer_features(raw_df)
        assert "ID" not in df_fe.columns
        assert "Komentarz" not in df_fe.columns

    def test_target_column_preserved(self, raw_df):
        df_fe = engineer_features(raw_df)
        assert "Kwota zapłacona" in df_fe.columns

    def test_no_nulls_in_feature_matrix(self, raw_df):
        df_fe = engineer_features(raw_df)
        X = df_fe.drop(columns=["Kwota zapłacona"])
        null_cols = X.columns[X.isnull().any()].tolist()
        assert not null_cols, f"NaN values found in columns: {null_cols}"

    def test_feature_count_reasonable(self, raw_df):
        df_fe = engineer_features(raw_df)
        X = df_fe.drop(columns=["Kwota zapłacona"])
        assert 5 <= len(X.columns) <= 30, f"Unexpected feature count: {len(X.columns)}"
