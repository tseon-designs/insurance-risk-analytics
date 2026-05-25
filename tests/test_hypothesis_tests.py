"""
tests/test_hypothesis_tests.py
================================
Unit tests for src/hypothesis_tests.py
"""
import numpy as np
import pandas as pd
import pytest

from src.hypothesis_tests import (
    TestResult,
    claim_frequency,
    claim_severity,
    chi_squared_test,
    kruskal_test,
    margin,
    results_table,
    test_risk_by_gender,
    test_risk_by_province,
    two_sample_ttest,
)


@pytest.fixture
def insurance_df():
    np.random.seed(42)
    n = 1000
    provinces = ["Gauteng", "Western Cape", "KwaZulu-Natal", "Eastern Cape", "Limpopo"]
    gender_vals = ["Male", "Female"]
    postal_codes = [str(p) for p in range(1000, 1020)]

    claims = np.random.exponential(3000, n) * (np.random.rand(n) > 0.75)
    premiums = np.random.uniform(200, 2000, n)

    return pd.DataFrame(
        {
            "PolicyID": range(n),
            "TotalPremium": premiums,
            "TotalClaims": claims,
            "Province": pd.Categorical(np.random.choice(provinces, n)),
            "Gender": pd.Categorical(np.random.choice(gender_vals, n)),
            "PostalCode": np.random.choice(postal_codes, n),
        }
    )


class TestStatisticalFunctions:
    def test_ttest_returns_float_tuple(self):
        a = pd.Series(np.random.normal(100, 10, 100))
        b = pd.Series(np.random.normal(120, 10, 100))
        stat, p = two_sample_ttest(a, b)
        assert isinstance(stat, float)
        assert 0.0 <= p <= 1.0

    def test_chi_squared_returns_valid_p(self, insurance_df):
        insurance_df["HasClaim"] = (insurance_df["TotalClaims"] > 0).astype(int)
        chi2, p = chi_squared_test(insurance_df, "Province", "HasClaim")
        assert chi2 >= 0
        assert 0.0 <= p <= 1.0

    def test_kruskal_returns_float_tuple(self):
        g1 = pd.Series(np.random.normal(100, 10, 50))
        g2 = pd.Series(np.random.normal(200, 10, 50))
        g3 = pd.Series(np.random.normal(150, 10, 50))
        stat, p = kruskal_test(g1, g2, g3)
        assert stat >= 0
        assert 0.0 <= p <= 1.0


class TestClaimMetrics:
    def test_claim_frequency_in_range(self, insurance_df):
        freq = claim_frequency(insurance_df)
        assert 0.0 <= freq <= 1.0

    def test_claim_severity_only_positive(self, insurance_df):
        sev = claim_severity(insurance_df)
        assert (sev > 0).all()

    def test_margin_formula(self, insurance_df):
        m = margin(insurance_df)
        expected = insurance_df["TotalPremium"] - insurance_df["TotalClaims"]
        pd.testing.assert_series_equal(m.reset_index(drop=True), expected.reset_index(drop=True))


class TestTestResult:
    def test_reject_h0_when_p_below_alpha(self):
        result = TestResult(
            hypothesis="Test H",
            kpi="Claim Freq",
            group_a_label="A",
            group_b_label="B",
            test_name="t-test",
            statistic=3.5,
            p_value=0.001,
            alpha=0.05,
        )
        assert result.reject_h0 is True
        assert result.decision == "Reject H₀"

    def test_fail_to_reject_when_p_above_alpha(self):
        result = TestResult(
            hypothesis="Test H",
            kpi="Claim Freq",
            group_a_label="A",
            group_b_label="B",
            test_name="t-test",
            statistic=0.5,
            p_value=0.45,
            alpha=0.05,
        )
        assert result.reject_h0 is False
        assert result.decision == "Fail to Reject H₀"

    def test_to_dict_has_required_keys(self):
        result = TestResult(
            hypothesis="H",
            kpi="KPI",
            group_a_label="A",
            group_b_label="B",
            test_name="test",
            statistic=1.0,
            p_value=0.05,
        )
        d = result.to_dict()
        for key in ["Hypothesis", "KPI", "Test", "p-value", "Decision"]:
            assert key in d


class TestHypothesisFunctions:
    def test_province_test_returns_list(self, insurance_df):
        results = test_risk_by_province(insurance_df)
        assert isinstance(results, list)
        assert len(results) >= 1

    def test_gender_test_returns_list(self, insurance_df):
        results = test_risk_by_gender(insurance_df)
        assert isinstance(results, list)
        assert len(results) >= 1

    def test_results_table_is_dataframe(self, insurance_df):
        results = test_risk_by_province(insurance_df)
        table = results_table(results)
        assert isinstance(table, pd.DataFrame)
        assert "p-value" in table.columns
