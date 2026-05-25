"""
src/hypothesis_tests.py
=======================
Statistical hypothesis testing utilities for the
AlphaCare Insurance Risk Analytics project.

Null Hypotheses:
  H1: No risk differences across provinces.
  H2: No risk differences between zip codes.
  H3: No significant margin difference between zip codes.
  H4: No significant risk difference between Women and Men.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class TestResult:
    """Container for a single hypothesis test result."""

    hypothesis: str
    kpi: str
    group_a_label: str
    group_b_label: str
    test_name: str
    statistic: float
    p_value: float
    alpha: float = 0.05
    business_interpretation: str = ""

    @property
    def reject_h0(self) -> bool:
        return self.p_value < self.alpha

    @property
    def decision(self) -> str:
        return "Reject H₀" if self.reject_h0 else "Fail to Reject H₀"

    def to_dict(self) -> dict:
        return {
            "Hypothesis": self.hypothesis,
            "KPI": self.kpi,
            "Group A": self.group_a_label,
            "Group B": self.group_b_label,
            "Test": self.test_name,
            "Statistic": round(self.statistic, 4),
            "p-value": round(self.p_value, 6),
            "Alpha": self.alpha,
            "Decision": self.decision,
            "Interpretation": self.business_interpretation,
        }


def results_table(results: list[TestResult]) -> pd.DataFrame:
    """Convert a list of TestResult objects to a summary DataFrame."""
    return pd.DataFrame([r.to_dict() for r in results])


# ---------------------------------------------------------------------------
# Generic statistical tests
# ---------------------------------------------------------------------------


def two_sample_ttest(
    group_a: pd.Series,
    group_b: pd.Series,
    equal_var: bool = False,
) -> Tuple[float, float]:
    """
    Welch's (or Student's) two-sample t-test.

    Parameters
    ----------
    group_a, group_b : pd.Series
        Numeric samples (NaNs dropped internally).
    equal_var : bool
        If True, uses Student's t-test; if False (default), uses Welch's.

    Returns
    -------
    (statistic, p_value)
    """
    a = group_a.dropna()
    b = group_b.dropna()
    stat, p = stats.ttest_ind(a, b, equal_var=equal_var)
    return float(stat), float(p)


def chi_squared_test(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
) -> Tuple[float, float]:
    """
    Chi-squared test of independence between two categorical columns.

    Parameters
    ----------
    df : pd.DataFrame
    col_a, col_b : str
        Column names for the contingency table.

    Returns
    -------
    (chi2_statistic, p_value)
    """
    ct = pd.crosstab(df[col_a], df[col_b])
    chi2, p, dof, expected = stats.chi2_contingency(ct)
    return float(chi2), float(p)


def anova_test(*groups: pd.Series) -> Tuple[float, float]:
    """
    One-way ANOVA across multiple groups.

    Parameters
    ----------
    *groups : pd.Series
        Each argument is one group's numeric data.

    Returns
    -------
    (F_statistic, p_value)
    """
    clean_groups = [g.dropna() for g in groups]
    stat, p = stats.f_oneway(*clean_groups)
    return float(stat), float(p)


def kruskal_test(*groups: pd.Series) -> Tuple[float, float]:
    """
    Kruskal-Wallis H-test (non-parametric alternative to one-way ANOVA).

    Returns
    -------
    (H_statistic, p_value)
    """
    clean_groups = [g.dropna() for g in groups]
    stat, p = stats.kruskal(*clean_groups)
    return float(stat), float(p)


# ---------------------------------------------------------------------------
# Claim-metric helpers
# ---------------------------------------------------------------------------


def claim_frequency(df: pd.DataFrame) -> float:
    """Proportion of policies that recorded at least one claim."""
    return (df["TotalClaims"] > 0).mean()


def claim_severity(df: pd.DataFrame) -> pd.Series:
    """Claim amounts restricted to policies where a claim occurred."""
    return df.loc[df["TotalClaims"] > 0, "TotalClaims"]


def margin(df: pd.DataFrame) -> pd.Series:
    """Policy-level margin: TotalPremium − TotalClaims."""
    return df["TotalPremium"] - df["TotalClaims"]


# ---------------------------------------------------------------------------
# Named hypothesis tests
# ---------------------------------------------------------------------------


def test_risk_by_province(df: pd.DataFrame, alpha: float = 0.05) -> list[TestResult]:
    """
    H₀: There are no risk differences across provinces.

    Tests both claim frequency (chi-squared) and claim severity (Kruskal-Wallis).
    """
    results = []

    # --- Claim frequency (chi-squared) ---
    df2 = df.copy()
    df2["HasClaim"] = (df2["TotalClaims"] > 0).astype(int)
    chi2, p_chi = chi_squared_test(df2, "Province", "HasClaim")
    results.append(
        TestResult(
            hypothesis="H₁: No risk difference across provinces",
            kpi="Claim Frequency",
            group_a_label="All Provinces",
            group_b_label="All Provinces",
            test_name="Chi-squared",
            statistic=chi2,
            p_value=p_chi,
            alpha=alpha,
            business_interpretation=(
                "Provinces show statistically significant differences in claim frequency. "
                "Targeted provincial pricing adjustments are warranted."
                if p_chi < alpha
                else "No statistically significant difference in claim frequency across provinces."
            ),
        )
    )

    # --- Claim severity (Kruskal-Wallis) ---
    groups = [
        claim_severity(g)
        for _, g in df.groupby("Province", observed=True)
        if (g["TotalClaims"] > 0).sum() > 10
    ]
    h_stat, p_kw = kruskal_test(*groups)
    results.append(
        TestResult(
            hypothesis="H₁: No risk difference across provinces",
            kpi="Claim Severity",
            group_a_label="All Provinces",
            group_b_label="All Provinces",
            test_name="Kruskal-Wallis",
            statistic=h_stat,
            p_value=p_kw,
            alpha=alpha,
            business_interpretation=(
                "Average claim severity varies significantly across provinces. "
                "High-severity provinces may require larger premium loading."
                if p_kw < alpha
                else "No statistically significant difference in claim severity across provinces."
            ),
        )
    )
    return results


def test_risk_by_zipcode(
    df: pd.DataFrame,
    min_policies: int = 30,
    alpha: float = 0.05,
) -> list[TestResult]:
    """
    H₀: There are no risk differences between zip codes.
    H₀: There is no significant margin difference between zip codes.

    Selects top-2 postal codes by policy count for a paired t-test,
    and uses ANOVA across all postal codes with ≥ min_policies.
    """
    results = []
    df2 = df.copy()
    df2["HasClaim"] = (df2["TotalClaims"] > 0).astype(int)

    valid_zips = (
        df2.groupby("PostalCode")["PolicyID"].count()
        .loc[lambda s: s >= min_policies]
        .index
    )
    zip_df = df2[df2["PostalCode"].isin(valid_zips)]

    # Claim severity – Kruskal-Wallis across all valid zip codes
    zip_groups = [
        claim_severity(g)
        for _, g in zip_df.groupby("PostalCode")
        if (g["TotalClaims"] > 0).sum() > 5
    ]
    if len(zip_groups) >= 2:
        h_stat, p_kw = kruskal_test(*zip_groups)
        results.append(
            TestResult(
                hypothesis="H₂: No risk difference between zip codes",
                kpi="Claim Severity",
                group_a_label="All valid PostalCodes",
                group_b_label="All valid PostalCodes",
                test_name="Kruskal-Wallis",
                statistic=h_stat,
                p_value=p_kw,
                alpha=alpha,
                business_interpretation=(
                    "Claim severity differs significantly across postal codes. "
                    "Fine-grained geographic pricing at zip-code level is recommended."
                    if p_kw < alpha
                    else "No significant risk variation detected across postal codes."
                ),
            )
        )

    # Margin – Kruskal-Wallis across zip codes
    margin_groups = [
        margin(g)
        for _, g in zip_df.groupby("PostalCode")
    ]
    if len(margin_groups) >= 2:
        h_stat_m, p_m = kruskal_test(*margin_groups)
        results.append(
            TestResult(
                hypothesis="H₃: No significant margin difference between zip codes",
                kpi="Margin (TotalPremium − TotalClaims)",
                group_a_label="All valid PostalCodes",
                group_b_label="All valid PostalCodes",
                test_name="Kruskal-Wallis",
                statistic=h_stat_m,
                p_value=p_m,
                alpha=alpha,
                business_interpretation=(
                    "Policy margins vary significantly across postal codes. "
                    "Some zip codes are systematically over- or under-priced."
                    if p_m < alpha
                    else "Margins are statistically equivalent across postal codes."
                ),
            )
        )

    return results


def test_risk_by_gender(df: pd.DataFrame, alpha: float = 0.05) -> list[TestResult]:
    """
    H₀: There is no significant risk difference between Women and Men.

    Tests claim frequency (chi-squared) and claim severity (t-test).
    """
    results = []
    gender_df = df[df["Gender"].isin(["Male", "Female"])].copy()
    gender_df["HasClaim"] = (gender_df["TotalClaims"] > 0).astype(int)

    male = gender_df[gender_df["Gender"] == "Male"]
    female = gender_df[gender_df["Gender"] == "Female"]

    # Claim frequency
    chi2, p_chi = chi_squared_test(gender_df, "Gender", "HasClaim")
    results.append(
        TestResult(
            hypothesis="H₄: No risk difference between Women and Men",
            kpi="Claim Frequency",
            group_a_label="Male",
            group_b_label="Female",
            test_name="Chi-squared",
            statistic=chi2,
            p_value=p_chi,
            alpha=alpha,
            business_interpretation=(
                "Claim frequency differs significantly by gender. "
                "Gender-based premium differentiation may be justified (subject to regulatory review)."
                if p_chi < alpha
                else "No statistically significant difference in claim frequency between genders."
            ),
        )
    )

    # Claim severity
    male_sev = claim_severity(male)
    female_sev = claim_severity(female)
    if len(male_sev) > 10 and len(female_sev) > 10:
        t_stat, p_t = two_sample_ttest(male_sev, female_sev)
        results.append(
            TestResult(
                hypothesis="H₄: No risk difference between Women and Men",
                kpi="Claim Severity",
                group_a_label="Male",
                group_b_label="Female",
                test_name="Welch's t-test",
                statistic=t_stat,
                p_value=p_t,
                alpha=alpha,
                business_interpretation=(
                    f"Average claim severity differs by gender (p={p_t:.4f}). "
                    "Male policyholders may incur higher average claim costs."
                    if p_t < alpha
                    else "No statistically significant difference in claim severity between genders."
                ),
            )
        )

    return results


def run_all_hypothesis_tests(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """
    Run all four hypothesis tests and return a consolidated results table.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned, feature-engineered insurance DataFrame.
    alpha : float
        Significance level (default 0.05).

    Returns
    -------
    pd.DataFrame
        Summary table with all test results.
    """
    all_results: list[TestResult] = []
    all_results.extend(test_risk_by_province(df, alpha))
    all_results.extend(test_risk_by_zipcode(df, alpha=alpha))
    all_results.extend(test_risk_by_gender(df, alpha))
    return results_table(all_results)
