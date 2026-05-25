"""
tests/test_data_loader.py
=========================
Unit tests for src/data_loader.py
"""
import numpy as np
import pandas as pd
import pytest

from src.data_loader import (
    assess_missing,
    coerce_types,
    engineer_features,
    handle_missing,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    """Minimal DataFrame mimicking the insurance dataset."""
    return pd.DataFrame(
        {
            "PolicyID": [1, 2, 3, 4, 5],
            "TransactionMonth": [
                "2014-02-01", "2014-05-01", "2015-01-01", "2015-06-01", "2015-08-01"
            ],
            "TotalPremium": [500.0, 200.0, 0.0, 1500.0, 800.0],
            "TotalClaims": [0.0, 150.0, 0.0, 2000.0, 0.0],
            "SumInsured": [50000, 20000, 10000, 150000, 80000],
            "CalculatedPremiumPerTerm": [500.0, 200.0, 0.0, 1500.0, 800.0],
            "CustomValueEstimate": [np.nan, 20000, 10000, 150000, 80000],
            "CapitalOutstanding": [50000, 20000, 10000, 150000, 80000],
            "RegistrationYear": [2010, 2015, 2008, 2012, 2014],
            "Gender": ["Male", "Female", "Male", "Not specified", "Female"],
            "Province": ["Gauteng", "Western Cape", "Gauteng", "KwaZulu-Natal", "Western Cape"],
            "VehicleType": ["Passenger Vehicle"] * 5,
            "make": ["TOYOTA", "BMW", "FORD", "MERCEDES-BENZ", "VW"],
            "TotalClaims": [0.0, 150.0, 0.0, 2000.0, 0.0],
        }
    )


# ---------------------------------------------------------------------------
# coerce_types
# ---------------------------------------------------------------------------

class TestCoerceTypes:
    def test_transaction_month_becomes_datetime(self, sample_df):
        result = coerce_types(sample_df)
        assert pd.api.types.is_datetime64_any_dtype(result["TransactionMonth"])

    def test_total_premium_is_numeric(self, sample_df):
        result = coerce_types(sample_df)
        assert pd.api.types.is_numeric_dtype(result["TotalPremium"])

    def test_no_rows_dropped(self, sample_df):
        result = coerce_types(sample_df)
        assert len(result) == len(sample_df)


# ---------------------------------------------------------------------------
# assess_missing
# ---------------------------------------------------------------------------

class TestAssessMissing:
    def test_returns_dataframe(self, sample_df):
        result = assess_missing(sample_df)
        assert isinstance(result, pd.DataFrame)

    def test_detects_known_missing(self, sample_df):
        result = assess_missing(sample_df)
        assert "CustomValueEstimate" in result.index

    def test_missing_pct_in_range(self, sample_df):
        result = assess_missing(sample_df)
        assert result["missing_pct"].between(0, 100).all()


# ---------------------------------------------------------------------------
# handle_missing
# ---------------------------------------------------------------------------

class TestHandleMissing:
    def test_no_nans_after_handling(self, sample_df):
        df_typed = coerce_types(sample_df)
        result = handle_missing(df_typed)
        numeric_cols = result.select_dtypes(include="number").columns
        assert result[numeric_cols].isnull().sum().sum() == 0

    def test_shape_preserved_for_low_missing(self, sample_df):
        df_typed = coerce_types(sample_df)
        result = handle_missing(df_typed, threshold=50.0)
        # Only 1/5 = 20% missing in CustomValueEstimate → should be kept
        assert "CustomValueEstimate" in result.columns

    def test_high_missing_col_dropped(self):
        df = pd.DataFrame({
            "A": [1, 2, 3, 4, 5],
            "B": [np.nan, np.nan, np.nan, np.nan, 1.0],  # 80% missing
        })
        result = handle_missing(df, threshold=50.0)
        assert "B" not in result.columns


# ---------------------------------------------------------------------------
# engineer_features
# ---------------------------------------------------------------------------

class TestEngineerFeatures:
    def test_loss_ratio_created(self, sample_df):
        df_typed = coerce_types(sample_df)
        df_clean = handle_missing(df_typed)
        result = engineer_features(df_clean)
        assert "LossRatio" in result.columns

    def test_margin_created(self, sample_df):
        df_typed = coerce_types(sample_df)
        df_clean = handle_missing(df_typed)
        result = engineer_features(df_clean)
        assert "Margin" in result.columns

    def test_has_claim_binary(self, sample_df):
        df_typed = coerce_types(sample_df)
        df_clean = handle_missing(df_typed)
        result = engineer_features(df_clean)
        assert set(result["HasClaim"].unique()).issubset({0, 1})

    def test_margin_equals_premium_minus_claims(self, sample_df):
        df_typed = coerce_types(sample_df)
        df_clean = handle_missing(df_typed)
        result = engineer_features(df_clean)
        expected = result["TotalPremium"] - result["TotalClaims"]
        pd.testing.assert_series_equal(result["Margin"], expected, check_names=False)

    def test_vehicle_age_non_negative(self, sample_df):
        df_typed = coerce_types(sample_df)
        df_clean = handle_missing(df_typed)
        result = engineer_features(df_clean)
        if "VehicleAge" in result.columns:
            assert (result["VehicleAge"] >= 0).all()
