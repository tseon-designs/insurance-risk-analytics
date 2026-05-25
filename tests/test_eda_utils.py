"""
tests/test_eda_utils.py
=======================
Unit tests for src/eda_utils.py
"""
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.eda_utils import (
    categorical_summary,
    numeric_summary,
    plot_boxplots,
    plot_correlation_matrix,
    plot_loss_ratio_by_province,
)


@pytest.fixture
def sample_df():
    np.random.seed(0)
    n = 200
    provinces = ["Gauteng", "Western Cape", "KwaZulu-Natal", "Eastern Cape"]
    return pd.DataFrame(
        {
            "TotalPremium": np.random.exponential(500, n),
            "TotalClaims": np.random.exponential(300, n) * (np.random.rand(n) > 0.7),
            "SumInsured": np.random.uniform(10000, 200000, n),
            "LossRatio": np.random.uniform(0, 2, n),
            "Margin": np.random.normal(200, 300, n),
            "Province": pd.Categorical(np.random.choice(provinces, n)),
            "Gender": pd.Categorical(np.random.choice(["Male", "Female"], n)),
            "VehicleType": pd.Categorical(np.random.choice(["Passenger Vehicle", "Commercial"], n)),
        }
    )


class TestNumericSummary:
    def test_returns_dataframe(self, sample_df):
        result = numeric_summary(sample_df)
        assert isinstance(result, pd.DataFrame)

    def test_contains_skewness(self, sample_df):
        result = numeric_summary(sample_df)
        assert "skewness" in result.columns

    def test_contains_all_numeric_cols(self, sample_df):
        result = numeric_summary(sample_df)
        numeric_cols = sample_df.select_dtypes(include="number").columns.tolist()
        for col in numeric_cols:
            assert col in result.index


class TestCategoricalSummary:
    def test_returns_dataframe(self, sample_df):
        result = categorical_summary(sample_df)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, sample_df):
        result = categorical_summary(sample_df)
        assert "n_unique" in result.columns


class TestPlots:
    def test_boxplots_returns_figure(self, sample_df):
        fig = plot_boxplots(sample_df, cols=["TotalPremium", "TotalClaims"])
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_correlation_matrix_returns_figure(self, sample_df):
        fig = plot_correlation_matrix(sample_df, cols=["TotalPremium", "TotalClaims", "SumInsured"])
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_loss_ratio_by_province_returns_figure(self, sample_df):
        fig = plot_loss_ratio_by_province(sample_df)
        assert isinstance(fig, plt.Figure)
        plt.close("all")
