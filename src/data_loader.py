"""
src/data_loader.py
==================
Data ingestion, cleaning, and feature engineering utilities for the
AlphaCare Insurance Risk Analytics project.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RAW_DATA_PATH = Path("data/raw/MachineLearningRating_v3.txt")
PROCESSED_DATA_PATH = Path("data/processed/insurance_cleaned.csv")

# Columns that should be numeric but may arrive as object/string
NUMERIC_COLS = [
    "TotalPremium",
    "TotalClaims",
    "SumInsured",
    "CalculatedPremiumPerTerm",
    "CustomValueEstimate",
    "CapitalOutstanding",
    "Cylinders",
    "cubiccapacity",
    "kilowatts",
    "NumberOfDoors",
    "NumberOfVehiclesInFleet",
    "RegistrationYear",
    "ExcessSelected",
]

CATEGORICAL_COLS = [
    "IsVATRegistered",
    "Citizenship",
    "LegalType",
    "Title",
    "Language",
    "Bank",
    "AccountType",
    "MaritalStatus",
    "Gender",
    "Country",
    "Province",
    "MainCrestaZone",
    "SubCrestaZone",
    "ItemType",
    "VehicleType",
    "make",
    "Model",
    "bodytype",
    "AlarmImmobiliser",
    "TrackingDevice",
    "NewVehicle",
    "WrittenOff",
    "Rebuilt",
    "Converted",
    "CrossBorder",
    "TermFrequency",
    "CoverCategory",
    "CoverType",
    "CoverGroup",
    "Section",
    "Product",
    "StatutoryClass",
    "StatutoryRiskType",
]

DATE_COLS = ["TransactionMonth", "VehicleIntroDate"]


# ---------------------------------------------------------------------------
# Core loading function
# ---------------------------------------------------------------------------


def load_raw_data(filepath: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw pipe-delimited insurance data file.

    Parameters
    ----------
    filepath : str or Path
        Path to the raw .txt file.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame with all original columns.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"Raw data file not found at '{filepath}'. "
            "Run `dvc pull` to fetch the data from the remote store."
        )

    logger.info(f"Loading raw data from {filepath} …")
    df = pd.read_csv(
        filepath,
        sep="|",
        low_memory=False,
        encoding="utf-8",
        on_bad_lines="warn",
    )
    logger.info(f"Loaded {len(df):,} rows × {len(df.columns)} columns.")
    return df


# ---------------------------------------------------------------------------
# Type coercion
# ---------------------------------------------------------------------------


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce columns to their correct dtypes (numeric, categorical, datetime).

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        DataFrame with corrected dtypes.
    """
    df = df.copy()

    # Numeric columns
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Date columns
    for col in DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Categorical columns – strip leading/trailing whitespace first
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({"nan": np.nan, "": np.nan})
            df[col] = df[col].astype("category")

    # PostalCode – treat as string (zone identifier, not numeric)
    if "PostalCode" in df.columns:
        df["PostalCode"] = df["PostalCode"].astype(str).str.strip().replace("nan", np.nan)

    logger.info("Column dtypes coerced.")
    return df


# ---------------------------------------------------------------------------
# Missing-value handling
# ---------------------------------------------------------------------------


def assess_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a summary DataFrame of missing-value counts and percentages.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Columns: ['missing_count', 'missing_pct'].
    """
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100).round(2)
    summary = pd.DataFrame({"missing_count": missing, "missing_pct": pct})
    return summary[summary["missing_count"] > 0].sort_values("missing_pct", ascending=False)


def handle_missing(df: pd.DataFrame, threshold: float = 50.0) -> pd.DataFrame:
    """
    Handle missing values with a tiered strategy:

    - Columns with > ``threshold`` % missing → drop entirely.
    - Numeric columns with ≤ threshold missing → median imputation.
    - Categorical columns with ≤ threshold missing → mode imputation.

    Parameters
    ----------
    df : pd.DataFrame
    threshold : float
        Percentage above which a column is dropped (default 50 %).

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    missing_pct = df.isnull().mean() * 100

    cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
    if cols_to_drop:
        logger.info(f"Dropping {len(cols_to_drop)} columns with >{threshold}% missing: {cols_to_drop}")
        df.drop(columns=cols_to_drop, inplace=True)

    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                fill_val = df[col].median()
                df[col].fillna(fill_val, inplace=True)
            else:
                mode_vals = df[col].mode()
                fill_val = mode_vals.iloc[0] if len(mode_vals) > 0 else "Unknown"
                df[col].fillna(fill_val, inplace=True)

    logger.info("Missing values handled.")
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features relevant to insurance risk analysis.

    New columns:
    - ``LossRatio``         : TotalClaims / TotalPremium
    - ``Margin``            : TotalPremium - TotalClaims
    - ``HasClaim``          : 1 if TotalClaims > 0 else 0
    - ``VehicleAge``        : TransactionMonth.year - RegistrationYear
    - ``TransactionYear``   : Year of TransactionMonth
    - ``TransactionQuarter``: Quarter of TransactionMonth

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()

    # Core financial metrics
    df["LossRatio"] = np.where(
        df["TotalPremium"] > 0,
        df["TotalClaims"] / df["TotalPremium"],
        np.nan,
    )
    df["Margin"] = df["TotalPremium"] - df["TotalClaims"]

    # Claim indicator
    df["HasClaim"] = (df["TotalClaims"] > 0).astype(int)

    # Vehicle age at transaction time
    if "TransactionMonth" in df.columns and "RegistrationYear" in df.columns:
        tx_year = df["TransactionMonth"].dt.year
        df["VehicleAge"] = (tx_year - df["RegistrationYear"]).clip(lower=0)

    # Temporal features
    if "TransactionMonth" in df.columns:
        df["TransactionYear"] = df["TransactionMonth"].dt.year
        df["TransactionQuarter"] = df["TransactionMonth"].dt.quarter
        df["TransactionMonthNum"] = df["TransactionMonth"].dt.month

    logger.info("Feature engineering complete.")
    return df


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def load_and_prepare(
    raw_path: str | Path = RAW_DATA_PATH,
    save_processed: bool = True,
    processed_path: str | Path = PROCESSED_DATA_PATH,
) -> pd.DataFrame:
    """
    Full end-to-end data preparation pipeline:
    load → coerce types → handle missing → engineer features → (optionally save).

    Parameters
    ----------
    raw_path : str or Path
    save_processed : bool
    processed_path : str or Path

    Returns
    -------
    pd.DataFrame
        Cleaned and feature-engineered DataFrame.
    """
    df = load_raw_data(raw_path)
    df = coerce_types(df)
    df = handle_missing(df)
    df = engineer_features(df)

    if save_processed:
        processed_path = Path(processed_path)
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(processed_path, index=False)
        logger.info(f"Processed data saved to {processed_path}")

    return df


# ---------------------------------------------------------------------------
# Entry point (used by DVC pipeline)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s – %(message)s")
    load_and_prepare()
