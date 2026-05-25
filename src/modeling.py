"""
src/modeling.py
===============
Model training, evaluation, and interpretability utilities for the
AlphaCare Insurance Risk Analytics predictive modeling task.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

# Optional heavy imports
try:
    import xgboost as xgb

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logger.warning("XGBoost not installed; XGB models will be skipped.")

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not installed; SHAP analysis will be skipped.")


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------


FEATURE_COLS = [
    "SumInsured",
    "CalculatedPremiumPerTerm",
    "CustomValueEstimate",
    "CapitalOutstanding",
    "VehicleAge",
    "RegistrationYear",
    "Cylinders",
    "cubiccapacity",
    "kilowatts",
    "NumberOfDoors",
    "TransactionYear",
    "TransactionQuarter",
    "TransactionMonthNum",
    "Province",
    "VehicleType",
    "make",
    "bodytype",
    "CoverType",
    "CoverGroup",
    "Gender",
    "MaritalStatus",
    "AlarmImmobiliser",
    "TrackingDevice",
    "NewVehicle",
]


def prepare_features(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Optional[list[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Prepare features and target for modeling:
    - Drop rows with missing target.
    - Label-encode categorical columns.
    - Split into train/test.

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str
    feature_cols : list of str, optional
    test_size : float
    random_state : int

    Returns
    -------
    X_train, X_test, y_train, y_test, feature_names
    """
    if feature_cols is None:
        feature_cols = [c for c in FEATURE_COLS if c in df.columns]

    subset = df[feature_cols + [target_col]].copy().dropna(subset=[target_col])

    # Label-encode categorical columns
    for col in subset.select_dtypes(include=["category", "object"]).columns:
        if col == target_col:
            continue
        le = LabelEncoder()
        subset[col] = le.fit_transform(subset[col].astype(str))

    # Fill any remaining NaNs with median
    subset.fillna(subset.median(numeric_only=True), inplace=True)

    X = subset[feature_cols].values
    y = subset[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.info(
        f"Train size: {len(X_train):,} | Test size: {len(X_test):,} | Features: {len(feature_cols)}"
    )
    return X_train, X_test, y_train, y_test, feature_cols


# ---------------------------------------------------------------------------
# Regression models (Claim Severity)
# ---------------------------------------------------------------------------


def train_linear_regression(
    X_train: np.ndarray, y_train: np.ndarray
) -> Pipeline:
    """Train a Linear Regression model with standard scaling."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def train_random_forest_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 200,
    max_depth: int = 12,
    random_state: int = 42,
) -> RandomForestRegressor:
    """Train a Random Forest Regressor."""
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def train_xgb_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    random_state: int = 42,
) -> Any:
    """Train an XGBoost Regressor."""
    if not XGB_AVAILABLE:
        raise ImportError("XGBoost is not installed.")
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)
    return model


def evaluate_regression(
    model: Any, X_test: np.ndarray, y_test: np.ndarray, model_name: str = "Model"
) -> Dict[str, float]:
    """
    Evaluate a regression model with RMSE and R².

    Returns
    -------
    dict with keys: model, RMSE, R2
    """
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    logger.info(f"{model_name} – RMSE: {rmse:,.2f} | R²: {r2:.4f}")
    return {"Model": model_name, "RMSE": round(rmse, 2), "R2": round(r2, 4)}


# ---------------------------------------------------------------------------
# Classification models (Claim Probability)
# ---------------------------------------------------------------------------


def train_logistic_regression(
    X_train: np.ndarray, y_train: np.ndarray
) -> Pipeline:
    """Train a Logistic Regression classifier with standard scaling."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=500, random_state=42)),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def train_random_forest_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 200,
    max_depth: int = 12,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Train a Random Forest Classifier."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def train_xgb_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    random_state: int = 42,
) -> Any:
    """Train an XGBoost Classifier."""
    if not XGB_AVAILABLE:
        raise ImportError("XGBoost is not installed.")
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
        use_label_encoder=False,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train, verbose=False)
    return model


def evaluate_classification(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
) -> Dict[str, Any]:
    """
    Evaluate a classification model with accuracy, precision, recall, F1, AUC.

    Returns
    -------
    dict with evaluation metrics.
    """
    y_pred = model.predict(X_test)
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    except Exception:
        auc = None

    metrics = {
        "Model": model_name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "AUC": round(auc, 4) if auc is not None else "N/A",
    }
    logger.info(f"{model_name} – " + " | ".join(f"{k}: {v}" for k, v in metrics.items() if k != "Model"))
    return metrics


# ---------------------------------------------------------------------------
# Feature importance & SHAP
# ---------------------------------------------------------------------------


def get_feature_importance(
    model: Any,
    feature_names: list[str],
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Extract feature importances from tree-based models (RF, XGB).

    Parameters
    ----------
    model : fitted sklearn/xgboost model
    feature_names : list of str
    top_n : int

    Returns
    -------
    pd.DataFrame with columns ['Feature', 'Importance']
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "named_steps"):
        inner = model.named_steps.get("model")
        if inner and hasattr(inner, "coef_"):
            importances = np.abs(inner.coef_).flatten()
        else:
            raise ValueError("Model does not expose feature importances.")
    else:
        raise ValueError("Cannot extract feature importances from this model type.")

    df_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    return df_imp.sort_values("Importance", ascending=False).head(top_n).reset_index(drop=True)


def compute_shap_values(
    model: Any,
    X: np.ndarray,
    feature_names: list[str],
    max_samples: int = 500,
) -> Tuple[Any, np.ndarray]:
    """
    Compute SHAP values for a fitted model.

    Parameters
    ----------
    model : fitted model (RF or XGB)
    X : np.ndarray
        Feature matrix (preferably X_test).
    feature_names : list of str
    max_samples : int
        Maximum number of samples to compute SHAP for (for speed).

    Returns
    -------
    (explainer, shap_values_array)
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP library is not installed.")

    X_sample = X[:max_samples]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    return explainer, shap_values


# ---------------------------------------------------------------------------
# Risk-based pricing formula
# ---------------------------------------------------------------------------


def compute_risk_premium(
    p_claim: np.ndarray,
    predicted_severity: np.ndarray,
    expense_loading: float = 0.15,
    profit_margin: float = 0.10,
) -> np.ndarray:
    """
    Compute a risk-based premium using:
      Premium = (P(claim) × Predicted Severity) + Expense Loading + Profit Margin

    Parameters
    ----------
    p_claim : np.ndarray
        Probability of a claim for each policy.
    predicted_severity : np.ndarray
        Predicted claim severity for each policy.
    expense_loading : float
        Fraction of pure premium added for expenses (default 15%).
    profit_margin : float
        Fraction of pure premium added as profit (default 10%).

    Returns
    -------
    np.ndarray of risk-based premiums.
    """
    pure_premium = p_claim * predicted_severity
    loading = pure_premium * (expense_loading + profit_margin)
    return pure_premium + loading
