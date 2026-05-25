"""
src/eda_utils.py
================
Reusable EDA plotting and summarisation utilities for the
AlphaCare Insurance Risk Analytics project.
"""

from __future__ import annotations

import warnings
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

PALETTE = "viridis"
ACCENT = "#4C72B0"
BACKGROUND = "#F8F9FA"
FIG_DPI = 120


def set_style() -> None:
    """Apply a consistent, publication-ready Matplotlib style."""
    plt.rcParams.update(
        {
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": FIG_DPI,
        }
    )


set_style()


# ---------------------------------------------------------------------------
# Descriptive summaries
# ---------------------------------------------------------------------------


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extended descriptive statistics for all numeric columns.

    Returns
    -------
    pd.DataFrame
        Includes count, mean, std, min, quartiles, max, skewness, kurtosis.
    """
    num_df = df.select_dtypes(include="number")
    summary = num_df.describe().T
    summary["skewness"] = num_df.skew()
    summary["kurtosis"] = num_df.kurtosis()
    summary["missing_pct"] = (df.isnull().sum()[summary.index] / len(df) * 100).round(2)
    return summary.round(3)


def categorical_summary(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Summary of categorical columns: unique count, top value, top frequency %.

    Parameters
    ----------
    df : pd.DataFrame
    top_n : int
        Number of top categories to consider.

    Returns
    -------
    pd.DataFrame
    """
    cat_cols = df.select_dtypes(include=["category", "object"]).columns
    records = []
    for col in cat_cols:
        vc = df[col].value_counts(dropna=True)
        records.append(
            {
                "column": col,
                "n_unique": df[col].nunique(),
                "top_value": vc.index[0] if len(vc) > 0 else None,
                "top_freq_pct": round(vc.iloc[0] / len(df) * 100, 2) if len(vc) > 0 else 0,
                "missing_pct": round(df[col].isnull().mean() * 100, 2),
            }
        )
    return pd.DataFrame(records).set_index("column")


# ---------------------------------------------------------------------------
# Univariate analysis
# ---------------------------------------------------------------------------


def plot_numeric_distributions(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    ncols: int = 3,
    figsize_per_cell: tuple = (5, 3.5),
) -> plt.Figure:
    """
    Plot histograms + KDE for numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
    cols : list of str, optional
        Columns to plot. Defaults to all numeric columns.
    ncols : int
        Number of subplot columns.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if cols is None:
        cols = df.select_dtypes(include="number").columns.tolist()

    nrows = (len(cols) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_cell[0] * ncols, figsize_per_cell[1] * nrows),
    )
    axes = np.array(axes).flatten()

    for ax, col in zip(axes, cols):
        data = df[col].dropna()
        ax.hist(data, bins=40, color=ACCENT, alpha=0.75, edgecolor="white")
        ax2 = ax.twinx()
        data.plot.kde(ax=ax2, color="crimson", linewidth=1.5)
        ax2.set_ylabel("")
        ax2.set_yticks([])
        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel("Count")

    # Hide unused axes
    for ax in axes[len(cols):]:
        ax.set_visible(False)

    fig.suptitle("Distribution of Numeric Features", fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def plot_categorical_distributions(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    top_n: int = 10,
    ncols: int = 2,
    figsize_per_cell: tuple = (7, 3.5),
) -> plt.Figure:
    """
    Plot bar charts for categorical columns (top N categories).

    Parameters
    ----------
    df : pd.DataFrame
    cols : list of str, optional
    top_n : int
        Number of top categories to show.
    ncols : int

    Returns
    -------
    matplotlib.figure.Figure
    """
    if cols is None:
        cols = df.select_dtypes(include=["category", "object"]).columns.tolist()

    nrows = (len(cols) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_cell[0] * ncols, figsize_per_cell[1] * nrows),
    )
    axes = np.array(axes).flatten()

    cmap = plt.colormaps.get_cmap(PALETTE)
    for ax, col in zip(axes, cols):
        vc = df[col].value_counts(dropna=True).head(top_n)
        colors = [cmap(i / len(vc)) for i in range(len(vc))]
        bars = ax.barh(vc.index.astype(str)[::-1], vc.values[::-1], color=colors[::-1])
        ax.set_title(col)
        ax.set_xlabel("Count")
        for bar, val in zip(bars, vc.values[::-1]):
            ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:,}", va="center", fontsize=8)

    for ax in axes[len(cols):]:
        ax.set_visible(False)

    fig.suptitle("Categorical Feature Distributions (Top N)", fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Bivariate & multivariate analysis
# ---------------------------------------------------------------------------


def plot_premium_vs_claims(df: pd.DataFrame, hue_col: str = "Province") -> plt.Figure:
    """
    Scatter plot of TotalPremium vs TotalClaims coloured by a categorical variable.

    Parameters
    ----------
    df : pd.DataFrame
    hue_col : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    subset = df[(df["TotalPremium"] > 0) & (df["TotalClaims"] >= 0)].copy()
    categories = subset[hue_col].dropna().unique()

    cmap = plt.colormaps.get_cmap("tab20")
    for i, cat in enumerate(categories[:20]):
        mask = subset[hue_col] == cat
        ax.scatter(
            subset.loc[mask, "TotalPremium"],
            subset.loc[mask, "TotalClaims"],
            label=str(cat),
            alpha=0.4,
            s=10,
            color=cmap(i / 20),
        )

    # 45-degree reference line (break-even: LossRatio = 1)
    max_val = max(subset["TotalPremium"].quantile(0.99), subset["TotalClaims"].quantile(0.99))
    ax.plot([0, max_val], [0, max_val], "r--", linewidth=1.5, label="Loss Ratio = 1 (break-even)")
    ax.set_xlim(0, subset["TotalPremium"].quantile(0.99))
    ax.set_ylim(0, subset["TotalClaims"].quantile(0.99))
    ax.set_xlabel("Total Premium (ZAR)")
    ax.set_ylabel("Total Claims (ZAR)")
    ax.set_title(f"Total Premium vs Total Claims – coloured by {hue_col}")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def plot_correlation_matrix(df: pd.DataFrame, cols: Optional[List[str]] = None) -> plt.Figure:
    """
    Heatmap of the Pearson correlation matrix for numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
    cols : list of str, optional
        Subset of columns to include.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if cols is None:
        cols = df.select_dtypes(include="number").columns.tolist()

    corr = df[cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(max(8, len(cols)), max(6, len(cols) - 1)))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        ax=ax,
        annot_kws={"size": 9},
        vmin=-1,
        vmax=1,
    )
    ax.set_title("Pearson Correlation Matrix", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Geographic analysis
# ---------------------------------------------------------------------------


def plot_loss_ratio_by_province(df: pd.DataFrame) -> plt.Figure:
    """
    Bar chart of mean Loss Ratio by Province, sorted descending.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    matplotlib.figure.Figure
    """
    prov = (
        df.groupby("Province", observed=True)["LossRatio"]
        .agg(["mean", "median", "count"])
        .rename(columns={"mean": "Mean LR", "median": "Median LR", "count": "Policies"})
        .sort_values("Mean LR", ascending=True)
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.colormaps.get_cmap("RdYlGn_r")(
        np.linspace(0.15, 0.85, len(prov))
    )
    bars = ax.barh(prov["Province"], prov["Mean LR"], color=colors)
    ax.axvline(1.0, color="crimson", linestyle="--", linewidth=1.5, label="Break-even (LR=1)")
    ax.axvline(prov["Mean LR"].mean(), color="steelblue", linestyle=":", linewidth=1.5,
               label=f"Portfolio avg ({prov['Mean LR'].mean():.2f})")

    for bar, val, cnt in zip(bars, prov["Mean LR"], prov["Policies"]):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}  (n={cnt:,})", va="center", fontsize=9)

    ax.set_xlabel("Mean Loss Ratio")
    ax.set_title("Mean Loss Ratio by Province", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(0, prov["Mean LR"].max() * 1.35)
    fig.tight_layout()
    return fig


def plot_geographic_premium_heatmap(df: pd.DataFrame) -> plt.Figure:
    """
    Grouped bar chart: mean TotalPremium, mean TotalClaims, mean Margin by Province.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    matplotlib.figure.Figure
    """
    prov = (
        df.groupby("Province", observed=True)[["TotalPremium", "TotalClaims", "Margin"]]
        .mean()
        .sort_values("TotalPremium", ascending=False)
    )

    x = np.arange(len(prov))
    width = 0.28
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width, prov["TotalPremium"], width, label="Avg Premium", color="#2196F3", alpha=0.85)
    ax.bar(x, prov["TotalClaims"], width, label="Avg Claims", color="#F44336", alpha=0.85)
    ax.bar(x + width, prov["Margin"], width, label="Avg Margin", color="#4CAF50", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(prov.index, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("ZAR")
    ax.set_title("Average Premium, Claims & Margin by Province", fontsize=14, fontweight="bold")
    ax.legend()
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"R{v:,.0f}"))
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------


def plot_boxplots(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    ncols: int = 3,
) -> plt.Figure:
    """
    Box plots for detecting outliers in numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
    cols : list of str, optional
    ncols : int

    Returns
    -------
    matplotlib.figure.Figure
    """
    if cols is None:
        cols = ["TotalPremium", "TotalClaims", "SumInsured", "CustomValueEstimate",
                "CalculatedPremiumPerTerm", "LossRatio", "Margin"]
        cols = [c for c in cols if c in df.columns]

    nrows = (len(cols) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for ax, col in zip(axes, cols):
        data = df[col].dropna()
        bp = ax.boxplot(
            data,
            vert=True,
            patch_artist=True,
            boxprops=dict(facecolor=ACCENT, alpha=0.6),
            medianprops=dict(color="crimson", linewidth=2),
            flierprops=dict(marker=".", markersize=3, alpha=0.3, color="gray"),
        )
        ax.set_title(col)
        ax.set_ylabel("Value")
        q1, q3 = data.quantile(0.25), data.quantile(0.75)
        iqr = q3 - q1
        n_outliers = ((data < q1 - 1.5 * iqr) | (data > q3 + 1.5 * iqr)).sum()
        ax.text(0.98, 0.98, f"Outliers: {n_outliers:,}\n({n_outliers/len(data)*100:.1f}%)",
                transform=ax.transAxes, ha="right", va="top", fontsize=8, color="gray")

    for ax in axes[len(cols):]:
        ax.set_visible(False)

    fig.suptitle("Outlier Detection – Box Plots", fontsize=15, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Creative/insight plots
# ---------------------------------------------------------------------------


def plot_loss_ratio_by_vehicle_type(df: pd.DataFrame) -> plt.Figure:
    """
    Violin + swarm plot of Loss Ratio distribution by VehicleType.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    matplotlib.figure.Figure
    """
    plot_df = df[df["LossRatio"].between(0, 5)].copy()
    fig, ax = plt.subplots(figsize=(12, 6))
    order = (
        plot_df.groupby("VehicleType", observed=True)["LossRatio"]
        .median()
        .sort_values(ascending=False)
        .index
    )
    sns.violinplot(
        data=plot_df,
        x="VehicleType",
        y="LossRatio",
        order=order,
        palette="Set2",
        inner="quartile",
        ax=ax,
        cut=0,
    )
    ax.axhline(1.0, color="crimson", linestyle="--", linewidth=1.5, label="Break-even (LR=1)")
    ax.set_xlabel("Vehicle Type")
    ax.set_ylabel("Loss Ratio")
    ax.set_title("Loss Ratio Distribution by Vehicle Type", fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_monthly_claim_trends(df: pd.DataFrame) -> plt.Figure:
    """
    Dual-axis time-series: monthly claim frequency vs average claim severity.

    Parameters
    ----------
    df : pd.DataFrame (must have 'TransactionMonth', 'HasClaim', 'TotalClaims').

    Returns
    -------
    matplotlib.figure.Figure
    """
    monthly = (
        df.groupby("TransactionMonth")
        .agg(
            ClaimFrequency=("HasClaim", "mean"),
            AvgClaimSeverity=("TotalClaims", lambda x: x[x > 0].mean()),
            PolicyCount=("PolicyID", "count"),
        )
        .reset_index()
        .sort_values("TransactionMonth")
    )

    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax1.fill_between(monthly["TransactionMonth"], monthly["ClaimFrequency"] * 100,
                     alpha=0.3, color="#2196F3")
    ax1.plot(monthly["TransactionMonth"], monthly["ClaimFrequency"] * 100,
             color="#2196F3", linewidth=2, label="Claim Frequency (%)")
    ax1.set_ylabel("Claim Frequency (%)", color="#2196F3")
    ax1.tick_params(axis="y", labelcolor="#2196F3")

    ax2 = ax1.twinx()
    ax2.plot(monthly["TransactionMonth"], monthly["AvgClaimSeverity"],
             color="#F44336", linewidth=2, marker="o", markersize=4, label="Avg Claim Severity (ZAR)")
    ax2.set_ylabel("Avg Claim Severity (ZAR)", color="#F44336")
    ax2.tick_params(axis="y", labelcolor="#F44336")
    ax2.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"R{v:,.0f}"))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)
    ax1.set_title("Monthly Claim Frequency & Average Claim Severity (2014–2015)",
                  fontsize=14, fontweight="bold")
    ax1.set_xlabel("Transaction Month")
    fig.tight_layout()
    return fig


def plot_top_makes_by_claims(df: pd.DataFrame, top_n: int = 15) -> plt.Figure:
    """
    Horizontal bar chart of top N vehicle makes by average TotalClaims.

    Parameters
    ----------
    df : pd.DataFrame
    top_n : int

    Returns
    -------
    matplotlib.figure.Figure
    """
    make_claims = (
        df[df["TotalClaims"] > 0]
        .groupby("make", observed=True)["TotalClaims"]
        .agg(["mean", "count"])
        .query("count >= 50")
        .sort_values("mean", ascending=True)
        .tail(top_n)
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.colormaps.get_cmap("YlOrRd")
    colors = [cmap(i / len(make_claims)) for i in range(len(make_claims))]
    bars = ax.barh(make_claims["make"], make_claims["mean"], color=colors)

    for bar, val, cnt in zip(bars, make_claims["mean"], make_claims["count"]):
        ax.text(bar.get_width() + 200, bar.get_y() + bar.get_height() / 2,
                f"R{val:,.0f}  (n={cnt:,})", va="center", fontsize=8)

    ax.set_xlabel("Average Claim Amount (ZAR)")
    ax.set_title(f"Top {top_n} Vehicle Makes by Average Claim Amount\n(min. 50 claims)",
                 fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"R{v:,.0f}"))
    ax.set_xlim(0, make_claims["mean"].max() * 1.4)
    fig.tight_layout()
    return fig


def plot_gender_loss_ratio(df: pd.DataFrame) -> plt.Figure:
    """
    Side-by-side box plot of Loss Ratio by Gender.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    matplotlib.figure.Figure
    """
    plot_df = df[df["Gender"].isin(["Male", "Female"]) & df["LossRatio"].between(0, 5)].copy()
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=plot_df, x="Gender", y="LossRatio", palette=["#2196F3", "#E91E63"],
                width=0.4, flierprops=dict(marker=".", alpha=0.3, markersize=3), ax=ax)
    ax.axhline(1.0, color="crimson", linestyle="--", linewidth=1.5, label="Break-even (LR=1)")
    ax.set_title("Loss Ratio Distribution by Gender", fontsize=13, fontweight="bold")
    ax.set_ylabel("Loss Ratio")
    ax.legend()
    fig.tight_layout()
    return fig
