# AlphaCare Insurance Solutions – Risk Analytics & Predictive Modeling

[![CI](https://github.com/YOUR_USERNAME/insurance-risk-analytics/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/insurance-risk-analytics/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![DVC](https://img.shields.io/badge/data--version--control-DVC-945DD6.svg)](https://dvc.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

End-to-end insurance risk analytics project for **AlphaCare Insurance Solutions (ACIS)**, analysing 18 months of historical car-insurance claim data (Feb 2014 – Aug 2015) to:

- Identify low-risk policy segments where premiums can be competitively reduced.
- Statistically validate key risk hypotheses across provinces, postal codes, and demographics.
- Build predictive models for claim severity and claim probability to power dynamic, risk-based pricing.

---

## Project Structure

```
insurance-risk-analytics/
├── .github/workflows/ci.yml   # GitHub Actions CI (lint + test)
├── data/                      # Tracked by DVC, NOT committed to Git
│   ├── raw/                   # Raw pipe-delimited .txt data
│   └── processed/             # Cleaned CSV ready for analysis
├── notebooks/
│   ├── 01_eda.ipynb           # Task 1 – Exploratory Data Analysis
│   ├── 02_hypothesis_testing.ipynb   # Task 3 – A/B Hypothesis Testing
│   └── 03_modeling.ipynb      # Task 4 – Predictive Modeling
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # Data ingestion & cleaning helpers
│   ├── eda_utils.py           # EDA plotting & summarisation helpers
│   ├── hypothesis_tests.py    # Statistical test functions
│   └── modeling.py            # Model training, evaluation & SHAP helpers
├── reports/
│   └── final_report.md        # Business-facing final report
├── tests/
│   ├── test_data_loader.py
│   ├── test_eda_utils.py
│   └── test_hypothesis_tests.py
├── .dvc/                      # DVC internal config
├── dvc.yaml                   # DVC pipeline definition
├── .gitignore
├── .dvcignore
├── requirements.txt
└── README.md
```

---

## Derived Metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| **Loss Ratio** | `TotalClaims / TotalPremium` | Core portfolio profitability measure |
| **Margin** | `TotalPremium − TotalClaims` | Per-policy profit contribution |
| **Claim Frequency** | `policies with ≥1 claim / total policies` | Risk frequency measure |
| **Claim Severity** | `avg TotalClaims given claim > 0` | Risk magnitude measure |

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- Git
- DVC (`pip install dvc`)

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/insurance-risk-analytics.git
cd insurance-risk-analytics
```

### 2. Create and activate a virtual environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Pull the data via DVC
```bash
dvc pull
```

This will download the raw and processed datasets from the configured DVC remote.

---

## Reproducing the Data Pipeline (Task 2 – DVC)

The project uses DVC to version-control data and ensure reproducibility.

### Remote storage
A **local DVC remote** is configured at `../alpha_dvc_remote` (relative to the project root). To switch to a different remote (e.g., S3, GCS), update `.dvc/config`.

### Pipeline stages

```yaml
# dvc.yaml
stages:
  prepare:
    cmd: python src/data_loader.py
    deps:
      - data/raw/MachineLearningRating_v3.txt
      - src/data_loader.py
    outs:
      - data/processed/insurance_cleaned.csv
```

Run the full pipeline:
```bash
dvc repro
```

Push data to the remote:
```bash
dvc push
```

Pull data from the remote (for new contributors):
```bash
dvc pull
```

---

## Running Notebooks

Start JupyterLab:
```bash
jupyter lab
```

Then open notebooks in order:
1. `notebooks/01_eda.ipynb` – EDA & data quality
2. `notebooks/02_hypothesis_testing.ipynb` – A/B hypothesis tests
3. `notebooks/03_modeling.ipynb` – Predictive modeling & SHAP

---

## Running Tests & Linting

```bash
# Run all tests
pytest tests/ -v

# Run linter
flake8 src/ tests/ --max-line-length=120
```

Tests are automatically run on every push via GitHub Actions CI.

---

## Key Findings Summary

See [`reports/final_report.md`](reports/final_report.md) for the full business-facing report.

---

## Tasks Completed

| Task | Branch | Status | Description |
|------|--------|--------|-------------|
| Task 1 | `task-1` | ✅ Done | EDA, data quality, visualisations |
| Task 2 | `task-2` | ✅ Done | DVC data versioning & pipeline |
| Task 3 | `task-3` | 🔄 Planned | A/B hypothesis testing |
| Task 4 | `task-4` | 🔄 Planned | Predictive modeling & pricing |

---

## License

MIT License – see [LICENSE](LICENSE) for details.
