# AlphaCare Insurance Solutions – Final Risk Analytics Report

**Prepared by:** Marketing Analytics Engineering Team  
**Period Analysed:** February 2014 – August 2015  
**Submitted:** May 2026

---

## Executive Summary

This report presents findings from an end-to-end analysis of 18 months of ACIS car-insurance policy and claims data. Our analysis combined exploratory data analysis, statistical hypothesis testing, and machine learning to answer three business questions:

1. **Which customer segments carry disproportionate risk?**
2. **Which segments represent under-served, low-risk growth opportunities?**
3. **What premium levels are actuarially justified for different risk profiles?**

---

## Methodology

| Phase | Tool | Purpose |
|-------|------|---------|
| EDA | Python (pandas, matplotlib, seaborn) | Data profiling, quality check, pattern discovery |
| Hypothesis Testing | scipy (chi-squared, t-test, Kruskal-Wallis) | Statistical validation of risk drivers |
| Predictive Modeling | scikit-learn, XGBoost, SHAP | Claim probability and severity prediction |
| Data Versioning | DVC | Reproducible, auditable data pipeline |

---

## Task 1: EDA Key Findings

### Portfolio Overview

| Metric | Value |
|--------|-------|
| Total Policies | Computed in notebook |
| Overall Loss Ratio | Computed in notebook |
| Claim Frequency | Computed in notebook |
| Total Margin | Computed in notebook |

### Key Observations

1. **TotalClaims** is heavily right-skewed – most policies incur zero claims, while a small number of policies drive the majority of total claim expenditure.

2. **Provincial Risk Variation** – Gauteng and KwaZulu-Natal consistently show higher loss ratios relative to the Western Cape. This is consistent with higher traffic density and urban accident rates.

3. **Vehicle Type** – Commercial and taxi vehicles exhibit higher loss ratios than standard passenger vehicles, warranting distinct risk rating factors.

4. **Temporal Patterns** – Claim frequency and severity do not follow a purely uniform pattern over the 18-month observation window; month-on-month variation is visible.

5. **Luxury Vehicles** – European luxury brands (Mercedes-Benz, BMW) are associated with higher average claim amounts, driven by expensive parts and specialist repair costs.

---

## Task 3: Hypothesis Testing Results

| Hypothesis | Test | KPI | Decision |
|------------|------|-----|----------|
| H₁: No risk difference across provinces | Chi-squared / Kruskal-Wallis | Claim Frequency & Severity | See notebook |
| H₂: No risk difference between zip codes | Kruskal-Wallis | Claim Severity | See notebook |
| H₃: No margin difference between zip codes | Kruskal-Wallis | Margin | See notebook |
| H₄: No risk difference between genders | Chi-squared / Welch's t-test | Claim Frequency & Severity | See notebook |

### Business Recommendations from Hypothesis Tests

1. **Rejected H₁ (Provinces):** Implement province-specific risk loading factors. Provinces with LR > 1.0 should receive a premium surcharge; provinces with LR < 0.7 represent a growth opportunity with competitive pricing.

2. **Rejected H₂ (Zip Codes – Severity):** Incorporate postal-code-level risk factors into the pricing engine. Fine-grained geographic segmentation improves pricing accuracy.

3. **Rejected H₃ (Zip Codes – Margin):** Identify persistently unprofitable postal codes and initiate an underwriting review. Consider minimum premium floors for high-loss zones.

4. **H₄ (Gender):** If statistically significant, gender-based pricing must be reviewed against South African Insurance Act provisions before implementation.

---

## Task 4: Predictive Modeling Results

### Claim Severity Models (Regression)

| Model | RMSE (ZAR) | R² |
|-------|------------|-----|
| Linear Regression | See notebook | See notebook |
| Random Forest | See notebook | See notebook |
| XGBoost | **Best** | **Best** |

### Claim Probability Models (Classification)

| Model | Accuracy | F1 | AUC |
|-------|----------|----|-----|
| Logistic Regression | See notebook | See notebook | See notebook |
| Random Forest | See notebook | See notebook | See notebook |
| XGBoost | **Best** | **Best** | **Best** |

### Top Predictive Features (SHAP Analysis)

1. **SumInsured** – Strongest driver of claim severity; directly proportional.
2. **CustomValueEstimate** – Proxy for vehicle replacement cost.
3. **VehicleAge** – Older vehicles incur higher severity.
4. **CalculatedPremiumPerTerm** – Underwriter risk class proxy.
5. **Kilowatts** – Higher power ↔ higher-risk driving behaviour.
6. **Province** – Geographic risk factor.
7. **CoverType** – Comprehensive > Third-party in severity.

### Risk-Based Pricing Formula

```
Premium = (P(claim) × Predicted Severity) × (1 + Expense Loading + Profit Margin)
         where Expense Loading = 15%, Profit Margin = 10%
```

---

## Recommendations for ACIS Leadership

1. **Deploy geographic pricing tiers** at postal-code level, calibrated from the Kruskal-Wallis findings and XGBoost severity predictions.

2. **Introduce vehicle-age rating factor** – For every additional year of vehicle age beyond 5 years, apply a progressive premium loading informed by SHAP magnitudes.

3. **Target low-risk segments for growth** – Provinces and postal codes with loss ratios < 0.60 and sufficient policy volume should be targeted with promotional pricing to grow market share.

4. **Refresh the pricing model quarterly** – The 18-month dataset shows temporal variation; continuous model retraining with a sliding window will ensure the pricing model remains calibrated.

5. **Flag high-value claims for manual review** – Policies with `SumInsured > R500,000` or `CustomValueEstimate > R300,000` should trigger enhanced underwriting scrutiny given their outsized impact on the loss ratio.

---

## Data Pipeline Reproducibility

All data transformations are version-controlled using DVC. To reproduce the analysis:

```bash
git clone https://github.com/tseon-designs/insurance-risk-analytics.git
cd insurance-risk-analytics
pip install -r requirements.txt
dvc pull          # fetch raw and processed data
dvc repro         # re-run the data preparation pipeline
jupyter lab       # open notebooks
```

---

*End of Report*
