# AlphaCare Insurance Solutions (ACIS) 
## Comprehensive Risk Analytics & Predictive Modeling Report
**Prepared By:** Marketing Analytics Engineering Team
**Date:** May 2026

---

## 1. Executive Summary

This comprehensive report details the findings, statistical testing, and predictive modeling efforts applied to AlphaCare Insurance Solutions' historical auto-insurance claims dataset (Feb 2014 – Aug 2015). In preparation for an aggressive growth phase in the South African auto-insurance market, ACIS commissioned a deep dive into the underlying factors driving claim frequency, severity, and overall profitability.

### 1.1 Core Business Problem
ACIS currently faces an **unprofitable overall loss ratio of 1.0477**, indicating that for every R1 collected in premiums, approximately R1.05 is paid out in claims. Over the 18-month period analyzed:
- **Total Policies:** 1,000,098
- **Total Premium Collected:** R61,911,562.70
- **Total Claims Paid:** R64,867,546.17
- **Net Margin:** R-2,955,983.47
- **Claim Frequency:** 0.28% (2,788 claims)

While the frequency of claims is relatively low, the high severity of individual claims is completely eroding the premium base.

### 1.2 Strategic Objectives
1. **Optimize Marketing Strategy:** Identify low-risk targets for premium discounting and aggressive customer acquisition.
2. **Refine Pricing Models:** Transition from intuition-based pricing to a statistically rigorous, machine-learning-driven risk premium calculator.
3. **Geographic Targeting:** Isolate profitable regions and postal codes to deploy targeted marketing spend.

### 1.3 Key Findings
- **Geographic Location is the Strongest Predictor:** Claim severity, claim frequency, and total margin vary dramatically and significantly across provinces and postal codes. (p < 0.0001).
- **Gender is NOT a Predictor:** Despite conventional assumptions, there are zero statistically significant differences in claim behavior between male and female drivers.
- **Machine Learning Pricing is Viable:** We successfully developed a dual-model predictive architecture (XGBoost/Random Forest) capable of scoring any new policyholder for both *probability of claiming* and *expected severity*.

---

## 2. Exploratory Data Analysis (EDA)

The dataset encompasses over 1 million records representing 18 months of auto insurance policies. We systematically handled missing values, corrected data types, and engineered new features (such as `VehicleAge` and `Margin`) prior to analysis.

### 2.1 Portfolio Health and Key Performance Indicators
The portfolio's fundamental metrics immediately highlight the pricing mismatch:
- **Total Portfolio Size:** 1M+ policies provides high statistical confidence.
- **Loss Ratio:** The 104.7% loss ratio requires urgent remediation. A sustainable target is typically 65-75%.

### 2.2 Geographic Distribution
Our spatial analysis revealed stark differences between provinces:
- Certain provinces exhibit extreme volatility in claim severity.
- Postal codes (Zip Codes) show an even higher variance, indicating that risk is highly localized. High-density urban postal codes generally exhibit higher frequency but varying severity compared to rural postal codes.

### 2.3 Feature Distributions
- **Vehicle Age:** Older vehicles show a different risk profile compared to newer vehicles, though newer vehicles often lead to higher claim severity due to replacement part costs.
- **Engine Capacity & Cylinders:** High-performance vehicles strongly correlate with elevated risk severity.

---

## 3. Statistical Hypothesis Testing

To ensure that pricing and marketing strategies are built on mathematical facts rather than assumptions, we conducted rigorous A/B and hypothesis testing across four primary business questions.

### Hypothesis 1: Risk Differences Across Provinces
- **Objective:** Determine if there are significant risk differences across provinces.
- **Claim Frequency Test:** Chi-squared Test (Statistic: 104.19, **p-value: 0.0000**)
- **Claim Severity Test:** Kruskal-Wallis Test (Statistic: 105.75, **p-value: 0.0000**)
- **Decision:** **Reject the Null Hypothesis.**
- **Interpretation:** Provinces show highly statistically significant differences in both the likelihood of claims and the cost of those claims. We *must* implement targeted provincial pricing. Marketing should immediately pivot to acquiring customers in the historically lowest-severity provinces.

### Hypothesis 2: Risk Differences Between Postal Codes
- **Objective:** Determine if risk is localized at the postal code level.
- **Test Utilized:** Kruskal-Wallis Test
- **Statistic:** 234.90
- **p-value:** **0.0000**
- **Decision:** **Reject the Null Hypothesis.**
- **Interpretation:** Risk is not just provincial; it is hyper-localized. Fine-grained geographic pricing at the zip-code level is strongly recommended. 

### Hypothesis 3: Profit Margin Differences Between Postal Codes
- **Objective:** Determine if some zip codes are systematically more profitable.
- **Test Utilized:** Kruskal-Wallis Test
- **Statistic:** 88,471.56
- **p-value:** **0.0000**
- **Decision:** **Reject the Null Hypothesis.**
- **Interpretation:** The current pricing strategy fails to account for local risk, leading to massive variations in profitability (Margin = Premium - Claims). Certain postal codes are subsidizing others.

### Hypothesis 4: Risk Differences Between Genders
- **Objective:** Determine if gender is a valid pricing factor.
- **Claim Frequency (Male vs Female):** Chi-squared Test (Statistic: 0.0037, **p-value: 0.951**)
- **Claim Severity (Male vs Female):** Welch's t-test (Statistic: -0.579, **p-value: 0.568**)
- **Decision:** **Fail to Reject the Null Hypothesis.**
- **Interpretation:** There is **no statistically significant difference** in risk between men and women. Any pricing model or marketing strategy that discriminates based on gender is mathematically flawed and potentially legally problematic. Gender should be dropped as a primary risk rating factor.

---

## 4. Predictive Modeling Methodology

To transition ACIS from reactive analysis to proactive risk management, we engineered a dual-stage Machine Learning pipeline. Since only 0.28% of policies experience a claim, predicting a single "Total Claim Amount" directly on the full dataset is highly inefficient due to zero-inflation.

### 4.1 The Dual-Model Architecture
Instead of a single model, we implemented a Two-Stage (Frequency-Severity) approach:
1. **Classification Model (Frequency):** Predicts the *probability* that a policyholder will file a claim ($P(Claim)$).
2. **Regression Model (Severity):** Predicts the *expected cost* of the claim, assuming a claim actually occurs.
3. **Risk Premium Calculation:** Expected Loss = $P(Claim) \times PredictedSeverity$.

### 4.2 Data Preparation and Encoding
Categorical variables (e.g., `Province`, `VehicleType`, `Make`) were encoded using Target Encoding to preserve their relationship with the risk variables. Missing values were median/mode imputed, and the dataset was standardized. For the classification model, Synthetic Minority Over-sampling Technique (SMOTE) was evaluated to handle the extreme 99.7% class imbalance.

---

## 5. Model Evaluation and Results

We evaluated three advanced algorithms for both stages: Linear/Logistic Regression, Random Forest, and XGBoost.

### 5.1 Stage 1: Classification (Claim Probability)
The classification model was evaluated primarily on **Recall** (ability to identify actual claims) and **AUC** (Area Under the ROC Curve).

| Model | Accuracy | Precision | Recall | F1 Score | AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.9971 | 0.0000 | 0.0000 | 0.0000 | 0.7566 |
| Random Forest | 0.8040 | 0.0131 | **0.8950** | 0.0258 | 0.9030 |
| XGBoost | 0.8151 | 0.0136 | **0.8761** | 0.0268 | **0.9053** |

**Conclusion:** XGBoost and Random Forest vastly outperform Logistic Regression. While Logistic Regression achieved 99.7% accuracy by simply predicting "No Claim" for everyone, it failed entirely at identifying risk (Recall = 0.0). Both Random Forest and XGBoost achieved ~88-89% Recall with an AUC > 0.90, meaning they successfully identified 9 out of 10 future claimants. We selected **XGBoost** for its superior probability calibration.

### 5.2 Stage 2: Regression (Claim Severity)
The regression models were trained strictly on the subset of data where `TotalClaims > 0` to learn the drivers of cost.

| Model | RMSE (ZAR) | R² Score |
| :--- | :--- | :--- |
| Linear Regression | 34,305.46 | 0.2682 |
| Random Forest | **33,522.43** | **0.3013** |
| XGBoost | 36,498.87 | 0.1717 |

**Conclusion:** The **Random Forest Regressor** achieved the lowest Root Mean Squared Error (R 33,522) and the highest explained variance (R² = 30.13%). Predicting exact claim costs is notoriously difficult due to extreme outliers (total losses vs. minor dents), but the Random Forest model captures the underlying patterns best.

---

## 6. Feature Importance and SHAP Interpretability

Machine Learning models in insurance cannot be "black boxes." Regulators and underwriters require transparency. We utilized SHAP (SHapley Additive exPlanations) to interpret the XGBoost probability model.

### Top Drivers of Risk
1. **PostalCode / Province:** By far the most predictive feature. Geographic location dictates traffic density, theft rates, and road quality.
2. **Vehicle Age:** The age of the vehicle has a non-linear relationship with risk.
3. **Vehicle Make & Model:** Certain manufacturers correlate highly with specific driver demographics and repair costs.
4. **Cylinders & Cubic Capacity:** Higher-performance vehicles consistently show a higher probability of severe claims.
5. **Gender:** Confirmed by our SHAP analysis and hypothesis testing, gender contributed almost zero predictive power to the models.

---

## 7. Dynamic Risk-Based Pricing Engine

Using the selected models (XGBoost for Probability, Random Forest for Severity), we constructed a dynamic pricing engine capable of calculating a mathematically fair risk premium for every policy.

### Pricing Formula
$$ Pure Premium = P(Claim) \times PredictedSeverity $$
$$ Commercial Premium = Pure Premium \times (1 + ExpenseLoading + ProfitMargin) $$

*(Assuming an expense loading of 15% and a target profit margin of 10%)*

### Pricing Results on Holdout Set
When scoring the portfolio using the new ML pricing engine, we observed the following distribution:
- **Mean Recommended Premium:** R 3,341.07
- **Median Recommended Premium:** R 1.03
- **Maximum Recommended Premium:** R 387,312.44

The stark difference between the Mean and Median premium proves that the vast majority of drivers are extremely low risk and should be charged very low premiums (to gain market share), while a small segment of ultra-high-risk drivers requires enormous premiums to offset their expected losses. 

By charging a flat or lightly-adjusted premium across the board historically, ACIS was dramatically overcharging safe drivers (causing them to leave for competitors) and vastly undercharging dangerous drivers (attracting them, leading to the 1.04 loss ratio).

---

## 8. Strategic Business Recommendations

Based on the culmination of the data analysis, statistical testing, and predictive modeling, we submit the following actionable recommendations to the ACIS executive board:

### Recommendation 1: Aggressive Geographic Segmentation
Risk is hyper-localized. ACIS must implement strict zip-code-level pricing multipliers. We have identified exactly which postal codes yield positive margins. Marketing should immediately isolate these "green zones" and heavily discount premiums to acquire low-risk volume.

### Recommendation 2: Transition to ML-Based Underwriting
The current pricing strategy is failing. By implementing the Dual-Stage XGBoost/Random Forest pipeline, ACIS can instantly flag the top 10% riskiest applicants at the point of quote. High-risk applicants should either be declined or priced at the maximum bounds of the risk premium calculator (upwards of R100k+ annually for the most dangerous profiles).

### Recommendation 3: Eliminate Gender-Based Pricing Strategies
The statistical evidence is conclusive: gender does not impact claim frequency or severity. Any marketing budget allocated specifically toward "female drivers" under the assumption that they are safer is wasted. Resources should be reallocated to geographic and vehicle-class targeting.

### Recommendation 4: Sub-segment High Performance Vehicles
Vehicles with high cubic capacity and cylinder counts are driving severe losses. Underwriting should introduce steep non-linear premium curves for high-performance vehicles, especially in urban postal codes where the risk compounds.

---
**End of Report**
*Technical Addendum: All code, Jupyter Notebooks (01_eda.ipynb, 02_hypothesis_testing.ipynb, 03_modeling.ipynb), and unit tests are version-controlled via Git and DVC in the official project repository.*
