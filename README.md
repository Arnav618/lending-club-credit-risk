# Credit Risk Prediction & Portfolio Analysis — Lending Club

Predicting loan default risk using XGBoost, validated with statistical testing and SHAP interpretability, with a cost-sensitive threshold analysis and a 3-page Power BI dashboard for portfolio risk monitoring.

## Project Overview

Lenders don't just need to know *who* will default — they need to know how much a wrong call actually costs. This project builds a loan default classifier on Lending Club data, but treats the model as one piece of a larger risk decision: every step, from feature selection to the final probability threshold, is checked against whether it would survive scrutiny in a real underwriting or portfolio-risk context. That meant catching data leakage before it inflated results, validating the top risk drivers statistically rather than trusting feature importances at face value, and setting the classification threshold using actual interest-margin economics instead of an arbitrary 0.5 cutoff.

## Key Findings

- **Caught and fixed data leakage**: the top feature's mutual information score dropped from 0.47 to 0.04 once post-outcome columns (fields only known after a loan's fate is decided) were removed — the "before" model was learning the answer, not predicting it.
- **Tuned XGBoost reached AUC 0.7265** (baseline: 0.7177), with the improvement confirmed via bootstrap confidence interval [0.7127, 0.7226].
- **Statistically validated key risk drivers** using t-tests (p < 0.001), so the drivers reported aren't just high on a feature-importance chart — they hold up under a formal test.
- **Cost-sensitive threshold analysis**: an initial assumption of 10% cost-per-default was replaced with the real interest-margin figure (27.2%), which shifted the optimal decision threshold from 0.25 to 0.50 — a concrete example of how a wrong cost assumption changes the actual lending decision, not just a metric.

## Tech Stack

Python (pandas, XGBoost, scikit-learn, SHAP), PostgreSQL/SQL, Power BI, DAX

## Pipeline

1. Data cleaning & leakage detection
2. Feature engineering
3. Model training & hyperparameter tuning
4. Statistical validation of risk drivers
5. Cost-sensitive threshold analysis
6. Power BI dashboard for portfolio monitoring

## Dashboard Screenshots

**Page 1 — Portfolio Overview**
![Power BI Overview](images/01_Portfolio_Risk_Overview.png)

**Page 2 — Segmentation**
![Power BI Segmentation](images/02_Risk_Segmentation.png)

**Page 3 — Model Output**
![Power BI Model Output](images/03_Model_Output.png)

**SHAP Summary**
![SHAP Summary](images/shap_summary.png)

## Results

| Metric | Baseline | Tuned Model |
|---|---|---|
| AUC-ROC | 0.7177 | 0.7265 |
| Bootstrap 95% CI | — | [0.7127, 0.7226] |
| Optimal threshold (cost-corrected) | 0.25 (assumed cost) | 0.50 (real cost) |

## Repository Structure

```
lending-club-credit-risk/
├── README.md
├── notebooks/
│   └── credit_risk_analysis.ipynb
├── images/
│   ├── shap_summary.png
│   ├── powerbi_page1_overview.png
│   ├── powerbi_page2_segmentation.png
│   └── powerbi_page3_model_output.png
├── dashboard/
│   └── credit_risk_dashboard.pbix
└── requirements.txt
```

## Data

This project uses the [Lending Club Loan Data](https://www.kaggle.com/datasets/adarshsng/lending-club-loan-data-csv) dataset from Kaggle. The raw dataset is not included in this repo — download it directly from Kaggle to reproduce the analysis.

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/credit_risk_analysis.ipynb
```

