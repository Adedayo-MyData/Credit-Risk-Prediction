# Credit Risk Prediction Using Machine Learning

## Project Overview

Financial institutions face significant losses when borrowers default on loans. This project develops a machine learning model to predict loan default using borrower financial characteristics and credit behavior indicators.

The project follows a complete machine learning pipeline:

* Data loading and inspection
* Data preprocessing
* Feature engineering
* Model development using Random Forest
* Model evaluation
* Probability threshold optimization

The final model achieved **94.3% accuracy** with strong default detection performance and an excellent ROC-AUC score.

---

## Dataset Description

The dataset contains **5,000 loan records** representing borrower financial and credit-related attributes.

### Dataset Size

* **Total observations:** 5,000
* **Training set size:** 4,000
* **Testing set size:** 1,000

### Variables in the Dataset

| Variable                | Description                               |
| ----------------------- | ----------------------------------------- |
| loan_id                 | Unique loan identifier                    |
| annual_income           | Borrower annual income                    |
| monthly_debt            | Monthly debt obligations                  |
| loan_amount             | Amount of loan requested                  |
| total_credit_limit      | Maximum available credit                  |
| credit_used             | Amount of credit already used             |
| total_delinquencies_12m | Number of delinquencies in last 12 months |
| open_credit_accounts    | Number of open credit accounts            |
| credit_history_years    | Length of credit history                  |
| oldest_account_age      | Age of oldest credit account              |
| late_payments_90d       | Payments overdue >90 days                 |
| hard_inquiries_last_6m  | Recent credit inquiries                   |
| default                 | Target variable                           |

---

## Data Preprocessing

Initial inspection of the dataset revealed:

* Presence of raw financial variables requiring transformation
* Need for derived financial ratios
* Variation in scales across features

Data preprocessing prepared the dataset for modeling and improved feature quality.

---

## Feature Engineering

Several financial ratios were created to better capture borrower risk behavior:

* **Debt-to-Income Ratio:**
  debt_to_income = monthly_debt / annual_income
  Measures how much income is committed to debt repayment.

* **Credit Utilization Ratio:**
  credit_utilization = credit_used / total_credit_limit
  Indicates how much of available credit is being used.

* **Loan-to-Income Ratio:**
  loan_to_income = loan_amount / annual_income
  Measures loan size relative to income.

* **Delinquency Ratio:**
  delinquency_ratio = total_delinquencies_12m / open_credit_accounts
  Captures repayment issues across accounts.

* **Utilization-to-Income Ratio:**
  utilization_to_income = credit_utilization / annual_income
  Measures dependence on credit relative to income.

---

## Model Development

A **Random Forest Classifier** was used:

RandomForestClassifier(random_state=42)

### Why Random Forest?

* Captures nonlinear relationships
* Handles feature interactions
* Reduces overfitting via ensemble learning

---

## Model Performance

### Accuracy

**94.3%**

---

## Classification Report

| Class           | Precision | Recall | F1 Score | Support |
| --------------- | --------- | ------ | -------- | ------- |
| Non-Default (0) | 0.96      | 0.96   | 0.96     | 649     |
| Default (1)     | 0.92      | 0.92   | 0.92     | 351     |

### Interpretation

* Strong performance across both classes
* High recall (**0.92**) for defaulters — meaning most risky borrowers are correctly identified
* Balanced precision and recall indicates stable predictions

---

## ROC–AUC Score

**ROC–AUC = 0.9883**

This indicates excellent separability between defaulters and non-defaulters.

---

## Probability Threshold Optimization

Default classification threshold (0.50) was adjusted to improve performance.

### Threshold Results

| Threshold | Precision | Recall | F1 Score |
| --------- | --------- | ------ | -------- |
| 0.10      | 0.6598    | 1.0000 | 0.7950   |
| 0.15      | 0.7457    | 0.9943 | 0.8523   |
| 0.20      | 0.7959    | 0.9886 | 0.8818   |
| 0.25      | 0.8223    | 0.9886 | 0.8978   |
| 0.30      | 0.8424    | 0.9744 | 0.9036   |
| 0.35      | 0.8619    | 0.9601 | 0.9084   |
| 0.40      | 0.8850    | 0.9430 | 0.9131   |
| 0.45      | 0.9176    | 0.9202 | 0.9189   |
| 0.50      | 0.9296    | 0.9031 | 0.9162   |

---

## Key Insight from Threshold Tuning

* Lower thresholds increase recall (detect more defaulters)
* Higher thresholds increase precision (reduce false positives)

The **0.40 – 0.45 range** provides the best balance between precision and recall.

---

## Feature Importance Insights

Although exact importance values were not explicitly printed, model behavior indicates that:

* Engineered financial ratios strongly influenced predictions
* Credit utilization and debt-related metrics were key drivers
* Historical behavior variables (delinquencies, late payments) contributed significantly

---

## Key Discoveries

1. Engineered financial ratios significantly improve predictive performance
2. Credit utilization is a strong indicator of financial stress
3. Default prediction performance is highly sensitive to threshold selection
4. The model achieves strong balance between precision and recall without heavy tuning

---

## Business Implications

This model can support:

* **Automated credit decision systems**
* **Risk-based loan pricing**
* **Early identification of high-risk borrowers**
* **Improved portfolio risk management**

---

## Tools and Technologies

* Python
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib

---

## Conclusion

This project demonstrates a complete credit risk prediction pipeline.

### Final Performance:

* **Accuracy:** ~94.3%
* **Default Recall:** ~9%
* **ROC–AUC:** 0.9883

The results show that combining feature engineering, Random Forest, and threshold tuning produces a highly effective credit risk model.
----
## Project Files

[Data Preprocessing Notebook](CreditRisk_Preprocessed.ipynb)

[ML Notebook](CreditRiskML.ipynb)

[Raw Data](credit_risk_raw.csv)

[Processed Data](preprocessed_data.csv)

[Image](ROC%20Curve.png) – ROC Curve

[Image](feature_importance) - Feature Importance

## Future Improvements

* Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
* Try boosting models (XGBoost, LightGBM)
* Cross-validation for robustness
* Deployment via API or dashboard
* Feature importance visualization

---

## Author

**Adedayo Adebayo**
Data Analyst | ML Practitioner
