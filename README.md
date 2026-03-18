# Credit Risk Prediction Using Machine Learning

## Project Overview
Financial institutions face significant financial losses when borrowers default on loans. Accurate credit risk assessment enables lenders to make informed loan approval decisions, reduce exposure to bad debt, and maintain a healthy lending portfolio.

This project develops a machine learning model that predicts whether a borrower will default on a loan using financial characteristics and credit behavior indicators.

The project follows a complete end-to-end machine learning workflow, including:

- Raw data inspection  
- Data cleaning and preprocessing  
- Feature engineering  
- Multicollinearity checks  
- Model development using Random Forest  
- Model evaluation  
- Probability threshold optimization  

After optimization, the final model achieved **93.75% accuracy** with significantly improved ability to detect high-risk borrowers.

---

## Dataset Description
The dataset contains **5,000 loan records** representing borrower financial conditions and credit history indicators.

### Dataset Size
- **Total observations:** 5,000  
- **Total variables:** 13  

### Variables in the Dataset

| Variable                     | Description |
|----------------------------|------------|
| loan_id                     | Unique loan identifier |
| annual_income               | Borrower annual income |
| monthly_debt                | Monthly debt obligations |
| loan_amount                 | Amount of loan requested |
| total_credit_limit          | Maximum available credit |
| credit_used                 | Amount of credit already used |
| total_delinquencies_12m     | Number of delinquencies in the last 12 months |
| open_credit_accounts        | Number of open credit accounts |
| credit_history_years        | Length of borrower credit history |
| oldest_account_age          | Age of oldest credit account |
| late_payments_90d           | Number of payments overdue by more than 90 days |
| hard_inquiries_last_6m      | Credit inquiries within the last six months |
| default                     | Target variable (1 = Default, 0 = Non-default) |

---

## Data Cleaning and Preprocessing

The first stage of the project focused on preparing the raw dataset for machine learning modeling.

### Dataset Inspection
Initial data exploration revealed several issues that required preprocessing:

- Missing values in multiple predictor variables  
- Differences in variable scales across numerical features  
- Raw financial variables that could be transformed into more informative financial indicators  

**Initial dataset size:** 5,000 observations  

---

## Handling Missing Values

Rows containing missing values were removed to ensure the model was trained only on complete observations.

**After cleaning:**
- Final dataset size: **4,558 observations**  
- Total removed records: **442**  

Removing incomplete records ensured the reliability of the engineered features and prevented bias during model training.

---

## Feature Engineering

Feature engineering was applied to derive more meaningful financial indicators from the raw variables. These engineered variables help capture borrower risk behavior more effectively than raw financial values alone.

Five financial ratios were created.

---

### Debt-to-Income Ratio
Measures debt burden relative to income.

debt_to_income = monthly_debt / annual_income

This ratio measures the proportion of a borrower’s income already committed to debt repayment. Higher values indicate that a large share of income is used to service debt, increasing the likelihood of financial distress.

---

### Credit Utilization Ratio
Measures the proportion of available credit currently being used.

credit_utilization = credit_used / total_credit_limit

High credit utilization often signals financial strain and is widely recognized as a strong predictor of credit risk.

---

### Loan-to-Income Ratio
Measures the size of the requested loan relative to the borrower’s income.

loan_to_income = loan_amount / annual_income

Borrowers requesting loans that are large relative to their income may face greater difficulty meeting repayment obligations.

---

### Delinquency Ratio
Measures the frequency of delinquent accounts relative to the number of active credit accounts.

delinquency_ratio = total_delinquencies_12m / open_credit_accounts

A higher ratio indicates a pattern of repayment problems across the borrower’s credit portfolio.

---

### Utilization-to-Income Ratio
Captures the relationship between credit utilization and income.

utilization_to_income = credit_utilization / annual_income

This metric helps detect borrowers who rely heavily on credit relative to their income level, which may signal elevated financial risk.

## Multicollinearity Check

Highly correlated variables can reduce model interpretability and introduce redundancy. A multicollinearity assessment was therefore performed to ensure that the model learned from independent financial signals.

The following variables were retained for modeling:

- late_payments_90d  
- credit_history_years  
- oldest_account_age  
- hard_inquiries_last_6m  
- debt_to_income  
- credit_utilization  
- loan_to_income  
- total_delinquencies_12m  
- delinquency_ratio  
- utilization_to_income  

These variables collectively represent borrower credit history, financial pressure, and borrowing behavior.

---

## Preparing Data for Modeling

After preprocessing and feature engineering, the dataset was divided into training and testing sets.

### Training Set
- x_train: 3,646 observations  

### Testing Set
- x_test: 912 observations  

The target variable is **loan default status**.

---

## Feature Scaling

All predictors were standardized using StandardScaler.

Standardization ensures that variables with larger numeric ranges do not dominate the learning process, allowing the model to treat all features equally during training.

---

## Machine Learning Model

The project uses a **Random Forest Classifier**.

Random Forest is particularly suitable for credit risk prediction because it:

- Captures nonlinear relationships between variables  
- Handles complex interactions between predictors  
- Reduces overfitting through ensemble learning  

Model initialization:  
RandomForestClassifier (random_state = 42)

---

## Initial Model Performance

The initial Random Forest model produced the following results on the test dataset.

### Accuracy
**0.9013**

### Confusion Matrix

|                      | Predicted Non-Default | Predicted Default |
|----------------------|----------------------|-------------------|
| **Actual Non-Default** | 735                  | 6                 |
| **Actual Default**     | 84                   | 87                |

### Interpretation

- The model correctly classified most safe borrowers.  
- However, many actual defaulters were incorrectly classified as safe borrowers, which could expose lenders to financial losses.  

---

## Classification Report

| Class        | Precision | Recall | F1 Score |
|-------------|----------|--------|----------|
| Non-Default | 0.90     | 0.99   | 0.94     |
| Default     | 0.94     | 0.51   | 0.66     |

The recall of **0.51** for defaulters indicates that the model was failing to identify nearly half of the risky borrowers.

---

## ROC–AUC Score

The model achieved:

**ROC–AUC = 0.947**

This indicates excellent ability to distinguish between defaulters and non-defaulters.

## Probability Threshold Optimization

Classification models normally use **0.50** as the decision threshold.

However, in credit risk applications this threshold may not be optimal because:

- Missing a defaulter is far more costly than incorrectly rejecting a safe borrower  

To improve risk detection, several probability thresholds were evaluated.

---

## Threshold Experiment Results

| Threshold | Precision | Recall | F1 Score |
|----------|----------|--------|----------|
| 0.10     | 0.317    | 0.976  | 0.479    |
| 0.20     | 0.623    | 0.900  | 0.737    |
| 0.25     | 0.760    | 0.854  | 0.804    |
| 0.30     | 0.848    | 0.813  | 0.830    |
| 0.50     | 0.935    | 0.509  | 0.659    |

The **0.30 threshold** produced the best balance between precision and recall.

---

## Final Model Performance

Using a **0.30 probability threshold**, model performance improved significantly.

### Accuracy
**93.75%**

### Confusion Matrix

|                      | Predicted Non-Default | Predicted Default |
|----------------------|----------------------|-------------------|
| **Actual Non-Default** | 716                  | 25                |
| **Actual Default**     | 32                   | 139               |

### Key Improvement

Default detection recall improved from:

**51% ? 81%**

This means the model now identifies most risky borrowers.

---

## Feature Importance Analysis

Random Forest models allow extraction of feature importance scores, which indicate how strongly each variable influences predictions.

Feature importance was obtained using:  
rf.feature_importances_

### Most Important Predictors

| Feature                 | Importance Insight |
|------------------------|-------------------|
| debt_to_income         | High debt burden increases default risk |
| credit_utilization     | High utilization signals financial strain |
| utilization_to_income  | Indicates reliance on credit relative to income |
| loan_to_income         | Larger loans relative to income increase risk |
| credit_history_years   | Longer credit history generally indicates reliability |
| oldest_account_age     | Older accounts suggest established credit behavior |

### Key Discovery

Engineered financial ratios such as **credit_utilization** and **debt_to_income** proved to be stronger predictors than many raw financial variables.

This demonstrates the importance of feature engineering in financial risk modeling.

---

## Why Threshold Tuning Improved Recall

At the default threshold of **0.50**, many borrowers with moderate default probabilities were classified as safe.

**Example:**

Predicted probability of default = 0.35  

- At threshold 0.50 ? Classified as Non-Default  
- At threshold 0.30 ? Classified as Default  

Lowering the threshold enabled the model to capture more risky borrowers, significantly improving recall.

---

## Key Discoveries From the Project

1. Financial ratios outperform raw financial variables in predicting credit risk  
2. Credit utilization is one of the strongest indicators of financial stress  
3. Late payment history strongly predicts future default behavior  
4. Adjusting the classification threshold significantly improves default detection  

---

## Business Implications

The model can support financial institutions in several ways:

- **Automated Loan Approval**  
  Borrowers can be automatically classified as low-risk or high-risk  

- **Risk-Based Lending**  
  Interest rates can be adjusted based on predicted borrower risk  

- **Early Risk Detection**  
  Financial institutions can identify borrowers likely to default and intervene early  

- **Portfolio Risk Management**  
  Helps banks maintain a healthier loan portfolio by reducing exposure to high-risk lending  

---

## Tools and Technologies

- Python  
- Pandas  
- NumPy  
- Scikit-Learn  
- Matplotlib  
- Seaborn  

---

## Conclusion

This project demonstrates a complete machine learning pipeline for credit risk prediction.

Through feature engineering and probability threshold optimization, the Random Forest model achieved:

- **Accuracy:** 93.75%  
- **Default Recall:** 81%  
- **ROC–AUC:** 0.947  

These results highlight how machine learning can significantly improve loan default prediction and credit risk management.

---

## Project Files

- [CreditRisk_Preprocessed Notebook](notebooks/CreditRisk_Preprocessed.ipynb) – Data cleaning, preprocessing, and exploratory analysis.  
- [CreditRiskML Notebook](notebooks/CreditRiskML.ipynb) – Machine Learning model building, evaluation, and predictions.  
- [Raw Dataset](data/credit_risk_raw.csv) – Original dataset used for analysis.  
- [Preprocessed Dataset](data/data_preprocessed.csv) – Cleaned dataset ready for ML models.  
- [ROC Curve](images/ROC%20Curve.png) – Model evaluation visualization.

## Future Improvements

Several enhancements can further improve the model:

- Testing additional algorithms such as XGBoost, LightGBM, and Logistic Regression  
- Applying hyperparameter tuning to optimize model performance  
- Implementing cross-validation to improve model robustness  
- Incorporating additional financial behavior variables  
- Building an interactive dashboard (Power BI or Streamlit) for credit risk visualization  

---

## Author

**Adedayo Adebayo**  
Data Scientist | Business Analyst
