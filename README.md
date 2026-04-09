# 📊 Credit Risk Prediction Using Machine Learning

## 📌 Project Overview
Loan default prediction is a critical task in financial risk management. Incorrect lending decisions can lead to significant financial losses.  

This project builds a **robust machine learning pipeline** to predict whether a borrower will default on a loan using financial, demographic, and credit behavior data.

The project follows an **end-to-end ML workflow**:

- Data cleaning and preprocessing  
- Feature engineering  
- Handling class imbalance (SMOTE – applied selectively)  
- Model training (Logistic Regression, Random Forest, XGBoost)  
- Model evaluation and comparison  
- Cross-validation  
- ROC-AUC analysis  

---

## 📂 Dataset Description

- **Total observations:** 32,409  
- **Training set:** 25,927  
- **Testing set:** 6,482  
- **Total features after preprocessing:** 21  

### 🎯 Target Variable
- **loan_status**
  - `0` → Non-default  
  - `1` → Default  

### 🧾 Key Features
- person_age  
- person_income  
- person_emp_length  
- loan_amnt  
- loan_int_rate  
- loan_grade  
- loan_intent  
- person_home_ownership  
- cb_person_cred_hist_length  
- cb_person_default_on_file  

---

## 🛠️ Data Preprocessing

Key steps performed:

- Removed duplicates  
- Handled missing values:
  - Numerical → median  
  - Categorical → mode  
- Removed unrealistic values:
  - Age > 100  
  - Employment length > age  
- Outlier treatment:
  - Income capped at 1st and 99th percentiles  

---

## ⚙️ Feature Engineering

Created meaningful domain-based features:

- **Monthly Payment Estimate**  
  `monthly_payment_est = loan_amnt / 12`

- **Income per Employment Year**  
  `income_per_year_of_emp = person_income / (person_emp_length + 1)`

- **Interest-to-Income Ratio**  
  `interest_income_ratio = loan_int_rate / (person_income + 1)`

- **Credit Experience Ratio**  
  `credit_exp_ratio = cb_person_cred_hist_length / person_age`

---

## 🔄 Data Preparation

- Train-test split: **80/20**
- Feature scaling: **StandardScaler**

### ⚖️ Handling Class Imbalance
- **SMOTE applied only to Logistic Regression**
- Random Forest and XGBoost handled imbalance using:
  - `class_weight='balanced'` (Random Forest)
  - `scale_pos_weight` (XGBoost)

---

## 🤖 Models Trained

### 1. Logistic Regression (with SMOTE)
- Trained on **scaled + SMOTE-balanced data**
- Designed to improve recall of minority class (defaulters)

### 2. Random Forest
- n_estimators = 200  
- max_depth = 10  
- min_samples_split = 10  
- class_weight = 'balanced'  

### 3. XGBoost
- n_estimators = 300  
- max_depth = 6  
- learning_rate = 0.05  
- scale_pos_weight applied  

---

## 📈 Model Evaluation Results

### 🔹 Logistic Regression (SMOTE Applied)
- **Accuracy:** 80.31%  
- **ROC-AUC:** 0.8738  
- **Cross-Val ROC-AUC:** 0.8676  

**Confusion Matrix**


| Class | Precision | Recall | F1-score |
|------|----------|--------|---------|
| Non-default (0) | 0.93 | 0.81 | 0.87 |
| Default (1)     | 0.54 | 0.78 | 0.63 |

**Insight:**  
SMOTE improved recall (0.78), increasing detection of defaulters, but reduced precision.

---

### 🔹 Random Forest
- **Accuracy:** 90.73%  
- **ROC-AUC:** 0.9341  

**Confusion Matrix**


| Class | Precision | Recall | F1-score |
|------|----------|--------|---------|
| Non-default (0) | 0.94 | 0.95 | 0.94 |
| Default (1)     | 0.80 | 0.77 | 0.78 |

**Insight:**  
Balanced performance with reduced false positives.

---

### 🔹 XGBoost (Best Model)
- **Accuracy:** **91.62%**  
- **ROC-AUC:** **0.9487**  

**Confusion Matrix**


| Class | Precision | Recall | F1-score |
|------|----------|--------|---------|
| Non-default (0) | 0.93 | 0.96 | 0.95 |
| Default (1)     | 0.85 | 0.75 | 0.80 |

**Insight:**  
Best overall performance with highest ROC-AUC and strong precision.

---

## 📊 Model Comparison

| Model | Accuracy | ROC-AUC | Key Strength |
|------|---------|--------|-------------|
| Logistic Regression (SMOTE) | 80.3% | 0.8738 | High recall |
| Random Forest | 90.7% | 0.9341 | Balanced performance |
| **XGBoost** | **91.6%** | **0.9487** | **Best overall performance** |

---

## 🔍 Key Insights

- Selective use of SMOTE improved model reliability  
- Logistic Regression benefited from SMOTE → higher recall  
- Tree-based models handled imbalance without oversampling  
- XGBoost achieved the best overall performance  
- Feature engineering significantly improved predictions  

---

## 💼 Business Implications

This model can support:

- Automated loan approval systems  
- Risk-based pricing strategies  
- Credit scoring enhancement  
- Early detection of high-risk borrowers  
- Portfolio risk management  

---

## 🧰 Tools & Technologies

- Python  
- Pandas, NumPy  
- Scikit-learn  
- XGBoost  
- Imbalanced-learn (SMOTE)  
- Matplotlib  

---

## ✅ Conclusion

This project demonstrates a **complete and production-ready machine learning pipeline** for credit risk prediction.

### 🏆 Final Model (XGBoost)
- Accuracy: **91.6%**  
- ROC-AUC: **0.9487**  

The model effectively distinguishes between defaulters and non-defaulters and is suitable for real-world financial applications.

---

## 🚀 Future Improvements

- Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)  
- Model interpretability using SHAP  
- Deployment via Flask or FastAPI  
- Dashboard development (Streamlit / Power BI)  
- Ensemble stacking  

---

## 📁 Project Structure

[Data Preprocessing Notebook](Credit_Risk_Preprocessing)

[ML Notebook](Credit_Risk_ML.ipynb)

[Raw Dataset](credit_risk_dataset.csv)

[Processed Dataset](data_preprocessed.csv)

[Image](ROC_Curve_Comparison.png)

[Image](feature_importance.png)


---

## 👤 Author
**Adedayo Adebayo**  
Data Analyst | Machine Learning Practitioner  

---

## ⭐ If you found this useful
Give this repository a star ⭐
