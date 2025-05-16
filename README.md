# Customer Churn Prediction 📉

This project aims to predict whether a customer will churn (i.e., leave a service) using the XGBoost machine learning model. Predicting churn is crucial for businesses to retain customers and reduce loss.

> ✅ This project was completed as part of a **graduation project from DEPI (Digital Egypt Pioneers Initiative)**.

---

## 🔍 Problem Statement

Churn prediction helps identify customers likely to stop using a service. By analyzing historical data, we can build a model that accurately predicts churn and enables businesses to take preventive actions.

---

## 📦 Dataset

- **Source:** [Telco Customer Churn - Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn)
- **Details:** The dataset includes customer information such as demographics, services signed up for, account information, and whether they churned.

---

## 🛠️ Tools & Technologies Used

- Python
- Pandas, NumPy
- XGBoost
- Scikit-learn
- Matplotlib, Seaborn
- Jupyter Notebook

---

## 🧠 Model Used

- **XGBoost (Extreme Gradient Boosting)**
  - Chosen for its high performance and ability to handle imbalanced datasets.
  - Early stopping was applied to prevent overfitting and improve generalization.

---

## 📈 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

---

## 📊 Results

The final model was trained using **XGBoost** with **early stopping**, and evaluated on the test set:

| Metric                        | Value |
|-------------------------------|--------|
| **Accuracy**                  | 79%   |
| **Precision (Class 0)**       | 0.57 |
| **Precision (Class 1)**       | 0.94 |
| **Recall (Class 0)**          | 0.85 |
| **Recall (Class 1)**          | 0.77 |
| **F1-Score (Class 0)**        | 0.68 |
| **F1-Score (Class 1)**        | 0.84 |
| **Macro Avg F1**              | 0.76 |
| **Weighted Avg F1**           | 0.80 |

> 🔎 The model performs very well at identifying customers likely to churn (class 1), though it is less precise in identifying those who won’t churn. This could be due to class imbalance in the dataset.

---

## 📌 Key Insights

- Customers on **month-to-month contracts** are more likely to churn.
- Longer **tenure** is associated with lower churn risk.
- Customers with **higher monthly charges** and **no tech support** are more prone to churn.
- **Automatic payment methods** are associated with lower churn.

---

📬 Contact
👤 Name: Ayman Negm

📧 Email: aymansamy1772004@gmail.com

🔗 LinkedIn: ayman-negm-690b322b4




