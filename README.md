# 💳 Fraud Detection Using Machine Learning

## 📌 Project Overview
This project implements an **end-to-end Machine Learning solution for detecting fraudulent financial transactions**.  
The primary objective is to **identify fraud with high recall**, minimizing financial loss caused by missed fraud cases.

The project covers the complete ML lifecycle:
**Data Analysis → Feature Engineering → Model Training → Evaluation → Model Selection → Deployment Readiness**

---

## 🎯 Business Problem
Financial institutions face challenges in fraud detection due to:
- Highly **imbalanced datasets**
- High cost of **false negatives (missed fraud)**
- Complex and evolving fraud patterns

**Goal:**  
Build a reliable fraud detection model that prioritizes **catching fraudulent transactions** while maintaining model stability.

---

## 📊 Dataset Description
The dataset contains transaction-level information such as:
- Transaction type
- Transaction amount
- Sender and receiver balances
- Fraud label (`isFraud`)

📌 **Key Challenge:**  
The dataset is **highly imbalanced**, which is common in real-world fraud detection problems.

---

## 🔍 Exploratory Data Analysis (EDA)
Key EDA steps performed:
- Checked missing values and data types
- Analyzed fraud vs non-fraud distribution
- Studied fraud patterns across transaction types
- Compared transaction amounts for fraud and non-fraud cases

These insights helped guide feature engineering and model selection.

---

## 🛠 Feature Engineering
Created a new feature to capture suspicious behavior:

### ➕ Balance Difference
```python
balance_diff = oldbalanceOrg - newbalanceOrig


- This helps capture abnormal balance behavior, a strong fraud indicator.

---

## 🤖 Models Used
- **Logistic Regression** (Baseline model)
- **Random Forest Classifier** (Final model)

Class imbalance handled using:
- `class_weight='balanced'`

---

## 📈 Model Evaluation
Evaluation metrics used:
- Recall (Primary metric)
- Precision
- F1-score
- ROC-AUC
- Confusion Matrix

📌 **Why Recall?**
In fraud detection, missing a fraud transaction is more costly than flagging a genuine one.

---

## 🏆 Results
- Random Forest outperformed Logistic Regression
- Better recall and F1-score
- Model generalized well without overfitting

---

## 🔎 Feature Importance
Random Forest feature importance was used to identify key fraud indicators such as:
- Transaction amount
- Balance difference
- Transaction type

---

##🚀 Streamlit Deployment
##📌 Objective

-Deploy the trained fraud detection model as an interactive web application using Streamlit.


##🧱 Application Features

-User-friendly input form
-Real-time fraud prediction
-Displays:
   -Fraud / Non-Fraud result
   -Prediction confidence
