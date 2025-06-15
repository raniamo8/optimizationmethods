# Predicting Corporate Bankruptcy with Logistic Regression

This project focuses on predicting corporate bankruptcy using real financial data from Taiwanese companies. A logistic regression model is used, combined with class balancing and hyperparameter tuning, to identify companies at high risk of bankruptcy.

---

## Installation

Install all required dependencies with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

---

## Overview

- Dataset: 6,819 companies (1999–2009), 96 financial features  
- Goal: Predict whether a company will go bankrupt (`Bankrupt?`)
- Challenges: Strong class imbalance (very few bankruptcies)
- Key Tools: Logistic Regression, SMOTE, GridSearchCV

---

## Methods

- **Data Scaling:** `StandardScaler` to normalize all features
- **Class Balancing:** `SMOTE` to oversample the minority class
- **Model:** `LogisticRegression` with L1 and L2 regularization
- **Hyperparameter Tuning:** `GridSearchCV` with 5-fold CV  
- **Evaluation Metrics:** F1-score, ROC-AUC, confusion matrix
- **Feature Importance:** via L1 regularization coefficients

---