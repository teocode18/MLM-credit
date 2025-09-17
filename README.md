
# Credit Score Prediction (XGBoost)

## Overview
This project involves building and tuning an **XGBoost** machine learning model to predict **credit scores** based on customer data. The model classifies customer credit scores into three categories: **Good**, **Poor**, and **Standard**.

The dataset contains various features such as **Outstanding Debt**, **Total EMI per month**, and **Annual Income**. This project demonstrates an end-to-end machine learning pipeline, including data preprocessing, model training, hyperparameter tuning, model evaluation, and visualizations.

## Key Features:
- Data cleaning and preprocessing (handling missing values, encoding categorical features).
- Training an **XGBoost** model for **multi-class classification**.
- **Hyperparameter optimization** using **RandomizedSearchCV**.
- **Model evaluation** using **classification report** and **confusion matrix**.
- **Feature importance visualization** to identify key predictive features.
- **SHAP (Shapley Additive Explanations)** for model interpretability (optional).
- Model saving for future use (saved as `xgb_best_model.pkl`).

## Dataset:
- The dataset is loaded from a CSV file `train.csv` and contains the following columns:
  - **Credit_Score**: Target variable (Good, Poor, Standard).
  - **Outstanding_Debt**: Total outstanding debt of the customer.
  - **Total_EMI_per_month**: Total monthly EMI (Equated Monthly Installment).
  - **Annual_Income**: Annual income of the customer.
  - **Other features**: Various financial attributes related to the customer.

## Installation and Requirements:
To run the project locally, make sure you have the following Python libraries installed:
- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `matplotlib`
- `seaborn`
- `joblib`

You can install the dependencies using `pip`:
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib
