
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
- Model saving for future use (saved as `xgb_best_model.pkl`).

## Dataset:
- The dataset is loaded from a CSV file `train.csv` and contains the following columns:
  - **Credit_Score**: Target variable (Good, Poor, Standard).
  - **Outstanding_Debt**: Total outstanding debt of the customer.
  - **Total_EMI_per_month**: Total monthly EMI (Equated Monthly Installment).
  - **Annual_Income**: Annual income of the customer.
  - **Other features**: Various financial attributes related to the customer.
 

# Visualisations:
## 1) Confusion matrix
### The confusion matrix plot shows how well the model predicts each class (Good, Poor, Standard). Here is an example of the output:

<img width="277" height="215" alt="confusionmatrix" src="https://github.com/user-attachments/assets/8d4590d1-f173-40f8-a3cd-533ec8ea5fc0" />

## Insights
### Rows represent actual classes (true labels), columns represent predicted classes (predicted labels), the values in the matrix are the counts of instances.
### -Good Class: The model correctly identified 234 instances as Good, but it did misclassify 110 as Standard and 6 as Poor.
### -Poor Class: The model identified 389 instances as Poor, but 39 instances were misclassified as Good, and 161 as Standard.
### -Standard Class: The model predicted 808 instances correctly as Standard, though some instances of Good (111) and Poor (142) were misclassified as Standard.
#### The model correctly predicted a high percentage of instances in each class, with accuracy around 72%. While the model is performing well, there is room to improve its ability to distinguish between "Good" and "Standard" categories. The misclassifications are expected in real-world scenarios and could be improved with further tuning and balancing.


## 2) Feature importance:
### The feature importance plot shows the most important features that contribute to the model's predictions. For instance:

<img width="410" height="254" alt="featureimportance" src="https://github.com/user-attachments/assets/39993adc-ff75-4dc8-a8db-729af790779f" />

## Insights
### The top three features contributing the most to predicting credit scores are:
### - Outstanding Debt: This is likely a major factor, as the more debt someone has, the more likely they are to have a lower credit score.
### - Total EMI per month: The monthly EMI payments would directly influence the ability to repay loans, impacting credit score.
### - Changed Credit Limit: This could indicate credit behavior changes and how credit limits impact score predictions
#### The plot suggests that these financial factors are key indicators of creditworthiness, as expected




# How to run
## Clone the repository to your local machine:
git clone https://github.com/your-username/credit-score-prediction.git
cd credit-score-prediction

## run the Python script:
python credit_score_model.py

# Check for results:
The model performance will be displayed in the notebook, including a confusion matrix and feature importance.
Saved model file (xgb_best_model.pkl) for future use.


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
bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib














