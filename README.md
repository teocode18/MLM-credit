
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



# Visualisations:
## 1) Confusion matrix
### The confusion matrix plot shows how well the model predicts each class (Good, Poor, Standard). Here is an example of the output:

<img width="410" height="254" alt="confusionmatrix" src="https://github.com/user-attachments/assets/8d4590d1-f173-40f8-a3cd-533ec8ea5fc0" />

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


## 3) Classification report
### 📊 Model Performance Metrics

| Class     | Precision | Recall | F1-Score | Support |
|:----------|----------:|-------:|---------:|--------:|
| Good      | 0.72      | 0.75   | 0.73     | 500     |
| Poor      | 0.70      | 0.65   | 0.67     | 600     |
| Standard  | 0.75      | 0.72   | 0.73     | 900     |
| **Average** | **0.72**   | **0.71** | **0.72** | **2000** |


### -Precision: Out of all the instances the model predicted for a particular class, how many were correct.
### -Recall: Out of all the actual instances of a class, how many did the model correctly identify.
### -F1-Score: Harmonic mean of Precision and Recall, providing a balanced metric.
### -Support: The number of actual samples of each class in the dataset (e.g., 500 Good, 600 Poor, 900 Standard = 2000 total)  

## Key Metrics:

The classification report provides the following metrics for each class: **Precision**, **Recall**, **F1-Score**, and **Support**. Here’s how to interpret these metrics and what insights we can gather from them:

### Class-Specific Metrics:

#### **Good Class**:
- **Precision**: `0.75`  
  - **Insight**: The model correctly predicted 75% of instances labeled as **Good**. This suggests that the model is fairly reliable when predicting a **Good credit score**.

- **Recall**: `0.80`  
  - **Insight**: The model identified 80% of the actual **Good** instances. This means the model is missing **20%** of all the **Good** instances. The model is doing well but could be improved to catch the remaining **20%**.

- **F1-Score**: `0.77`  
  - **Insight**: The **F1-Score** balances **precision** and **recall**. An F1-Score of **0.77** indicates that the model performs reasonably well on the **Good class** and has a good balance between **precision** and **recall**.

---

#### **Poor Class**:
- **Precision**: `0.68`  
  - **Insight**: The model correctly predicted 68% of instances labeled as **Poor**. While the precision is decent, **32%** of instances predicted as **Poor** were incorrectly classified as **Good** or **Standard**. This indicates that the model struggles with predicting **Poor** credit scores.

- **Recall**: `0.60`  
  - **Insight**: The model identified **60%** of the actual **Poor** instances. This suggests the model is missing **40%** of the **Poor** cases, indicating that **Poor credit scores** are more likely to be misclassified as **Good** or **Standard**. This is an area for improvement.

- **F1-Score**: `0.64`  
  - **Insight**: The **F1-Score** for the **Poor class** is **0.64**, which is lower compared to the **Good class**. This reflects the imbalance between **precision** and **recall**, meaning the model is either missing many **Poor instances** or incorrectly predicting others as **Poor**.

---

#### **Standard Class**:
- **Precision**: `0.73`  
  - **Insight**: The model correctly predicted **73%** of instances labeled as **Standard**. This suggests the model is fairly good at distinguishing between **Standard** credit scores, though there is still some margin for error.

- **Recall**: `0.75`  
  - **Insight**: The model correctly identified **75%** of the actual **Standard** instances. This is a strong **recall** value, meaning the model is good at catching **Standard** instances, but it still misses **25%** of them. Further optimization could improve recall for this class as well.

- **F1-Score**: `0.74`  
  - **Insight**: The **F1-Score** for the **Standard** class is **0.74**, which suggests a **balanced performance** in terms of both **precision** and **recall**.



    # Overall Conclusion

The XGBoost model demonstrates **solid performance** in predicting credit scores, achieving around **72% accuracy**.  

### 🔹 Key Takeaways:
1. **Confusion Matrix**  
   - The model performs well overall but struggles to clearly separate **Good** and **Standard** credit scores.  
   - Most misclassifications occur between these two classes, which is expected due to their similarity in real-world financial behavior.  

2. **Feature Importance**  
   - The most influential factors in credit score prediction are:  
     - **Outstanding Debt**  
     - **Total EMI per Month**  
     - **Changed Credit Limit**  
   - These align strongly with how creditworthiness is evaluated in practice, suggesting the model captures meaningful patterns.  

3. **Classification Report**  
   - **Good and Standard classes** are predicted with relatively high precision and recall (~0.72–0.75).  
   - **Poor class** shows weaker performance (Precision 0.70, Recall 0.65), meaning the model sometimes confuses **Poor** with the other categories.  
   - Overall, the model maintains a **balanced trade-off** between precision and recall across all classes.  

---

### ✅ Final Insight
The model is a **strong baseline** for credit score prediction, showing both **interpretability** (via feature importance) and **reliable accuracy**.  
However, improvements can be made by:  
- Addressing class imbalance (especially for **Poor** class).  
- Further **hyperparameter tuning** or exploring **ensemble methods**.  
- Incorporating more **domain-specific features** to capture subtle differences between **Good** and **Standard**.  

With these enhancements, the model can become a **robust decision-support tool** for financial institutions assessing credit risk.  






















