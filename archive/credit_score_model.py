import matplotlib.pyplot as plt
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # To save the model

# 1. Load training data
train = pd.read_csv("train.csv").sample(10000, random_state=42)

# 2. Separate features and target
X = train.drop(["Credit_Score"], axis=1)
y = train["Credit_Score"]

# Drop non-informative columns
X = X.drop(["ID", "Customer_ID", "Name", "SSN"], axis=1)

# Fill missing values
X = X.fillna(X.mean(numeric_only=True))

# Convert categorical variables
X = pd.get_dummies(X, drop_first=True)

# Encode target labels
le = LabelEncoder()
y = le.fit_transform(y)  # Good/Poor/Standard -> 0/1/2

# 3. Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Smaller parameter search space (Fast Mode)
param_dist = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 4, 5],
    "learning_rate": [0.05, 0.1, 0.2],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}

# 5. Initialize base model
xgb_clf = XGBClassifier(
    objective="multi:softmax",
    num_class=3,
    eval_metric="mlogloss",
    random_state=42
)

# 6. Hyperparameter search (Fast Mode)
random_search = RandomizedSearchCV(
    estimator=xgb_clf,
    param_distributions=param_dist,
    n_iter=10,              # try only 10 combos
    scoring="accuracy",
    cv=2,                   # 2-fold CV (faster)
    verbose=2,
    random_state=42,
    n_jobs=-1,
    return_train_score=True
)

random_search.fit(X_train, y_train)

# 7. Save tuning results
results_df = pd.DataFrame(random_search.cv_results_)
results_df.to_csv("tuning_results_fast.csv", index=False)
print("\n✅ Fast tuning results saved to tuning_results_fast.csv")

# 8. Best model
best_model = random_search.best_estimator_
print("\n✅ Best Parameters Found (Fast Mode):", random_search.best_params_)

# 9. Evaluate tuned model on validation set
y_pred = best_model.predict(X_val)
print("\nClassification Report (Tuned XGBoost - Fast Mode):\n")
print(classification_report(le.inverse_transform(y_val),
                            le.inverse_transform(y_pred)))

# Confusion matrix
cm = confusion_matrix(le.inverse_transform(y_val),
                      le.inverse_transform(y_pred),
                      labels=le.classes_)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Tuned XGBoost (Fast Mode)")

# Show the plot
plt.show()

# Save the confusion matrix plot as PNG (for portfolio/CV)
plt.savefig("confusion_matrix.png")


# 10. Feature Importance
xgb.plot_importance(best_model, importance_type="weight", max_num_features=10)
plt.show()

# Save feature importance plot as PNG (for portfolio/CV)
plt.savefig("feature_importance.png")

# 11. Save the best model for future use
joblib.dump(best_model, "xgb_best_model.pkl")
print("\n💾 Model saved as xgb_best_model.pkl")

# --- HOW TO LOAD LATER (in another notebook or session) ---
# loaded_model = joblib.load("xgb_best_model.pkl")
# preds = loaded_model.predict(X_val)




