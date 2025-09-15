import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load a sample (so it runs faster in Codespaces)
train = pd.read_csv("train.csv").sample(10000, random_state=42)

# -----------------------------
# TARGET DISTRIBUTION
# -----------------------------
plt.figure(figsize=(6,4))
train["Credit_Score"].value_counts().plot(
    kind="bar", color=["skyblue", "salmon", "lightgreen"]
)
plt.title("Distribution of Credit Scores")
plt.xlabel("Credit Score")
plt.ylabel("Count")
plt.show()

# -----------------------------
# INCOME VS CREDIT SCORE
# -----------------------------
plt.figure(figsize=(6,4))
sns.boxplot(x="Credit_Score", y="Annual_Income", data=train)
plt.title("Annual Income vs Credit Score")
plt.show()

# -----------------------------
# AGE DISTRIBUTION BY CREDIT SCORE
# -----------------------------
plt.figure(figsize=(6,4))
sns.histplot(data=train, x="Age", hue="Credit_Score", kde=True, element="step")
plt.title("Age Distribution by Credit Score")
plt.show()

# -----------------------------
# FEATURE CORRELATIONS (numeric only)
# -----------------------------
plt.figure(figsize=(10,6))
sns.heatmap(train.corr(numeric_only=True), cmap="coolwarm", annot=False)
plt.title("Feature Correlation Heatmap")
plt.show()
