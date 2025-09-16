# Copilot Instructions for MLM-credit

## Project Overview
- **Purpose:** Predict credit scores using machine learning (XGBoost) on tabular customer data.
- **Main workflow:** Data loading, cleaning, feature engineering, model training/tuning, evaluation, explainability (SHAP), and test set prediction for submission.
- **Key notebook:** `archive/credit_score_notebook.ipynb` contains the end-to-end workflow and is the best reference for project logic and conventions.

## Directory Structure
- `archive/` — All data, models, scripts, and the main notebook are here.
  - `train.csv`, `test.csv` — Main datasets.
  - `credit_score_notebook.ipynb` — Main analysis and modeling notebook.
  - `credit_score_model.py` — (If present) Standalone model code.
  - `xgb_best_model.pkl`, `credit_score_model.pkl` — Saved model artifacts.
  - `*.pkl`, `*.json` — Encoders and model files.
  - `eda.py`, `get_dummies.py` — Utility/data prep scripts.
  - `submission.csv` — Output predictions for competition/test set.

## Data & Modeling Patterns
- **Data cleaning:** Drop columns `ID`, `Customer_ID`, `Name`, `SSN`. Fill missing values with column means (numeric only).
- **Categorical encoding:** Use `pd.get_dummies(..., drop_first=True)` for categorical features.
- **Label encoding:** Map target labels to integers using a dictionary (`label_map`).
- **Model:** XGBoost (`xgboost.XGBClassifier`) with tuned hyperparameters (see notebook for details).
- **Evaluation:** Use `classification_report`, `confusion_matrix`, and SHAP for explainability.
- **Submission:** Ensure test set columns match training features (use `reindex(columns=X.columns, fill_value=0)`).

## Developer Workflows
- **Run notebook:** Use Jupyter or VS Code to execute `archive/credit_score_notebook.ipynb` step by step.
- **Model training:** All training and evaluation is done in the notebook. Artifacts are saved with `joblib.dump`.
- **Prediction:** Run the last notebook cell to generate `submission.csv`.
- **Dependencies:** Install with `pip install -r requirements.txt` (if present) or manually: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `shap`, `matplotlib`, `seaborn`, `joblib`.

## Conventions & Tips
- **Keep all new scripts and data in `archive/`** to avoid cluttering the root.
- **Notebook is source of truth** for data prep, modeling, and evaluation logic.
- **For new features:** Prototype in the notebook, then refactor to scripts if needed.
- **Model/encoder files:** Use `.pkl` for joblib/pickle, `.json` for XGBoost models.
- **No custom build/test system:** All logic is in Python scripts and notebooks.

## Integration Points
- **No external APIs/services** — all data is local.
- **No web UI/backend** — this is a pure data science/ML workflow.

## Example: Adding a New Model
1. Add new code cells to the notebook for your model.
2. Save the model artifact in `archive/`.
3. Update the prediction cell to use your new model.

---

For questions, see the notebook or ask the project owner.
