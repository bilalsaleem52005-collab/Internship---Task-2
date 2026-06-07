Task 1-
# 🌸 Term Deposit Prediction – Training Pipeline

**3 models + SHAP explanations + automatic plots**

Predict if a bank client will subscribe to a term deposit.

---

## 📌 What & Why

- **Business need**: Banks run campaigns – only ~11.7% of clients subscribe.  
  Predicting “yes” helps target the right customers.

- **My solution**: Train & compare 3 models, explain predictions with SHAP.

---

## 🧠 What I Did (Step by Step)

| Step | Action | Why |
|------|--------|-----|
| 1 | Load & explore data | Check shape, missing values, class balance |
| 2 | Preprocess | One‑hot encode categories, standard scale numbers |
| 3 | Train 3 models | Logistic Regression (baseline), Random Forest, XGBoost |
| 4 | Evaluate | Accuracy, F1‑score, AUC‑ROC, confusion matrices, ROC curves |
| 5 | Compare | Table shows XGBoost performs best |
| 6 | Feature importance | Top 15 features for RF & XGB |
| 7 | SHAP explanations | Global summary + 5 waterfall plots (subsampled for speed) |
| 8 | Save best model | `best_model.pkl` for later use |

---

## 📊 Results (on test set)

| Model                 | Accuracy | F1‑Score | AUC‑ROC |
|-----------------------|----------|----------|---------|
| Logistic Regression   | 0.891    | 0.507    | 0.908   |
| Random Forest         | 0.902    | 0.580    | 0.931   |
| **XGBoost**           | **0.905**| **0.585**| **0.936** |

> **Top features** (from SHAP): `duration`, `poutcome_success`, `month_may`, `housing_yes`, `age`

---

## 🖥️ How to Run

### 1. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap joblib
