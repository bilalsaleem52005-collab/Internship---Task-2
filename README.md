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

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 🛍️ Customer Segmentation – Mall Customers

**Unsupervised Learning: K‑Means + PCA/t‑SNE visualisation + actionable marketing strategies**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Overview

A mall wants to understand its customers better – not just who they are, but **how they behave** (spending vs income).  
Using **unsupervised learning**, we segment customers into meaningful groups and propose targeted marketing strategies.

**What this project does:**
- Exploratory Data Analysis (age, income, spending, gender)
- K‑Means clustering to find natural customer segments
- PCA & t‑SNE visualisations of the clusters
- Interpret each segment and recommend marketing actions

---

## 🧠 What I Did (Step by Step)

| Step | Action | Why |
|------|--------|-----|
| 1 | Load & explore data | Check shape, missing values, distributions |
| 2 | Select features | `Annual Income` + `Spending Score` – the two most behaviour‑rich columns |
| 3 | Scale features | StandardScaler – income (k$) and spending (0‑100) live on different scales |
| 4 | Find optimal k | Elbow method + silhouette score → k = 5 |
| 5 | Run K‑Means | Segment customers into 5 clusters |
| 6 | Visualise | Scatter plots (original space, PCA, t‑SNE) |
| 7 | Profile clusters | Average age, income, spending, dominant gender per cluster |
| 8 | Suggest strategies | Tailored marketing actions for each segment |
| 9 | Save outputs | All plots + CSV summary |

---

## 📊 Results & Insights

### Optimal Number of Clusters: k = 5

| Cluster | Age | Income (k$) | Spending Score | Dominant Gender | Segment | Marketing Strategy |
|---------|-----|-------------|----------------|-----------------|---------|---------------------|
| 0 | 42.7 | 55.3 | 49.5 | Female | Standard (mid income, mid spending) | Loyalty programs, cross‑sell |
| 1 | 41.1 | 88.2 | 17.1 | Male | High income, low spending (price sensitive) | Show value, free trials |
| 2 | 45.2 | 26.3 | 20.9 | Female | Low income, low spending (budget) | Discounts, bundle deals |
| 3 | 25.3 | 25.7 | 79.4 | Female | Low income, high spending (aspirational) | BNPL, aspirational ads |
| 4 | 32.7 | 86.5 | 82.1 | Female | High income, high spending (VIP) | Premium products, exclusive events |

### Visualisations (saved automatically)

- **EDA distributions** – age, income, spending, gender  
- **Elbow & silhouette** – proving k=5 is optimal  
- **Clusters (original space)** – income vs spending coloured by cluster  
- **PCA projection** – linear dimensionality reduction  
- **t‑SNE projection** – better separation of clusters  
- **Centroids** – red X marks the cluster centers  

---

## 🖥️ How to Run

### 1. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
