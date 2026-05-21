
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Metrics
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    roc_curve,
    auc,
    ConfusionMatrixDisplay
)

# SHAP
import shap

# =========================
# LOAD DATASET
# =========================

# Make sure bank-full.csv is in same folder

df = pd.read_csv('bank-full.csv', sep=';')

print(df.head())
print(df.info())
print(df.shape)

# =========================
# PREPROCESSING
# =========================

X = df.drop('y', axis=1)
y = df['y']

# Encode target variable
le = LabelEncoder()
y = le.fit_transform(y)

# Separate columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(exclude=['object']).columns

# Pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# LOGISTIC REGRESSION
# =========================

log_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

log_model.fit(X_train, y_train)

# Predictions

y_pred_log = log_model.predict(X_test)
y_prob_log = log_model.predict_proba(X_test)[:, 1]

# =========================
# RANDOM FOREST
# =========================

rf_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ))
])

rf_model.fit(X_train, y_train)

# Predictions

y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

# =========================
# PRINT RESULTS
# =========================

print("\n============================")
print("LOGISTIC REGRESSION RESULTS")
print("============================")

print("F1 Score:", f1_score(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))

print("\n=====================")
print("RANDOM FOREST RESULTS")
print("=====================")

print("F1 Score:", f1_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# =========================
# ROC DATA
# =========================

fpr_log, tpr_log, _ = roc_curve(y_test, y_prob_log)
roc_auc_log = auc(fpr_log, tpr_log)

fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

# =========================
# CREATE SINGLE PAGE DASHBOARD
# =========================

fig = plt.figure(figsize=(22, 16))

# ---------------------------------
# 1. Target Distribution
# ---------------------------------

ax1 = plt.subplot(2, 3, 1)
sns.countplot(x=df['y'], ax=ax1)
ax1.set_title('Target Variable Distribution')
ax1.set_xlabel('Subscribed')
ax1.set_ylabel('Count')

# ---------------------------------
# 2. Correlation Heatmap
# ---------------------------------

ax2 = plt.subplot(2, 3, 2)

numeric_df = df.select_dtypes(include=np.number)

sns.heatmap(
    numeric_df.corr(),
    cmap='coolwarm',
    annot=True,
    fmt='.2f',
    ax=ax2
)

ax2.set_title('Correlation Heatmap')

# ---------------------------------
# 3. Logistic Regression Confusion Matrix
# ---------------------------------

ax3 = plt.subplot(2, 3, 3)

cm_log = confusion_matrix(y_test, y_pred_log)

sns.heatmap(
    cm_log,
    annot=True,
    fmt='d',
    cmap='Blues',
    ax=ax3
)

ax3.set_title('Logistic Regression Confusion Matrix')
ax3.set_xlabel('Predicted')
ax3.set_ylabel('Actual')

# ---------------------------------
# 4. Random Forest Confusion Matrix
# ---------------------------------

ax4 = plt.subplot(2, 3, 4)

cm_rf = confusion_matrix(y_test, y_pred_rf)

sns.heatmap(
    cm_rf,
    annot=True,
    fmt='d',
    cmap='Greens',
    ax=ax4
)

ax4.set_title('Random Forest Confusion Matrix')
ax4.set_xlabel('Predicted')
ax4.set_ylabel('Actual')

# ---------------------------------
# 5. ROC Curve
# ---------------------------------

ax5 = plt.subplot(2, 3, 5)

ax5.plot(
    fpr_log,
    tpr_log,
    label=f'Logistic Regression AUC = {roc_auc_log:.2f}'
)

ax5.plot(
    fpr_rf,
    tpr_rf,
    label=f'Random Forest AUC = {roc_auc_rf:.2f}'
)

ax5.plot([0, 1], [0, 1], linestyle='--')

ax5.set_title('ROC Curve')
ax5.set_xlabel('False Positive Rate')
ax5.set_ylabel('True Positive Rate')
ax5.legend()

# ---------------------------------
# 6. F1 Score Comparison
# ---------------------------------

ax6 = plt.subplot(2, 3, 6)

models = ['Logistic Regression', 'Random Forest']
f1_scores = [
    f1_score(y_test, y_pred_log),
    f1_score(y_test, y_pred_rf)
]

ax6.bar(models, f1_scores)
ax6.set_ylim(0, 1)
ax6.set_title('F1 Score Comparison')
ax6.set_ylabel('F1 Score')

# ---------------------------------
# FINAL LAYOUT
# ---------------------------------

plt.suptitle(
    'Bank Marketing Machine Learning Dashboard',
    fontsize=22,
    fontweight='bold'
)

plt.tight_layout()
plt.show()

# =========================
# SHAP EXPLAINABILITY
# =========================

print("\nGenerating SHAP explanations...")

# Transform data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Train separate RF for SHAP
rf_explain = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf_explain.fit(X_train_processed, y_train)

# SHAP explainer
explainer = shap.TreeExplainer(rf_explain)

# SHAP values
shap_values = explainer.shap_values(X_test_processed[:5])

# Summary plot
shap.summary_plot(
    shap_values[1],
    X_test_processed[:5],
    show=True
)

print("\nTask Completed Successfully!")