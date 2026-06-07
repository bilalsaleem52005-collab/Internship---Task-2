"""
Term Deposit Subscription Prediction – Full Pipeline
3 Models: Logistic Regression, Random Forest, XGBoost
Includes evaluation, comparison, SHAP explainability (optimised), and auto-saving of plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
import sys
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, f1_score,
    roc_curve, roc_auc_score, accuracy_score
)

# -------------------------------
# 1. PATHS & SETUP
# -------------------------------
def get_script_dir():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

SCRIPT_DIR = get_script_dir()
DATASET_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "dataset", "bank-full.csv"))
SCREENSHOTS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "screenshots"))
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

# -------------------------------
# 2. LOAD & EXPLORE DATA
# -------------------------------
print("="*70)
print(" BANK MARKETING: TERM DEPOSIT PREDICTION (3 MODELS)")
print("="*70)

df = pd.read_csv(DATASET_PATH, sep=";")
df['y'] = df['y'].map({'yes': 1, 'no': 0})

X = df.drop('y', axis=1)
y = df['y']

categorical_cols = X.select_dtypes(include='object').columns.tolist()
numerical_cols = X.select_dtypes(exclude='object').columns.tolist()

print(f"\nDataset shape: {df.shape}")
print(f"Positive class rate: {y.mean():.2%}")
print(f"Categorical features: {len(categorical_cols)}")
print(f"Numerical features: {len(numerical_cols)}")

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
    ('num', StandardScaler(), numerical_cols)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# 3. DEFINE MODELS
# -------------------------------
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=200, learning_rate=0.1, random_state=42, eval_metric='logloss')
}

results = {}

# -------------------------------
# 4. TRAIN & EVALUATE
# -------------------------------
print("\n" + "="*70)
print(" MODEL TRAINING & EVALUATION")
print("="*70)

for name, clf in models.items():
    print(f"\n--- Training {name} ---")
    pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', clf)])
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    results[name] = {
        'pipeline': pipeline,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'acc': acc,
        'f1': f1,
        'auc': auc
    }
    
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=['No', 'Yes'], zero_division=0))

# -------------------------------
# 5. COMPARISON TABLE
# -------------------------------
comparison = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[m]['acc'] for m in results],
    'F1-Score': [results[m]['f1'] for m in results],
    'AUC-ROC': [results[m]['auc'] for m in results]
}).round(4)

print("\n" + "="*70)
print(" MODEL COMPARISON")
print("="*70)
print(comparison.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 2))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=comparison.values, colLabels=comparison.columns, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
plt.savefig(os.path.join(SCREENSHOTS_DIR, 'model_comparison_table.png'), bbox_inches='tight')
plt.close()

# -------------------------------
# 6. CONFUSION MATRICES
# -------------------------------
for name, res in results.items():
    cm = confusion_matrix(y_test, res['y_pred'])
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    plt.title(f'{name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(SCREENSHOTS_DIR, f'confusion_matrix_{name.replace(" ", "_")}.png'))
    plt.close()

# -------------------------------
# 7. ROC CURVES
# -------------------------------
plt.figure(figsize=(8,6))
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    plt.plot(fpr, tpr, label=f"{name} (AUC = {res['auc']:.3f})")
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves – All Models')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(os.path.join(SCREENSHOTS_DIR, 'roc_curves_all_models.png'))
plt.close()

# -------------------------------
# 8. FEATURE IMPORTANCE (RF & XGB)
# -------------------------------
preprocessor.fit(X_train)
cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
all_features = np.concatenate([cat_features, numerical_cols])

for name in ['Random Forest', 'XGBoost']:
    if name in results:
        clf = results[name]['pipeline'].named_steps['classifier']
        importances = clf.feature_importances_
        indices = np.argsort(importances)[-15:]
        plt.figure(figsize=(10,6))
        plt.barh(range(len(indices)), importances[indices], color='steelblue')
        plt.yticks(range(len(indices)), [all_features[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title(f'{name} - Top 15 Features')
        plt.tight_layout()
        plt.savefig(os.path.join(SCREENSHOTS_DIR, f'feature_importance_{name.replace(" ", "_")}.png'))
        plt.close()

# -------------------------------
# 9. SHAP EXPLANATIONS (optimised – subset)
# -------------------------------
print("\n" + "="*70)
print(" SHAP EXPLANATIONS (Random Forest – subset of 100 test samples)")
print("="*70)

X_train_encoded = preprocessor.transform(X_train)
X_test_encoded = preprocessor.transform(X_test)

sample_size = min(100, X_test_encoded.shape[0])
np.random.seed(42)
indices = np.random.choice(X_test_encoded.shape[0], sample_size, replace=False)
X_test_sample = X_test_encoded[indices]

print(f"Using {sample_size} random test samples for SHAP (total test: {X_test_encoded.shape[0]})")

rf_shap = RandomForestClassifier(n_estimators=200, random_state=42)
rf_shap.fit(X_train_encoded, y_train)

explainer = shap.TreeExplainer(rf_shap)
print("Computing SHAP values...")
shap_values = explainer.shap_values(X_test_sample)

shap_values_positive = shap_values[1] if isinstance(shap_values, list) else shap_values[:,:,1]

plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values_positive, X_test_sample, feature_names=all_features, show=False)
plt.title('SHAP Summary Plot – Impact on “Yes” Prediction')
plt.tight_layout()
plt.savefig(os.path.join(SCREENSHOTS_DIR, 'shap_summary.png'), bbox_inches='tight')
plt.close()

for i in range(min(5, sample_size)):
    plt.figure()
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values_positive[i],
            base_values=explainer.expected_value[1],
            data=X_test_sample[i],
            feature_names=all_features
        ),
        show=False
    )
    plt.title(f"Prediction Explanation – Test Customer #{i+1}")
    plt.tight_layout()
    plt.savefig(os.path.join(SCREENSHOTS_DIR, f'shap_prediction_{i+1}.png'), bbox_inches='tight')
    plt.close()

print(f"✅ SHAP plots saved in: {SCREENSHOTS_DIR}")

# -------------------------------
# 10. SAVE BEST MODEL
# -------------------------------
best_model_name = comparison.loc[comparison['F1-Score'].idxmax(), 'Model']
best_pipeline = results[best_model_name]['pipeline']
models_dir = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "models"))
os.makedirs(models_dir, exist_ok=True)
import joblib
joblib.dump(best_pipeline, os.path.join(models_dir, "best_model.pkl"))
print(f"\n🏆 Best model: {best_model_name} (F1 = {comparison['F1-Score'].max():.4f})")
print(f"✅ Model saved to: {os.path.join(models_dir, 'best_model.pkl')}")

print("\n🎉 Script completed successfully!")