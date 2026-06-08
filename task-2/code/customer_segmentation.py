"""
Customer Segmentation – Mall Customers Dataset
Unsupervised Learning: K-Means clustering + PCA/t-SNE visualisation + marketing strategies.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

# -------------------------------
# 1. PATHS & SETUP
# -------------------------------
def get_script_dir():
    return os.path.dirname(os.path.abspath(__file__))

SCRIPT_DIR = get_script_dir()
DATASET_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "dataset", "Mall_Customers.csv"))
OUTPUT_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "outputs"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

# -------------------------------
# 2. LOAD & EXPLORE DATA
# -------------------------------
print("="*60)
print(" CUSTOMER SEGMENTATION (MALL CUSTOMERS)")
print("="*60)

df = pd.read_csv(DATASET_PATH)
print(f"\nShape: {df.shape}")
print(df.head())
print("\nMissing values:\n", df.isnull().sum())
print("\nBasic statistics:\n", df.describe())

# Visualise distributions
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
sns.histplot(df['Age'], bins=20, kde=True, ax=axes[0,0])
axes[0,0].set_title('Age Distribution')
sns.histplot(df['Annual Income (k$)'], bins=20, kde=True, ax=axes[0,1])
axes[0,1].set_title('Annual Income (k$)')
sns.histplot(df['Spending Score (1-100)'], bins=20, kde=True, ax=axes[1,0])
axes[1,0].set_title('Spending Score')
sns.countplot(data=df, x='Genre', ax=axes[1,1])
axes[1,1].set_title('Gender')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'eda_distributions.png'))
plt.close()

# -------------------------------
# 3. SELECT FEATURES FOR CLUSTERING
# -------------------------------
# Use Annual Income and Spending Score (classic 2D segmentation)
X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

# Standardise (important for K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# 4. DETERMINE OPTIMAL K (ELBOW + SILHOUETTE)
# -------------------------------
inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Elbow plot
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method')

plt.subplot(1,2,2)
plt.plot(K_range, silhouette_scores, 'ro-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'elbow_silhouette.png'))
plt.close()

# Choose k=5 (clear elbow, decent silhouette)
optimal_k = 5
print(f"\nOptimal clusters (based on elbow & silhouette): {optimal_k}")

# -------------------------------
# 5. APPLY K-MEANS
# -------------------------------
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# -------------------------------
# 6. VISUALISE CLUSTERS (original 2D space)
# -------------------------------
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', 
                hue='Cluster', palette='Set2', s=80, alpha=0.7)
plt.title(f'Customer Segments (K-Means, k={optimal_k})')
plt.savefig(os.path.join(OUTPUT_DIR, 'clusters_kmeans.png'))
plt.close()

# -------------------------------
# 7. PCA & t-SNE VISUALISATIONS
# -------------------------------
# PCA (2 components)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='Set2', s=80, alpha=0.7)
plt.title('PCA projection of clusters')
plt.savefig(os.path.join(OUTPUT_DIR, 'clusters_pca.png'))
plt.close()

# t-SNE (2 components)
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)
df['TSNE1'] = X_tsne[:, 0]
df['TSNE2'] = X_tsne[:, 1]

plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='TSNE1', y='TSNE2', hue='Cluster', palette='Set2', s=80, alpha=0.7)
plt.title('t-SNE projection of clusters')
plt.savefig(os.path.join(OUTPUT_DIR, 'clusters_tsne.png'))
plt.close()

# -------------------------------
# 8. CLUSTER PROFILES & MARKETING STRATEGIES
# -------------------------------
cluster_summary = df.groupby('Cluster').agg({
    'Age': 'mean',
    'Annual Income (k$)': 'mean',
    'Spending Score (1-100)': 'mean',
    'Genre': lambda x: x.mode()[0]  # most frequent gender
}).round(2).rename(columns={'Gender': 'Dominant Gender'})

# Add descriptive labels
cluster_labels = {
    0: 'Standard (mid income, mid spending)',
    1: 'High income, high spending (VIP)',
    2: 'Low income, low spending (budget)',
    3: 'High income, low spending (price sensitive)',
    4: 'Low income, high spending (aspirational)'
}
cluster_summary['Segment'] = [cluster_labels.get(i, '') for i in cluster_summary.index]

# Marketing strategies
strategies = {
    0: 'Offer loyalty programs and cross‑sell mid‑range products.',
    1: 'Premium products, exclusive events, and personalized offers.',
    2: 'Discounts, bundle deals, and value‑oriented promotions.',
    3: 'Show value of premium products, free trials, and education.',
    4: 'Flexible payment plans, aspirational product visibility, and BNPL options.'
}
cluster_summary['Marketing Strategy'] = [strategies.get(i, '') for i in cluster_summary.index]

print("\n" + "="*60)
print(" CLUSTER PROFILES & MARKETING STRATEGIES")
print("="*60)
print(cluster_summary.to_string())

# Save to CSV
cluster_summary.to_csv(os.path.join(OUTPUT_DIR, 'cluster_profiles.csv'))
print(f"\n✅ Cluster profiles saved to: {os.path.join(OUTPUT_DIR, 'cluster_profiles.csv')}")

# -------------------------------
# 9. EXTRA: VISUALISE EACH CLUSTER'S INCOME VS SPENDING
# -------------------------------
plt.figure(figsize=(12,8))
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', 
                hue='Cluster', style='Cluster', palette='Set2', s=100, alpha=0.7)
# Add cluster centroids (in original scale)
centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centroids_original[:,0], centroids_original[:,1], 
            c='red', marker='X', s=200, label='Centroids')
plt.title(f'K-Means Clusters (k={optimal_k}) with Centroids')
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, 'clusters_with_centroids.png'))
plt.close()

print("\n🎉 Segmentation completed successfully!")
print(f"All outputs saved in: {OUTPUT_DIR}")