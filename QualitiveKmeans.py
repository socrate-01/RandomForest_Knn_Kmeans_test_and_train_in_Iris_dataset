import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# Chargement du dataset Iris
iris = load_iris()
X = iris.data

# Normalisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Réduction de dimension avec PCA pour la visualisation en 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Entraînement du modèle K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

# Visualisation des clusters
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans, cmap='viridis')
plt.title('Visualisation des Clusters avec K-Means sur Iris')
plt.xlabel('Composante Principale 1')
plt.ylabel('Composante Principale 2')
plt.show()
