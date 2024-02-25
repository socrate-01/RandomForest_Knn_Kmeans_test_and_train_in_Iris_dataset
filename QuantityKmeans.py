import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.datasets import load_iris

# Chargement du dataset Iris
iris = load_iris()
X = iris.data

# Normalisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Entraînement du modèle K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

# Évaluation quantitative avec la silhouette
silhouette_avg = silhouette_score(X_scaled, y_kmeans)
print(f"Score de silhouette : {silhouette_avg}")
