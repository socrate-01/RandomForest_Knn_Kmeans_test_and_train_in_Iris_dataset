from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib  # Importez joblib pour sauvegarder le modèle

# Chargement du dataset Iris
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Normalisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Réduction de dimension avec PCA pour la visualisation (2D)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Création du modèle k-means (par exemple, pour 3 clusters)
kmeans_model = KMeans(n_clusters=3, random_state=42)

# Entraînement du modèle
kmeans_model.fit(X_scaled)

# Sauvegarde du modèle dans un fichier Joblib
joblib.dump(kmeans_model, 'kmeans_model.joblib')

# Prédictions sur les clusters
y_kmeans = kmeans_model.predict(X_scaled)
accuracy = accuracy_score(y, y_kmeans)
classification_report_result = classification_report(y, y_kmeans)

# Affichage des résultats
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report_result)
