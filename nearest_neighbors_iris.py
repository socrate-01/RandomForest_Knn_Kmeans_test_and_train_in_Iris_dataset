# Import des bibliothèques nécessaires
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Chargement du dataset Iris
from sklearn.datasets import load_iris
iris = load_iris()

# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Création du modèle k-NN
knn_model = KNeighborsClassifier(n_neighbors=3)  # Vous pouvez ajuster le nombre de voisins selon vos besoins

# Entraînement du modèle
knn_model.fit(X_train, y_train)

# Prédictions sur l'ensemble de test
y_pred = knn_model.predict(X_test)

# Évaluation du modèle
accuracy = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)

# Affichage des résultats
print(f"Accuracy: {accuracy}")
print("Classification Report:\n", classification_report_str)
