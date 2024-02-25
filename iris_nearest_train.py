from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import joblib

# Étape 1: Importez les bibliothèques nécessaires.
# Étape 2: Chargez le jeu de données Iris.
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Étape 3: Créez et entraînez le modèle k-NN.
knn_model = KNeighborsClassifier(n_neighbors=3)
print("Entraînement du modèle Nearest Neighbors (kNN) en cours...")
knn_model.fit(X_train, y_train)
print("Entraînement terminé.")

# Étape 4: Évaluez la précision sur l'ensemble d'entraînement.
train_accuracy = knn_model.score(X_train, y_train)
print(f'Précision sur l\'ensemble d\'entraînement : {train_accuracy:.2f}')

# Sauvegardez le modèle dans un fichier Joblib.
joblib.dump(knn_model, 'knn_model.joblib')
