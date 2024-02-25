from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import joblib

# Étape1: Importez les bibliothèques nécessaires.
# Étape2: Chargez le jeu de données Iris.
iris = load_iris()
X_train, _, y_train, _ = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Étape3: Créez et entraînez le modèle Random Forest.
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Étape4: Évaluez la précision sur l'ensemble d'entraînement.
train_accuracy = rf_model.score(X_train, y_train)
print(f'Précision sur l\'ensemble d\'entraînement : {train_accuracy:.2f}')

# Sauvegardez le modèle dans un fichier Joblib.
joblib.dump(rf_model, 'random_forest_model.joblib')
