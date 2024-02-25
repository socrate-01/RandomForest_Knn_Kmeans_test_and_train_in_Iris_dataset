# train_random_forest_quantitative.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Charger le jeu de données Iris
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Créer un modèle Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)  # Exemple, ajustez selon vos besoins

# Pendant l'entraînement
print("Entraînement du modèle Random Forest en cours...")
rf_model.fit(X_train, y_train)
print("Entraînement terminé.")

# Évaluer la précision sur l'ensemble de test
y_pred = rf_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f'Précision sur l\'ensemble de test : {test_accuracy:.2f}')

# Rapport de classification pour l'évaluation quantitative détaillée
classification_rep = classification_report(y_test, y_pred, target_names=iris.target_names)
print("Rapport de classification :\n", classification_rep)
