# random_forest_iris.py

# Étape 1 : Importer les bibliothèques nécessaires
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Étape 2 : Charger le jeu de données Iris
iris_data = pd.read_csv('bezdekiris.data', header=None, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target'])

# Étape 3 : Diviser le jeu de données en ensembles d'entraînement et de test
X = iris_data.drop('target', axis=1)
y = iris_data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Étape 4 : Créer et entraîner le modèle Random Forest
random_forest_model = RandomForestClassifier(random_state=42)
random_forest_model.fit(X_train, y_train)

# Étape 5 : Faire des prédictions sur l'ensemble de test
y_pred = random_forest_model.predict(X_test)

# Étape 6 : Évaluer le modèle
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Étape 7 : Afficher les résultats
print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(classification_rep)
