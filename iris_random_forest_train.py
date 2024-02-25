# train_random_forest.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Charger le jeu de données Iris
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Créer un modèle Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Entraîner le modèle sur l'ensemble d'entraînement
rf_model.fit(X_train, y_train)

# Évaluer la précision sur l'ensemble d'entraînement
train_accuracy = rf_model.score(X_train, y_train)
print(f'Précision sur l\'ensemble d\'entraînement : {train_accuracy:.2f}')

# Prédire les étiquettes sur l'ensemble de test
rf_pred = rf_model.predict(X_test)

# Calculer la précision sur l'ensemble de test
test_accuracy = accuracy_score(y_test, rf_pred)
print(f'Précision sur l\'ensemble de test : {test_accuracy:.2f}')

# Afficher l'importance des caractéristiques
feature_importances = rf_model.feature_importances_
print('Importance des caractéristiques :', feature_importances)

# Visualiser un arbre de la forêt (le premier arbre)
plt.figure(figsize=(10, 6))
plot_tree(rf_model.estimators_[0], filled=True, feature_names=iris.feature_names, class_names=iris.target_names, rounded=True)
plt.show()
