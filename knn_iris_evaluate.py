import joblib
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

iris = load_iris()

def load_new_data(file_path):
    new_data = pd.read_csv(file_path, header=None, names=iris.feature_names + ['Class'])
    label_encoder = LabelEncoder()
    new_data['Class'] = label_encoder.fit_transform(new_data['Class'])
    return new_data

# Charger le modèle à partir du fichier Joblib.
loaded_model = joblib.load('knn_model.joblib')

# Charger de nouvelles données de test.
new_data = load_new_data('testing.data')

# Faire des prédictions sur les nouvelles données.
predictions = loaded_model.predict(new_data.drop('Class', axis=1))

# Évaluation quantitative
accuracy = accuracy_score(new_data['Class'], predictions)
print(f'Précision sur l\'ensemble de test : {accuracy:.2f}')
print('Évaluation quantitative (rapport de classification) :\n', classification_report(new_data['Class'], predictions))

# Évaluation qualitative
# Afficher la matrice de confusion
cm = confusion_matrix(new_data['Class'], predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title('Matrice de Confusion')
plt.xlabel('Prédictions')
plt.ylabel('Vraies étiquettes')
plt.show()

# Imprimer les prédictions
print("Prédictions sur les nouvelles données :")
print(predictions)
