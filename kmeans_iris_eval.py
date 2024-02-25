import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

iris = load_iris()

# Charger le modèle KMeans à partir du fichier Joblib
kmeans_model = joblib.load('kmeans_model.joblib')

# Charger de nouvelles données de test (remplacez 'testing.data' par le nom de votre fichier de données)
new_data = pd.read_csv('testing.data', header=None, names=iris.feature_names + ['Class'])
label_encoder = LabelEncoder()
new_data['Class'] = label_encoder.fit_transform(new_data['Class'])

# Normalisation des nouvelles données avec le même scaler utilisé lors de l'entraînement
X = iris.data
scaler = StandardScaler()
scaler.fit(X)
new_data_scaled = scaler.transform(new_data.drop('Class', axis=1))

# Réduction de dimension avec PCA pour la visualisation (2D)
pca = PCA(n_components=2)
pca.fit(X)
new_data_pca = pca.transform(new_data_scaled)

# Prédictions sur les clusters
predictions = kmeans_model.predict(new_data_scaled)

# Évaluation quantitative
accuracy = accuracy_score(new_data['Class'], predictions)
print(f'Précision sur l\'ensemble de test : {accuracy:.2f}')
print('Évaluation quantitative (rapport de classification) :\n', classification_report(new_data['Class'], predictions))

# Évaluation qualitative
# Afficher la matrice de confusion
plt.figure(figsize=(10, 8))
sns.scatterplot(x=new_data_pca[:, 0], y=new_data_pca[:, 1], hue=predictions, palette='viridis', s=100)
plt.title('Clusters prédits par KMeans sur les nouvelles données')
plt.xlabel('Composante principale 1')
plt.ylabel('Composante principale 2')
plt.legend(title='Cluster')
plt.show()

# Imprimer les prédictions
print("Prédictions sur les nouvelles données :")
print(predictions)
