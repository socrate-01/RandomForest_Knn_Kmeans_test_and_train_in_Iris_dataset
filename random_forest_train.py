# random_forest_train.py
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import joblib

def train_random_forest_model(train_data_path, model_save_path):
    # Charger les données d'entraînement depuis le fichier CSV
    train_data = pd.read_csv(train_data_path)
    X_train = train_data.drop('target', axis=1)
    y_train = train_data['target']

    # Initialiser le modèle Random Forest
    random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Entraîner le modèle
    random_forest_model.fit(X_train, y_train)

    # Sauvegarder le modèle entraîné
    joblib.dump(random_forest_model, model_save_path)

if __name__ == "__main__":
    # Spécifier le chemin vers le fichier d'entraînement et le chemin pour sauvegarder le modèle
    train_data_path = 'bezdekIris.data'
    model_save_path = 'random_forest_model.joblib'

    # Entraîner le modèle et le sauvegarder
    train_random_forest_model(train_data_path, model_save_path)
