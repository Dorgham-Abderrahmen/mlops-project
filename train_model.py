import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Charger les données
iris = load_iris()
X = iris.data
y = iris.target

# Entraîner un modèle simple
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Sauvegarder
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")

print("✅ Modèle entraîné et sauvegardé dans models/model.pkl")
print(f"Accuracy (entraînement): {model.score(X, y):.2%}")
