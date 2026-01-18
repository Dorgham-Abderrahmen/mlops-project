# scripts/download_data.py
import pandas as pd
from sklearn.datasets import load_iris
import os

# Créer le dossier data
os.makedirs('data/raw', exist_ok=True)

# Charger le dataset Iris
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['target_name'] = iris.target_names[iris.target]

# Sauvegarder
df.to_csv('data/raw/iris.csv', index=False)
print("Dataset saved to data/raw/iris.csv")
print(f"Shape: {df.shape}")