Write-Host 'Création modèle Iris...' -ForegroundColor Green
python -c \"
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Créer dossier
os.makedirs('model', exist_ok=True)

# Créer modèle
iris = load_iris()
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(iris.data, iris.target)

# Sauvegarder
with open('model/iris_model.pkl', 'wb') as f:
    pickle.dump(model, f)
    
print(f'✅ Modèle sauvegardé dans model/iris_model.pkl')
print(f'📊 Accuracy: {model.score(iris.data, iris.target):.1%}')
\"
