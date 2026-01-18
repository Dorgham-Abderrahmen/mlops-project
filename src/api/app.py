# src/api/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import joblib
import numpy as np
import os
from typing import Optional
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="API de prédiction Iris")

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float 
    petal_length: float
    petal_width: float

# ===== CHARGEMENT DU MODÈLE =====
MODEL_PATHS = [
    "model/iris_model.pkl",
    "model/model.pkl", 
    "models/iris_model.joblib",
    "src/models/model.pkl",
    "../model.pkl",
    "iris_model.pkl"
]

model = None
model_name = "Simulation"
model_loaded = False

def load_model():
    """Cherche et charge le modèle automatiquement"""
    global model, model_name, model_loaded
    
    for model_path in MODEL_PATHS:
        try:
            logger.info(f"Tentative de chargement depuis: {model_path}")
            
            if os.path.exists(model_path):
                # Essayer différents formats
                if model_path.endswith('.pkl'):
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                elif model_path.endswith('.joblib'):
                    model = joblib.load(model_path)
                else:
                    continue
                    
                model_name = os.path.basename(model_path)
                model_loaded = True
                logger.info(f"✅ Modèle chargé: {model_name}")
                logger.info(f"   Type: {type(model)}")
                
                # Tester le modèle
                test_features = [[5.1, 3.5, 1.4, 0.2]]
                prediction = model.predict(test_features)
                logger.info(f"   Test prédiction: {prediction}")
                return True
                
        except Exception as e:
            logger.warning(f"❌ Échec chargement {model_path}: {e}")
            continue
    
    logger.warning("⚠️  Aucun modèle trouvé - Mode simulation activé")
    return False

# Charger le modèle au démarrage
load_model()

@app.get("/")
def home():
    return {
        "message": "API de prédiction Iris",
        "model_loaded": model_loaded,
        "model": model_name,
        "mode": "simulation" if not model_loaded else "production"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "model": model_name
    }

@app.get("/model-info")
def get_model_info():
    """Informations détaillées sur le modèle"""
    if model_loaded:
        return {
            "name": model_name,
            "type": str(type(model)),
            "features": getattr(model, 'n_features_in_', 'unknown'),
            "parameters": getattr(model, 'get_params', lambda: {})()
        }
    return {"message": "Mode simulation - Pas de modèle chargé"}

@app.post("/predict")
def predict(features: IrisFeatures):
    """Faire une prédiction"""
    
    # Préparer les données
    X = [[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width
    ]]
    
    # Utiliser le modèle si disponible
    if model_loaded:
        try:
            prediction = model.predict(X)[0]
            probabilities = None
            
            # Essayer d'obtenir les probabilités
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X)[0].tolist()
            
            species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
            species = species_map.get(int(prediction), "unknown")
            
            return {
                "prediction": int(prediction),
                "species": species,
                "probabilities": probabilities,
                "model": model_name,
                "mode": "production",
                "features": features.dict()
            }
            
        except Exception as e:
            logger.error(f"Erreur prédiction: {e}")
            # Fallback sur la simulation
            pass
    
    # MODE SIMULATION (fallback)
    petal_area = features.petal_length * features.petal_width
    
    if petal_area < 2.0:
        prediction = 0
        species = "setosa"
        confidence = 0.95
    elif petal_area < 7.0:
        prediction = 1  
        species = "versicolor"
        confidence = 0.85
    else:
        prediction = 2
        species = "virginica"
        confidence = 0.90
    
    return {
        "prediction": prediction,
        "species": species,
        "confidence": confidence,
        "model": "simulation",
        "mode": "simulation",
        "petal_area": petal_area,
        "features": features.dict()
    }

@app.post("/train")
def train_model():
    """Endpoint pour entraîner un modèle (exemple)"""
    # Ceci est un exemple - adaptez-le à votre code d'entraînement
    try:
        from sklearn.datasets import load_iris
        from sklearn.ensemble import RandomForestClassifier
        
        # Charger les données
        iris = load_iris()
        X, y = iris.data, iris.target
        
        # Entraîner un modèle simple
        new_model = RandomForestClassifier(n_estimators=100, random_state=42)
        new_model.fit(X, y)
        
        # Sauvegarder le modèle
        model_path = "model/iris_trained.pkl"
        os.makedirs("model", exist_ok=True)
        
        with open(model_path, 'wb') as f:
            pickle.dump(new_model, f)
        
        # Recharger le modèle
        load_model()
        
        return {
            "message": "Modèle entraîné et sauvegardé",
            "model_path": model_path,
            "accuracy": new_model.score(X, y)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur entraînement: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)