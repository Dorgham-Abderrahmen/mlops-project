# pipeline_zenml.py
import pandas as pd
import numpy as np
from typing import Tuple, Annotated
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from zenml import pipeline, step
from zenml.logger import get_logger

logger = get_logger(__name__)

@step
def load_data() -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"]
]:
    """Étape de chargement et préparation des données"""
    logger.info("Chargement des données Iris...")
    iris = load_iris()
    
    # Convertir en DataFrame pour meilleure traçabilité
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="target")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    logger.info(f"Données chargées: {X.shape[0]} échantillons")
    logger.info(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test

@step
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100
) -> RandomForestClassifier:
    """Étape d'entraînement du modèle"""
    logger.info(f"Entraînement RandomForest avec {n_estimators} arbres...")
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    logger.info("Modèle entraîné")
    
    return model

@step
def evaluate_model(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> float:
    """Étape d'évaluation du modèle"""
    logger.info("Évaluation du modèle...")
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    logger.info(f"Accuracy: {accuracy:.2%}")
    
    return accuracy

@pipeline
def iris_pipeline(n_estimators: int = 100):
    """Pipeline complet Iris"""
    # Définir le flux
    X_train, X_test, y_train, y_test = load_data()
    model = train_model(X_train, y_train, n_estimators)
    accuracy = evaluate_model(model, X_test, y_test)
    
    return accuracy

if __name__ == "__main__":
    # Exécution 1: Baseline
    print("🔧 Exécution 1: Baseline (n_estimators=100)")
    run1 = iris_pipeline(n_estimators=100)
    
    # Exécution 2: Variation
    print("\n🔧 Exécution 2: Variation (n_estimators=200)")
    run2 = iris_pipeline(n_estimators=200)
    
    print("\n✅ Pipeline ZenML exécuté deux fois!")
    print("Voir les runs: zenml pipeline runs list")
