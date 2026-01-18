# train_mlflow.py
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

def train_with_mlflow(experiment_name="Iris_Classification", run_name=None):
    """Entraînement avec tracking MLflow"""
    
    # Configuration MLflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=run_name) as run:
        # 1. Chargement des données
        iris = load_iris()
        X, y = iris.data, iris.target
        
        # 2. Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 3. Paramètres (à varier entre les runs)
        params = {
            "n_estimators": 100,
            "max_depth": 5,
            "random_state": 42,
            "test_size": 0.2
        }
        
        # Log des paramètres
        mlflow.log_params(params)
        
        # 4. Entraînement
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        # 5. Évaluation
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log des métriques
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("train_samples", len(X_train))
        mlflow.log_metric("test_samples", len(X_test))
        
        # 6. Classification report (artefact)
        report = classification_report(y_test, y_pred, target_names=iris.target_names, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_path = "classification_report.csv"
        report_df.to_csv(report_path)
        mlflow.log_artifact(report_path)
        
        # 7. Matrice de confusion (artefact)
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=iris.target_names, 
                    yticklabels=iris.target_names)
        plt.title("Matrice de confusion")
        plt.ylabel('Vrai')
        plt.xlabel('Prédit')
        
        conf_matrix_path = "confusion_matrix.png"
        plt.savefig(conf_matrix_path)
        plt.close()
        mlflow.log_artifact(conf_matrix_path)
        
        # 8. Importance des features (artefact)
        feature_importance = pd.DataFrame({
            'feature': iris.feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=feature_importance, x='importance', y='feature')
        plt.title("Importance des features")
        
        importance_path = "feature_importance.png"
        plt.savefig(importance_path)
        plt.close()
        mlflow.log_artifact(importance_path)
        
        # 9. Sauvegarde du modèle
        model_path = "model/iris_model.pkl"
        import joblib
        joblib.dump(model, model_path)
        
        # Log du modèle dans MLflow
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_artifact(model_path)
        
        # 10. Tags
        mlflow.set_tag("dataset", "iris")
        mlflow.set_tag("model_type", "RandomForest")
        mlflow.set_tag("author", "MLOps Project")
        
        print(f"✅ Run MLflow terminé: {run.info.run_id}")
        print(f"📊 Accuracy: {accuracy:.2%}")
        
        # Nettoyage
        for file in [report_path, conf_matrix_path, importance_path]:
            if os.path.exists(file):
                os.remove(file)
        
        return run.info.run_id

if __name__ == "__main__":
    # Run 1: Baseline
    print("🏃‍♂️ Run 1: Baseline")
    run1 = train_with_mlflow(run_name="baseline")
    
    # Run 2: Variation (plus d'arbres)
    print("\n🏃‍♂️ Run 2: Variation (n_estimators=200)")
    mlflow.start_run(run_name="variation_200_trees")
    mlflow.log_param("n_estimators", 200)
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")
    mlflow.end_run()
    
    print("\n✅ Deux runs MLflow terminés!")
    print(f"Accès: http://localhost:5000")
