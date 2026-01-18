# setup_data.py - Script pour récupérer les données avec DVC
import os
import subprocess
import sys

def setup_data():
    \"\"\"Récupérer les données avec DVC\"\"\"
    print(\"📥 Récupération des données avec DVC...\")
    
    # Vérifier si DVC est installé
    try:
        import dvc
    except ImportError:
        print(\"❌ DVC non installé. Installation...\")
        subprocess.check_call([sys.executable, \"-m\", \"pip\", \"install\", \"dvc\"])
    
    # Vérifier si les données existent déjà
    if os.path.exists(\"data/raw/iris.csv\"):
        print(\"✅ Données déjà présentes\")
        return
    
    # Initialiser DVC si nécessaire
    if not os.path.exists(\".dvc\"):
        print(\"Initialisation DVC...\")
        subprocess.run([\"dvc\", \"init\"], capture_output=True)
    
    # Récupérer les données
    print(\"Téléchargement des données...\")
    result = subprocess.run([\"dvc\", \"pull\"], capture_output=True, text=True)
    
    if result.returncode == 0:
        print(\"✅ Données récupérées avec succès\")
        print(f\"Emplacement: {os.path.abspath('data/raw/iris.csv')}\")
    else:
        print(\"❌ Erreur lors de la récupération:\")
        print(result.stderr)

if __name__ == \"__main__\":
    setup_data()
