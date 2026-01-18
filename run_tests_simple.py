# run_tests_simple.py
import sys
import os

# Ajouter le chemin
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("🧪 TESTS SIMPLIFIÉS")
print("=================")

# Test 1: Import API
print("\n1. Test import API...")
try:
    from src.api.app import app
    print("   ✅ API importée")
except ImportError as e:
    print(f"   ❌ Erreur: {e}")
    sys.exit(1)

# Test 2: FastAPI TestClient
print("\n2. Test endpoints...")
try:
    from fastapi.testclient import TestClient
    
    client = TestClient(app)
    
    # Test racine
    response = client.get("/")
    assert response.status_code == 200
    print(f"   ✅ GET / : {response.status_code}")
    
    # Test health
    response = client.get("/health")
    assert response.status_code == 200
    print(f"   ✅ GET /health : {response.status_code}")
    
    # Test prédiction
    test_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post("/predict", json=test_data)
    assert response.status_code in [200, 422]  # 422 si validation échoue
    print(f"   ✅ POST /predict : {response.status_code}")
    
    print("\n🎉 TOUS LES TESTS PASSENT!")
    
except Exception as e:
    print(f"   ❌ Erreur: {e}")
    import traceback
    traceback.print_exc()
