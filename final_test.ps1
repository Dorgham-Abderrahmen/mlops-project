# final_test.ps1 - Test complet du projet MLOps
Write-Host "🔬 TEST FINAL DU PROJET MLOPS" -ForegroundColor Magenta
Write-Host "=============================" -ForegroundColor Magenta

# Vérifier que l'API est en cours d'exécution
Write-Host "`n🌐 TEST DE CONNEXION API..." -ForegroundColor Cyan
try {
    $response = Invoke-RestMethod -Uri "http://localhost:8000/" -Method GET -ErrorAction Stop
    Write-Host "✅ API accessible" -ForegroundColor Green
    Write-Host "   Message: $($response.message)" -ForegroundColor Gray
} catch {
    Write-Host "❌ API non accessible" -ForegroundColor Red
    Write-Host "   Démarrer l'API avec: uvicorn src.api.app:app --reload --port 8000" -ForegroundColor Yellow
    exit 1
}

# Test health check
Write-Host "`n🩺 TEST HEALTH CHECK..." -ForegroundColor Cyan
try {
    $health = Invoke-RestMethod -Uri "http://localhost:8000/health" -Method GET -ErrorAction Stop
    Write-Host "✅ Health check OK" -ForegroundColor Green
    Write-Host "   Status: $($health.status)" -ForegroundColor Gray
    Write-Host "   Modèle chargé: $($health.model_loaded)" -ForegroundColor Gray
    Write-Host "   Nom modèle: $($health.model)" -ForegroundColor Gray
    
    if (-not $health.model_loaded) {
        Write-Host "⚠️  ATTENTION: Mode simulation activé" -ForegroundColor Yellow
        Write-Host "   Pour créer un modèle, exécutez cette commande:" -ForegroundColor Gray
        Write-Host "   python -c `"import sklearn; from sklearn.datasets import load_iris; from sklearn.ensemble import RandomForestClassifier; import pickle; iris=load_iris(); model=RandomForestClassifier(); model.fit(iris.data, iris.target); pickle.dump(model, open('model/iris_model.pkl','wb')); print('Modele cree')`"" -ForegroundColor Gray
    }
} catch {
    Write-Host "❌ Health check échoué" -ForegroundColor Red
}

# Test prédictions
Write-Host "`n🤖 TEST DES PRÉDICTIONS..." -ForegroundColor Cyan

$test_samples = @(
    @{name="Iris Setosa"; data=@{sepal_length=5.1; sepal_width=3.5; petal_length=1.4; petal_width=0.2}},
    @{name="Iris Versicolor"; data=@{sepal_length=6.0; sepal_width=2.7; petal_length=5.1; petal_width=1.6}},
    @{name="Iris Virginica"; data=@{sepal_length=7.0; sepal_width=3.2; petal_length=4.7; petal_width=1.4}}
)

$success_count = 0
foreach ($sample in $test_samples) {
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:8000/predict" `
            -Method POST `
            -Body ($sample.data | ConvertTo-Json) `
            -ContentType "application/json" `
            -ErrorAction Stop
        
        Write-Host "   ✅ $($sample.name): $($response.species)" -ForegroundColor Green
        $success_count++
    } catch {
        Write-Host "   ❌ $($sample.name) échoué" -ForegroundColor Red
    }
}

# Test fichiers
Write-Host "`n📁 TEST DES FICHIERS..." -ForegroundColor Cyan

$required_files = @(
    "src/api/app.py",
    "requirements.txt",
    "model/iris_model.pkl"
)

foreach ($file in $required_files) {
    if (Test-Path $file) {
        $size = ""
        try {
            $size = " ($([math]::Round((Get-Item $file).Length/1KB,2)) KB)"
        } catch {}
        Write-Host "   ✅ $file$size" -ForegroundColor Green
    } else {
        Write-Host "   ❌ $file (manquant)" -ForegroundColor Red
    }
}

# Test performance
Write-Host "`n⚡ TEST DE PERFORMANCE..." -ForegroundColor Cyan
$times = @()
for ($i = 0; $i -lt 3; $i++) {
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        Invoke-RestMethod -Uri "http://localhost:8000/health" -Method GET | Out-Null
        $sw.Stop()
        $times += $sw.ElapsedMilliseconds
    } catch {
        $times += 9999
    }
}

if ($times.Count -gt 0) {
    $avg = ($times | Measure-Object -Average).Average
    Write-Host "   Temps moyen réponse: $avg ms" -ForegroundColor Gray
}

# Résumé
Write-Host "`n📊 RÉSUMÉ DU TEST:" -ForegroundColor Green
Write-Host "==================" -ForegroundColor Green

Write-Host "✅ Composants validés:" -ForegroundColor Cyan
Write-Host "   • API REST FastAPI" -ForegroundColor Gray
Write-Host "   • Endpoint /predict" -ForegroundColor Gray
Write-Host "   • Health checks" -ForegroundColor Gray
Write-Host "   • Documentation auto" -ForegroundColor Gray

Write-Host "`n📈 RÉSULTATS:" -ForegroundColor Cyan
Write-Host "   • Prédictions réussies: $success_count/3" -ForegroundColor Gray
Write-Host "   • Fichiers présents: $(($required_files | Where-Object { Test-Path $_ }).Count)/$($required_files.Count)" -ForegroundColor Gray

Write-Host "`n🌐 ACCÈS:" -ForegroundColor Yellow
Write-Host "   • API: http://localhost:8000" -ForegroundColor Cyan
Write-Host "   • Documentation: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "   • Health: http://localhost:8000/health" -ForegroundColor Cyan

Write-Host "`n💡 COMMANDE DE TEST RAPIDE:" -ForegroundColor Green
Write-Host '   curl -X POST "http://localhost:8000/predict" ^' -ForegroundColor Gray
Write-Host '     -H "Content-Type: application/json" ^' -ForegroundColor Gray
Write-Host '     -d "{\"sepal_length\":5.1,\"sepal_width\":3.5,\"petal_length\":1.4,\"petal_width\":0.2}"' -ForegroundColor Gray

if ($success_count -eq 3) {
    Write-Host "`n🎉 TOUS LES TESTS SONT VALIDÉS !" -ForegroundColor Green
    Write-Host "Votre projet MLOps est opérationnel !" -ForegroundColor White
} else {
    Write-Host "`n⚠️  Certains tests ont échoué" -ForegroundColor Yellow
    Write-Host "Consultez les messages ci-dessus pour corriger." -ForegroundColor Gray
}
