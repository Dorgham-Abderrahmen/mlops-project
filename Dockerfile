# Dockerfile multi-stage
FROM python:3.9-slim AS builder

WORKDIR /app

# Copier les requirements
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Runtime image
FROM python:3.9-slim

WORKDIR /app

# Copier les dépendances depuis builder
COPY --from=builder /root/.local /root/.local

# Copier l'application
COPY . .

# Installer les dépendances système si nécessaire
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Créer les dossiers nécessaires
RUN mkdir -p /app/model /app/logs /app/data /app/mlruns

# Variables d'environnement
ENV PATH=/root/.local/bin:
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Exposition des ports
EXPOSE 8000

# Commande de démarrage
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
