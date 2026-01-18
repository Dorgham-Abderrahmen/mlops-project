from fastapi import FastAPI
import uvicorn
import sys

print("=== D?MARRAGE API ===")
print(f"Python: {sys.version}")
print(f"FastAPI test")

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "API fonctionne!", "status": "online"}

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "test-api"}

@app.get("/info")
async def info():
    return {
        "python": sys.version.split()[0],
        "fastapi": "working",
        "endpoints": ["/", "/health", "/info"]
    }

if __name__ == "__main__":
    print("D?marrage sur http://0.0.0.0:8006")
    print("Appuyez sur Ctrl+C pour arr?ter")
    uvicorn.run(app, host="0.0.0.0", port=8006, log_level="info")