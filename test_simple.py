print("Test Python simple")
print("=== D?marrage test ===")

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Test simple"}

@app.get("/health")
def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    print("D?marrage de l'API...")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
