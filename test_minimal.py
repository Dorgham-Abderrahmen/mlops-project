print("=== TEST MINIMAL ===")

# V?rifier les imports
try:
    from fastapi import FastAPI
    print("? FastAPI import?")
except Exception as e:
    print(f"? Erreur FastAPI: {e}")

try:
    import uvicorn
    print("? uvicorn import?")
except Exception as e:
    print(f"? Erreur uvicorn: {e}")

# Cr?er app
app = FastAPI()

@app.get("/")
def root():
    return {"test": "ok"}

print("? Application cr??e")

# D?marrer
print("D?marrage sur port 8001...")
import uvicorn
uvicorn.run(app, host="0.0.0.0", port=8001)
