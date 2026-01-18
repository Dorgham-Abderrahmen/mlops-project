# test_ultra_simple.py
print('?tape 1: D?but du script')
from fastapi import FastAPI
print('?tape 2: FastAPI import?')
app = FastAPI()
print('?tape 3: App cr??e')
@app.get('/')
def home(): return {'ok': True}
print('?tape 4: Route d?finie')
print('?tape 5: D?marrage...')
import uvicorn
uvicorn.run(app, host='0.0.0.0', port=8002)
print('?tape 6: Apr?s run (ne devrait pas appara?tre)')