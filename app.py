from fastapi import FastAPI, File, UploadFile
from fastapi.responses import RedirectResponse
from PIL import Image
import numpy as np
import io
from model import model_instance

app = FastAPI()

@app.get("/")
def root():
    return RedirectResponse(url="/index.html")

@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": model_instance.model_loaded}

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np = np.array(image)
        return model_instance.predict(img_np)
    except Exception as e:
        return {"ok": False, "error": str(e)}