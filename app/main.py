"""
FastAPI app serving the resnet50 model for the web demo - upload a chest
X-ray, get back per-class probabilities using the Day 4 tuned thresholds.
"""
import base64
import io
import json
import sys
from pathlib import Path

import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.loader import get_val_transforms
from models import ResNetTransfer
from utils import CLASS_NAMES, IMAGE_SIZE, METRICS_DIR, MODELS_DIR, get_device
from visualization.gradcam import GradCAM, overlay_heatmap

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Chest X-ray Classifier")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")

device = get_device()
model = ResNetTransfer(num_classes=len(CLASS_NAMES)).to(device)
checkpoint = torch.load(MODELS_DIR / "resnet50_best.pt", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

with open(METRICS_DIR / "resnet50_thresholds.json") as f:
    THRESHOLDS = json.load(f)

transform = get_val_transforms(IMAGE_SIZE)
cam = GradCAM(model, model.backbone.layer4)


@app.get("/health")
def health():
    return {"status": "ok", "device": str(device)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.sigmoid(logits).squeeze(0).cpu().tolist()

    results = {}
    for class_name, prob in zip(CLASS_NAMES, probs):
        results[class_name] = {
            "probability": round(prob, 4),
            "predicted": prob >= THRESHOLDS[class_name],
        }

    # explain whichever class the model is most confident about
    top_class_idx = max(range(len(CLASS_NAMES)), key=lambda i: probs[i])
    heatmap = cam(image_tensor, class_idx=top_class_idx)
    overlay = overlay_heatmap(image_tensor.squeeze(0), heatmap)

    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")
    gradcam_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "predictions": results,
        "gradcam": {
            "class": CLASS_NAMES[top_class_idx],
            "image_base64": gradcam_b64,
        },
    }
