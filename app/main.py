"""
FastAPI app serving the resnet50 model for the web demo - upload a chest
X-ray, get back per-class probabilities using the Day 4 tuned thresholds.
"""
import json
import sys
from pathlib import Path

import torch
from fastapi import FastAPI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models import ResNetTransfer
from utils import CLASS_NAMES, METRICS_DIR, MODELS_DIR, get_device

app = FastAPI(title="Chest X-ray Classifier")

device = get_device()
model = ResNetTransfer(num_classes=len(CLASS_NAMES)).to(device)
checkpoint = torch.load(MODELS_DIR / "resnet50_best.pt", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

with open(METRICS_DIR / "resnet50_thresholds.json") as f:
    THRESHOLDS = json.load(f)


@app.get("/health")
def health():
    return {"status": "ok", "device": str(device)}
