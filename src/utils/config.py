"""
Configuration and constants for the Chest X-ray Multi-label Classification project.
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
METRICS_DIR = RESULTS_DIR / "metrics"
VIZ_DIR = RESULTS_DIR / "visualizations"

# Create directories if they don't exist
for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, METADATA_DIR, MODELS_DIR, METRICS_DIR, VIZ_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Dataset configuration
CLASS_NAMES = ["No Finding", "Atelectasis", "Cardiomegaly", "Effusion", "Pneumonia"]
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# Image configuration
IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]  # ImageNet stats
STD = [0.229, 0.224, 0.225]

# Training configuration
BATCH_SIZE = 32
VAL_BATCH_SIZE = 64
NUM_WORKERS = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 30
PATIENCE = 5  # for early stopping

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2

# Random seed for reproducibility
SEED = 42

# Device
DEVICE = "cuda"  # Will be set dynamically
