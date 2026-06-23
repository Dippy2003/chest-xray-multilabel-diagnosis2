"""
Tune per-class decision thresholds on the validation set for a trained model,
then re-evaluate the test set with those thresholds instead of the flat 0.5
default. AUC-ROC doesn't change (it's threshold-independent) but F1 should
improve a lot for the rare classes.

Usage:
    python scripts/tune_thresholds.py --model resnet50
    python scripts/tune_thresholds.py --model efficientnet_b0
    python scripts/tune_thresholds.py --model baseline_cnn
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data import create_dataloaders
from evaluation import format_metrics
from evaluation.metrics import compute_metrics
from evaluation.threshold_tuning import find_best_thresholds
from models import BaselineCNN, EfficientNetTransfer, ResNetTransfer
from utils import CLASS_NAMES, IMAGE_SIZE, METADATA_DIR, METRICS_DIR, MODELS_DIR, RAW_DATA_DIR, get_device

MODEL_BUILDERS = {
    "baseline_cnn": lambda n: BaselineCNN(num_classes=n),
    "resnet50": lambda n: ResNetTransfer(num_classes=n),
    "efficientnet_b0": lambda n: EfficientNetTransfer(num_classes=n),
}


@torch.no_grad()
def get_logits(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        all_logits.append(logits.cpu())
        all_labels.append(labels)
    return torch.cat(all_labels), torch.cat(all_logits)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_BUILDERS.keys(), required=True)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    dataset_df = pd.read_csv(METADATA_DIR / "dataset_splits.csv")
    _, val_loader, test_loader = create_dataloaders(
        data_dir=str(RAW_DATA_DIR / "sample" / "images"),
        labels_df=dataset_df,
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
    )

    model = MODEL_BUILDERS[args.model](len(CLASS_NAMES)).to(device)
    checkpoint_path = MODELS_DIR / f"{args.model}_best.pt"
    model.load_state_dict(torch.load(checkpoint_path)["model_state_dict"])

    val_labels, val_logits = get_logits(model, val_loader, device)
    thresholds = find_best_thresholds(val_labels, val_logits, CLASS_NAMES)

    print("Tuned thresholds (from validation set):")
    for class_name, t in thresholds.items():
        print(f"  {class_name:<20s} {t:.2f}")

    with open(METRICS_DIR / f"{args.model}_thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)


if __name__ == "__main__":
    main()
