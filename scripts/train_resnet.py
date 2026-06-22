"""
Train ResNet50 transfer learning model.

Usage:
    python scripts/train_resnet.py [--epochs N] [--lr LR]
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data import create_dataloaders
from evaluation import format_metrics
from models import ResNetTransfer
from training import EarlyStopping, compute_pos_weight, evaluate, save_checkpoint, train_one_epoch
from utils import (
    CLASS_NAMES,
    IMAGE_SIZE,
    METADATA_DIR,
    METRICS_DIR,
    MODELS_DIR,
    NUM_EPOCHS,
    PATIENCE,
    RAW_DATA_DIR,
    WEIGHT_DECAY,
    get_device,
    set_seed,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    dataset_df = pd.read_csv(METADATA_DIR / "dataset_splits.csv")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=str(RAW_DATA_DIR / "sample" / "images"),
        labels_df=dataset_df,
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
    )

    train_df = dataset_df[dataset_df["split"] == "train"]
    pos_weight = compute_pos_weight(train_df, CLASS_NAMES).to(device)

    model = ResNetTransfer(num_classes=len(CLASS_NAMES)).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=WEIGHT_DECAY)
    early_stopping = EarlyStopping(patience=PATIENCE, mode="max")

    history = []
    checkpoint_path = MODELS_DIR / "resnet50_best.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device, CLASS_NAMES)
        macro_auc = val_metrics["macro_avg"]["auc_roc"]
        macro_f1 = val_metrics["macro_avg"]["f1"]

        print(
            f"Epoch {epoch:>3d}/{args.epochs} | train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_macro_auc={macro_auc:.4f} val_macro_f1={macro_f1:.4f}"
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_macro_auc": macro_auc,
                "val_macro_f1": macro_f1,
            }
        )

        if early_stopping.step(macro_auc):
            save_checkpoint(model, checkpoint_path, epoch, val_metrics)
            print(f"  -> new best val_macro_auc={macro_auc:.4f}, checkpoint saved")

        if early_stopping.should_stop:
            print(f"Early stopping triggered at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(METRICS_DIR / "resnet50_history.csv", index=False)

    model.load_state_dict(torch.load(checkpoint_path)["model_state_dict"])
    test_loss, test_metrics = evaluate(model, test_loader, criterion, device, CLASS_NAMES)
    print(f"\nTest loss: {test_loss:.4f}")
    print(format_metrics(test_metrics))

    with open(METRICS_DIR / "resnet50_test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)


if __name__ == "__main__":
    main()
