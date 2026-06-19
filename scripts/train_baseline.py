"""
Train the custom baseline CNN from scratch on the chest X-ray sample dataset.

Usage:
    python scripts/train_baseline.py [--epochs N] [--lr LR]
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
from models import BaselineCNN
from training import EarlyStopping, evaluate, save_checkpoint, train_one_epoch
from utils import (
    CLASS_NAMES,
    IMAGE_SIZE,
    LEARNING_RATE,
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


def compute_pos_weight(train_df: pd.DataFrame, class_names: list, max_weight: float = 20.0) -> torch.Tensor:
    """
    pos_weight[c] = (negative count) / (positive count) for class c — the standard
    recipe for BCEWithLogitsLoss under class imbalance, since it equalizes the
    expected positive- and negative-term loss contributions per class regardless
    of how rare the positive class is.

    Capped at max_weight: Pneumonia has ~46 positives out of 3924 train samples,
    giving an uncapped ratio of ~84x. Weighting that aggressively risks unstable
    gradients / the model overcorrecting into predicting Pneumonia too liberally.
    Capping at 20x is a judgment call, not a textbook value — worth revisiting in
    Day 4 if Pneumonia's recall/precision trade-off looks off.
    """
    pos_counts = train_df[class_names].sum()
    neg_counts = len(train_df) - pos_counts
    raw_weight = (neg_counts / pos_counts.clip(lower=1)).clip(upper=max_weight)
    return torch.tensor(raw_weight.values, dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
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
    print(f"pos_weight per class: {dict(zip(CLASS_NAMES, pos_weight.tolist()))}")

    model = BaselineCNN(num_classes=len(CLASS_NAMES)).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    early_stopping = EarlyStopping(patience=PATIENCE, mode="max")

    history = []
    checkpoint_path = MODELS_DIR / "baseline_cnn_best.pt"

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
    pd.DataFrame(history).to_csv(METRICS_DIR / "baseline_cnn_history.csv", index=False)

    # Final evaluation on the test set using the best checkpoint
    model.load_state_dict(torch.load(checkpoint_path)["model_state_dict"])
    test_loss, test_metrics = evaluate(model, test_loader, criterion, device, CLASS_NAMES)
    print(f"\nTest loss: {test_loss:.4f}")
    print(format_metrics(test_metrics))

    with open(METRICS_DIR / "baseline_cnn_test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)


if __name__ == "__main__":
    main()
