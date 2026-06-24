"""
Run Grad-CAM on a handful of test set images using the trained resnet50,
save heatmap overlays so we can sanity check what the model is actually
looking at instead of just trusting the AUC/F1 numbers.

Usage:
    python scripts/run_gradcam.py --num-images 8
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.loader import get_val_transforms
from data.dataset import ChestXrayDataset
from models import ResNetTransfer
from utils import CLASS_NAMES, IMAGE_SIZE, METADATA_DIR, MODELS_DIR, RAW_DATA_DIR, VIZ_DIR, get_device
from visualization.gradcam import GradCAM, overlay_heatmap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-images", type=int, default=8)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    dataset_df = pd.read_csv(METADATA_DIR / "dataset_splits.csv")
    test_df = dataset_df[dataset_df["split"] == "test"].reset_index(drop=True)

    test_dataset = ChestXrayDataset(
        img_dir=str(RAW_DATA_DIR / "sample" / "images"),
        labels_df=test_df,
        transform=get_val_transforms(IMAGE_SIZE),
        class_names=CLASS_NAMES,
    )

    model = ResNetTransfer(num_classes=len(CLASS_NAMES)).to(device)
    checkpoint = torch.load(MODELS_DIR / "resnet50_best.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    cam = GradCAM(model, model.backbone.layer4)

    # only pick images that have at least one positive label, so there's an
    # actual disease class to explain instead of "No Finding" every time
    candidate_idxs = [
        i for i in range(len(test_dataset))
        if test_df.iloc[i][CLASS_NAMES[1:]].sum() > 0
    ][: args.num_images]

    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    for idx in candidate_idxs:
        image, labels = test_dataset[idx]
        image_batch = image.unsqueeze(0).to(device)

        positive_classes = [CLASS_NAMES[i] for i, v in enumerate(labels) if v == 1]
        class_idx = CLASS_NAMES.index(positive_classes[0])

        heatmap = cam(image_batch, class_idx=class_idx)
        overlay = overlay_heatmap(image, heatmap)

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(overlay)
        ax.set_title(f"{positive_classes[0]} (idx {idx})", fontsize=10)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(VIZ_DIR / f"gradcam_{idx}_{positive_classes[0].replace(' ', '_')}.png", dpi=120)
        plt.close(fig)

        print(f"saved gradcam for idx {idx}, class={positive_classes[0]}")


if __name__ == "__main__":
    main()
