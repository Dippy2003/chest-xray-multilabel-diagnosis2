# Multi-label Chest X-ray Disease Classifier

A deep learning portfolio project demonstrating advanced PyTorch modeling skills for multi-label medical image classification.

## 🎯 Project Goal

Build a multi-label chest X-ray disease classifier with:
- Custom CNN baseline trained from scratch
- Transfer learning with ResNet50 and EfficientNet-B0
- Proper handling of severe class imbalance
- AUC-ROC and per-class F1 evaluation metrics
- Grad-CAM visualizations for model interpretability
- Clean, modular codebase

## 📊 Dataset

NIH ChestX-ray14 sample set (via Kaggle, `nih-chest-xrays/sample`) - 5,606 images, 5 classes:
- No Finding (54.3%)
- Effusion (11.5%)
- Atelectasis (9.1%)
- Cardiomegaly (2.5%)
- Pneumonia (1.1%)

Using the official 5,606-image sample rather than the full 112k-image dataset (45GB) -
fits on a 4GB VRAM GPU within the project's time budget, still keeps the same severe
imbalance (~49x between most/least common class) that the project is about handling.

## 📁 Project Structure

```
├── data/
│   ├── raw/                # Raw images from Kaggle (gitignored)
│   └── metadata/           # dataset_splits.csv, EDA plots
├── src/
│   ├── data/               # Dataset & DataLoader utilities
│   ├── models/              # baseline CNN, ResNet50/EfficientNet-B0 transfer learning
│   ├── training/            # train/eval engine, early stopping, pos_weight calc
│   ├── evaluation/          # per-class AUC-ROC/F1 metrics
│   ├── visualization/       # Grad-CAM & plotting (not yet built)
│   └── utils/               # Config & helpers
├── scripts/
│   ├── train_baseline.py
│   ├── train_resnet.py
│   └── train_efficientnet.py
├── notebooks/
│   └── 01_eda.ipynb
└── results/
    └── metrics/             # training history + test metrics per model, model_comparison.md
```

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download + explore data:**
   ```bash
   jupyter notebook notebooks/01_eda.ipynb
   ```
   downloads the dataset via Kaggle API, builds dataset_splits.csv

3. **Train a model:**
   ```bash
   python scripts/train_baseline.py
   python scripts/train_resnet.py
   python scripts/train_efficientnet.py
   ```

## 📚 Progress (5-day plan)

- ✅ **Day 1:** Data loading, EDA, train/val/test splits
- ✅ **Day 2:** Baseline CNN from scratch - test macro AUC 0.677
- ✅ **Day 3:** Transfer learning (ResNet50 0.755, EfficientNet-B0 0.728) - both beat baseline
- ✅ **Day 4:** Per-class threshold tuning - resnet50 macro F1 0.308 → 0.318
- ✅ **Day 5:** Grad-CAM visualizations + final write-up

See [results/metrics/model_comparison.md](results/metrics/model_comparison.md) for full per-class numbers.

## 🔥 Grad-CAM

Grad-CAM on resnet50's last conv block (`layer4`), showing what the model actually looks at for each predicted class:

| Cardiomegaly | Effusion | Atelectasis |
|---|---|---|
| ![cardiomegaly](results/visualizations/gradcam_13_Cardiomegaly.png) | ![effusion](results/visualizations/gradcam_5_Effusion.png) | ![atelectasis](results/visualizations/gradcam_9_Atelectasis.png) |

Cardiomegaly's heatmap lands right on the heart silhouette, and Effusion lights up the lower lung / costophrenic region - both match where a radiologist would actually look, which is a good sign the model learned real features and not some dataset artifact.

Full write-up: [results/RESULTS.md](results/RESULTS.md)

## 🛠️ Tech Stack

- **Framework:** PyTorch + torchvision
- **Metrics:** scikit-learn (AUC-ROC, F1)
- **Viz:** matplotlib, seaborn
- **Dataset:** Kaggle API

## 📝 Design Decisions

- **BCEWithLogitsLoss:** For multi-label classification (not CrossEntropyLoss which is single-label only)
- **Weighted loss:** To handle severe class imbalance without modifying training data
- **AUC-ROC/F1:** Better than accuracy for imbalanced multi-label problems
- **224x224 images:** Standard for ImageNet pretrained models
- **Modular code:** Easy to add new models and evaluate systematically

---

**Author:** Dippy2003  
**Goal:** CV portfolio project demonstrating deep learning mastery
