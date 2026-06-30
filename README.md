---
title: Chest X-ray Multilabel Diagnosis
emoji: 🫁
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

<div align="center">

# Multi-label Chest X-ray Disease Classifier

> Deep learning for pathology detection across 5 disease classes from a single chest X-ray

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)
![Docker](https://img.shields.io/badge/Docker-Hugging%20Face%20Spaces-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

**Live demo:** [dipnas-chest-xray-multilabel-diagnosis2.hf.space](https://dipnas-chest-xray-multilabel-diagnosis2.hf.space)

</div>

---

## What It Does

| Input | Action | Output |
|---|---|---|
| Chest X-ray image (PNG/JPG) | Multi-label classification via ResNet50 | Per-class disease probabilities with confidence |
| Same image | Grad-CAM on `layer4` | Heatmap overlay showing which region triggered each prediction |
| Raw training data | Weighted BCE loss + threshold tuning | Calibrated thresholds per class to maximize F1 under severe imbalance |

---

## How It Works

```
  X-ray image
       |
       v
  [ Preprocessing ]  224x224 resize, ImageNet normalization
       |
       v
  [ ResNet50 backbone ]  pretrained on ImageNet, fine-tuned on ChestX-ray14
       |
       v
  [ BCEWithLogitsLoss ]  with pos_weight to handle 49x class imbalance
       |
       v
  [ Per-class thresholds ]  tuned on validation set to maximize macro F1
       |
       v
  [ Grad-CAM ]  gradients from layer4 -> spatial heatmap per predicted class
       |
       v
  Predicted labels + probability scores + heatmap overlay
```

1. Images are resized to 224x224 and normalized with ImageNet mean/std.
2. ResNet50 replaces its final FC layer with a 5-output head (one logit per class).
3. `BCEWithLogitsLoss` with `pos_weight` penalizes false negatives on rare classes more heavily.
4. After training, per-class thresholds are tuned on the validation set to push macro F1 above the default 0.5 cutoff.
5. Grad-CAM backpropagates through `layer4` to produce a spatial activation map, resized and alpha-blended over the original image.

---

## Model Results

| Model | Macro AUC-ROC | Macro F1 (tuned) |
|---|---|---|
| Baseline CNN (from scratch) | 0.677 | baseline |
| ResNet50 (transfer learning) | **0.755** | **0.318** |
| EfficientNet-B0 (transfer learning) | 0.728 | 0.311 |

Full per-class breakdown: [results/metrics/model_comparison.md](results/metrics/model_comparison.md)

---

## Grad-CAM Visualizations

Grad-CAM on ResNet50's last conv block (`layer4`), showing what the model attends to for each predicted class:

| Cardiomegaly | Effusion | Atelectasis |
|---|---|---|
| ![cardiomegaly](results/visualizations/gradcam_13_Cardiomegaly.png) | ![effusion](results/visualizations/gradcam_5_Effusion.png) | ![atelectasis](results/visualizations/gradcam_9_Atelectasis.png) |

The Cardiomegaly heatmap lands on the heart silhouette; Effusion lights up the lower lung and costophrenic region. Both match where a radiologist would look, which suggests the model learned real anatomical features rather than dataset artifacts.

Full write-up: [results/RESULTS.md](results/RESULTS.md)

---

## Tech Stack

| Tool | Role |
|---|---|
| PyTorch + torchvision | Model training, ResNet50/EfficientNet-B0 transfer learning |
| scikit-learn | AUC-ROC, F1 metrics, per-class threshold tuning |
| matplotlib + seaborn | Training curves, EDA plots, Grad-CAM overlays |
| Kaggle API | Dataset download (`nih-chest-xrays/sample`) |
| FastAPI | REST API: `/predict` and `/gradcam` endpoints |
| React + Vite + TypeScript | Frontend UI |
| Tailwind v4 + shadcn/ui | Component styling |
| Framer Motion | Animated UI transitions |
| Docker | Containerized deployment |
| Hugging Face Spaces | Cloud hosting |

---

## Setup

**Prerequisites:**

- Python 3.10+
- Node.js 18+ (for the frontend)
- A Kaggle account with `~/.kaggle/kaggle.json` configured
- GPU with 4GB+ VRAM recommended (tested on RTX 3050 4GB)

**Steps:**

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Download the dataset and run EDA:
   ```bash
   jupyter notebook notebooks/01_eda.ipynb
   ```
   This downloads the NIH ChestX-ray14 sample via the Kaggle API and writes `data/metadata/dataset_splits.csv`.

3. Train models (each script saves its best checkpoint to `results/models/`):
   ```bash
   python scripts/train_baseline.py
   python scripts/train_resnet.py
   python scripts/train_efficientnet.py
   ```

4. Tune per-class thresholds:
   ```bash
   python scripts/tune_thresholds.py
   ```

5. Build the frontend (one-time, or after changing `web/`):
   ```bash
   cd web
   npm install
   npm run build
   cd ..
   ```

6. Run the web demo:
   ```bash
   uvicorn app.main:app --reload
   ```
   Open [http://127.0.0.1:8000](http://127.0.0.1:8000), upload a chest X-ray, and get per-class probabilities plus a Grad-CAM heatmap for the top predicted class.

   For frontend development with hot reload, run `npm run dev` inside `web/` (proxies `/predict` and `/health` to port 8000) and open [http://localhost:5173](http://localhost:5173).

---

## Project Structure

```
.
├── data/
│   ├── raw/                      # Raw images from Kaggle (gitignored)
│   └── metadata/                 # dataset_splits.csv, EDA plots
├── src/
│   ├── data/                     # Dataset class and DataLoader utilities
│   ├── models/                   # Baseline CNN, ResNet50, EfficientNet-B0
│   ├── training/                 # Train/eval engine, early stopping, pos_weight
│   ├── evaluation/               # Per-class AUC-ROC/F1, threshold tuning
│   ├── visualization/            # Grad-CAM implementation
│   └── utils/                    # Config and shared helpers
├── scripts/
│   ├── train_baseline.py         # Train CNN from scratch
│   ├── train_resnet.py           # Fine-tune ResNet50
│   ├── train_efficientnet.py     # Fine-tune EfficientNet-B0
│   ├── tune_thresholds.py        # Per-class threshold search
│   └── run_gradcam.py            # Generate Grad-CAM overlays
├── app/
│   └── main.py                   # FastAPI server (predict + gradcam, serves web/dist)
├── web/                          # React + Vite + Tailwind + shadcn/ui frontend
├── notebooks/
│   └── 01_eda.ipynb              # Dataset download, EDA, split creation
└── results/
    ├── metrics/                  # Training history, test metrics, model_comparison.md
    ├── visualizations/           # Grad-CAM example overlays
    └── RESULTS.md                # Final write-up with analysis
```

---

## Dataset

NIH ChestX-ray14 sample set via Kaggle (`nih-chest-xrays/sample`): 5,606 images across 5 classes.

| Class | Prevalence |
|---|---|
| No Finding | 54.3% |
| Effusion | 11.5% |
| Atelectasis | 9.1% |
| Cardiomegaly | 2.5% |
| Pneumonia | 1.1% |

The 5,606-image sample rather than the full 112k dataset (45 GB) was used to fit within a 4 GB VRAM budget while preserving the same ~49x class imbalance ratio that the project is built to handle.

---

## Design Decisions

| Decision | Reason |
|---|---|
| `BCEWithLogitsLoss` | Multi-label classification requires independent per-class binary predictions, not a single softmax |
| `pos_weight` in loss | Handles severe class imbalance without modifying or oversampling training data |
| AUC-ROC + F1 metrics | Accuracy is misleading on heavily imbalanced data; these metrics reflect true detection quality |
| 224x224 input size | Standard for ImageNet-pretrained models; required to use pretrained weights without shape mismatch |
| Per-class threshold tuning | Default 0.5 threshold is suboptimal when class priors are extreme; tuning per class lifts macro F1 |

---

## Progress

| Day | Milestone | Result |
|---|---|---|
| Day 1 | Data loading, EDA, train/val/test splits | Done |
| Day 2 | Baseline CNN from scratch | Test macro AUC 0.677 |
| Day 3 | Transfer learning: ResNet50 + EfficientNet-B0 | ResNet50 AUC 0.755, EfficientNet AUC 0.728 |
| Day 4 | Per-class threshold tuning | ResNet50 macro F1: 0.308 to 0.318 |
| Day 5 | Grad-CAM visualizations + final write-up | Done |

---

## License

MIT

**Author:** [Dippy2003](https://github.com/Dippy2003)
