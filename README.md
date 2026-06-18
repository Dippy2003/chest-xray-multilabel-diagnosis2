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

NIH ChestX-ray14 (via Kaggle) - 5 classes:
- No Finding (majority class)
- Pneumonia
- Effusion
- Atelectasis
- Cardiomegaly

## 📁 Project Structure

```
├── data/
│   ├── raw/                # Raw images from Kaggle
│   ├── processed/          # Resized/normalized images
│   └── metadata/           # Train/val/test splits
├── src/
│   ├── data/              # Dataset & DataLoader utilities
│   ├── models/            # Model architectures
│   ├── training/          # Training loops & metrics
│   ├── visualization/     # Grad-CAM & plotting
│   └── utils/             # Config & helpers
├── notebooks/
│   └── 01_eda.ipynb      # Exploratory data analysis
└── results/               # Saved models & visualizations
```

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download dataset:**
   Run the EDA notebook to automatically download via Kaggle API.

3. **Phase 1 - Data Exploration:**
   ```bash
   jupyter notebook notebooks/01_eda.ipynb
   ```

## 📚 Phases

- **Phase 1:** Data loading & EDA
- **Phase 2:** Baseline CNN architecture
- **Phase 3:** Transfer learning (ResNet50, EfficientNet-B0)
- **Phase 4:** Class imbalance handling & evaluation
- **Phase 5:** Grad-CAM & final report

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
