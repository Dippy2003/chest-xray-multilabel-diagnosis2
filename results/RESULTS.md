# final results

5-day multi-label chest X-ray classifier, NIH ChestX-ray14 sample set (5,606 images, 5 classes).

## best model: resnet50 (transfer learning)

- macro AUC-ROC: 0.755
- macro F1 (tuned thresholds): 0.318
- ~15M trainable params (layer4 + fc head unfrozen, rest of backbone frozen)

## what worked

- **pos_weight in BCEWithLogitsLoss** - stopped the model from just predicting "No Finding" for everything. without it, accuracy looks fine but AUC is useless
- **transfer learning** beat the from-scratch baseline on every class, biggest jump on Cardiomegaly (0.522 -> 0.710) since that's the class with the fewest positives (~140) - pretrained ImageNet features matter most exactly where we have the least data to learn from scratch
- **per-class threshold tuning** - flat 0.5 threshold was hiding real model quality behind bad F1 numbers. tuning thresholds per class on the val set (not test) recovered a lot of that, resnet50 went 0.308 -> 0.318 macro F1
- **freezing most of the backbone** - only ~3.9k training images, fine-tuning the whole resnet50/efficientnet would've overfit fast. freezing everything except the last block + head was the right call for this data size
- **Grad-CAM sanity check** - heatmaps for Cardiomegaly and Effusion land exactly where a radiologist would look (heart silhouette, costophrenic region), which is a real signal the model learned actual disease features and not some shortcut/artifact in the dataset

## what didn't work as well

- **baseline CNN from scratch** - capped around 0.677 macro AUC, no amount of training helped much past that since it has no prior knowledge of what to look for in 3.9k images
- **threshold tuning on the baseline specifically** made F1 slightly worse (0.257 -> 0.250) instead of better - its predictions are the noisiest of the 3 models, so thresholds picked on a small 841-image val set didn't generalize
- **Pneumonia** stayed the hardest class throughout (AUC 0.67-0.82 depending on model, F1 near 0 after tuning) - only ~1% of the dataset is positive, not enough signal even with pos_weight capped at 20x

## model comparison (full numbers)

see [metrics/model_comparison.md](metrics/model_comparison.md)

## scope notes

used the official Kaggle `nih-chest-xrays/sample` dataset (5,606 images) instead of the full 112k-image
NIH set - fits on a 4GB VRAM GPU (RTX 3050) and a 5-day timeline while keeping the same severe class
imbalance the project is about handling. framed as a deliberate scope decision given hardware/time
constraints, not a shortcut.
