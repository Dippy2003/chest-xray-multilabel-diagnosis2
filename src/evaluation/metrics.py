"""
Per-class evaluation metrics for multi-label chest X-ray classification.

Plain accuracy is misleading here: with "No Finding" at ~54% and "Pneumonia"
at ~1% of samples, a model that always predicts the majority class scores high
accuracy while being useless. AUC-ROC (threshold-independent) and F1 (threshold-
dependent, balances precision/recall) are reported per class instead.
"""
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score


def compute_metrics(
    y_true: torch.Tensor,
    y_logits: torch.Tensor,
    class_names: List[str],
    threshold: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-class AUC-ROC and F1 from raw model logits.

    Args:
        y_true: Tensor of shape (N, num_classes), binary ground-truth labels.
        y_logits: Tensor of shape (N, num_classes), raw model outputs (pre-sigmoid).
        class_names: Class names in the same order as the columns.
        threshold: Probability threshold for converting sigmoid outputs to
            binary predictions when computing F1. A single global threshold
            is a simplification — per-class threshold tuning comes in Day 4.

    Returns:
        Dict with one entry per class_name (each holding 'auc_roc' and 'f1'),
        plus a 'macro_avg' entry. AUC-ROC is NaN for a class with only one
        label value present (e.g. zero positive samples in a small eval set).
    """
    y_true_np = y_true.detach().cpu().numpy()
    y_prob_np = torch.sigmoid(y_logits).detach().cpu().numpy()
    y_pred_np = (y_prob_np >= threshold).astype(int)

    results: Dict[str, Dict[str, float]] = {}
    aucs, f1s = [], []

    for i, class_name in enumerate(class_names):
        f1 = f1_score(y_true_np[:, i], y_pred_np[:, i], zero_division=0)

        # roc_auc_score requires both classes present; guard against the
        # degenerate case (e.g. a tiny eval batch with no positive samples)
        unique_labels = np.unique(y_true_np[:, i])
        auc = roc_auc_score(y_true_np[:, i], y_prob_np[:, i]) if len(unique_labels) > 1 else float("nan")

        results[class_name] = {"auc_roc": auc, "f1": f1}
        f1s.append(f1)
        if not np.isnan(auc):
            aucs.append(auc)

    results["macro_avg"] = {
        "auc_roc": float(np.mean(aucs)) if aucs else float("nan"),
        "f1": float(np.mean(f1s)),
    }
    return results


def format_metrics(metrics: Dict[str, Dict[str, float]]) -> str:
    """Pretty-print per-class metrics as an aligned table."""
    lines = [f"{'Class':<20s} {'AUC-ROC':>10s} {'F1':>10s}"]
    for class_name, vals in metrics.items():
        if class_name == "macro_avg":
            continue
        lines.append(f"{class_name:<20s} {vals['auc_roc']:>10.4f} {vals['f1']:>10.4f}")
    lines.append("-" * 42)
    macro = metrics["macro_avg"]
    lines.append(f"{'Macro Average':<20s} {macro['auc_roc']:>10.4f} {macro['f1']:>10.4f}")
    return "\n".join(lines)
