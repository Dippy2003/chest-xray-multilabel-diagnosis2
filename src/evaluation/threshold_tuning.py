"""
Per-class probability threshold tuning.

compute_metrics() so far has used one flat threshold (0.5) for every class to
turn probabilities into predictions. That's a bad fit here - classes like
Pneumonia have very few positives, and their optimal decision threshold is not
the same as a class like "No Finding" which is over half the dataset. This
sweeps thresholds per class on the validation set and picks whatever maximizes
F1 for that class.
"""
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import f1_score


def find_best_thresholds(
    y_true: torch.Tensor,
    y_logits: torch.Tensor,
    class_names: List[str],
    thresholds=np.arange(0.05, 0.96, 0.05),
) -> Dict[str, float]:
    """
    For each class, sweep candidate thresholds and keep the one with the
    highest F1 score. Meant to be run on the validation set, then reused
    (frozen) on the test set so the test set stays untouched by tuning.
    """
    y_true_np = y_true.detach().cpu().numpy()
    y_prob_np = torch.sigmoid(y_logits).detach().cpu().numpy()

    best_thresholds: Dict[str, float] = {}
    for i, class_name in enumerate(class_names):
        best_f1 = -1.0
        best_t = 0.5
        for t in thresholds:
            preds = (y_prob_np[:, i] >= t).astype(int)
            f1 = f1_score(y_true_np[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
        best_thresholds[class_name] = best_t

    return best_thresholds
