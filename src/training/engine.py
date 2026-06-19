"""
Shared training/evaluation loop, reused by the baseline CNN and later by the
transfer-learning models so training logic isn't duplicated per architecture.
"""
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from evaluation.metrics import compute_metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one training epoch. Returns the mean training loss."""
    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: list,
) -> Tuple[float, Dict[str, Dict[str, float]]]:
    """
    Run inference over a full loader and compute loss + per-class metrics.

    Metrics are computed once over all accumulated predictions (not averaged
    per-batch) — AUC-ROC in particular is not meaningfully averageable across
    small batches, since it needs the full score distribution per class.
    """
    model.eval()
    total_loss = 0.0
    all_logits, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(torch.cat(all_labels), torch.cat(all_logits), class_names)
    return avg_loss, metrics


class EarlyStopping:
    """
    Stops training when a monitored metric stops improving, and tracks
    whether the current epoch produced a new best (for checkpointing).

    Monitors macro AUC-ROC by default — rising training/val loss alone
    doesn't tell us whether the model is actually getting better on the
    rare classes, which is the whole point of this project.
    """

    def __init__(self, patience: int = 5, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.best_score: float = float("-inf") if mode == "max" else float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, score: float) -> bool:
        """Update state with the latest score. Returns True if it's a new best."""
        is_better = score > self.best_score if self.mode == "max" else score < self.best_score

        if is_better:
            self.best_score = score
            self.counter = 0
            return True

        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


def save_checkpoint(model: nn.Module, path: Path, epoch: int, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict(), "epoch": epoch, "metrics": metrics},
        path,
    )
