"""Evaluation metric.

Mean Dice over the 6 foreground classes. Background is excluded because it
covers ~76% of every image and would otherwise dominate the score.
"""

from __future__ import annotations

import numpy as np


def dice_metric(pred: np.ndarray, target: np.ndarray, num_classes: int = 7) -> float:
    """Mean per-class Dice over foreground classes 1..num_classes-1.

    A class that is absent from *both* the prediction and the ground truth
    scores 1.0 (a correct "nothing here" call). See the README's
    "Honest notes" section for why this convention matters.
    """
    scores = []
    for c in range(1, num_classes):
        p = pred == c
        t = target == c
        intersection = (p & t).sum()
        cardinality = p.sum() + t.sum()
        scores.append(1.0 if cardinality == 0 else 2.0 * intersection / cardinality)
    return float(np.mean(scores))


def per_class_dice(pred: np.ndarray, target: np.ndarray, num_classes: int = 7) -> dict:
    """Same as ``dice_metric`` but returns the individual class scores."""
    out = {}
    for c in range(1, num_classes):
        p = pred == c
        t = target == c
        cardinality = p.sum() + t.sum()
        out[c] = 1.0 if cardinality == 0 else float(2.0 * (p & t).sum() / cardinality)
    return out
