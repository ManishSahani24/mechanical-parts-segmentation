"""Loss functions.

The objective is ``Dice + Focal``. The two terms fix different failure modes:

* **Soft Dice (foreground only)** optimises region overlap directly, which is
  the competition metric, and is insensitive to the 76% background prior.
* **Focal (gamma=2)** is a per-pixel term that keeps pushing on the small set
  of genuinely hard pixels — mainly object boundaries and the thin spring
  coils — long after Dice has flattened out.

Dice alone gives unstable early gradients when predictions are near-uniform;
cross-entropy alone converges to a background-heavy solution. The sum trains
smoothly from random init, which matters when there is no pretrained encoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft multi-class Dice, computed over foreground classes only."""

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(1)
        probs = torch.softmax(logits, dim=1)
        onehot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = (probs[:, 1:] * onehot[:, 1:]).sum(dims)
        cardinality = probs[:, 1:].sum(dims) + onehot[:, 1:].sum(dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """Cross-entropy down-weighted on already-confident pixels."""

    def __init__(self, gamma: float = 2.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


class CombinedLoss(nn.Module):
    """``DiceLoss + FocalLoss`` — the objective used for the submitted run."""

    def __init__(self, dice_smooth: float = 1.0, focal_gamma: float = 2.0) -> None:
        super().__init__()
        self.dice = DiceLoss(smooth=dice_smooth)
        self.focal = FocalLoss(gamma=focal_gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.dice(logits, targets) + self.focal(logits, targets)
