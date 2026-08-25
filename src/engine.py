"""Train and validation loops."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .metrics import dice_metric
from .utils import autocast


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler,
    device: torch.device,
    aux_weights: Sequence[float] = (0.4, 0.2),
    grad_clip: float = 1.0,
    amp: bool = True,
) -> float:
    """One pass over the training set. Returns mean loss per sample."""
    model.train()
    running = 0.0

    for images, masks, _ in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device.type, enabled=amp):
            main, aux3, aux2 = model(images)
            loss = (
                criterion(main, masks)
                + aux_weights[0] * criterion(aux3, masks)
                + aux_weights[1] * criterion(aux2, masks)
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        running += loss.item() * images.size(0)

    return running / len(loader.dataset)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int = 7,
    amp: bool = True,
) -> Tuple[float, float]:
    """Returns ``(mean loss, mean foreground Dice)``.

    Note the loss here is the *main head only* — the aux heads are inactive in
    eval mode — so validation loss is not directly comparable to training loss.
    """
    model.eval()
    total_loss = 0.0
    total_dice = 0.0

    for images, masks, _ in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with autocast(device.type, enabled=amp):
            logits = model(images)
            loss = criterion(logits, masks)

        preds = logits.argmax(1).cpu().numpy()
        targets = masks.cpu().numpy()
        for pred, target in zip(preds, targets):
            total_dice += dice_metric(pred, target, num_classes)
        total_loss += loss.item() * images.size(0)

    n = len(loader.dataset)
    return total_loss / n, total_dice / n
