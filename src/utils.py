"""Seeding, device selection, checkpoint I/O and mask visualisation."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from .constants import CLASS_COLORS


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Seed every RNG that touches the training loop."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def autocast(device_type: str, enabled: bool):
    """``torch.amp.autocast`` with a fallback for torch < 2.4."""
    try:
        return torch.amp.autocast(device_type=device_type, enabled=enabled)
    except (AttributeError, TypeError):  # pragma: no cover - old torch
        return torch.cuda.amp.autocast(enabled=enabled)


def make_grad_scaler(device_type: str, enabled: bool):
    """``torch.amp.GradScaler`` with a fallback for torch < 2.4."""
    try:
        return torch.amp.GradScaler(device_type, enabled=enabled)
    except (AttributeError, TypeError):  # pragma: no cover - old torch
        return torch.cuda.amp.GradScaler(enabled=enabled)


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Drop the ``module.`` prefix that ``nn.DataParallel`` adds.

    The submitted run trained on 2x T4 under DataParallel, so its checkpoint
    keys are prefixed. Without this, loading that checkpoint into a plain
    single-GPU (or CPU) model raises a KeyError.
    """
    if not any(k.startswith("module.") for k in state_dict):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def save_checkpoint(path: str | Path, model: torch.nn.Module, epoch: int, val_dice: float) -> None:
    state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epoch, "state_dict": state, "val_dice": val_dice}, path)


def load_checkpoint(path: str | Path, model: torch.nn.Module, device: torch.device) -> dict:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    target = model.module if hasattr(model, "module") else model
    target.load_state_dict(strip_module_prefix(ckpt["state_dict"]))
    return ckpt


def colorise_mask(mask: np.ndarray) -> np.ndarray:
    """Map an integer class mask to an RGB image for plotting."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, colour in CLASS_COLORS.items():
        rgb[mask == class_id] = colour
    return rgb


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
