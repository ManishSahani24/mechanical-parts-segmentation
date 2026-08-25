"""8-fold dihedral test-time augmentation.

The dataset has no canonical orientation, so the 8 elements of the dihedral
group D4 are all valid views of the same scene. Predicting on each, mapping the
probability maps back through the inverse transform and averaging removes a
chunk of the model's orientation-specific error.

Averaging happens in **softmax probability space**, not on argmax labels —
averaging hard labels would throw away exactly the confidence information that
makes the ensemble useful.

Note on the group
-----------------
D4 is generated cleanly as ``{rot90(k)} ∪ {rot90(k) then flip(width)}`` for
k = 0..3. Enumerating flips and rotations by hand instead is error-prone: the
original notebook's list contained ``flip([2, 3])`` *and* ``rot90(2)``, which
are the same 180-degree rotation, so it averaged 7 distinct views with one
double-weighted and omitted one diagonal reflection. The construction below is
provably the full group; ``tests/test_tta.py`` asserts all 8 views are
distinct and every forward/inverse pair round-trips.
"""

from __future__ import annotations

from typing import Callable, List, Tuple

import torch

from .utils import autocast


def _rot(k: int) -> Callable:
    return lambda t: t.rot90(k, [2, 3])


def _rot_flip(k: int) -> Callable:
    return lambda t: t.rot90(k, [2, 3]).flip(3)


def _rot_flip_inverse(k: int) -> Callable:
    return lambda t: t.flip(3).rot90((4 - k) % 4, [2, 3])


# (forward, inverse) pairs — the 8 elements of the dihedral group D4.
# k = 0..3 rotations, then the same rotations composed with a horizontal flip
# (which yields both axis flips and both diagonal reflections).
_TRANSFORMS: List[Tuple[Callable, Callable]] = [
    *[(_rot(k), _rot((4 - k) % 4)) for k in range(4)],
    *[(_rot_flip(k), _rot_flip_inverse(k)) for k in range(4)],
]


@torch.no_grad()
def predict_tta(
    model: torch.nn.Module,
    image: torch.Tensor,
    device: torch.device,
    amp: bool = True,
) -> torch.Tensor:
    """Average softmax probabilities over the 8 dihedral views.

    Args:
        model: model in eval mode.
        image: a single ``(C, H, W)`` tensor (no batch dimension).

    Returns:
        ``(num_classes, H, W)`` probability tensor on CPU.
    """
    model.eval()
    x = image.unsqueeze(0).to(device)
    accum = None

    for forward, inverse in _TRANSFORMS:
        with autocast(device.type, enabled=amp):
            probs = torch.softmax(model(forward(x)), dim=1)
        probs = inverse(probs.float())
        accum = probs if accum is None else accum + probs

    return (accum / len(_TRANSFORMS)).squeeze(0).cpu()


@torch.no_grad()
def predict_plain(
    model: torch.nn.Module,
    image: torch.Tensor,
    device: torch.device,
    amp: bool = True,
) -> torch.Tensor:
    """Single forward pass — the no-TTA baseline."""
    model.eval()
    x = image.unsqueeze(0).to(device)
    with autocast(device.type, enabled=amp):
        probs = torch.softmax(model(x), dim=1)
    return probs.float().squeeze(0).cpu()
