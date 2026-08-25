"""Albumentations pipelines.

The parts are photographed on a flat tray with no canonical orientation, so
the full dihedral group (flips + 90-degree rotations) is label-preserving and
free to apply. Elastic/grid warps are what keep the thin classes (springs,
o-rings) from being memorised by shape template rather than texture.

API note
--------
The original notebook was written against the albumentations 1.x signatures
(``GaussNoise(var_limit=...)``, ``CoarseDropout(max_holes=..., ...)``,
``ShiftScaleRotate``). Under albumentations 2.x those keyword arguments are
**silently dropped** with only a ``UserWarning``, so both transforms ran with
library defaults — in particular ``CoarseDropout`` left the *mask* untouched,
which is not what ``mask_fill_value=0`` was asking for. This module uses the
2.x signatures so the intended augmentation actually happens; the numbers below
are the 1.x values converted (e.g. var_limit (10, 50) on a 0-255 scale is
std 0.012-0.028 on the 0-1 scale 2.x expects).
"""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

from ..constants import MEAN, STD


def build_train_transform() -> A.Compose:
    return A.Compose(
        [
            # --- geometric: dihedral group is label-preserving here ---
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(
                translate_percent=(-0.1, 0.1),
                scale=(0.8, 1.2),
                rotate=(-45, 45),
                border_mode=0,
                fill=0,
                fill_mask=0,
                p=0.6,
            ),
            A.ElasticTransform(alpha=120, sigma=6, p=0.3),
            A.GridDistortion(p=0.2),
            # --- photometric: lighting/finish varies across the tray ---
            A.OneOf(
                [
                    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
                    A.HueSaturationValue(
                        hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20
                    ),
                    A.CLAHE(clip_limit=4.0),
                ],
                p=0.7,
            ),
            A.GaussNoise(std_range=(0.012, 0.028), p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            # --- occlusion: forces context rather than whole-object matching ---
            # up to 8 holes of at most 32px on a 384px image (32/384 = 0.083)
            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(0.02, 0.084),
                hole_width_range=(0.02, 0.084),
                fill=0,
                fill_mask=0,
                p=0.2,
            ),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ]
    )


def build_eval_transform() -> A.Compose:
    """Validation / test: normalise only, no augmentation."""
    return A.Compose([A.Normalize(mean=MEAN, std=STD), ToTensorV2()])
