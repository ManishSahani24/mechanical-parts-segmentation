"""Dataset and dataloader construction."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ..config import Config
from .transforms import build_eval_transform, build_train_transform


class SegmentationDataset(Dataset):
    """Paired image/mask dataset. ``mask_dir=None`` yields images only (test).

    Note: masks are read with ``np.array(Image.open(...))`` and deliberately
    **not** converted to RGB or "L". The PNGs store class ids directly as pixel
    values; any conversion would remap them and silently destroy the labels.
    """

    def __init__(
        self,
        img_ids: Sequence[str],
        img_dir: Path,
        mask_dir: Optional[Path] = None,
        transform=None,
    ) -> None:
        self.img_ids = list(img_ids)
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir) if mask_dir is not None else None
        self.transform = transform

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int):
        stem = self.img_ids[idx]
        image = np.array(Image.open(self.img_dir / f"{stem}.png").convert("RGB"))

        if self.mask_dir is None:
            if self.transform:
                return self.transform(image=image)["image"], stem
            return torch.from_numpy(image).permute(2, 0, 1).float() / 255.0, stem

        mask = np.array(Image.open(self.mask_dir / f"{stem}.png"))
        if self.transform:
            out = self.transform(image=image, mask=mask)
            return out["image"], out["mask"].long(), stem
        return (
            torch.from_numpy(image).permute(2, 0, 1).float() / 255.0,
            torch.from_numpy(mask).long(),
            stem,
        )


def split_ids(img_dir: Path, val_frac: float, seed: int) -> Tuple[List[str], List[str]]:
    """Deterministic random train/val split over image stems."""
    ids = sorted(p.stem for p in Path(img_dir).glob("*.png"))
    if not ids:
        raise FileNotFoundError(f"No PNGs found in {img_dir}. Did you download the dataset?")
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(ids))
    n_val = int(len(ids) * val_frac)
    val = [ids[i] for i in perm[:n_val]]
    train = [ids[i] for i in perm[n_val:]]
    return train, val


def build_dataloaders(cfg: Config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Construct the train / val / test dataloaders from a config."""
    train_ids, val_ids = split_ids(cfg.data.train_img_dir, cfg.data.val_frac, cfg.seed)
    test_ids = sorted(p.stem for p in cfg.data.test_img_dir.glob("*.png"))

    train_ds = SegmentationDataset(
        train_ids, cfg.data.train_img_dir, cfg.data.train_mask_dir, build_train_transform()
    )
    val_ds = SegmentationDataset(
        val_ids, cfg.data.train_img_dir, cfg.data.train_mask_dir, build_eval_transform()
    )
    test_ds = SegmentationDataset(
        test_ids, cfg.data.test_img_dir, None, build_eval_transform()
    )

    common = {"num_workers": cfg.data.num_workers, "pin_memory": True}
    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, shuffle=True, drop_last=True, **common
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False, **common)
    test_loader = DataLoader(
        test_ds, batch_size=cfg.predict.batch_size, shuffle=False, **common
    )
    return train_loader, val_loader, test_loader
