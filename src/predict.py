"""Inference + submission generation.

    python -m src.predict --config configs/default.yaml \
        --checkpoint outputs/best_model.pth --out outputs/submission.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import Config
from .constants import CLASS_ID, CLASS_LABELS
from .data import build_dataloaders
from .models import build_model
from .rle import EMPTY_RLE, rle_encode
from .tta import predict_plain, predict_tta
from .utils import get_device, load_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Kaggle submission")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="outputs/best_model.pth")
    parser.add_argument("--out", default="outputs/submission.csv")
    parser.add_argument("--no-tta", action="store_true", help="Disable 8-fold TTA")
    parser.add_argument("--data-root", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config.from_yaml(args.config)
    if args.data_root is not None:
        cfg.data.root = args.data_root
    use_tta = cfg.predict.tta and not args.no_tta

    set_seed(cfg.seed)
    device = get_device()
    amp = cfg.train.amp and device.type == "cuda"

    model = build_model(cfg).to(device)
    ckpt = load_checkpoint(args.checkpoint, model, device)
    model.eval()
    print(f"Loaded {args.checkpoint} (epoch {ckpt['epoch']}, val Dice {ckpt['val_dice']:.4f})")
    print(f"TTA: {'8-fold dihedral' if use_tta else 'off'}")

    _, _, test_loader = build_dataloaders(cfg)
    predict_fn = predict_tta if use_tta else predict_plain

    results: dict[str, str] = {}
    for images, stems in test_loader:
        for i, stem in enumerate(stems):
            probs = predict_fn(model, images[i], device, amp=amp)
            mask = probs.argmax(0).numpy().astype(np.uint8)
            for label in CLASS_LABELS:
                binary = mask == CLASS_ID[label]
                results[f"{stem}.png_{label}"] = (
                    rle_encode(binary) if binary.any() else EMPTY_RLE
                )

    submission = pd.read_csv(cfg.data.sample_submission)
    submission["EncodedPixels"] = (
        submission["ImageId_ClassId"].map(results).fillna(EMPTY_RLE)
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False)

    missing = submission["ImageId_ClassId"].map(results).isna().sum()
    print(f"Wrote {out_path} ({len(submission)} rows, {missing} unmatched ids)")


if __name__ == "__main__":
    main()
