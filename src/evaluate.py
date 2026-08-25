"""Score a checkpoint on the held-out validation split.

Also measures the TTA delta, which is the number worth quoting rather than
asserting that TTA "improves robustness":

    python -m src.evaluate --checkpoint outputs/best_model.pth --compare-tta
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from .config import Config
from .constants import CLASS_NAMES
from .data import build_dataloaders
from .metrics import dice_metric, per_class_dice
from .models import build_model
from .tta import predict_plain, predict_tta
from .utils import get_device, load_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on the val split")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="outputs/best_model.pth")
    parser.add_argument("--compare-tta", action="store_true", help="Report with and without TTA")
    parser.add_argument("--data-root", default=None)
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, loader, device, predict_fn, num_classes: int, amp: bool):
    overall, per_class = [], {c: [] for c in range(1, num_classes)}
    for images, masks, _ in loader:
        for i in range(images.size(0)):
            probs = predict_fn(model, images[i], device, amp=amp)
            pred = probs.argmax(0).numpy()
            target = masks[i].numpy()
            overall.append(dice_metric(pred, target, num_classes))
            for c, score in per_class_dice(pred, target, num_classes).items():
                per_class[c].append(score)
    return float(np.mean(overall)), {c: float(np.mean(v)) for c, v in per_class.items()}


def main() -> None:
    args = parse_args()
    cfg = Config.from_yaml(args.config)
    if args.data_root is not None:
        cfg.data.root = args.data_root

    set_seed(cfg.seed)
    device = get_device()
    amp = cfg.train.amp and device.type == "cuda"

    model = build_model(cfg).to(device)
    load_checkpoint(args.checkpoint, model, device)
    model.eval()

    _, val_loader, _ = build_dataloaders(cfg)

    variants = [("no TTA", predict_plain)]
    if args.compare_tta or cfg.predict.tta:
        variants.append(("8-fold TTA", predict_tta))

    scores = {}
    for name, fn in variants:
        mean_dice, per_class = evaluate(model, val_loader, device, fn, cfg.data.num_classes, amp)
        scores[name] = mean_dice
        print(f"\n{name}  —  mean foreground Dice: {mean_dice:.4f}")
        for class_id, value in per_class.items():
            print(f"    {CLASS_NAMES[class_id]:<14} {value:.4f}")

    if len(scores) == 2:
        delta = scores["8-fold TTA"] - scores["no TTA"]
        print(f"\nTTA delta: {delta:+.4f} Dice ({delta * 100:+.2f} points)")


if __name__ == "__main__":
    main()
