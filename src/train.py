"""Training entry point.

    python -m src.train --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from .config import Config
from .data import build_dataloaders
from .engine import train_one_epoch, validate
from .losses import CombinedLoss
from .models import build_model
from .utils import (
    count_parameters,
    get_device,
    make_grad_scaler,
    save_checkpoint,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ScratchUNet on the parts dataset")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--epochs", type=int, default=None, help="Override config epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override config batch size")
    parser.add_argument("--data-root", default=None, help="Override dataset root")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config.from_yaml(args.config)
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.data_root is not None:
        cfg.data.root = args.data_root

    set_seed(cfg.seed)
    device = get_device()
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")

    train_loader, val_loader, _ = build_dataloaders(cfg)
    print(
        f"Train: {len(train_loader.dataset)} images | "
        f"Val: {len(val_loader.dataset)} images"
    )

    model = build_model(cfg).to(device)
    print(f"Model parameters: {count_parameters(model) / 1e6:.2f}M")
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs (DataParallel)")
        model = nn.DataParallel(model)

    criterion = CombinedLoss(cfg.train.dice_smooth, cfg.train.focal_gamma)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=cfg.train.t_0, T_mult=cfg.train.t_mult, eta_min=cfg.train.lr_min
    )
    amp = cfg.train.amp and device.type == "cuda"
    scaler = make_grad_scaler(device.type, enabled=amp)

    best_dice = 0.0
    ckpt_path = output_dir / "best_model.pth"
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_dice": [], "lr": []}

    for epoch in range(1, cfg.train.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device,
            aux_weights=cfg.train.aux_weights,
            grad_clip=cfg.train.grad_clip,
            amp=amp,
        )
        val_loss, val_dice = validate(
            model, val_loader, criterion, device, cfg.data.num_classes, amp=amp
        )
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["lr"].append(lr)

        flag = ""
        if val_dice > best_dice:
            best_dice = val_dice
            save_checkpoint(ckpt_path, model, epoch, val_dice)
            flag = "  <- saved"

        print(
            f"Ep {epoch:03d}/{cfg.train.epochs}  "
            f"Loss {train_loss:.4f}/{val_loss:.4f}  "
            f"Dice {val_dice:.4f}  LR {lr:.2e}{flag}"
        )

    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps({"best_val_dice": best_dice, "epochs": cfg.train.epochs}, indent=2)
    )
    print(f"\nBest val Dice: {best_dice:.4f}")
    print(f"Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
