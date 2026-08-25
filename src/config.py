"""Typed configuration loaded from a YAML file."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml


@dataclass
class DataConfig:
    root: str = "data/Dataset"
    img_size: int = 384
    num_classes: int = 7
    val_frac: float = 0.15
    num_workers: int = 2

    @property
    def train_img_dir(self) -> Path:
        return Path(self.root) / "train" / "images"

    @property
    def train_mask_dir(self) -> Path:
        return Path(self.root) / "train" / "masks"

    @property
    def test_img_dir(self) -> Path:
        return Path(self.root) / "test" / "images"

    @property
    def sample_submission(self) -> Path:
        return Path(self.root) / "sample_submission.csv"


@dataclass
class ModelConfig:
    stem_channels: int = 64
    encoder_channels: List[int] = field(default_factory=lambda: [128, 256, 512, 1024])
    encoder_blocks: List[int] = field(default_factory=lambda: [2, 2, 3, 3])
    aspp_channels: int = 512
    aspp_rates: List[int] = field(default_factory=lambda: [1, 4, 8, 16])
    decoder_channels: List[int] = field(default_factory=lambda: [256, 128, 64, 64])


@dataclass
class TrainConfig:
    batch_size: int = 16
    epochs: int = 60
    lr: float = 3e-4
    lr_min: float = 1e-6
    weight_decay: float = 1e-4
    t_0: int = 15
    t_mult: int = 2
    grad_clip: float = 1.0
    amp: bool = True
    focal_gamma: float = 2.0
    dice_smooth: float = 1.0
    aux_weights: List[float] = field(default_factory=lambda: [0.4, 0.2])


@dataclass
class PredictConfig:
    batch_size: int = 4
    tta: bool = True


@dataclass
class Config:
    seed: int = 42
    output_dir: str = "outputs"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    predict: PredictConfig = field(default_factory=PredictConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
        return cls(
            seed=raw.get("seed", 42),
            output_dir=raw.get("output_dir", "outputs"),
            data=DataConfig(**raw.get("data", {})),
            model=ModelConfig(**raw.get("model", {})),
            train=TrainConfig(**raw.get("train", {})),
            predict=PredictConfig(**raw.get("predict", {})),
        )
