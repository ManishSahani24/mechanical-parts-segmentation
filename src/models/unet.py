"""ScratchUNet — a U-Net trained from scratch (no pretrained encoder).

Design notes
------------
* **No ImageNet backbone.** The competition images are synthetic renders of
  machined parts on a plain tray; ImageNet's low-level filters buy little here,
  and a from-scratch encoder made the whole architecture free to tune.
* **ASPP at the bottleneck** gives multi-scale context without a fifth
  downsample, which matters for the two thin classes (spring, o_ring).
* **Deep supervision.** Two auxiliary heads at decoder stages 4 and 3 are
  supervised with the same loss at weights 0.4 / 0.2. They exist only during
  training and are dropped at eval, so inference cost is unchanged.
"""

from __future__ import annotations

from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Config
from .blocks import ASPP, ConvBnAct, DecoderBlock, EncoderBlock

TrainOutput = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class ScratchUNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 7,
        stem_channels: int = 64,
        encoder_channels: Tuple[int, ...] = (128, 256, 512, 1024),
        encoder_blocks: Tuple[int, ...] = (2, 2, 3, 3),
        aspp_channels: int = 512,
        aspp_rates: Tuple[int, ...] = (1, 4, 8, 16),
        decoder_channels: Tuple[int, ...] = (256, 128, 64, 64),
    ) -> None:
        super().__init__()
        c1, c2, c3, c4 = encoder_channels
        b1, b2, b3, b4 = encoder_blocks
        d4, d3, d2, d1 = decoder_channels

        self.stem = nn.Sequential(
            ConvBnAct(3, stem_channels), ConvBnAct(stem_channels, stem_channels)
        )
        self.enc1 = EncoderBlock(stem_channels, c1, n_res=b1)
        self.enc2 = EncoderBlock(c1, c2, n_res=b2)
        self.enc3 = EncoderBlock(c2, c3, n_res=b3)
        self.enc4 = EncoderBlock(c3, c4, n_res=b4)

        self.aspp = ASPP(c4, aspp_channels, rates=aspp_rates)

        self.dec4 = DecoderBlock(aspp_channels, c3, d4)
        self.dec3 = DecoderBlock(d4, c2, d3)
        self.dec2 = DecoderBlock(d3, c1, d2)
        self.dec1 = DecoderBlock(d2, stem_channels, d1)

        self.head_main = nn.Conv2d(d1, num_classes, 1)
        self.head_aux3 = nn.Conv2d(d4, num_classes, 1)
        self.head_aux2 = nn.Conv2d(d3, num_classes, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, TrainOutput]:
        size = x.shape[-2:]

        s0 = self.stem(x)      # 1/1
        s1 = self.enc1(s0)     # 1/2
        s2 = self.enc2(s1)     # 1/4
        s3 = self.enc3(s2)     # 1/8
        s4 = self.enc4(s3)     # 1/16

        bottleneck = self.aspp(s4)

        y4 = self.dec4(bottleneck, s3)
        y3 = self.dec3(y4, s2)
        y2 = self.dec2(y3, s1)
        y1 = self.dec1(y2, s0)

        out = self.head_main(y1)
        if not self.training:
            return out

        aux3 = F.interpolate(
            self.head_aux3(y4), size=size, mode="bilinear", align_corners=False
        )
        aux2 = F.interpolate(
            self.head_aux2(y3), size=size, mode="bilinear", align_corners=False
        )
        return out, aux3, aux2


def build_model(cfg: Config) -> ScratchUNet:
    return ScratchUNet(
        num_classes=cfg.data.num_classes,
        stem_channels=cfg.model.stem_channels,
        encoder_channels=tuple(cfg.model.encoder_channels),
        encoder_blocks=tuple(cfg.model.encoder_blocks),
        aspp_channels=cfg.model.aspp_channels,
        aspp_rates=tuple(cfg.model.aspp_rates),
        decoder_channels=tuple(cfg.model.decoder_channels),
    )
