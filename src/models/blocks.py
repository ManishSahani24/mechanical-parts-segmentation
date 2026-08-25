"""Building blocks for the from-scratch U-Net."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnAct(nn.Module):
    """Conv -> BatchNorm -> GELU, with dilation-aware padding.

    BatchNorm uses momentum=0.01: at batch size 16 the default momentum of 0.1
    makes the running statistics noisy enough that eval-mode Dice lags
    train-mode by a visible margin.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 3,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel // 2)
        self.block = nn.Sequential(
            nn.Conv2d(
                in_ch, out_ch, kernel, stride=stride, padding=padding,
                dilation=dilation, groups=groups, bias=False,
            ),
            nn.BatchNorm2d(out_ch, momentum=0.01, eps=1e-3),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResBlock(nn.Module):
    """Two 3x3 convs with an identity skip — keeps gradients alive at depth."""

    def __init__(self, ch: int, dilation: int = 1) -> None:
        super().__init__()
        self.c1 = ConvBnAct(ch, ch, dilation=dilation)
        self.c2 = ConvBnAct(ch, ch, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.c2(self.c1(x))


class EncoderBlock(nn.Module):
    """Strided downsample followed by ``n_res`` residual blocks."""

    def __init__(self, in_ch: int, out_ch: int, n_res: int = 2) -> None:
        super().__init__()
        self.down = ConvBnAct(in_ch, out_ch, stride=2)
        self.res = nn.Sequential(*[ResBlock(out_ch) for _ in range(n_res)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res(self.down(x))


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling.

    At the bottleneck the feature map is 24x24 (384 / 16). Parallel dilated
    convs plus a global-average branch let one pixel see anything from a
    washer's rim to the whole tray without extra downsampling.
    """

    def __init__(self, in_ch: int, out_ch: int, rates: Sequence[int] = (1, 6, 12, 18)) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            [ConvBnAct(in_ch, out_ch, kernel=1 if r == 1 else 3, dilation=r) for r in rates]
        )
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch, momentum=0.01),
            nn.GELU(),
        )
        self.proj = ConvBnAct(out_ch * (len(rates) + 1), out_ch, kernel=1)
        self.drop = nn.Dropout2d(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [conv(x) for conv in self.convs]
        feats.append(self.gap(x).expand_as(feats[0]))
        return self.drop(self.proj(torch.cat(feats, dim=1)))


class DecoderBlock(nn.Module):
    """Bilinear upsample to the skip's resolution, concatenate, then refine."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            ConvBnAct(in_ch + skip_ch, out_ch),
            ConvBnAct(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))
