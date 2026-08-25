"""Shape and mode contracts for ScratchUNet.

Uses a narrow model so the test runs on CPU in a couple of seconds; the width
does not affect the contracts being checked.
"""

import torch

from src.losses import CombinedLoss
from src.models import ScratchUNet

TINY = dict(
    num_classes=7,
    stem_channels=8,
    encoder_channels=(8, 16, 16, 32),
    encoder_blocks=(1, 1, 1, 1),
    aspp_channels=16,
    aspp_rates=(1, 2, 4),
    decoder_channels=(16, 8, 8, 8),
)


def test_train_mode_returns_three_heads_at_input_resolution():
    model = ScratchUNet(**TINY).train()
    x = torch.randn(2, 3, 64, 64)
    main, aux3, aux2 = model(x)
    for out in (main, aux3, aux2):
        assert out.shape == (2, 7, 64, 64)


def test_eval_mode_returns_only_the_main_head():
    model = ScratchUNet(**TINY).eval()
    out = model(torch.randn(1, 3, 64, 64))
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 7, 64, 64)


def test_deep_supervision_loss_is_differentiable():
    model = ScratchUNet(**TINY).train()
    criterion = CombinedLoss()
    x = torch.randn(2, 3, 64, 64)
    y = torch.randint(0, 7, (2, 64, 64))

    main, aux3, aux2 = model(x)
    loss = criterion(main, y) + 0.4 * criterion(aux3, y) + 0.2 * criterion(aux2, y)
    loss.backward()

    assert torch.isfinite(loss)
    assert model.stem[0].block[0].weight.grad is not None
