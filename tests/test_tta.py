"""The TTA inverse transforms are easy to get subtly wrong.

If a forward/inverse pair is mismatched, predictions still *look* plausible —
they just quietly blur toward the mean. So the round-trip is asserted rather
than trusted.
"""

import torch
import torch.nn as nn

from src.tta import _TRANSFORMS, predict_plain, predict_tta

DEVICE = torch.device("cpu")


class LogitPassthrough(nn.Module):
    """Returns its input unchanged — an exactly equivariant 'model'."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def test_transform_pairs_are_inverses():
    x = torch.randn(1, 7, 16, 16)
    for i, (forward, inverse) in enumerate(_TRANSFORMS):
        assert torch.equal(inverse(forward(x)), x), f"transform {i} is not invertible"


def test_group_has_eight_distinct_views():
    x = torch.arange(16 * 16, dtype=torch.float32).reshape(1, 1, 16, 16)
    views = {forward(x).numpy().tobytes() for forward, _ in _TRANSFORMS}
    assert len(views) == 8


def test_tta_is_identity_for_an_equivariant_model():
    """Averaging 8 correctly-inverted views of an equivariant model changes nothing."""
    model = LogitPassthrough().eval()
    logits = torch.randn(7, 24, 24)
    tta = predict_tta(model, logits, DEVICE, amp=False)
    plain = predict_plain(model, logits, DEVICE, amp=False)
    assert torch.allclose(tta, plain, atol=1e-5)


def test_tta_output_is_a_probability_distribution():
    model = LogitPassthrough().eval()
    probs = predict_tta(model, torch.randn(7, 24, 24), DEVICE, amp=False)
    assert probs.shape == (7, 24, 24)
    assert torch.allclose(probs.sum(0), torch.ones(24, 24), atol=1e-5)
