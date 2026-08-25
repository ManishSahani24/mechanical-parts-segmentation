"""RLE round-trip tests.

Column-major vs row-major RLE is the classic silent-zero bug in this kind of
competition, so it gets a test rather than a comment.
"""

import numpy as np
import pytest

from src.rle import EMPTY_RLE, rle_decode, rle_encode


def test_roundtrip_rectangle():
    mask = np.zeros((384, 384), np.uint8)
    mask[5:50, 10:60] = 1
    assert np.array_equal(rle_decode(rle_encode(mask)), mask)


def test_roundtrip_random():
    rng = np.random.RandomState(0)
    mask = (rng.rand(384, 384) > 0.7).astype(np.uint8)
    assert np.array_equal(rle_decode(rle_encode(mask)), mask)


def test_column_major_order():
    """A single pixel at (row=1, col=0) must encode as start index 2."""
    mask = np.zeros((4, 4), np.uint8)
    mask[1, 0] = 1
    assert rle_encode(mask) == "2 1"


def test_empty_mask_encodes_to_nothing():
    assert rle_encode(np.zeros((384, 384), np.uint8)) == ""


def test_empty_placeholder_decodes_to_single_pixel():
    decoded = rle_decode(EMPTY_RLE)
    assert decoded.sum() == 1


@pytest.mark.parametrize("bad", [None, "", "   ", float("nan")])
def test_invalid_input_returns_blank(bad):
    assert rle_decode(bad).sum() == 0
