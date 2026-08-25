"""Run-length encoding for the Kaggle submission format.

The competition uses **column-major (Fortran) order** with 1-based start
indices, i.e. pixels are numbered top-to-bottom then left-to-right. Getting
this backwards is the single easiest way to score ~0 with a perfectly good
model, so both directions are covered by tests/test_rle.py.
"""

from __future__ import annotations

import numpy as np

EMPTY_RLE = "1 1"  # placeholder the competition accepts for "no prediction"


def rle_encode(mask: np.ndarray) -> str:
    """Encode a binary 2-D mask as a run-length string.

    Args:
        mask: 2-D array-like of 0/1 (or bool).

    Returns:
        Space-separated "start length start length ..." string.
    """
    pixels = np.asarray(mask, dtype=np.uint8).T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(int(x)) for x in runs)


def rle_decode(rle: str, size: int = 384) -> np.ndarray:
    """Decode a run-length string back into a ``size x size`` uint8 mask."""
    flat = np.zeros(size * size, dtype=np.uint8)
    if not isinstance(rle, str) or not rle.strip():
        return flat.reshape(size, size, order="F")
    values = np.asarray(rle.split(), dtype=np.int64)
    for start, length in zip(values[0::2], values[1::2]):
        flat[start - 1 : start - 1 + length] = 1
    return flat.reshape(size, size, order="F")
