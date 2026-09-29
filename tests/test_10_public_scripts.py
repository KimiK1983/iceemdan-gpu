"""Small public-data parser check without downloading or redistributing recordings."""

import numpy as np
import pytest

from scripts.analyze_cudb import decode_212


def test_decode_212_signed_limits_and_length():
    samples = decode_212(bytes((0x00, 0xF0, 0xFF, 0xFF, 0x87, 0x00)))
    np.testing.assert_array_equal(samples, [0, -1, 2047, -2048])
    with pytest.raises(ValueError, match="divisible by three"):
        decode_212(b"\x00")
