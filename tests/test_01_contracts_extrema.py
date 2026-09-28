import numpy as np
import pytest


@pytest.mark.parametrize(
    "x",
    [
        [0.0],
        [0.0, 0.0],
        [0.0] * 9,
        [2.0] * 9,
        list(range(9)),
        [0, 1, 1, 0],
        [0, 0, 1, 0],
        [0, 1, 1, 0, -1, -1, 0],
        [0, 1, 0, -1, 0],
        [0, 1e-240, 0, -1e-240, 0],
    ],
)
def test_exact_extrema_and_zeros(cpu, gpu, cp, x):
    from iceemdan_cupy.contracts import extrema

    s = np.array(x, dtype=float)
    t = np.arange(len(s), dtype=float)
    a = cpu._extrema(t, s)
    b = extrema(cp.asarray(t), cp.asarray(s))
    for u, v in zip(a, b, strict=True):
        np.testing.assert_array_equal(u, cp.asnumpy(v))


@pytest.mark.parametrize(
    "x", [[], [[1, 2], [3, 4]], [0, float("nan"), 1], [0, float("inf"), 1], [1 + 1j, 2, 3]]
)
def test_invalid_signal(cpu, gpu, x):
    with pytest.raises((ValueError, TypeError)):
        gpu.EMD().emd(x)


@pytest.mark.parametrize("x", [[0], [7], [2, 2], [0] * 16, list(range(16))])
def test_degenerate_no_noise(cpu, gpu, cp, x):
    expected = cpu.ICEEMDAN(trials=2)(np.asarray(x))
    actual = gpu.ICEEMDAN(trials=2)(x)
    np.testing.assert_array_equal(expected, cp.asnumpy(actual))


@pytest.mark.parametrize("ddof", [0, 1])
def test_std_and_even_median(cpu, gpu, cp, ddof):
    from iceemdan_cupy.contracts import stable_std, time_vector

    x = np.array([1, 2, 7, 3, 8, 4.0]) * 1e-200
    assert np.isclose(stable_std(cp.asarray(x), ddof) / cpu._stable_std(x, ddof), 1, rtol=1e-14)
    out = time_vector(np.arange(8) * 0.2, 8)
    assert out[1] == cpu._time_vector(np.arange(8) * 0.2, 8)[1]


@pytest.mark.parametrize(
    "t", [np.zeros(8), np.arange(8)[::-1], np.r_[np.arange(7), 9], np.r_[np.arange(7), np.nan]]
)
def test_time_rejected(gpu, t):
    with pytest.raises(ValueError):
        gpu.EMD()(np.arange(8.0), T=t)
