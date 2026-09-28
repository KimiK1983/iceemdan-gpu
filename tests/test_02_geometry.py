from pathlib import Path

import numpy as np
import pytest

KINDS = ["cubic", "pchip", "akima", "cubic_hermite", "linear", "slinear", "quadratic"]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("seed", [1, 7, 21])
def test_interpolators(cpu, gpu, cp, kind, seed):
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.uniform(0.5, 2, 9))
    y = rng.normal(size=9)
    t = np.linspace(x[0], x[-1], 101)
    ext = np.stack((x, y))
    expected = cpu.EMD(spline_kind=kind).spline_points(t, ext)[1]
    actual = gpu.EMD(spline_kind=kind).spline_points(cp.asarray(t), cp.asarray(ext))[1]
    np.testing.assert_allclose(cp.asnumpy(actual), expected, rtol=1e-10, atol=1e-10)


def test_three_point_natural_not_not_a_knot(cpu, gpu, cp):
    from iceemdan_cupy.geometry import cubic_spline_3pts

    x = np.array([0.0, 1.0, 3.0])
    y = np.array([0.0, 2.0, 1.0])
    t = np.linspace(0, 3, 301)
    expect = cpu.cubic_spline_3pts(x, y, t)[1]
    actual = cubic_spline_3pts(cp.asarray(x), cp.asarray(y), cp.asarray(t))[1]
    np.testing.assert_allclose(cp.asnumpy(actual), expect, rtol=1e-13, atol=1e-13)
    assert abs(float(actual[200]) - 2.125) < 1e-13


@pytest.mark.parametrize("nbsym", [1, 2, 3, 6])
@pytest.mark.parametrize("seed", [3, 9, 14])
def test_mirror_exact_indices_and_values(cpu, gpu, cp, nbsym, seed):
    x = np.random.default_rng(seed).normal(size=80)
    t = np.arange(80.0)
    a = cpu.EMD(nbsym=nbsym)
    b = gpu.EMD(nbsym=nbsym)
    ar = a.find_extrema(t, x)
    br = b.find_extrema(cp.asarray(t), cp.asarray(x))
    aa = a.prepare_points_simple(t, x, *ar[:4])
    bb = b.prepare_points_simple(cp.asarray(t), cp.asarray(x), *br[:4])
    for u, v in zip(aa, bb, strict=True):
        np.testing.assert_array_equal(u, cp.asnumpy(v))


def test_duplicate_removal_keeps_last(gpu, cp):
    from iceemdan_cupy.geometry import keep_last_adjacent

    ext = cp.asarray([[0, 1, 1, 2, 2, 2, 3], [3, 4, 5, 6, 7, 8, 9.0]])
    out = keep_last_adjacent(ext)
    np.testing.assert_array_equal(cp.asnumpy(out), [[0, 1, 2, 3], [3, 5, 8, 9]])


def test_adversarial_mirror_same_input_exact(cpu, gpu, cp):
    p = Path(__file__).parent / "fixtures/first_divergent_mirror.npz"
    fixture = np.load(p)
    tested = 0
    for key in fixture.files:
        x = fixture[key]
        if x.ndim != 1 or x.size < 16 or x.dtype.kind != "f":
            continue
        t = np.arange(len(x), dtype=float)
        a = cpu.EMD()
        b = gpu.EMD()
        ar = a.find_extrema(t, x)
        br = b.find_extrema(cp.asarray(t), cp.asarray(x))
        if len(ar[0]) + len(ar[2]) < 3:
            continue
        for u, v in zip(
            a.prepare_points_simple(t, x, *ar[:4]),
            b.prepare_points_simple(cp.asarray(t), cp.asarray(x), *br[:4]),
            strict=True,
        ):
            np.testing.assert_array_equal(u, cp.asnumpy(v))
        tested += 1
    assert tested > 0
