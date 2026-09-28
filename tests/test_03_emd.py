import numpy as np
import pytest


@pytest.mark.parametrize(
    "kind", ["cubic", "pchip", "akima", "cubic_hermite", "linear", "slinear", "quadratic"]
)
def test_single_imf_and_local_mean(cpu, gpu, cp, kind):
    n = np.arange(256.0)
    x = np.sin(0.73 * n) + 0.4 * np.cos(0.12 * n)
    a = cpu.EMD(spline_kind=kind)
    b = gpu.EMD(spline_kind=kind)
    expected = a(x, max_imf=1)
    actual = b(x, max_imf=1)
    np.testing.assert_allclose(cp.asnumpy(actual), expected, rtol=1e-9, atol=1e-10)
    assert a.diagnostics_[0]["sift_iterations"] == b.diagnostics_[0]["sift_iterations"]
    from iceemdan_cupy.emd import local_mean

    mean, _ = local_mean(b, cp.asarray(x), cp.asarray(n))
    np.testing.assert_allclose(
        cp.asnumpy(mean), x - a.get_imfs_and_residue()[0][0], rtol=1e-9, atol=1e-10
    )


@pytest.mark.parametrize("n", [128, 256])
def test_zero_residue_is_not_an_imf(cpu, gpu, cp, n):
    x = np.sin(2 * np.pi * np.arange(n) / 16)
    b = gpu.EMD()
    b(x, max_imf=1)
    from iceemdan_cupy.emd import local_mean

    m, _ = local_mean(b, cp.asarray(x), None)
    np.testing.assert_allclose(cp.asnumpy(m), 0, atol=1e-13)


@pytest.mark.parametrize("x", [np.linspace(-1, 1, 20), np.ones(16)])
def test_nonextractable_local_mean_equals_input(gpu, cp, x):
    from iceemdan_cupy.emd import local_mean

    m, _ = local_mean(gpu.EMD(), cp.asarray(x), None)
    np.testing.assert_array_equal(cp.asnumpy(m), x)


def test_no_false_convergence(gpu):
    with pytest.raises(gpu.SiftingConvergenceError):
        gpu.EMD(MAX_ITERATION=1)(np.random.default_rng(9).normal(size=128))


def test_masked_division_zero_cases(gpu, cp):
    e = gpu.EMD()
    # Directly isolate ratio from real extrema: upper/lower zeros at one sample.
    x = cp.asarray([0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0])
    t = cp.asarray(np.arange(9.0))
    e.extract_max_min_spline = lambda *args: (
        cp.asarray([0.0, 1, 1, 1, 1, 1, 1, 1, 1]),
        cp.asarray([0.0, -1, -1, -1, -1, -1, -1, -1, -1]),
        None,
        None,
    )
    info, m = e._criterion(x, t)
    assert info["envelope_ratio_max"] == 0
    assert info["is_imf"]


def test_trace_does_not_change_results(gpu, cp):
    n = np.arange(128.0)
    x = np.sin(0.9 * n) + 0.6 * np.cos(0.19 * n)
    events = []
    a = gpu.EMD()
    b = gpu.EMD(trace=lambda event, data: events.append(event))
    np.testing.assert_array_equal(cp.asnumpy(a(x, max_imf=1)), cp.asnumpy(b(x, max_imf=1)))
    assert "envelopes" in events and "criterion" in events


def test_zero_amplitude_nonzero_mean_remains_infinite(gpu, cp):
    e = gpu.EMD()
    x = cp.asarray([0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0])
    t = cp.asarray(np.arange(9.0))
    e.extract_max_min_spline = lambda *args: (cp.ones(9), cp.ones(9), None, None)
    info, mean = e._criterion(x, t)
    assert info["envelope_ratio_max"] == float("inf")
    assert info["fraction_above_threshold"] == 1
    assert not info["is_imf"]
