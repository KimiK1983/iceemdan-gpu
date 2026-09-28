"""Numerical contract regressions requiring real CUDA."""

import numpy as np
import pytest


@pytest.mark.parametrize("route", ["scalar", "batch", "graph"])
def test_first_noise_imf_scale_extremes(gpu, cp, route):
    t = np.arange(192.0)
    x = np.sin(0.73 * t) + 0.5 * np.cos(0.14 * t)
    w = np.sin(0.34 * t)[None, :]
    opts = {"batch_emd": route != "scalar", "graph_control": route == "graph"}
    outputs = []
    for scale in (1.0, 1e-310, 1e150):
        model = gpu.ICEEMDAN(trials=1, **opts)
        out = cp.asnumpy(model(x, noise=scale * w, max_imf=1))
        assert np.isfinite(out).all()
        assert model.diagnostics_["stop_reason"] == "max_imf"
        outputs.append(out)
    for out in outputs[1:]:
        # Subnormal input samples are quantized before any backend sees them.
        np.testing.assert_allclose(out, outputs[0], rtol=1e-7, atol=1e-8)


def test_ordinary_shared_noise_preserves_components_and_states(cpu, gpu, cp):
    t = np.arange(192.0)
    x = np.sin(0.73 * t) + 0.5 * np.cos(0.14 * t)
    W = np.random.default_rng(11).normal(size=(4, len(t)))
    reference = cpu.ICEEMDAN(trials=4)
    rows = iter(W)
    reference.generate_noise = lambda scale, size: next(rows).copy() * scale
    expected = reference(x, max_imf=2)
    for route in ("scalar", "batch", "graph"):
        model = gpu.ICEEMDAN(trials=4, batch_emd=route != "scalar", graph_control=route == "graph")
        actual = cp.asnumpy(model(x, noise=W, max_imf=2))
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-10)
        assert model.diagnostics_["stop_reason"] == reference.diagnostics_["stop_reason"]
        assert (
            model.diagnostics_["natural_termination"]
            == reference.diagnostics_["natural_termination"]
        )
        assert (
            model.diagnostics_["noise_mode_counts"] == reference.diagnostics_["noise_mode_counts"]
        )
        assert len(model.diagnostics_["stages"]) == len(reference.diagnostics_["stages"])
        for stage, baseline in zip(
            model.diagnostics_["stages"], reference.diagnostics_["stages"], strict=True
        ):
            assert stage["sift_iterations"] == baseline["sift_iterations"]


def test_three_stages_shared_noise_matches_cpu_scalar_batch_graph(cpu, gpu, cp):
    t = np.arange(192.0)
    x = np.sin(0.71 * t) + 0.5 * np.sin(0.21 * t) + 0.3 * np.sin(0.035 * t)
    W = np.random.default_rng(7).normal(size=(4, len(t)))
    reference = cpu.ICEEMDAN(trials=4)
    rows = iter(W)
    reference.generate_noise = lambda scale, size: next(rows).copy() * scale
    expected = reference(x, max_imf=3)
    assert expected.shape == (4, len(t))
    for route in ("scalar", "batch", "graph"):
        model = gpu.ICEEMDAN(trials=4, batch_emd=route != "scalar", graph_control=route == "graph")
        actual = cp.asnumpy(model(x, noise=W, max_imf=3))
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-10)
        assert model.diagnostics_["stop_reason"] == reference.diagnostics_["stop_reason"]
        assert (
            model.diagnostics_["noise_mode_counts"] == reference.diagnostics_["noise_mode_counts"]
        )
        assert [s["sift_iterations"] for s in model.diagnostics_["stages"]] == [
            s["sift_iterations"] for s in reference.diagnostics_["stages"]
        ]


def test_backend_reconstruction_extreme_values(gpu, cp):
    from iceemdan_cupy.emd import backend_parts

    class Backend:
        def __init__(self, mode, residue):
            self.mode, self.residue = mode, residue

        def emd(self, *args, **kwargs):
            pass

        def get_imfs_and_residue(self):
            return self.mode, self.residue

    t = cp.arange(8.0)
    tiny = cp.full(8, 1e-300)
    invalid = Backend(cp.full((1, 8), 1e308), cp.full(8, -1e308))
    with pytest.raises(gpu.BackendContractError, match="reconstruction"):
        backend_parts(invalid, tiny, t, 1)
    for signal, modes, residue in (
        (cp.zeros(8), cp.ones((1, 8)), -cp.ones(8)),
        (tiny, cp.asarray([[1e308] * 8, [-1e308] * 8]), tiny.copy()),
    ):
        actual_modes, actual_residue, _ = backend_parts(
            Backend(modes, residue), signal, t, len(modes)
        )
        np.testing.assert_array_equal(cp.asnumpy(actual_modes), cp.asnumpy(modes))
        np.testing.assert_array_equal(cp.asnumpy(actual_residue), cp.asnumpy(residue))
    with pytest.raises(gpu.BackendContractError, match="non-finite"):
        backend_parts(Backend(cp.full((1, 8), np.nan), tiny), tiny, t, 1)
    for scale in (1e-320, 1e-300, 1e300):
        signal = cp.full(8, scale)
        valid = Backend(cp.full((1, 8), scale / 2), cp.full(8, scale / 2))
        modes, residue, _ = backend_parts(valid, signal, t, 1)
        np.testing.assert_allclose(cp.asnumpy(modes[0] + residue), scale, rtol=1e-14)


@pytest.mark.parametrize("batch", [False, True])
def test_epsilon_zero_no_extractable_imf_is_terminal(gpu, cp, batch):
    from scipy.interpolate import CubicSpline

    x = CubicSpline(
        np.linspace(0, 63, 7),
        [
            1.6324326927834527,
            -1.877901750973295,
            -1.6079837666476737,
            -1.317525148354537,
            -0.08275273999265884,
            0.033899292086251885,
            0.029076569410149824,
        ],
    )(np.arange(64.0))
    assert gpu.EMD()(x).shape == (1, len(x))
    model = gpu.ICEEMDAN(trials=2, epsilon=0, batch_emd=batch)
    out = cp.asnumpy(model(x))
    np.testing.assert_allclose(out, x[None, :], rtol=0, atol=1e-14)
    assert model.diagnostics_["stop_reason"] == "emd_has_no_extractable_imf"
    assert model.diagnostics_["natural_termination"] is True


def test_epsilon_zero_real_stagnation_still_raises(gpu, cp):
    class ZeroMode:
        def emd(self, signal, t, max_imf):
            self.signal = signal

        def get_imfs_and_residue(self):
            return cp.zeros((1, len(self.signal))), self.signal.copy()

    t = np.arange(128.0)
    x = np.sin(0.7 * t) + 0.3 * np.cos(0.13 * t)
    with pytest.raises(gpu.SiftingConvergenceError, match="no progress"):
        gpu.ICEEMDAN(trials=1, epsilon=0, ext_EMD=ZeroMode())(x, max_imf=1)


@pytest.mark.parametrize("route", ["batch", "graph"])
def test_accelerated_emd_configuration_is_enforced(gpu, route):
    t = np.arange(192.0)
    x = np.sin(0.73 * t) + 0.5 * np.cos(0.14 * t)
    W = np.random.default_rng(11).normal(size=(2, len(t)))
    opts = {"batch_emd": True, "graph_control": route == "graph"}
    model = gpu.ICEEMDAN(trials=2, emd_params={"max_modes": 1}, **opts)
    with pytest.raises(ValueError, match="max_modes"):
        model(x, noise=W, max_imf=2)
    assert model(x, noise=W, max_imf=1).shape == (2, len(t))
    with pytest.raises(ValueError, match="DTYPE|float64"):
        gpu.ICEEMDAN(trials=2, emd_params={"DTYPE": "float32"}, **opts)


@pytest.mark.parametrize("route", ["scalar", "batch", "graph"])
def test_initial_threshold_is_not_natural_termination(gpu, cp, route):
    t = np.arange(192.0)
    x = np.sin(0.73 * t) + 0.5 * np.cos(0.14 * t)
    model = gpu.ICEEMDAN(
        trials=2, range_thr=100, batch_emd=route != "scalar", graph_control=route == "graph"
    )
    out = cp.asnumpy(model(x))
    np.testing.assert_array_equal(out, x[None, :])
    assert model.diagnostics_["stop_reason"] == "range_threshold"
    assert model.diagnostics_["natural_termination"] is False


@pytest.mark.parametrize("route", ["scalar", "batch", "graph"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_reconstruction_diagnostic_describes_public_output(gpu, cp, route, dtype):
    t = np.arange(192.0)
    x = np.sin(0.73 * t) + 0.5 * np.cos(0.14 * t)
    W = np.random.default_rng(11).normal(size=(2, len(t)))
    model = gpu.ICEEMDAN(
        trials=2, dtype=dtype, batch_emd=route != "scalar", graph_control=route == "graph"
    )
    out = cp.asnumpy(model(x, noise=W, max_imf=1))
    peak = np.max(np.abs(x))
    expected = np.max(np.abs(np.sum(out.astype(np.float64) / peak, axis=0) - x / peak))
    assert model.diagnostics_["reconstruction_scaled_linf"] == pytest.approx(expected, abs=1e-15)
    assert "reconstruction_scaled_linf_internal" in model.diagnostics_
    if dtype == "float32":
        assert (
            model.diagnostics_["reconstruction_scaled_linf"]
            > model.diagnostics_["reconstruction_scaled_linf_internal"]
        )
    assert model.diagnostics_["noise_bytes_is_peak_vram"] is False
