import inspect
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize("shape", [(3, 800_000_000), (800_000_000, 3)])
def test_batched_emd_rejects_index_products_before_allocation(cp, monkeypatch, shape):
    from types import SimpleNamespace

    from iceemdan_cupy import batched_emd as module

    class HugeView:
        ndim = 2
        dtype = np.dtype("float64")
        flags = SimpleNamespace(c_contiguous=True)

        def __init__(self, shape):
            self.shape = shape

    monkeypatch.setattr(module, "require_device_array", lambda *_: None)
    with pytest.raises(ValueError, match="int32"):
        module.batched_emd(HugeView(shape), max_modes=1, _device_result=True)


def test_batched_emd_device_skip_does_not_extract(cp):
    from iceemdan_cupy.batched_emd import batched_emd

    t = cp.arange(128, dtype="float64")
    signals = (cp.sin(0.73 * t) + 0.3 * cp.cos(0.11 * t))[None, :]
    modes, available, _, _, state = batched_emd(
        signals, max_modes=1, _device_result=True, _skip=cp.ones((), dtype="int32")
    )
    assert bool(cp.all(modes == 0))
    assert not bool(cp.any(available))
    assert int(cp.asnumpy(state)[0]) == -1


def _compare_reference(cp, data):
    from iceemdan_cupy.batched_emd import batched_extrema
    from iceemdan_cupy.contracts import extrema

    maxima, minima, zeros, counts = batched_extrema(cp.asarray(data))
    host_counts = cp.asnumpy(counts)
    host_maxima = cp.asnumpy(maxima)
    host_minima = cp.asnumpy(minima)
    host_zeros = cp.asnumpy(zeros)
    t = cp.arange(data.shape[1], dtype="float64")
    for row, signal in enumerate(data):
        reference = extrema(t, cp.asarray(signal))
        expected = [cp.asnumpy(reference[i]) for i in (0, 2, 4)]
        actual = [host_maxima[row], host_minima[row], host_zeros[row]]
        for kind in range(3):
            np.testing.assert_array_equal(actual[kind][: host_counts[row, kind]], expected[kind])
            assert host_counts[row, kind] == len(expected[kind])


def test_batched_extrema_plateaus_and_zero_runs(cp):
    rows = np.array(
        [
            [0] * 16,
            [2] * 16,
            list(range(16)),
            [0, 1, 1, 0, -1, -1, 0, 2, 2, 0, 0, 0, -2, 0, 1, 0],
            [0, 0, 1, 0, 1, 1, 0, -1, -1, 0, 0, 2, 0, 0, 0, 0],
            [0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1],
            [0, 0, 0, 1, 1, 1, 1, 0, 0, -1, -1, 0, 0, 0, 0, 0],
        ],
        dtype="float64",
    )
    _compare_reference(cp, rows)
    _compare_reference(cp, rows * 1e-240)


def test_batched_extrema_random_with_plateaus(cp):
    rng = np.random.default_rng(4)
    rows = rng.normal(size=(32, 257))
    rows[:, 21:25] = rows[:, 20:21]
    rows[:, 120:126] = 0
    _compare_reference(cp, rows)


def test_batched_extrema_kernel_is_capture_compatible(cp):
    from iceemdan_cupy.batched_emd import extrema_into

    signals = cp.asarray(np.tile([0, 1, 1, 0, -1, -1, 0, 2], (4, 1)), dtype="float64")
    outputs = [cp.empty(signals.shape, dtype="int32") for _ in range(3)]
    counts = cp.empty((len(signals), 3), dtype="int32")
    extrema_into(signals, *outputs, counts)
    cp.cuda.Device().synchronize()  # Compile before capture.
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        stream.begin_capture()
        extrema_into(signals, *outputs, counts)
        graph = stream.end_capture()
        graph.launch(stream=stream)
    stream.synchronize()
    np.testing.assert_array_equal(cp.asnumpy(counts[:, :2]), [[1, 1]] * 4)


def test_batched_reflection_matches_reference(cpu, cp):
    from iceemdan_cupy.batched_emd import batched_extrema, batched_reflection

    rng = np.random.default_rng(19)
    data = rng.normal(size=(24, 80))
    data[:, 10:13] = data[:, 9:10]
    data[:, 41:45] = 0
    signals = cp.asarray(data)
    hi, lo, zeros, counts = batched_extrema(signals)
    del zeros
    for nbsym in (1, 2, 3, 6):
        upper_x, upper_y, lower_x, lower_y, knot_counts, status = batched_reflection(
            signals, hi, lo, counts, nbsym
        )
        np.testing.assert_array_equal(cp.asnumpy(status), 0)
        lengths = cp.asnumpy(knot_counts)
        knot_arrays = [cp.asnumpy(a) for a in (upper_x, upper_y, lower_x, lower_y)]
        model = cpu.EMD(nbsym=nbsym)
        for row, signal in enumerate(data):
            t = np.arange(len(signal), dtype=float)
            reference = model.prepare_points_simple(t, signal, *model.find_extrema(t, signal)[:4])
            for kind, target in enumerate(reference):
                size = lengths[row, kind]
                np.testing.assert_array_equal(knot_arrays[2 * kind][row, :size], target[0])
                np.testing.assert_array_equal(knot_arrays[2 * kind + 1][row, :size], target[1])


def test_batched_reflection_regression_fixture(cpu, cp):
    from iceemdan_cupy.batched_emd import batched_extrema, batched_reflection

    fixture = np.load(Path(__file__).parent / "fixtures/first_divergent_mirror.npz")
    data = np.stack([fixture[name] for name in ("x_left", "x_right", "mean_left", "mean_right")])
    signals = cp.asarray(data)
    hi, lo, _, counts = batched_extrema(signals)
    ux, uy, lx, ly, lengths, status = batched_reflection(signals, hi, lo, counts)
    np.testing.assert_array_equal(cp.asnumpy(status), 0)
    arrays = [cp.asnumpy(v) for v in (ux, uy, lx, ly)]
    lengths = cp.asnumpy(lengths)
    reference = cpu.EMD()
    for row, signal in enumerate(data):
        t = np.arange(len(signal), dtype=float)
        upper, lower = reference.prepare_points_simple(
            t, signal, *reference.find_extrema(t, signal)[:4]
        )
        for kind, expected in enumerate((upper, lower)):
            size = lengths[row, kind]
            np.testing.assert_array_equal(arrays[2 * kind][row, :size], expected[0])
            np.testing.assert_array_equal(arrays[2 * kind + 1][row, :size], expected[1])


def test_batched_reflection_terminal_row_is_initialized(cp):
    from iceemdan_cupy.batched_emd import batched_extrema, reflection_into

    signals = cp.arange(16, dtype="float64")[None, :]
    hi, lo, _, counts = batched_extrema(signals)
    knots = [cp.full((1, 32), 7.0) for _ in range(4)]
    knot_counts = cp.full((1, 2), 7, dtype="int32")
    status = cp.full(1, 7, dtype="int32")
    reflection_into(signals, hi, lo, counts, *knots, knot_counts, status)
    np.testing.assert_array_equal(cp.asnumpy(status), [1])
    np.testing.assert_array_equal(cp.asnumpy(knot_counts), 0)
    for array in knots:
        np.testing.assert_array_equal(cp.asnumpy(array), 0)


def test_extrema_and_reflection_capture_together(cp):
    from iceemdan_cupy.batched_emd import extrema_into, reflection_into

    signal = cp.asarray(np.tile([0, 1, 0, -1], (4, 6)), dtype="float64")
    indices = [cp.empty(signal.shape, dtype="int32") for _ in range(3)]
    extrema_counts = cp.empty((4, 3), dtype="int32")
    knots = [cp.empty((4, 2 * signal.shape[1]), dtype="float64") for _ in range(4)]
    knot_counts = cp.empty((4, 2), dtype="int32")
    status = cp.empty(4, dtype="int32")

    def launch():
        extrema_into(signal, *indices, extrema_counts)
        reflection_into(signal, indices[0], indices[1], extrema_counts, *knots, knot_counts, status)

    launch()
    cp.cuda.Device().synchronize()  # Compile both kernels before capture.
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        stream.begin_capture()
        launch()
        graph = stream.end_capture()
        graph.launch(stream=stream)
    stream.synchronize()
    np.testing.assert_array_equal(cp.asnumpy(status), 0)
    assert np.all(cp.asnumpy(knot_counts) >= 3)


def test_batched_cubic_spline_matches_reference(cpu, cp):
    from iceemdan_cupy.batched_emd import batched_extrema, batched_reflection, batched_spline

    rng = np.random.default_rng(25)
    data = rng.normal(size=(20, 96))
    data[0] = np.sin(np.arange(96) * 0.17)
    signals = cp.asarray(data)
    hi, lo, _, counts = batched_extrema(signals)
    knots = batched_reflection(signals, hi, lo, counts)
    upper, lower, status = batched_spline(*knots[:5], samples=96)
    np.testing.assert_array_equal(cp.asnumpy(status), 0)
    actual = [cp.asnumpy(upper), cp.asnumpy(lower)]
    sizes = cp.asnumpy(knots[4])
    model = cpu.EMD()
    for row, signal in enumerate(data):
        t = np.arange(96, dtype=float)
        expected_knots = model.prepare_points_simple(t, signal, *model.find_extrema(t, signal)[:4])
        for kind in range(2):
            assert sizes[row, kind] == expected_knots[kind].shape[1]
            _, expected = model.spline_points(t, expected_knots[kind])
            np.testing.assert_allclose(actual[kind][row], expected, rtol=2e-12, atol=2e-12)


def test_batched_cubic_spline_three_knots(cpu, cp):
    from iceemdan_cupy.batched_emd import batched_spline

    x = np.array([-4.0, 5.0, 15.0])
    y = np.array([0.3, 2.0, -0.5])
    knot_x = cp.asarray(np.pad(x, (0, 17)))[None, :]
    knot_y = cp.asarray(np.pad(y, (0, 17)))[None, :]
    upper, lower, status = batched_spline(
        knot_x, knot_y, knot_x, knot_y, cp.asarray([[3, 3]], dtype="int32"), samples=10
    )
    np.testing.assert_array_equal(cp.asnumpy(status), 0)
    t = np.arange(10, dtype=float)
    _, expected = cpu.EMD().spline_points(t, np.vstack((x, y)))
    np.testing.assert_allclose(cp.asnumpy(upper)[0], expected, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(cp.asnumpy(lower)[0], expected, rtol=1e-13, atol=1e-13)


def test_batched_cubic_spline_invalid_knots_are_initialized(cp):
    from iceemdan_cupy.batched_emd import batched_spline

    x = cp.asarray([[0.0, 1.0, 1.0, 9.0] + [0.0] * 16])
    y = cp.asarray([[0.0, 1.0, 2.0, 3.0] + [0.0] * 16])
    upper, lower, status = batched_spline(x, y, x, y, cp.asarray([[4, 4]], dtype="int32"), 10)
    np.testing.assert_array_equal(cp.asnumpy(status), [[2, 2]])
    np.testing.assert_array_equal(cp.asnumpy(upper), 0)
    np.testing.assert_array_equal(cp.asnumpy(lower), 0)


def test_batched_full_geometry_capture(cp):
    from iceemdan_cupy.batched_emd import extrema_into, reflection_into, spline_into

    signal = cp.asarray(np.tile([0, 1, 0, -1], (4, 16)), dtype="float64")
    batch, n = signal.shape
    indices = [cp.empty(signal.shape, dtype="int32") for _ in range(3)]
    extrema_counts = cp.empty((batch, 3), dtype="int32")
    knots = [cp.empty((batch, 2 * n), dtype="float64") for _ in range(4)]
    knot_counts = cp.empty((batch, 2), dtype="int32")
    reflection_status = cp.empty(batch, dtype="int32")
    envelopes = [cp.empty(signal.shape, dtype="float64") for _ in range(2)]
    work = [cp.empty((batch, 2, 2 * n), dtype="float64") for _ in range(2)]
    spline_status = cp.empty((batch, 2), dtype="int32")

    def launch():
        extrema_into(signal, *indices, extrema_counts)
        reflection_into(
            signal, indices[0], indices[1], extrema_counts, *knots, knot_counts, reflection_status
        )
        spline_into(*knots, knot_counts, *envelopes, *work, spline_status)

    launch()
    cp.cuda.Device().synchronize()
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        stream.begin_capture()
        launch()
        graph = stream.end_capture()
        graph.launch(stream=stream)
    stream.synchronize()
    np.testing.assert_array_equal(cp.asnumpy(reflection_status), 0)
    np.testing.assert_array_equal(cp.asnumpy(spline_status), 0)
    assert np.all(np.isfinite(cp.asnumpy(envelopes[0])))


def test_batched_rilling_matches_reference(cpu, cp):
    from iceemdan_cupy.batched_emd import (
        batched_criterion,
        batched_extrema,
        batched_reflection,
        batched_spline,
    )

    rng = np.random.default_rng(36)
    data = rng.normal(size=(16, 128))
    data[0] = np.sin(np.arange(128) * 0.13)
    signal = cp.asarray(data)
    hi, lo, _, counts = batched_extrema(signal)
    knots = batched_reflection(signal, hi, lo, counts)
    upper, lower, spline_status = batched_spline(*knots[:5], samples=128)
    mean, metrics, decision = batched_criterion(upper, lower, counts, spline_status)
    mean, metrics, decision = (cp.asnumpy(v) for v in (mean, metrics, decision))
    model = cpu.EMD()
    t = np.arange(128, dtype=float)
    for row in range(len(data)):
        info, expected_mean = model._criterion(data[row], t)
        np.testing.assert_allclose(mean[row], expected_mean, rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(
            metrics[row],
            [info["envelope_ratio_max"], info["fraction_above_threshold"]],
            rtol=2e-12,
            atol=2e-12,
        )
        assert decision[row] == int(info["is_imf"])


def test_batched_first_imf_matches_reference(cpu, cp):
    from iceemdan_cupy.batched_emd import batched_first_imf

    t = np.arange(192, dtype=float)
    rng = np.random.default_rng(52)
    data = np.stack(
        [
            np.sin(0.18 * t) + 0.3 * np.sin(0.055 * t),
            np.sin(0.37 * t) + 0.5 * np.sin(0.07 * t),
            rng.normal(size=len(t)),
            np.linspace(-1, 1, len(t)),
        ]
    )
    modes, iterations, state = batched_first_imf(cp.asarray(data), max_iteration=100)
    modes = cp.asnumpy(modes)
    iterations = cp.asnumpy(iterations)
    state = cp.asnumpy(state)
    model = cpu.EMD(MAX_ITERATION=100)
    for row, signal in enumerate(data):
        expected, info = model._first(signal, t)
        if expected is None:
            assert state[row] == 2
            np.testing.assert_array_equal(modes[row], 0)
        else:
            assert state[row] == 1
            np.testing.assert_allclose(modes[row], expected, rtol=1e-10, atol=1e-10)
        assert iterations[row] == info["sift_iterations"]


def test_batched_emd_matches_reference_modes(cpu, cp):
    from iceemdan_cupy.batched_emd import batched_emd

    t = np.arange(192, dtype=float)
    rng = np.random.default_rng(74)
    data = np.stack(
        [
            np.sin(0.22 * t) + 0.3 * np.sin(0.055 * t),
            np.sin(0.31 * t) + 0.2 * np.sin(0.04 * t),
            rng.normal(size=len(t)),
            np.linspace(-1, 1, len(t)),
        ]
    )
    modes, available, residue, _ = batched_emd(cp.asarray(data), max_modes=3, max_iteration=100)
    modes = [cp.asnumpy(v) for v in modes]
    available = [cp.asnumpy(v) for v in available]
    residue = cp.asnumpy(residue)
    model = cpu.EMD(MAX_ITERATION=100)
    for row, signal in enumerate(data):
        expected = model.emd(signal, max_imf=3)
        for k in range(len(modes)):
            assert bool(available[k][row]) == (k < len(expected) - 1)
            np.testing.assert_allclose(
                modes[k][row], expected[k] if k < len(expected) - 1 else 0.0, rtol=1e-9, atol=1e-9
            )
        np.testing.assert_allclose(residue[row], expected[-1], rtol=1e-9, atol=1e-9)


def test_batched_emd_rejects_workspace_larger_than_device(cp):
    from iceemdan_cupy.batched_emd import batched_emd

    signal = cp.zeros((1, 128), dtype="float64")
    with pytest.raises(MemoryError, match="requires at least .* GiB"):
        batched_emd(signal, max_modes=2**31 - 1)


def test_batched_emd_device_result_keeps_status_on_gpu(cpu, cp, monkeypatch):
    from iceemdan_cupy.batched_emd import batched_emd

    t = np.arange(192.0)
    data = np.stack((np.sin(0.73 * t) + 0.3 * np.cos(0.14 * t), np.linspace(-1, 1, len(t))))
    signal = cp.asarray(data)
    original_asnumpy = cp.asnumpy

    def reject_transfer(*args, **kwargs):
        pytest.fail("EMD downloaded a device status before returning the device result.")

    with monkeypatch.context() as patcher:
        patcher.setattr(cp, "asnumpy", reject_transfer)
        modes, available, residue, counts, state = batched_emd(
            signal, max_modes=2, max_iteration=100, _device_result=True
        )
    assert all(
        isinstance(value, cp.ndarray) for value in (modes, available, residue, counts, state)
    )
    assert modes.shape == (2, 2, len(t))
    np.testing.assert_array_equal(original_asnumpy(state), [0, 0])
    np.testing.assert_array_equal(original_asnumpy(available[:, 1]), [False, False])
    model = cpu.EMD(MAX_ITERATION=100)
    expected = model.emd(data[0], max_imf=2)
    assert original_asnumpy(available[0, 0])
    np.testing.assert_allclose(original_asnumpy(modes[0, 0]), expected[0], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(original_asnumpy(residue[0]), expected[-1], rtol=1e-9, atol=1e-9)
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        stream.begin_capture()
        captured = batched_emd(signal, max_modes=2, max_iteration=100, _device_result=True)
        graph = stream.end_capture()
        graph.launch(stream=stream)
        stream.synchronize()
    np.testing.assert_array_equal(original_asnumpy(captured[-1]), [0, 0])


def test_ordered_mean_preserves_kahan_order(cp):
    from iceemdan_cupy.batched_emd import ordered_mean
    from iceemdan_cupy.ensemble import kahan_add

    rng = np.random.default_rng(82)
    data = cp.asarray(rng.normal(size=(23, 257)))
    total = cp.zeros(257, dtype="float64")
    correction = cp.zeros_like(total)
    for row in data:
        total, correction = kahan_add(total, correction, row)
    np.testing.assert_array_equal(cp.asnumpy(ordered_mean(data)), cp.asnumpy(total / len(data)))


def test_batched_ensemble_two_stages_matches_cpu(cpu, cp):
    from iceemdan_cupy import ICEEMDAN

    t = np.arange(192.0)
    signal = np.sin(0.73 * t) + 0.5 * np.cos(0.14 * t)
    noise = np.random.default_rng(11).normal(size=(6, len(signal)))
    reference = cpu.ICEEMDAN(trials=6)
    samples = iter(noise)
    reference.generate_noise = lambda scale, size: next(samples).copy() * scale
    expected = reference(signal, max_imf=2)
    model = ICEEMDAN(trials=6, batch_emd=True)
    actual = cp.asnumpy(model(signal, max_imf=2, noise=noise))
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-10)
    assert model.diagnostics_["noise_mode_counts"] == reference.diagnostics_["noise_mode_counts"]
    for a, b in zip(model.diagnostics_["stages"], reference.diagnostics_["stages"], strict=True):
        assert a["sift_iterations"] == b["sift_iterations"]


def test_batched_repeated_calls_with_partial_lengths_match_cpu(cpu, cp):
    from iceemdan_cupy import ICEEMDAN

    model = ICEEMDAN(trials=4, batch_emd=True, graph_control=False)
    reference = cpu.ICEEMDAN(trials=4)
    for n in (127, 129, 127):
        t = np.arange(float(n))
        signal = np.sin(0.71 * t) + 0.5 * np.sin(0.21 * t) + 0.3 * np.sin(0.035 * t)
        noise = np.random.default_rng(n).normal(size=(4, n))
        rows = iter(noise)
        reference.generate_noise = lambda scale, size, rows=rows: next(rows).copy() * scale
        expected = reference(signal, max_imf=2)
        actual = cp.asnumpy(model(signal, max_imf=2, noise=noise))

        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-10)
        for key in ("stop_reason", "natural_termination", "noise_mode_counts"):
            assert model.diagnostics_[key] == reference.diagnostics_[key]
        for stage, baseline in zip(
            model.diagnostics_["stages"], reference.diagnostics_["stages"], strict=True
        ):
            assert stage["missing_noise_modes"] == baseline["missing_noise_modes"]
            assert stage["sift_iterations"] == baseline["sift_iterations"]


def test_batched_stage_diagnostics_transfer_after_last_stage(cpu, cp, monkeypatch):
    from iceemdan_cupy import ICEEMDAN, ensemble

    t = np.arange(192.0)
    signal = np.sin(0.73 * t) + 0.5 * np.cos(0.14 * t)
    noise = np.random.default_rng(11).normal(size=(6, len(signal)))
    reference = cpu.ICEEMDAN(trials=6)
    rows = iter(noise)
    reference.generate_noise = lambda scale, size: next(rows).copy() * scale
    expected = reference(signal, max_imf=2)

    original_emd = ensemble.batched_emd
    original_asnumpy = cp.asnumpy
    stages_run = 0
    in_stage_emd = False

    def observed_emd(*args, **kwargs):
        nonlocal stages_run, in_stage_emd
        in_stage_emd = kwargs["max_modes"] == 1 and args[0].shape[0] == len(noise)
        try:
            result = original_emd(*args, **kwargs)
            if in_stage_emd:
                stages_run += 1
            return result
        finally:
            in_stage_emd = False

    def guarded_asnumpy(*args, **kwargs):
        caller = inspect.currentframe().f_back
        if in_stage_emd:
            pytest.fail("A batched stage EMD downloaded its status before returning.")
        if caller.f_code.co_name == "ceemdan" and stages_run < 2:
            pytest.fail("Stage diagnostics were transferred before the last EMD stage.")
        return original_asnumpy(*args, **kwargs)

    monkeypatch.setattr(ensemble, "batched_emd", observed_emd)
    monkeypatch.setattr(cp, "asnumpy", guarded_asnumpy)
    model = ICEEMDAN(trials=6, batch_emd=True)
    actual = model(signal, max_imf=2, noise=noise)
    assert stages_run == 2
    np.testing.assert_allclose(original_asnumpy(actual), expected, rtol=1e-9, atol=1e-10)
    for a, b in zip(model.diagnostics_["stages"], reference.diagnostics_["stages"], strict=True):
        assert a["missing_noise_modes"] == b["missing_noise_modes"]
        np.testing.assert_allclose(a["perturbation_stds"], b["perturbation_stds"], atol=1e-10)
        assert a["sift_iterations"] == b["sift_iterations"]


def test_batched_ensemble_missing_noise_modes_keep_denominator(cpu, cp):
    from iceemdan_cupy import ICEEMDAN

    t = np.arange(192.0)
    signal = np.sin(0.43 * t) + 0.3 * np.sin(0.07 * t)
    rng = np.random.default_rng(9)
    noise = np.stack(
        [
            np.sin(0.34 * t),
            rng.normal(size=len(t)),
            np.sin(0.27 * t),
            rng.normal(size=len(t)),
        ]
    )
    reference = cpu.ICEEMDAN(trials=4)
    samples = iter(noise)
    reference.generate_noise = lambda scale, size: next(samples).copy() * scale
    expected = reference(signal, max_imf=3)
    model = ICEEMDAN(trials=4, batch_emd=True)
    actual = cp.asnumpy(model(signal, max_imf=3, noise=noise))
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-10)
    assert model.diagnostics_["noise_mode_counts"] == [1, 3, 1, 3]
    assert [stage["missing_noise_modes"] for stage in model.diagnostics_["stages"]] == [0, 2, 2]
    assert all(stage["ensemble_denominator"] == 4 for stage in model.diagnostics_["stages"])


def test_batched_ensemble_generated_noise_matches_scalar_port(cp):
    from iceemdan_cupy import ICEEMDAN

    t = np.arange(128.0)
    signal = np.sin(0.49 * t) + 0.4 * np.sin(0.09 * t)
    scalar = ICEEMDAN(trials=5, seed=42)
    batched = ICEEMDAN(trials=5, seed=42, batch_emd=True)
    expected = cp.asnumpy(scalar(signal, max_imf=2))
    actual = cp.asnumpy(batched(signal, max_imf=2))
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-10)
    assert scalar.diagnostics_["noise_mode_counts"] == batched.diagnostics_["noise_mode_counts"]


@pytest.mark.parametrize("ddof", [0, 1])
def test_batched_early_stop_codes_preserve_reason_priority(cp, ddof):
    from iceemdan_cupy import ICEEMDAN

    t = np.arange(128.0)
    wave = np.sin(0.7 * t)
    cases = [
        (np.linspace(-1, 1, len(t)), {}, 1, "fewer_than_three_extrema"),
        (1e-16 * wave, {}, 2, "numerical_zero"),
        (wave, {"residue_std_threshold": 2.0}, 3, "residue_std_threshold"),
        (wave, {"range_thr": 3.0}, 4, "range_threshold"),
        (wave, {"total_power_thr": 200.0}, 5, "residue_l1_threshold"),
        (wave, {}, 0, None),
        (
            np.linspace(-1, 1, len(t)),
            {"residue_std_threshold": 2.0, "range_thr": 3.0},
            1,
            "fewer_than_three_extrema",
        ),
        (
            wave,
            {"residue_std_threshold": 2.0, "range_thr": 3.0},
            3,
            "residue_std_threshold",
        ),
    ]
    for signal, kwargs, expected_code, expected_reason in cases:
        model = ICEEMDAN(trials=2, batch_emd=True, ddof=ddof, **kwargs)
        residue = cp.asarray(signal)
        assert int(cp.asnumpy(model._early_stop_code(residue))) == expected_code
        assert model._stop_reason(residue, cp.asarray(t), False) == expected_reason


def test_batched_early_stop_avoids_variable_length_extrema(cp, monkeypatch):
    from iceemdan_cupy import ICEEMDAN, ensemble

    def reject_extrema(*args):
        pytest.fail("The batched stop path used variable-length extrema.")

    monkeypatch.setattr(ensemble, "extrema", reject_extrema)
    t = cp.arange(128, dtype="float64")
    residue = cp.linspace(-1, 1, len(t))
    assert ICEEMDAN(trials=2, batch_emd=True)._stop_reason(residue, t, False) == (
        "fewer_than_three_extrema"
    )


@pytest.mark.parametrize("ddof", [0, 1])
def test_batched_residue_std_matches_stable_reference(cp, ddof):
    from iceemdan_cupy import ICEEMDAN
    from iceemdan_cupy.contracts import stable_std

    model = ICEEMDAN(trials=2, batch_emd=True, ddof=ddof)
    for samples in (
        np.zeros(128),
        np.sin(0.7 * np.arange(128.0)),
        1e300 * np.sin(0.7 * np.arange(128.0)),
    ):
        residue = cp.asarray(samples)
        np.testing.assert_allclose(
            cp.asnumpy(model._device_std(residue)), stable_std(residue, ddof), rtol=1e-15
        )


def test_batched_stage_std_stays_on_device(cpu, cp, monkeypatch):
    from iceemdan_cupy import ICEEMDAN, ensemble

    t = np.arange(192.0)
    signal = np.sin(0.73 * t) + 0.5 * np.cos(0.14 * t)
    noise = np.random.default_rng(11).normal(size=(6, len(signal)))
    reference = cpu.ICEEMDAN(trials=6)
    rows = iter(noise)
    reference.generate_noise = lambda scale, size: next(rows).copy() * scale
    expected = reference(signal, max_imf=2)

    original_std = ensemble.stable_std
    calls = 0

    def observed_std(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_std(*args, **kwargs)

    monkeypatch.setattr(ensemble, "stable_std", observed_std)
    model = ICEEMDAN(trials=6, batch_emd=True)
    actual = cp.asnumpy(model(signal, max_imf=2, noise=noise))
    assert calls == 1  # Initial input normalization only.
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(
        [stage["residue_std_before"] for stage in model.diagnostics_["stages"]],
        [stage["residue_std_before"] for stage in reference.diagnostics_["stages"]],
        rtol=1e-15,
    )


def test_batched_stage_status_preserves_error_priority(cp):
    from iceemdan_cupy.ensemble import _stage_status

    previous = cp.asarray([1.0, 2.0, 3.0])
    for candidate, expected in (
        ([0.5, 2.0, 3.0], 0),
        ([1.0, 2.0, 3.0], 2),
        ([float("nan"), 2.0, 3.0], 1),
        ([float("inf"), 2.0, 3.0], 1),
    ):
        assert int(cp.asnumpy(_stage_status(previous, cp.asarray(candidate)))) == expected


def test_batched_noise_bank_keeps_dense_modes_and_counts_on_device(cp):
    from iceemdan_cupy import ICEEMDAN

    t = np.arange(192.0)
    rng = np.random.default_rng(9)
    noise = cp.asarray(
        np.stack(
            [
                np.sin(0.34 * t),
                rng.normal(size=len(t)),
                np.sin(0.27 * t),
                rng.normal(size=len(t)),
            ]
        )
    )
    model = ICEEMDAN(trials=4, batch_emd=True)
    model._noise_cap = 3
    model._supplied_noise = noise
    modes, available, first_std, counts, used, _, _ = model._batched_noise_bank((len(t),))
    assert isinstance(modes, cp.ndarray) and modes.shape == (3, 4, len(t))
    assert isinstance(available, cp.ndarray) and available.shape == (3, 4)
    assert isinstance(first_std, cp.ndarray) and first_std.shape == (4,)
    assert isinstance(counts, cp.ndarray) and counts.shape == (4,)
    assert isinstance(used, cp.ndarray) and used.shape == ()
    assert cp.asnumpy(counts).tolist() == [1, 3, 1, 3]
    assert int(cp.asnumpy(used)) == 3


def test_batched_imf_stop_codes_match_existing_reason(cpu, cp):
    from scipy.interpolate import CubicSpline

    from iceemdan_cupy import ICEEMDAN

    model = ICEEMDAN(trials=2, batch_emd=True)
    reference = cpu.ICEEMDAN(trials=2)
    t = np.arange(192.0)
    short_t = np.arange(64.0)
    terminal = CubicSpline(
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
    )(short_t)
    for signal, expected_code, expected_reason in (
        (np.sin(0.73 * t), 6, "residue_is_imf"),
        (np.sin(0.73 * t) + 0.5 * np.cos(0.14 * t), 0, None),
        (terminal, 7, "emd_has_no_extractable_imf"),
    ):
        residue = cp.asarray(signal)
        assert int(cp.asnumpy(model._early_stop_code(residue))) == 0
        assert int(cp.asnumpy(model._imf_stop_code(residue))) == expected_code
        assert reference._stop_reason(signal, np.arange(len(signal), dtype=float), True) == (
            expected_reason
        )
        assert model._stop_reason(residue, cp.arange(len(residue), dtype="float64"), True) == (
            expected_reason
        )


def test_batched_imf_stop_preserves_convergence_error(cp):
    from iceemdan_cupy import ICEEMDAN, SiftingConvergenceError

    t = cp.arange(192, dtype="float64")
    residue = cp.asarray(np.random.default_rng(0).normal(size=len(t)))
    model = ICEEMDAN(trials=2, batch_emd=True, emd_params={"MAX_ITERATION": 1})
    assert int(cp.asnumpy(model._imf_stop_code(residue))) == 8
    with pytest.raises(SiftingConvergenceError):
        model._stop_reason(residue, t, True)


def test_batched_imf_stop_uses_device_emd(cp, monkeypatch):
    from iceemdan_cupy import ICEEMDAN

    model = ICEEMDAN(trials=2, batch_emd=True)
    t = cp.arange(192, dtype="float64")
    residue = cp.sin(0.73 * t)

    def reject_scalar_criterion(*args):
        pytest.fail("The batched stop path used scalar EMD diagnostics.")

    monkeypatch.setattr(model.EMD, "imf_diagnostics", reject_scalar_criterion)
    assert model._stop_reason(residue, t, True) == "residue_is_imf"
