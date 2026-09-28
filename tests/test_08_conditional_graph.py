"""CUDA conditional WHILE compatibility on the production EMD kernel."""

import numpy as np
import pytest


def test_graph_stages_rejects_overflow_before_device_allocation(cp, monkeypatch):
    from types import SimpleNamespace

    from iceemdan_cupy.gpu_loop import graph_stages

    def reject_allocation(*args, **kwargs):
        raise AssertionError("device allocation happened before index validation")

    monkeypatch.setattr(cp, "zeros", reject_allocation)
    model = SimpleNamespace(epsilon=1.0, trials=3)
    bank = SimpleNamespace(shape=(1, 3, 800_000_000))
    residue = SimpleNamespace(shape=(800_000_000,))
    with pytest.raises(ValueError, match="int32"):
        graph_stages(model, residue, bank, None, None, requested=1)


def test_capture_keeps_unreturned_workspace_alive_across_replays(cp):
    from iceemdan_cupy.conditional_graph import CUDAWhileGraph

    kernel = cp.RawKernel(
        r"""
        extern "C" __device__ void cudaGraphSetConditional(unsigned long long, unsigned int);
        extern "C" __global__ void work(unsigned long long handle, double* temp, double* out) {
            int q = threadIdx.x;
            temp[q] = 17.; out[q] = temp[q];
            if (q == 0) cudaGraphSetConditional(handle, 0);
        }
        """,
        "work",
    )
    kernel.compile()
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        warm = [cp.empty(128) for _ in range(4)]
        del warm
    stream.synchronize()
    with CUDAWhileGraph(stream) as graph:

        def body(handle):
            out = cp.empty(128)
            temp = cp.empty(128)
            temp_ptr = temp.data.ptr
            kernel((1,), (128,), (np.uint64(handle), temp, out))
            return out, temp_ptr

        out, temp_ptr = graph.capture(body)
        with stream:
            sentinel = cp.full(128, 123.0)
        for _ in range(2):
            graph.launch()
            stream.synchronize()
            np.testing.assert_array_equal(cp.asnumpy(sentinel), np.full(128, 123.0))
            np.testing.assert_array_equal(cp.asnumpy(out), np.full(128, 17.0))
        assert sentinel.data.ptr != temp_ptr


def test_captured_batched_emd_does_not_reuse_workspace_for_sentinels(cp, monkeypatch):
    from iceemdan_cupy.batched_emd import batched_emd
    from iceemdan_cupy.conditional_graph import CUDAWhileGraph

    stop = cp.RawKernel(
        r"""
        extern "C" __device__ void cudaGraphSetConditional(unsigned long long, unsigned int);
        extern "C" __global__ void stop(unsigned long long handle) {
            cudaGraphSetConditional(handle, 0);
        }
        """,
        "stop",
    )
    stop.compile()
    t = cp.arange(128, dtype="float64")
    signal = (cp.sin(0.73 * t) + 0.3 * cp.cos(0.11 * t))[None, :]
    stream = cp.cuda.Stream(non_blocking=True)
    allocations = []
    original_empty = cp.empty

    def tracked_empty(*args, **kwargs):
        array = original_empty(*args, **kwargs)
        allocations.append((array.nbytes, array.data.ptr))
        return array

    with CUDAWhileGraph(stream) as graph:
        graph.prepare(lambda: batched_emd(signal, max_modes=1, _device_result=True))

        def body(handle):
            modes, available, _, _, state = batched_emd(signal, max_modes=1, _device_result=True)
            stop((1,), (1,), (np.uint64(handle),))
            return modes, available, state

        with monkeypatch.context() as patch:
            patch.setattr(cp, "empty", tracked_empty)
            modes, available, state = graph.capture(body)
        captured = {pointer for _, pointer in allocations}
        assert len(captured) >= 10
        with stream:
            sentinels = [cp.full(size, 165, dtype="uint8") for size, _ in allocations]
        assert not any(array.data.ptr in captured for array in sentinels)
        for _ in range(2):
            graph.launch()
            stream.synchronize()
            assert all(bool(cp.all(array == 165)) for array in sentinels)
            assert bool(cp.all(cp.isfinite(modes)))
            assert bool(cp.all(state < 3))
            assert bool(cp.any(available))


def test_close_attempts_both_destructors_and_preserves_body_error():
    import ctypes as ct

    from iceemdan_cupy.conditional_graph import CUDAWhileGraph

    graph = object.__new__(CUDAWhileGraph)
    graph._executable = ct.c_void_p(1)
    graph._graph = ct.c_void_p(2)
    graph._result = object()
    calls = []

    class BadStream:
        def synchronize(self):
            calls.append("sync")
            raise RuntimeError("sync failed")

    graph._stream = BadStream()
    graph._exec_destroy = "exec"
    graph._destroy = "graph"

    def check(which, pointer):
        calls.append(which)
        if which == "exec":
            raise RuntimeError("exec failed")

    graph._check = check
    with pytest.raises(ValueError, match="body failed"):
        with graph:
            raise ValueError("body failed")
    assert calls == ["sync", "exec", "graph"]


def test_end_capture_invalidated_after_body_does_not_destroy_stale_graph(monkeypatch):
    import ctypes as ct
    from contextlib import nullcontext
    from types import SimpleNamespace

    import iceemdan_cupy.conditional_graph as module

    class Stream:
        ptr = 7

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    destroyed = []
    graph = object.__new__(module.CUDAWhileGraph)
    graph._stream = Stream()
    graph._body = 456
    graph._if_body = None
    graph._graph = ct.c_void_p(123)
    graph._executable = ct.c_void_p()
    graph._owners = []
    graph._result = None
    graph.handle = 1
    graph._begin = lambda *args: 0

    def invalid_end(stream, pointer):
        ct.cast(pointer, ct.POINTER(ct.c_void_p))[0] = ct.c_void_p()
        return 900

    graph._end = invalid_end
    graph._error_string = lambda code: b"invalidated"
    graph._check = lambda fn, *args: fn(*args)
    graph._destroy = lambda pointer: destroyed.append(pointer.value) or 0
    monkeypatch.setattr(
        module,
        "cp",
        SimpleNamespace(
            cuda=SimpleNamespace(
                get_allocator=lambda: lambda size: object(),
                using_allocator=lambda fn: nullcontext(),
            )
        ),
    )
    with pytest.raises(RuntimeError, match="invalidated|capture"):
        graph.capture(lambda handle: "completed")
    graph.close()
    assert destroyed == []
    assert not graph._graph.value


def test_parent_device_gate_skips_setup_and_loop_for_initial_stop(cp):
    from iceemdan_cupy.conditional_graph import CUDAWhileGraph

    setup = cp.RawKernel(r'extern "C" __global__ void setup(int* p) { p[0] += 1; }', "setup")
    step = cp.RawKernel(
        r"""
        extern "C" __device__ void cudaGraphSetConditional(unsigned long long,unsigned int);
        extern "C" __global__ void step(unsigned long long h,int* p) {
            p[0] += 1; cudaGraphSetConditional(h,0);
        }
        """,
        "step",
    )
    setup.compile()
    step.compile()
    initial = cp.asarray(1, dtype="int32")
    reason = cp.zeros(1, dtype="int32")
    setup_count = cp.zeros(1, dtype="int32")
    stage_count = cp.zeros(1, dtype="int32")
    stream = cp.cuda.Stream(non_blocking=True)
    with CUDAWhileGraph(stream, initial_code=initial, reason=reason) as graph:
        graph.capture_setup(lambda: setup((1,), (1,), (setup_count,)))

        def body(handle):
            step((1,), (1,), (np.uint64(handle), stage_count))

        graph.capture(body)
        graph.launch()
        stream.synchronize()
        assert int(cp.asnumpy(reason)[0]) == 1
        assert int(cp.asnumpy(setup_count)[0]) == 0
        assert int(cp.asnumpy(stage_count)[0]) == 0
        with stream:
            initial.fill(0)
        graph.launch()
        stream.synchronize()
        assert int(cp.asnumpy(reason)[0]) == 0
        assert int(cp.asnumpy(setup_count)[0]) == 1
        assert int(cp.asnumpy(stage_count)[0]) == 1


def test_graph_terminal_ignores_large_algorithm_limit_with_small_workspace(cp):
    from iceemdan_cupy import ICEEMDAN

    model = ICEEMDAN(
        trials=1,
        batch_emd=True,
        graph_control=True,
        seed=17,
        max_imf_iterations=100_000_000,
        graph_workspace_modes=2,
    )
    before = model._noise.snapshot()
    result = cp.asnumpy(model(np.ones(64)))
    np.testing.assert_array_equal(result, np.ones((1, 64)))
    assert model.diagnostics_["stop_reason"] == "constant_or_short"
    assert model._noise.snapshot() == before
    assert (
        0
        < model.diagnostics_["graph_arena_used_bytes"]
        <= model.diagnostics_["graph_arena_reserved_bytes"]
    )
    assert (
        model.diagnostics_["graph_arena_reserved_bytes"]
        < model.diagnostics_["device_memory_total_bytes"]
    )


def test_graph_workspace_exhaustion_is_resource_error(cp):
    from iceemdan_cupy import ICEEMDAN

    t = np.arange(192.0)
    signal = np.sin(0.73 * t) + 0.4 * np.cos(0.27 * t) + 0.2 * np.sin(0.055 * t)
    noise = np.random.default_rng(14).normal(size=(6, len(t)))
    model = ICEEMDAN(
        trials=6,
        batch_emd=True,
        graph_control=True,
        max_imf_iterations=8,
        graph_workspace_modes=1,
    )
    with pytest.raises(MemoryError, match="workspace"):
        model(signal, max_imf=2, noise=noise)
    assert model.diagnostics_ is None
    recovered = cp.asnumpy(model(signal, max_imf=1, noise=noise))
    assert recovered.shape == (2, len(signal))
    assert np.isfinite(recovered).all()


def test_initial_device_gate_skips_noise_bank_for_whole_call(cp, monkeypatch):
    from iceemdan_cupy import ICEEMDAN

    mark = cp.RawKernel(r'extern "C" __global__ void mark(int* p) { p[0] += 1; }', "mark")
    mark.compile()
    marker = cp.zeros(1, dtype="int32")
    noise_marker = cp.zeros(1, dtype="int32")
    model = ICEEMDAN(trials=1, batch_emd=True, graph_control=True, seed=4)
    original_bank = model._batched_noise_bank
    original_draw = model._noise.draw

    def draw(*args, **kwargs):
        result = original_draw(*args, **kwargs)
        mark((1,), (1,), (noise_marker,))
        return result

    def bank(*args, **kwargs):
        result = original_bank(*args, **kwargs)
        mark((1,), (1,), (marker,))
        return result

    monkeypatch.setattr(model, "_batched_noise_bank", bank)
    monkeypatch.setattr(model._noise, "draw", draw)
    model(np.ones(64))
    assert int(cp.asnumpy(marker)[0]) == 0
    assert int(cp.asnumpy(noise_marker)[0]) == 0
    t = np.arange(64.0)
    model(np.sin(0.4 * t) + 0.3 * np.cos(0.1 * t), max_imf=1)
    assert int(cp.asnumpy(marker)[0]) == 1
    assert int(cp.asnumpy(noise_marker)[0]) == 1


def test_while_graph_replays_batched_emd_without_host_decisions(cp):
    from iceemdan_cupy.batched_emd import batched_emd
    from iceemdan_cupy.conditional_graph import CUDAWhileGraph

    update = cp.RawKernel(
        r"""
        extern "C" __device__ void cudaGraphSetConditional(unsigned long long, unsigned int);
        extern "C" __global__ void update(unsigned long long handle, int* count) {
            int next = ++count[0];
            cudaGraphSetConditional(handle, next < 3);
        }
        """,
        "update",
    )
    update.compile()
    t = cp.arange(128, dtype="float64")
    signals = cp.stack((cp.sin(0.73 * t) + 0.3 * cp.cos(0.11 * t), cp.sin(0.52 * t)))
    expected, expected_available, _, _, expected_state = batched_emd(
        signals, max_modes=1, _device_result=True
    )
    count = cp.zeros(1, dtype="int32")
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        for _ in range(3):
            batched_emd(signals, max_modes=1, _device_result=True)
    stream.synchronize()

    with CUDAWhileGraph(stream) as graph:
        graph.prepare(lambda: batched_emd(signals, max_modes=1, _device_result=True))

        def body(handle):
            modes, available, _, _, state = batched_emd(signals, max_modes=1, _device_result=True)
            update((1,), (1,), (np.uint64(handle), count))
            return modes, available, state

        modes, available, state = graph.capture(body)
        for _ in range(2):
            with stream:
                count.fill(0)
            graph.launch()
            stream.synchronize()
            assert int(cp.asnumpy(count)[0]) == 3
            np.testing.assert_array_equal(cp.asnumpy(modes), cp.asnumpy(expected))
            np.testing.assert_array_equal(cp.asnumpy(available), cp.asnumpy(expected_available))
            np.testing.assert_array_equal(cp.asnumpy(state), cp.asnumpy(expected_state))


@pytest.mark.parametrize(
    "threshold, requested, expected_stages",
    [(0.0, 2, 2), (0.7, 2, 1), (2.0, 2, 0), (0.0, -1, 4)],
)
def test_graph_stages_match_existing_batched_recurrence(cp, threshold, requested, expected_stages):
    from iceemdan_cupy import ICEEMDAN
    from iceemdan_cupy.gpu_loop import graph_stages

    t = np.arange(192.0)
    signal = np.sin(0.73 * t) + 0.4 * np.cos(0.27 * t) + 0.2 * np.sin(0.055 * t)
    noise = np.random.default_rng(14).normal(size=(6, len(t)))
    reference = ICEEMDAN(
        trials=6, batch_emd=True, residue_std_threshold=threshold, max_imf_iterations=8
    )
    expected = cp.asnumpy(reference(signal, max_imf=requested, noise=noise))

    model = ICEEMDAN(
        trials=6, batch_emd=True, residue_std_threshold=threshold, max_imf_iterations=8
    )
    model._noise_cap = requested if requested > 0 else model.max_imf_iterations
    model._supplied_noise = cp.asarray(noise)
    bank, available, first_std, _, _, bank_status, bank_state = model._batched_noise_bank((len(t),))
    peak = float(np.max(np.abs(signal)))
    std = float(np.std(signal / peak, ddof=model.ddof))
    r = cp.asarray(signal / peak / std)
    modes, residue, reason, diagnostics = graph_stages(
        model,
        r,
        bank,
        available,
        first_std,
        requested=requested,
        initial_std=cp.asarray(std),
        bank_status=bank_status,
        bank_state=bank_state,
    )
    actual = cp.asnumpy(cp.vstack((*modes, residue)) * std * peak)
    assert len(modes) == expected_stages
    assert reason == reference.diagnostics_["stop_reason"]
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-10)
    assert len(diagnostics) == expected_stages
    for stage, expected_stage in zip(diagnostics, reference.diagnostics_["stages"], strict=True):
        assert stage["missing_noise_modes"] == expected_stage["missing_noise_modes"]
        assert stage["sift_iterations"] == expected_stage["sift_iterations"]
        np.testing.assert_allclose(
            stage["perturbation_stds"], expected_stage["perturbation_stds"], rtol=1e-13
        )
        np.testing.assert_allclose(
            stage["residue_std_before"], expected_stage["residue_std_before"], rtol=1e-13
        )


@pytest.mark.parametrize("requested", [2, -1])
def test_public_graph_control_matches_batched_output_and_diagnostics(cp, requested):
    from iceemdan_cupy import ICEEMDAN

    t = np.arange(192.0)
    signal = np.sin(0.73 * t) + 0.4 * np.cos(0.27 * t) + 0.2 * np.sin(0.055 * t)
    noise = np.random.default_rng(14).normal(size=(6, len(t)))
    previous = ICEEMDAN(trials=6, batch_emd=True, max_imf_iterations=8)
    expected = cp.asnumpy(previous(signal, max_imf=requested, noise=noise))
    migrated = ICEEMDAN(trials=6, batch_emd=True, graph_control=True, max_imf_iterations=8)
    actual = cp.asnumpy(migrated(signal, max_imf=requested, noise=noise))
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-10)
    assert migrated.diagnostics_["stop_reason"] == previous.diagnostics_["stop_reason"]
    assert migrated.diagnostics_["noise_mode_counts"] == previous.diagnostics_["noise_mode_counts"]
    assert migrated.diagnostics_["stages"] == pytest.approx(previous.diagnostics_["stages"])


def test_graph_control_does_not_call_host_stop_or_std(cp, monkeypatch):
    from iceemdan_cupy import ICEEMDAN, ensemble

    t = np.arange(128.0)
    signal = np.sin(0.49 * t) + 0.4 * np.sin(0.09 * t)
    noise = np.random.default_rng(11).normal(size=(4, len(t)))
    model = ICEEMDAN(trials=4, batch_emd=True, graph_control=True)

    def reject(*args, **kwargs):
        pytest.fail("Graph control used the host stop or standard deviation path.")

    monkeypatch.setattr(model, "_stop_reason", reject)
    monkeypatch.setattr(ensemble, "stable_std", reject)
    assert model(signal, max_imf=2, noise=noise).shape == (3, len(t))


def test_graph_control_generated_noise_and_terminal_rng_match(cp):
    from iceemdan_cupy import ICEEMDAN

    t = np.arange(128.0)
    signal = np.sin(0.49 * t) + 0.4 * np.sin(0.09 * t)
    previous = ICEEMDAN(trials=5, seed=42, batch_emd=True)
    migrated = ICEEMDAN(trials=5, seed=42, batch_emd=True, graph_control=True)
    expected = cp.asnumpy(previous(signal, max_imf=2))
    actual = cp.asnumpy(migrated(signal, max_imf=2))
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-10)
    assert migrated._noise.snapshot() == previous._noise.snapshot()

    previous_terminal = ICEEMDAN(trials=5, seed=42, batch_emd=True, residue_std_threshold=2.0)
    migrated_terminal = ICEEMDAN(
        trials=5,
        seed=42,
        batch_emd=True,
        graph_control=True,
        residue_std_threshold=2.0,
    )
    terminal_expected = cp.asnumpy(previous_terminal(signal, max_imf=2))
    terminal_actual = cp.asnumpy(migrated_terminal(signal, max_imf=2))
    np.testing.assert_array_equal(terminal_actual, terminal_expected)
    assert migrated_terminal._noise.snapshot() == previous_terminal._noise.snapshot()
    assert migrated_terminal.diagnostics_["stop_reason"] == "residue_std_threshold"


def test_deferred_noise_draw_preserves_philox_without_host_check(cp, monkeypatch):
    from iceemdan_cupy import noise as noise_module

    reference = noise_module.NoiseStream(seed=42)
    expected = cp.asnumpy(reference.draw(1.0, 128))
    deferred = noise_module.NoiseStream(seed=42)

    def reject(*args, **kwargs):
        pytest.fail("Deferred noise draw synchronized with the host.")

    monkeypatch.setattr(noise_module, "scalar", reject)
    monkeypatch.setattr(noise_module, "synchronize", reject)
    actual = deferred.draw(1.0, 128, defer_validation=True)
    np.testing.assert_array_equal(cp.asnumpy(actual), expected)
    assert deferred.snapshot() == reference.snapshot()


def test_deferred_noise_bank_keeps_validation_state_on_device(cp, monkeypatch):
    from iceemdan_cupy import ICEEMDAN, ensemble

    t = np.arange(128.0)
    model = ICEEMDAN(trials=4, batch_emd=True, graph_control=True)
    model._noise_cap = 2
    model._supplied_noise = cp.asarray(np.random.default_rng(7).normal(size=(4, len(t))))

    def reject(*args, **kwargs):
        pytest.fail("Deferred noise bank read a device validation scalar.")

    monkeypatch.setattr(ensemble, "scalar", reject)
    bank, available, first_std, counts, used, status, state = model._batched_noise_bank(
        (len(t),), defer_errors=True
    )
    assert bank.shape == (2, 4, len(t))
    assert available.shape == (2, 4)
    assert first_std.shape == counts.shape == (4,)
    assert used.shape == status.shape == ()
    assert state.shape == (4,)
    assert int(cp.asnumpy(status)) == 0


def test_graph_control_reports_invalid_noise_after_device_loop(cp):
    from iceemdan_cupy import ICEEMDAN, BackendContractError

    t = np.arange(128.0)
    signal = np.sin(0.49 * t) + 0.4 * np.sin(0.09 * t)
    monotone_noise = np.stack([t + i for i in range(4)])
    model = ICEEMDAN(trials=4, batch_emd=True, graph_control=True, seed=17)
    before = model._noise.snapshot()
    with pytest.raises(BackendContractError, match="first noise IMF"):
        model(signal, max_imf=2, noise=monotone_noise)
    assert model.diagnostics_ is None
    assert model._noise.snapshot() == before


@pytest.mark.parametrize(
    "signal, requested, reason",
    [
        (np.ones(64), -1, "constant_or_short"),
        (np.arange(64.0), -1, "fewer_than_three_extrema"),
        (np.sin(0.7 * np.arange(64.0)), 0, "max_imf"),
    ],
)
def test_graph_control_initial_terminal_contract(cp, signal, requested, reason):
    from iceemdan_cupy import ICEEMDAN

    previous = ICEEMDAN(trials=4, seed=3, batch_emd=True)
    migrated = ICEEMDAN(trials=4, seed=3, batch_emd=True, graph_control=True)
    expected = cp.asnumpy(previous(signal, max_imf=requested))
    actual = cp.asnumpy(migrated(signal, max_imf=requested))
    np.testing.assert_array_equal(actual, expected)
    assert migrated.diagnostics_["stop_reason"] == reason
    assert migrated._noise.snapshot() == previous._noise.snapshot()


def test_graph_control_reuses_its_capture_stream(cp):
    from iceemdan_cupy import ICEEMDAN

    t = np.arange(128.0)
    signal = np.sin(0.49 * t) + 0.4 * np.sin(0.09 * t)
    noise = np.random.default_rng(11).normal(size=(4, len(t)))
    model = ICEEMDAN(trials=4, batch_emd=True, graph_control=True)
    first = cp.asnumpy(model(signal, max_imf=2, noise=noise))
    stream = model._graph_stream
    second = cp.asnumpy(model(signal, max_imf=2, noise=noise))
    assert model._graph_stream is stream
    np.testing.assert_array_equal(second, first)
    longer = np.sin(0.49 * np.arange(192.0)) + 0.4 * np.sin(0.09 * np.arange(192.0))
    longer_noise = np.random.default_rng(12).normal(size=(4, len(longer)))
    resized = cp.asnumpy(model(longer, max_imf=2, noise=longer_noise))
    expected = ICEEMDAN(trials=4, batch_emd=True)
    np.testing.assert_allclose(
        resized, cp.asnumpy(expected(longer, max_imf=2, noise=longer_noise)), rtol=1e-9, atol=1e-10
    )
    assert model._graph_stream is stream


@pytest.mark.parametrize("case", ["close_tones", "chirp_trend", "colored_tone", "plateaus"])
def test_graph_control_stress_parity_with_previous_batch(cp, case):
    from iceemdan_cupy import ICEEMDAN

    t = np.arange(192.0)
    rng = np.random.default_rng(23)
    if case == "close_tones":
        signal = np.sin(0.16 * t) + 0.8 * np.sin(0.172 * t)
    elif case == "chirp_trend":
        signal = np.sin(0.04 * t + 0.0012 * t**2) + 0.002 * t
    elif case == "colored_tone":
        signal = np.sin(0.37 * t) + 0.2 * np.cumsum(rng.normal(size=len(t))) / len(t)
    else:
        signal = np.round(np.sin(0.22 * t), 1)
    noise = rng.normal(size=(8, len(t)))
    previous = ICEEMDAN(trials=8, batch_emd=True, max_imf_iterations=6)
    migrated = ICEEMDAN(trials=8, batch_emd=True, graph_control=True, max_imf_iterations=6)
    try:
        expected = cp.asnumpy(previous(signal, max_imf=3, noise=noise))
    except RuntimeError as exc:
        with pytest.raises(type(exc)):
            migrated(signal, max_imf=3, noise=noise)
    else:
        actual = cp.asnumpy(migrated(signal, max_imf=3, noise=noise))
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-10)
        assert migrated.diagnostics_["stop_reason"] == previous.diagnostics_["stop_reason"]
        assert migrated.diagnostics_["stages"] == pytest.approx(previous.diagnostics_["stages"])


def test_graph_control_preserves_safety_limit_error(cp):
    from iceemdan_cupy import ICEEMDAN, DecompositionLimitError

    t = np.arange(192.0)
    signal = np.sin(0.73 * t) + 0.4 * np.cos(0.27 * t) + 0.2 * np.sin(0.055 * t)
    noise = np.random.default_rng(14).normal(size=(6, len(t)))
    for graph_control in (False, True):
        model = ICEEMDAN(
            trials=6, batch_emd=True, graph_control=graph_control, max_imf_iterations=1
        )
        with pytest.raises(DecompositionLimitError, match="safety limit"):
            model(signal, noise=noise)
        assert model.diagnostics_ is None
