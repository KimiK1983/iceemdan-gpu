"""Device-controlled ICEEMDAN stage loop for the cubic batched EMD path."""

from __future__ import annotations

from numpy import uint64

from .batched_emd import _check_index_products, batched_emd, ordered_mean
from .conditional_graph import CUDAWhileGraph
from .contracts import BackendContractError, DecompositionLimitError, SiftingConvergenceError
from .ensemble import _EARLY_STOP_REASONS, _stage_status
from .runtime import cp

_PREPARE = cp.RawKernel(
    r"""
    extern "C" __global__ void prepare_stage(
        const double* r, const double* bank, const double* first_std,
        const double* sd, const int* count, const int* skip,
        double* y, double* eta,
        int n, int trials, int cap, double epsilon) {
        int q = blockIdx.x * blockDim.x + threadIdx.x;
        if (q >= n * trials) return;
        if (skip[0]) { eta[q] = 0.0; y[q] = r[q % n]; return; }
        int k = count[0], i = q / n;
        if (k >= cap) k = cap - 1;
        double mode = bank[((size_t)k * trials * n) + q];
        double amplitude = epsilon * sd[0];
        double perturbation = k == 0 ? amplitude * (mode / first_std[i]) : amplitude * mode;
        eta[q] = perturbation;
        y[q] = r[q % n] + perturbation;
    }
    """,
    "prepare_stage",
)
_DECIDE = cp.RawKernel(
    r"""
    extern "C" __global__ void decide_stage(
        const int* early, const long long* imf, const long long* progress,
        const int* state, const int* count, const double* initial_std,
        const long long* bank_status, int* action, int* reason,
        int trials, int requested, int limit, int algorithm_limit) {
        int k = count[0];
        int code = requested >= 0 && k >= requested ? 9 : 0;
        if (!code && k == 0 && initial_std[0] == 0.0) code = 14;
        if (!code) code = early[0];
        if (!code && k == 0 && bank_status[0]) code = 14 + (int)bank_status[0];
        if (!code && k > 0) code = (int)imf[0];
        if (!code && k >= limit) code = limit < algorithm_limit ? 19 : 10;
        if (!code) {
            for (int i = 0; i < trials; ++i) {
                if (state[i] >= 3) { code = 11; break; }
            }
        }
        if (!code && progress[0] == 1) code = 12;
        if (!code && progress[0] == 2) code = 13;
        action[0] = code == 0;
        reason[0] = code;
    }
    """,
    "decide_stage",
)
_COMMIT = cp.RawKernel(
    r"""
    extern "C" __global__ void commit_stage(
        double* r, const double* next, double* modes,
        const int* count, const int* action, int n) {
        int q = blockIdx.x * blockDim.x + threadIdx.x;
        if (q >= n || !action[0]) return;
        int k = count[0];
        modes[(size_t)k * n + q] = r[q] - next[q];
        r[q] = next[q];
    }
    """,
    "commit_stage",
)
_DIAGNOSTICS = cp.RawKernel(
    r"""
    extern "C" __global__ void record_stage(
        const int* count, const int* action, const bool* noise_available,
        const double* perturbation_stds, const int* sift_counts,
        const bool* local_available, const double* residue_std,
        int* missing, double* stds, int* sifts, double* residue_stds,
        int trials) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= trials || !action[0]) return;
        int k = count[0];
        if (i == 0) residue_stds[k] = residue_std[0];
        if (!noise_available[k * trials + i]) atomicAdd(missing + k, 1);
        stds[k * trials + i] = perturbation_stds[i];
        sifts[k * trials + i] = local_available[i] ? sift_counts[i] : -1;
    }
    """,
    "record_stage",
)
_ADVANCE = cp.RawKernel(
    r"""
    extern "C" __device__ void cudaGraphSetConditional(unsigned long long, unsigned int);
    extern "C" __global__ void advance_stage(
        unsigned long long handle, int* count, const int* action,
        int* reason, int requested) {
        if (!action[0]) {
            cudaGraphSetConditional(handle, 0);
            return;
        }
        int next = ++count[0];
        if (requested >= 0 && next >= requested) {
            reason[0] = 9;
            cudaGraphSetConditional(handle, 0);
        } else {
            cudaGraphSetConditional(handle, 1);
        }
    }
    """,
    "advance_stage",
)


def graph_stages(
    model,
    residue,
    bank=None,
    noise_available=None,
    first_std=None,
    *,
    requested,
    initial_std=None,
    bank_status=None,
    bank_state=None,
    initial_early=None,
    setup=None,
    resource_info=None,
):
    """Run stage recurrence in a CUDA WHILE graph; transfer status after exit.

    Parameters
    ----------
    model : ICEEMDAN
        Configured ensemble and graph stream owner.
    residue : cupy.ndarray
        Initial float64 residue with shape ``(samples,)``; updated in place.
    bank : cupy.ndarray, optional
        Noise modes with shape ``(capacity, trials, samples)``.
    noise_available : cupy.ndarray, optional
        Boolean availability with shape ``(capacity, trials)``.
    first_std : cupy.ndarray, optional
        First noise IMF standard deviation for each realization.
    requested : int
        User's component limit; ``-1`` requests natural termination.
    initial_std : cupy.ndarray, optional
        Device scalar with initial signal standard deviation.
    bank_status : cupy.ndarray, optional
        Device scalar describing noise-bank setup errors.
    bank_state : cupy.ndarray, optional
        Device row states used when reporting a bank failure.
    initial_early : cupy.ndarray, optional
        Device int32 scalar for the initial termination gate.
    setup : callable, optional
        Captured operation that builds the bank before the stage loop.
    resource_info : dict, optional
        Receives observed device capacity and graph arena byte counts.

    Returns
    -------
    tuple
        Extracted modes, updated residue, stop reason and stage diagnostics.
    """
    if requested == 0 or model.epsilon <= 0:
        raise ValueError("Graph stages require positive epsilon and a nonzero component limit.")
    limit, trials, n = (
        (model._noise_cap, model.trials, residue.size) if setup is not None else bank.shape
    )
    if limit < 1 or trials != model.trials or residue.shape != (n,):
        raise ValueError("Noise bank and residue shapes do not match the ensemble.")
    _check_index_products(trials, n)
    if initial_std is None:
        initial_std = cp.asarray(1.0, dtype="float64")
    if bank_status is None:
        bank_status = cp.zeros((), dtype="int64")
    if initial_early is None:
        initial_early = cp.zeros((), dtype="int32")
    if resource_info is not None:
        resource_info["device_memory_total_bytes"] = cp.cuda.Device().mem_info[1]
    modes = cp.zeros((limit, n), dtype="float64")
    count = cp.zeros(1, dtype="int32")
    reason = cp.zeros(1, dtype="int32")
    action = cp.zeros(1, dtype="int32")
    y = cp.empty((trials, n), dtype="float64")
    eta = cp.empty(y.shape, dtype="float64")
    missing = cp.zeros(limit, dtype="int32")
    perturbation_stds = cp.zeros((limit, trials), dtype="float64")
    sift_counts = cp.full((limit, trials), -1, dtype="int32")
    residue_stds = cp.zeros(limit, dtype="float64")
    for kernel in (_PREPARE, _DECIDE, _COMMIT, _DIAGNOSTICS, _ADVANCE):
        kernel.compile()
    stream = model._graph_stream
    if stream is None:
        stream = cp.cuda.Stream(non_blocking=True)
        model._graph_stream = stream
    ready = cp.cuda.get_current_stream().record()
    stream.wait_event(ready)

    def calculate():
        early = model._early_stop_code(residue)
        imf = model._imf_stop_code(residue, skip=initial_early)
        sd = model._device_std(residue)
        _PREPARE(
            ((y.size + 127) // 128,),
            (128,),
            (
                residue,
                bank,
                first_std,
                sd,
                count,
                initial_early,
                y,
                eta,
                n,
                trials,
                limit,
                model.epsilon,
            ),
        )
        local_modes, local_available, _, iterations, state = batched_emd(
            y,
            max_modes=1,
            nbsym=model.EMD.nbsym,
            max_iteration=model.EMD.MAX_ITERATION,
            thresholds=model.EMD.rilling_thresholds,
            _device_result=True,
            _skip=initial_early,
        )
        next_residue = ordered_mean(y - local_modes[0])
        progress = _stage_status(residue, next_residue)
        stds = model._batched_std(eta)
        return early, imf, sd, state, next_residue, progress, stds, local_available, iterations

    arena_bytes = 0
    if setup is not None:
        # ponytail: sized for CuPy 14.2 Philox and captured temporaries; remeasure on upgrade.
        philox_bytes = 16_384_000 if model._supplied_noise is None else 0
        arena_bytes = philox_bytes + 4 * (8 * (limit + 64) + 12) * trials * n + 1_048_576
    with CUDAWhileGraph(
        stream,
        initial_code=initial_early if setup is not None else None,
        reason=reason if setup is not None else None,
        arena_bytes=arena_bytes,
    ) as graph:
        if setup is not None:
            bank, noise_available, first_std, _, _, bank_status, bank_state = graph.capture_setup(
                lambda: setup(graph.rng_state_allocator)
            )
        else:
            graph.prepare(calculate)

        def body(handle):
            early, imf, sd, state, next_r, progress, stds, local_available, iterations = calculate()
            _DECIDE(
                (1,),
                (1,),
                (
                    early,
                    imf,
                    progress,
                    state,
                    count,
                    initial_std,
                    bank_status,
                    action,
                    reason,
                    trials,
                    requested,
                    limit,
                    model.max_imf_iterations,
                ),
            )
            _COMMIT(
                ((n + 127) // 128,),
                (128,),
                (residue, next_r, modes, count, action, n),
            )
            _DIAGNOSTICS(
                ((trials + 127) // 128,),
                (128,),
                (
                    count,
                    action,
                    noise_available,
                    stds,
                    iterations[0],
                    local_available[0],
                    sd,
                    missing,
                    perturbation_stds,
                    sift_counts,
                    residue_stds,
                    trials,
                ),
            )
            _ADVANCE((1,), (1,), (uint64(handle), count, action, reason, requested))
            return state

        last_state = graph.capture(body)
        if resource_info is not None:
            resource_info["graph_arena_reserved_bytes"] = arena_bytes
            resource_info["graph_arena_used_bytes"] = graph._arena_used
        graph.launch()
        stream.synchronize()

    completed, code = cp.asnumpy(cp.stack((count[0], reason[0]))).tolist()
    if code == 8:
        raise SiftingConvergenceError("Batched stop-IMF extraction did not converge.")
    if code == 10:
        raise DecompositionLimitError("ICEEMDAN safety limit reached before natural termination.")
    if code == 19:
        raise MemoryError("CUDA graph workspace mode capacity exhausted before termination.")
    if code == 11:
        failed = cp.asnumpy(last_state)
        raise SiftingConvergenceError(
            f"Batched EMD failed for rows {list((failed >= 3).nonzero()[0])}; "
            "states: 3=bad spline, 4=stagnation, 5=iteration limit, 6=non-finite."
        )
    if code == 12:
        raise FloatingPointError("Non-finite ICEEMDAN residue.")
    if code == 13:
        raise SiftingConvergenceError("ICEEMDAN made no progress before its terminal condition.")
    if code == 15:
        failed = cp.asnumpy(bank_state) if bank_state is not None else None
        rows = list((failed >= 3).nonzero()[0]) if failed is not None else []
        raise SiftingConvergenceError(f"Batched noise EMD failed for rows {rows}.")
    if code == 16:
        raise BackendContractError("Every realization needs a first noise IMF.")
    if code == 17:
        raise BackendContractError("Every realization needs a finite nonzero first noise IMF.")
    if code == 18:
        raise FloatingPointError("Noise generation overflowed.")
    if not 1 <= code <= 14:
        raise RuntimeError(f"Unknown CUDA stage termination code {code}.")
    diagnostics = []
    if completed:
        host_missing = cp.asnumpy(missing[:completed])
        host_stds = cp.asnumpy(perturbation_stds[:completed])
        host_sifts = cp.asnumpy(sift_counts[:completed])
        host_residue_stds = cp.asnumpy(residue_stds[:completed])
        for k in range(completed):
            diagnostics.append(
                dict(
                    index=k + 1,
                    noise_mode_index=k + 1,
                    ensemble_denominator=trials,
                    missing_noise_modes=int(host_missing[k]),
                    perturbation_stds=host_stds[k].tolist(),
                    residue_std_before=float(host_residue_stds[k]),
                    sift_iterations=[int(v) if v >= 0 else None for v in host_sifts[k]],
                )
            )
    reason_text = (
        "max_imf" if code == 9 else "constant_or_short" if code == 14 else _EARLY_STOP_REASONS[code]
    )
    return modes[:completed], residue, reason_text, diagnostics
