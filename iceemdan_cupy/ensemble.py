"""CuPy ICEEMDAN with reference and batched cubic EMD routes.

Recurrence, thresholds and Kahan order follow frozen ICEEMDAN(2).py.
Modified 2026-09-25. Original wrapper: Javier F. Santamaria; rights retained.
The reference path remains the default; batched and graph control are opt-in.
"""

from __future__ import annotations

import hashlib
import logging
import math
import warnings
from typing import Any

from numpy import bool_ as numpy_bool

from .contracts import (
    EPS,
    BackendContractError,
    DecompositionLimitError,
    SiftingConvergenceError,
    extrema,
    float_dtype,
    integer,
    nonnegative,
    stable_std,
    time_vector,
    vector,
)
from .emd import EMD, backend_parts, local_mean
from .noise import NoiseStream
from .runtime import cp, device_array, on_device, scalar, synchronize

__version__ = "0.2.0"


def batched_emd(*args, **kwargs):
    from .batched_emd import batched_emd as run

    return run(*args, **kwargs)


def batched_extrema(*args, **kwargs):
    from .batched_emd import batched_extrema as run

    return run(*args, **kwargs)


def ordered_mean(*args, **kwargs):
    from .batched_emd import ordered_mean as run

    return run(*args, **kwargs)


_EARLY_STOP_REASONS = (
    None,
    "fewer_than_three_extrema",
    "numerical_zero",
    "residue_std_threshold",
    "range_threshold",
    "residue_l1_threshold",
    "residue_is_imf",
    "emd_has_no_extractable_imf",
)
_NON_NATURAL_REASONS = (
    "max_imf",
    "range_threshold",
    "residue_l1_threshold",
    "residue_std_threshold",
)


def kahan_add(total, correction, value):
    """Add one value to an ordered compensated sum.

    Parameters
    ----------
    total : cupy.ndarray
        Current sum.
    correction : cupy.ndarray
        Compensation carried from the preceding addition.
    value : cupy.ndarray
        Next contribution; order is significant.

    Returns
    -------
    tuple of cupy.ndarray
        Updated sum and compensation, suitable for the next chunk.
    """
    term = value - correction
    summed = total + term
    correction = (summed - total) - term
    return summed, correction


def _stage_status(previous, candidate):
    """Encode stage progress on device: 0 progressed, 1 non-finite, 2 unchanged.

    Parameters
    ----------
    previous, candidate : cupy.ndarray
        Residues before and after one stage.

    Returns
    -------
    cupy.ndarray
        Device scalar status code.
    """
    return cp.where(
        cp.all(cp.isfinite(candidate)),
        cp.where(cp.array_equal(previous, candidate), 2, 0),
        1,
    )


class ICEEMDAN:
    """Device-resident ICEEMDAN using the CPU reference's mathematical rules.

    Return shape (K+1,N); the final row is always the residue. Default float64
    INTERNAL precision is fixed. dtype selects OUTPUT float64/float32.

    rng_mode='cupy_philox' generates noise on device. For CPU-reference
    comparisons, supply the same noise matrix explicitly to both backends.
    Supplied noise=W bypasses either generator without changing its state.

    parallel=True and processes other than None/1 are rejected: the reference
    GPU port uses one process per device. The opt-in batched EMD processes
    realizations together, while the reference path retains their order.
    trace(event,payload) is opt-in; it receives device arrays and host metadata.
    It may add cost/storage. There is no automatic transfer inside the trace.

    Parameters
    ----------
    trials : int, default 100
        Number of noise realizations in each ensemble.
    epsilon : float, default 0.2
        Nonnegative perturbation scale.
    ext_EMD : object, optional
        External EMD backend returning device-resident IMFs and residue.
    parallel : bool, default False
        ``True`` is rejected; this port uses one process per device.
    emd_params : dict, optional
        Parameters passed to the bundled EMD when ``ext_EMD`` is absent.
    dtype : {"float64", "float32"}, default "float64"
        Output precision; internal calculations remain float64.
    seed : int, optional
        Initial random seed. ``None`` selects a fresh seed.
    processes : int, optional
        Only ``None`` or ``1`` is accepted.
    noise_kind : {"normal", "uniform"}, default "normal"
        Noise distribution. Uniform noise is an experimental variant.
    range_thr : float, optional
        Residue range stopping threshold.
    total_power_thr : float, optional
        Residue absolute-sum stopping threshold.
    residue_std_threshold : float, default 0
        Residue standard-deviation stopping threshold.
    max_imf_iterations : int, default 100
        Safety limit on the number of extracted components.
    ddof : {0, 1}, default 1
        Degrees-of-freedom correction used in standard deviations.
    device : int, default 0
        CUDA device index.
    rng_mode : {"cupy_philox"}, default "cupy_philox"
        Device-generated noise stream.
    trace : callable, optional
        Callback receiving an event name and payload for diagnostics.
    batch_emd : bool, default False
        Use the CUDA batched cubic EMD path for noise and ensemble extractions.
        The default remains the established scalar CuPy reference path.
    graph_control : bool, default False
        Run the batched stage loop inside a CUDA conditional WHILE graph.
    graph_workspace_modes : int, default 64
        Physical mode slots reserved by graph control; exhaustion raises a resource error.
    """

    logger = logging.getLogger(__name__)

    def __init__(
        self,
        trials=100,
        epsilon=0.2,
        ext_EMD=None,
        parallel=False,
        emd_params=None,
        dtype="float64",
        *,
        seed=None,
        processes=None,
        noise_kind="normal",
        range_thr=None,
        total_power_thr=None,
        residue_std_threshold=0.0,
        max_imf_iterations=100,
        ddof=1,
        device=0,
        rng_mode="cupy_philox",
        trace=None,
        batch_emd=False,
        graph_control=False,
        graph_workspace_modes=64,
    ):
        self.device_id = integer("device", device, 0)
        self.trials = integer("trials", trials)
        self.epsilon = nonnegative("epsilon", epsilon)
        self.dtype = float_dtype(dtype)
        if not isinstance(parallel, (bool, numpy_bool)):
            raise TypeError("parallel must be bool.")
        if parallel:
            raise ValueError(
                "parallel=True is a CPU ProcessPool option. This GPU reference uses one process; no silent fallback."
            )
        if processes is not None and integer("processes", processes) != 1:
            raise ValueError(
                "GPU reference uses one process per device; processes must be None or 1."
            )
        self.parallel = False
        self.processes = 1
        self.max_imf_iterations = integer("max_imf_iterations", max_imf_iterations)
        self.range_thr = nonnegative("range_thr", range_thr, True)
        self.total_power_thr = nonnegative("total_power_thr", total_power_thr, True)
        self.residue_std_threshold = nonnegative("residue_std_threshold", residue_std_threshold)
        self.ddof = integer("ddof", ddof, 0)
        if self.ddof not in (0, 1):
            raise ValueError("ddof must be 0 or 1.")
        if noise_kind not in ("normal", "uniform"):
            raise ValueError("noise_kind must be normal or uniform.")
        self.noise_kind = noise_kind
        if noise_kind == "uniform":
            warnings.warn(
                "Uniform noise is an experimental variant, not the Gaussian paper protocol.",
                UserWarning,
                stacklevel=2,
            )
        if emd_params is not None and not isinstance(emd_params, dict):
            raise TypeError("emd_params must be a dictionary or None.")
        if ext_EMD is not None and emd_params:
            raise ValueError("Configure ext_EMD itself; do not also pass emd_params.")
        if trace is not None and not callable(trace):
            raise TypeError("trace must be callable or None.")
        self.trace = trace
        if not isinstance(batch_emd, (bool, numpy_bool)):
            raise TypeError("batch_emd must be bool.")
        self.batch_emd = bool(batch_emd)
        if not isinstance(graph_control, (bool, numpy_bool)):
            raise TypeError("graph_control must be bool.")
        self.graph_control = bool(graph_control)
        self.graph_workspace_modes = integer("graph_workspace_modes", graph_workspace_modes)
        if self.graph_control and (not self.batch_emd or self.epsilon == 0):
            raise ValueError("graph_control requires batch_emd=True and epsilon>0.")
        self.emd_params_internal = dict(emd_params or {})
        if ext_EMD is None:
            if (
                "device" in self.emd_params_internal
                and self.emd_params_internal["device"] != self.device_id
            ):
                raise ValueError("EMD device conflicts with ICEEMDAN device.")
            self.emd_params_internal.setdefault("max_modes", self.max_imf_iterations)
            self.emd_params_internal.setdefault("device", self.device_id)
            self.emd_params_internal.setdefault("trace", trace)
            self.EMD = EMD(**self.emd_params_internal)
        else:
            for method in ("emd", "get_imfs_and_residue"):
                if not callable(getattr(ext_EMD, method, None)):
                    raise TypeError(f"ext_EMD must implement {method}().")
            if isinstance(ext_EMD, EMD) and ext_EMD.device_id != self.device_id:
                raise ValueError("External EMD device does not match ICEEMDAN device.")
            self.EMD = ext_EMD
        if self.batch_emd and (
            not isinstance(self.EMD, EMD) or self.EMD.spline_kind != "cubic" or trace is not None
        ):
            raise ValueError("batch_emd requires the bundled cubic EMD without trace.")
        if self.batch_emd and self.EMD.output_dtype.name != "float64":
            raise ValueError(
                "batch_emd requires EMD DTYPE='float64'; its kernels use float64 output."
            )
        self._noise = NoiseStream(rng_mode, seed)
        self.C_IMF = self.residue = self.diagnostics_ = None
        self.all_noise_IMFs_for_CEEMDAN = []
        self._active = False
        self._supplied_noise = None
        self._noise_cap = self.max_imf_iterations
        self._last_iterations = None
        self._graph_stream = None

    def noise_seed(self, seed):
        """Reset the noise stream before the next decomposition.

        Parameters
        ----------
        seed : int or None
            Nonnegative seed, or ``None`` for a fresh random seed.
        """
        if self._active:
            raise RuntimeError("Cannot reset the RNG during a decomposition.")
        self._noise.reset(seed)

    @on_device
    def generate_noise(self, scale, size):
        return self._noise.draw(scale, size, self.noise_kind)

    def _get_emd_local_mean(self, y, t):
        mean, self._last_iterations = local_mean(self.EMD, y, t)
        return mean

    def _context(self, text):
        if isinstance(self.EMD, EMD):
            self.EMD.trace_context = text

    def _emit(self, event, **payload):
        if self.trace is not None:
            self.trace(
                event, {k: v.copy() if isinstance(v, cp.ndarray) else v for k, v in payload.items()}
            )

    def _pre_decompose_noise_for_ceemdan(self, S_shape, T):
        self.all_noise_IMFs_for_CEEMDAN = []
        for i in range(self.trials):
            w = (
                self.generate_noise(1.0, S_shape)
                if self._supplied_noise is None
                else self._supplied_noise[i].copy()
            )
            self._context(f"noise:{i}")
            modes, _, _ = backend_parts(self.EMD, w, T, self._noise_cap)
            self.all_noise_IMFs_for_CEEMDAN.append(modes)
            self._emit("noise_modes", realization=i, noise=w, modes=modes)

    def _batched_noise_bank(self, shape, *, defer_errors=False, skip=None, _state_allocator=None):
        if self._noise_cap > self.EMD.max_modes:
            raise ValueError("max_imf exceeds the EMD max_modes safety limit.")
        if self._supplied_noise is None:
            if defer_errors:
                noise = cp.empty((self.trials, *shape), dtype="float64")
                for i in range(self.trials):
                    noise[i] = self._noise.draw(
                        1.0,
                        shape,
                        self.noise_kind,
                        defer_validation=True,
                        _state_allocator=_state_allocator,
                    )
            else:
                noise = cp.stack([self.generate_noise(1.0, shape) for _ in range(self.trials)])
        else:
            noise = self._supplied_noise.copy()
        modes, available, _, _, state = batched_emd(
            noise,
            max_modes=self._noise_cap,
            nbsym=self.EMD.nbsym,
            max_iteration=self.EMD.MAX_ITERATION,
            thresholds=self.EMD.rilling_thresholds,
            _device_result=True,
            _skip=skip,
        )
        first_std = self._batched_std(modes[0])
        invalid_std = (first_std <= 0) | ~cp.isfinite(first_std)
        status = cp.where(
            cp.any(state >= 3),
            1,
            cp.where(cp.all(~available[0]), 2, cp.where(cp.any(invalid_std), 3, 0)),
        )
        if defer_errors:
            status = cp.where(cp.all(cp.isfinite(noise)), status, 4)
        else:
            code = int(scalar(status))
            if code == 1:
                failed = cp.asnumpy(state)
                raise SiftingConvergenceError(
                    f"Batched EMD failed for rows {list((failed >= 3).nonzero()[0])}; "
                    "states: 3=bad spline, 4=stagnation, 5=iteration limit, 6=non-finite."
                )
            if code == 2:
                raise BackendContractError("Every realization needs a first noise IMF.")
            if code == 3:
                raise BackendContractError(
                    "Every realization needs a finite nonzero first noise IMF."
                )
        counts = cp.sum(available, axis=0)
        used = cp.sum(cp.any(available, axis=1))
        return modes, available, first_std, counts, used, status, state

    def _batched_std(self, rows):
        peak = cp.max(cp.absolute(rows), axis=1)
        divisor = cp.where(peak > 0, peak, 1.0)
        return peak * cp.std(rows / divisor[:, None], axis=1, ddof=self.ddof)

    def _device_std(self, r):
        if r.size <= self.ddof:
            return cp.zeros((), dtype="float64")
        peak = cp.max(cp.absolute(r))
        divisor = cp.where(peak > 0, peak, 1.0)
        return peak * cp.std(r / divisor, ddof=self.ddof)

    def _early_stop_code(self, r):
        """Return the first applicable non-IMF stop reason as a device scalar.

        Parameters
        ----------
        r : cupy.ndarray
            Current float64 residue.

        Returns
        -------
        cupy.ndarray
            Device int32 scalar; zero means continue.
        """
        _, _, _, counts = batched_extrema(r[None, :])
        sd = self._device_std(r)
        code = cp.zeros((), dtype="int32")
        if self.total_power_thr is not None:
            code = cp.where(cp.sum(cp.absolute(r)) < self.total_power_thr, 5, code)
        if self.range_thr is not None:
            code = cp.where(cp.ptp(r) < self.range_thr, 4, code)
        if self.residue_std_threshold > 0:
            code = cp.where(sd < self.residue_std_threshold, 3, code)
        code = cp.where(sd <= 64 * EPS, 2, code)
        return cp.where(counts[0, 0] + counts[0, 1] < 3, 1, code)

    def _imf_stop_code(self, r, skip=None):
        """Evaluate first-IMF acceptance and extractability without a host read.

        Parameters
        ----------
        r : cupy.ndarray
            Current float64 residue.
        skip : cupy.ndarray, optional
            Device int32 scalar that suppresses the EMD kernel when nonzero.

        Returns
        -------
        cupy.ndarray
            Device int32 scalar stop code; zero means continue.
        """
        _, available, _, iterations, state = batched_emd(
            r[None, :],
            max_modes=1,
            nbsym=self.EMD.nbsym,
            max_iteration=self.EMD.MAX_ITERATION,
            thresholds=self.EMD.rilling_thresholds,
            _device_result=True,
            _skip=skip,
        )
        code = cp.where(available[0, 0] & (iterations[0, 0] == 0), 6, 0)
        code = cp.where(~available[0, 0], 7, code)
        return cp.where(state[0] >= 3, 8, code)

    def _stop_reason(self, r, t, check_imf):
        if self.batch_emd:
            code = int(scalar(self._early_stop_code(r)))
            if code:
                return _EARLY_STOP_REASONS[code]
        else:
            ext = extrema(t, r)
            if len(ext[0]) + len(ext[2]) < 3:
                return "fewer_than_three_extrema"
            sd = stable_std(r, self.ddof)
            if sd <= 64 * EPS:
                return "numerical_zero"
            if self.residue_std_threshold > 0 and sd < self.residue_std_threshold:
                return "residue_std_threshold"
            if self.range_thr is not None and scalar(cp.ptp(r)) < self.range_thr:
                return "range_threshold"
            if (
                self.total_power_thr is not None
                and scalar(cp.sum(cp.absolute(r))) < self.total_power_thr
            ):
                return "residue_l1_threshold"
        if check_imf and self.batch_emd:
            code = int(scalar(self._imf_stop_code(r)))
            if code == 8:
                raise SiftingConvergenceError("Batched stop-IMF extraction did not converge.")
            return _EARLY_STOP_REASONS[code]
        if check_imf:
            native = isinstance(self.EMD, EMD)
            if native and self.EMD.imf_diagnostics(r, t)["is_imf"]:
                return "residue_is_imf"
            modes, _, _ = backend_parts(self.EMD, r, t, 1)
            if not len(modes):
                return "emd_has_no_extractable_imf"
            if not native:
                mean = r - modes[0]
                if scalar(cp.max(cp.absolute(mean))) <= 64 * EPS * max(
                    1.0, scalar(cp.max(cp.absolute(r)))
                ):
                    return "residue_is_imf_external"
        return None

    @on_device
    def end_condition(self, current_residue_r_k, max_imf_user_request, num_extracted_cimfs, T=None):
        r = vector(current_residue_r_k)
        t, _, _ = time_vector(T, len(r))
        requested = integer("max_imf", max_imf_user_request, -1)
        done = integer("num_extracted_cimfs", num_extracted_cimfs, 0)
        if requested >= 0 and done >= requested:
            return True
        return self._stop_reason(r, t, done > 0) is not None

    def __call__(self, S, T=None, max_imf=-1, progress=False, *, noise=None):
        return self.ceemdan(S, T, max_imf, progress, noise=noise)

    @on_device
    def ceemdan(self, S, T=None, max_imf=-1, progress=False, *, noise=None):
        """Decompose one uniformly sampled real signal on the selected GPU.

        Parameters
        ----------
        S : array_like
            Finite one-dimensional signal of length ``N``.
        T : array_like, optional
            Strictly increasing, uniformly spaced times of length ``N``.
            Sampling metadata is retained; the computation uses relative indices.
        max_imf : int, default -1
            Maximum extracted components; ``-1`` permits natural termination
            and ``0`` returns only the residue.
        progress : bool, default False
            Print progress after each completed component.
        noise : array_like, optional
            Finite real matrix of shape ``(trials, N)``. It is copied and
            bypasses the internal random stream without advancing its state.

        Returns
        -------
        cupy.ndarray
            Components with shape ``(K+1, N)``; the last row is the residue.
            The output dtype is configured by ``dtype`` at construction.
        """
        if self._active:
            raise RuntimeError("An ICEEMDAN instance cannot run concurrent decompositions.")
        self.C_IMF = self.residue = self.diagnostics_ = None
        self.all_noise_IMFs_for_CEEMDAN = []
        self._supplied_noise = None
        self._active = True
        saved = self._noise.snapshot()
        try:
            x = vector(S)
            t, dt, t0 = time_vector(T, len(x))
            requested = integer("max_imf", max_imf, -1)
            if requested > self.max_imf_iterations:
                raise ValueError(
                    "max_imf exceeds max_imf_iterations; raise the safety limit explicitly."
                )
            if not isinstance(progress, (bool, numpy_bool)):
                raise TypeError("progress must be bool.")
            if self.graph_control and progress:
                raise ValueError("graph_control cannot print per-stage progress.")
            info = dict(
                version=__version__,
                backend=f"{type(self.EMD).__module__}.{type(self.EMD).__qualname__}",
                cupy_version=cp.__version__,
                device=self.device_id,
                noise_kind=self.noise_kind,
                trials=self.trials,
                epsilon=self.epsilon,
                ddof=self.ddof,
                internal_dtype="float64",
                output_dtype=self.dtype.name,
                sample_interval=dt,
                sample_origin=t0,
                parallel=False,
                processes=1,
                rng=self._noise.mode,
                rng_policy="philox-request-v1",
                batch_emd=self.batch_emd,
                graph_control=self.graph_control,
                noise_input="supplied_W" if noise is not None else "generated",
                rng_state_sha256=hashlib.sha256(repr(saved).encode()).hexdigest(),
                noise_mode_counts=[],
                noise_bytes=0,
                noise_bytes_is_peak_vram=False,
                stages=[],
            )
            if noise is not None:
                W = device_array(noise)
                if W.ndim != 2 or W.shape != (self.trials, len(x)) or W.dtype.kind not in "iuf":
                    raise ValueError("noise must be a real (trials,N) matrix.")
                W = W.astype("float64", copy=True)
                if not scalar(cp.all(cp.isfinite(W))):
                    raise ValueError("Supplied noise contains non-finite values.")
                self._supplied_noise = W
            if self.graph_control:
                return self._ceemdan_graph(x, requested, info, saved)
            peak = float(scalar(cp.max(cp.absolute(x))))
            scaled = x / peak if peak else x.copy()
            std_scaled = stable_std(scaled, self.ddof)
            if requested == 0 or std_scaled == 0 or len(x) < 3:
                return self._finish(
                    x.reshape(1, -1),
                    info,
                    "max_imf" if requested == 0 else "constant_or_short",
                    requested != 0,
                    x,
                )
            r = scaled / std_scaled
            reason = self._stop_reason(r, t, False)
            if reason:
                return self._finish(
                    x.reshape(1, -1), info, reason, reason not in _NON_NATURAL_REASONS, x
                )
            self._noise_cap = requested if requested > 0 else self.max_imf_iterations
            if self.epsilon > 0:
                if self.batch_emd:
                    bank_modes, bank_available, first_std, mode_counts, noise_slots, _, _ = (
                        self._batched_noise_bank(r.shape)
                    )
                else:
                    self._pre_decompose_noise_for_ceemdan(r.shape, t)
                    banks = self.all_noise_IMFs_for_CEEMDAN
                    if len(banks) != self.trials:
                        raise BackendContractError("Noise decomposition changed the ensemble size.")
                    info["noise_mode_counts"] = [len(b) for b in banks]
                    info["noise_bytes"] = sum(b.nbytes for b in banks)
                    first_std = [stable_std(b[0], self.ddof) if len(b) else 0.0 for b in banks]
                    if any(v <= 0 or not math.isfinite(v) for v in first_std):
                        raise BackendContractError(
                            "Every realization needs a finite nonzero first noise IMF."
                        )
            else:
                banks = [cp.empty((0, len(x)), dtype="float64") for _ in range(self.trials)]
                first_std = [1.0] * self.trials
                info["noise_mode_counts"] = [0] * self.trials
            modes: list[Any] = []  # CuPy 14.2.0 has no typed ndarray interface.
            deferred_stage_diagnostics = []
            while True:
                if requested >= 0 and len(modes) >= requested:
                    reason = "max_imf"
                    break
                self._context(f"stop:{len(modes)}")
                reason = self._stop_reason(r, t, len(modes) > 0)
                if reason:
                    break
                if len(modes) >= self.max_imf_iterations:
                    raise DecompositionLimitError(
                        "ICEEMDAN safety limit reached before natural termination."
                    )
                k = len(modes) + 1
                sd = (
                    self._device_std(r)
                    if self.batch_emd and self.epsilon > 0
                    else stable_std(r, self.ddof)
                )
                availability = []
                perturbation_stds = []
                counts = []
                if self.epsilon == 0:
                    self._context(f"stage:{k}:unperturbed")
                    extracted, _, self._last_iterations = backend_parts(self.EMD, r, t, 1)
                    if not len(extracted):
                        reason = "emd_has_no_extractable_imf"
                        break
                    new_r = r - extracted[0]
                    counts = [self._last_iterations] * self.trials
                    availability = [False] * self.trials
                    perturbation_stds = [0.0] * self.trials
                else:
                    if self.batch_emd:
                        exists = (
                            bank_available[k - 1]
                            if k <= len(bank_available)
                            else cp.zeros(self.trials, dtype="bool")
                        )
                        noise_mode = (
                            bank_modes[k - 1]
                            if k <= len(bank_modes)
                            else cp.zeros((self.trials, len(r)), dtype="float64")
                        )
                        amplitude = self.epsilon * sd
                        eta = (
                            amplitude * noise_mode
                            if k > 1
                            else amplitude * (noise_mode / first_std[:, None])
                        )
                        y = r[None, :] + eta
                        local_modes, local_available, _, sift_counts, state = batched_emd(
                            y,
                            max_modes=1,
                            nbsym=self.EMD.nbsym,
                            max_iteration=self.EMD.MAX_ITERATION,
                            thresholds=self.EMD.rilling_thresholds,
                            _device_result=True,
                        )
                        if scalar(cp.any(state >= 3)):
                            failed = cp.asnumpy(state)
                            raise SiftingConvergenceError(
                                f"Batched EMD failed for rows {list((failed >= 3).nonzero()[0])}; "
                                "states: 3=bad spline, 4=stagnation, 5=iteration limit, "
                                "6=non-finite."
                            )
                        mean = y - local_modes[0]
                        new_r = ordered_mean(mean)
                        stage_arrays = (
                            exists,
                            self._batched_std(eta),
                            sift_counts[0],
                            local_available[0],
                            cp.broadcast_to(sd, (self.trials,)),
                        )
                    else:
                        total = cp.zeros_like(r)
                        correction = cp.zeros_like(r)
                        # All realizations see the SAME r until the complete reduction.
                        for i in range(self.trials):
                            exists = len(banks[i]) >= k
                            availability.append(exists)
                            if exists:
                                amplitude = self.epsilon * sd
                                eta = (
                                    amplitude * (banks[i][0] / first_std[i])
                                    if k == 1
                                    else amplitude * banks[i][k - 1]
                                )
                                perturbation_stds.append(stable_std(eta, self.ddof))
                                y = r + eta
                            else:
                                y = r.copy()
                                perturbation_stds.append(0.0)
                            self._context(f"stage:{k}:realization:{i}")
                            mean = self._get_emd_local_mean(y, t)
                            self._emit("local_mean", stage=k, realization=i, input=y, mean=mean)
                            total, correction = kahan_add(total, correction, mean)
                            counts.append(self._last_iterations)
                        new_r = total / self.trials
                if self.batch_emd and self.epsilon > 0:
                    stage_status = int(scalar(_stage_status(r, new_r)))
                    if stage_status == 1:
                        raise FloatingPointError("Non-finite ICEEMDAN residue.")
                    if stage_status == 2:
                        raise SiftingConvergenceError(
                            "ICEEMDAN made no progress before its terminal condition."
                        )
                else:
                    if not scalar(cp.all(cp.isfinite(new_r))):
                        raise FloatingPointError("Non-finite ICEEMDAN residue.")
                    if scalar(cp.array_equal(r, new_r)):
                        raise SiftingConvergenceError(
                            "ICEEMDAN made no progress before its terminal condition."
                        )
                component = r - new_r
                modes.append(component)
                stage_info = dict(
                    index=k,
                    noise_mode_index=k,
                    ensemble_denominator=self.trials,
                    missing_noise_modes=None,
                    perturbation_stds=None,
                    residue_std_before=None if self.batch_emd and self.epsilon > 0 else sd,
                    sift_iterations=None,
                )
                if self.batch_emd and self.epsilon > 0:
                    deferred_stage_diagnostics.append((stage_info, stage_arrays))
                else:
                    stage_info.update(
                        missing_noise_modes=availability.count(False),
                        perturbation_stds=perturbation_stds,
                        sift_iterations=counts,
                    )
                info["stages"].append(stage_info)
                self._emit("stage", index=k, previous_residue=r, residue=new_r, component=component)
                r = new_r
                if progress:
                    print(
                        f"ICEEMDAN: component {k}, residual std={stable_std(r, self.ddof):.6g}",
                        flush=True,
                    )
            if deferred_stage_diagnostics:
                details = cp.asnumpy(
                    cp.stack([cp.stack(arrays) for _, arrays in deferred_stage_diagnostics])
                )
                for (stage_info, _), (exists, stds, iterations, present, residue_stds) in zip(
                    deferred_stage_diagnostics, details, strict=True
                ):
                    stage_info.update(
                        missing_noise_modes=int(self.trials - exists.sum()),
                        perturbation_stds=stds.tolist(),
                        residue_std_before=float(residue_stds[0]),
                        sift_iterations=[
                            int(count) if available else None
                            for count, available in zip(iterations, present, strict=True)
                        ],
                    )
            if self.batch_emd and self.epsilon > 0:
                info["noise_mode_counts"] = cp.asnumpy(mode_counts).tolist()
                info["noise_bytes"] = int(scalar(noise_slots)) * self.trials * len(r) * 8
            normalized = cp.vstack((*modes, r))
            components = (normalized * std_scaled) * peak
            if not scalar(cp.all(cp.isfinite(components))):
                raise FloatingPointError("ICEEMDAN components exceed float64 range.")
            info["reconstruction_scaled_linf_internal"] = float(
                scalar(cp.max(cp.absolute(cp.sum(components / peak, axis=0) - x / peak)))
            )
            natural = reason not in _NON_NATURAL_REASONS
            return self._finish(components, info, reason, natural, x)
        except BaseException:
            self.C_IMF = self.residue = self.diagnostics_ = None
            self._noise.restore(saved)
            raise
        finally:
            self.all_noise_IMFs_for_CEEMDAN = []
            self._supplied_noise = None
            self._active = False
            self._context(None)

    def _ceemdan_graph(self, x, requested, info, saved):
        from .gpu_loop import graph_stages

        if requested == 0 or len(x) < 3:
            return self._finish(
                x.reshape(1, -1),
                info,
                "max_imf" if requested == 0 else "constant_or_short",
                requested != 0,
                x,
            )
        peak = cp.max(cp.absolute(x))
        divisor = cp.where(peak > 0, peak, 1.0)
        scaled = x / divisor
        std_scaled = self._device_std(scaled)
        r = scaled / cp.where(std_scaled > 0, std_scaled, 1.0)
        self._noise_cap = min(
            requested if requested > 0 else self.max_imf_iterations,
            self.graph_workspace_modes,
        )
        initial_early = cp.where(std_scaled == 0, 14, self._early_stop_code(r))
        bank_info = {}

        def setup(state_allocator):
            result = self._batched_noise_bank(
                r.shape, defer_errors=True, _state_allocator=state_allocator
            )
            bank_info["counts"], bank_info["used"] = result[3:5]
            return result

        modes, r, reason, stages = graph_stages(
            self,
            r,
            requested=requested,
            initial_std=std_scaled,
            initial_early=initial_early,
            setup=setup,
            resource_info=info,
        )
        if not len(modes):
            self._noise.restore(saved)
            return self._finish(
                x.reshape(1, -1),
                info,
                reason,
                reason not in _NON_NATURAL_REASONS,
                x,
            )
        info["stages"] = stages
        info["noise_mode_counts"] = cp.asnumpy(bank_info["counts"]).tolist()
        info["noise_bytes"] = int(scalar(bank_info["used"])) * self.trials * len(r) * 8
        components = (cp.vstack((modes, r)) * std_scaled) * peak
        if not scalar(cp.all(cp.isfinite(components))):
            raise FloatingPointError("ICEEMDAN components exceed float64 range.")
        info["reconstruction_scaled_linf_internal"] = float(
            scalar(cp.max(cp.absolute(cp.sum(components / divisor, axis=0) - x / divisor)))
        )
        natural = reason not in _NON_NATURAL_REASONS
        return self._finish(components, info, reason, natural, x)

    def _finish(self, components, info, reason, complete, original):
        output = components.astype(self.dtype, copy=False)
        if not scalar(cp.all(cp.isfinite(output))):
            raise FloatingPointError(f"Results are not representable in output dtype {self.dtype}.")
        peak = cp.max(cp.absolute(original))
        divisor = cp.where(peak > 0, peak, 1.0)
        info["reconstruction_scaled_linf"] = float(
            scalar(
                cp.max(
                    cp.absolute(
                        cp.sum(output.astype("float64", copy=False) / divisor, axis=0)
                        - original / divisor
                    )
                )
            )
        )
        imfs = output[:-1].copy()
        residue = output[-1].copy()
        info.update(
            stop_reason=reason,
            natural_termination=bool(complete),
            n_components=len(output) - 1,
            residue_row=len(output) - 1,
        )
        diagnostic_kernel = (
            self.EMD if isinstance(self.EMD, EMD) else EMD(device=self.device_id, trace=self.trace)
        )
        info["component_diagnostic_rule"] = dict(
            kind="Rilling",
            spline_kind=diagnostic_kernel.spline_kind,
            nbsym=diagnostic_kernel.nbsym,
            thresholds=list(diagnostic_kernel.rilling_thresholds),
            matches_decomposition_backend=isinstance(self.EMD, EMD),
        )
        self._context("output_diagnostics")
        info["component_imf_diagnostics"] = [
            diagnostic_kernel.imf_diagnostics(row) for row in output[:-1]
        ]
        result = output.copy()
        synchronize()
        self.C_IMF = imfs
        self.residue = residue
        self.diagnostics_ = info
        return result

    @on_device
    def get_imfs_and_residue(self):
        """Return copies of the last successful decomposition's parts.

        Returns
        -------
        tuple of cupy.ndarray
            IMFs of shape ``(K, N)`` and residue of shape ``(N,)``. Both are
            device-resident copies that do not alias the stored state.
        """
        if self.C_IMF is None or self.residue is None:
            raise ValueError("No successful ICEEMDAN decomposition is available.")
        return self.C_IMF.copy(), self.residue.copy()


CEEMDAN = ICEEMDAN
