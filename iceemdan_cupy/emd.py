"""Stage 3: CuPy EMD and explicit IMF/residue contract.

Control/stopping conditions follow reference lines 285–482. No CPU envelope
or numeric fallback; all output arrays are device-resident.
"""

from __future__ import annotations

import math
from typing import Any

from numpy import finfo as finfo_metadata

from .contracts import (
    EPS,
    BackendContractError,
    DecompositionLimitError,
    SiftingConvergenceError,
    extrema,
    float_dtype,
    integer,
    nonnegative,
    time_vector,
    vector,
)
from .geometry import Geometry
from .runtime import cp, device_array, on_device, require_device_array, scalar, synchronize


class EMD(Geometry):
    """Bounded real EMD with explicit IMF/residue storage.

    The default criterion is Rilling's envelope ratio test (0.05, 0.5, 0.05)
    AND |number of extrema - number of zeros| <= 1. Unlike the upstream
    Cauchy-style loop, it does not accept MAX_ITERATION as convergence.

    Cubic uses the PyEMD geometry: SciPy not-a-knot for >3 support points,
    and the upstream three-point spline otherwise. Other spline choices
    are numerical variants, not an exact reproduction of the paper's kernel.

    Parameters
    ----------
    spline_kind : str, default "cubic"
        Interpolator for the upper and lower envelopes.
    nbsym : int, default 2
        Number of extrema reflected at each boundary.
    MAX_ITERATION : int, default 2000
        Maximum siftings per candidate IMF; exhaustion raises an error.
    rilling_thresholds : tuple of float, default (0.05, 0.5, 0.05)
        Lower ratio, upper ratio, and allowed fraction above the lower ratio.
    max_modes : int, default 100
        Safety limit on extracted modes.
    extrema_detection : str, default "simple"
        Only discrete ``"simple"`` detection is supported.
    DTYPE : {"float64", "float32"}, default "float64"
        Output dtype; internal calculation remains float64.
    device : int, default 0
        CUDA device index.
    trace : callable, optional
        Callback receiving diagnostic events and device arrays.
    """

    def __init__(
        self,
        spline_kind="cubic",
        nbsym=2,
        *,
        MAX_ITERATION=2000,
        rilling_thresholds=(0.05, 0.5, 0.05),
        max_modes=100,
        extrema_detection="simple",
        DTYPE="float64",
        device=0,
        trace=None,
    ):
        self.device_id = integer("device", device, 0)
        self.trace = trace
        self.trace_context = None
        if trace is not None and not callable(trace):
            raise TypeError("trace must be callable or None.")
        if extrema_detection != "simple":
            raise ValueError(
                "The bundled kernel supports discrete extrema only; "
                "use an explicit external backend for other definitions."
            )
        if spline_kind not in (
            "cubic",
            "pchip",
            "akima",
            "cubic_hermite",
            "linear",
            "slinear",
            "quadratic",
        ):
            raise ValueError("Unsupported spline_kind.")
        self.spline_kind = spline_kind
        self.nbsym = integer("nbsym", nbsym)
        self.MAX_ITERATION = integer("MAX_ITERATION", MAX_ITERATION)
        self.max_modes = integer("max_modes", max_modes)
        self.output_dtype = float_dtype(DTYPE)
        self.DTYPE = "float64"
        self.extrema_detection = extrema_detection
        values = tuple(nonnegative("rilling threshold", v) for v in rilling_thresholds)
        if len(values) != 3 or values[0] <= 0 or values[1] < values[0] or not 0 <= values[2] < 1:
            raise ValueError("Require 0 < threshold1 <= threshold2 and 0 <= fraction < 1.")
        self.rilling_thresholds = values
        self.imfs = self.residue = None
        self.diagnostics_ = []
        self.stop_reason_ = None

    @on_device
    def find_extrema(self, T, S):
        return extrema(device_array(T), device_array(S))

    @on_device
    def extract_max_min_spline(self, T, S):
        hi, hiv, lo, lov, _ = self.find_extrema(T, S)
        if len(hi) + len(lo) < 3 or not len(hi) or not len(lo):
            raise ValueError("At least three alternating extrema are needed for envelopes.")
        ehi, elo = self.prepare_points_simple(T, S, hi, hiv, lo, lov)
        _, upper = self.spline_points(T, ehi)
        _, lower = self.spline_points(T, elo)
        if upper.shape != S.shape or lower.shape != S.shape:
            raise SiftingConvergenceError("Envelopes do not cover the complete record.")
        if not cp.all(cp.isfinite(upper)) or not cp.all(cp.isfinite(lower)):
            raise SiftingConvergenceError("Envelope interpolation produced non-finite values.")
        self._emit(
            "envelopes", signal=S, upper=upper, lower=lower, upper_knots=ehi, lower_knots=elo
        )
        return upper, lower, ehi, elo

    @on_device
    def _criterion(self, x, t):
        ext = self.find_extrema(t, x)
        ne, nz = len(ext[0]) + len(ext[2]), len(ext[4])
        info: dict[str, int | float | bool | None] = dict(
            extrema=ne,
            zeros=nz,
            is_imf=False,
            envelope_ratio_max=None,
            fraction_above_threshold=None,
        )
        if ne < 3:
            return info, None
        upper, lower, _, _ = self.extract_max_min_spline(t, x)
        mean = 0.5 * upper + 0.5 * lower
        amplitude = cp.absolute(0.5 * upper - 0.5 * lower)
        # CuPy divide has no documented where= argument. Avoid even transient
        # division by zero; reproduce the original inf / zero conventions.
        positive = amplitude > 0
        denominator = cp.where(positive, amplitude, 1.0)
        ratio = cp.where(positive, cp.absolute(mean) / denominator, float("inf"))
        ratio = cp.where((amplitude == 0) & (mean == 0), 0.0, ratio)
        threshold1, threshold2, allowed = self.rilling_thresholds
        fraction = float(cp.mean(ratio > threshold1))
        max_ratio = float(cp.max(ratio))
        info.update(
            envelope_ratio_max=max_ratio,
            fraction_above_threshold=fraction,
            is_imf=bool(abs(ne - nz) <= 1 and fraction <= allowed and max_ratio <= threshold2),
        )
        self._emit("criterion", signal=x, mean=mean, **info)
        return info, mean

    @on_device
    def imf_diagnostics(self, S, T=None):
        x = vector(S)
        t, _, _ = time_vector(T, len(x))
        peak = float(cp.max(cp.absolute(x)))
        # Rescaling prevents underflow in spline algebra without altering criteria.
        info, _ = self._criterion(x / peak if peak else x, t)
        return info

    def _first(self, signal, t):
        h = signal.copy()
        for count in range(self.MAX_ITERATION + 1):
            info, mean = self._criterion(h, t)
            if mean is None:
                return None, dict(
                    info, sift_iterations=count, converged=False, reason="fewer_than_three_extrema"
                )
            if info["is_imf"]:
                return h, dict(info, sift_iterations=count, converged=True, reason="rilling")
            if count == self.MAX_ITERATION:
                raise SiftingConvergenceError(
                    f"No IMF convergence after {count} siftings; "
                    f"extrema={info['extrema']}, zeros={info['zeros']}, "
                    f"max envelope ratio={info['envelope_ratio_max']:.6g}."
                )
            candidate = h - mean
            if cp.array_equal(candidate, h):
                raise SiftingConvergenceError(
                    "Sifting stagnated before satisfying the IMF criterion."
                )
            h = candidate
        raise AssertionError("Unreachable sifting state")

    def __call__(self, S, T=None, max_imf=-1):
        return self.emd(S, T=T, max_imf=max_imf)

    @on_device
    def emd(self, S, T=None, max_imf=-1):
        """Extract bounded IMFs and return them with the final residue.

        Parameters
        ----------
        S : array_like
            Finite one-dimensional real signal of length ``N``.
        T : array_like, optional
            Strictly increasing, uniformly spaced sample times.
        max_imf : int, default -1
            Maximum number of IMFs; ``-1`` uses natural termination up to
            ``max_modes``, and ``0`` returns only the residue.

        Returns
        -------
        cupy.ndarray
            Device array of shape ``(K+1, N)`` with the residue in the final
            row. Values use the configured output dtype.
        """
        self.imfs = self.residue = None
        self.diagnostics_ = []
        self.stop_reason_ = None
        x = vector(S)
        t, _, _ = time_vector(T, len(x))
        max_imf = integer("max_imf", max_imf, -1)
        if max_imf > self.max_modes:
            raise ValueError("max_imf exceeds the EMD max_modes safety limit.")
        peak = float(cp.max(cp.absolute(x)))
        u = x / peak if peak else x.copy()
        r = u.copy()
        modes: list[Any] = []  # CuPy 14.2.0 has no typed ndarray interface.
        limit = max_imf if max_imf >= 0 else self.max_modes
        floor = 64 * EPS
        while len(modes) < limit:
            ext = self.find_extrema(t, r)
            if len(ext[0]) + len(ext[2]) < 3:
                self.stop_reason_ = "fewer_than_three_extrema"
                break
            if cp.max(cp.absolute(r)) <= floor:
                self.stop_reason_ = "numerical_zero"
                break
            h, info = self._first(r, t)
            if h is None:
                self.stop_reason_ = "fewer_than_three_extrema"
                break
            new_r = r - h
            if cp.array_equal(new_r, r):
                raise SiftingConvergenceError("EMD extraction made no progress.")
            modes.append(h)
            self.diagnostics_.append(info)
            r = new_r
        else:
            if max_imf < 0:
                ext = self.find_extrema(t, r)
                if len(ext[0]) + len(ext[2]) >= 3 and cp.max(cp.absolute(r)) > floor:
                    raise DecompositionLimitError("EMD max_modes reached before termination.")
                self.stop_reason_ = "terminal_at_limit"
            else:
                self.stop_reason_ = "max_imf"
        restored = (cp.stack(modes) if modes else cp.empty((0, len(x)), dtype="float64")) * peak
        residue = r * peak if peak else r
        result = cp.vstack((restored, residue)).astype(self.output_dtype, copy=False)
        self.imfs, self.residue = result[:-1].copy(), result[-1].copy()
        if not cp.all(cp.isfinite(result)):
            self.imfs = self.residue = None
            raise FloatingPointError(f"EMD output is not representable in {self.output_dtype}.")
        # IMF and residue remain unambiguously separate; never omit a zero row.
        synchronize()
        return result

    @on_device
    def get_imfs_and_residue(self):
        """Return copies of the last successful EMD extraction.

        Returns
        -------
        tuple of cupy.ndarray
            IMFs of shape ``(K, N)`` and residue of shape ``(N,)``.
        """
        if self.imfs is None or self.residue is None:
            raise ValueError("No successful EMD decomposition is available.")
        return self.imfs.copy(), self.residue.copy()

    def _emit(self, event, **data):
        if self.trace is not None:
            payload = {k: v.copy() if isinstance(v, cp.ndarray) else v for k, v in data.items()}
            payload["context"] = self.trace_context
            self.trace(event, payload)


def backend_parts(backend, signal, t, max_imf):
    require_device_array(signal, "signal")
    backend.emd(signal.copy(), t, max_imf=max_imf)
    parts, remainder = backend.get_imfs_and_residue()
    if not isinstance(parts, cp.ndarray) or not isinstance(remainder, cp.ndarray):
        raise BackendContractError(
            "EMD must return GPU-resident IMFs AND residue; CPU fallback rejected."
        )
    require_device_array(parts, "IMFs")
    require_device_array(remainder, "residue")
    raw_dtype = parts.dtype
    if raw_dtype.kind not in "iuf" or remainder.dtype.kind not in "iuf":
        raise BackendContractError("EMD getter must return real numeric arrays.")
    modes = parts.astype("float64", copy=False)
    remainder = remainder.astype("float64", copy=False)
    n = len(signal)
    if modes.ndim != 2 or modes.shape[1] != n or remainder.shape != (n,):
        raise BackendContractError("EMD getter must return (K,N) IMFs and an (N,) residue.")
    if max_imf >= 0 and len(modes) > max_imf:
        raise BackendContractError("EMD returned more IMFs than requested.")
    if not scalar(cp.all(cp.isfinite(modes))) or not scalar(cp.all(cp.isfinite(remainder))):
        raise BackendContractError("EMD returned non-finite values.")
    eps = finfo_metadata(raw_dtype).eps if raw_dtype.kind == "f" else EPS
    peak = float(scalar(cp.max(cp.absolute(signal))))
    tolerance = 256 * eps * max(1, len(modes))
    reconstructed = cp.sum(modes, axis=0) + remainder
    defect = (
        float(scalar(cp.max(cp.absolute(reconstructed / peak - signal / peak))))
        if peak
        else float(scalar(cp.max(cp.absolute(reconstructed))))
    )
    if not math.isfinite(defect) or defect > (tolerance if peak else 0.0):
        raise BackendContractError(
            f"EMD getter violates reconstruction; scaled defect={defect:.3g}."
        )
    records = getattr(backend, "diagnostics_", None)
    iterations = None
    if records and len(modes) and isinstance(records[0], dict):
        iterations = records[0].get("sift_iterations")
    return modes.copy(), remainder.copy(), iterations


def local_mean(backend, signal, t):
    modes, _, iterations = backend_parts(backend, signal, t, 1)
    return (signal - modes[0] if len(modes) else signal.copy()), iterations
