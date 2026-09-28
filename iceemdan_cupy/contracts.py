"""Stage 1: scalar metadata validation and GPU-only signal statistics/extrema.

Derived from ICEEMDAN(2).py lines 169–282; changed 2026-09-25 for CuPy.
NumPy is imported only for dtype/type metadata, never for signal arithmetic.
"""

from __future__ import annotations

import math
import numbers

from numpy import bool_ as numpy_bool
from numpy import dtype as dtype_metadata
from numpy import finfo as finfo_metadata

from .runtime import cp, device_array, scalar

EPS = finfo_metadata("float64").eps

_COMPACT_INDICES = cp.RawKernel(
    r"""
extern "C" __global__ void compact_indices(const bool* mask, const long long* prefix,
                                            long long* out, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && mask[i]) out[prefix[i] - 1] = i;
}
""",
    "compact_indices",
)


def _flatnonzero(mask):
    """Compact a boolean mask without CuPy 14.2's race-reported nonzero scan."""
    n = mask.size
    if not n:
        return cp.empty(0, dtype="int64")
    prefix = cp.cumsum(mask, dtype="int64")
    result = cp.empty(int(prefix[-1].item()), dtype="int64")
    _COMPACT_INDICES(((n + 127) // 128,), (128,), (mask, prefix, result, n))
    return result


class SiftingConvergenceError(RuntimeError):
    """Candidate has not satisfied the declared IMF criterion."""


class DecompositionLimitError(RuntimeError):
    """Safety limit reached without natural termination."""


class BackendContractError(RuntimeError):
    """An EMD backend violated its GPU IMF/residue contract."""


def integer(name, value, minimum=1):
    if isinstance(value, (bool, numpy_bool)) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer, not {type(value).__name__}.")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    return int(value)


def nonnegative(name, value, allow_none=False):
    if value is None and allow_none:
        return None
    if isinstance(value, (bool, numpy_bool)) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a finite real number.")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return value


def float_dtype(dtype):
    d = dtype_metadata(dtype)
    if d not in (dtype_metadata("float32"), dtype_metadata("float64")):
        raise ValueError("dtype must be float32 or float64; arithmetic uses float64.")
    return d


def vector(signal, name="S"):
    raw = device_array(signal)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array.")
    if raw.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numeric values, not {raw.dtype}.")
    out = raw.astype("float64", copy=True)
    if not scalar(cp.all(cp.isfinite(out))):
        raise ValueError(f"{name} contains NaN, infinity, or values outside float64.")
    return out


def time_vector(time, n):
    if time is None:
        return cp.arange(n, dtype="float64"), None, None
    t = vector(time, "T")
    if len(t) != n:
        raise ValueError("S and T must have the same length.")
    if n == 1:
        return cp.zeros(1, dtype="float64"), None, float(scalar(t[0]))
    diff = cp.diff(t)
    if not scalar(cp.all(cp.isfinite(diff))) or scalar(cp.any(diff <= 0)):
        raise ValueError("T must be finite and strictly increasing.")
    dt = float(scalar(cp.median(diff)))
    ulp = EPS * max(float(scalar(cp.max(cp.absolute(t)))), dt)
    if dt < 32 * ulp:
        raise ValueError("T has insufficient numerical resolution; use relative times.")
    tolerance = max(1e-7 * dt, 8 * ulp)
    if scalar(cp.any(cp.absolute(diff - dt) > tolerance)):
        raise ValueError("ICEEMDAN requires uniform sampling; resample explicitly first.")
    return cp.arange(n, dtype="float64"), dt, float(scalar(t[0]))


def stable_std(x, ddof=1):
    x = device_array(x, dtype="float64")
    if x.size <= ddof:
        return 0.0
    peak = float(scalar(cp.max(cp.absolute(x))))
    if peak == 0:
        return 0.0
    return peak * float(scalar(cp.std(x / peak, ddof=ddof)))


def extrema(T, S):
    """Interior plateau midpoints (ties to even) and zero-run locations.

    No multiplication for sign comparisons and no out-of-range gathers.
    Compacting arrays is allowed to synchronize in this reference port.

    Parameters
    ----------
    T : cupy.ndarray
        One-dimensional time coordinates in the same device as ``S``.
    S : cupy.ndarray
        One-dimensional signal values.

    Returns
    -------
    tuple of cupy.ndarray
        Maximum times and values, minimum times and values, and zero-crossing
        sample locations, in that order. Each array has a variable length.
    """
    n = len(S)
    crosses = _flatnonzero(((S[:-1] < 0) & (S[1:] > 0)) | ((S[:-1] > 0) & (S[1:] < 0)))
    zero = cp.concatenate((cp.zeros(1, dtype="bool"), S == 0, cp.zeros(1, dtype="bool"))).astype(
        "int8"
    )
    edges = cp.diff(zero)
    start = _flatnonzero(edges == 1)
    end = _flatnonzero(edges == -1) - 1
    zeros = cp.sort(cp.concatenate((crosses, cp.rint((start + end) / 2.0))))
    starts = cp.concatenate((cp.zeros(1, dtype="int64"), _flatnonzero(S[1:] != S[:-1]) + 1))
    ends = cp.concatenate((starts[1:] - 1, cp.full(1, n - 1, dtype="int64")))
    values = S[starts]
    if len(values) < 3:
        empty = cp.empty(0, dtype="float64")
        return empty, empty, empty, empty, zeros
    mids = cp.rint((starts[1:-1] + ends[1:-1]) / 2.0).astype("int64")
    high = mids[(values[1:-1] > values[:-2]) & (values[1:-1] > values[2:])]
    low = mids[(values[1:-1] < values[:-2]) & (values[1:-1] < values[2:])]
    return T[high], S[high], T[low], S[low], zeros
