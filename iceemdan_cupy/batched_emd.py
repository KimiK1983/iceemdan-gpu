"""Fixed-shape CUDA intermediates for batched EMD."""

from __future__ import annotations

from numpy import float64, int32

from .contracts import EPS, SiftingConvergenceError, integer, nonnegative
from .runtime import cp, require_device_array


def _check_index_products(batch, n):
    limit = 2**31 - 1
    if batch < 1 or n < 1 or batch > limit // 3 or n > limit // 4 or batch > limit // n:
        raise ValueError("Batch/sample index products exceed the CUDA int32 contract.")


_EXTREMA_SOURCE = r"""
__device__ void detect_extrema(const double* signal, int* hi, int* lo,
                               int* zz, int* counts, int n) {
    int nh = 0, nl = 0, nz = 0;

    for (int start = 0; start < n;) {
        int end = start;
        double value = signal[start];
        while (end + 1 < n && signal[end + 1] == value) ++end;
        int midpoint = __double2int_rn(0.5 * ((double)start + (double)end));

        if (value == 0.0) zz[nz++] = midpoint;
        if (start > 0 && end + 1 < n) {
            double left = signal[start - 1];
            double right = signal[end + 1];
            if (value > left && value > right) hi[nh++] = midpoint;
            if (value < left && value < right) lo[nl++] = midpoint;
        }
        if (end + 1 < n &&
            ((value < 0.0 && signal[end + 1] > 0.0) ||
             (value > 0.0 && signal[end + 1] < 0.0))) {
            zz[nz++] = end;
        }
        start = end + 1;
    }
    for (int i = nh; i < n; ++i) hi[i] = -1;
    for (int i = nl; i < n; ++i) lo[i] = -1;
    for (int i = nz; i < n; ++i) zz[i] = -1;
    counts[0] = nh;
    counts[1] = nl;
    counts[2] = nz;
}
extern "C" __global__
void extrema_batch(const double* signals, int* maxima, int* minima,
                   int* zeros, int* counts, int n, int batch) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batch) return;
    size_t offset = (size_t)row * n;
    detect_extrema(signals + offset, maxima + offset, minima + offset,
                   zeros + offset, counts + 3 * row, n);
}
"""

_EXTREMA_KERNEL = cp.RawKernel(_EXTREMA_SOURCE, "extrema_batch", backend="nvrtc")


def extrema_into(signals, maxima, minima, zeros, counts):
    """Write sorted sample indices and device counts for each signal row.

    All arrays must be contiguous on one GPU. Unused index slots become -1.
    The input must already be validated as finite real float64 data.

    Parameters
    ----------
    signals : cupy.ndarray
        Input with shape ``(batch, samples)``.
    maxima, minima, zeros : cupy.ndarray
        Caller-owned int32 index buffers shaped like ``signals``.
    counts : cupy.ndarray
        Caller-owned int32 counts with shape ``(batch, 3)``.
    """
    require_device_array(signals, "signals")
    if signals.ndim != 2 or signals.dtype != float64 or not signals.flags.c_contiguous:
        raise ValueError("signals must be a contiguous float64 (batch, samples) CuPy array.")
    batch, n = signals.shape
    if batch < 1 or batch > 2**31 - 1 or n < 1 or n > 2**31 - 1:
        raise ValueError("Require a nonempty batch with <= 2^31-1 samples per row.")
    _check_index_products(batch, n)
    for name, output in (("maxima", maxima), ("minima", minima), ("zeros", zeros)):
        require_device_array(output, name)
        if output.shape != signals.shape or output.dtype != int32 or not output.flags.c_contiguous:
            raise ValueError(f"{name} must be a contiguous int32 array shaped like signals.")
    require_device_array(counts, "counts")
    if counts.shape != (batch, 3) or counts.dtype != int32 or not counts.flags.c_contiguous:
        raise ValueError("counts must be a contiguous int32 (batch, 3) CuPy array.")
    _EXTREMA_KERNEL(
        ((batch + 127) // 128,),
        (128,),
        (signals, maxima, minima, zeros, counts, int32(n), int32(batch)),
    )


def batched_extrema(signals):
    """Allocate fixed-shape device outputs and detect extrema without host reads.

    Parameters
    ----------
    signals : cupy.ndarray
        Contiguous float64 input with shape ``(batch, samples)``.

    Returns
    -------
    tuple of cupy.ndarray
        Maxima, minima, zeros and counts on the device.
    """
    require_device_array(signals, "signals")
    if signals.ndim != 2:
        raise ValueError("signals must have shape (batch, samples).")
    _check_index_products(*signals.shape)
    outputs = [cp.empty(signals.shape, dtype="int32") for _ in range(3)]
    counts = cp.empty((len(signals), 3), dtype="int32")
    extrema_into(signals, outputs[0], outputs[1], outputs[2], counts)
    return *outputs, counts


_REFLECTION_SOURCE = r"""
struct Seq {
    const int* data;
    int start, stop, extra, extra_first, reverse;
};
__device__ int small(int a, int b) { return a < b ? a : b; }
__device__ int large(int a, int b) { return a > b ? a : b; }
__device__ Seq seq(const int* data, int start, int stop, int extra = -1,
                   int extra_first = 0, int reverse = 1) {
    Seq q = {data, start, stop, extra, extra_first, reverse};
    return q;
}
__device__ int length(Seq q) { return q.stop - q.start + (q.extra >= 0); }
__device__ int at(Seq q, int j) {
    if (q.extra_first) {
        if (j == 0) return q.extra;
        --j;
    }
    int inner = q.stop - q.start;
    if (j >= inner) return q.extra;
    return q.data[q.reverse ? q.stop - 1 - j : q.start + j];
}
__device__ int write_knots(double* x, double* y, int capacity, const double* signal,
                           Seq left, const int* core, int core_count, Seq right,
                           int left_sym, int right_sym) {
    int used = 0;
    for (int j = 0; j < length(left); ++j) {
        int index = at(left, j);
        x[used] = 2.0 * left_sym - index;
        y[used++] = signal[index];
    }
    for (int j = 0; j < core_count; ++j) {
        int index = core[j];
        x[used] = index;
        y[used++] = signal[index];
    }
    for (int j = 0; j < length(right); ++j) {
        int index = at(right, j);
        x[used] = 2.0 * right_sym - index;
        y[used++] = signal[index];
    }
    int kept = 0;
    for (int j = 0; j < used; ++j) {
        if (j + 1 == used || x[j] != x[j + 1]) {
            x[kept] = x[j];
            y[kept++] = y[j];
        }
    }
    for (int j = kept; j < capacity; ++j) { x[j] = 0.0; y[j] = 0.0; }
    return kept;
}
__device__ void clear_knots(double* upper_x, double* upper_y,
                            double* lower_x, double* lower_y, int capacity) {
    for (int j = 0; j < capacity; ++j) {
        upper_x[j] = upper_y[j] = 0.0;
        lower_x[j] = lower_y[j] = 0.0;
    }
}
__device__ int reflect_one(const double* s, const int* hi, const int* lo,
                           const int* extrema_counts, double* upper_x,
                           double* upper_y, double* lower_x, double* lower_y,
                           int* knot_counts, int n, int nbsym) {
    int nh = extrema_counts[0], nl = extrema_counts[1];
    int capacity = 2 * n;
    knot_counts[0] = knot_counts[1] = 0;
    if (nh == 0 || nl == 0 || nh + nl < 3) {
        clear_knots(upper_x, upper_y, lower_x, lower_y, capacity);
        return 1;
    }
    Seq lhi, llo, rhi, rlo;
    int lsym, rsym;

    if (hi[0] < lo[0]) {
        if (s[0] > s[lo[0]]) {
            lhi = seq(hi, 1, small(nh, nbsym + 1));
            llo = seq(lo, 0, small(nl, nbsym));
            lsym = hi[0];
        } else {
            lhi = seq(hi, 0, small(nh, nbsym));
            llo = seq(lo, 0, small(nl, nbsym - 1), 0);
            lsym = 0;
        }
    } else {
        if (s[0] < s[hi[0]]) {
            lhi = seq(hi, 0, small(nh, nbsym));
            llo = seq(lo, 1, small(nl, nbsym + 1));
            lsym = lo[0];
        } else {
            lhi = seq(hi, 0, small(nh, nbsym - 1), 0);
            llo = seq(lo, 0, small(nl, nbsym));
            lsym = 0;
        }
    }
    if (hi[nh - 1] < lo[nl - 1]) {
        if (s[n - 1] < s[hi[nh - 1]]) {
            rhi = seq(hi, large(nh - nbsym, 0), nh);
            rlo = seq(lo, large(nl - nbsym - 1, 0), nl - 1);
            rsym = lo[nl - 1];
        } else {
            rhi = seq(hi, large(nh - nbsym + 1, 0), nh, n - 1, 1);
            rlo = seq(lo, large(nl - nbsym, 0), nl);
            rsym = n - 1;
        }
    } else {
        if (s[n - 1] > s[lo[nl - 1]]) {
            rhi = seq(hi, large(nh - nbsym - 1, 0), nh - 1);
            rlo = seq(lo, large(nl - nbsym, 0), nl);
            rsym = hi[nh - 1];
        } else {
            rhi = seq(hi, large(nh - nbsym, 0), nh);
            rlo = seq(lo, large(nl - nbsym + 1, 0), nl, n - 1, 1);
            rsym = n - 1;
        }
    }
    if (length(llo) == 0) llo = seq(lo, 0, nl, -1, 0, 0);
    if (length(rlo) == 0) rlo = seq(lo, 0, nl, -1, 0, 0);
    if (length(lhi) == 0) lhi = seq(hi, 0, nh, -1, 0, 0);
    if (length(rhi) == 0) rhi = seq(hi, 0, nh, -1, 0, 0);

    if (2 * lsym - at(llo, 0) > 0 || 2 * lsym - at(lhi, 0) > 0) {
        if (lsym == hi[0]) lhi = seq(hi, 0, small(nh, nbsym));
        else llo = seq(lo, 0, small(nl, nbsym));
        if (lsym == 0) {
            clear_knots(upper_x, upper_y, lower_x, lower_y, capacity);
            return 2;
        }
        lsym = 0;
    }
    if (2 * rsym - at(rlo, length(rlo) - 1) < n - 1 ||
        2 * rsym - at(rhi, length(rhi) - 1) < n - 1) {
        if (rsym == hi[nh - 1]) rhi = seq(hi, large(nh - nbsym, 0), nh);
        else rlo = seq(lo, large(nl - nbsym, 0), nl);
        if (rsym == n - 1) {
            clear_knots(upper_x, upper_y, lower_x, lower_y, capacity);
            return 3;
        }
        rsym = n - 1;
    }
    knot_counts[0] = write_knots(upper_x, upper_y, capacity,
                                 s, lhi, hi, nh, rhi, lsym, rsym);
    knot_counts[1] = write_knots(lower_x, lower_y, capacity,
                                 s, llo, lo, nl, rlo, lsym, rsym);
    return 0;
}
extern "C" __global__
void reflect_batch(const double* signals, const int* maxima, const int* minima,
                   const int* extrema_counts, double* upper_x, double* upper_y,
                   double* lower_x, double* lower_y, int* knot_counts, int* status,
                   int n, int batch, int nbsym) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batch) return;
    size_t signal_offset = (size_t)row * n;
    size_t knot_offset = (size_t)row * (2 * n);
    status[row] = reflect_one(
        signals + signal_offset, maxima + signal_offset, minima + signal_offset,
        extrema_counts + 3 * row, upper_x + knot_offset, upper_y + knot_offset,
        lower_x + knot_offset, lower_y + knot_offset, knot_counts + 2 * row,
        n, nbsym);
}
"""

_REFLECTION_KERNEL = cp.RawKernel(_REFLECTION_SOURCE, "reflect_batch", backend="nvrtc")


def reflection_into(
    signals,
    maxima,
    minima,
    extrema_counts,
    upper_x,
    upper_y,
    lower_x,
    lower_y,
    knot_counts,
    status,
    nbsym=2,
):
    """Write reflected knots and device status into caller-owned buffers.

    Parameters
    ----------
    signals : cupy.ndarray
        Contiguous float64 input with shape ``(batch, samples)``.
    maxima, minima : cupy.ndarray
        Int32 extrema indices shaped like ``signals``.
    extrema_counts : cupy.ndarray
        Int32 counts with shape ``(batch, 3)``.
    upper_x, upper_y, lower_x, lower_y : cupy.ndarray
        Float64 knot buffers with shape ``(batch, 2 * samples)``.
    knot_counts : cupy.ndarray
        Int32 knot counts with shape ``(batch, 2)``.
    status : cupy.ndarray
        Int32 reflection status with shape ``(batch,)``.
    nbsym : int, default 2
        Number of extrema reflected at each edge.
    """
    require_device_array(signals, "signals")
    if signals.ndim != 2 or signals.dtype != float64 or not signals.flags.c_contiguous:
        raise ValueError("signals must be a contiguous float64 (batch, samples) CuPy array.")
    batch, n = signals.shape
    nbsym = integer("nbsym", nbsym)
    if batch < 1 or batch > 2**31 - 1 or n < 1 or n > (2**31 - 1) // 2 or nbsym > 2**31 - 2:
        raise ValueError("Batch, sample length, or nbsym exceeds the CUDA int32 contract.")
    _check_index_products(batch, n)
    for name, array in (("maxima", maxima), ("minima", minima)):
        require_device_array(array, name)
        if array.shape != signals.shape or array.dtype != int32 or not array.flags.c_contiguous:
            raise ValueError(f"{name} must be a contiguous int32 array shaped like signals.")
    require_device_array(extrema_counts, "extrema_counts")
    if (
        extrema_counts.shape != (batch, 3)
        or extrema_counts.dtype != int32
        or not extrema_counts.flags.c_contiguous
    ):
        raise ValueError("extrema_counts must be contiguous (batch, 3) int32.")
    knots = (upper_x, upper_y, lower_x, lower_y)
    for name, array in zip(("upper_x", "upper_y", "lower_x", "lower_y"), knots, strict=True):
        require_device_array(array, name)
        if array.shape != (batch, 2 * n) or array.dtype != float64 or not array.flags.c_contiguous:
            raise ValueError(f"{name} must be contiguous (batch, 2*samples) float64.")
    for name, array, shape in (
        ("knot_counts", knot_counts, (batch, 2)),
        ("status", status, (batch,)),
    ):
        require_device_array(array, name)
        if array.shape != shape or array.dtype != int32 or not array.flags.c_contiguous:
            raise ValueError(f"{name} must be contiguous int32 with shape {shape}.")
    _REFLECTION_KERNEL(
        ((batch + 127) // 128,),
        (128,),
        (
            signals,
            maxima,
            minima,
            extrema_counts,
            *knots,
            knot_counts,
            status,
            int32(n),
            int32(batch),
            int32(nbsym),
        ),
    )


def batched_reflection(signals, maxima, minima, extrema_counts, nbsym=2):
    """Mirror extrema into fixed-shape device knot buffers for cubic envelopes.

    Parameters
    ----------
    signals : cupy.ndarray
        Contiguous float64 input with shape ``(batch, samples)``.
    maxima, minima : cupy.ndarray
        Int32 extrema indices shaped like ``signals``.
    extrema_counts : cupy.ndarray
        Int32 counts with shape ``(batch, 3)``.
    nbsym : int, default 2
        Number of extrema reflected at each edge.

    Returns
    -------
    tuple of cupy.ndarray
        Four knot buffers, knot counts and reflection status on the device.
    """
    require_device_array(signals, "signals")
    if signals.ndim != 2:
        raise ValueError("signals must have shape (batch, samples).")
    batch, n = signals.shape
    _check_index_products(batch, n)
    knots = [cp.empty((batch, 2 * n), dtype="float64") for _ in range(4)]
    knot_counts = cp.empty((batch, 2), dtype="int32")
    status = cp.empty(batch, dtype="int32")
    reflection_into(
        signals,
        maxima,
        minima,
        extrema_counts,
        knots[0],
        knots[1],
        knots[2],
        knots[3],
        knot_counts,
        status,
        nbsym=nbsym,
    )
    return *knots, knot_counts, status


_SPLINE_SOURCE = r"""
__device__ int interpolate(const double* x, const double* y, int m, int n,
                           double* output, double* upper, double* deriv) {
    if (m < 3) return 1;
    if (x[0] > 0.0 || x[m - 1] < n - 1) return 3;
    for (int i = 0; i < m; ++i) {
        if (!isfinite(x[i]) || !isfinite(y[i]) ||
            (i > 0 && x[i] <= x[i - 1])) return 2;
    }
    // Thomas elimination of the first-derivative system used by CubicSpline.
    // The three-knot branch preserves the reference EMD's distinct system.
    for (int i = 0; i < m; ++i) {
        double lower, diagonal, higher, rhs;
        if (m == 3) {
            double h0 = x[1] - x[0], h1 = x[2] - x[1];
            double v0 = (y[1] - y[0]) / (h0 * h0);
            double v1 = (y[2] - y[1]) / (h1 * h1);
            if (i == 0) {
                lower = 0.0; diagonal = 2.0 / h0;
                higher = 1.0 / h0; rhs = 3.0 * v0;
            } else if (i == 1) {
                lower = 1.0 / h0; diagonal = 2.0 * (1.0 / h0 + 1.0 / h1);
                higher = 1.0 / h1; rhs = 3.0 * (v0 + v1);
            } else {
                lower = 1.0 / h1; diagonal = 2.0 / h1;
                higher = 0.0; rhs = 3.0 * v1;
            }
        } else if (i == 0) {
            double h0 = x[1] - x[0], h1 = x[2] - x[1], span = h0 + h1;
            double v0 = (y[1] - y[0]) / h0, v1 = (y[2] - y[1]) / h1;
            lower = 0.0; diagonal = h1; higher = span;
            rhs = ((h0 + 2.0 * span) * h1 * v0 + h0 * h0 * v1) / span;
        } else if (i == m - 1) {
            double h0 = x[m - 2] - x[m - 3];
            double h1 = x[m - 1] - x[m - 2], span = h0 + h1;
            double v0 = (y[m - 2] - y[m - 3]) / h0;
            double v1 = (y[m - 1] - y[m - 2]) / h1;
            lower = span; diagonal = h0; higher = 0.0;
            rhs = (h1 * h1 * v0 + (2.0 * span + h1) * h0 * v1) / span;
        } else {
            double h0 = x[i] - x[i - 1], h1 = x[i + 1] - x[i];
            double v0 = (y[i] - y[i - 1]) / h0;
            double v1 = (y[i + 1] - y[i]) / h1;
            lower = h1; diagonal = 2.0 * (h0 + h1); higher = h0;
            rhs = 3.0 * (h1 * v0 + h0 * v1);
        }
        double pivot = i ? diagonal - lower * upper[i - 1] : diagonal;
        if (!isfinite(pivot) || pivot == 0.0) return 4;
        upper[i] = i + 1 < m ? higher / pivot : 0.0;
        deriv[i] = (rhs - (i ? lower * deriv[i - 1] : 0.0)) / pivot;
    }
    for (int i = m - 2; i >= 0; --i) deriv[i] -= upper[i] * deriv[i + 1];
    int segment = 0;
    for (int q = 0; q < n; ++q) {
        while (segment + 1 < m - 1 && x[segment + 1] <= q) ++segment;
        double h = x[segment + 1] - x[segment];
        double t = (q - x[segment]) / h;
        double delta = y[segment + 1] - y[segment];
        double a = deriv[segment] * h - delta;
        double b = delta - deriv[segment + 1] * h;
        output[q] = (1.0 - t) * y[segment] + t * y[segment + 1] +
                    t * (1.0 - t) * (a * (1.0 - t) + b * t);
        if (!isfinite(output[q])) return 4;
    }
    return 0;
}

extern "C" __global__
void spline_batch(const double* upper_x, const double* upper_y,
                  const double* lower_x, const double* lower_y,
                  const int* knot_counts, double* upper_out, double* lower_out,
                  double* work_upper, double* work_deriv, int* status,
                  int n, int batch) {
    int task = blockIdx.x * blockDim.x + threadIdx.x;
    if (task >= 2 * batch) return;
    int row = task / 2, kind = task % 2, capacity = 2 * n;
    size_t knot_offset = (size_t)row * capacity;
    size_t work_offset = (size_t)task * capacity;
    double* output = (kind ? lower_out : upper_out) + (size_t)row * n;
    const double* x = (kind ? lower_x : upper_x) + knot_offset;
    const double* y = (kind ? lower_y : upper_y) + knot_offset;
    int m = knot_counts[task];
    int result = m > capacity ? 1 : interpolate(x, y, m, n, output,
                                                work_upper + work_offset,
                                                work_deriv + work_offset);
    status[task] = result;
    if (result) for (int q = 0; q < n; ++q) output[q] = 0.0;
}
"""

_SPLINE_KERNEL = cp.RawKernel(_SPLINE_SOURCE, "spline_batch", backend="nvrtc")


def spline_into(
    upper_x,
    upper_y,
    lower_x,
    lower_y,
    knot_counts,
    upper_out,
    lower_out,
    work_upper,
    work_deriv,
    status,
):
    """Evaluate both cubic envelopes for each row into fixed device buffers.

    Status columns correspond to upper/lower: 0 success, 1 too few knots,
    2 invalid knots, 3 incomplete coverage, 4 singular or non-finite result.

    Parameters
    ----------
    upper_x, upper_y, lower_x, lower_y : cupy.ndarray
        Float64 knot buffers with shape ``(batch, 2 * samples)``.
    knot_counts : cupy.ndarray
        Int32 knot counts with shape ``(batch, 2)``.
    upper_out, lower_out : cupy.ndarray
        Float64 envelopes with shape ``(batch, samples)``.
    work_upper, work_deriv : cupy.ndarray
        Float64 spline workspace with shape ``(batch, 2, 2 * samples)``.
    status : cupy.ndarray
        Int32 spline status with shape ``(batch, 2)``.
    """
    require_device_array(upper_out, "upper_out")
    if upper_out.ndim != 2 or upper_out.dtype != float64 or not upper_out.flags.c_contiguous:
        raise ValueError("upper_out must be contiguous float64 (batch, samples).")
    batch, n = upper_out.shape
    if batch < 1 or batch > (2**31 - 1) // 2 or n < 1 or n > (2**31 - 1) // 2:
        raise ValueError("Batch or sample length exceeds the CUDA int32 contract.")
    _check_index_products(batch, n)
    for name, array, shape, dtype in (
        ("upper_x", upper_x, (batch, 2 * n), float64),
        ("upper_y", upper_y, (batch, 2 * n), float64),
        ("lower_x", lower_x, (batch, 2 * n), float64),
        ("lower_y", lower_y, (batch, 2 * n), float64),
        ("knot_counts", knot_counts, (batch, 2), int32),
        ("lower_out", lower_out, (batch, n), float64),
        ("work_upper", work_upper, (batch, 2, 2 * n), float64),
        ("work_deriv", work_deriv, (batch, 2, 2 * n), float64),
        ("status", status, (batch, 2), int32),
    ):
        require_device_array(array, name)
        if array.shape != shape or array.dtype != dtype or not array.flags.c_contiguous:
            raise ValueError(f"{name} must be contiguous {dtype} with shape {shape}.")
    _SPLINE_KERNEL(
        ((2 * batch + 127) // 128,),
        (128,),
        (
            upper_x,
            upper_y,
            lower_x,
            lower_y,
            knot_counts,
            upper_out,
            lower_out,
            work_upper,
            work_deriv,
            status,
            int32(n),
            int32(batch),
        ),
    )


def batched_spline(upper_x, upper_y, lower_x, lower_y, knot_counts, samples):
    """Allocate and evaluate both cubic envelopes, returning device status.

    Parameters
    ----------
    upper_x, upper_y, lower_x, lower_y : cupy.ndarray
        Float64 knot buffers with shape ``(batch, 2 * samples)``.
    knot_counts : cupy.ndarray
        Int32 knot counts with shape ``(batch, 2)``.
    samples : int
        Number of output samples per row.

    Returns
    -------
    tuple of cupy.ndarray
        Upper envelope, lower envelope and spline status on the device.
    """
    samples = integer("samples", samples)
    batch = len(knot_counts)
    _check_index_products(batch, samples)
    outputs = [cp.empty((batch, samples), dtype="float64") for _ in range(2)]
    work = [cp.empty((batch, 2, 2 * samples), dtype="float64") for _ in range(2)]
    status = cp.empty((batch, 2), dtype="int32")
    spline_into(
        upper_x,
        upper_y,
        lower_x,
        lower_y,
        knot_counts,
        outputs[0],
        outputs[1],
        work[0],
        work[1],
        status,
    )
    return *outputs, status


_CRITERION_SOURCE = r"""
__device__ int rilling_one(const double* upper, const double* lower,
                           const int* extrema_counts, const int* spline_status,
                           double* means, double* metrics, int n,
                           double threshold1, double threshold2, double allowed) {
    int ne = extrema_counts[0] + extrema_counts[1];
    int nz = extrema_counts[2];
    int above = 0;
    double max_ratio = 0.0;
    for (int q = 0; q < n; ++q) {
        double u = upper[q], l = lower[q];
        double mean = 0.5 * u + 0.5 * l;
        means[q] = mean;
        double amplitude = fabs(0.5 * u - 0.5 * l);
        double ratio = amplitude > 0.0 ? fabs(mean) / amplitude :
                       (mean == 0.0 ? 0.0 : __longlong_as_double(0x7ff0000000000000LL));
        if (ratio > threshold1) ++above;
        if (ratio > max_ratio) max_ratio = ratio;
    }
    double fraction = (double)above / n;
    metrics[0] = max_ratio;
    metrics[1] = fraction;
    return ne < 3 ? 2 :
                    (spline_status[0] || spline_status[1]) ? 3 :
                    (abs(ne - nz) <= 1 && fraction <= allowed &&
                     max_ratio <= threshold2) ? 1 : 0;
}
extern "C" __global__
void rilling_batch(const double* upper, const double* lower,
                   const int* extrema_counts, const int* spline_status,
                   double* means, double* metrics, int* decision,
                   int n, int batch, double threshold1, double threshold2,
                   double allowed) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batch) return;
    size_t offset = (size_t)row * n;
    decision[row] = rilling_one(
        upper + offset, lower + offset, extrema_counts + 3 * row,
        spline_status + 2 * row, means + offset, metrics + 2 * row,
        n, threshold1, threshold2, allowed);
}
"""

_CRITERION_KERNEL = cp.RawKernel(_CRITERION_SOURCE, "rilling_batch", backend="nvrtc")

_FUSED_SOURCE = (
    _EXTREMA_SOURCE
    + _REFLECTION_SOURCE
    + _SPLINE_SOURCE
    + _CRITERION_SOURCE
    + r"""
__device__ int sift_one(double* h, int* hi, int* lo, int* zz, int* counts,
                        double* ux, double* uy, double* lx, double* ly,
                        int* knots, double* upper, double* lower,
                        double* work_upper, double* work_deriv, int* spline,
                        double* mean, double* metrics, int* iterations,
                        int n, int nbsym, int limit,
                        double threshold1, double threshold2, double allowed) {
    for (;;) {
        detect_extrema(h, hi, lo, zz, counts, n);
        if (counts[0] + counts[1] < 3) return 2;
        if (reflect_one(h, hi, lo, counts, ux, uy, lx, ly, knots, n, nbsym)) {
            return 3;
        }
        spline[0] = interpolate(ux, uy, knots[0], n, upper,
                                work_upper, work_deriv);
        spline[1] = interpolate(lx, ly, knots[1], n, lower,
                                work_upper + 2 * n, work_deriv + 2 * n);
        if (spline[0] || spline[1]) return 3;
        int verdict = rilling_one(upper, lower, counts, spline, mean, metrics, n,
                                  threshold1, threshold2, allowed);
        if (verdict) return verdict;
        if (*iterations == limit) return 5;
        bool progressed = false;
        for (int q = 0; q < n; ++q) {
            double before = h[q], after = before - mean[q];
            if (!isfinite(after)) return 6;
            progressed |= after != before;
            h[q] = after;
        }
        if (!progressed) return 4;
        ++*iterations;
    }
}
extern "C" __global__
void first_imf_batch(double* signals, int* maxima, int* minima, int* zeros,
                     int* extrema_counts, double* upper_x, double* upper_y,
                     double* lower_x, double* lower_y, int* knot_counts,
                     double* upper_out, double* lower_out, double* work_upper,
                     double* work_deriv, int* spline_status, double* means,
                     double* metrics, int* state, int* iterations,
                     int n, int batch, int nbsym, int limit,
                     double threshold1, double threshold2, double allowed) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batch) return;
    size_t sample = (size_t)row * n;
    size_t knot = (size_t)row * (2 * n);
    size_t work = (size_t)row * (4 * n);
    state[row] = sift_one(
        signals + sample, maxima + sample, minima + sample, zeros + sample,
        extrema_counts + 3 * row, upper_x + knot, upper_y + knot,
        lower_x + knot, lower_y + knot, knot_counts + 2 * row,
        upper_out + sample, lower_out + sample, work_upper + work,
        work_deriv + work, spline_status + 2 * row, means + sample,
        metrics + 2 * row, iterations + row, n, nbsym, limit,
        threshold1, threshold2, allowed);
}

extern "C" __global__
void full_emd_batch(double* residue, const double* peaks, double* modes,
                    int* available, int* sift_counts, double* candidates,
                    int* maxima, int* minima, int* zeros, int* extrema_counts,
                    double* upper_x, double* upper_y, double* lower_x,
                    double* lower_y, int* knot_counts, double* upper_out,
                    double* lower_out, double* work_upper, double* work_deriv,
                    int* spline_status, double* means, double* metrics,
                    int* state, const int* skip, int skip_enabled,
                    int n, int batch, int cap, int nbsym, int limit,
                    double floor, double threshold1, double threshold2,
                    double allowed) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batch) return;
    if (skip_enabled && skip[0]) { state[row] = -1; return; }
    size_t sample = (size_t)row * n;
    size_t knot = (size_t)row * (2 * n);
    size_t work = (size_t)row * (4 * n);
    double* r = residue + sample;
    double* h = candidates + sample;
    for (int mode = 0; mode < cap; ++mode) {
        double maximum = 0.0;
        for (int q = 0; q < n; ++q) {
            double value = fabs(r[q]);
            if (value > maximum) maximum = value;
        }
        if (maximum <= floor) break;
        for (int q = 0; q < n; ++q) h[q] = r[q];
        int* count = sift_counts + (size_t)mode * batch + row;
        int result = sift_one(
            h, maxima + sample, minima + sample, zeros + sample,
            extrema_counts + 3 * row, upper_x + knot, upper_y + knot,
            lower_x + knot, lower_y + knot, knot_counts + 2 * row,
            upper_out + sample, lower_out + sample, work_upper + work,
            work_deriv + work, spline_status + 2 * row, means + sample,
            metrics + 2 * row, count, n, nbsym, limit,
            threshold1, threshold2, allowed);
        if (result == 2) break;
        if (result != 1) { state[row] = result; return; }
        double* output = modes + ((size_t)mode * batch + row) * n;
        bool progressed = false;
        for (int q = 0; q < n; ++q) {
            double before = r[q];
            double after = before - h[q];
            progressed |= after != before;
            r[q] = after;
            output[q] = h[q] * peaks[row];
        }
        if (!progressed) { state[row] = 4; return; }
        available[(size_t)mode * batch + row] = 1;
    }
    state[row] = 0;
}
"""
)

_FUSED_KERNEL = cp.RawKernel(_FUSED_SOURCE, "first_imf_batch", backend="nvrtc")
_FULL_EMD_KERNEL = cp.RawKernel(_FUSED_SOURCE, "full_emd_batch", backend="nvrtc")


def criterion_into(
    upper,
    lower,
    extrema_counts,
    spline_status,
    means,
    metrics,
    decision,
    thresholds=(0.05, 0.5, 0.05),
):
    """Write Rilling mean, max ratio, fraction and decision on device.

    Decision: 0 continue, 1 IMF accepted, 2 terminal, 3 bad spline.

    Parameters
    ----------
    upper, lower : cupy.ndarray
        Float64 envelopes with shape ``(batch, samples)``.
    extrema_counts : cupy.ndarray
        Int32 counts with shape ``(batch, 3)``.
    spline_status : cupy.ndarray
        Int32 spline status with shape ``(batch, 2)``.
    means : cupy.ndarray
        Caller-owned float64 local means shaped like ``upper``.
    metrics : cupy.ndarray
        Caller-owned float64 max ratio and fraction with shape ``(batch, 2)``.
    decision : cupy.ndarray
        Caller-owned int32 decisions with shape ``(batch,)``.
    thresholds : tuple of float, default (0.05, 0.5, 0.05)
        Lower ratio, upper ratio and allowed fraction.
    """
    require_device_array(upper, "upper")
    if upper.ndim != 2 or upper.dtype != float64 or not upper.flags.c_contiguous:
        raise ValueError("upper must be contiguous float64 (batch, samples).")
    batch, n = upper.shape
    if batch < 1 or batch > 2**31 - 1 or n < 1 or n > 2**31 - 1:
        raise ValueError("Batch or sample length exceeds the CUDA int32 contract.")
    _check_index_products(batch, n)
    for name, array, shape, dtype in (
        ("lower", lower, (batch, n), float64),
        ("extrema_counts", extrema_counts, (batch, 3), int32),
        ("spline_status", spline_status, (batch, 2), int32),
        ("means", means, (batch, n), float64),
        ("metrics", metrics, (batch, 2), float64),
        ("decision", decision, (batch,), int32),
    ):
        require_device_array(array, name)
        if array.shape != shape or array.dtype != dtype or not array.flags.c_contiguous:
            raise ValueError(f"{name} must be contiguous {dtype} with shape {shape}.")
    if len(thresholds) != 3:
        raise ValueError("thresholds must contain lower, upper, and allowed fraction.")
    t1, t2, allowed = (nonnegative("threshold", value) for value in thresholds)
    if t1 <= 0 or t2 < t1 or allowed >= 1:
        raise ValueError("Require 0 < lower <= upper and 0 <= allowed < 1.")
    _CRITERION_KERNEL(
        ((batch + 127) // 128,),
        (128,),
        (
            upper,
            lower,
            extrema_counts,
            spline_status,
            means,
            metrics,
            decision,
            int32(n),
            int32(batch),
            float64(t1),
            float64(t2),
            float64(allowed),
        ),
    )


def batched_criterion(upper, lower, extrema_counts, spline_status, thresholds=(0.05, 0.5, 0.05)):
    """Allocate fixed device outputs for the Rilling criterion.

    Parameters
    ----------
    upper, lower : cupy.ndarray
        Float64 envelopes with shape ``(batch, samples)``.
    extrema_counts : cupy.ndarray
        Int32 counts with shape ``(batch, 3)``.
    spline_status : cupy.ndarray
        Int32 spline status with shape ``(batch, 2)``.
    thresholds : tuple of float, default (0.05, 0.5, 0.05)
        Lower ratio, upper ratio and allowed fraction.

    Returns
    -------
    tuple of cupy.ndarray
        Local means, ratio metrics and decisions on the device.
    """
    require_device_array(upper, "upper")
    if upper.ndim != 2:
        raise ValueError("upper must have shape (batch, samples).")
    batch, n = upper.shape
    _check_index_products(batch, n)
    means = cp.empty((batch, n), dtype="float64")
    metrics = cp.empty((batch, 2), dtype="float64")
    decision = cp.empty(batch, dtype="int32")
    criterion_into(
        upper, lower, extrema_counts, spline_status, means, metrics, decision, thresholds
    )
    return means, metrics, decision


def batched_first_imf(signals, *, nbsym=2, max_iteration=2000, thresholds=(0.05, 0.5, 0.05)):
    """Extract one IMF per row using device geometry and Rilling reductions.

    The entire sifting loop runs inside one CUDA kernel per batch. The host
    reads final status once for explicit error reporting.

    Parameters
    ----------
    signals : cupy.ndarray
        Contiguous float64 input with shape ``(batch, samples)``.
    nbsym : int, default 2
        Number of extrema reflected at each edge.
    max_iteration : int, default 2000
        Maximum siftings per candidate IMF.
    thresholds : tuple of float, default (0.05, 0.5, 0.05)
        Rilling lower ratio, upper ratio and allowed fraction.

    Returns
    -------
    tuple of cupy.ndarray
        Accepted first IMFs, sift counts and per-row device status.
    """
    require_device_array(signals, "signals")
    if signals.ndim != 2 or signals.dtype != float64 or not signals.flags.c_contiguous:
        raise ValueError("signals must be contiguous float64 (batch, samples).")
    batch, n = signals.shape
    nbsym = integer("nbsym", nbsym)
    max_iteration = integer("max_iteration", max_iteration)
    if batch < 1 or batch > (2**31 - 1) // 2 or n < 1 or n > (2**31 - 1) // 2:
        raise ValueError("Batch or sample length exceeds the CUDA int32 contract.")
    _check_index_products(batch, n)
    if max_iteration > 2**31 - 1:
        raise ValueError("max_iteration exceeds the CUDA int32 contract.")
    if nbsym > 2**31 - 2:
        raise ValueError("nbsym exceeds the CUDA int32 contract.")
    if len(thresholds) != 3:
        raise ValueError("thresholds must contain lower, upper, and allowed fraction.")
    t1, t2, allowed = (nonnegative("threshold", value) for value in thresholds)
    if t1 <= 0 or t2 < t1 or allowed >= 1:
        raise ValueError("Require 0 < lower <= upper and 0 <= allowed < 1.")
    h = signals.copy()
    indices = [cp.empty((batch, n), dtype="int32") for _ in range(3)]
    extrema_counts = cp.empty((batch, 3), dtype="int32")
    knots = [cp.empty((batch, 2 * n), dtype="float64") for _ in range(4)]
    knot_counts = cp.empty((batch, 2), dtype="int32")
    envelopes = [cp.empty((batch, n), dtype="float64") for _ in range(2)]
    spline_work = [cp.empty((batch, 2, 2 * n), dtype="float64") for _ in range(2)]
    spline_status = cp.empty((batch, 2), dtype="int32")
    means = cp.empty((batch, n), dtype="float64")
    metrics = cp.empty((batch, 2), dtype="float64")
    state = cp.zeros(batch, dtype="int32")
    iterations = cp.zeros(batch, dtype="int32")
    _FUSED_KERNEL(
        ((batch + 127) // 128,),
        (128,),
        (
            h,
            *indices,
            extrema_counts,
            *knots,
            knot_counts,
            *envelopes,
            *spline_work,
            spline_status,
            means,
            metrics,
            state,
            iterations,
            int32(n),
            int32(batch),
            int32(nbsym),
            int32(max_iteration),
            float64(t1),
            float64(t2),
            float64(allowed),
        ),
    )
    host_state = cp.asnumpy(state)
    if (host_state >= 3).any():
        raise SiftingConvergenceError(
            f"Batched sifting failed for rows {list((host_state >= 3).nonzero()[0])}; "
            "states: 3=bad spline, 4=stagnation, 5=iteration limit, 6=non-finite."
        )
    return cp.where((state == 1)[:, None], h, 0.0), iterations, state


def batched_emd(
    signals,
    *,
    max_modes,
    nbsym=2,
    max_iteration=2000,
    thresholds=(0.05, 0.5, 0.05),
    _device_result=False,
    _skip=None,
):
    """Decompose equal-length rows; private device result retains unchecked status.

    Parameters
    ----------
    signals : cupy.ndarray
        Contiguous float64 input with shape ``(batch, samples)``.
    max_modes : int
        Maximum modes to extract per row; zero returns the inputs as residues.
    nbsym : int, default 2
        Number of extrema reflected at each edge.
    max_iteration : int, default 2000
        Maximum siftings per candidate IMF.
    thresholds : tuple of float, default (0.05, 0.5, 0.05)
        Rilling lower ratio, upper ratio and allowed fraction.
    _device_result : bool, default False
        Retain fixed-shape modes and unchecked status on device for internal callers.
    _skip : cupy.ndarray, optional
        Device int32 scalar that skips the kernel when nonzero.

    Returns
    -------
    tuple
        Public form: mode and availability lists, residues and sift-count list.
        Device form: fixed-shape modes, availability, residues, counts and status.
    """
    require_device_array(signals, "signals")
    if signals.ndim != 2 or signals.dtype != float64 or not signals.flags.c_contiguous:
        raise ValueError("signals must be contiguous float64 (batch, samples).")
    batch, n = signals.shape
    if batch < 1 or batch > (2**31 - 1) // 2 or n < 1 or n > (2**31 - 1) // 2:
        raise ValueError("Batch or sample length exceeds the CUDA int32 contract.")
    _check_index_products(batch, n)
    max_modes = integer("max_modes", max_modes, 0)
    nbsym = integer("nbsym", nbsym)
    max_iteration = integer("max_iteration", max_iteration)
    if max_modes > 2**31 - 1 or nbsym > 2**31 - 2 or max_iteration > 2**31 - 1:
        raise ValueError("A mode, mirror, or iteration limit exceeds the CUDA int32 contract.")
    if len(thresholds) != 3:
        raise ValueError("thresholds must contain lower, upper, and allowed fraction.")
    t1, t2, allowed = (nonnegative("threshold", value) for value in thresholds)
    if t1 <= 0 or t2 < t1 or allowed >= 1:
        raise ValueError("Require 0 < lower <= upper and 0 <= allowed < 1.")
    if _skip is not None:
        require_device_array(_skip, "skip")
        if _skip.dtype != int32 or _skip.shape not in ((), (1,)):
            raise ValueError("skip must be a device int32 scalar.")
    if max_modes == 0:
        if _device_result:
            return (
                cp.empty((0, batch, n), dtype="float64"),
                cp.empty((0, batch), dtype="bool"),
                signals.copy(),
                cp.empty((0, batch), dtype="int32"),
                cp.zeros(batch, dtype="int32"),
            )
        return [], [], signals.copy(), []
    # Modes and sifting buffers coexist; this excludes inputs and small metadata.
    minimum_bytes = (8 * (max_modes + 21) + 3 * 4) * batch * n
    device_bytes = cp.cuda.Device().mem_info[1]
    if minimum_bytes > device_bytes:
        raise MemoryError(
            f"Batched EMD workspace requires at least {minimum_bytes / 2**30:.2f} GiB; "
            f"device capacity is {device_bytes / 2**30:.2f} GiB."
        )
    peak = cp.max(cp.absolute(signals), axis=1)
    divisor = cp.where(peak > 0, peak, 1.0)
    residue = signals / divisor[:, None]
    modes = cp.zeros((max_modes, batch, n), dtype="float64")
    available = cp.zeros((max_modes, batch), dtype="int32")
    sift_counts = cp.zeros((max_modes, batch), dtype="int32")
    candidate = cp.empty((batch, n), dtype="float64")
    indices = [cp.empty((batch, n), dtype="int32") for _ in range(3)]
    extrema_counts = cp.empty((batch, 3), dtype="int32")
    knots = [cp.empty((batch, 2 * n), dtype="float64") for _ in range(4)]
    knot_counts = cp.empty((batch, 2), dtype="int32")
    envelopes = [cp.empty((batch, n), dtype="float64") for _ in range(2)]
    spline_work = [cp.empty((batch, 2, 2 * n), dtype="float64") for _ in range(2)]
    spline_status = cp.empty((batch, 2), dtype="int32")
    means = cp.empty((batch, n), dtype="float64")
    metrics = cp.empty((batch, 2), dtype="float64")
    state = cp.zeros(batch, dtype="int32")
    _FULL_EMD_KERNEL(
        ((batch + 127) // 128,),
        (128,),
        (
            residue,
            peak,
            modes,
            available,
            sift_counts,
            candidate,
            *indices,
            extrema_counts,
            *knots,
            knot_counts,
            *envelopes,
            *spline_work,
            spline_status,
            means,
            metrics,
            state,
            state if _skip is None else _skip,
            int32(_skip is not None),
            int32(n),
            int32(batch),
            int32(max_modes),
            int32(nbsym),
            int32(max_iteration),
            float64(64 * EPS),
            float64(t1),
            float64(t2),
            float64(allowed),
        ),
    )
    if _device_result:
        return modes, available != 0, residue * peak[:, None], sift_counts, state
    host_state = cp.asnumpy(state)
    if (host_state >= 3).any():
        raise SiftingConvergenceError(
            f"Batched EMD failed for rows {list((host_state >= 3).nonzero()[0])}; "
            "states: 3=bad spline, 4=stagnation, 5=iteration limit, 6=non-finite."
        )
    used = sum(cp.asnumpy(cp.any(available != 0, axis=1)).tolist())
    return (
        [modes[k] for k in range(used)],
        [available[k] != 0 for k in range(used)],
        residue * peak[:, None],
        [sift_counts[k] for k in range(used)],
    )


_KAHAN_SOURCE = r"""
extern "C" __global__
void ordered_mean(const double* values, double* result, int n, int batch) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= n) return;
    double total = 0.0, correction = 0.0;
    for (int row = 0; row < batch; ++row) {
        double term = values[(size_t)row * n + q] - correction;
        double summed = total + term;
        correction = (summed - total) - term;
        total = summed;
    }
    result[q] = total / batch;
}
"""

_KAHAN_KERNEL = cp.RawKernel(_KAHAN_SOURCE, "ordered_mean", backend="nvrtc")


def ordered_mean(values):
    """Average rows with the reference's Kahan summation order on CUDA.

    Parameters
    ----------
    values : cupy.ndarray
        Contiguous float64 input with shape ``(batch, samples)``.

    Returns
    -------
    cupy.ndarray
        Mean with shape ``(samples,)`` on the device.
    """
    require_device_array(values, "values")
    if values.ndim != 2 or values.dtype != float64 or not values.flags.c_contiguous:
        raise ValueError("values must be contiguous float64 (batch, samples).")
    batch, n = values.shape
    if batch < 1 or batch > 2**31 - 1 or n < 1 or n > 2**31 - 1:
        raise ValueError("Batch or sample length exceeds the CUDA int32 contract.")
    _check_index_products(batch, n)
    result = cp.empty(n, dtype="float64")
    _KAHAN_KERNEL(((n + 127) // 128,), (128,), (values, result, int32(n), int32(batch)))
    return result
