"""Stage 2: CuPy mirror geometry and interpolation.

Modified from ICEEMDAN(2).py lines 47–163 on 2026-09-25.
PyEMD-derived portions: Copyright 2017 Dawid Laszuk; Apache-2.0.
Original wrapper: Javier F. Santamaria, 2025; rights retained.
All supported interpolators execute via cupyx, never scipy CPU.
"""

from __future__ import annotations

from cupyx import errstate
from cupyx.scipy.interpolate import (
    Akima1DInterpolator,
    CubicHermiteSpline,
    CubicSpline,
    PchipInterpolator,
    interp1d,
)

from .runtime import cp, on_device, require_device_array


def append_scalar(a, value):
    return cp.concatenate((a, cp.full(1, value, dtype=a.dtype)))


def keep_last_adjacent(extrema):
    keep = cp.concatenate((extrema[0, 1:] != extrema[0, :-1], cp.ones(1, dtype="bool")))
    return extrema[:, keep]


def cubic_spline_3pts(x, y, T):
    # Exact first-derivative system from the reference, not default not-a-knot.
    x0, x1, x2 = x
    y0, y1, y2 = y
    x1x0, x2x1 = x1 - x0, x2 - x1
    y1y0, y2y1 = y1 - y0, y2 - y1
    _x1x0, _x2x1 = 1.0 / x1x0, 1.0 / x2x1
    m11, m12 = 2 * _x1x0, _x1x0
    m21, m22, m23 = _x1x0, 2.0 * (_x1x0 + _x2x1), _x2x1
    m32, m33 = _x2x1, 2.0 * _x2x1
    v1 = 3 * y1y0 * _x1x0 * _x1x0
    v3 = 3 * y2y1 * _x2x1 * _x2x1
    v2 = v1 + v3
    z = cp.zeros_like(x0)
    M = cp.stack((cp.stack((m11, m12, z)), cp.stack((m21, m22, m23)), cp.stack((z, m32, m33))))
    v = cp.stack((v1, v2, v3))
    with errstate(linalg="raise"):
        k = cp.linalg.solve(M, v)
    a1 = k[0] * x1x0 - y1y0
    b1 = -k[1] * x1x0 + y1y0
    a2 = k[1] * x2x1 - y2y1
    b2 = -k[2] * x2x1 + y2y1
    t = T[(T >= x0) & (T <= x2)]
    t1 = (T[(T >= x0) & (T < x1)] - x0) / x1x0
    t2 = (T[(T >= x1) & (T <= x2)] - x1) / x2x1
    t11, t22 = 1.0 - t1, 1.0 - t2
    q1 = t11 * y0 + t1 * y1 + t1 * t11 * (a1 * t11 + b1 * t1)
    q2 = t22 * y1 + t2 * y2 + t2 * t22 * (a2 * t22 + b2 * t2)
    return t, cp.concatenate((q1, q2))


class Geometry:
    nbsym: int
    spline_kind: str
    DTYPE: str

    @on_device
    def prepare_points_simple(self, T, S, max_pos, max_val, min_pos, min_val):
        for name, value in (("T", T), ("S", S), ("max_pos", max_pos), ("min_pos", min_pos)):
            require_device_array(value, name)
        ind_min = min_pos.astype("int64")
        ind_max = max_pos.astype("int64")
        nbsym = self.nbsym
        end_min, end_max = len(min_pos), len(max_pos)
        if ind_max[0] < ind_min[0]:
            if S[0] > S[ind_min[0]]:
                lmax = ind_max[1 : min(end_max, nbsym + 1)][::-1]
                lmin = ind_min[0 : min(end_min, nbsym + 0)][::-1]
                lsym = ind_max[0]
            else:
                lmax = ind_max[0 : min(end_max, nbsym)][::-1]
                lmin = append_scalar(ind_min[0 : min(end_min, nbsym - 1)][::-1], 0)
                lsym = 0
        else:
            if S[0] < S[ind_max[0]]:
                lmax = ind_max[0 : min(end_max, nbsym + 0)][::-1]
                lmin = ind_min[1 : min(end_min, nbsym + 1)][::-1]
                lsym = ind_min[0]
            else:
                lmax = append_scalar(ind_max[0 : min(end_max, nbsym - 1)][::-1], 0)
                lmin = ind_min[0 : min(end_min, nbsym)][::-1]
                lsym = 0
        if ind_max[-1] < ind_min[-1]:
            if S[-1] < S[ind_max[-1]]:
                rmax = ind_max[max(end_max - nbsym, 0) :][::-1]
                rmin = ind_min[max(end_min - nbsym - 1, 0) : -1][::-1]
                rsym = ind_min[-1]
            else:
                rmax = append_scalar(ind_max[max(end_max - nbsym + 1, 0) :], len(S) - 1)[::-1]
                rmin = ind_min[max(end_min - nbsym, 0) :][::-1]
                rsym = len(S) - 1
        else:
            if S[-1] > S[ind_min[-1]]:
                rmax = ind_max[max(end_max - nbsym - 1, 0) : -1][::-1]
                rmin = ind_min[max(end_min - nbsym, 0) :][::-1]
                rsym = ind_max[-1]
            else:
                rmax = ind_max[max(end_max - nbsym, 0) :][::-1]
                rmin = append_scalar(ind_min[max(end_min - nbsym + 1, 0) :], len(S) - 1)[::-1]
                rsym = len(S) - 1
        if not lmin.size:
            lmin = ind_min
        if not rmin.size:
            rmin = ind_min
        if not lmax.size:
            lmax = ind_max
        if not rmax.size:
            rmax = ind_max
        tlmin = 2 * T[lsym] - T[lmin]
        tlmax = 2 * T[lsym] - T[lmax]
        trmin = 2 * T[rsym] - T[rmin]
        trmax = 2 * T[rsym] - T[rmax]
        if tlmin[0] > T[0] or tlmax[0] > T[0]:
            if lsym == ind_max[0]:
                lmax = ind_max[0 : min(end_max, nbsym)][::-1]
            else:
                lmin = ind_min[0 : min(end_min, nbsym)][::-1]
            if lsym == 0:
                raise Exception("Left edge BUG")
            lsym = 0
            tlmin = 2 * T[lsym] - T[lmin]
            tlmax = 2 * T[lsym] - T[lmax]
        if trmin[-1] < T[-1] or trmax[-1] < T[-1]:
            if rsym == ind_max[-1]:
                rmax = ind_max[max(end_max - nbsym, 0) :][::-1]
            else:
                rmin = ind_min[max(end_min - nbsym, 0) :][::-1]
            if rsym == len(S) - 1:
                raise Exception("Right edge BUG")
            rsym = len(S) - 1
            trmin = 2 * T[rsym] - T[rmin]
            trmax = 2 * T[rsym] - T[rmax]
        zlmax, zlmin, zrmax, zrmin = S[lmax], S[lmin], S[rmax], S[rmin]
        tmin = cp.concatenate((tlmin, T[ind_min], trmin))
        tmax = cp.concatenate((tlmax, T[ind_max], trmax))
        zmin = cp.concatenate((zlmin, S[ind_min], zrmin))
        zmax = cp.concatenate((zlmax, S[ind_max], zrmax))
        max_extrema = cp.stack((tmax, zmax))
        min_extrema = cp.stack((tmin, zmin))
        max_extrema = keep_last_adjacent(max_extrema)
        min_extrema = keep_last_adjacent(min_extrema)
        return max_extrema, min_extrema

    @on_device
    def spline_points(self, T, extrema):
        require_device_array(T, "T")
        require_device_array(extrema, "extrema")
        kind = self.spline_kind.lower()
        t = T[(T >= extrema[0, 0]) & (T <= extrema[0, -1])]
        # cupyx linalg checking is scoped; unsupported numpy over/invalid options
        # are intentionally NOT copied to cupyx.errstate.
        with errstate(linalg="raise"):
            if kind == "akima":
                return t, Akima1DInterpolator(extrema[0], extrema[1])(t)
            if kind == "cubic":
                if extrema.shape[1] > 3:
                    return t, CubicSpline(extrema[0], extrema[1])(t)
                return cubic_spline_3pts(extrema[0], extrema[1], t)
            if kind == "pchip":
                return t, PchipInterpolator(extrema[0], extrema[1])(t)
            if kind == "cubic_hermite":
                slopes = cp.gradient(extrema[1], extrema[0])
                return t, CubicHermiteSpline(extrema[0], extrema[1], slopes)(t)
            if kind in ("linear", "slinear", "quadratic"):
                return T, interp1d(extrema[0], extrema[1], kind=kind)(t).astype(self.DTYPE)
        raise ValueError("No such interpolation method!")
