#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ICEEMDAN, Colominas et al. (2014), with explicit numerical contracts.

Original wrapper: Javier F. Santamaria, copyright 2025; rights retained.
Corrections prepared 2026-09-25. No relicensing of the original is implied.
PyEMD-derived geometry: Dawid Laszuk, Apache-2.0, identified below.
The complete Apache-2.0 terms are included at the end of this file.

Runtime dependencies: NumPy and SciPy ONLY. PyEMD is not required.
Default EMD: discrete extrema, mirrored cubic envelopes, Rilling stopping.
Output: rows are ICEEMDAN components; the LAST ROW is always the residue.
An ICEEMDAN component is not automatically a structural mode or an exact IMF.

Minimal use:
    from ICEEMDAN import ICEEMDAN
    model = ICEEMDAN(trials=100, epsilon=0.2, seed=42)
    components = model(signal, T=time)
    imfs, residue = model.get_imfs_and_residue()

With parallel=True, invoke from an importable script guarded by
``if __name__ == "__main__":`` (spawn is used on every platform).
"""
from __future__ import annotations
import copy
import hashlib
import logging
import multiprocessing as mp
import numbers
import pickle
import warnings
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from typing import Any, Optional
import numpy as np

__version__ = "2.0.0"
__all__ = ["ICEEMDAN", "CEEMDAN", "EMD", "SiftingConvergenceError",
           "DecompositionLimitError", "BackendContractError"]


# BEGIN PyEMD-derived geometry (Copyright 2017 Dawid Laszuk; Apache-2.0).
# PyEMD v1.10.0 EMD.py/splines.py; only mirror and interpolation functions.
# Source metadata and changes are documented in THIRD_PARTY_NOTICES.md.
from scipy.interpolate import (interp1d, CubicSpline, Akima1DInterpolator,
                               PchipInterpolator, CubicHermiteSpline)
def cubic_spline_3pts(x, y, T):
    x0, x1, x2 = x
    y0, y1, y2 = y
    x1x0, x2x1 = x1-x0, x2-x1
    y1y0, y2y1 = y1-y0, y2-y1
    _x1x0, _x2x1 = 1.0/x1x0, 1.0/x2x1
    m11, m12, m13 = 2*_x1x0, _x1x0, 0
    m21, m22, m23 = _x1x0, 2.0*(_x1x0+_x2x1), _x2x1
    m31, m32, m33 = 0, _x2x1, 2.0*_x2x1
    v1 = 3*y1y0*_x1x0*_x1x0
    v3 = 3*y2y1*_x2x1*_x2x1
    v2 = v1+v3
    M = np.array([[m11,m12,m13],[m21,m22,m23],[m31,m32,m33]])
    v = np.array([v1,v2,v3]).T
    k = np.linalg.solve(M,v)
    a1 = k[0]*x1x0-y1y0
    b1 = -k[1]*x1x0+y1y0
    a2 = k[1]*x2x1-y2y1
    b2 = -k[2]*x2x1+y2y1
    t = T[np.r_[T>=x0] & np.r_[T<=x2]]
    t1 = (T[np.r_[T>=x0] & np.r_[T<x1]]-x0)/x1x0
    t2 = (T[np.r_[T>=x1] & np.r_[T<=x2]]-x1)/x2x1
    t11,t22 = 1.0-t1,1.0-t2
    q1 = t11*y0+t1*y1+t1*t11*(a1*t11+b1*t1)
    q2 = t22*y1+t2*y2+t2*t22*(a2*t22+b2*t2)
    return t, np.append(q1,q2)

class _Geometry:
    def prepare_points_simple(self,T,S,max_pos,max_val,min_pos,min_val):
        ind_min=min_pos.astype(int)
        ind_max=max_pos.astype(int)
        nbsym=self.nbsym
        end_min,end_max=len(min_pos),len(max_pos)
        if ind_max[0]<ind_min[0]:
            if S[0]>S[ind_min[0]]:
                lmax=ind_max[1:min(end_max,nbsym+1)][::-1]
                lmin=ind_min[0:min(end_min,nbsym+0)][::-1]
                lsym=ind_max[0]
            else:
                lmax=ind_max[0:min(end_max,nbsym)][::-1]
                lmin=np.append(ind_min[0:min(end_min,nbsym-1)][::-1],0)
                lsym=0
        else:
            if S[0]<S[ind_max[0]]:
                lmax=ind_max[0:min(end_max,nbsym+0)][::-1]
                lmin=ind_min[1:min(end_min,nbsym+1)][::-1]
                lsym=ind_min[0]
            else:
                lmax=np.append(ind_max[0:min(end_max,nbsym-1)][::-1],0)
                lmin=ind_min[0:min(end_min,nbsym)][::-1]
                lsym=0
        if ind_max[-1]<ind_min[-1]:
            if S[-1]<S[ind_max[-1]]:
                rmax=ind_max[max(end_max-nbsym,0):][::-1]
                rmin=ind_min[max(end_min-nbsym-1,0):-1][::-1]
                rsym=ind_min[-1]
            else:
                rmax=np.append(ind_max[max(end_max-nbsym+1,0):],len(S)-1)[::-1]
                rmin=ind_min[max(end_min-nbsym,0):][::-1]
                rsym=len(S)-1
        else:
            if S[-1]>S[ind_min[-1]]:
                rmax=ind_max[max(end_max-nbsym-1,0):-1][::-1]
                rmin=ind_min[max(end_min-nbsym,0):][::-1]
                rsym=ind_max[-1]
            else:
                rmax=ind_max[max(end_max-nbsym,0):][::-1]
                rmin=np.append(ind_min[max(end_min-nbsym+1,0):],len(S)-1)[::-1]
                rsym=len(S)-1
        if not lmin.size: lmin=ind_min
        if not rmin.size: rmin=ind_min
        if not lmax.size: lmax=ind_max
        if not rmax.size: rmax=ind_max
        tlmin=2*T[lsym]-T[lmin]
        tlmax=2*T[lsym]-T[lmax]
        trmin=2*T[rsym]-T[rmin]
        trmax=2*T[rsym]-T[rmax]
        if tlmin[0]>T[0] or tlmax[0]>T[0]:
            if lsym==ind_max[0]: lmax=ind_max[0:min(end_max,nbsym)][::-1]
            else: lmin=ind_min[0:min(end_min,nbsym)][::-1]
            if lsym==0: raise Exception('Left edge BUG')
            lsym=0
            tlmin=2*T[lsym]-T[lmin]
            tlmax=2*T[lsym]-T[lmax]
        if trmin[-1]<T[-1] or trmax[-1]<T[-1]:
            if rsym==ind_max[-1]: rmax=ind_max[max(end_max-nbsym,0):][::-1]
            else: rmin=ind_min[max(end_min-nbsym,0):][::-1]
            if rsym==len(S)-1: raise Exception('Right edge BUG')
            rsym=len(S)-1
            trmin=2*T[rsym]-T[rmin]
            trmax=2*T[rsym]-T[rmax]
        zlmax,zlmin,zrmax,zrmin=S[lmax],S[lmin],S[rmax],S[rmin]
        tmin=np.concatenate((tlmin,T[ind_min],trmin))
        tmax=np.concatenate((tlmax,T[ind_max],trmax))
        zmin=np.concatenate((zlmin,S[ind_min],zrmin))
        zmax=np.concatenate((zlmax,S[ind_max],zrmax))
        max_extrema=np.array([tmax,zmax])
        min_extrema=np.array([tmin,zmin])
        max_dup_idx=np.where(max_extrema[0,1:]==max_extrema[0,:-1])
        max_extrema=np.delete(max_extrema,max_dup_idx,axis=1)
        min_dup_idx=np.where(min_extrema[0,1:]==min_extrema[0,:-1])
        min_extrema=np.delete(min_extrema,min_dup_idx,axis=1)
        return max_extrema,min_extrema
    def spline_points(self,T,extrema):
        kind=self.spline_kind.lower()
        t=T[(T>=extrema[0,0])&(T<=extrema[0,-1])]
        if kind=='akima': return t,Akima1DInterpolator(extrema[0],extrema[1])(t)
        elif kind=='cubic':
            if extrema.shape[1]>3: return t,CubicSpline(extrema[0],extrema[1])(t)
            return cubic_spline_3pts(extrema[0],extrema[1],t)
        elif kind=='pchip': return t,PchipInterpolator(extrema[0],extrema[1])(t)
        elif kind=='cubic_hermite':
            return t,CubicHermiteSpline(extrema[0],extrema[1],np.gradient(extrema[1],extrema[0]))(t)
        elif kind in ['slinear','quadratic','linear']:
            return T,interp1d(extrema[0],extrema[1],kind=kind)(t).astype(self.DTYPE)
        raise ValueError('No such interpolation method!')
# END PyEMD-derived geometry.


# Numerical contracts and Rilling-sift implementation; replaces the stock
# PyEMD sifting loop. The envelope geometry above is the reused portion.

class SiftingConvergenceError(RuntimeError):
    """An EMD candidate did not satisfy the declared IMF stopping criterion."""


class DecompositionLimitError(RuntimeError):
    """A safety limit was reached before a natural terminal state."""


class BackendContractError(RuntimeError):
    """An external EMD backend returned inconsistent IMFs and/or residue."""


def _integer(name, value, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise TypeError(f'{name} must be an integer, not {type(value).__name__}.')
    if value < minimum:
        raise ValueError(f'{name} must be >= {minimum}.')
    return int(value)


def _nonnegative(name, value, allow_none=False):
    if value is None and allow_none:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise TypeError(f'{name} must be a finite real number.')
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative.')
    return value


def _float_dtype(dtype):
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype('float32'), np.dtype('float64')):
        raise ValueError('dtype must be float32 or float64; arithmetic uses float64.')
    return dtype


def _vector(signal, name='S'):
    raw = np.asarray(signal)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError(f'{name} must be a nonempty one-dimensional array.')
    if raw.dtype.kind not in 'iuf':
        raise TypeError(f'{name} must contain real numeric values, not {raw.dtype}.')
    out = np.array(raw, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(out)):
        raise ValueError(f'{name} contains NaN, infinity, or values outside float64.')
    return out


def _time_vector(time, n):
    """Validate uniform sampling, then use dimensionless sample indices.

    This is an explicit domain restriction, not implicit interpolation. EMD
    geometry is invariant to positive affine changes of time. Nonuniform
    acquisition requires an explicit resampling decision outside this module.
    """
    if time is None:
        return np.arange(n, dtype=np.float64), None, None
    t = _vector(time, 'T')
    if len(t) != n:
        raise ValueError('S and T must have the same length.')
    if n == 1:
        return np.zeros(1), None, float(t[0])
    with np.errstate(over='ignore', invalid='ignore'):
        diff = np.diff(t)
    if not np.all(np.isfinite(diff)) or np.any(diff <= 0):
        raise ValueError('T must be finite and strictly increasing.')
    dt = float(np.median(diff))
    ulp = np.finfo(float).eps * max(float(np.max(np.abs(t))), dt)
    if dt < 32 * ulp:
        raise ValueError('T has insufficient numerical resolution; use relative times.')
    tolerance = max(1e-7 * dt, 8 * ulp)
    if np.any(np.abs(diff - dt) > tolerance):
        raise ValueError('ICEEMDAN requires uniform sampling; resample explicitly first.')
    return np.arange(n, dtype=np.float64), dt, float(t[0])


def _stable_std(x, ddof=1):
    x = np.asarray(x, dtype=np.float64)
    if x.size <= ddof:
        return 0.0
    peak = float(np.max(np.abs(x)))
    if peak == 0:
        return 0.0
    return peak * float(np.std(x / peak, ddof=ddof))


def _extrema(T, S):
    """Discrete extrema, including interior plateaus; endpoints are excluded.

    Plateaus are represented by the round-to-even midpoint index, matching
    PyEMD's midpoint convention. Signs are compared without multiplying tiny
    or huge samples. A maximal run touching either boundary is not an extremum.
    """
    n = len(S)
    # Every zero run contributes one zero location, in addition to sign changes.
    crosses = np.flatnonzero(((S[:-1] < 0) & (S[1:] > 0)) |
                             ((S[:-1] > 0) & (S[1:] < 0)))
    zero = np.r_[False, S == 0, False].astype(np.int8)
    edges = np.diff(zero)
    start, end = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1
    zeros = np.sort(np.r_[crosses, np.round((start + end) / 2.0)])
    # Run-length representation avoids both plateau boundary-index defects.
    starts = np.r_[0, np.flatnonzero(S[1:] != S[:-1]) + 1]
    ends = np.r_[starts[1:] - 1, n - 1]
    values = S[starts]
    if len(values) < 3:
        empty = np.empty(0, dtype=float)
        return empty, empty, empty, empty, zeros
    mids = np.round((starts[1:-1] + ends[1:-1]) / 2.0).astype(np.intp)
    high = mids[(values[1:-1] > values[:-2]) & (values[1:-1] > values[2:])]
    low = mids[(values[1:-1] < values[:-2]) & (values[1:-1] < values[2:])]
    return T[high], S[high], T[low], S[low], zeros


class EMD(_Geometry):
    """Bounded real EMD with explicit IMF/residue storage.

    The default criterion is Rilling's envelope ratio test (0.05, 0.5, 0.05)
    AND |number of extrema - number of zeros| <= 1. Unlike the upstream
    Cauchy-style loop, it does not accept MAX_ITERATION as convergence.

    Cubic uses the PyEMD geometry: SciPy not-a-knot for >3 support points,
    and the upstream three-point spline otherwise. Other spline choices
    are numerical variants, not an exact reproduction of the paper's kernel.
    """
    def __init__(self, spline_kind='cubic', nbsym=2, *, MAX_ITERATION=2000,
                 rilling_thresholds=(0.05, 0.5, 0.05), max_modes=100,
                 extrema_detection='simple', DTYPE=np.float64):
        if extrema_detection != 'simple':
            raise ValueError('The bundled kernel supports discrete extrema only; '
                             'use an explicit external backend for other definitions.')
        if spline_kind not in ('cubic','pchip','akima','cubic_hermite',
                               'linear','slinear','quadratic'):
            raise ValueError('Unsupported spline_kind.')
        self.spline_kind = spline_kind
        self.nbsym = _integer('nbsym', nbsym)
        self.MAX_ITERATION = _integer('MAX_ITERATION', MAX_ITERATION)
        self.max_modes = _integer('max_modes', max_modes)
        self.output_dtype = _float_dtype(DTYPE)
        self.DTYPE = np.float64
        self.extrema_detection = extrema_detection
        values = tuple(_nonnegative('rilling threshold', v) for v in rilling_thresholds)
        if len(values) != 3 or values[0] <= 0 or values[1] < values[0] or not 0 <= values[2] < 1:
            raise ValueError('Require 0 < threshold1 <= threshold2 and 0 <= fraction < 1.')
        self.rilling_thresholds = values
        self.imfs = self.residue = None
        self.diagnostics_ = []
        self.stop_reason_ = None

    def find_extrema(self, T, S):
        return _extrema(np.asarray(T), np.asarray(S))

    def extract_max_min_spline(self, T, S):
        hi, hiv, lo, lov, _ = self.find_extrema(T, S)
        if len(hi) + len(lo) < 3 or not len(hi) or not len(lo):
            raise ValueError('At least three alternating extrema are needed for envelopes.')
        ehi, elo = self.prepare_points_simple(T, S, hi, hiv, lo, lov)
        _, upper = self.spline_points(T, ehi)
        _, lower = self.spline_points(T, elo)
        if upper.shape != S.shape or lower.shape != S.shape:
            raise SiftingConvergenceError('Envelopes do not cover the complete record.')
        if not np.all(np.isfinite(upper)) or not np.all(np.isfinite(lower)):
            raise SiftingConvergenceError('Envelope interpolation produced non-finite values.')
        return upper, lower, ehi, elo

    def _criterion(self, x, t):
        ext = self.find_extrema(t, x)
        ne, nz = len(ext[0]) + len(ext[2]), len(ext[4])
        info = dict(extrema=ne, zeros=nz, is_imf=False,
                    envelope_ratio_max=None, fraction_above_threshold=None)
        if ne < 3:
            return info, None
        upper, lower, _, _ = self.extract_max_min_spline(t, x)
        mean = 0.5 * upper + 0.5 * lower
        amplitude = np.abs(0.5 * upper - 0.5 * lower)
        ratio = np.full(x.shape, np.inf)
        np.divide(np.abs(mean), amplitude, out=ratio, where=amplitude > 0)
        ratio[(amplitude == 0) & (mean == 0)] = 0
        threshold1, threshold2, allowed = self.rilling_thresholds
        fraction = float(np.mean(ratio > threshold1))
        max_ratio = float(np.max(ratio))
        info.update(envelope_ratio_max=max_ratio, fraction_above_threshold=fraction,
                    is_imf=bool(abs(ne - nz) <= 1 and fraction <= allowed and
                                max_ratio <= threshold2))
        return info, mean

    def imf_diagnostics(self, S, T=None):
        x = _vector(S)
        t, _, _ = _time_vector(T, len(x))
        peak = float(np.max(np.abs(x)))
        # Rescaling prevents underflow in spline algebra without altering criteria.
        info, _ = self._criterion(x / peak if peak else x, t)
        return info

    def _first(self, signal, t):
        h = signal.copy()
        for count in range(self.MAX_ITERATION + 1):
            info, mean = self._criterion(h, t)
            if mean is None:
                return None, dict(info, sift_iterations=count, converged=False,
                                  reason='fewer_than_three_extrema')
            if info['is_imf']:
                return h, dict(info, sift_iterations=count, converged=True, reason='rilling')
            if count == self.MAX_ITERATION:
                raise SiftingConvergenceError(
                    f'No IMF convergence after {count} siftings; '
                    f'extrema={info["extrema"]}, zeros={info["zeros"]}, '
                    f'max envelope ratio={info["envelope_ratio_max"]:.6g}.')
            candidate = h - mean
            if np.array_equal(candidate, h):
                raise SiftingConvergenceError('Sifting stagnated before satisfying the IMF criterion.')
            h = candidate
        raise AssertionError('Unreachable sifting state')

    def __call__(self, S, T=None, max_imf=-1):
        return self.emd(S, T=T, max_imf=max_imf)

    def emd(self, S, T=None, max_imf=-1):
        self.imfs = self.residue = None
        self.diagnostics_ = []
        self.stop_reason_ = None
        x = _vector(S)
        t, _, _ = _time_vector(T, len(x))
        max_imf = _integer('max_imf', max_imf, -1)
        if max_imf > self.max_modes:
            raise ValueError('max_imf exceeds the EMD max_modes safety limit.')
        peak = float(np.max(np.abs(x)))
        u = x / peak if peak else x.copy()
        r = u.copy()
        modes = []
        limit = max_imf if max_imf >= 0 else self.max_modes
        floor = 64 * np.finfo(float).eps
        while len(modes) < limit:
            ext = self.find_extrema(t, r)
            if len(ext[0]) + len(ext[2]) < 3:
                self.stop_reason_ = 'fewer_than_three_extrema'; break
            if np.max(np.abs(r)) <= floor:
                self.stop_reason_ = 'numerical_zero'; break
            h, info = self._first(r, t)
            if h is None:
                self.stop_reason_ = 'fewer_than_three_extrema'; break
            new_r = r - h
            if np.array_equal(new_r, r):
                raise SiftingConvergenceError('EMD extraction made no progress.')
            modes.append(h)
            self.diagnostics_.append(info)
            r = new_r
        else:
            if max_imf < 0:
                ext = self.find_extrema(t, r)
                if len(ext[0]) + len(ext[2]) >= 3 and np.max(np.abs(r)) > floor:
                    raise DecompositionLimitError('EMD max_modes reached before termination.')
                self.stop_reason_ = 'terminal_at_limit'
            else:
                self.stop_reason_ = 'max_imf'
        restored = np.array(modes, dtype=float).reshape(-1, len(x)) * peak
        residue = r * peak if peak else r
        with np.errstate(over='ignore',invalid='ignore'):
            result = np.asarray(np.vstack((restored,residue)), dtype=self.output_dtype)
        self.imfs, self.residue = result[:-1].copy(), result[-1].copy()
        if not np.all(np.isfinite(result)):
            self.imfs = self.residue = None
            raise FloatingPointError(f'EMD output is not representable in {self.output_dtype}.')
        # IMF and residue remain unambiguously separate; never omit a zero row.
        return result

    def get_imfs_and_residue(self):
        if self.imfs is None or self.residue is None:
            raise ValueError('No successful EMD decomposition is available.')
        return self.imfs.copy(), self.residue.copy()


# ICEEMDAN recurrence. Original wrapper attribution retained in module header.


def _backend_parts(backend, signal, t, max_imf):
    """Use the explicit getter, never infer a residual from the last raw row."""
    backend.emd(signal.copy(), t, max_imf=max_imf)
    parts, remainder = backend.get_imfs_and_residue()
    raw_dtype = np.asarray(parts).dtype
    if raw_dtype.kind not in 'iuf' or np.asarray(remainder).dtype.kind not in 'iuf':
        raise BackendContractError('EMD getter must return real numeric arrays.')
    modes = np.asarray(parts, dtype=float)
    remainder = np.asarray(remainder, dtype=float)
    n = len(signal)
    if modes.ndim != 2 or modes.shape[1] != n or remainder.shape != (n,):
        raise BackendContractError('EMD getter must return (K,N) IMFs and an (N,) residue.')
    if max_imf >= 0 and len(modes) > max_imf:
        raise BackendContractError('EMD returned more IMFs than requested.')
    if not np.all(np.isfinite(modes)) or not np.all(np.isfinite(remainder)):
        raise BackendContractError('EMD returned non-finite values.')
    eps = np.finfo(raw_dtype).eps if raw_dtype.kind == 'f' else np.finfo(float).eps
    # Relative, scale-aware reconstruction check; IMF semantics remain the backend contract.
    peak = float(np.max(np.abs(signal)))
    scale = peak if peak else 1.0
    defect = np.max(np.abs((modes / scale).sum(axis=0) + remainder / scale - signal / scale))
    if defect > 256 * eps * max(1, len(modes)):
        raise BackendContractError(f'EMD getter violates reconstruction; scaled defect={defect:.3g}.')
    records = getattr(backend, 'diagnostics_', None)
    iterations = None
    if records and len(modes):
        record = records[0]
        if isinstance(record, dict):
            iterations = record.get('sift_iterations')
    return modes.copy(), remainder.copy(), iterations


def _local_mean(backend, signal, t):
    modes, _, iterations = _backend_parts(backend, signal, t, 1)
    # E1=0 when no IMF exists; M(y)=y in that case. If E1=y, M(y)=0.
    mean = signal - modes[0] if len(modes) else signal.copy()
    return mean, iterations


_WORKER_BACKEND = None


def _init_worker(backend):
    global _WORKER_BACKEND
    _WORKER_BACKEND = backend


def _noise_worker(args):
    signal, t, max_imf = args
    modes, _, _ = _backend_parts(_WORKER_BACKEND, signal, t, max_imf)
    return modes


def _mean_worker(args):
    signal, t = args
    return _local_mean(_WORKER_BACKEND, signal, t)


class ICEEMDAN:
    """Improved CEEMDAN following Colominas, Schlotthauer and Torres (2014).

    Parameters
    ----------
    trials : positive int, default 100
        Fixed ensemble size I; missing later noise modes NEVER shrink it.
    epsilon : nonnegative float, default 0.2
        First-stage perturbation standard deviation divided by std(input).
        E1(noise) is normalized separately for EVERY realization. Later noise
        modes are used raw, as in section 3.3 (SNRFlag=1 in the author's code).
    ext_EMD : optional backend
        Must expose emd(S,T,max_imf) AND get_imfs_and_residue(). A backend is
        required to be deterministic for each input. Parallel mode serializes
        the actual object, including custom state; it does not infer parameters.
        Its own sifting/convergence guarantees replace those of the bundled EMD.
    emd_params : dict, optional
        Passed only to the bundled EMD constructor. May not be combined with
        ext_EMD, to avoid ambiguous or mismatched worker configurations.
    parallel : bool, default False
        Parallelizes noise decomposition AND ensemble local means, with ordered
        reductions. Uses spawn on all platforms. Guard the calling script.
    processes : positive int or None
        None uses min(trials, available CPUs).
    dtype : float64 or float32, default float64
        OUTPUT precision. Core calculations and sums remain float64. Input
        integer dtype is never restored: that would truncate the components.
    seed : nonnegative integer or None
        NumPy Generator(PCG64). Calls advance the RNG; noise_seed resets it.
    max_imf_iterations : positive int, default 100
        Independent safety limit. Hitting it without natural termination raises.
    noise_kind : 'normal' (default) or 'uniform'
        Uniform is a documented experimental variant, not the paper's Gaussian
        experiment. Both generators have unit population variance before EMD.
    range_thr, total_power_thr : nonnegative float or None
        Optional normalized-residue thresholds. The latter is an L1 SUM, not
        energy/power. Defaults None preserve the mathematical stopping rule.
    residue_std_threshold : nonnegative float, default 0
        Optional normalized-residue standard-deviation threshold. A separate
        machine-precision safeguard is always present.
    ddof : 0 or 1, default 1
        Explicit sample-standard-deviation convention. Default agrees with
        MATLAB std; ddof=0 is a legitimate alternate convention, not a repair.

    Notes
    -----
    Uniformly sampled finite real one-dimensional signals only. No implicit
    resampling, filtering, component removal, or modal/SHM interpretation.
    Positive scale normalization does NOT center the input or remove its DC.
    A terminal oscillatory IMF may remain in the LAST (residue) row, as in the
    paper's early terminal-IMF criterion. Inspect the residue before discarding.
    The averaged components need not satisfy the exact IMF conditions; their
    measured diagnostics are exposed rather than asserted without evidence.
    """
    logger = logging.getLogger(__name__)

    def __init__(self, trials=100, epsilon=0.2, ext_EMD=None, parallel=False,
                 emd_params=None, dtype=np.float64, *, seed=None, processes=None,
                 noise_kind='normal', range_thr=None, total_power_thr=None,
                 residue_std_threshold=0.0, max_imf_iterations=100, ddof=1):
        self.trials = _integer('trials', trials)
        self.epsilon = _nonnegative('epsilon', epsilon)
        self.dtype = _float_dtype(dtype)
        if not isinstance(parallel, (bool, np.bool_)):
            raise TypeError('parallel must be bool.')
        self.parallel = bool(parallel)
        self.processes = (min(self.trials, mp.cpu_count()) if processes is None
                          else _integer('processes', processes))
        self.max_imf_iterations = _integer('max_imf_iterations', max_imf_iterations)
        self.range_thr = _nonnegative('range_thr', range_thr, True)
        self.total_power_thr = _nonnegative('total_power_thr', total_power_thr, True)
        self.residue_std_threshold = _nonnegative('residue_std_threshold', residue_std_threshold)
        self.ddof = _integer('ddof', ddof, 0)
        if self.ddof not in (0,1):
            raise ValueError('ddof must be 0 or 1.')
        if noise_kind not in ('normal','uniform'):
            raise ValueError('noise_kind must be normal or uniform.')
        self.noise_kind = noise_kind
        if noise_kind == 'uniform':
            warnings.warn('Uniform noise is an experimental variant, not the Gaussian '
                          'protocol in Colominas (2014).', UserWarning, stacklevel=2)
        if emd_params is not None and not isinstance(emd_params, dict):
            raise TypeError('emd_params must be a dictionary or None.')
        if ext_EMD is not None and emd_params:
            raise ValueError('Configure ext_EMD itself; do not also pass emd_params.')
        self.emd_params_internal = dict(emd_params or {})
        if ext_EMD is None:
            self.emd_params_internal.setdefault('max_modes',self.max_imf_iterations)
            self.EMD = EMD(**self.emd_params_internal)
        else:
            for method in ('emd','get_imfs_and_residue'):
                if not callable(getattr(ext_EMD, method, None)):
                    raise TypeError(f'ext_EMD must implement {method}().')
            self.EMD = ext_EMD
        self.random = None
        self.noise_seed(seed)
        self.C_IMF = self.residue = None
        self.diagnostics_ = None
        self.all_noise_IMFs_for_CEEMDAN = []
        self._active = False
        self._executor = None
        self._noise_cap = self.max_imf_iterations
        self._last_iterations = None

    def noise_seed(self, seed):
        if seed is not None:
            seed = _integer('seed', seed, 0)
        self.random = np.random.default_rng(seed)

    def generate_noise(self, scale, size):
        scale = _nonnegative('scale', scale)
        if isinstance(size, numbers.Integral) and not isinstance(size, (bool,np.bool_)):
            shape = _integer('size',size,0)
        else:
            if not isinstance(size,(tuple,list)):
                raise TypeError('size must be an integer or a sequence of integer dimensions.')
            shape = tuple(_integer('size dimension', v, 0) for v in size)
        if self.noise_kind == 'normal':
            out = self.random.normal(0.0, scale, size=shape)
        else:
            limit = scale * np.sqrt(3.0)
            if not np.isfinite(limit):
                raise ValueError('Uniform noise scale is too large.')
            out = self.random.uniform(-limit,limit,size=shape)
        if not np.all(np.isfinite(out)):
            raise FloatingPointError('Noise generation overflowed.')
        return out

    def _get_emd_local_mean(self, S_plus_noise, T):
        mean, self._last_iterations = _local_mean(self.EMD, S_plus_noise, T)
        return mean

    def _pre_decompose_noise_for_ceemdan(self, S_shape, T):
        # Raw noise vectors are generated in the parent for serial/spawn identity.
        jobs = ((self.generate_noise(1.0, S_shape), T, self._noise_cap)
                for _ in range(self.trials))
        if self._executor is None:
            banks = [_backend_parts(self.EMD, y, t, cap)[0] for y,t,cap in jobs]
        else:
            banks = list(self._executor.map(_noise_worker, jobs, chunksize=1))
        self.all_noise_IMFs_for_CEEMDAN = banks

    def _stop_reason(self, r, t, check_imf):
        ext = _extrema(t, r)
        if len(ext[0]) + len(ext[2]) < 3:
            return 'fewer_than_three_extrema'
        sd = _stable_std(r,self.ddof)
        if sd <= 64*np.finfo(float).eps:
            return 'numerical_zero'
        if self.residue_std_threshold > 0 and sd < self.residue_std_threshold:
            return 'residue_std_threshold'
        if self.range_thr is not None and np.ptp(r) < self.range_thr:
            return 'range_threshold'
        if self.total_power_thr is not None and np.sum(np.abs(r)) < self.total_power_thr:
            return 'residue_l1_threshold'
        if check_imf:
            native = isinstance(self.EMD, EMD)
            if native and self.EMD.imf_diagnostics(r,t)['is_imf']:
                return 'residue_is_imf'
            # Starting with >=3 extrema does not ensure an IMF can be extracted:
            # sifting itself can reduce that number. Inspect the explicit getter.
            modes, _, _ = _backend_parts(self.EMD,r,t,1)
            if not len(modes):
                return 'emd_has_no_extractable_imf'
            if not native:
                local = r-modes[0]
                if np.max(np.abs(local)) <= 64*np.finfo(float).eps*max(1.0,np.max(np.abs(r))):
                    return 'residue_is_imf_external'
        return None

    def end_condition(self, current_residue_r_k, max_imf_user_request,
                      num_extracted_cimfs, T=None):
        r = _vector(current_residue_r_k)
        t,_,_ = _time_vector(T,len(r))
        requested = _integer('max_imf',max_imf_user_request,-1)
        done = _integer('num_extracted_cimfs',num_extracted_cimfs,0)
        if requested >= 0 and done >= requested:
            return True
        return self._stop_reason(r,t,check_imf=done>0) is not None

    def __call__(self,S,T=None,max_imf=-1,progress=False):
        return self.ceemdan(S,T=T,max_imf=max_imf,progress=progress)

    def ceemdan(self,S,T=None,max_imf=-1,progress=False):
        if self._active:
            raise RuntimeError('An ICEEMDAN instance cannot run concurrent decompositions.')
        self.C_IMF = self.residue = self.diagnostics_ = None
        self.all_noise_IMFs_for_CEEMDAN = []
        self._active = True
        rng_state = copy.deepcopy(self.random.bit_generator.state)
        try:
            x = _vector(S)
            t,dt,t0 = _time_vector(T,len(x))
            requested = _integer('max_imf',max_imf,-1)
            if requested > self.max_imf_iterations:
                raise ValueError('max_imf exceeds max_imf_iterations; raise the safety limit explicitly.')
            if not isinstance(progress,(bool,np.bool_)):
                raise TypeError('progress must be bool.')
            info = dict(version=__version__,backend=f'{type(self.EMD).__module__}.{type(self.EMD).__qualname__}',
                        noise_kind=self.noise_kind,trials=self.trials,epsilon=self.epsilon,
                        ddof=self.ddof,internal_dtype='float64',output_dtype=self.dtype.name,
                        sample_interval=dt,sample_origin=t0,parallel=self.parallel,
                        processes=self.processes if self.parallel else 1,
                        rng='PCG64',rng_state_sha256=hashlib.sha256(repr(rng_state).encode()).hexdigest(),
                        noise_mode_counts=[],stages=[],noise_bytes=0)
            peak = float(np.max(np.abs(x)))
            # Scale only: no mean removal and no hidden preprocessing.
            scaled = x / peak if peak else x.copy()
            std_scaled = _stable_std(scaled,self.ddof)
            if requested == 0 or std_scaled == 0 or len(x)<3:
                return self._finish(x.reshape(1,-1),info,
                                    'max_imf' if requested==0 else 'constant_or_short',requested!=0)
            r = scaled / std_scaled
            reason = self._stop_reason(r,t,check_imf=False)
            if reason:
                return self._finish(x.reshape(1,-1),info,reason,True)
            if self.parallel:
                try:
                    worker_backend = pickle.loads(pickle.dumps(self.EMD))
                except Exception as exc:
                    raise TypeError('ext_EMD is not serializable for spawn; use parallel=False '
                                    'or a module-level serializable backend.') from exc
                context = ProcessPoolExecutor(max_workers=self.processes,
                                              mp_context=mp.get_context('spawn'),
                                              initializer=_init_worker,initargs=(worker_backend,))
            else:
                context = nullcontext(None)
            self._noise_cap = requested if requested > 0 else self.max_imf_iterations
            with context as executor:
                self._executor = executor
                if self.epsilon > 0:
                    self._pre_decompose_noise_for_ceemdan(r.shape,t)
                    banks = self.all_noise_IMFs_for_CEEMDAN
                    if len(banks)!=self.trials:
                        raise BackendContractError('Noise decomposition changed the ensemble size.')
                    info['noise_mode_counts'] = [len(b) for b in banks]
                    info['noise_bytes'] = sum(b.nbytes for b in banks)
                    first_std = np.array([_stable_std(b[0],self.ddof) if len(b) else 0.0 for b in banks])
                    if np.any(first_std <= 0) or not np.all(np.isfinite(first_std)):
                        raise BackendContractError('Every realization needs a finite nonzero first noise IMF.')
                else:
                    banks = [np.empty((0,len(x))) for _ in range(self.trials)]
                    first_std = np.ones(self.trials)
                    info['noise_mode_counts'] = [0]*self.trials
                modes = []
                reason = None
                while True:
                    # The requested cap truncates intentionally, independently of safety limits.
                    if requested >= 0 and len(modes) >= requested:
                        reason='max_imf'; break
                    reason=self._stop_reason(r,t,check_imf=len(modes)>0)
                    if reason:
                        break
                    if len(modes) >= self.max_imf_iterations:
                        raise DecompositionLimitError('ICEEMDAN safety limit reached before natural termination.')
                    k=len(modes)+1
                    sd=_stable_std(r,self.ddof)
                    availability=[]
                    perturbation_stds=[]
                    if self.epsilon == 0:
                        new_r=self._get_emd_local_mean(r,t)
                        counts=[self._last_iterations]*self.trials
                        availability=[False]*self.trials
                        perturbation_stds=[0.0]*self.trials
                    else:
                        def jobs():
                            for i in range(self.trials):
                                exists=len(banks[i])>=k
                                availability.append(exists)
                                if exists:
                                    coefficient=(self.epsilon*sd/first_std[i] if k==1 else self.epsilon*sd)
                                    eta=coefficient*banks[i][k-1]
                                    perturbation_stds.append(_stable_std(eta,self.ddof))
                                    y=r+eta
                                else:
                                    # Author's fallback, made explicit: E_k(w_i)=0.
                                    # Include M(r) and KEEP the denominator I.
                                    y=r.copy()
                                    perturbation_stds.append(0.0)
                                yield y,t
                        if executor is None:
                            def local_results():
                                for y,ti in jobs():
                                    m=self._get_emd_local_mean(y,ti)
                                    yield m,self._last_iterations
                            results=local_results()
                        else:
                            results=executor.map(_mean_worker,jobs(),chunksize=1)
                        total=np.zeros_like(r)
                        correction=np.zeros_like(r)
                        counts=[]
                        # Kahan reduction in realization order: deterministic serial/spawn.
                        for local,count in results:
                            term=local-correction
                            summed=total+term
                            correction=(summed-total)-term
                            total=summed
                            counts.append(count)
                        if len(counts)!=self.trials:
                            raise BackendContractError('The local-mean ensemble lost realizations.')
                        new_r=total/self.trials
                    if not np.all(np.isfinite(new_r)):
                        raise FloatingPointError('Non-finite ICEEMDAN residue.')
                    component=r-new_r
                    if np.array_equal(r,new_r):
                        raise SiftingConvergenceError('ICEEMDAN made no progress before its terminal condition.')
                    modes.append(component)
                    info['stages'].append(dict(index=k,noise_mode_index=k,ensemble_denominator=self.trials,
                                                missing_noise_modes=availability.count(False),
                                                perturbation_stds=perturbation_stds,
                                                residue_std_before=sd,sift_iterations=counts))
                    r=new_r
                    if progress:
                        print(f'ICEEMDAN: component {k}, residual std={_stable_std(r,self.ddof):.6g}',flush=True)
                normalized=np.vstack((*modes,r))
                with np.errstate(over='ignore',invalid='ignore'):
                    components=(normalized*std_scaled)*peak
                if not np.all(np.isfinite(components)):
                    raise FloatingPointError('ICEEMDAN components exceed float64 range.')
            # No reconstruction repair: the stored residue is the one from the recurrence.
            info['reconstruction_scaled_linf']=float(np.max(np.abs((components/peak).sum(0)-x/peak)))
            natural=reason not in ('max_imf','range_threshold','residue_l1_threshold','residue_std_threshold')
            return self._finish(components,info,reason,natural)
        except BaseException:
            self.C_IMF=self.residue=self.diagnostics_=None
            self.random.bit_generator.state=rng_state
            raise
        finally:
            self.all_noise_IMFs_for_CEEMDAN=[]
            self._executor=None
            self._active=False

    def _finish(self,components,info,reason,complete):
        with np.errstate(over='ignore',invalid='ignore'):
            output=np.asarray(components,dtype=self.dtype)
        if not np.all(np.isfinite(output)):
            raise FloatingPointError(f'Results are not representable in output dtype {self.dtype}.')
        self.C_IMF=output[:-1].copy()
        self.residue=output[-1].copy()
        info.update(stop_reason=reason,natural_termination=bool(complete),n_components=len(output)-1,
                    residue_row=len(output)-1)
        # Diagnose, do not force the averaged components to become different signals.
        diagnostic_kernel=self.EMD if isinstance(self.EMD,EMD) else EMD()
        info['component_diagnostic_rule']=dict(
            kind='Rilling',spline_kind=diagnostic_kernel.spline_kind,
            nbsym=diagnostic_kernel.nbsym,thresholds=list(diagnostic_kernel.rilling_thresholds),
            matches_decomposition_backend=isinstance(self.EMD,EMD))
        info['component_imf_diagnostics']=[diagnostic_kernel.imf_diagnostics(row) for row in output[:-1]]
        self.diagnostics_=info
        return output.copy()

    def get_imfs_and_residue(self):
        if self.C_IMF is None or self.residue is None:
            raise ValueError('No successful ICEEMDAN decomposition is available.')
        return self.C_IMF.copy(),self.residue.copy()


# Compatibility with the supplied file's class/import name.
CEEMDAN = ICEEMDAN



# Third-party license (applies to PyEMD-derived portions):
APACHE_2_0_LICENSE = '\n                                 Apache License\n                           Version 2.0, January 2004\n                        http://www.apache.org/licenses/\n\n   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION\n\n   1. Definitions.\n\n      "License" shall mean the terms and conditions for use, reproduction,\n      and distribution as defined by Sections 1 through 9 of this document.\n\n      "Licensor" shall mean the copyright owner or entity authorized by\n      the copyright owner that is granting the License.\n\n      "Legal Entity" shall mean the union of the acting entity and all\n      other entities that control, are controlled by, or are under common\n      control with that entity. For the purposes of this definition,\n      "control" means (i) the power, direct or indirect, to cause the\n      direction or management of such entity, whether by contract or\n      otherwise, or (ii) ownership of fifty percent (50%) or more of the\n      outstanding shares, or (iii) beneficial ownership of such entity.\n\n      "You" (or "Your") shall mean an individual or Legal Entity\n      exercising permissions granted by this License.\n\n      "Source" form shall mean the preferred form for making modifications,\n      including but not limited to software source code, documentation\n      source, and configuration files.\n\n      "Object" form shall mean any form resulting from mechanical\n      transformation or translation of a Source form, including but\n      not limited to compiled object code, generated documentation,\n      and conversions to other media types.\n\n      "Work" shall mean the work of authorship, whether in Source or\n      Object form, made available under the License, as indicated by a\n      copyright notice that is included in or attached to the work\n      (an example is provided in the Appendix below).\n\n      "Derivative Works" shall mean any work, whether in Source or Object\n      form, that is based on (or derived from) the Work and for which the\n      editorial revisions, annotations, elaborations, or other modifications\n      represent, as a whole, an original work of authorship. For the purposes\n      of this License, Derivative Works shall not include works that remain\n      separable from, or merely link (or bind by name) to the interfaces of,\n      the Work and Derivative Works thereof.\n\n      "Contribution" shall mean any work of authorship, including\n      the original version of the Work and any modifications or additions\n      to that Work or Derivative Works thereof, that is intentionally\n      submitted to Licensor for inclusion in the Work by the copyright owner\n      or by an individual or Legal Entity authorized to submit on behalf of\n      the copyright owner. For the purposes of this definition, "submitted"\n      means any form of electronic, verbal, or written communication sent\n      to the Licensor or its representatives, including but not limited to\n      communication on electronic mailing lists, source code control systems,\n      and issue tracking systems that are managed by, or on behalf of, the\n      Licensor for the purpose of discussing and improving the Work, but\n      excluding communication that is conspicuously marked or otherwise\n      designated in writing by the copyright owner as "Not a Contribution."\n\n      "Contributor" shall mean Licensor and any individual or Legal Entity\n      on behalf of whom a Contribution has been received by Licensor and\n      subsequently incorporated within the Work.\n\n   2. Grant of Copyright License. Subject to the terms and conditions of\n      this License, each Contributor hereby grants to You a perpetual,\n      worldwide, non-exclusive, no-charge, royalty-free, irrevocable\n      copyright license to reproduce, prepare Derivative Works of,\n      publicly display, publicly perform, sublicense, and distribute the\n      Work and such Derivative Works in Source or Object form.\n\n   3. Grant of Patent License. Subject to the terms and conditions of\n      this License, each Contributor hereby grants to You a perpetual,\n      worldwide, non-exclusive, no-charge, royalty-free, irrevocable\n      (except as stated in this section) patent license to make, have made,\n      use, offer to sell, sell, import, and otherwise transfer the Work,\n      where such license applies only to those patent claims licensable\n      by such Contributor that are necessarily infringed by their\n      Contribution(s) alone or by combination of their Contribution(s)\n      with the Work to which such Contribution(s) was submitted. If You\n      institute patent litigation against any entity (including a\n      cross-claim or counterclaim in a lawsuit) alleging that the Work\n      or a Contribution incorporated within the Work constitutes direct\n      or contributory patent infringement, then any patent licenses\n      granted to You under this License for that Work shall terminate\n      as of the date such litigation is filed.\n\n   4. Redistribution. You may reproduce and distribute copies of the\n      Work or Derivative Works thereof in any medium, with or without\n      modifications, and in Source or Object form, provided that You\n      meet the following conditions:\n\n      (a) You must give any other recipients of the Work or\n          Derivative Works a copy of this License; and\n\n      (b) You must cause any modified files to carry prominent notices\n          stating that You changed the files; and\n\n      (c) You must retain, in the Source form of any Derivative Works\n          that You distribute, all copyright, patent, trademark, and\n          attribution notices from the Source form of the Work,\n          excluding those notices that do not pertain to any part of\n          the Derivative Works; and\n\n      (d) If the Work includes a "NOTICE" text file as part of its\n          distribution, then any Derivative Works that You distribute must\n          include a readable copy of the attribution notices contained\n          within such NOTICE file, excluding those notices that do not\n          pertain to any part of the Derivative Works, in at least one\n          of the following places: within a NOTICE text file distributed\n          as part of the Derivative Works; within the Source form or\n          documentation, if provided along with the Derivative Works; or,\n          within a display generated by the Derivative Works, if and\n          wherever such third-party notices normally appear. The contents\n          of the NOTICE file are for informational purposes only and\n          do not modify the License. You may add Your own attribution\n          notices within Derivative Works that You distribute, alongside\n          or as an addendum to the NOTICE text from the Work, provided\n          that such additional attribution notices cannot be construed\n          as modifying the License.\n\n      You may add Your own copyright statement to Your modifications and\n      may provide additional or different license terms and conditions\n      for use, reproduction, or distribution of Your modifications, or\n      for any such Derivative Works as a whole, provided Your use,\n      reproduction, and distribution of the Work otherwise complies with\n      the conditions stated in this License.\n\n   5. Submission of Contributions. Unless You explicitly state otherwise,\n      any Contribution intentionally submitted for inclusion in the Work\n      by You to the Licensor shall be under the terms and conditions of\n      this License, without any additional terms or conditions.\n      Notwithstanding the above, nothing herein shall supersede or modify\n      the terms of any separate license agreement you may have executed\n      with Licensor regarding such Contributions.\n\n   6. Trademarks. This License does not grant permission to use the trade\n      names, trademarks, service marks, or product names of the Licensor,\n      except as required for reasonable and customary use in describing the\n      origin of the Work and reproducing the content of the NOTICE file.\n\n   7. Disclaimer of Warranty. Unless required by applicable law or\n      agreed to in writing, Licensor provides the Work (and each\n      Contributor provides its Contributions) on an "AS IS" BASIS,\n      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or\n      implied, including, without limitation, any warranties or conditions\n      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A\n      PARTICULAR PURPOSE. You are solely responsible for determining the\n      appropriateness of using or redistributing the Work and assume any\n      risks associated with Your exercise of permissions under this License.\n\n   8. Limitation of Liability. In no event and under no legal theory,\n      whether in tort (including negligence), contract, or otherwise,\n      unless required by applicable law (such as deliberate and grossly\n      negligent acts) or agreed to in writing, shall any Contributor be\n      liable to You for damages, including any direct, indirect, special,\n      incidental, or consequential damages of any character arising as a\n      result of this License or out of the use or inability to use the\n      Work (including but not limited to damages for loss of goodwill,\n      work stoppage, computer failure or malfunction, or any and all\n      other commercial damages or losses), even if such Contributor\n      has been advised of the possibility of such damages.\n\n   9. Accepting Warranty or Additional Liability. While redistributing\n      the Work or Derivative Works thereof, You may choose to offer,\n      and charge a fee for, acceptance of support, warranty, indemnity,\n      or other liability obligations and/or rights consistent with this\n      License. However, in accepting such obligations, You may act only\n      on Your own behalf and on Your sole responsibility, not on behalf\n      of any other Contributor, and only if You agree to indemnify,\n      defend, and hold each Contributor harmless for any liability\n      incurred by, or claims asserted against, such Contributor by reason\n      of your accepting any such warranty or additional liability.\n\n   END OF TERMS AND CONDITIONS\n\n   APPENDIX: How to apply the Apache License to your work.\n\n      To apply the Apache License to your work, attach the following\n      boilerplate notice, with the fields enclosed by brackets "[]"\n      replaced with your own identifying information. (Don\'t include\n      the brackets!)  The text should be enclosed in the appropriate\n      comment syntax for the file format. We also recommend that a\n      file or class name and description of purpose be included on the\n      same "printed page" as the copyright notice for easier\n      identification within third-party archives.\n\n   Copyright [yyyy] [name of copyright owner]\n\n   Licensed under the Apache License, Version 2.0 (the "License");\n   you may not use this file except in compliance with the License.\n   You may obtain a copy of the License at\n\n       http://www.apache.org/licenses/LICENSE-2.0\n\n   Unless required by applicable law or agreed to in writing, software\n   distributed under the License is distributed on an "AS IS" BASIS,\n   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n   See the License for the specific language governing permissions and\n   limitations under the License.\n'
