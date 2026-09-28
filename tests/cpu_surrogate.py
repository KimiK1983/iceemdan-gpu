"""Test double ONLY. Exercises Python port logic using NumPy/SciPy, NOT CuPy.

Never installed/imported by the production package. It provides no evidence
about CUDA compilation, allocation, synchronization, kernels or GPU precision.
"""

import contextlib
import sys
import types

import numpy as np
import scipy.interpolate as si


class Device:
    def __init__(self, device=None):
        self.id = 0 if device is None else int(device)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def synchronize(self):
        return None

    @property
    def mem_info(self):
        return (0, 0)


class Array(np.ndarray):
    @property
    def device(self):
        return Device(0)


def arr(x, dtype=None, **kwargs):
    return np.array(x, dtype=dtype, copy=True).view(Array)


def wrap(fn):
    def call(*a, **kw):
        kw.pop("blocking", None)
        y = fn(*a, **kw)
        if isinstance(y, (np.ndarray, np.generic)):
            return np.asarray(y).view(Array)
        return y

    return call


class Philox:
    def __init__(self, seed=None, **kwargs):
        self.seed = seed


class Generator:
    def __init__(self, bitgen):
        self.rng = np.random.default_rng(bitgen.seed)

    def standard_normal(self, size=None, dtype=np.float64):
        return self.rng.standard_normal(size).astype(dtype).view(Array)

    def uniform(self, low=0, high=1, size=None, dtype=np.float64):
        return self.rng.uniform(low, high, size).astype(dtype).view(Array)


def install():
    cp = types.ModuleType("cupy")
    cp.__version__ = "14.2.0"
    cp._test_surrogate = True
    cp.ndarray = Array
    cp.asarray = arr
    cp.array = arr
    for name in [
        "empty",
        "zeros",
        "ones",
        "zeros_like",
        "full",
        "arange",
        "concatenate",
        "stack",
        "vstack",
        "diff",
        "flatnonzero",
        "sort",
        "rint",
        "absolute",
        "max",
        "mean",
        "std",
        "median",
        "sum",
        "ptp",
        "all",
        "any",
        "isfinite",
        "array_equal",
        "where",
        "gradient",
    ]:
        setattr(cp, name, wrap(getattr(np, name)))
    cp.asnumpy = lambda x, **kw: np.array(x, copy=True)
    cp.linalg = types.SimpleNamespace(solve=wrap(np.linalg.solve))
    cp.random = types.SimpleNamespace(Philox4x3210=Philox, Generator=Generator)
    cp.cuda = types.SimpleNamespace(
        Device=Device, runtime=types.SimpleNamespace(getDeviceCount=lambda: 1)
    )
    cx = types.ModuleType("cupyx")
    cx.errstate = lambda **kw: contextlib.nullcontext()
    sc = types.ModuleType("cupyx.scipy")
    it = types.ModuleType("cupyx.scipy.interpolate")
    for name in [
        "CubicSpline",
        "CubicHermiteSpline",
        "PchipInterpolator",
        "Akima1DInterpolator",
        "interp1d",
    ]:
        cls = getattr(si, name)

        def factory(*a, _cls=cls, **kw):
            obj = _cls(*a, **kw)
            return lambda q: arr(obj(q))

        setattr(it, name, factory)
    for name, obj in [
        ("cupy", cp),
        ("cupyx", cx),
        ("cupyx.scipy", sc),
        ("cupyx.scipy.interpolate", it),
    ]:
        sys.modules[name] = obj
