"""Explicit CUDA boundaries. See docs/CUPY_API_LEDGER.json before adding APIs."""

from __future__ import annotations

from functools import wraps

try:
    import cupy as cp
except ImportError as exc:
    raise ImportError(
        "Install CuPy 14.2.0 (cupy-cuda12x or the appropriate CUDA wheel); "
        "this package has no automatic CPU fallback."
    ) from exc

CUPY_VERSION = "14.2.0"
if cp.__version__ != CUPY_VERSION:
    raise RuntimeError(
        f"This reference port is pinned to CuPy {CUPY_VERSION}; "
        f"found {cp.__version__}. Revalidate before changing the pin."
    )


def scalar(value):
    """Transfer one device value to a Python scalar when necessary.

    Parameters
    ----------
    value : cupy.ndarray or scalar
        A single device element or an existing host scalar.

    Returns
    -------
    scalar
        Host value used for branching or diagnostics.
    """
    if isinstance(value, cp.ndarray):
        if value.size != 1:
            raise ValueError("Expected exactly one device element for scalar control.")
        return value.item()
    # Values already in host metadata need no device transfer.
    return value.item() if hasattr(value, "item") else value


def on_device(fn):
    """Run an instance method in its configured CUDA device context.

    Parameters
    ----------
    fn : callable
        Method whose instance has a ``device_id`` attribute.

    Returns
    -------
    callable
        Wrapped method; no peer-to-peer array transfer is performed.
    """

    @wraps(fn)
    def wrapped(self, *args, **kwargs):
        with cp.cuda.Device(self.device_id):
            return fn(self, *args, **kwargs)

    return wrapped


def device_array(value, *, dtype=None, copy=False):
    """Import an array into the current device with blocking host transfer.

    Parameters
    ----------
    value : array_like
        Host or device input. Device arrays must belong to the current GPU.
    dtype : dtype or str, optional
        Requested device dtype.
    copy : bool, default False
        Request an owned copy. A host input is transferred synchronously.

    Returns
    -------
    cupy.ndarray
        Array resident on the current device.
    """
    if isinstance(value, cp.ndarray):
        if value.device.id != cp.cuda.Device().id:
            raise ValueError("Input belongs to another GPU. Transfer it explicitly first.")
    return cp.asarray(value, dtype=dtype, copy=copy if copy else None, blocking=True)


def require_device_array(value, name="array"):
    if not isinstance(value, cp.ndarray):
        raise TypeError(f"{name} must be a CuPy array; CPU backends are not accepted.")
    if value.device.id != cp.cuda.Device().id:
        raise ValueError(f"{name} belongs to a different device.")
    return value


def to_numpy(value):
    """Download a CuPy array to host memory explicitly.

    Parameters
    ----------
    value : cupy.ndarray
        Array to transfer from its owning GPU.

    Returns
    -------
    numpy.ndarray
        Host copy after a blocking transfer. The numerical core does not call
        this helper.
    """
    if not isinstance(value, cp.ndarray):
        raise TypeError("to_numpy expects a CuPy array.")
    with cp.cuda.Device(value.device.id):
        return cp.asnumpy(value, blocking=True)


def synchronize():
    """Surface asynchronous failures before a result/state is committed."""
    cp.cuda.Device().synchronize()
