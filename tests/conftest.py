"""CUDA suite by default. Explicit CPU surrogate is testing-only, never runtime."""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if os.environ.get("ICEEMDAN_TEST_MODE") == "cpu-surrogate":
    sys.path.insert(0, str(Path(__file__).parent))
    from cpu_surrogate import install

    install()


@pytest.fixture(scope="session")
def cp():
    try:
        import cupy

        if cupy.__version__ != "14.2.0":
            pytest.skip("CuPy 14.2.0 is required for this acceptance run")
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("No CUDA device")
        # Exercise runtime, not import alone. APIs were documented before test use.
        with cupy.cuda.Device(0):
            cupy.zeros(1).item()
    except (ImportError, RuntimeError) as e:
        pytest.skip(f"CUDA unavailable: {e}")
    return cupy


@pytest.fixture(scope="session")
def cpu():
    spec = importlib.util.spec_from_file_location(
        "frozen_iceemdan_cpu", ROOT / "reference/ICEEMDAN_cpu.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="session")
def gpu(cp):
    import iceemdan_cupy

    return iceemdan_cupy
