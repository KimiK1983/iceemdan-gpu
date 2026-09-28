"""Run AFTER passing stages 0–4 on the target GPU.

python examples/compare_reference.py
Output arrays stay on GPU until the explicit to_numpy boundary below.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from iceemdan_cupy import ICEEMDAN, to_numpy


def main():
    spec = importlib.util.spec_from_file_location("reference", ROOT / "reference/ICEEMDAN_cpu.py")
    cpu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cpu)
    n = np.arange(256.0)
    x = np.sin(0.73 * n) + 0.45 * np.cos(0.14 * n)
    W = np.random.default_rng(42).normal(size=(8, len(x)))
    a = cpu.ICEEMDAN(trials=8)
    rows = iter(W)
    a.generate_noise = lambda scale, size: next(rows).copy() * scale
    gpu = ICEEMDAN(trials=8, epsilon=0.2, device=0, rng_mode="cupy_philox")
    expected = a(x, max_imf=2)
    actual = to_numpy(gpu(x, max_imf=2, noise=W))
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-10)
    print(
        "Compared components using IDENTICAL W. Max difference:", np.max(np.abs(actual - expected))
    )
    print("GPU stop reason:", gpu.diagnostics_["stop_reason"])


if __name__ == "__main__":
    main()
