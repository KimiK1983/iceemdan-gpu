"""One explicit-W CPU/CUDA batch parity case for Compute Sanitizer."""

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import cupy as cp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from iceemdan_cupy import ICEEMDAN
from reference.ICEEMDAN_cpu import ICEEMDAN as CPU

CASES = {
    "odd_i1": (127, 1, 1),
    "block_i5": (129, 5, 2),
    "three_stage": (192, 4, 3),
    "colominas_i50": (1000, 50, 1),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=CASES)
    args = parser.parse_args()
    if cp.__version__ != "14.2.0" or cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError("This gate requires real CuPy 14.2.0 and a CUDA device.")
    cp.zeros(1).item()

    n, trials, max_imf = CASES[args.case]
    t = np.arange(float(n))
    if args.case == "colominas_i50":
        x = np.sin(2 * np.pi * 0.065 * t)
        x[500:750] += np.sin(2 * np.pi * 0.255 * (t[500:750] - 500))
    else:
        x = np.sin(0.71 * t) + 0.5 * np.sin(0.21 * t) + 0.3 * np.sin(0.035 * t)
    W = np.random.default_rng(7 + n + trials).normal(size=(trials, n))

    reference = CPU(trials=trials)
    rows = iter(W)
    reference.generate_noise = lambda scale, size: next(rows).copy() * scale
    expected = reference(x, max_imf=max_imf)
    model = ICEEMDAN(trials=trials, batch_emd=True, graph_control=False)
    actual = cp.asnumpy(model(x, noise=W, max_imf=max_imf))

    assert expected.shape == actual.shape
    assert np.isfinite(expected).all() and np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-10)
    for key in ("stop_reason", "natural_termination", "noise_mode_counts"):
        assert model.diagnostics_[key] == reference.diagnostics_[key], key
    assert len(model.diagnostics_["stages"]) == len(reference.diagnostics_["stages"])
    for stage, baseline in zip(
        model.diagnostics_["stages"], reference.diagnostics_["stages"], strict=True
    ):
        for key in ("missing_noise_modes", "sift_iterations"):
            assert stage[key] == baseline[key], key
    if args.case == "three_stage":
        assert len(model.diagnostics_["stages"]) == 3

    print(
        "BATCH_CASE_PASS "
        + json.dumps(
            {
                "case": args.case,
                "n": n,
                "trials": trials,
                "max_imf": max_imf,
                "noise_sha256": hashlib.sha256(W.tobytes(order="C")).hexdigest(),
                "max_abs_difference": float(np.max(np.abs(actual - expected))),
                "stages": len(model.diagnostics_["stages"]),
                "stop_reason": model.diagnostics_["stop_reason"],
                "python": platform.python_version(),
                "cupy": cp.__version__,
                "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
                "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
