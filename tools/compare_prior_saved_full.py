"""Compare a saved full CPU decomposition with a real CUDA run using identical W."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from iceemdan_cupy import ICEEMDAN, to_numpy
from reference.ICEEMDAN_cpu import ICEEMDAN as CPU


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--trials", type=int, default=100)
    args = parser.parse_args()
    with np.load(args.baseline) as saved:
        x, t, expected = saved["x"], saved["t"], saved["components"]
        W = saved["W"] if "W" in saved else None
    if W is None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "status": "NON_REPRODUCIBLE",
                    "reason": "Saved baseline has no W; equal seeds do not establish equal noise.",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return 2
    if W.shape != (args.trials, len(x)):
        raise ValueError("Saved W shape does not match trials and signal length.")
    rows = iter(W)
    reference = CPU(trials=args.trials, epsilon=0.2)
    reference.generate_noise = lambda scale, size: next(rows).copy() * scale
    start = time.perf_counter()
    cpu_parts = reference(x, T=t)
    cpu_seconds = time.perf_counter() - start
    model = ICEEMDAN(trials=args.trials, epsilon=0.2)
    start = time.perf_counter()
    gpu_parts = to_numpy(model(x, T=t, noise=W, progress=True))
    gpu_seconds = time.perf_counter() - start
    same_shape = cpu_parts.shape == gpu_parts.shape == expected.shape
    report = {
        "baseline": str(args.baseline),
        "trials": args.trials,
        "noise": "saved W",
        "cpu_seconds": cpu_seconds,
        "gpu_seconds": gpu_seconds,
        "shapes": {
            "saved": list(expected.shape),
            "cpu": list(cpu_parts.shape),
            "gpu": list(gpu_parts.shape),
        },
        "stop": {
            "cpu": reference.diagnostics_["stop_reason"],
            "gpu": model.diagnostics_["stop_reason"],
        },
        "same_noise_counts": reference.diagnostics_["noise_mode_counts"]
        == model.diagnostics_["noise_mode_counts"],
        "same_natural_termination": reference.diagnostics_["natural_termination"]
        == model.diagnostics_["natural_termination"],
        "same_sift_counts": [s["sift_iterations"] for s in reference.diagnostics_["stages"]]
        == [s["sift_iterations"] for s in model.diagnostics_["stages"]],
        "cpu_vs_saved_max_abs": float(np.max(np.abs(cpu_parts - expected))) if same_shape else None,
        "gpu_vs_cpu_max_abs": float(np.max(np.abs(gpu_parts - cpu_parts))) if same_shape else None,
        "per_row_max_abs": [
            float(np.max(np.abs(a - b))) for a, b in zip(gpu_parts, cpu_parts, strict=True)
        ]
        if same_shape
        else None,
        "finite": bool(
            np.isfinite(expected).all()
            and np.isfinite(cpu_parts).all()
            and np.isfinite(gpu_parts).all()
        ),
        "cpu_matches_saved": bool(
            same_shape and np.allclose(cpu_parts, expected, rtol=1e-9, atol=1e-10)
        ),
        "allclose": bool(same_shape and np.allclose(gpu_parts, cpu_parts, rtol=1e-9, atol=1e-10)),
    }
    report["passed"] = bool(
        report["finite"]
        and report["cpu_matches_saved"]
        and report["allclose"]
        and report["same_noise_counts"]
        and report["same_natural_termination"]
        and report["same_sift_counts"]
        and report["stop"]["cpu"] == report["stop"]["gpu"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(args.output.with_suffix(".npz"), components=gpu_parts)
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
